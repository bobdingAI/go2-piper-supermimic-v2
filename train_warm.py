"""Initialize from the procedural teacher, then run the upstream SuperMimic trainer."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.config import TrainConfig
from supermimic.robots.go2_piper import HOME
from supermimic.training.loop import Trainer


@dataclass
class WarmTrainConfig(TrainConfig):
    align_teacher: bool = False


def initialize_policy(trainer: Trainer):
    env, command = trainer.env, trainer.command
    files = sorted(Path(trainer.cfg.data.npz_dir).glob("*.npz"))
    series = []
    for path in files:
        with np.load(path, allow_pickle=False) as clip:
            series.append(clip["teacher_joint_target"])
    targets = torch.as_tensor(np.stack(series), device=env.device)
    home = torch.tensor(list(HOME.values()), device=env.device)
    targets = (targets - home) / trainer.spec.action_scale
    clips = torch.arange(env.num_envs, device=env.device) % len(files)
    starts = trainer.lib.clip_start[clips]
    command.forced_frame[:] = starts
    term = env.action_manager.get_term("joint_pos")
    noise = term.noise_std
    # Legacy runs keep their original one-frame lead for reproduction.
    lead = 0 if trainer.cfg.align_teacher else 1
    term.noise_std = 0.0
    states, goals, actions = [], [], []
    try:
        obs, _ = env.reset()
        with torch.no_grad():
            for _ in range(targets.shape[1] - 1):
                active_clips = trainer.lib.frame_clip[command.frame]
                frame = command.frame - trainer.lib.clip_start[active_clips]
                # mjlab advances the command on reset. Its current frame is the
                # destination of the next control step, also the frame the goal window starts at.
                target_frame = (frame + lead).clamp(max=targets.shape[1] - 1)
                action = targets[active_clips, target_frame]
                states.append(obs["policy"].detach().clone())
                goals.append(obs["goal"].detach().clone())
                actions.append(action)
                obs, _, _, _, _ = env.step(action)
    finally:
        term.noise_std = noise
        command.forced_frame.fill_(-1)

    states, goals, actions = map(torch.cat, (states, goals, actions))
    optimizer = torch.optim.AdamW(trainer.policy.parameters(), lr=0.0003)
    for step in range(401):
        indices = torch.randint(len(actions), (128,), device=env.device)
        estimate = trainer.policy(states[indices], goals[indices])
        loss = (estimate - actions[indices]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Teacher initialization step {step}, action MSE {loss.item():.6f}")
    initial_dir = trainer.log_dir.parent / f"{trainer.cfg.run_name}_initial"
    initial_dir.mkdir(exist_ok=True)
    torch.save({"policy": trainer.policy.state_dict(), "config": asdict(trainer.cfg)},
               initial_dir / "policy.pt")


def main():
    cfg = tyro.cli(WarmTrainConfig)
    trainer = Trainer(cfg)
    try:
        initialize_policy(trainer)
        trainer.run()
        print(f"PyTorch CUDA peak allocated: {torch.cuda.max_memory_allocated(trainer.device) / 2**20:.1f} MiB; "
              f"reserved: {torch.cuda.max_memory_reserved(trainer.device) / 2**20:.1f} MiB")
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
