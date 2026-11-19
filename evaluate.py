"""Measure policy tracking of a held-out reach and render reference vs rollout."""

import argparse
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from reference import DATA, DWELL_END, DWELL_START, FPS, ROOT, floor_model
from supermimic.robots.go2_piper import BASE, HOME, JOINTS, LEGS, TIP


def rollout(checkpoint, path, steps, teacher=None, probe_paths=()):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    policy, config = (load_policy_from_checkpoint(checkpoint, "cuda:0") if checkpoint else (None, {"env": {}}))
    if checkpoint and config["robot"] != "go2_piper":
        raise ValueError("Expected a Go2 + PiPER checkpoint")
    if not checkpoint:
        from supermimic.robots import set_robot_overrides

        set_robot_overrides(stiffness=None, action_scale=None)
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in (path, *probe_paths)], robot="go2_piper", num_envs=1, episode_length_s=10.0,
        future_frames=config["env"].get("future_frames", 10), anchor_obs=config["env"].get("anchor_obs", False),
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
        record_transitions=False, sampling_mode="start", seed=0,
    ))
    cfg.terminations.pop("clip_end", None)
    cfg.auto_reset = False
    cfg.scene.sensors = (ContactSensorCfg(
        name="ground_contacts", primary=ContactMatch("geom", LEGS, "robot"),
        secondary=ContactMatch("geom", "terrain"), fields=("found",), reduce="mindist",
    ),)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        robot = env.scene["robot"]
        root_idx = robot.body_names.index(BASE)
        tip_idx = robot.body_names.index(TIP)
        origin = env.scene.env_origins[0].cpu().numpy()
        command = env.command_manager.get_term("motion")
        term = env.action_manager.get_term("joint_pos")
        if tuple(robot.joint_names) != JOINTS or tuple(term.target_names) != JOINTS:
            raise ValueError("Robot or actuator joint order differs from the demonstrations")
        scale = torch.broadcast_to(torch.as_tensor(term.scale, device=env.device), (1, len(JOINTS)))
        offset = term.offset
        torch.testing.assert_close(offset[0], torch.tensor(list(HOME.values()), device=env.device))
        if teacher is not None:
            targets = torch.as_tensor(teacher, device=env.device)
        start = int(command.library.clip_start[0])
        command.forced_frame[:] = start
        observations, _ = env.reset()
        positions, tips, support, frames = [], [], [], []
        audit = {"joint_names": list(JOINTS), "scale": scale[0].tolist(), "offset_rad": offset[0].tolist(),
                 "encoder_bias_rad": robot.data.encoder_bias[0].tolist(), "fixed_observation_probes": []}
        with torch.inference_mode():
            for tick in range(steps):
                done = False
                if tick:
                    if teacher is None:
                        action = policy(observations["policy"], observations["goal"])
                    else:
                        action = (targets[tick] - offset) / scale
                    observations, _, terminated, truncated, _ = env.step(action)
                    done = bool((terminated | truncated)[0])
                state = robot.data
                root = state.body_link_pos_w[0, root_idx].cpu().numpy() - origin
                quat = state.body_link_quat_w[0, root_idx].cpu().numpy()
                joints = state.joint_pos[0].cpu().numpy()
                positions.append(np.concatenate((root, quat, joints)))
                tips.append(state.body_link_pos_w[0, tip_idx].cpu().numpy() - origin)
                support.append(int((env.scene["ground_contacts"].data.found[0] > 0).sum()))
                frames.append(int(command.frame[0]) - start)
                if policy is not None and probe_paths and tick in (20, 30, 40):
                    windows = command.library.goal_window(command.library.clip_start[1:] + frames[-1],
                                                          command.cfg.future_frames)
                    fixed_obs = observations["policy"].expand(len(windows), -1)
                    actions = policy(fixed_obs, windows)[:, -6:]
                    audit["fixed_observation_probes"].append({
                        "observation_frame": tick, "window_frames": [frames[-1], frames[-1] + command.cfg.future_frames - 1],
                        "clips": [str(p) for p in probe_paths], "actions": actions.tolist(),
                        "delta_right_minus_left_rad": ((actions[1] - actions[0]) * scale[0, -6:]).tolist(),
                    })
                if done:
                    break
        audit["command_frames_for_observations"] = frames
        audit["terminated"] = done
        return np.asarray(positions), np.asarray(tips), np.asarray(support), audit
    finally:
        env.close()


def render(destination, reference_positions, actual_positions=None, actual_label="SuperMimic policy"):
    model = floor_model()
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (.12, 0, .45)
    camera.azimuth, camera.elevation, camera.distance = 130, -17, 1.45
    width, height = 480, 360
    paired = actual_positions is not None
    sequence = [("IK and PD reference", reference_positions)]
    if paired:
        sequence.append((actual_label, actual_positions))
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{width * len(sequence)}x{height}",
        "-framerate", str(FPS // 2), "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    with mujoco.Renderer(model, height, width) as renderer, subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for tick in range(0, len(reference_positions), 2):
            panels = []
            for label, poses in sequence:
                if tick >= len(poses):
                    panel = Image.new("RGB", (width, height), (24, 30, 39))
                    ImageDraw.Draw(panel).text((12, 12), f"{label} finished early", fill="white")
                else:
                    data.qpos[:] = poses[tick]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera)
                    panel = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, 210, 30), fill=(24, 30, 39))
                    draw.text((12, 10), f"{label}   {tick / FPS:.2f}s", fill="white")
                panels.append(np.asarray(panel))
            encoder.stdin.write(np.concatenate(panels, axis=1).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode the video")


def read_reference(path):
    with np.load(path, allow_pickle=False) as clip:
        if tuple(clip["joint_names"]) != JOINTS or int(clip["fps"]) != FPS:
            raise ValueError(f"Unexpected joint order or frame rate in {path}")
        base = list(clip["body_names"]).index(BASE)
        tip = list(clip["body_names"]).index(TIP)
        poses = np.concatenate((clip["body_pos_w"][:, base], clip["body_quat_w"][:, base], clip["joint_pos"]), axis=1)
        return poses, clip["body_pos_w"][:, tip], clip["reach_goal_w"], clip["teacher_joint_target"]


def measure(reference_tip, goal, actual, actual_tip, support, terminated=False):
    count = len(actual)
    orientation = actual[:, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (orientation[:, 1] ** 2 + orientation[:, 2] ** 2), -1, 1)))
    error = np.linalg.norm(actual_tip - reference_tip[:count], axis=1)
    time = np.arange(len(reference_tip)) / FPS
    dwell = (time >= DWELL_START) & (time <= DWELL_END)
    disp = actual_tip - actual_tip[0]
    ref_disp = reference_tip - reference_tip[0]
    direction = ref_disp[dwell].mean(axis=0)
    full_dwell = count > np.flatnonzero(dwell)[-1]
    held = disp[dwell[:count]].mean(axis=0) if full_dwell else None
    return {
        "steps_completed": count, "steps_expected": len(reference_tip), "terminated": terminated,
        "direction_completion": float(held @ direction / (direction @ direction)) if full_dwell else None,
        "dwell_displacement_xyz_m": held.tolist() if full_dwell else None,
        "reference_dwell_displacement_xyz_m": direction.tolist(),
        "dwell_tip_tracking_error_m": float(error[dwell[:count]].mean()) if full_dwell else None,
        "reach_goal_error_m": float(np.linalg.norm(actual_tip[dwell[:count]] - goal, axis=1).mean()) if full_dwell else None,
        "mean_tip_tracking_error_m": float(error.mean()),
        "peak_tip_excursion_m": float(np.linalg.norm(disp, axis=1).max()),
        "reference_peak_tip_excursion_m": float(np.linalg.norm(ref_disp, axis=1).max()),
        "return_residual_m": float(np.linalg.norm(disp[-1])) if count == len(reference_tip) and not terminated else None,
        "max_tilt_deg": float(tilt.max()),
        "full_support_fraction": float(np.mean(support == 4)) if support is not None else None,
    }


def plot_xyz(destination, reference_tip, series):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    for axis, label in enumerate("XYZ"):
        for name, tip in [("IK+PD reference", reference_tip), *series.items()]:
            axes[axis].plot(np.arange(len(tip)) / FPS, 100 * (tip[:, axis] - tip[0, axis]), label=name)
        axes[axis].axvspan(DWELL_START, DWELL_END, color="gray", alpha=.18)
        axes[axis].set_ylabel(f"{label} (cm)")
        axes[axis].grid(alpha=.2)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(destination, dpi=140)


def evaluate_trial(path, output, checkpoints, *, video=True, probe_paths=(), reference_path=None):
    """Run all controllers from the same clip start and score against a common reference."""
    output.mkdir(parents=True, exist_ok=True)
    poses, tip, goal, targets = read_reference(path)
    reference_poses, reference_tip, _, _ = read_reference(reference_path or path)
    if len(poses) != len(reference_poses) or not np.allclose(poses[0], reference_poses[0], atol=1e-5):
        raise ValueError("Controller and scoring clips must share initial pose and frame count")
    with np.load(path, allow_pickle=False) as clip:
        support = clip["foot_contact_count"] if "foot_contact_count" in clip else None
    metrics = {"ik_pd_cpu": measure(reference_tip, goal, poses, tip, support)}
    series = {"IK+PD CPU replay": tip} if reference_path else {}
    for label, checkpoint in {"ik_pd_warp": None, **checkpoints}.items():
        actual, actual_tip, contacts, audit = rollout(checkpoint, path, len(poses),
                                                      teacher=targets if checkpoint is None else None,
                                                      probe_paths=probe_paths if checkpoint else ())
        metrics[label] = measure(reference_tip, goal, actual, actual_tip, contacts, audit["terminated"])
        metrics[label]["checkpoint"] = str(checkpoint) if checkpoint else None
        metrics[label]["frame_zero_joint_max_error_rad"] = float(np.max(np.abs(actual[0, 7:] - poses[0, 7:])))
        metrics[label]["frame_zero_tip_error_m"] = float(np.linalg.norm(actual_tip[0] - tip[0]))
        series[label] = actual_tip
        (output / f"{label}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        np.savez_compressed(output / f"{label}_rollout.npz", actual_qpos=actual, actual_tip=actual_tip,
                            reference_qpos=reference_poses, reference_tip=reference_tip,
                            foot_contact_count=contacts, fps=FPS)
        if video:
            render(output / f"{label}.mp4", reference_poses, actual, label)
    plot_xyz(output / "displacement_xyz.png", reference_tip, series)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--group", choices=("train", "holdout"), default="holdout")
    parser.add_argument("--clip", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    path = args.data_dir / args.group / f"reach_{args.clip:02d}.npz"
    reference_positions, reference_tip, reach_goal, _ = read_reference(path)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    label = f"{args.group}_{path.stem}"
    if args.checkpoint is None:
        render(output / f"reference_{label}.mp4", reference_positions)
        return

    actual, actual_tip, support, audit = rollout(args.checkpoint, path, len(reference_positions))
    count = len(actual)
    joint_error = actual[:, 7:] - reference_positions[:count, 7:]
    measurements = {
        "clip": f"{args.group}/{path.name}", "checkpoint": str(args.checkpoint),
        **measure(reference_tip, reach_goal, actual, actual_tip, support, audit["terminated"]),
        "leg_joint_rmse_rad": float(np.sqrt(np.mean(joint_error[:, :12] ** 2))),
        "arm_joint_rmse_rad": float(np.sqrt(np.mean(joint_error[:, 12:] ** 2))),
    }
    print(json.dumps(measurements, indent=2))
    stem = f"{args.checkpoint.parent.name}_{label}"
    (output / f"{stem}_metrics.json").write_text(json.dumps(measurements, indent=2) + "\n")
    np.savez_compressed(output / f"{stem}_rollout.npz", actual_qpos=actual, actual_tip=actual_tip,
                        reference_qpos=reference_positions, reference_tip=reference_tip,
                        foot_contact_count=support, fps=FPS)
    plot_xyz(output / f"{stem}_xyz.png", reference_tip, {"Policy": actual_tip})
    if not args.no_video:
        render(output / f"{stem}.mp4", reference_positions, actual)


if __name__ == "__main__":
    main()
