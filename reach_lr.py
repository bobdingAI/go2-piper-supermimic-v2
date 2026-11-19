"""Reproduce the planar left/right reach experiment with isolated outputs."""

import argparse
import json
from pathlib import Path

from evaluate import evaluate_trial
from reference import DATA, OFFSETS, floor_model, record


BASELINE = {
    "legacy_initial": Path("artifacts/training/reach_warm_initial/policy.pt"),
    "legacy_updated": Path("artifacts/training/reach_warm/ckpt_latest.pt"),
}
HOLDOUT = ((.055, .032, 0.), (.062, -.034, 0.))
TRAIN = ((.045, .025, 0.), OFFSETS["train"][1], (.065, .05, 0.),
         (.045, -.025, 0.), OFFSETS["train"][2], (.065, -.05, 0.))


def baseline(output, video):
    model = floor_model()
    for index, offset in enumerate(OFFSETS["holdout"]):
        name = f"reach_{index:02d}.npz"
        saved = DATA / "holdout" / name
        replay = output / "replay" / name
        record(model, offset, replay)
        evaluate_trial(replay, output / saved.stem, BASELINE, video=video, reference_path=saved,
                       probe_paths=tuple(DATA / "train" / f"reach_{i:02d}.npz" for i in (1, 2)))


def generate(output, offsets):
    model = floor_model()
    for calibrate, label in ((False, "ik"), (True, "calibrated")):
        stats = [record(model, offset, output / label / f"reach_{i:02d}.npz", calibrate=calibrate)
                 for i, offset in enumerate(offsets)]
        (output / f"{label}_metrics.json").write_text(json.dumps(stats, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "train-data", "holdout-data", "compare"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--clip", type=int, action="append")
    parser.add_argument("--checkpoint", nargs=2, metavar=("LABEL", "PATH"), action="append", default=[])
    parser.add_argument("--probe-paths", nargs=2, type=Path, default=())
    args = parser.parse_args()
    if args.stage == "compare" and args.data_dir is None:
        parser.error("compare requires --data-dir")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.stage == "baseline":
        baseline(args.output, not args.no_video)
    elif args.stage == "compare":
        clips = ([args.data_dir / f"reach_{i:02d}.npz" for i in args.clip] if args.clip
                 else sorted(args.data_dir.glob("reach_*.npz")))
        if not clips:
            parser.error("no reach clips in --data-dir")
        for clip in clips:
            evaluate_trial(clip, args.output / clip.stem, dict(args.checkpoint),
                           video=not args.no_video, probe_paths=args.probe_paths)
    else:
        generate(args.output, TRAIN if args.stage == "train-data" else HOLDOUT)


if __name__ == "__main__":
    main()
