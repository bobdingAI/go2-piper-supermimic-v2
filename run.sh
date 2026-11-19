#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
runtime="${GO2_RUNTIME_DIR:-$HOME/.cache/go2-piper-supermimic-v2}"
python="$runtime/venv/bin/python"
export PYTHONPATH="$PWD/vendor/supermimic-quadruped-0.2.0/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p artifacts

case "${1:-reference}" in
  reference)
    MUJOCO_GL=disable "$python" reference.py 2>&1 | tee artifacts/reference.log
    ;;
  train|warm|reach-lr-warm)
    log_name="${2:-reach_smoke}"
    entry=(-m supermimic.scripts.train)
    extra=()
    output=artifacts
    data=data/train
    if [[ "$1" != train ]]; then
      entry=(train_warm.py)
      extra=(--policy-trainer.lr 0.00001 --tf32)
    fi
    if [[ "$1" == reach-lr-warm ]]; then
      output="${4:?provide a new experiment output directory}"
      data="${5:?provide a training data directory}"
      mkdir -- "$output"
      extra+=(--align-teacher "${@:6}")
    fi
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -lms 200 > "$output/${log_name}_gpu.csv" &
      gpu_monitor=$!
      trap 'kill "$gpu_monitor" 2>/dev/null || true' EXIT
    fi
    MUJOCO_GL=disable "$python" "${entry[@]}" "${extra[@]}" \
      --robot go2_piper --run-name "$log_name" --log-root "$output/training" --data.npz-dir "$data" --seed 0 \
      --env.num-envs 48 --env.replay-capacity 300 --env.action-noise-std 0.035 \
      --env.no-obs-noise --env.no-reset-noise --env.no-domain-randomization \
      --wm-trainer.batch-size 64 --policy-trainer.batch-size 64 \
      --wm-trainer.compile None --policy-trainer.compile None --policy-trainer.horizon 12 \
      --schedule.total-iterations "${3:-300}" --schedule.log-interval 25 --schedule.checkpoint-interval 10000 \
      2>&1 | tee "$output/$log_name.log"
    ;;
  evaluate)
    MUJOCO_GL=egl "$python" evaluate.py "${@:2}"
    ;;
  reach-lr)
    MUJOCO_GL=egl "$python" reach_lr.py "${@:2}"
    ;;
  *)
    printf 'Usage: bash run.sh [reference|train|warm|evaluate|reach-lr|reach-lr-warm]\n' >&2
    exit 2
    ;;
esac
