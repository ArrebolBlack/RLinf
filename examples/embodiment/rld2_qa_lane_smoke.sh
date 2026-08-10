#!/usr/bin/env bash
set -u

# Run one short, isolated reset/step smoke on every approved A800 lane.
# The host-specific environment variables are supplied by the caller.

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <lane-prefix> <gpu-csv> <repo> <runtime> <runs>" >&2
  exit 2
fi

lane_prefix="$1"
gpu_csv="$2"
repo="$3"
runtime="$4"
runs="$5"
image="${SE3WAM_IMAGE:-se3wam/a800-benchmark-runtime:mujoco311-rs140-v1}"
mkdir -p "$runs"

pids=()
IFS=',' read -r -a gpus <<< "$gpu_csv"
for gpu in "${gpus[@]}"; do
  (
    SE3WAM_CONTAINER_NAME="rld2qa-${lane_prefix}-g${gpu}" \
    SE3WAM_REPO="$repo" \
    SE3WAM_IMAGE="$image" \
    SE3WAM_RUNTIME="$runtime" \
    SE3WAM_CACHE_MOUNT_MODE=ro \
    se3wam-dev "$gpu" env \
      PYTHONPATH=/workspace/runtime/pydeps-portable-v1:/workspace/runtime/pydeps-a800-core-v1:/workspace/SE3-WAM:/workspace/SE3-WAM/src \
      python /workspace/SE3-WAM/examples/embodiment/rld2_qa_runtime_smoke.py \
      --task p0_grasp --split validation \
      --manifest-seed "$((20262150 + gpu))" --steps 2 \
      --output "/workspace/runs/RLD2-QA/lane-${lane_prefix}-g${gpu}.json" \
      > "${runs}/lane-${lane_prefix}-g${gpu}.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

exit "$status"
