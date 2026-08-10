#!/usr/bin/env bash
set -euo pipefail

# Formal RLD2-QA privileged-planner metric-calibration wave.
#
# The caller/scheduler owns the live N0 availability and conflict check.  This
# launcher consumes only the handed-over GPU indices, maps the frozen exact-14
# task order round-robin, and runs at most one task at a time on each lane.

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 <lane-prefix> <gpu-csv> <repo> <runtime> <runs> <tmp-root> [log-root]" >&2
  exit 2
fi

lane_prefix="$1"
gpu_csv="$2"
repo="$3"
runtime="$4"
runs="$5"
tmp_root="$6"
requested_log_root="${7:-}"

image_ref="${SE3WAM_IMAGE:-se3wam/a800-benchmark-runtime:mujoco311-rs140-v1}"
image_id="${SE3WAM_IMAGE_ID:-}"
dev_bin="${SE3WAM_DEV_BIN:-se3wam-dev}"
host_python="${RLD2_QA_PYTHON:-python3}"
wave_id="${RLD2_QA_CALIBRATION_WAVE_ID:-planner-calibration-metric-v03-s20261350}"
manifest_seed="${RLD2_QA_CALIBRATION_SEED:-20261350}"
validation_seed="${RLD2_QA_VALIDATION_SEED:-20261150}"
review_seed="${RLD2_QA_REVIEW_SEED:-20261250}"
test_id_seed="${RLD2_QA_TEST_ID_SEED:-20262040}"
test_ood_seed="${RLD2_QA_TEST_OOD_SEED:-20262041}"
benchmark_commit="${RLD2_QA_BENCHMARK_COMMIT:-1e58962122d1d0caef904f2e8597d7692802951c}"
benchmark_source_root="${RLD2_QA_BENCHMARK_SOURCE_ROOT:-}"

helper_relative="examples/embodiment/rld2_qa_planner_calibration_contract.py"
helper_host="${repo}/${helper_relative}"
helper_container="/workspace/SE3-WAM/${helper_relative}"
container_repo="/workspace/SE3-WAM"
wave_relative="RLD2-QA/${wave_id}"
wave_host="${runs}/${wave_relative}"
wave_container="/workspace/runs/${wave_relative}"
log_root="${requested_log_root:-${wave_host}/logs}"

fail() {
  echo "planner calibration launcher error: $*" >&2
  exit 2
}

[[ "$lane_prefix" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || \
  fail "lane-prefix must contain only safe identifier characters"
[[ "$wave_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || \
  fail "wave ID must contain only safe identifier characters"
(( ${#lane_prefix} <= 20 )) || fail "lane-prefix is too long for isolated container names"
(( ${#wave_id} <= 40 )) || fail "wave ID is too long for isolated container names"
[[ -d "$repo/.git" || -f "$repo/.git" ]] || fail "repo is not a Git worktree: $repo"
[[ -d "$runtime" ]] || fail "runtime directory is absent: $runtime"
[[ -n "$benchmark_source_root" ]] || \
  fail "RLD2_QA_BENCHMARK_SOURCE_ROOT must name the mounted clean SE3-WAM worktree"
[[ -d "$benchmark_source_root/.git" || -f "$benchmark_source_root/.git" ]] || \
  fail "benchmark source is not a Git worktree: $benchmark_source_root"
command -v "$dev_bin" >/dev/null 2>&1 || fail "lane wrapper is unavailable: $dev_bin"
command -v "$host_python" >/dev/null 2>&1 || fail "host Python is unavailable: $host_python"
command -v docker >/dev/null 2>&1 || fail "docker is unavailable"
command -v flock >/dev/null 2>&1 || fail "flock is unavailable"
command -v realpath >/dev/null 2>&1 || fail "realpath is unavailable"

repo="$(realpath -e -- "$repo")" || fail "cannot resolve evaluator repo"
runtime="$(realpath -e -- "$runtime")" || fail "cannot resolve runtime root"
benchmark_source_root="$(realpath -e -- "$benchmark_source_root")" || \
  fail "cannot resolve benchmark source root"
case "$benchmark_source_root" in
  "$runtime"/*) ;;
  *) fail "benchmark source root must be a child of the mounted runtime root" ;;
esac
benchmark_relative="${benchmark_source_root#"$runtime"/}"
[[ "$benchmark_relative" =~ ^[A-Za-z0-9._/-]+$ ]] || \
  fail "benchmark source relative path contains unsupported characters"
benchmark_container="/workspace/runtime/${benchmark_relative}"
helper_host="${repo}/${helper_relative}"
[[ -f "$helper_host" ]] || fail "contract helper is absent: $helper_host"

case "$manifest_seed,$validation_seed,$review_seed,$test_id_seed,$test_ood_seed" in
  *[!0-9,]*) fail "all manifest seeds must be nonnegative decimal integers" ;;
esac

IFS=',' read -r -a gpus <<< "$gpu_csv"
(( ${#gpus[@]} >= 1 && ${#gpus[@]} <= 8 )) || \
  fail "gpu-csv must contain between 1 and 8 N0 lanes"
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-7]$ ]] || fail "GPU index must be in [0,7]: $gpu"
  [[ -z "${seen_gpus[$gpu]:-}" ]] || fail "duplicate GPU index: $gpu"
  seen_gpus[$gpu]=1
done

if [[ -z "$image_id" ]]; then
  image_id="$(docker image inspect --format '{{.Id}}' "$image_ref")" || \
    fail "cannot resolve runtime image: $image_ref"
fi
[[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || \
  fail "runtime image must resolve to an immutable sha256 identity"

evaluator_head="$(git -C "$repo" rev-parse HEAD)" || fail "cannot resolve evaluator Git HEAD"
evaluator_commit="${RLD2_QA_EVALUATOR_COMMIT:-$evaluator_head}"
[[ "$evaluator_commit" =~ ^[0-9a-f]{40}$ ]] || fail "evaluator commit is invalid"
[[ "$benchmark_commit" =~ ^[0-9a-f]{40}$ ]] || fail "benchmark commit is invalid"
[[ "$evaluator_head" == "$evaluator_commit" ]] || \
  fail "evaluator commit does not equal source HEAD"

mkdir -p "$runs/RLD2-QA" "$tmp_root" "$log_root"
lock_path="${runs}/RLD2-QA/.${wave_id}.launcher.lock"
exec 9>"$lock_path"
flock -n 9 || fail "another launcher owns wave $wave_id"

"$host_python" "$helper_host" init-wave \
  --wave-root "$wave_host" \
  --wave-id "$wave_id" \
  --lane-prefix "$lane_prefix" \
  --gpus "$gpu_csv" \
  --source-root "$repo" \
  --benchmark-source-root "$benchmark_source_root" \
  --benchmark-source-container-root "$benchmark_container" \
  --image-ref "$image_ref" \
  --image-id "$image_id" \
  --evaluator-commit "$evaluator_commit" \
  --benchmark-commit "$benchmark_commit" \
  --manifest-seed "$manifest_seed" \
  --validation-seed "$validation_seed" \
  --review-seed "$review_seed" \
  --test-id-seed "$test_id_seed" \
  --test-ood-seed "$test_ood_seed"

mapfile -t tasks < <("$host_python" "$helper_host" list-tasks)
(( ${#tasks[@]} == 14 )) || fail "contract helper did not return exact-14 tasks"

run_container_task() {
  local phase="$1"
  local gpu="$2"
  local task="$3"
  local task_tmp="${tmp_root}/${wave_id}/${lane_prefix}/g${gpu}/${phase}-${task}"
  local container_name="rld2qa-cal-${wave_id}-${phase}-${task}-${lane_prefix}-g${gpu}"
  local phase_log="${log_root}/${phase}-${task}.launcher.log"
  local -a helper_args
  mkdir -p "$task_tmp"
  if [[ "$phase" == "predeclare" ]]; then
    helper_args=(
      predeclare-task
      --wave-root "$wave_container"
      --task "$task"
      --source-root "$container_repo"
      --benchmark-source-root "$benchmark_container"
    )
  elif [[ "$phase" == "evaluate" ]]; then
    helper_args=(
      run-task
      --wave-root "$wave_container"
      --task "$task"
      --source-root "$container_repo"
      --benchmark-source-root "$benchmark_container"
      --image-size 64
    )
  else
    fail "unknown lane phase: $phase"
  fi
  SE3WAM_CONTAINER_NAME="$container_name" \
  SE3WAM_REPO="$repo" \
  SE3WAM_IMAGE="$image_id" \
  SE3WAM_RUNTIME="$runtime" \
  SE3WAM_RUNS="$runs" \
  SE3WAM_TMP="$task_tmp" \
  SE3WAM_CACHE_MOUNT_MODE=ro \
    "$dev_bin" "$gpu" env \
      PYTHONHASHSEED=0 \
      PYTHONDONTWRITEBYTECODE=1 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 \
      PYTHONPATH="${benchmark_container}/src:${container_repo}:/workspace/runtime/pydeps-portable-v1:/workspace/runtime/pydeps-a800-core-v1" \
      python "$helper_container" "${helper_args[@]}" \
      >"$phase_log" 2>&1
}

run_phase() {
  local phase="$1"
  local lane_count="${#gpus[@]}"
  local -a pids=()
  local lane_slot
  for lane_slot in "${!gpus[@]}"; do
    (
      local ordinal
      for ((ordinal=lane_slot; ordinal<${#tasks[@]}; ordinal+=lane_count)); do
        run_container_task "$phase" "${gpus[$lane_slot]}" "${tasks[$ordinal]}"
      done
    ) &
    pids+=("$!")
  done
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  (( status == 0 )) || fail "$phase phase failed; no later phase was started"
}

# No rollout can begin until all 280 reset identities exist and the root seal
# covers every per-task manifest/contract.
run_phase predeclare
"$host_python" "$helper_host" seal-wave \
  --wave-root "$wave_host" \
  --source-root "$repo" \
  --benchmark-source-root "$benchmark_source_root"

run_phase evaluate

# Final receipt validation replays the frozen evidence builder and therefore
# runs in the same bound image/source environment as task finalization, not in
# the dependency-light host Python used for init/seal.
finalize_gpu="${gpus[0]}"
finalize_tmp="${tmp_root}/${wave_id}/${lane_prefix}/g${finalize_gpu}/finalize-wave"
finalize_container="rld2qa-cal-${wave_id}-finalize-${lane_prefix}-g${finalize_gpu}"
mkdir -p "$finalize_tmp"
SE3WAM_CONTAINER_NAME="$finalize_container" \
SE3WAM_REPO="$repo" \
SE3WAM_IMAGE="$image_id" \
SE3WAM_RUNTIME="$runtime" \
SE3WAM_RUNS="$runs" \
SE3WAM_TMP="$finalize_tmp" \
SE3WAM_CACHE_MOUNT_MODE=ro \
  "$dev_bin" "$finalize_gpu" env \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONPATH="${benchmark_container}/src:${container_repo}:/workspace/runtime/pydeps-portable-v1:/workspace/runtime/pydeps-a800-core-v1" \
    python "$helper_container" finalize-wave \
      --wave-root "$wave_container" \
      --source-root "$container_repo" \
      --benchmark-source-root "$benchmark_container" \
    >"${log_root}/finalize-wave.launcher.log" 2>&1
