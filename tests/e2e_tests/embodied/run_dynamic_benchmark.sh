#!/usr/bin/env bash
set -Eeuo pipefail

: "${SE3_WAM_PATH:?Set SE3_WAM_PATH to the research-only SE3-WAM checkout}"
repo_path="${REPO_PATH:-$(git rev-parse --show-toplevel)}"
rlinf_commit="${RLINF_COMMIT:-$(git -C "$repo_path" rev-parse HEAD)}"
benchmark_commit="${BENCHMARK_COMMIT:-$(git -C "$SE3_WAM_PATH" rev-parse HEAD)}"
if [[ -n "${DYNAMIC_BENCHMARK_E2E_OUTPUT:-}" ]]; then
    run_root="$DYNAMIC_BENCHMARK_E2E_OUTPUT"
    mkdir -p "$run_root"
else
    run_root="$(mktemp -d "${TMPDIR:-/tmp}/rlinf-dynamic-benchmark-e2e.XXXXXX")"
    trap 'rm -rf "$run_root"' EXIT
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="$repo_path:$SE3_WAM_PATH/src${PYTHONPATH:+:$PYTHONPATH}"

python "$repo_path/examples/embodiment/benchmark_dynamic_benchmark_throughput.py" \
    --task t2_trans \
    --num-envs 1 \
    --steps 8 \
    --rlinf-commit "$rlinf_commit" \
    --benchmark-commit "$benchmark_commit" \
    --output "$run_root/throughput.json"

python "$repo_path/examples/embodiment/train_dynamic_benchmark_expert.py" \
    --config "$repo_path/examples/embodiment/config/dynamic_benchmark_t2_rlpd.yaml" \
    --rlinf-commit "$rlinf_commit" \
    --benchmark-commit "$benchmark_commit" \
    --output "$run_root/trainer" \
    --num-envs 1 \
    --eval-num-envs 1 \
    --total-env-steps 16 \
    --random-env-steps 4 \
    --demo-episodes 1 \
    --demo-max-attempts 10 \
    --bc-steps 1 \
    --batch-size 4 \
    --replay-capacity 64 \
    --updates-per-vector-step 1 \
    --q-heads 2 \
    --q-target-subset 2 \
    --eval-interval 16 \
    --eval-episodes 1 \
    --checkpoint-interval 16 \
    --log-interval 8 \
    --manifest-size 32

python - "$run_root" "$rlinf_commit" "$benchmark_commit" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rlinf_commit, benchmark_commit = sys.argv[2:]
throughput = json.loads((root / "throughput.json").read_text())
summary = json.loads((root / "trainer" / "summary.json").read_text())
assert throughput["terminal_count"] == 0
assert throughput["reward_schema"]["schema_version"] == "rlinf-dynamic-benchmark-reward-v0.2"
assert throughput["rlinf_commit"] == rlinf_commit
assert throughput["benchmark_commit"] == benchmark_commit
assert summary["status"] == "complete"
assert summary["config"]["rlinf_commit"] == rlinf_commit
assert summary["config"]["benchmark_commit"] == benchmark_commit
assert summary["env_steps"] == 16
PY
