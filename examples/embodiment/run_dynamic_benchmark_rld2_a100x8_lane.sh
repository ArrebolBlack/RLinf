#!/usr/bin/env bash
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -Eeuo pipefail
umask 027
export GIT_OPTIONAL_LOCKS=0
export PYTHONDONTWRITEBYTECODE=1

die() {
  printf 'RLD2_LAUNCH_FATAL: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 4 ]] ||
  die "usage: PACKAGE_ROOT LANE RUN_ID RUN_ROOT"
package_root="$1"
lane="$2"
run_id="$3"
run_root="$4"

[[ -d "$package_root" ]] || die "package root is missing"
[[ "$lane" =~ ^L[0-7]$ ]] || die "lane must be one of L0-L7"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe run id"
[[ ! -e "$run_root" ]] || die "refusing to overwrite run root"

project_root=/vepfs-mlp2/mlp-public/haoce/yjq/se3wam-a100
resource_wrapper="$project_root/bootstrap/benchmark-infra/run_a100x8_sim_lane.sh"
python_wrapper="$project_root/bootstrap/benchmark-infra/benchmark_python_a100.sh"
core_python=/opt/venvs/se3wam-core-py312-cu130/bin/python
[[ -x "$resource_wrapper" && -x "$python_wrapper" && -x "$core_python" ]] ||
  die "authoritative A100X8 runtime wrappers are missing"

(
  cd "$package_root"
  sha256sum --strict -c SHA256SUMS
) >/dev/null || die "launch package signature verification failed"

mapfile -t identity < <(
  "$core_python" - "$package_root" "$lane" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
lane = sys.argv[2]
package = json.loads((root / "launch_package.json").read_text())
plan = json.loads((root / "lanes" / f"{lane}.json").read_text())
allowed_lanes = [f"L{index}" for index in range(8)]
if (
    package.get("schema_version")
    != "rlinf-dynamic-benchmark-rld2-launch-package-v0.3"
    or package.get("status") != "blocked-awaiting-allocation"
    or package.get("allowed_lanes") != allowed_lanes
    or list(package.get("lanes", {})) != allowed_lanes
    or "forbidden_lane" in package
):
    raise SystemExit("launch package state mismatch")
if lane not in package["lanes"]:
    raise SystemExit("lane is not declared by package")
if plan.get("lane") != lane:
    raise SystemExit("lane plan identity mismatch")
print(plan["expected_gpu_uuid"])
print(package["rlinf_source"]["path"])
print(package["rlinf_source"]["commit"])
print(package["se3_source"]["path"])
print(package["se3_source"]["commit"])
print(package["runtime_deps"]["path"])
print(package["runtime_deps"]["sha256sums_sha256"])
PY
)
[[ ${#identity[@]} -eq 7 ]] || die "cannot read package source identity"
expected_uuid="${identity[0]}"
rlinf_root="${identity[1]}"
rlinf_commit="${identity[2]}"
se3_root="${identity[3]}"
se3_commit="${identity[4]}"
runtime_deps_root="${identity[5]}"
runtime_deps_sha256="${identity[6]}"

for item in "$expected_uuid" "$rlinf_root" "$rlinf_commit" "$se3_root" "$se3_commit" \
  "$runtime_deps_root" "$runtime_deps_sha256"; do
  [[ "$item" != *$'\n'* && "$item" != *$'\r'* ]] || die "unsafe package identity"
done
[[ "$(git -C "$rlinf_root" rev-parse HEAD)" == "$rlinf_commit" ]] ||
  die "RLinf snapshot commit mismatch"
[[ -z "$(git -C "$rlinf_root" status --porcelain --untracked-files=all)" ]] ||
  die "RLinf snapshot is not clean"
[[ "$(git -C "$se3_root" rev-parse HEAD)" == "$se3_commit" ]] ||
  die "SE3-WAM snapshot commit mismatch"
[[ -z "$(git -C "$se3_root" status --porcelain --untracked-files=all)" ]] ||
  die "SE3-WAM snapshot is not clean"
[[ -d "$runtime_deps_root" && -f "$runtime_deps_root/SHA256SUMS" ]] ||
  die "runtime dependency snapshot is missing"
[[ "$(sha256sum "$runtime_deps_root/SHA256SUMS" | awk '{print $1}')" == \
  "$runtime_deps_sha256" ]] || die "runtime dependency manifest mismatch"
(
  cd "$runtime_deps_root"
  sha256sum --strict -c SHA256SUMS
) >/dev/null || die "runtime dependency snapshot verification failed"

lane_number="${lane#L}"
runtime_root="$run_root/runtime"
output_root="$run_root/output"
export PYTHONPATH="$runtime_deps_root:$rlinf_root:$se3_root/src"

# The authoritative wrapper obtains the resource lock and fails if keepalive or
# another task still owns the lane.  This launcher never stops keepalive itself.
exec "$resource_wrapper" \
  "$lane_number" \
  "$expected_uuid" \
  "$run_id" \
  "$runtime_root" \
  "$output_root" \
  "$python_wrapper" \
  "$rlinf_root/examples/embodiment/run_dynamic_benchmark_rld2_launch_gate.py" \
  lane \
  --package-root "$package_root" \
  --lane "$lane" \
  --output-root "$output_root"
