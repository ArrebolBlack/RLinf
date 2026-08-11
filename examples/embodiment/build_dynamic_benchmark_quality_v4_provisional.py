#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the explicit, fail-closed Qv4 provisional threshold inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from se3_wam.benchmark.contracts import stable_sha256
from se3_wam.benchmark.trajectory_quality_v4 import (
    EXACT14_ORIENTATION_CONTRACT,
    QUALITY_V4_THRESHOLDS_SCHEMA,
    orientation_contract_manifest,
)

from examples.embodiment.dynamic_benchmark_quality_v4 import (
    QUALITY_V4_VISION_TOLERANCE_SCHEMA,
    quality_v4_threshold_check_inventory,
    validate_quality_v4_thresholds,
)


def _check(phase: str, metric: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "metric": metric,
        "direction": "minimize",
        "max": None,
        "paired_non_worse_tolerance": None,
        "strict_improvement_margin": None,
        "calibration_status": "pending",
    }


def _vision_tolerance(task_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": QUALITY_V4_VISION_TOLERANCE_SCHEMA,
        "task_id": task_id,
        "calibration_status": "provisional",
        "rgb_max_abs_lsb": 2,
        "rgb_max_changed_fraction_per_frame": 0.001,
        "depth_abs_tolerance_m": 0.0001,
        "depth_relative_tolerance": 0.00001,
        "segmentation_exact": True,
        "rationale": (
            "Per-task replay identity is registered now; numerical values remain "
            "provisional until calibration evidence and owner review are complete."
        ),
    }
    payload["tolerance_sha256"] = stable_sha256(payload)
    return payload


def build_provisional_thresholds() -> dict[str, Any]:
    """Return exact-14 task-by-phase checks with no invented numeric freeze."""

    tasks: dict[str, Any] = {}
    check_inventory = quality_v4_threshold_check_inventory()
    for task_id in EXACT14_ORIENTATION_CONTRACT:
        checks = [_check(phase, metric) for phase, metric in check_inventory[task_id]]
        tasks[task_id] = {
            "vision_tolerance": _vision_tolerance(task_id),
            "checks": checks,
        }
    payload: dict[str, Any] = {
        "schema_version": QUALITY_V4_THRESHOLDS_SCHEMA,
        "contract_name": "Dynamic Benchmark trajectory quality Qv4",
        "calibration_status": "provisional",
        "formal_freeze_eligible": False,
        "orientation_contract_sha256": orientation_contract_manifest()[
            "contract_sha256"
        ],
        "allowed_tuning_splits": ["metric_calibration"],
        "splits_read": [],
        "calibration_sources": {
            "exact14x20_planner": {
                "split": "metric_calibration",
                "task_count": 14,
                "episodes_per_task": 20,
                "total_reset_count": 280,
                "status": "qv4_replay_pending",
                "inherited_qv3_receipt_sha256": (
                    "ccda1567e64b6a51e5cb1c631a0b03e94d48e6c9f4bee0c63df3f2f491ad0d20"
                ),
                "note": (
                    "The Qv3 receipt is inventory only and is not evidence that Qv4 "
                    "metrics were calibrated."
                ),
            },
            "known_good_bad_trajectories": {
                "status": "pending",
                "required_labels": [
                    "known_good",
                    "issued_action_jitter",
                    "applied_action_jitter",
                    "physics_rate_jerk",
                    "path_detour",
                    "path_backtrack",
                    "orientation_drift",
                    "object_tilt_or_slip",
                ],
            },
            "fresh_deterministic_rl_pilot": {
                "status": "pending",
                "same_reset_pairing_required": True,
                "return_is_diagnostic_only": True,
            },
        },
        "owner_review": {
            "approved": False,
            "reviewer": None,
            "reviewed_at": None,
            "decision_record": None,
        },
        "freeze_requirements": [
            "all three calibration sources complete",
            "no test_id or test_ood samples read for tuning",
            "all bounds, paired tolerances, and strict margins numeric",
            "project-owner review approved",
        ],
        "tasks": tasks,
    }
    payload["thresholds_sha256"] = stable_sha256(payload)
    validate_quality_v4_thresholds(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "rld2_qa"
            / "quality_v4"
            / "quality_v4_thresholds.provisional.json"
        ),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_provisional_thresholds(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
