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

"""Assess Qv4 evidence before materializing an exact-14 threshold candidate.

The command is intentionally fail closed.  It writes a machine-readable readiness
receipt even when evidence is incomplete, but it never invents a threshold value or
changes Owner-review state.  A future aggregator can consume the receipt only after
all blockers are absent and every value is bound to the corresponding evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from se3_wam.benchmark.contracts import stable_sha256
from se3_wam.benchmark.trajectory_quality_v4 import EXACT14_ORIENTATION_CONTRACT

from examples.embodiment.dynamic_benchmark_quality_v4 import (
    quality_v4_threshold_check_inventory,
)

READINESS_SCHEMA = "rlinf-dynamic-benchmark-quality-v4-candidate-readiness-v0.1"
PLANNER_SCHEMA = "se3-wam-qv4-planner-calibration-aggregator-manifest-v0.1"
GOOD_BAD_SCHEMA = "se3-wam-qv4-known-good-bad-evidence-v0.1"
RL_PAIR_SCHEMA = "qv4-rl-pilot-pair-delta-summary-v0.1"
_FORBIDDEN_SPLITS = frozenset({"test_id", "test_ood"})
_REQUIRED_GOOD_BAD_LABELS = frozenset(
    {
        "known_good",
        "issued_action_jitter",
        "applied_action_jitter",
        "physics_rate_jerk",
        "path_detour",
        "path_backtrack",
        "orientation_drift",
        "object_tilt_or_slip",
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"evidence must be a JSON mapping: {path}")
    return dict(payload), _file_sha256(path)


def _planner_blockers(payload: Mapping[str, Any] | None) -> list[str]:
    if payload is None:
        return ["PLANNER_EVIDENCE_MISSING"]
    blockers: list[str] = []
    if payload.get("schema_version") != PLANNER_SCHEMA:
        blockers.append("PLANNER_SCHEMA_MISMATCH")
    if (
        payload.get("status") != "complete_calibration_evidence"
        or payload.get("scientific_partition") != "metric_calibration"
        or payload.get("task_count") != 14
        or payload.get("episodes_per_task") != 20
        or payload.get("expected_reset_count") != 280
        or payload.get("complete_reset_count") != 280
        or payload.get("source_tape_count") != 280
    ):
        blockers.append("PLANNER_NOT_COMPLETE_EXACT14X20_METRIC_CALIBRATION")
    # The current manifest locates source tapes but contains no per-check numeric
    # aggregate.  A formal candidate cannot derive a hard bound from identities.
    if not isinstance(payload.get("task_check_value_evidence"), Mapping):
        blockers.append("PLANNER_TASK_CHECK_HARD_BOUND_EVIDENCE_MISSING")
    if not isinstance(payload.get("task_vision_tolerance_evidence"), Mapping):
        blockers.append("PLANNER_TASK_VISION_TOLERANCE_EVIDENCE_MISSING")
    return blockers


def _good_bad_blockers(payload: Mapping[str, Any] | None) -> list[str]:
    if payload is None:
        return ["GOOD_BAD_EVIDENCE_MISSING"]
    blockers: list[str] = []
    if payload.get("schema_version") != GOOD_BAD_SCHEMA:
        blockers.append("GOOD_BAD_SCHEMA_MISMATCH")
    classification = payload.get("classification")
    labels = (
        {
            str(row.get("expected_label"))
            for row in classification
            if isinstance(row, Mapping)
        }
        if isinstance(classification, Sequence)
        and not isinstance(classification, (str, bytes))
        else set()
    )
    acceptance = payload.get("acceptance")
    if (
        labels != _REQUIRED_GOOD_BAD_LABELS
        or not isinstance(acceptance, Mapping)
        or acceptance.get("known_good_retained") is not True
        or acceptance.get("all_corresponding_bad_rejected") is not True
    ):
        blockers.append("GOOD_BAD_REQUIRED_LABELS_OR_CLASSIFICATION_INCOMPLETE")
    boundary = payload.get("scientific_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("source_runtime_split") != "metric_calibration"
        or boundary.get("may_set_or_freeze_absolute_qv4_thresholds") is not True
        or payload.get("task_count") != 14
        or not isinstance(payload.get("task_check_discriminability_evidence"), Mapping)
    ):
        blockers.append("GOOD_BAD_NOT_FORMAL_EXACT14_METRIC_CALIBRATION_EVIDENCE")
    return blockers


def _rl_blockers(payload: Mapping[str, Any] | None) -> list[str]:
    if payload is None:
        return ["RL_PAIR_EVIDENCE_MISSING"]
    blockers: list[str] = []
    if payload.get("schema_version") != RL_PAIR_SCHEMA:
        blockers.append("RL_PAIR_SCHEMA_MISMATCH")
    distributions = payload.get("qv4_metric_delta_distributions")
    if (
        payload.get("pair_count") != 840
        or not isinstance(distributions, Mapping)
        or set(distributions) != set(EXACT14_ORIENTATION_CONTRACT)
    ):
        blockers.append("RL_NOT_COMPLETE_EXACT14X3X20_SAME_RESET")
    if payload.get("same_reset_pairing_verified") is not True:
        blockers.append("RL_SAME_RESET_PAIRING_NOT_VERIFIED")
    if (
        payload.get("action_mode") != "deterministic_mean"
        or payload.get("stochastic_step_sample_count") != 0
    ):
        blockers.append("RL_NOT_FRESH_DETERMINISTIC_MEAN")
    if not isinstance(payload.get("task_check_paired_value_evidence"), Mapping):
        blockers.append("RL_TASK_CHECK_TOLERANCE_MARGIN_EVIDENCE_MISSING")
    return blockers


def _declared_splits(payloads: Sequence[Mapping[str, Any] | None]) -> set[str]:
    splits: set[str] = set()
    for payload in payloads:
        if payload is None:
            continue
        for name in ("split", "scientific_partition", "source_runtime_split"):
            value = payload.get(name)
            if isinstance(value, str):
                splits.add(value)
        boundary = payload.get("scientific_boundary")
        if isinstance(boundary, Mapping):
            value = boundary.get("source_runtime_split")
            if isinstance(value, str):
                splits.add(value)
    return splits


def assess_candidate_evidence(
    *,
    planner: Mapping[str, Any] | None,
    good_bad: Mapping[str, Any] | None,
    rl_pairs: Mapping[str, Any] | None,
    evidence_files: Mapping[str, str | None],
) -> dict[str, Any]:
    """Return a stable, machine-readable readiness decision."""

    splits_read = _declared_splits((planner, good_bad, rl_pairs))
    blockers = [
        *_planner_blockers(planner),
        *_good_bad_blockers(good_bad),
        *_rl_blockers(rl_pairs),
    ]
    if splits_read & _FORBIDDEN_SPLITS:
        blockers.append("FORBIDDEN_TEST_SPLIT_DECLARED")
    expected_check_count = sum(
        len(checks) for checks in quality_v4_threshold_check_inventory().values()
    )
    receipt: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA,
        "candidate_materialized": False,
        "formal_candidate_ready": not blockers,
        "formal_freeze_eligible": False,
        "owner_review": {
            "approved": False,
            "reviewer": None,
            "reviewed_at": None,
            "decision_record": None,
        },
        "expected_inventory": {
            "task_count": 14,
            "check_count": expected_check_count,
            "planner_reset_count": 280,
            "rl_same_reset_pair_count": 840,
        },
        "splits_declared_by_evidence": sorted(splits_read),
        "test_splits_read": sorted(splits_read & _FORBIDDEN_SPLITS),
        "evidence_file_sha256": dict(evidence_files),
        "blockers": sorted(set(blockers)),
        "next_action": (
            "materialize numeric exact-14 candidate for Owner review"
            if not blockers
            else "resolve every blocker; do not assign unevidenced threshold values"
        ),
    }
    receipt["payload_sha256"] = stable_sha256(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-evidence", type=Path)
    parser.add_argument("--good-bad-evidence", type=Path)
    parser.add_argument("--rl-pair-evidence", type=Path)
    parser.add_argument("--output-readiness", type=Path, required=True)
    args = parser.parse_args()
    planner, planner_sha = _load_optional(args.planner_evidence)
    good_bad, good_bad_sha = _load_optional(args.good_bad_evidence)
    rl_pairs, rl_sha = _load_optional(args.rl_pair_evidence)
    receipt = assess_candidate_evidence(
        planner=planner,
        good_bad=good_bad,
        rl_pairs=rl_pairs,
        evidence_files={
            "exact14x20_planner": planner_sha,
            "known_good_bad_trajectories": good_bad_sha,
            "fresh_deterministic_rl_pilot": rl_sha,
        },
    )
    args.output_readiness.parent.mkdir(parents=True, exist_ok=True)
    args.output_readiness.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_readiness)
    return 0 if receipt["formal_candidate_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
