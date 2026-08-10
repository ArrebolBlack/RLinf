#!/usr/bin/env python3
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

"""Export audited best-known Dynamic Benchmark trajectories from a frozen pool.

``optimal`` in this entrypoint means best-known under the supplied immutable
candidate manifest, reset manifest, escalation budget, and lexicographic score.
It does not claim a globally optimal continuous-control solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    # Keep the documented ``python examples/embodiment/<script>.py`` entrypoint
    # usable without relying on an ambient PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
    validate_compatibility_evidence,
)
from examples.embodiment.evaluate_dynamic_benchmark_expert import (
    _device,
    _load_inference_policy,
    _manifest_row,
    _sha256,
    _validate_policy_payload,
)
from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    _compose_residual_actions,
    _planner_actions,
    _policy_action,
)

CANDIDATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-candidates-v0.1"
CANDIDATE_SCHEMA_V2 = "rlinf-dynamic-benchmark-optimal-candidates-v0.2"
CANDIDATE_SCHEMAS = (CANDIDATE_SCHEMA, CANDIDATE_SCHEMA_V2)
EVALUATOR_IDENTITY_SCHEMA = "rlinf-dynamic-benchmark-quality-evaluator-identity-v0.1"
CALIBRATION_EVIDENCE_SCHEMA = (
    "rlinf-dynamic-benchmark-planner-calibration-evidence-v0.1"
)
CANDIDATE_RELEASE_SCHEMA = "rlinf-dynamic-benchmark-rld2-candidate-release-v0.2"
EXPORT_SCHEMA = "rlinf-dynamic-benchmark-optimal-export-v0.1"
ATTEMPT_SCHEMA = "rlinf-dynamic-benchmark-optimal-attempt-v0.3"
STATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-export-state-v0.1"
PROGRESS_SCHEMA = "rlinf-dynamic-benchmark-optimal-progress-v0.1"
RENDER_PARITY_SKIP_SCHEMA = "rlinf-dynamic-benchmark-render-parity-skip-v0.1"
LEGACY_SELECTION_MODE = "legacy-lexicographic"
PLANNER_PARETO_SELECTION_MODE = "planner-pareto"
SELECTION_MODES = (LEGACY_SELECTION_MODE, PLANNER_PARETO_SELECTION_MODE)
FIRST_ELIGIBLE_SEARCH_MODE = "first-eligible"
FULL_POOL_SEARCH_MODE = "full-pool"


def select_quality_v4_same_reset_pair(
    *,
    planner_attempt: Mapping[str, Any],
    rl_attempt: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the Qv4 absolute gates plus same-reset paired-Pareto selector."""

    from examples.embodiment.dynamic_benchmark_quality_v4 import (
        paired_pareto_winner,
    )

    return paired_pareto_winner(
        planner_attempt=planner_attempt,
        rl_attempt=rl_attempt,
        thresholds=thresholds,
    )


def export_quality_v4_winner(
    output: Path,
    *,
    source: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    """Write and independently re-gate the selected Qv4 winner's full source."""

    from examples.embodiment.dynamic_benchmark_evaluation_attempt import (
        materialize_quality_v4_winner_export,
    )

    return materialize_quality_v4_winner_export(
        output,
        source=source,
        attempt=attempt,
    )


CANDIDATE_SEARCH_MODES = (FIRST_ELIGIBLE_SEARCH_MODE, FULL_POOL_SEARCH_MODE)
PLANNER_DOMINANCE_SCHEMA = "rlinf-dynamic-benchmark-planner-dominance-v0.1"
QUALITY_V2_THRESHOLDS_SCHEMA = "se3-wam-trajectory-quality-v2-thresholds-v0.3"
QUALITY_V2_SUMMARY_SCHEMA = "se3-wam-trajectory-quality-v2"
QUALITY_V2_GATE_SCHEMA = "se3-wam-trajectory-quality-v2-gate-v0.1"
QUALITY_V2_DOMINANCE_SCHEMA = "rlinf-dynamic-benchmark-quality-v2-dominance-v0.1"
QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA = (
    "rld2-qa-planner-calibration-wave-receipt-v0.1"
)
QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES = 20
QUALITY_V2_MINIMUM_SUCCESSFUL_EPISODES = 8
QUALITY_V2_CALIBRATION_TASKS = (
    "p0_grasp",
    "t1_xyz",
    "t1_belt",
    "t1_so3",
    "t1_occ",
    "t2_trans",
    "t2_se3",
    "t3_phase",
    "t3_full",
    "t4_sphere",
    "t4_sphere_tabletop",
    "t4_slider",
    "t4_can",
    "t5_replan",
)
_QUALITY_V2_CORE_CHECK_SPECS = (
    (
        "action_second_difference",
        "action",
        "full_episode",
        "action.action_second_difference_l2_mean_per_transition",
        "action_l2",
    ),
    (
        "action_max_second_difference",
        "action",
        "full_episode",
        "action.action_max_second_difference_l2",
        "action_l2",
    ),
    (
        "action_total_variation",
        "action",
        "full_episode",
        "action.action_total_variation_l2_mean_per_transition",
        "action_l2",
    ),
    (
        "eef_translation_path_length",
        "eef_motion",
        "full_episode",
        "eef_motion.eef_translation_path_length_m",
        "translation_path_m",
    ),
    (
        "eef_rotation_path_length",
        "eef_motion",
        "full_episode",
        "eef_motion.eef_rotation_path_length_rad",
        "rotation_or_orientation_rad",
    ),
    (
        "eef_angular_jerk",
        "eef_motion",
        "full_episode",
        "eef_motion.eef_angular_jerk_max_rad_s3",
        "angular_jerk_rad_s3",
    ),
    (
        "eef_linear_jerk",
        "eef_motion",
        "full_episode",
        "eef_motion.eef_linear_jerk_max_m_s3",
        "linear_jerk_m_s3",
    ),
    (
        "eef_angular_jerk_rms",
        "eef_motion",
        "full_episode",
        "eef_motion.eef_angular_jerk_rms_rad_s3",
        "angular_jerk_rad_s3",
    ),
    (
        "eef_linear_jerk_rms",
        "eef_motion",
        "full_episode",
        "eef_motion.eef_linear_jerk_rms_m_s3",
        "linear_jerk_m_s3",
    ),
)
_QUALITY_V2_APPROACH_CHECK_SPEC = (
    "approach_verticality",
    "grasp_geometry",
    "acquisition_window",
    "approach_axis.approach_axis_error_max_rad",
    "rotation_or_orientation_rad",
)
_QUALITY_V2_ORIENTATION_CHECK_SPEC = (
    "orientation_reference",
    "grasp_geometry",
    "full_episode",
    "orientation_reference.orientation_reference_error_max_rad",
    "rotation_or_orientation_rad",
)
_QUALITY_V2_JAW_CHECK_SPECS = {
    "world_down_tool_axis": (
        "jaw_angle",
        "grasp_geometry",
        "acquisition_window",
        "jaw_axis.jaw_axis_error_max_rad",
        "rotation_or_orientation_rad",
    ),
    "reset_frozen_full_orientation": (
        "jaw_angle",
        "grasp_geometry",
        "post_hold",
        "jaw_axis.jaw_axis_error_max_rad",
        "rotation_or_orientation_rad",
    ),
}
BASE_DOMINANCE_METRICS = (
    "trajectory_completion",
    "completion_time_s",
    "control_steps",
    "action_l2_sum",
)
SELECTION_CONTRACT = (
    "success,safety,trajectory_completion,return,-control_steps,-action_l2_sum"
)
PLANNER_PARETO_SELECTION_CONTRACT = (
    "success,safety,t5-replan-causal-timing,quality-v2-absolute-gate,"
    "planner-pareto(trajectory_completion,"
    "task_quality.*,-completion_time_s,-control_steps,"
    "quality-v2.threshold-checks,-t5-impact-to-applied-correction-s,"
    "-action_l2_sum);return=diagnostic-only"
)
T5_ACTION_HISTORY_SCHEMA = "se3-wam-t5-issued-applied-action-history-v0.1"
T5_ACTION_VALUE_SEMANTIC_LABELS = (
    "arm_translation_x",
    "arm_translation_y",
    "arm_translation_z",
    "arm_rotation_x",
    "arm_rotation_y",
    "arm_rotation_z",
    "gripper",
)
T5_TIMING_VALUE_SEMANTIC_LABELS = (
    "impact_end_time_s",
    "first_contact_time_s",
    "control_hz",
)
T5_TIMING_COUNT_SEMANTIC_LABELS = (
    "expected_issued_action_count",
    "expected_action_delay_steps",
)


@dataclass(frozen=True)
class CandidateSpec:
    """One immutable planner or policy candidate."""

    candidate_id: str
    kind: str
    policy_path: Path | None = None
    policy_sha256: str | None = None
    stochastic: bool = False
    exploration_seed_offset: int = 0
    residual_scale: float | None = None
    provenance: Mapping[str, Any] | None = None


@dataclass
class LoadedCandidate:
    """Candidate plus the reconstructed model and normalizer, when applicable."""

    spec: CandidateSpec
    index: int
    config: dict[str, Any] | None = None
    state_schema: dict[str, Any] | None = None
    model: Any | None = None
    normalizer: RunningNormalizer | None = None


@dataclass(frozen=True)
class CompatibilityEvidence:
    """One portable policy/evaluator benchmark compatibility proof."""

    policy_benchmark_commit: str
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class CalibrationEvidence:
    """Validated replay calibration evidence to make dataset-local."""

    source_path: Path
    sha256: str


@dataclass(frozen=True)
class ProvenanceFile:
    """One source file copied into an immutable dataset provenance path."""

    source_path: Path
    relative_path: str
    sha256: str


def _candidate_identity(spec: CandidateSpec) -> dict[str, Any]:
    """Return a canonical-JSON-safe candidate identity."""

    return {
        "candidate_id": spec.candidate_id,
        "kind": spec.kind,
        "policy_path": None if spec.policy_path is None else str(spec.policy_path),
        "policy_sha256": spec.policy_sha256,
        "stochastic": spec.stochastic,
        "exploration_seed_offset": spec.exploration_seed_offset,
        "residual_scale": spec.residual_scale,
        "provenance": spec.provenance,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--candidate-release-manifest", type=Path)
    parser.add_argument("--expected-candidate-release-manifest-sha256")
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument(
        "--evaluator-benchmark-commit",
        help="required by candidate schema v0.2; evaluator SE3-WAM commit",
    )
    parser.add_argument("--rlinf-commit", help="candidate schema v0.1 policy commit")
    parser.add_argument(
        "--benchmark-commit", help="candidate schema v0.1 policy benchmark commit"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quality-v2-thresholds",
        type=Path,
        required=True,
        help="Frozen task×phase quality-v2 gate contract.",
    )
    parser.add_argument(
        "--expected-quality-v2-thresholds-sha256",
        required=True,
    )
    parser.add_argument(
        "--quality-v2-calibration-wave-receipt",
        type=Path,
        required=True,
        help="authoritative receipt source copied into dataset provenance",
    )
    parser.add_argument(
        "--expected-quality-v2-calibration-wave-receipt-sha256",
        required=True,
    )
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--manifest-seed", type=int, required=True)
    parser.add_argument("--accepted-episodes", type=int, default=100)
    parser.add_argument("--max-resets", type=int, default=200)
    parser.add_argument("--initial-k", type=int, default=8)
    parser.add_argument("--max-k", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument(
        "--candidate-search-mode",
        choices=CANDIDATE_SEARCH_MODES,
        default=FIRST_ELIGIBLE_SEARCH_MODE,
        help=(
            "first-eligible preserves K escalation; full-pool evaluates every "
            "manifest candidate for every reset"
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default=LEGACY_SELECTION_MODE,
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted export at its last committed reset boundary",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def _full_commit(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


def _expected_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("payload_sha256", None)
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _safe_dataset_relative_path(relative: Any, *, label: str) -> str:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty dataset-relative path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or pure.as_posix() != relative
        or "\\" in relative
    ):
        raise ValueError(f"unsafe {label}: {relative!r}")
    return relative


def _quality_v2_calibration_receipt_binding(
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    wave = thresholds.get("calibration_wave_receipt")
    if not isinstance(wave, Mapping) or wave.get("binding_status") != "bound":
        raise ValueError(
            "quality-v2 threshold has no bound calibration receipt artifact"
        )
    relative = _safe_dataset_relative_path(
        wave.get("relative_path"),
        label="quality-v2 calibration receipt path",
    )
    pure = PurePosixPath(relative)
    if pure.parts[0] != "provenance" or pure.name != "wave_receipt.json":
        raise ValueError(
            "quality-v2 calibration receipt must be under provenance/ and end in "
            "wave_receipt.json"
        )
    file_sha256 = _expected_sha256(
        wave.get("file_sha256"),
        "quality-v2 calibration receipt file SHA-256",
    )
    payload_sha256 = _expected_sha256(
        wave.get("payload_sha256"),
        "quality-v2 calibration receipt payload SHA-256",
    )
    legacy_sha256 = _expected_sha256(
        wave.get("sha256"),
        "quality-v2 calibration receipt compatibility SHA-256",
    )
    if file_sha256 != payload_sha256 or file_sha256 != legacy_sha256:
        raise ValueError(
            "canonical quality-v2 calibration receipt file/payload identities disagree"
        )
    return {
        "relative_path": relative,
        "file_sha256": file_sha256,
        "payload_sha256": payload_sha256,
    }


def _validate_quality_v2_calibration_receipt_artifact(
    thresholds: Mapping[str, Any],
    receipt_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_benchmark_commit: str | None = None,
) -> ProvenanceFile:
    """Reopen and cross-check the distributable exact-14 calibration receipt."""

    binding = _quality_v2_calibration_receipt_binding(thresholds)
    if expected_sha256 is not None:
        expected = _expected_sha256(
            expected_sha256,
            "expected quality-v2 calibration receipt SHA-256",
        )
        if expected != binding["file_sha256"]:
            raise ValueError(
                "expected calibration receipt SHA-256 disagrees with frozen thresholds"
            )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(
            "quality-v2 calibration receipt artifact is missing or symlinked"
        )
    receipt_bytes = receipt_path.read_bytes()
    file_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if file_sha256 != binding["file_sha256"]:
        raise ValueError("quality-v2 calibration receipt file SHA-256 mismatch")
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("quality-v2 calibration receipt is not UTF-8 JSON") from error
    if not isinstance(receipt, Mapping):
        raise TypeError("quality-v2 calibration receipt must be a mapping")
    canonical_bytes = _canonical_json_bytes(receipt)
    if receipt_bytes != canonical_bytes:
        raise ValueError("quality-v2 calibration receipt is not canonical JSON")
    if hashlib.sha256(canonical_bytes).hexdigest() != binding["payload_sha256"]:
        raise ValueError("quality-v2 calibration receipt payload SHA-256 mismatch")

    wave = thresholds["calibration_wave_receipt"]
    assert isinstance(wave, Mapping)
    expected_top_level = {
        "schema_version": QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA,
        "scientific_partition": "metric_calibration",
        "transport_split": "validation",
        "task_count": len(QUALITY_V2_CALIBRATION_TASKS),
        "episodes_per_task": QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES,
        "total_reset_count": (
            len(QUALITY_V2_CALIBRATION_TASKS) * QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES
        ),
        "task_order": list(QUALITY_V2_CALIBRATION_TASKS),
    }
    for key, value in expected_top_level.items():
        if receipt.get(key) != value or wave.get(key) != value:
            raise ValueError(f"quality-v2 calibration receipt/threshold {key} mismatch")
    for key in (
        "manifest_seed",
        "wave_contract_sha256",
        "predeclaration_receipt_sha256",
        "source_identity",
        "disjointness",
    ):
        if wave.get(key) != receipt.get(key):
            raise ValueError(f"quality-v2 calibration receipt/threshold {key} mismatch")
    source_identity = receipt.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("quality-v2 calibration receipt source identity is missing")
    receipt_benchmark_commit = _full_commit(
        "quality-v2 calibration receipt benchmark commit",
        source_identity.get("benchmark_commit"),
    )
    if (
        expected_benchmark_commit is not None
        and receipt_benchmark_commit
        != _full_commit(
            "expected quality-v2 calibration benchmark commit",
            expected_benchmark_commit,
        )
    ):
        raise ValueError(
            "quality-v2 calibration receipt benchmark commit differs from the "
            "authenticated evaluator benchmark commit"
        )
    raw_receipt_tasks = receipt.get("tasks")
    raw_binding_tasks = wave.get("tasks")
    if (
        not isinstance(raw_receipt_tasks, list)
        or not isinstance(raw_binding_tasks, list)
        or len(raw_receipt_tasks) != len(QUALITY_V2_CALIBRATION_TASKS)
        or len(raw_binding_tasks) != len(QUALITY_V2_CALIBRATION_TASKS)
    ):
        raise ValueError("quality-v2 calibration receipt task inventory is not exact14")
    identity_keys = (
        "task_contract_sha256",
        "task_receipt_sha256",
        "task_config_sha256",
        "task_quality_schema_version",
        "task_quality_schema_sha256",
        "reset_manifest_relative_path",
        "reset_manifest_sha256",
        "reset_identity_set_sha256",
        "reset_row_set_sha256",
        "evaluation_relative_path",
        "evaluation_sha256",
        "evaluation_payload_sha256",
    )
    for ordinal, (task_id, receipt_task, binding_task) in enumerate(
        zip(
            QUALITY_V2_CALIBRATION_TASKS,
            raw_receipt_tasks,
            raw_binding_tasks,
            strict=True,
        )
    ):
        if not isinstance(receipt_task, Mapping) or not isinstance(
            binding_task, Mapping
        ):
            raise TypeError("quality-v2 calibration receipt task row must be a mapping")
        if (
            receipt_task.get("ordinal") != ordinal
            or binding_task.get("ordinal") != ordinal
            or receipt_task.get("task_id") != task_id
            or binding_task.get("task_id") != task_id
            or receipt_task.get("reset_count") != QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES
            or binding_task.get("reset_identity_count")
            != QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES
        ):
            raise ValueError("quality-v2 calibration receipt task order/count mismatch")
        for key in identity_keys:
            if binding_task.get(key) != receipt_task.get(key):
                raise ValueError(
                    f"quality-v2 calibration task {task_id} {key} mismatch"
                )
        for key in (
            "task_contract_sha256",
            "task_receipt_sha256",
            "task_config_sha256",
            "task_quality_schema_sha256",
            "reset_manifest_sha256",
            "reset_identity_set_sha256",
            "reset_row_set_sha256",
            "evaluation_sha256",
            "evaluation_payload_sha256",
        ):
            _expected_sha256(
                receipt_task.get(key),
                f"quality-v2 calibration task {task_id} {key}",
            )
    return ProvenanceFile(
        source_path=receipt_path.resolve(),
        relative_path=str(binding["relative_path"]),
        sha256=str(binding["file_sha256"]),
    )


def _rewrite_last_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Atomically replace the last committed row of a JSONL file."""

    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} has no rows to rewrite")
    lines[-1] = json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
    body = "\n".join(lines) + "\n"
    temporary = path.with_suffix(path.suffix + ".drop.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _quality_score(record: Mapping[str, Any]) -> tuple[float, ...]:
    """Return the frozen quality score, excluding deterministic identity tie-break."""

    return (
        float(bool(record["success"])),
        float(not bool(record["safety_failure"])),
        float(record["trajectory_completion"]),
        float(record["return"]),
        -float(record["control_steps"]),
        -float(record["action_l2_sum"]),
    )


def _eligible(record: Mapping[str, Any]) -> bool:
    replay = record.get("replay_validation")
    for key in ("success", "safety_failure", "finite_and_bounded"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"attempt {key} must be boolean")
    if not isinstance(replay, Mapping) or not isinstance(replay.get("passed"), bool):
        raise ValueError("attempt replay-validation passed flag must be boolean")
    quality_gate = record.get("quality_v2_gate")
    if not isinstance(quality_gate, Mapping) or not isinstance(
        quality_gate.get("passed"), bool
    ):
        raise ValueError("quality-v2 gate passed flag must be boolean")
    if not quality_gate["passed"]:
        return False
    causal_timing_passed = True
    if record.get("schema_version") == ATTEMPT_SCHEMA:
        issued_equals_applied = record.get("issued_equals_applied")
        if not isinstance(issued_equals_applied, bool):
            raise ValueError(
                "attempt issued_equals_applied declaration must be boolean"
            )
        if record.get("task_id") == "t5_replan":
            if issued_equals_applied:
                raise ValueError(
                    "T5 Replan must preserve distinct issued/applied histories"
                )
            raw_causal_gate = record.get("t5_replan_causal_timing_passed")
            if not isinstance(raw_causal_gate, bool):
                raise ValueError("T5 Replan causal-timing gate must be boolean")
            causal_timing_passed = raw_causal_gate
        elif not issued_equals_applied:
            raise ValueError("non-T5 attempt must declare issued_equals_applied=true")
    return bool(
        record.get("success")
        and not record.get("safety_failure")
        and record.get("finite_and_bounded")
        and replay.get("passed")
        and causal_timing_passed
    )


def _selection_contract(selection_mode: str) -> str:
    if selection_mode == LEGACY_SELECTION_MODE:
        return SELECTION_CONTRACT
    if selection_mode == PLANNER_PARETO_SELECTION_MODE:
        return PLANNER_PARETO_SELECTION_CONTRACT
    raise ValueError(f"unsupported selection mode {selection_mode!r}")


def _metric_contract(
    value: Any,
    *,
    metric_name: str,
    direction: str,
) -> dict[str, Any]:
    """Validate one replay-calibrated planner-dominance metric contract."""

    if not isinstance(value, Mapping):
        raise ValueError(f"planner-dominance metric {metric_name!r} must be a mapping")
    common = {
        "direction",
        "max_observed_replay_drift",
        "scientific_resolution",
    }
    expected = (
        common | {"numeric_floor_absolute", "numeric_floor_relative"}
        if metric_name == "action_l2_sum"
        else common | {"numeric_floor"}
    )
    if set(value) != expected or value.get("direction") != direction:
        raise ValueError(f"planner-dominance metric {metric_name!r} contract mismatch")
    drift = _finite_number(
        value["max_observed_replay_drift"],
        f"planner-dominance metric {metric_name!r} drift",
    )
    resolution = _finite_number(
        value["scientific_resolution"],
        f"planner-dominance metric {metric_name!r} scientific resolution",
    )
    if drift < 0.0:
        raise ValueError(f"planner-dominance metric {metric_name!r} drift is invalid")
    if resolution <= 0.0:
        raise ValueError(
            f"planner-dominance metric {metric_name!r} resolution is invalid"
        )
    normalized = {
        "direction": direction,
        "max_observed_replay_drift": drift,
        "scientific_resolution": resolution,
    }
    if metric_name == "action_l2_sum":
        absolute = _finite_number(
            value["numeric_floor_absolute"],
            "action_l2_sum absolute numeric floor",
        )
        relative = _finite_number(
            value["numeric_floor_relative"],
            "action_l2_sum relative numeric floor",
        )
        if absolute != 1.0e-6 or relative != 1.0e-6:
            raise ValueError(
                "action_l2_sum numeric floor must be max(1e-6, 1e-6*|planner|)"
            )
        normalized.update(
            numeric_floor_absolute=absolute,
            numeric_floor_relative=relative,
        )
    else:
        floor = _finite_number(
            value["numeric_floor"],
            f"planner-dominance metric {metric_name!r} numeric floor",
        )
        expected_floor = 0.0 if metric_name == "control_steps" else 1.0e-6
        if floor != expected_floor:
            raise ValueError(
                f"planner-dominance metric {metric_name!r} numeric floor mismatch"
            )
        if metric_name == "control_steps" and resolution < 1.0:
            raise ValueError(
                "control_steps strict scientific resolution must be at least one"
            )
        if metric_name == "completion_time_s" and resolution != 0.002:
            raise ValueError(
                "completion_time_s scientific resolution must be one 0.002 s physics step"
            )
        normalized["numeric_floor"] = floor
    return normalized


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number, not bool or string")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validate_planner_dominance_contract(
    payload: Mapping[str, Any],
    *,
    task: str,
    selection_mode: str,
) -> dict[str, Any] | None:
    """Validate the task/backend-specific utility and replay calibration contract."""

    raw = payload.get("planner_dominance")
    if selection_mode == LEGACY_SELECTION_MODE:
        if raw is not None:
            raise ValueError("planner_dominance requires selection-mode=planner-pareto")
        return None
    if selection_mode != PLANNER_PARETO_SELECTION_MODE or not isinstance(raw, Mapping):
        raise ValueError(
            "planner-pareto requires a planner_dominance candidate-manifest contract"
        )
    expected_keys = {
        "schema_version",
        "task",
        "backend_id",
        "quality_schema",
        "calibration",
        "metrics",
        "tie_break_order",
    }
    if (
        set(raw) != expected_keys
        or raw.get("schema_version") != PLANNER_DOMINANCE_SCHEMA
    ):
        raise ValueError("planner-dominance schema or field inventory mismatch")
    if raw.get("task") != task:
        raise ValueError("planner-dominance task identity mismatch")
    backend_id = raw.get("backend_id")
    if (
        not isinstance(backend_id, str)
        or not backend_id
        or backend_id.strip() != backend_id
    ):
        raise ValueError("planner-dominance backend identity is missing")
    quality_schema = raw.get("quality_schema")
    if not isinstance(quality_schema, Mapping) or set(quality_schema) != {
        "schema_version",
        "task_id",
        "task_config_sha256",
        "components",
        "schema_sha256",
    }:
        raise ValueError("planner-dominance quality schema inventory is invalid")
    quality_schema_version = quality_schema.get("schema_version")
    if not isinstance(quality_schema_version, str) or not quality_schema_version:
        raise ValueError("planner-dominance quality schema version is missing")
    if quality_schema.get("task_id") != task:
        raise ValueError("planner-dominance quality schema task identity mismatch")
    task_config_sha256 = _expected_sha256(
        quality_schema.get("task_config_sha256"),
        "planner-dominance task config SHA-256",
    )
    quality_schema_sha256 = _expected_sha256(
        quality_schema.get("schema_sha256"),
        "planner-dominance quality schema SHA-256",
    )
    components = quality_schema.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("planner-dominance quality components are missing")
    normalized_components: list[dict[str, Any]] = []
    component_names: set[str] = set()
    for metadata in components:
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "name",
            "direction",
            "unit",
            "scientific_resolution",
            "reducer",
            "source",
            "description",
        }:
            raise ValueError("planner-dominance quality component metadata is invalid")
        name = metadata.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or "." in name
            or name in component_names
        ):
            raise ValueError("planner-dominance quality component name is invalid")
        direction = metadata.get("direction")
        unit = metadata.get("unit")
        reducer = metadata.get("reducer")
        source = metadata.get("source")
        description = metadata.get("description")
        resolution = _finite_number(
            metadata.get("scientific_resolution"),
            f"quality component {name!r} scientific resolution",
        )
        if direction not in {"minimize", "maximize"}:
            raise ValueError(f"quality component {name!r} direction is invalid")
        if not isinstance(unit, str) or not unit:
            raise ValueError(f"quality component {name!r} unit is missing")
        if reducer not in {"minimum", "maximum", "terminal"}:
            raise ValueError(f"quality component {name!r} reducer is invalid")
        if not isinstance(source, str) or not source or source.strip() != source:
            raise ValueError(f"quality component {name!r} source is missing")
        if (
            not isinstance(description, str)
            or not description
            or description.strip() != description
        ):
            raise ValueError(f"quality component {name!r} description is missing")
        if resolution <= 0.0:
            raise ValueError(f"quality component {name!r} resolution is invalid")
        component_names.add(name)
        # Preserve the canonical upstream row byte-for-byte at the JSON-value
        # level.  The list order and every source/description field are part of
        # ``schema_sha256``; the name index used below is deliberately separate.
        normalized_components.append(dict(metadata))
    normalized_quality_schema = {
        "schema_version": quality_schema_version,
        "task_id": task,
        "task_config_sha256": task_config_sha256,
        "components": normalized_components,
        "schema_sha256": quality_schema_sha256,
    }
    recomputed_quality_schema_sha256 = _payload_sha256(
        {
            "schema_version": quality_schema_version,
            "task_id": task,
            "task_config_sha256": task_config_sha256,
            "components": list(components),
        }
    )
    if quality_schema_sha256 != recomputed_quality_schema_sha256:
        raise ValueError("planner-dominance quality schema SHA-256 does not recompute")
    calibration = raw.get("calibration")
    if not isinstance(calibration, Mapping) or set(calibration) != {
        "replay_count",
        "reset_episode_id",
        "reset_manifest_sha256",
        "evidence_path",
        "evidence_sha256",
    }:
        raise ValueError("planner-dominance calibration evidence inventory mismatch")
    replay_count = calibration.get("replay_count")
    reset_episode_id = calibration.get("reset_episode_id")
    if (
        isinstance(replay_count, bool)
        or not isinstance(replay_count, int)
        or replay_count < 3
    ):
        raise ValueError(
            "planner-dominance calibration requires at least three replays"
        )
    if (
        not isinstance(reset_episode_id, str)
        or not reset_episode_id
        or reset_episode_id.strip() != reset_episode_id
    ):
        raise ValueError("planner-dominance calibration reset identity is missing")
    evidence_path = calibration.get("evidence_path")
    if (
        not isinstance(evidence_path, str)
        or not evidence_path
        or evidence_path.strip() != evidence_path
    ):
        raise ValueError("planner-dominance calibration evidence path is missing")
    normalized_calibration = {
        "replay_count": replay_count,
        "reset_episode_id": reset_episode_id,
        "reset_manifest_sha256": _expected_sha256(
            calibration.get("reset_manifest_sha256"),
            "planner-dominance calibration reset manifest SHA-256",
        ),
        "evidence_path": evidence_path,
        "evidence_sha256": _expected_sha256(
            calibration.get("evidence_sha256"),
            "planner-dominance calibration evidence SHA-256",
        ),
    }
    metrics = raw.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "trajectory_completion",
        "task_quality",
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    }:
        raise ValueError("planner-dominance metric inventory mismatch")
    quality_metrics = metrics.get("task_quality")
    component_index = {row["name"]: row for row in normalized_components}
    if not isinstance(quality_metrics, Mapping) or set(quality_metrics) != set(
        component_index
    ):
        raise ValueError("planner-dominance quality metric mapping is incomplete")
    normalized_metrics = {
        "trajectory_completion": _metric_contract(
            metrics["trajectory_completion"],
            metric_name="trajectory_completion",
            direction="max",
        ),
        "task_quality": {
            name: _metric_contract(
                quality_metrics[name],
                metric_name=f"task_quality.{name}",
                direction=(
                    "max" if component_index[name]["direction"] == "maximize" else "min"
                ),
            )
            for name in component_index
        },
        "completion_time_s": _metric_contract(
            metrics["completion_time_s"],
            metric_name="completion_time_s",
            direction="min",
        ),
        "control_steps": _metric_contract(
            metrics["control_steps"],
            metric_name="control_steps",
            direction="min",
        ),
        "action_l2_sum": _metric_contract(
            metrics["action_l2_sum"],
            metric_name="action_l2_sum",
            direction="min",
        ),
    }
    for name in component_index:
        if (
            normalized_metrics["task_quality"][name]["scientific_resolution"]
            != component_index[name]["scientific_resolution"]
        ):
            raise ValueError(
                f"quality component {name!r} calibration resolution differs from schema"
            )
    metric_keys = [
        "trajectory_completion",
        *(f"task_quality.{name}" for name in component_index),
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    ]
    tie_break_order = raw.get("tie_break_order")
    if (
        not isinstance(tie_break_order, list)
        or len(tie_break_order) != len(metric_keys)
        or set(tie_break_order) != set(metric_keys)
    ):
        raise ValueError(
            "planner-dominance tie-break order must name every quality metric once"
        )
    normalized = {
        "schema_version": PLANNER_DOMINANCE_SCHEMA,
        "task": task,
        "backend_id": backend_id,
        "quality_schema": normalized_quality_schema,
        "calibration": normalized_calibration,
        "metrics": normalized_metrics,
        "tie_break_order": list(tie_break_order),
    }
    normalized["payload_sha256"] = _payload_sha256(normalized)
    return normalized


def _dominance_metric_keys(contract: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        "trajectory_completion",
        *(
            f"task_quality.{component['name']}"
            for component in contract["quality_schema"]["components"]
        ),
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    )


def _metric_spec(contract: Mapping[str, Any], metric_name: str) -> Mapping[str, Any]:
    if metric_name.startswith("task_quality."):
        return contract["metrics"]["task_quality"][metric_name.split(".", 1)[1]]
    return contract["metrics"][metric_name]


def _metric_value(record: Mapping[str, Any], metric_name: str) -> float:
    if metric_name.startswith("task_quality."):
        component = metric_name.split(".", 1)[1]
        summary = record.get("task_quality")
        values = summary.get("components") if isinstance(summary, Mapping) else None
        component_row = values.get(component) if isinstance(values, Mapping) else None
        if not isinstance(component_row, Mapping) or "value" not in component_row:
            raise ValueError(
                f"task quality mapping gap: attempt is missing component {component!r}"
            )
        value = _finite_number(
            component_row["value"],
            f"task quality component {component!r} value",
        )
    else:
        raw_value = record.get(metric_name)
        if metric_name == "control_steps":
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 1
            ):
                raise ValueError(
                    "planner-dominance control_steps must be a positive integer"
                )
            value = float(raw_value)
        else:
            value = _finite_number(
                raw_value,
                f"planner-dominance metric {metric_name!r}",
            )
            if metric_name == "trajectory_completion" and not 0.0 <= value <= 1.0:
                raise ValueError("trajectory_completion must be in [0, 1]")
            if metric_name in {"completion_time_s", "action_l2_sum"} and value < 0.0:
                raise ValueError(
                    f"planner-dominance metric {metric_name!r} is negative"
                )
    return value


def _validate_attempt_quality(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    summary = record.get("task_quality")
    schema = contract["quality_schema"]
    if not isinstance(summary, Mapping):
        raise ValueError(
            "task quality mapping gap: attempt has no task_quality summary"
        )
    if set(summary) != {
        "schema_version",
        "episode_id",
        "task_id",
        "evaluator_backend_id",
        "schema_sha256",
        "physics_sample_count",
        "terminal",
        "components",
        "summary_sha256",
    }:
        raise ValueError("task quality summary field inventory mismatch")
    if (
        summary.get("schema_version") != schema["schema_version"]
        or summary.get("task_id") != contract["task"]
        or summary.get("schema_sha256") != schema["schema_sha256"]
        or summary.get("episode_id") != record.get("episode_id")
        or summary.get("evaluator_backend_id") != contract["backend_id"]
    ):
        raise ValueError("task quality summary identity mismatch")
    if summary.get("terminal") is not True:
        raise ValueError("task quality summary must be terminal")
    summary_sha256 = _expected_sha256(
        summary.get("summary_sha256"),
        "task quality summary SHA-256",
    )
    summary_payload = dict(summary)
    summary_payload.pop("summary_sha256")
    if summary_sha256 != _payload_sha256(summary_payload):
        raise ValueError("task quality summary SHA-256 does not recompute")
    sample_count = summary.get("physics_sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("task quality summary physics sample count is invalid")
    values = summary.get("components")
    schema_components = schema["components"]
    expected_names = [component["name"] for component in schema_components]
    expected = set(expected_names)
    if not isinstance(values, Mapping) or set(values) != expected:
        missing = sorted(
            expected - set(values) if isinstance(values, Mapping) else expected
        )
        extra = sorted(set(values) - expected if isinstance(values, Mapping) else set())
        raise ValueError(f"task quality mapping gap: missing={missing}, extra={extra}")
    if list(values) != expected_names:
        raise ValueError(
            "task quality component order differs from the canonical schema"
        )
    for frozen_schema in schema_components:
        name = frozen_schema["name"]
        component = values[name]
        frozen = {
            key: frozen_schema[key]
            for key in ("direction", "unit", "scientific_resolution", "reducer")
        }
        if not isinstance(component, Mapping) or set(component) != {
            "value",
            "direction",
            "unit",
            "scientific_resolution",
            "reducer",
        }:
            raise ValueError(f"task quality component {name!r} inventory mismatch")
        _finite_number(
            component["scientific_resolution"],
            f"task quality component {name!r} scientific resolution",
        )
        if any(component.get(key) != frozen[key] for key in frozen):
            raise ValueError(f"task quality component {name!r} metadata mismatch")
        _metric_value(record, f"task_quality.{name}")


def _metric_thresholds(
    metric_name: str,
    reference_value: float,
    spec: Mapping[str, Any],
) -> tuple[float, float]:
    if metric_name == "action_l2_sum":
        floor = max(
            float(spec["numeric_floor_absolute"]),
            float(spec["numeric_floor_relative"]) * abs(reference_value),
        )
    else:
        floor = float(spec["numeric_floor"])
    epsilon = max(floor, 2.0 * float(spec["max_observed_replay_drift"]))
    strict_margin = max(float(spec["scientific_resolution"]), 2.0 * epsilon)
    if metric_name == "control_steps":
        strict_margin = max(1.0, strict_margin)
    return epsilon, strict_margin


def _quality_v2_expected_check_specs(
    task_contract: Mapping[str, Any],
) -> tuple[list[dict[str, str]], str, str]:
    """Return the exact task-derived Qv3 check inventory.

    The calibrator freezes nine whole-episode control/motion checks for every
    task.  The task contract then selects exactly one grasp-orientation check
    and determines whether a jaw-angle check is applicable.  Keeping this
    inventory here makes every formal consumer fail closed on the same
    phase/metric/key/group identity instead of merely checking a count.
    """

    orientation_mode = task_contract.get("orientation_mode")
    if orientation_mode == "world_down_tool_axis":
        orientation_spec = _QUALITY_V2_APPROACH_CHECK_SPEC
    elif orientation_mode == "reset_frozen_full_orientation":
        orientation_spec = _QUALITY_V2_ORIENTATION_CHECK_SPEC
    else:
        raise ValueError(
            "quality-v2 task orientation_mode must be "
            "'world_down_tool_axis' or 'reset_frozen_full_orientation'"
        )

    jaw_axis_mode = task_contract.get("jaw_axis_mode")
    if (
        not isinstance(jaw_axis_mode, str)
        or not jaw_axis_mode
        or jaw_axis_mode.strip() != jaw_axis_mode
    ):
        raise ValueError("quality-v2 task jaw_axis_mode is invalid")

    raw_specs = [*_QUALITY_V2_CORE_CHECK_SPECS, orientation_spec]
    if jaw_axis_mode != "unconstrained":
        raw_specs.append(_QUALITY_V2_JAW_CHECK_SPECS[orientation_mode])
    expected_count = 10 if jaw_axis_mode == "unconstrained" else 11
    if len(raw_specs) != expected_count:
        raise AssertionError("internal quality-v2 canonical inventory is inconsistent")
    specs = [
        {
            "key": key,
            "group": group,
            "phase": phase,
            "metric": metric,
            "paired_comparison_family": family,
        }
        for key, group, phase, metric, family in raw_specs
    ]
    return specs, orientation_mode, jaw_axis_mode


def _quality_v2_dominance_contract(
    payload: Mapping[str, Any],
    *,
    task: str,
    thresholds_sha256: str,
    require_formal_freeze: bool = True,
) -> dict[str, Any]:
    """Derive every paired Qv2 comparison dimension from the frozen checks."""

    threshold_sha256 = _expected_sha256(
        thresholds_sha256,
        "quality-v2 threshold SHA-256",
    )
    if payload.get("schema_version") != QUALITY_V2_THRESHOLDS_SCHEMA:
        raise ValueError(
            "planner-pareto requires the frozen quality-v2 threshold schema v0.3"
        )
    formal_freeze_eligible = payload.get("formal_freeze_eligible")
    if not isinstance(formal_freeze_eligible, bool):
        raise ValueError(
            "quality-v2 threshold formal-freeze eligibility must be boolean"
        )
    if require_formal_freeze and not formal_freeze_eligible:
        raise ValueError(
            "quality-v2 threshold contract is not eligible for formal freeze"
        )
    if require_formal_freeze:
        minimum_attempted = payload.get("minimum_attempted_episodes")
        minimum_successful = payload.get("minimum_successful_episodes")
        wave = payload.get("calibration_wave_receipt")
        if (
            payload.get("calibration_status") != "frozen"
            or isinstance(minimum_attempted, bool)
            or not isinstance(minimum_attempted, int)
            or minimum_attempted < QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES
            or isinstance(minimum_successful, bool)
            or not isinstance(minimum_successful, int)
            or minimum_successful < QUALITY_V2_MINIMUM_SUCCESSFUL_EPISODES
            or not isinstance(wave, Mapping)
            or wave.get("binding_status") != "bound"
            or wave.get("schema_version") != QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA
            or wave.get("scientific_partition") != "metric_calibration"
            or wave.get("task_count") != 14
            or wave.get("episodes_per_task") != 20
            or wave.get("total_reset_count") != 280
        ):
            raise ValueError(
                "quality-v2 threshold contract has invalid formal calibration provenance"
            )
        _expected_sha256(
            wave.get("sha256"),
            "quality-v2 calibration wave receipt SHA-256",
        )
    tasks = payload.get("tasks")
    task_contract = tasks.get(task) if isinstance(tasks, Mapping) else None
    if not isinstance(task_contract, Mapping):
        raise ValueError(f"quality-v2 threshold contract has no task {task!r}")
    if require_formal_freeze:
        provenance = task_contract.get("provenance")
        attempted = (
            provenance.get("attempted_episode_count")
            if isinstance(provenance, Mapping)
            else None
        )
        successful = (
            provenance.get("successful_episode_count")
            if isinstance(provenance, Mapping)
            else None
        )
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("formal_freeze_eligible") is not True
            or isinstance(attempted, bool)
            or not isinstance(attempted, int)
            or attempted < minimum_attempted
            or isinstance(successful, bool)
            or not isinstance(successful, int)
            or successful < minimum_successful
        ):
            raise ValueError(
                f"quality-v2 threshold task {task!r} has invalid formal calibration provenance"
            )
    expected_specs, orientation_mode, jaw_axis_mode = _quality_v2_expected_check_specs(
        task_contract
    )
    raw_checks = task_contract.get("checks")
    if not isinstance(raw_checks, list) or len(raw_checks) != len(expected_specs):
        raise ValueError(
            f"quality-v2 checks for {task!r} must contain exactly "
            f"{len(expected_specs)} task-derived entries"
        )

    paired_fields = (
        "paired_nonworse_absolute_tolerance",
        "paired_nonworse_relative_tolerance",
        "paired_strict_improvement_absolute",
        "paired_strict_improvement_relative",
    )
    required_fields = {
        "phase",
        "metric",
        "max",
        "direction",
        "paired_comparison_family",
        *paired_fields,
    }
    checks_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_check in enumerate(raw_checks):
        if not isinstance(raw_check, Mapping) or not required_fields.issubset(
            raw_check
        ):
            raise ValueError(
                f"quality-v2 check {index} is missing frozen paired-comparison metadata"
            )
        phase = raw_check.get("phase")
        metric = raw_check.get("metric")
        family = raw_check.get("paired_comparison_family")
        if (
            not isinstance(phase, str)
            or not phase
            or phase.strip() != phase
            or "." in phase
        ):
            raise ValueError(f"quality-v2 check {index} phase is invalid")
        if (
            not isinstance(metric, str)
            or not metric
            or metric.strip() != metric
            or any(not part for part in metric.split("."))
        ):
            raise ValueError(f"quality-v2 check {index} metric is invalid")
        if raw_check.get("direction") != "minimize":
            raise ValueError(
                f"quality-v2 check {phase}.{metric} must use direction='minimize'"
            )
        if not isinstance(family, str) or not family or family.strip() != family:
            raise ValueError(
                f"quality-v2 check {phase}.{metric} comparison family is invalid"
            )
        identity = (phase, metric)
        if identity in checks_by_identity:
            raise ValueError(f"duplicate quality-v2 check {phase}.{metric}")
        maximum = _finite_number(
            raw_check.get("max"),
            f"quality-v2 check {phase}.{metric} maximum",
        )
        normalized_paired = {
            name: _finite_number(
                raw_check.get(name),
                f"quality-v2 check {phase}.{metric} {name}",
            )
            for name in paired_fields
        }
        if maximum < 0.0 or any(value < 0.0 for value in normalized_paired.values()):
            raise ValueError(
                f"quality-v2 check {phase}.{metric} thresholds must be non-negative"
            )
        if (
            normalized_paired["paired_strict_improvement_absolute"] == 0.0
            and normalized_paired["paired_strict_improvement_relative"] == 0.0
        ):
            raise ValueError(
                f"quality-v2 check {phase}.{metric} has no strict-improvement resolution"
            )
        checks_by_identity[identity] = {
            "maximum": maximum,
            "paired_comparison_family": family,
            **normalized_paired,
        }

    expected_by_identity = {
        (spec["phase"], spec["metric"]): spec for spec in expected_specs
    }
    actual_identities = set(checks_by_identity)
    expected_identities = set(expected_by_identity)
    if actual_identities != expected_identities:
        missing = sorted(expected_identities - actual_identities)
        extra = sorted(actual_identities - expected_identities)
        raise ValueError(
            "quality-v2 v0.3 task-derived check inventory mismatch: "
            f"missing={missing}, extra={extra}"
        )
    metrics: list[dict[str, Any]] = []
    for identity, frozen in checks_by_identity.items():
        spec = expected_by_identity[identity]
        expected_family = spec["paired_comparison_family"]
        if frozen["paired_comparison_family"] != expected_family:
            raise ValueError(
                f"quality-v2 check {identity[0]}.{identity[1]} comparison family "
                f"must be {expected_family!r}"
            )
        metrics.append(
            {
                "name": f"quality_v2.{identity[0]}.{identity[1]}",
                "key": spec["key"],
                "group": spec["group"],
                "phase": identity[0],
                "metric": identity[1],
                "maximum": frozen["maximum"],
                "direction": "minimize",
                "paired_comparison_family": expected_family,
                **{field: frozen[field] for field in paired_fields},
            }
        )
    normalized = {
        "schema_version": QUALITY_V2_DOMINANCE_SCHEMA,
        "threshold_schema_version": QUALITY_V2_THRESHOLDS_SCHEMA,
        "threshold_sha256": threshold_sha256,
        "formal_freeze_eligible": formal_freeze_eligible,
        "task": task,
        "orientation_mode": orientation_mode,
        "jaw_axis_mode": jaw_axis_mode,
        "metrics": metrics,
    }
    normalized["payload_sha256"] = _payload_sha256(normalized)
    return normalized


def _quality_v2_metric_value(
    record: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> float:
    summary = record.get("quality_v2")
    if not isinstance(summary, Mapping):
        raise ValueError("quality-v2 mapping gap: attempt has no quality_v2 summary")
    phase = str(spec["phase"])
    if phase == "full_episode":
        value: Any = summary
    else:
        phases = summary.get("phases")
        value = phases.get(phase) if isinstance(phases, Mapping) else None
        if not isinstance(value, Mapping):
            raise ValueError(
                f"quality-v2 mapping gap: attempt is missing phase {phase!r}"
            )
    metric = str(spec["metric"])
    for part in metric.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(
                f"quality-v2 mapping gap: attempt is missing metric {phase}.{metric}"
            )
        value = value[part]
    result = _finite_number(value, f"quality-v2 metric {phase}.{metric}")
    if result < 0.0:
        raise ValueError(f"quality-v2 metric {phase}.{metric} must be non-negative")
    return result


def _validate_quality_v2_attempt(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, float]:
    """Validate the summary/gate/hash binding used by planner-pareto selection."""

    if record.get("schema_version") != ATTEMPT_SCHEMA:
        raise ValueError("planner-pareto requires attempt schema v0.3")
    if record.get("task_id") != contract.get("task"):
        raise ValueError("quality-v2 attempt task identity mismatch")
    summary = record.get("quality_v2")
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version") != QUALITY_V2_SUMMARY_SCHEMA
    ):
        raise ValueError("quality-v2 summary schema mismatch")
    summary_sha256 = _expected_sha256(
        record.get("quality_v2_sha256"),
        "quality-v2 summary SHA-256",
    )
    if summary_sha256 != _payload_sha256(summary):
        raise ValueError("quality-v2 summary SHA-256 does not recompute")

    gate = record.get("quality_v2_gate")
    expected_gate_keys = {
        "schema_version",
        "contract_schema_version",
        "contract_sha256",
        "task_id",
        "passed",
        "checks",
    }
    if not isinstance(gate, Mapping) or set(gate) != expected_gate_keys:
        raise ValueError("quality-v2 gate field inventory mismatch")
    if (
        gate.get("schema_version") != QUALITY_V2_GATE_SCHEMA
        or gate.get("contract_schema_version")
        != contract.get("threshold_schema_version")
        or gate.get("contract_sha256") != contract.get("threshold_sha256")
        or gate.get("task_id") != contract.get("task")
        or gate.get("passed") is not True
    ):
        raise ValueError("quality-v2 gate identity or absolute decision mismatch")
    raw_gate_checks = gate.get("checks")
    metric_specs = contract.get("metrics")
    if (
        not isinstance(raw_gate_checks, list)
        or not isinstance(metric_specs, list)
        or len(raw_gate_checks) != len(metric_specs)
    ):
        raise ValueError("quality-v2 gate check inventory mismatch")

    values: dict[str, float] = {}
    expected_check_keys = {"metric", "phase", "actual", "max", "passed"}
    for raw_gate_check, spec in zip(raw_gate_checks, metric_specs, strict=True):
        if (
            not isinstance(raw_gate_check, Mapping)
            or set(raw_gate_check) != expected_check_keys
        ):
            raise ValueError("quality-v2 gate check field inventory mismatch")
        actual = _quality_v2_metric_value(record, spec)
        gate_actual = _finite_number(
            raw_gate_check.get("actual"),
            f"quality-v2 gate actual {spec['name']}",
        )
        gate_maximum = _finite_number(
            raw_gate_check.get("max"),
            f"quality-v2 gate maximum {spec['name']}",
        )
        if (
            raw_gate_check.get("phase") != spec["phase"]
            or raw_gate_check.get("metric") != spec["metric"]
            or gate_actual != actual
            or gate_maximum != spec["maximum"]
            or raw_gate_check.get("passed") is not True
            or actual > float(spec["maximum"])
        ):
            raise ValueError(f"quality-v2 gate check {spec['name']} does not recompute")
        values[str(spec["name"])] = actual
    return values


def _t5_causal_latency(record: Mapping[str, Any]) -> float | None:
    """Return the required T5 causal latency metric; non-T5 tasks have none."""

    if record.get("task_id") != "t5_replan":
        if (
            record.get("impact_end_to_first_qualifying_applied_correction_s")
            is not None
        ):
            raise ValueError("non-T5 attempt cannot declare a T5 causal latency")
        return None
    if record.get("t5_replan_causal_timing_passed") is not True:
        raise ValueError("T5 planner comparison requires a passing causal-timing gate")
    value = _finite_number(
        record.get("impact_end_to_first_qualifying_applied_correction_s"),
        "T5 impact-end-to-applied-correction latency",
    )
    if value < 0.0:
        raise ValueError("T5 causal correction latency must be nonnegative")
    return value


def _quality_v2_metric_thresholds(
    reference_value: float,
    spec: Mapping[str, Any],
) -> tuple[float, float]:
    tolerance = max(
        float(spec["paired_nonworse_absolute_tolerance"]),
        float(spec["paired_nonworse_relative_tolerance"]) * abs(reference_value),
    )
    strict_margin = max(
        float(spec["paired_strict_improvement_absolute"]),
        float(spec["paired_strict_improvement_relative"]) * abs(reference_value),
        2.0 * tolerance,
    )
    return tolerance, strict_margin


def _planner_pareto_dominates(
    record: Mapping[str, Any],
    reference: Mapping[str, Any],
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
) -> bool:
    """Return whether record is non-worse on every frozen quality metric."""

    _validate_attempt_quality(record, contract)
    _validate_attempt_quality(reference, contract)
    quality_v2_values = _validate_quality_v2_attempt(record, quality_v2_contract)
    reference_quality_v2_values = _validate_quality_v2_attempt(
        reference, quality_v2_contract
    )
    strictly_better = False
    for metric_name in _dominance_metric_keys(contract):
        candidate_value = _metric_value(record, metric_name)
        reference_value = _metric_value(reference, metric_name)
        spec = _metric_spec(contract, metric_name)
        epsilon, strict_margin = _metric_thresholds(
            metric_name,
            reference_value,
            spec,
        )
        if spec["direction"] == "max":
            if candidate_value < reference_value - epsilon:
                return False
            strictly_better |= candidate_value > reference_value + strict_margin
        else:
            if candidate_value > reference_value + epsilon:
                return False
            strictly_better |= candidate_value < reference_value - strict_margin
    for spec in quality_v2_contract["metrics"]:
        metric_name = str(spec["name"])
        candidate_value = quality_v2_values[metric_name]
        reference_value = reference_quality_v2_values[metric_name]
        tolerance, strict_margin = _quality_v2_metric_thresholds(
            reference_value,
            spec,
        )
        if candidate_value > reference_value + tolerance:
            return False
        strictly_better |= candidate_value < reference_value - strict_margin
    candidate_causal_latency = _t5_causal_latency(record)
    reference_causal_latency = _t5_causal_latency(reference)
    if candidate_causal_latency is not None:
        assert reference_causal_latency is not None
        if candidate_causal_latency > reference_causal_latency + 1.0e-9:
            return False
        strictly_better |= candidate_causal_latency < reference_causal_latency - 1.0e-9
    return strictly_better


def _pareto_frontier(
    records: list[dict[str, Any]],
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if not any(
            other is not record
            and _planner_pareto_dominates(
                other,
                record,
                contract,
                quality_v2_contract,
            )
            for other in records
        )
    ]


def _planner_pareto_tie_key(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
) -> tuple[float, ...]:
    values = []
    quality_v2_values = _validate_quality_v2_attempt(record, quality_v2_contract)
    quality_v2_tie_values = [
        -quality_v2_values[str(spec["name"])] for spec in quality_v2_contract["metrics"]
    ]
    for metric_name in contract["tie_break_order"]:
        if metric_name == "action_l2_sum":
            # The frozen scientific order is task utility, duration,
            # smoothness/path, then aggregate control effort.
            values.extend(quality_v2_tie_values)
            causal_latency = _t5_causal_latency(record)
            if causal_latency is not None:
                values.append(-causal_latency)
        value = _metric_value(record, metric_name)
        values.append(
            value
            if _metric_spec(contract, metric_name)["direction"] == "max"
            else -value
        )
    values.append(-float(int(record["candidate_index"])))
    return tuple(values)


def _select_winner(
    records: list[dict[str, Any]],
    *,
    selection_mode: str = LEGACY_SELECTION_MODE,
    planner_dominance: Mapping[str, Any] | None = None,
    quality_v2_dominance: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Select one eligible winner under the requested planner comparison."""

    eligible = [record for record in records if _eligible(record)]
    if not eligible:
        return None
    if selection_mode == LEGACY_SELECTION_MODE:
        if planner_dominance is not None or quality_v2_dominance is not None:
            raise ValueError("legacy selection must not declare dominance contracts")
        selectable = eligible
        return max(
            selectable,
            key=lambda record: (
                _quality_score(record),
                -int(record["candidate_index"]),
            ),
        )
    if selection_mode == PLANNER_PARETO_SELECTION_MODE:
        if planner_dominance is None or quality_v2_dominance is None:
            raise ValueError(
                "planner-pareto selection requires frozen task and quality-v2 dominance contracts"
            )
        planner = next(
            (record for record in records if int(record["candidate_index"]) == 0),
            None,
        )
        if planner is None:
            raise ValueError("planner-pareto selection requires candidate index zero")
        eligible_rl = [
            record for record in eligible if int(record["candidate_index"]) != 0
        ]
        for record in eligible_rl:
            _validate_attempt_quality(record, planner_dominance)
            _validate_quality_v2_attempt(record, quality_v2_dominance)
        if not _eligible(planner):
            selectable = eligible_rl
        else:
            _validate_attempt_quality(planner, planner_dominance)
            _validate_quality_v2_attempt(planner, quality_v2_dominance)
            selectable = [
                record
                for record in eligible_rl
                if _planner_pareto_dominates(
                    record,
                    planner,
                    planner_dominance,
                    quality_v2_dominance,
                )
            ]
            if not selectable:
                return planner
        if not selectable:
            return None
        frontier = _pareto_frontier(
            selectable,
            planner_dominance,
            quality_v2_dominance,
        )
        return max(
            frontier,
            key=lambda record: _planner_pareto_tie_key(
                record,
                planner_dominance,
                quality_v2_dominance,
            ),
        )
    raise ValueError(f"unsupported selection mode {selection_mode!r}")


def _selection_result(
    records: list[dict[str, Any]],
    winner: Mapping[str, Any] | None,
    *,
    selection_mode: str,
) -> dict[str, Any]:
    """Record the algorithm-neutral source decision separately from quality metrics."""

    planner = next(
        (record for record in records if int(record["candidate_index"]) == 0),
        None,
    )
    if planner is None:
        raise ValueError("selection result requires candidate index zero")
    if selection_mode == LEGACY_SELECTION_MODE:
        source_kind = "legacy" if winner is not None else "rejected"
    elif winner is None:
        source_kind = "rejected"
    elif int(winner["candidate_index"]) == 0:
        source_kind = "planner_fallback"
    else:
        source_kind = "expert_dominant"
    return {
        "source_kind": source_kind,
        "planner_eligible": _eligible(planner),
        "winner_candidate_id": None if winner is None else winner["candidate_id"],
        "winner_candidate_index": None
        if winner is None
        else int(winner["candidate_index"]),
    }


def _render_parity_failure_reason(error: BaseException | str) -> str | None:
    message = str(error)
    if "parity failed" in message:
        return "render_parity_failed"
    if "canonical replay contract" in message:
        return "canonical_replay_contract_failed"
    return None


def _render_parity_skip(
    winner: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    """Bind a failed render replay to the independently selected light attempt."""

    reason = _render_parity_failure_reason(error)
    if reason is None:
        raise ValueError("render-parity skip requires a recognized replay failure")
    return {
        "schema_version": RENDER_PARITY_SKIP_SCHEMA,
        "reason": reason,
        "error_type": type(error).__name__,
        "error": str(error),
        "candidate_id": winner["candidate_id"],
        "candidate_index": int(winner["candidate_index"]),
        "attempt_tape": winner["attempt_tape"],
        "attempt_tape_sha256": winner["attempt_tape_sha256"],
        "action_sha256": winner["action_sha256"],
    }


def _budget_sequence(initial_k: int, max_k: int) -> tuple[int, ...]:
    if initial_k < 1 or max_k < initial_k:
        raise ValueError("candidate budgets require 1 <= initial_k <= max_k")
    values = []
    budget = initial_k
    while budget < max_k:
        values.append(budget)
        budget = min(max_k, budget * 2)
    values.append(max_k)
    return tuple(values)


def _candidate_budgets(
    search_mode: str,
    *,
    initial_k: int,
    max_k: int,
    candidate_pool_size: int,
) -> tuple[int, ...]:
    """Resolve the actual per-reset candidate budgets for a frozen pool."""

    if candidate_pool_size < 1:
        raise ValueError("candidate pool must not be empty")
    if search_mode == FULL_POOL_SEARCH_MODE:
        return (candidate_pool_size,)
    if search_mode != FIRST_ELIGIBLE_SEARCH_MODE:
        raise ValueError(f"unsupported candidate search mode {search_mode!r}")
    if candidate_pool_size < max_k:
        raise ValueError(
            f"candidate manifest must contain at least max_k={max_k} candidates"
        )
    return _budget_sequence(initial_k, max_k)


def _validate_candidate_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    rlinf_commit: str | None,
    benchmark_commit: str | None,
    max_k: int,
) -> tuple[str, tuple[CandidateSpec, ...]]:
    """Validate and resolve the frozen candidate pool without loading policies."""

    schema_version = payload.get("schema_version")
    if schema_version not in CANDIDATE_SCHEMAS:
        raise ValueError("unsupported optimal-trajectory candidate schema")
    if schema_version == CANDIDATE_SCHEMA:
        if rlinf_commit is None or benchmark_commit is None:
            raise ValueError(
                "candidate schema v0.1 requires policy RLinf/benchmark commits"
            )
        if payload.get("rlinf_commit") != rlinf_commit:
            raise ValueError("candidate manifest RLinf commit mismatch")
        if payload.get("benchmark_commit") != benchmark_commit:
            raise ValueError("candidate manifest benchmark commit mismatch")
    else:
        expected_v2_keys = {
            "schema_version",
            "task",
            "evaluator_identity",
            "policy_rlinf_commits",
            "policy_benchmark_commits",
            "candidates",
            "planner_dominance",
        }
        if set(payload) != expected_v2_keys:
            raise ValueError("candidate schema v0.2 top-level inventory mismatch")
        if rlinf_commit is not None or benchmark_commit is not None:
            raise ValueError(
                "candidate schema v0.2 forbids legacy singular policy commit arguments"
            )
    task = payload.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError("candidate manifest task identity is missing")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or len(rows) < max_k:
        raise ValueError(
            f"candidate manifest must contain at least max_k={max_k} candidates"
        )
    allowed = {
        "candidate_id",
        "kind",
        "policy_path",
        "policy_sha256",
        "stochastic",
        "exploration_seed_offset",
        "residual_scale",
        "provenance",
    }
    specs = []
    policy_rlinf_commits: set[str] = set()
    policy_benchmark_commits: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("candidate row is not a supported mapping")
        candidate_id = row.get("candidate_id")
        kind = row.get("kind")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if kind not in {"planner", "policy"}:
            raise ValueError(f"candidate {candidate_id!r} has unsupported kind")
        raw_stochastic = row.get("stochastic", False)
        if not isinstance(raw_stochastic, bool):
            raise ValueError("candidate stochastic must be boolean")
        stochastic = raw_stochastic
        raw_seed_offset = row.get("exploration_seed_offset", 0)
        if isinstance(raw_seed_offset, bool) or not isinstance(raw_seed_offset, int):
            raise ValueError("candidate exploration_seed_offset must be an integer")
        seed_offset = raw_seed_offset
        if not 0 <= seed_offset < 2**31:
            raise ValueError("candidate exploration_seed_offset must be in [0, 2**31)")
        residual_scale = row.get("residual_scale")
        if residual_scale is not None:
            residual_scale = _finite_number(
                residual_scale,
                f"candidate {candidate_id!r} residual_scale",
            )
            if not 0.0 < residual_scale <= 1.0:
                raise ValueError("candidate residual_scale must be in (0, 1]")
        provenance = row.get("provenance")
        if provenance is not None:
            if not isinstance(provenance, Mapping) or not provenance:
                raise ValueError("candidate provenance must be a non-empty mapping")
            try:
                provenance = json.loads(
                    json.dumps(provenance, allow_nan=False, sort_keys=True)
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "candidate provenance must be canonical-JSON-safe"
                ) from error
        if schema_version == CANDIDATE_SCHEMA_V2 and provenance is None:
            raise ValueError(
                "candidate schema v0.2 requires provenance for every candidate"
            )
        policy_path = None
        policy_sha256 = None
        if kind == "planner":
            if stochastic or seed_offset or residual_scale is not None:
                raise ValueError(
                    "planner candidate cannot declare policy exploration fields"
                )
            if (
                row.get("policy_path") is not None
                or row.get("policy_sha256") is not None
            ):
                raise ValueError("planner candidate cannot declare a policy file")
        else:
            raw_path = row.get("policy_path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("policy candidate is missing policy_path")
            policy_path = Path(raw_path)
            if not policy_path.is_absolute():
                policy_path = (manifest_path.parent / policy_path).resolve()
            policy_sha256 = _expected_sha256(
                row.get("policy_sha256"),
                f"candidate {candidate_id} policy_sha256",
            )
            if provenance is not None:
                source = provenance.get("source")
                checkpoint = provenance.get("checkpoint")
                benchmark = provenance.get("benchmark")
                if not isinstance(source, Mapping):
                    raise ValueError(
                        f"candidate {candidate_id!r} provenance has no source"
                    )
                source_rlinf_commit = _full_commit(
                    f"candidate {candidate_id!r} provenance source RLinf commit",
                    source.get("rlinf_commit"),
                )
                if not isinstance(checkpoint, Mapping):
                    raise ValueError(
                        f"candidate {candidate_id!r} provenance has no checkpoint"
                    )
                if checkpoint.get("sha256") != policy_sha256:
                    raise ValueError(
                        f"candidate {candidate_id!r} provenance checkpoint SHA-256 mismatch"
                    )
                provenance_path = checkpoint.get("path")
                if not isinstance(provenance_path, str) or not provenance_path:
                    raise ValueError(
                        f"candidate {candidate_id!r} provenance checkpoint path is missing"
                    )
                resolved_provenance_path = Path(provenance_path)
                if not resolved_provenance_path.is_absolute():
                    resolved_provenance_path = (
                        manifest_path.parent / resolved_provenance_path
                    ).resolve()
                if resolved_provenance_path != policy_path:
                    raise ValueError(
                        f"candidate {candidate_id!r} provenance checkpoint path mismatch"
                    )
                if not isinstance(benchmark, Mapping):
                    raise ValueError(
                        f"candidate {candidate_id!r} provenance has no benchmark"
                    )
                source_benchmark_commit = _full_commit(
                    f"candidate {candidate_id!r} provenance benchmark commit",
                    benchmark.get("commit"),
                )
                if (
                    schema_version == CANDIDATE_SCHEMA
                    and source_benchmark_commit != benchmark_commit
                ):
                    raise ValueError(
                        f"candidate {candidate_id!r} provenance benchmark mismatch"
                    )
                policy_rlinf_commits.add(source_rlinf_commit)
                policy_benchmark_commits.add(source_benchmark_commit)
        specs.append(
            CandidateSpec(
                candidate_id=candidate_id,
                kind=kind,
                policy_path=policy_path,
                policy_sha256=policy_sha256,
                stochastic=stochastic,
                exploration_seed_offset=seed_offset,
                residual_scale=residual_scale,
                provenance=provenance,
            )
        )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    if sum(spec.kind == "planner" for spec in specs) != 1:
        raise ValueError("candidate manifest must contain exactly one planner")
    if specs[0].kind != "planner":
        raise ValueError("the frozen candidate pool must put its planner at index zero")
    if schema_version == CANDIDATE_SCHEMA_V2:
        if payload.get("policy_rlinf_commits") != sorted(policy_rlinf_commits):
            raise ValueError("candidate v0.2 policy RLinf commit inventory mismatch")
        if payload.get("policy_benchmark_commits") != sorted(policy_benchmark_commits):
            raise ValueError(
                "candidate v0.2 policy benchmark commit inventory mismatch"
            )
    return task, tuple(specs)


def _policy_source_commits(
    spec: CandidateSpec,
    *,
    fallback_rlinf_commit: str | None,
    fallback_benchmark_commit: str | None,
) -> tuple[str, str]:
    """Return per-policy authority, falling back only for schema v0.1."""

    if spec.kind != "policy":
        raise ValueError("policy source commit requested for a non-policy candidate")
    if spec.provenance is None:
        if fallback_rlinf_commit is None or fallback_benchmark_commit is None:
            raise ValueError(
                f"candidate {spec.candidate_id!r} has no per-policy source authority"
            )
        return fallback_rlinf_commit, fallback_benchmark_commit
    source = spec.provenance.get("source")
    benchmark = spec.provenance.get("benchmark")
    if not isinstance(source, Mapping) or not isinstance(benchmark, Mapping):
        raise ValueError(
            f"candidate {spec.candidate_id!r} provenance source is incomplete"
        )
    return (
        _full_commit(
            f"candidate {spec.candidate_id!r} provenance source RLinf commit",
            source.get("rlinf_commit"),
        ),
        _full_commit(
            f"candidate {spec.candidate_id!r} provenance benchmark commit",
            benchmark.get("commit"),
        ),
    )


def _task_compatibility_inventory(
    specs: Sequence[CandidateSpec],
    *,
    task: str,
    policy_benchmark_commit: str,
) -> list[dict[str, Any]]:
    """Project unique task/checkpoint identities for compatibility coverage."""

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in specs:
        if spec.kind != "policy" or spec.provenance is None:
            continue
        source = spec.provenance.get("source")
        benchmark = spec.provenance.get("benchmark")
        state_schema = spec.provenance.get("state_schema")
        if (
            not isinstance(source, Mapping)
            or not isinstance(benchmark, Mapping)
            or not isinstance(state_schema, Mapping)
            or benchmark.get("commit") != policy_benchmark_commit
        ):
            continue
        state_dim = state_schema.get("state_dim")
        mask_dim = state_schema.get("mask_dim")
        if (
            isinstance(state_dim, bool)
            or not isinstance(state_dim, int)
            or state_dim < 1
            or isinstance(mask_dim, bool)
            or not isinstance(mask_dim, int)
            or mask_dim < 0
        ):
            raise ValueError("candidate compatibility state dimensions are invalid")
        if spec.policy_sha256 is None:
            raise ValueError("candidate compatibility policy SHA-256 is missing")
        row = {
            "task": task,
            "policy_sha256": spec.policy_sha256,
            "policy_rlinf_commit": _full_commit(
                f"candidate {spec.candidate_id!r} compatibility RLinf commit",
                source.get("rlinf_commit"),
            ),
            "policy_benchmark_commit": policy_benchmark_commit,
            "policy_state_schema_sha256": _expected_sha256(
                state_schema.get("sha256"),
                f"candidate {spec.candidate_id!r} compatibility state schema",
            ),
            "policy_state_dim": state_dim,
            "policy_mask_dim": mask_dim,
        }
        key = (task, spec.policy_sha256)
        previous = rows.get(key)
        if previous is not None and previous != row:
            raise ValueError(
                "candidate rollout expansions carry mixed compatibility provenance"
            )
        rows[key] = row
    return [rows[key] for key in sorted(rows)]


def _resolve_candidate_release_file(manifest_path: Path, relative: str) -> Path:
    """Resolve one portable input while forbidding release-root escapes."""

    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError("candidate release evidence path must be relative")
    release_root = manifest_path.resolve().parent.parent
    resolved = (manifest_path.parent / relative_path).resolve()
    if not resolved.is_relative_to(release_root) or not resolved.is_file():
        raise ValueError(
            "candidate release evidence escapes or is missing from the release"
        )
    return resolved


def _validate_evaluator_identity(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    specs: tuple[CandidateSpec, ...],
    evaluator_rlinf_commit: str,
    evaluator_benchmark_commit: str | None,
    planner_dominance: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, tuple[CompatibilityEvidence, ...]]:
    """Validate v0.2 evaluator identity and portable compatibility evidence."""

    if payload.get("schema_version") == CANDIDATE_SCHEMA:
        if payload.get("evaluator_identity") is not None:
            raise ValueError("candidate schema v0.1 cannot declare evaluator_identity")
        return None, ()
    if payload.get("schema_version") != CANDIDATE_SCHEMA_V2:
        raise ValueError("unsupported candidate schema for evaluator identity")
    if evaluator_benchmark_commit is None:
        raise ValueError("candidate schema v0.2 requires --evaluator-benchmark-commit")
    raw = payload.get("evaluator_identity")
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "evaluator_rlinf_commit",
        "evaluator_benchmark_commit",
        "backend_id",
        "policy_benchmark_relations",
    }:
        raise ValueError("candidate evaluator identity inventory mismatch")
    if raw.get("schema_version") != EVALUATOR_IDENTITY_SCHEMA:
        raise ValueError("candidate evaluator identity schema mismatch")
    if raw.get("evaluator_rlinf_commit") != evaluator_rlinf_commit:
        raise ValueError("candidate evaluator RLinf commit differs from the CLI")
    if raw.get("evaluator_benchmark_commit") != evaluator_benchmark_commit:
        raise ValueError("candidate evaluator benchmark commit differs from the CLI")
    backend_id = raw.get("backend_id")
    if (
        not isinstance(backend_id, str)
        or not backend_id
        or backend_id.strip() != backend_id
    ):
        raise ValueError("candidate evaluator backend identity is missing")
    if planner_dominance is None or planner_dominance.get("backend_id") != backend_id:
        raise ValueError("planner-dominance backend differs from evaluator identity")

    policy_benchmark_commits = payload.get("policy_benchmark_commits")
    relations = raw.get("policy_benchmark_relations")
    if (
        not isinstance(policy_benchmark_commits, list)
        or not isinstance(relations, list)
        or not relations
    ):
        raise ValueError("candidate policy benchmark relation inventory mismatch")
    normalized_relations: list[dict[str, Any]] = []
    evidence: list[CompatibilityEvidence] = []
    relation_commits: list[str] = []
    for relation in relations:
        if not isinstance(relation, Mapping) or set(relation) != {
            "policy_benchmark_commit",
            "relation",
            "evidence_path",
            "evidence_sha256",
        }:
            raise ValueError("candidate policy benchmark relation row mismatch")
        policy_commit = _full_commit(
            "policy benchmark relation commit",
            relation.get("policy_benchmark_commit"),
        )
        relation_commits.append(policy_commit)
        relation_name = relation.get("relation")
        evidence_path = relation.get("evidence_path")
        evidence_sha256 = relation.get("evidence_sha256")
        if relation_name == "identical":
            if (
                policy_commit != evaluator_benchmark_commit
                or evidence_path is not None
                or evidence_sha256 is not None
            ):
                raise ValueError("identical benchmark relation is inconsistent")
        elif relation_name == "checkpoint-compatible":
            if policy_commit == evaluator_benchmark_commit:
                raise ValueError(
                    "identical benchmark commits cannot claim compatibility"
                )
            if not isinstance(evidence_path, str) or not evidence_path:
                raise ValueError("checkpoint-compatible relation has no evidence path")
            expected_evidence_sha256 = _expected_sha256(
                evidence_sha256,
                "benchmark compatibility evidence SHA-256",
            )
            source_path = _resolve_candidate_release_file(
                manifest_path,
                evidence_path,
            )
            if _sha256(source_path) != expected_evidence_sha256:
                raise ValueError("benchmark compatibility evidence SHA-256 mismatch")
            proof = validate_compatibility_evidence(
                json.loads(source_path.read_text(encoding="utf-8"))
            )
            if (
                proof["policy_benchmark_commit"] != policy_commit
                or proof["evaluator_rlinf_commit"] != evaluator_rlinf_commit
                or proof["evaluator_benchmark_commit"] != evaluator_benchmark_commit
                or proof["backend_id"] != backend_id
            ):
                raise ValueError("benchmark compatibility evidence identity mismatch")
            expected_task_inventory = _task_compatibility_inventory(
                specs,
                task=payload["task"],
                policy_benchmark_commit=policy_commit,
            )
            proof_task_inventory = [
                {
                    "task": probe["task"],
                    "policy_sha256": probe["policy_sha256"],
                    "policy_rlinf_commit": probe["policy_rlinf_commit"],
                    "policy_benchmark_commit": proof["policy_benchmark_commit"],
                    "policy_state_schema_sha256": probe["policy_state_schema_sha256"],
                    "policy_state_dim": probe["policy_state_dim"],
                    "policy_mask_dim": probe["policy_mask_dim"],
                }
                for probe in proof["probes"]
                if probe["task"] == payload["task"]
            ]
            if proof_task_inventory != expected_task_inventory:
                raise ValueError(
                    "benchmark compatibility evidence does not cover the task policy pool"
                )
            evidence.append(
                CompatibilityEvidence(
                    policy_benchmark_commit=policy_commit,
                    source_path=source_path,
                    sha256=expected_evidence_sha256,
                )
            )
        else:
            raise ValueError(f"unsupported policy benchmark relation {relation_name!r}")
        normalized_relations.append(dict(relation))
    if (
        relation_commits != sorted(set(relation_commits))
        or relation_commits != policy_benchmark_commits
    ):
        raise ValueError(
            "candidate policy benchmark relations are not canonical or complete"
        )

    planner = specs[0]
    if planner.provenance is None:
        raise ValueError("candidate schema v0.2 planner provenance is missing")
    planner_source = planner.provenance.get("source")
    planner_runtime = planner.provenance.get("runtime")
    planner_benchmark = planner.provenance.get("benchmark")
    if (
        not isinstance(planner_source, Mapping)
        or planner_source.get("rlinf_commit") != evaluator_rlinf_commit
        or not isinstance(planner_runtime, Mapping)
        or planner_runtime.get("evaluator_rlinf_commit") != evaluator_rlinf_commit
        or not isinstance(planner_benchmark, Mapping)
        or planner_benchmark.get("commit") != evaluator_benchmark_commit
    ):
        raise ValueError("planner provenance is not bound to the evaluator identity")
    normalized = {
        "schema_version": EVALUATOR_IDENTITY_SCHEMA,
        "evaluator_rlinf_commit": evaluator_rlinf_commit,
        "evaluator_benchmark_commit": evaluator_benchmark_commit,
        "backend_id": backend_id,
        "policy_benchmark_relations": normalized_relations,
    }
    return normalized, tuple(evidence)


def _validate_calibration_evidence(
    *,
    manifest_path: Path,
    planner_dominance: Mapping[str, Any] | None,
    evaluator_identity: Mapping[str, Any] | None,
) -> CalibrationEvidence | None:
    """Verify raw fresh-environment planner replays and recompute every drift."""

    if planner_dominance is None:
        return None
    if evaluator_identity is None:
        raise ValueError("planner calibration requires a frozen evaluator identity")
    calibration = planner_dominance["calibration"]
    source_path = _resolve_candidate_release_file(
        manifest_path,
        calibration["evidence_path"],
    )
    if _sha256(source_path) != calibration["evidence_sha256"]:
        raise ValueError("planner calibration evidence file SHA-256 mismatch")
    evidence = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "schema_version",
        "task",
        "backend_id",
        "evaluator_identity_sha256",
        "split",
        "test_exposure",
        "reset_manifest_sha256",
        "replay_count",
        "replays",
        "payload_sha256",
    }:
        raise ValueError("planner calibration evidence inventory mismatch")
    if evidence.get("schema_version") != CALIBRATION_EVIDENCE_SCHEMA:
        raise ValueError("planner calibration evidence schema mismatch")
    if _payload_sha256(evidence) != evidence.get("payload_sha256"):
        raise ValueError("planner calibration evidence payload SHA-256 mismatch")
    if (
        evidence.get("task") != planner_dominance["task"]
        or evidence.get("backend_id") != planner_dominance["backend_id"]
        or evidence.get("evaluator_identity_sha256")
        != _payload_sha256(
            {
                "evaluator_rlinf_commit": evaluator_identity["evaluator_rlinf_commit"],
                "evaluator_benchmark_commit": evaluator_identity[
                    "evaluator_benchmark_commit"
                ],
                "backend_id": evaluator_identity["backend_id"],
            }
        )
    ):
        raise ValueError("planner calibration evaluator identity mismatch")
    if evidence.get("split") not in {"train", "validation"} or evidence.get(
        "test_exposure"
    ) != {"test_id": False, "test_ood": False}:
        raise ValueError("planner calibration evidence used a formal test split")
    if evidence.get("reset_manifest_sha256") != calibration["reset_manifest_sha256"]:
        raise ValueError("planner calibration reset manifest mismatch")
    replay_count = evidence.get("replay_count")
    replays = evidence.get("replays")
    if (
        replay_count != calibration["replay_count"]
        or isinstance(replay_count, bool)
        or not isinstance(replay_count, int)
        or replay_count < 3
        or not isinstance(replays, list)
        or len(replays) != replay_count
    ):
        raise ValueError("planner calibration replay count mismatch")

    replay_keys = {
        "replay_index",
        "environment_instance_id",
        "episode_id",
        "reset_request_sha256",
        "action_sha256",
        "success",
        "safety_failure",
        "finite_and_bounded",
        "termination_reason",
        "trajectory_completion",
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
        "task_quality",
    }
    environment_ids: set[str] = set()
    frozen_identity: tuple[Any, ...] | None = None
    metric_rows: list[dict[str, Any]] = []
    for index, raw_replay in enumerate(replays):
        if not isinstance(raw_replay, Mapping) or set(raw_replay) != replay_keys:
            raise ValueError("planner calibration replay row inventory mismatch")
        environment_id = raw_replay.get("environment_instance_id")
        if (
            raw_replay.get("replay_index") != index
            or not isinstance(environment_id, str)
            or not environment_id
            or environment_id.strip() != environment_id
            or environment_id in environment_ids
        ):
            raise ValueError(
                "planner calibration did not use unique fresh environments"
            )
        environment_ids.add(environment_id)
        episode_id = raw_replay.get("episode_id")
        if episode_id != calibration["reset_episode_id"]:
            raise ValueError("planner calibration reset episode identity drifted")
        reset_request_sha256 = _expected_sha256(
            raw_replay.get("reset_request_sha256"),
            "planner calibration reset request SHA-256",
        )
        action_sha256 = _expected_sha256(
            raw_replay.get("action_sha256"),
            "planner calibration action SHA-256",
        )
        if (
            raw_replay.get("success") is not True
            or raw_replay.get("safety_failure") is not False
            or raw_replay.get("finite_and_bounded") is not True
        ):
            raise ValueError(
                "planner calibration replay is not successful, safe, and finite"
            )
        termination_reason = raw_replay.get("termination_reason")
        if not isinstance(termination_reason, str) or not termination_reason:
            raise ValueError("planner calibration termination reason is missing")
        completion = _finite_number(
            raw_replay["trajectory_completion"],
            "planner calibration trajectory completion",
        )
        completion_time = _finite_number(
            raw_replay["completion_time_s"],
            "planner calibration completion time",
        )
        control_steps = raw_replay.get("control_steps")
        action_l2_sum = _finite_number(
            raw_replay["action_l2_sum"],
            "planner calibration action L2 sum",
        )
        if (
            not 0.0 <= completion <= 1.0
            or completion_time <= 0.0
            or isinstance(control_steps, bool)
            or not isinstance(control_steps, int)
            or control_steps < 1
            or action_l2_sum < 0.0
        ):
            raise ValueError("planner calibration replay metrics are invalid")
        replay = dict(raw_replay)
        _validate_attempt_quality(replay, planner_dominance)
        identity = (
            episode_id,
            reset_request_sha256,
            action_sha256,
            termination_reason,
            control_steps,
        )
        if frozen_identity is None:
            frozen_identity = identity
        elif identity != frozen_identity:
            raise ValueError(
                "planner calibration reset/action/outcome identity drifted"
            )
        metric_rows.append(replay)

    for metric_name in _dominance_metric_keys(planner_dominance):
        values = [_metric_value(row, metric_name) for row in metric_rows]
        observed_drift = max(values) - min(values)
        frozen_drift = float(
            _metric_spec(planner_dominance, metric_name)["max_observed_replay_drift"]
        )
        if not np.isclose(observed_drift, frozen_drift, rtol=0.0, atol=1.0e-15):
            raise ValueError(
                f"planner calibration drift does not recompute for {metric_name!r}"
            )
    return CalibrationEvidence(
        source_path=source_path,
        sha256=calibration["evidence_sha256"],
    )


def _validate_candidate_release_chain(
    *,
    candidate_manifest: Path,
    candidate_manifest_sha256: str,
    candidate_payload: Mapping[str, Any],
    release_manifest: Path | None,
    expected_release_manifest_sha256: str | None,
) -> tuple[str | None, tuple[ProvenanceFile, ...]]:
    """Bind one v0.2 task manifest to a production-validated exact-14 release."""

    if candidate_payload.get("schema_version") == CANDIDATE_SCHEMA:
        if release_manifest is not None or expected_release_manifest_sha256 is not None:
            raise ValueError(
                "candidate schema v0.1 cannot declare a v0.2 release chain"
            )
        return None, ()
    if candidate_payload.get("schema_version") != CANDIDATE_SCHEMA_V2:
        raise ValueError("unsupported candidate schema for release-chain validation")
    if release_manifest is None or expected_release_manifest_sha256 is None:
        raise ValueError(
            "candidate schema v0.2 requires candidate release manifest and pinned SHA-256"
        )
    expected_release_sha256 = _expected_sha256(
        expected_release_manifest_sha256,
        "candidate release manifest SHA-256",
    )
    candidate_manifest = candidate_manifest.resolve()
    release_root = candidate_manifest.parent.parent
    task_value = candidate_payload.get("task")
    if (
        not isinstance(task_value, str)
        or not task_value
        or task_value.strip() != task_value
    ):
        raise ValueError("candidate task identity is invalid")
    expected_candidate_path = (
        release_root / task_value / "candidate_manifest.json"
    ).resolve()
    expected_release_path = (release_root / "release_manifest.json").resolve()
    if candidate_manifest != expected_candidate_path:
        raise ValueError(
            "candidate manifest is orphaned from the canonical task release path"
        )
    if release_manifest.resolve() != expected_release_path:
        raise ValueError("candidate release manifest path is not canonical")
    if _sha256(expected_release_path) != expected_release_sha256:
        raise ValueError("candidate release manifest SHA-256 mismatch")

    try:
        from examples.embodiment.build_dynamic_benchmark_rld2_manifests import (
            validate_release,
        )
    except ModuleNotFoundError:
        # Direct ``python examples/embodiment/export_...py`` execution puts the
        # script directory, not the repository root, on sys.path.
        from build_dynamic_benchmark_rld2_manifests import validate_release

    validate_release(release_root, production=True)
    release = json.loads(expected_release_path.read_text(encoding="utf-8"))
    if release.get("schema_version") != CANDIDATE_RELEASE_SCHEMA:
        raise ValueError("candidate release schema mismatch")
    task_hashes = release.get("task_manifest_sha256")
    task = task_value
    task_policy_rlinf_commits = candidate_payload.get("policy_rlinf_commits")
    task_policy_benchmark_commits = candidate_payload.get("policy_benchmark_commits")
    release_policy_rlinf_commits = release.get("policy_rlinf_commits")
    release_policy_benchmark_commits = release.get("policy_benchmark_commits")
    if (
        not isinstance(task_hashes, Mapping)
        or task_hashes.get(task) != candidate_manifest_sha256
        or release.get("candidate_schema_version") != CANDIDATE_SCHEMA_V2
        or not isinstance(task_policy_rlinf_commits, list)
        or not all(isinstance(item, str) for item in task_policy_rlinf_commits)
        or not isinstance(task_policy_benchmark_commits, list)
        or not all(isinstance(item, str) for item in task_policy_benchmark_commits)
        or not isinstance(release_policy_rlinf_commits, list)
        or not all(isinstance(item, str) for item in release_policy_rlinf_commits)
        or not isinstance(release_policy_benchmark_commits, list)
        or not all(isinstance(item, str) for item in release_policy_benchmark_commits)
        or not set(task_policy_rlinf_commits).issubset(release_policy_rlinf_commits)
        or not set(task_policy_benchmark_commits).issubset(
            release_policy_benchmark_commits
        )
    ):
        raise ValueError(
            "candidate task manifest identity differs from its exact-14 release"
        )
    sha256sums_path = release_root / "SHA256SUMS"
    if not sha256sums_path.is_file():
        raise ValueError("candidate release SHA256SUMS is missing")
    provenance = (
        ProvenanceFile(
            source_path=expected_release_path,
            relative_path="provenance/candidate_release/release_manifest.json",
            sha256=expected_release_sha256,
        ),
        ProvenanceFile(
            source_path=sha256sums_path,
            relative_path="provenance/candidate_release/SHA256SUMS",
            sha256=_sha256(sha256sums_path),
        ),
    )
    return expected_release_sha256, provenance


def _load_candidates(
    specs: tuple[CandidateSpec, ...],
    *,
    task: str,
    rlinf_commit: str | None,
    benchmark_commit: str | None,
    device: torch.device,
) -> tuple[LoadedCandidate, ...]:
    loaded = []
    for index, spec in enumerate(specs):
        candidate = LoadedCandidate(spec=spec, index=index)
        if spec.kind == "policy":
            assert spec.policy_path is not None and spec.policy_sha256 is not None
            if not spec.policy_path.is_file():
                raise FileNotFoundError(spec.policy_path)
            if _sha256(spec.policy_path) != spec.policy_sha256:
                raise ValueError(
                    f"candidate {spec.candidate_id!r} policy SHA-256 mismatch"
                )
            payload = torch.load(
                spec.policy_path, map_location="cpu", weights_only=False
            )
            policy_rlinf_commit, policy_benchmark_commit = _policy_source_commits(
                spec,
                fallback_rlinf_commit=rlinf_commit,
                fallback_benchmark_commit=benchmark_commit,
            )
            config, state_schema = _validate_policy_payload(
                payload,
                rlinf_commit=policy_rlinf_commit,
                benchmark_commit=policy_benchmark_commit,
            )
            if config["task"] != task:
                raise ValueError(f"candidate {spec.candidate_id!r} task mismatch")
            if (
                spec.residual_scale is not None
                and config["algorithm"] != "residual_rlpd"
            ):
                raise ValueError(
                    "residual_scale override requires a residual-RLPD policy"
                )
            state_dim = int(state_schema["state_dim"])
            model = _load_inference_policy(config, state_dim, payload["model"], device)
            normalizer = RunningNormalizer(state_dim, int(state_schema["mask_dim"]))
            normalizer.load_state_dict(payload["normalizer"])
            candidate.config = config
            candidate.state_schema = state_schema
            candidate.model = model
            candidate.normalizer = normalizer
        loaded.append(candidate)
    return tuple(loaded)


def _candidate_seed(episode_id: str, candidate: LoadedCandidate) -> int:
    material = (
        f"{episode_id}\0{candidate.spec.candidate_id}\0"
        f"{candidate.spec.exploration_seed_offset}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from se3_wam.benchmark.contracts import canonical_json

    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(dict(payload)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _file_boundary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def _progress_payload(
    *,
    export_state_sha256: str,
    started_unix_s: float,
    next_reset_index: int,
    accepted_count: int,
    candidate_attempt_count: int,
    budget_histogram: Mapping[str, int],
    attempts_path: Path,
    reset_results_path: Path,
    winners_path: Path,
    resume_count: int,
    recovery_events: list[str],
) -> dict[str, Any]:
    payload = {
        "schema_version": PROGRESS_SCHEMA,
        "export_state_sha256": export_state_sha256,
        "started_unix_s": started_unix_s,
        "next_reset_index": next_reset_index,
        "accepted_count": accepted_count,
        "candidate_attempt_count": candidate_attempt_count,
        "budget_histogram": dict(budget_histogram),
        "resume_count": resume_count,
        "recovery_events": list(recovery_events),
        "file_boundaries": {
            "attempts.jsonl": _file_boundary(attempts_path),
            "reset_results.jsonl": _file_boundary(reset_results_path),
            "winner_manifest.jsonl": _file_boundary(winners_path),
        },
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def _recover_partial_output(
    *,
    output: Path,
    progress: Mapping[str, Any],
    reset_rows: list[Any],
    task: str,
    split: str,
) -> str | None:
    """Preserve and remove only data after the last committed reset boundary."""

    boundaries = progress.get("file_boundaries")
    if not isinstance(boundaries, dict):
        raise ValueError("resume progress is missing file boundaries")
    recovery = output.parent / f"{output.name}.recovery-{time.time_ns()}"
    recovery_created = False

    def ensure_recovery() -> None:
        nonlocal recovery_created
        if not recovery_created:
            recovery.mkdir(parents=False)
            recovery_created = True

    for name in ("attempts.jsonl", "reset_results.jsonl", "winner_manifest.jsonl"):
        path = output / name
        boundary = boundaries.get(name)
        if not isinstance(boundary, dict):
            raise ValueError(f"resume progress has no boundary for {name}")
        size = int(boundary.get("size", -1))
        expected = str(boundary.get("sha256", ""))
        data = path.read_bytes()
        if size < 0 or len(data) < size:
            raise ValueError(f"{name} is shorter than its committed boundary")
        prefix = data[:size]
        actual = hashlib.sha256(prefix).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} committed prefix checksum mismatch")
        if len(data) > size:
            ensure_recovery()
            shutil.copy2(path, recovery / name)
            temporary = path.with_suffix(path.suffix + ".resume.tmp")
            with temporary.open("wb") as stream:
                stream.write(prefix)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)

    next_reset_index = int(progress.get("next_reset_index", -1))
    if not 0 <= next_reset_index <= len(reset_rows):
        raise ValueError("resume progress reset index is outside the manifest")
    dirty_paths = [output / ".staging"]
    if next_reset_index < len(reset_rows):
        episode_id = reset_rows[next_reset_index].request.episode_id
        dirty_paths.extend(
            (
                output / "lightweight" / episode_id,
                output / "episodes" / task / split / episode_id,
            )
        )
    for path in dirty_paths:
        if not path.exists():
            continue
        if path.name == ".staging" and not any(path.iterdir()):
            path.rmdir()
            continue
        ensure_recovery()
        destination = recovery / path.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    return recovery.name if recovery_created else None


def _write_attempt_tape(
    output: Path,
    *,
    episode_id: str,
    candidate_index: int,
    arrays: Mapping[str, np.ndarray],
) -> tuple[str, str]:
    relative = Path("lightweight") / episode_id / f"candidate-{candidate_index:02d}.npz"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return relative.as_posix(), _sha256(path)


def _make_teacher(task: str, request: Any) -> Any:
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    teacher, _ = make_privileged_teacher(task, request=request)
    if hasattr(teacher, "reset"):
        teacher.reset()
    return teacher


class _ArmedResetReplayEnv:
    """Expose the canonical raw-env replay API while rearming hidden T5 events."""

    def __init__(self, vector_env: Any, raw_env: Any) -> None:
        self._vector_env = vector_env
        self._raw_env = raw_env

    def reset(self, request: Any) -> Any:
        observation = self._raw_env.reset(request)
        self._vector_env._arm_hidden_t5_event(self._raw_env, request)
        return observation

    def step(self, action: Any) -> Any:
        return self._raw_env.step(action)

    def save_state(self) -> bytes:
        return self._raw_env.save_state()


def _restore_candidate_start(env: Any, state: Mapping[str, Any]) -> None:
    """Restore wrapper bookkeeping, then rebuild the canonical request reset."""

    env.load_checkpoint_state(state)
    request = env._requests[0]
    if request is None:
        raise RuntimeError("candidate restore lost its reset request")
    raw_env = env.envs[0]
    observation = raw_env.reset(request)
    env._arm_hidden_t5_event(raw_env, request)
    env._raw_observations[0] = observation
    encoded = np.asarray(env._encode(observation, request), dtype=np.float32)
    env._last_obs = {"states": torch.as_tensor(encoded[None, :], dtype=torch.float32)}


def _task_quality_from_infos(infos: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read backend-defined task quality without inventing a fallback mapping."""

    raw = infos.get("task_quality")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError("task quality vector must contain exactly one environment")
        raw = raw[0]
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("task quality summary must be a non-empty mapping")
    try:
        # Keep the upstream component insertion order: it is part of the
        # canonical schema/summary contract and is checked independently.
        return json.loads(json.dumps(raw, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("task quality summary must be canonical-JSON-safe") from error


def _t5_replan_causal_evidence(
    raw_env: Any,
    *,
    control_steps: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Serialize and validate the canonical T5 issued/applied queue ledgers."""

    from se3_wam.benchmark.registry import get_task_spec
    from se3_wam.benchmark.t5_timing_contract import validate_t5_replan_timing

    history = raw_env.canonical_action_history
    if set(history) != {"issued_actions", "applied_actions"}:
        raise RuntimeError("T5 canonical action history inventory drifted")
    issued = tuple(history["issued_actions"])
    applied = tuple(history["applied_actions"])
    timing = raw_env.timing_summary()
    control_hz = float(get_task_spec("t5_replan").clock.control_hz)
    action_delay_steps = int(timing["action_delay_steps"])
    impact_end_time_s = timing.get("impact_end_time_s")
    first_contact_time_s = timing.get("first_contact_time_s")
    report = validate_t5_replan_timing(
        issued_actions=issued,
        applied_actions=applied,
        impact_end_time_s=impact_end_time_s,
        first_contact_time_s=first_contact_time_s,
        expected_issued_action_count=control_steps,
        expected_action_delay_steps=action_delay_steps,
        control_hz=control_hz,
    )
    applied_by_issue_step = {int(record["policy_step"]): record for record in applied}
    correction_latency_s: float | None = None
    if report.passed:
        if not report.qualifying_correction_steps or impact_end_time_s is None:
            raise RuntimeError(
                "passing T5 causal report lacks its qualifying correction"
            )
        first_step = report.qualifying_correction_steps[0]
        first_applied = applied_by_issue_step.get(first_step)
        if first_applied is None:
            raise RuntimeError(
                "T5 qualifying correction is absent from applied history"
            )
        correction_latency_s = float(first_applied["actual_apply_time_s"]) - float(
            impact_end_time_s
        )

    issued_count = len(issued)
    applied_count = len(applied)
    arrays = {
        "t5_action_history_schema": np.asarray(
            T5_ACTION_HISTORY_SCHEMA,
            dtype=f"<U{len(T5_ACTION_HISTORY_SCHEMA)}",
        ),
        "action_value_semantic_labels": np.asarray(
            T5_ACTION_VALUE_SEMANTIC_LABELS,
            dtype="<U32",
        ),
        "issued_action_values": np.asarray(
            [record["values"] for record in issued],
            dtype=np.float64,
        ).reshape(issued_count, 7),
        "issued_policy_step": np.asarray(
            [record["policy_step"] for record in issued],
            dtype=np.int64,
        ),
        "issued_time_s": np.asarray(
            [record["issue_time_s"] for record in issued],
            dtype=np.float64,
        ),
        "scheduled_apply_policy_step": np.asarray(
            [record["apply_policy_step"] for record in issued],
            dtype=np.int64,
        ),
        "scheduled_apply_time_s": np.asarray(
            [record["apply_time_s"] for record in issued],
            dtype=np.float64,
        ),
        "applied_action_values": np.asarray(
            [record["values"] for record in applied],
            dtype=np.float64,
        ).reshape(applied_count, 7),
        "applied_issue_policy_step": np.asarray(
            [record["policy_step"] for record in applied],
            dtype=np.int64,
        ),
        "actual_apply_policy_step": np.asarray(
            [record["actual_apply_policy_step"] for record in applied],
            dtype=np.int64,
        ),
        "actual_apply_time_s": np.asarray(
            [record["actual_apply_time_s"] for record in applied],
            dtype=np.float64,
        ),
        "t5_timing_value_semantic_labels": np.asarray(
            T5_TIMING_VALUE_SEMANTIC_LABELS,
            dtype="<U32",
        ),
        "t5_timing_values": np.asarray(
            [
                np.nan if impact_end_time_s is None else impact_end_time_s,
                np.nan if first_contact_time_s is None else first_contact_time_s,
                control_hz,
            ],
            dtype=np.float64,
        ),
        "t5_timing_count_semantic_labels": np.asarray(
            T5_TIMING_COUNT_SEMANTIC_LABELS,
            dtype="<U40",
        ),
        "t5_timing_counts": np.asarray(
            [control_steps, action_delay_steps],
            dtype=np.int64,
        ),
    }
    record_fields = {
        "issued_equals_applied": False,
        "t5_replan_causal_timing_passed": report.passed,
        "impact_end_to_first_qualifying_applied_correction_s": correction_latency_s,
    }
    return record_fields, arrays


def _trajectory_quality_v2_from_rollout(
    observations: Sequence[Any],
    action_array: np.ndarray,
    *,
    task_id: str,
    task_config: Mapping[str, object] | None,
    sample_period_s: float = 0.05,
) -> dict[str, Any]:
    """Compute replay-bound smoothness and EEF-orientation diagnostics.

    ``task_quality`` remains the task-specific success/utility contract.  This
    separate bundle is deliberately measurement-only: task/phase thresholds
    are calibrated from frozen planner/RL replays before they are allowed to
    reject candidates.  Missing privileged pose sources fail closed because a
    trajectory without orientation evidence cannot be used for RLD2-QA.
    """

    from se3_wam.benchmark.trajectory_quality import (
        trajectory_quality_v2_from_observations,
    )

    return trajectory_quality_v2_from_observations(
        observations,
        action_array,
        task_id=task_id,
        task_config=task_config,
        sample_period_s=sample_period_s,
        continuous_dimensions=max(1, int(action_array.shape[1]) - 1),
    )


def _rollout(
    *,
    env: Any,
    candidate: LoadedCandidate,
    device: torch.device,
    capture_trace: bool,
    trace_metadata: Mapping[str, Any] | None = None,
    replay_actions_array: np.ndarray | None = None,
    quality_v2_thresholds: Mapping[str, object] | None = None,
    quality_v2_thresholds_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], Any | None]:
    from se3_wam.benchmark.api import StepResult
    from se3_wam.benchmark.dataset import EpisodeTrace
    from se3_wam.benchmark.evaluation import replay_actions
    from se3_wam.benchmark.metrics import (
        completion_time_from_events,
        hierarchical_task_completion,
        validate_stage_event_order,
    )

    request = env._requests[0]
    observation = env._raw_observations[0]
    obs = env._last_obs
    if request is None or observation is None or obs is None:
        raise RuntimeError("optimal-trajectory environment is not initialized")
    row = _manifest_row(env, request.episode_id)
    raw_env = env.envs[0]
    task = env._get_task_spec(request.task_id)
    teacher = None
    residual = False
    residual_scale = None
    if candidate.spec.kind == "planner":
        teacher = _make_teacher(request.task_id, request)
    else:
        assert candidate.config is not None
        assert candidate.model is not None and candidate.normalizer is not None
        residual = candidate.config["algorithm"] == "residual_rlpd"
        if residual:
            teacher = _make_teacher(request.task_id, request)
            residual_scale = (
                candidate.spec.residual_scale
                if candidate.spec.residual_scale is not None
                else float(candidate.config.get("residual_scale", 0.25))
            )
    seed = _candidate_seed(request.episode_id, candidate)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    observations = [observation]
    states = [np.asarray(obs["states"][0], dtype=np.float32)]
    actions = []
    policy_actions = []
    rewards = []
    terminated_rows = []
    truncated_rows = []
    outcomes = []
    step_results = []
    result_info: dict[str, Any] | None = None
    terminated_value = False
    truncated_value = False
    while not (terminated_value or truncated_value):
        if replay_actions_array is not None:
            action_index = len(actions)
            if action_index >= replay_actions_array.shape[0]:
                raise RuntimeError(
                    "replayed action sequence is shorter than the rollout"
                )
            env_actions = torch.as_tensor(
                replay_actions_array[action_index], dtype=torch.float32
            ).unsqueeze(0)
            policy_action = env_actions.clone()
        elif candidate.spec.kind == "planner":
            env_actions = _planner_actions(env, [teacher])
            policy_action = env_actions.clone()
        else:
            with torch.inference_mode():
                policy_action, _ = _policy_action(
                    candidate.model,
                    candidate.normalizer,
                    obs["states"],
                    device,
                    stochastic=candidate.spec.stochastic,
                )
            policy_action = policy_action.cpu()
            env_actions = policy_action
            if residual:
                assert teacher is not None and residual_scale is not None
                env_actions = _compose_residual_actions(
                    _planner_actions(env, [teacher]),
                    policy_action,
                    residual_scale,
                )
        values = np.clip(np.asarray(env_actions[0], dtype=np.float64), -1.0, 1.0)
        action = env._ActionCommand(
            mode=request.action_mode,
            values=values,
            policy_step=observation.policy_step,
        )
        next_obs, reward, terminated, truncated, infos = env.step(
            env_actions,
            auto_reset=False,
        )
        next_observation = env._raw_observations[0]
        if next_observation is None:
            raise RuntimeError("optimal-trajectory rollout lost its raw observation")
        terminated_value = bool(terminated[0])
        truncated_value = bool(truncated[0])
        reason = infos["termination_reason"][0]
        active_progress = float(infos["reward_inputs"]["active_stage_progress"][0])
        observations.append(next_observation)
        states.append(np.asarray(next_obs["states"][0], dtype=np.float32))
        actions.append(action)
        policy_actions.append(np.asarray(policy_action[0], dtype=np.float32))
        rewards.append(float(reward[0]))
        terminated_rows.append(terminated_value)
        truncated_rows.append(truncated_value)
        outcomes.append(
            (
                terminated_value,
                truncated_value,
                bool(infos["success"][0]),
                reason,
                active_progress,
            )
        )
        task_quality = _task_quality_from_infos(infos)
        step_result_fields = {
            "observation": next_observation,
            "terminated": terminated_value,
            "truncated": truncated_value,
            "success": bool(infos["success"][0]),
            "termination_reason": reason,
            "active_stage_progress": active_progress,
        }
        if task_quality is not None:
            # Reconstruct the canonical typed object so the terminal EpisodeTrace
            # retains the same quality summary that governed selection.
            from se3_wam.benchmark.task_quality import EpisodeQualitySummary

            step_result_fields["task_quality"] = EpisodeQualitySummary.from_dict(
                task_quality
            )
        step_results.append(StepResult(**step_result_fields))
        result_info = {
            "success": bool(infos["success"][0]),
            "termination_reason": reason,
            "active_stage_progress": active_progress,
        }
        if task_quality is not None:
            result_info["task_quality"] = task_quality
        observation = next_observation
        obs = next_obs
        if len(actions) > int(env.horizon_steps):
            raise RuntimeError(
                "optimal-trajectory rollout exceeded the environment horizon"
            )
    if result_info is None:
        raise RuntimeError("optimal-trajectory candidate produced no action")

    events = tuple(raw_env._ledger.events)
    final_state = raw_env.save_state()
    completed = validate_stage_event_order(task, events)
    completion = hierarchical_task_completion(
        task,
        completed,
        result_info["active_stage_progress"],
    )
    completion_time = (
        completion_time_from_events(
            events,
            start_event=task.task_start_event,
            success_event=task.success_stages[-1],
        )
        if result_info["success"]
        else None
    )
    replay_validation = replay_actions(
        _ArmedResetReplayEnv(env, raw_env),
        request=request,
        expected_observations=tuple(observations),
        actions=tuple(actions),
        expected_outcomes=tuple(outcomes),
        expected_final_state=final_state,
    )
    action_array = np.stack([action.values for action in actions]).astype(np.float64)
    policy_action_array = np.stack(policy_actions).astype(np.float32)
    state_array = np.stack(states).astype(np.float32)
    reward_array = np.asarray(rewards, dtype=np.float32)
    quality_v2 = _trajectory_quality_v2_from_rollout(
        observations,
        action_array,
        task_id=env.task_id,
        task_config=getattr(raw_env, "task_config", None),
    )
    if quality_v2_thresholds is None or quality_v2_thresholds_sha256 is None:
        raise ValueError("quality-v2 threshold contract and identity are required")
    from se3_wam.benchmark.trajectory_quality import evaluate_quality_v2_gate

    quality_v2_gate = evaluate_quality_v2_gate(
        quality_v2,
        quality_v2_thresholds,
        task_id=env.task_id,
    )
    quality_v2_gate["contract_sha256"] = quality_v2_thresholds_sha256
    finite_and_bounded = bool(
        np.all(np.isfinite(state_array))
        and np.all(np.isfinite(action_array))
        and np.all(np.isfinite(policy_action_array))
        and np.all(np.isfinite(reward_array))
        and np.all(np.abs(action_array) <= 1.0)
        and np.all(np.abs(policy_action_array) <= 1.0)
    )
    safety_failures = set(env.reward_schema["safety_failures"])
    if request.task_id == "t5_replan":
        causal_record_fields, causal_arrays = _t5_replan_causal_evidence(
            raw_env,
            control_steps=len(actions),
        )
    else:
        causal_record_fields = {
            "issued_equals_applied": True,
            "t5_replan_causal_timing_passed": None,
            "impact_end_to_first_qualifying_applied_correction_s": None,
        }
        causal_arrays = {}
    record = {
        "schema_version": ATTEMPT_SCHEMA,
        "episode_id": request.episode_id,
        "task_id": request.task_id,
        "seed": request.seed,
        "factors": dict(request.factors),
        "source_group_id": row.source_group_id,
        "pair_id": row.pair_id,
        "pair_member_id": row.pair_member_id,
        "candidate_manifest_index": row.candidate_index,
        "candidate_id": candidate.spec.candidate_id,
        "candidate_index": candidate.index,
        "candidate_kind": candidate.spec.kind,
        "stochastic": candidate.spec.stochastic,
        "exploration_seed": seed,
        "residual_scale": residual_scale,
        "success": result_info["success"],
        "safety_failure": result_info["termination_reason"] in safety_failures,
        "termination_reason": result_info["termination_reason"],
        "trajectory_completion": completion,
        "task_quality": result_info.get("task_quality"),
        "quality_v2": quality_v2,
        "quality_v2_sha256": _payload_sha256(quality_v2),
        "quality_v2_gate": quality_v2_gate,
        "quality_v2_events_by_observation": [
            [str(event.name) for event in observation.events_since_last_observation]
            for observation in observations
        ],
        "completion_time_s": completion_time,
        "return": float(reward_array.sum(dtype=np.float64)),
        "control_steps": len(actions),
        "action_l2_sum": float(np.square(action_array).sum()),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(action_array).tobytes()
        ).hexdigest(),
        "policy_action_sha256": hashlib.sha256(
            np.ascontiguousarray(policy_action_array).tobytes()
        ).hexdigest(),
        "state_sha256": hashlib.sha256(
            np.ascontiguousarray(state_array).tobytes()
        ).hexdigest(),
        "reward_sha256": hashlib.sha256(
            np.ascontiguousarray(reward_array).tobytes()
        ).hexdigest(),
        "finite_and_bounded": finite_and_bounded,
        "replay_validation": replay_validation,
        "replay_validation_sha256": _payload_sha256(replay_validation),
        "events": [event.name for event in events],
        **causal_record_fields,
    }
    arrays = {
        "states": state_array,
        "policy_actions": policy_action_array,
        "actions": action_array,
        "rewards": reward_array,
        "terminated": np.asarray(terminated_rows, dtype=np.bool_),
        "truncated": np.asarray(truncated_rows, dtype=np.bool_),
        "eef_pose_xyzw": np.stack(
            [
                np.asarray(observation.privileged["eef_pose_xyzw"], dtype=np.float64)
                for observation in observations
            ]
        ),
        "fingerpad_closing_axis_world": np.stack(
            [
                np.asarray(
                    observation.privileged["fingerpad_closing_axis_world"],
                    dtype=np.float64,
                )
                for observation in observations
            ]
        ),
        "object_pose_wxyz": np.stack(
            [
                np.asarray(observation.privileged["object_pose_wxyz"], dtype=np.float64)
                for observation in observations
            ]
        ),
        "fingerpad_contact_flags": np.stack(
            [
                np.asarray(
                    observation.privileged["fingerpad_contact_flags"],
                    dtype=np.float64,
                )
                for observation in observations
            ]
        ),
        **causal_arrays,
    }
    trace = None
    if capture_trace:
        trace = EpisodeTrace(
            request=request,
            observations=tuple(observations),
            actions=tuple(actions),
            step_results=tuple(step_results),
            teacher_phases=tuple(
                f"best_known/{candidate.spec.candidate_id}" for _ in actions
            ),
            events=events,
            teacher_preparation={
                "method": "rlinf_best_known_candidate_selection",
                "candidate": _candidate_identity(candidate.spec),
                "candidate_index": candidate.index,
                "selection_contract": SELECTION_CONTRACT,
                **dict(trace_metadata or {}),
            },
            replay_validation=replay_validation,
            action_timing={
                "policy_rate_hz": 20.0,
                "control_steps": len(actions),
                "candidate_id": candidate.spec.candidate_id,
            },
        )
    return record, arrays, trace


def _make_env(
    *,
    task: str,
    split: str,
    manifest_seed: int,
    manifest_size: int,
    image_size: int,
    camera_observations: bool,
    policy: Mapping[str, Any] | None = None,
    task_quality_schema_version: str | None = None,
    task_quality_evaluator_backend_id: str | None = None,
) -> Any:
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    config = {
        "task_id": task,
        "split": split,
        "manifest_seed": manifest_seed,
        "manifest_size": manifest_size,
        "image_size": image_size,
        "camera_observations": camera_observations,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
    }
    if task_quality_schema_version is not None:
        config.update(
            task_quality_schema_version=task_quality_schema_version,
            task_quality_evaluator_backend_id=task_quality_evaluator_backend_id,
        )
    if policy is not None:
        config.update(
            features=policy.get("features", {}),
            reward_components=policy.get("reward_components", {}),
            reward_lift_shaping_weight=float(
                policy.get("reward_lift_shaping_weight", 0.0)
            ),
            reward_orientation_shaping_weight=float(
                policy.get("reward_orientation_shaping_weight", 0.0)
            ),
            state_derived_features=list(policy.get("state_derived_features", [])),
        )
    return DynamicBenchmarkEnv(
        cfg=config,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def _root_checksums(root: Path) -> int:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and ".staging" not in path.parts
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
        ),
        encoding="utf-8",
    )
    return len(paths)


def _compatibility_evidence_rows(
    evidence: tuple[CompatibilityEvidence, ...],
) -> list[dict[str, str]]:
    return [
        {
            "policy_benchmark_commit": row.policy_benchmark_commit,
            "relative_path": f"provenance/evidence/{row.sha256}.json",
            "sha256": row.sha256,
        }
        for row in evidence
    ]


def _materialize_compatibility_evidence(
    output: Path,
    evidence: tuple[CompatibilityEvidence, ...],
) -> None:
    """Copy validated source evidence into the self-contained dataset root."""

    for source, frozen in zip(
        evidence,
        _compatibility_evidence_rows(evidence),
        strict=True,
    ):
        _copy_provenance_file(
            output,
            source.source_path,
            frozen["relative_path"],
            source.sha256,
        )


def _calibration_evidence_row(
    evidence: CalibrationEvidence | None,
) -> dict[str, str] | None:
    if evidence is None:
        return None
    return {
        "relative_path": f"provenance/calibration/{evidence.sha256}.json",
        "sha256": evidence.sha256,
    }


def _provenance_file_rows(files: tuple[ProvenanceFile, ...]) -> list[dict[str, str]]:
    return [{"relative_path": row.relative_path, "sha256": row.sha256} for row in files]


def _copy_provenance_file(
    output: Path, source: Path, relative: str, sha256: str
) -> None:
    normalized = _safe_dataset_relative_path(relative, label="dataset provenance path")
    destination = output.joinpath(*PurePosixPath(normalized).parts)
    if not destination.resolve().is_relative_to(output.resolve()):
        raise ValueError("dataset provenance destination escapes the dataset root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("dataset provenance destination must not be a symbolic link")
    if not destination.exists():
        shutil.copyfile(source, destination)
    if not destination.is_file() or _sha256(destination) != sha256:
        raise ValueError("dataset provenance copy does not match its SHA-256")


def _materialize_additional_provenance(
    output: Path,
    *,
    calibration: CalibrationEvidence | None,
    release_files: tuple[ProvenanceFile, ...],
) -> None:
    if calibration is not None:
        row = _calibration_evidence_row(calibration)
        assert row is not None
        _copy_provenance_file(
            output,
            calibration.source_path,
            row["relative_path"],
            calibration.sha256,
        )
    for source in release_files:
        _copy_provenance_file(
            output,
            source.source_path,
            source.relative_path,
            source.sha256,
        )


def main() -> None:
    from se3_wam.benchmark.contracts import canonical_json
    from se3_wam.benchmark.dataset import write_episode_atomic
    from se3_wam.benchmark.evaluation import manifest_record

    args = _parser().parse_args()
    import yaml

    if not args.quality_v2_thresholds.is_file():
        raise FileNotFoundError(args.quality_v2_thresholds)
    quality_v2_thresholds_sha256 = _expected_sha256(
        args.expected_quality_v2_thresholds_sha256,
        "expected quality-v2 threshold SHA-256",
    )
    if _sha256(args.quality_v2_thresholds) != quality_v2_thresholds_sha256:
        raise ValueError("quality-v2 threshold contract SHA-256 mismatch")
    payload = yaml.safe_load(args.quality_v2_thresholds.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("quality-v2 threshold contract must be a YAML mapping")
    quality_v2_thresholds: Mapping[str, object] = dict(payload)
    if quality_v2_thresholds.get("formal_freeze_eligible") is not True:
        raise ValueError(
            "quality-v2 threshold contract is not eligible for formal freeze"
        )
    quality_v2_threshold_identity = {
        "schema_version": quality_v2_thresholds.get("schema_version"),
        "sha256": quality_v2_thresholds_sha256,
    }
    if (
        args.shard_count < 1
        or args.shard_index < 0
        or args.shard_index >= args.shard_count
    ):
        raise ValueError("shard-index must be in [0, shard-count)")
    sharded = args.shard_count > 1
    shard_output = (
        args.output / f"shard-{args.shard_index:02d}" if sharded else args.output
    )
    if shard_output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.resume and not shard_output.is_dir():
        raise FileNotFoundError("--resume requires an existing export directory")
    if args.accepted_episodes < 1 or args.max_resets < args.accepted_episodes:
        raise ValueError("max_resets must be at least accepted_episodes > 0")
    if args.image_size < 64:
        raise ValueError("image_size must be at least 64")
    if (
        args.selection_mode == PLANNER_PARETO_SELECTION_MODE
        and args.candidate_search_mode != FULL_POOL_SEARCH_MODE
    ):
        raise ValueError(
            "planner-pareto selection requires candidate-search-mode=full-pool"
        )
    selection_contract = _selection_contract(args.selection_mode)
    evaluator_commit = _full_commit("evaluator_commit", args.evaluator_commit)
    candidate_manifest_sha256 = _expected_sha256(
        args.expected_candidate_manifest_sha256,
        "expected candidate manifest SHA-256",
    )
    if _sha256(args.candidate_manifest) != candidate_manifest_sha256:
        raise ValueError("candidate manifest SHA-256 mismatch")
    candidate_payload = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    candidate_schema_version = candidate_payload.get("schema_version")
    if candidate_schema_version == CANDIDATE_SCHEMA:
        if args.rlinf_commit is None or args.benchmark_commit is None:
            raise ValueError(
                "candidate schema v0.1 requires --rlinf-commit and --benchmark-commit"
            )
        rlinf_commit = _full_commit("rlinf_commit", args.rlinf_commit)
        benchmark_commit = _full_commit("benchmark_commit", args.benchmark_commit)
        evaluator_benchmark_commit = (
            None
            if args.evaluator_benchmark_commit is None
            else _full_commit(
                "evaluator_benchmark_commit",
                args.evaluator_benchmark_commit,
            )
        )
    elif candidate_schema_version == CANDIDATE_SCHEMA_V2:
        if args.rlinf_commit is not None or args.benchmark_commit is not None:
            raise ValueError(
                "candidate schema v0.2 forbids legacy singular policy commit arguments"
            )
        if args.evaluator_benchmark_commit is None:
            raise ValueError(
                "candidate schema v0.2 requires --evaluator-benchmark-commit"
            )
        rlinf_commit = None
        benchmark_commit = None
        evaluator_benchmark_commit = _full_commit(
            "evaluator_benchmark_commit",
            args.evaluator_benchmark_commit,
        )
    else:
        raise ValueError("unsupported optimal-trajectory candidate schema")
    (
        candidate_release_manifest_sha256,
        candidate_release_provenance_files,
    ) = _validate_candidate_release_chain(
        candidate_manifest=args.candidate_manifest,
        candidate_manifest_sha256=candidate_manifest_sha256,
        candidate_payload=candidate_payload,
        release_manifest=args.candidate_release_manifest,
        expected_release_manifest_sha256=(
            args.expected_candidate_release_manifest_sha256
        ),
    )
    task, specs = _validate_candidate_manifest(
        candidate_payload,
        manifest_path=args.candidate_manifest.resolve(),
        rlinf_commit=rlinf_commit,
        benchmark_commit=benchmark_commit,
        max_k=args.max_k
        if args.candidate_search_mode == FIRST_ELIGIBLE_SEARCH_MODE
        else 1,
    )
    planner_dominance = _validate_planner_dominance_contract(
        candidate_payload,
        task=task,
        selection_mode=args.selection_mode,
    )
    quality_v2_dominance = (
        _quality_v2_dominance_contract(
            quality_v2_thresholds,
            task=task,
            thresholds_sha256=quality_v2_thresholds_sha256,
        )
        if args.selection_mode == PLANNER_PARETO_SELECTION_MODE
        else None
    )
    evaluator_identity, compatibility_evidence = _validate_evaluator_identity(
        candidate_payload,
        manifest_path=args.candidate_manifest.resolve(),
        specs=specs,
        evaluator_rlinf_commit=evaluator_commit,
        evaluator_benchmark_commit=evaluator_benchmark_commit,
        planner_dominance=planner_dominance,
    )
    calibration_benchmark_commit = _full_commit(
        "authenticated evaluator benchmark commit",
        benchmark_commit
        if evaluator_identity is None
        else evaluator_identity["evaluator_benchmark_commit"],
    )
    quality_v2_calibration_receipt = _validate_quality_v2_calibration_receipt_artifact(
        quality_v2_thresholds,
        args.quality_v2_calibration_wave_receipt,
        expected_sha256=(args.expected_quality_v2_calibration_wave_receipt_sha256),
        expected_benchmark_commit=calibration_benchmark_commit,
    )
    calibration_evidence = _validate_calibration_evidence(
        manifest_path=args.candidate_manifest.resolve(),
        planner_dominance=planner_dominance,
        evaluator_identity=evaluator_identity,
    )
    if planner_dominance is not None and any(spec.provenance is None for spec in specs):
        raise ValueError(
            "planner-pareto candidate manifest requires provenance for every candidate"
        )
    budgets = _candidate_budgets(
        args.candidate_search_mode,
        initial_k=args.initial_k,
        max_k=args.max_k,
        candidate_pool_size=len(specs),
    )
    device = _device(args.device)
    candidates = _load_candidates(
        specs,
        task=task,
        rlinf_commit=rlinf_commit,
        benchmark_commit=benchmark_commit,
        device=device,
    )
    manifest_size = args.max_resets + args.max_resets % 2
    quality_schema_version = (
        None
        if planner_dominance is None
        else str(planner_dominance["quality_schema"]["schema_version"])
    )
    quality_backend_id = (
        None if planner_dominance is None else str(planner_dominance["backend_id"])
    )

    def make_env_pair(
        policy: Mapping[str, Any] | None,
    ) -> tuple[Any, Any]:
        common = {
            "task": task,
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "manifest_size": manifest_size,
            "policy": policy,
            "task_quality_schema_version": quality_schema_version,
            "task_quality_evaluator_backend_id": quality_backend_id,
        }
        return (
            _make_env(
                **common,
                image_size=64,
                camera_observations=False,
            ),
            _make_env(
                **common,
                image_size=args.image_size,
                camera_observations=True,
            ),
        )

    light_env, render_env = make_env_pair(None)
    default_schema_key = canonical_json(light_env.state_schema)
    schema_configs: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if candidate.state_schema is None:
            continue
        assert candidate.config is not None
        schema_configs.setdefault(
            canonical_json(candidate.state_schema), candidate.config
        )
    env_pairs = {default_schema_key: (light_env, render_env)}
    for schema_key, policy_config in schema_configs.items():
        if schema_key == default_schema_key:
            continue
        pair = make_env_pair(policy_config)
        if canonical_json(pair[0].state_schema) != schema_key:
            pair[0].close()
            pair[1].close()
            raise ValueError(
                "export environment state schema does not match policy group"
            )
        env_pairs[schema_key] = pair
    candidate_env_keys = [
        default_schema_key
        if candidate.state_schema is None
        else canonical_json(candidate.state_schema)
        for candidate in candidates
    ]

    def reset_all_envs() -> None:
        for lightweight, rendered in env_pairs.values():
            lightweight.reset(options={"env_idx": [0]})
            rendered.reset(options={"env_idx": [0]})

    try:
        light_manifest = [manifest_record(row) for row in light_env._manifest_rows]
        render_manifest = [manifest_record(row) for row in render_env._manifest_rows]
        if canonical_json(light_manifest) != canonical_json(render_manifest):
            raise RuntimeError("lightweight and render manifests disagree")
        for variant_light, variant_render in env_pairs.values():
            if canonical_json(
                [manifest_record(row) for row in variant_light._manifest_rows]
            ) != canonical_json(light_manifest) or canonical_json(
                [manifest_record(row) for row in variant_render._manifest_rows]
            ) != canonical_json(light_manifest):
                raise RuntimeError("state-schema environment manifests disagree")
        rows = list(light_env._manifest_rows[: args.max_resets])
        if sharded:
            step = (len(rows) + args.shard_count - 1) // args.shard_count
            start = args.shard_index * step
            shard_rows = rows[start : start + step]
            run_output = shard_output
            for _ in range(start):
                reset_all_envs()
        else:
            start = 0
            shard_rows = rows
            run_output = args.output
        reset_manifest_text = "".join(
            canonical_json(manifest_record(row)) + "\n" for row in rows
        )
        reset_manifest_sha256 = hashlib.sha256(
            reset_manifest_text.encode("utf-8")
        ).hexdigest()
        if candidate_schema_version == CANDIDATE_SCHEMA:
            assert rlinf_commit is not None and benchmark_commit is not None
            source_identity = {
                "evaluator_rlinf_commit": evaluator_commit,
                "policy_rlinf_commit": rlinf_commit,
                "benchmark_commit": benchmark_commit,
            }
        else:
            assert evaluator_benchmark_commit is not None
            source_identity = {
                "evaluator_rlinf_commit": evaluator_commit,
                "evaluator_benchmark_commit": evaluator_benchmark_commit,
                "policy_rlinf_commits": list(candidate_payload["policy_rlinf_commits"]),
                "policy_benchmark_commits": list(
                    candidate_payload["policy_benchmark_commits"]
                ),
            }
        compatibility_evidence_rows = _compatibility_evidence_rows(
            compatibility_evidence
        )
        calibration_evidence_row = _calibration_evidence_row(calibration_evidence)
        candidate_release_provenance = _provenance_file_rows(
            candidate_release_provenance_files
        )
        export_state = {
            "schema_version": STATE_SCHEMA,
            "task": task,
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "manifest_size": manifest_size,
            "max_resets": args.max_resets,
            "accepted_target": args.accepted_episodes,
            "candidate_search_mode": args.candidate_search_mode,
            "candidate_pool_size": len(candidates),
            "initial_k": budgets[0],
            "max_k": budgets[-1],
            "budget_sequence": list(budgets),
            "selection_mode": args.selection_mode,
            "selection_contract": selection_contract,
            "planner_dominance": planner_dominance,
            "candidate_schema_version": candidate_schema_version,
            "evaluator_identity": evaluator_identity,
            "compatibility_evidence": compatibility_evidence_rows,
            "calibration_evidence": calibration_evidence_row,
            "candidate_release_manifest_sha256": candidate_release_manifest_sha256,
            "candidate_release_provenance": candidate_release_provenance,
            "image_size": args.image_size,
            "device": str(device),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "reset_manifest_sha256": reset_manifest_sha256,
            "source_identity": source_identity,
            "quality_v2_threshold_identity": quality_v2_threshold_identity,
            "state_schema": json.loads(
                json.dumps(light_env.state_schema, allow_nan=False)
            ),
            "candidates": [
                _candidate_identity(candidate.spec) for candidate in candidates
            ],
        }
        export_state["payload_sha256"] = _payload_sha256(export_state)
        attempts_path = run_output / "attempts.jsonl"
        winners_path = run_output / "winner_manifest.jsonl"
        reset_results_path = run_output / "reset_results.jsonl"
        reset_manifest_path = run_output / "reset_manifest.jsonl"
        export_state_path = run_output / "export_state.json"
        progress_path = run_output / "progress.json"
        if args.resume:
            if (run_output / "dataset_card.json").exists() or (
                run_output / "SHA256SUMS"
            ).exists():
                raise ValueError("refusing to resume a sealed export")
            if (
                _sha256(run_output / "candidate_manifest.json")
                != candidate_manifest_sha256
            ):
                raise ValueError("resume candidate-manifest copy checksum mismatch")
            if (
                _sha256(run_output / "quality_v2_thresholds.json")
                != quality_v2_thresholds_sha256
            ):
                raise ValueError("resume quality-v2 threshold copy checksum mismatch")
            _copy_provenance_file(
                run_output,
                quality_v2_calibration_receipt.source_path,
                quality_v2_calibration_receipt.relative_path,
                quality_v2_calibration_receipt.sha256,
            )
            _materialize_compatibility_evidence(run_output, compatibility_evidence)
            _materialize_additional_provenance(
                run_output,
                calibration=calibration_evidence,
                release_files=candidate_release_provenance_files,
            )
            if reset_manifest_path.read_text(encoding="utf-8") != reset_manifest_text:
                raise ValueError(
                    "resume reset manifest does not match the requested run"
                )
            stored_state = json.loads(export_state_path.read_text(encoding="utf-8"))
            if _payload_sha256(stored_state) != stored_state.get("payload_sha256"):
                raise ValueError("resume export-state payload checksum mismatch")
            if stored_state != export_state:
                raise ValueError(
                    "resume arguments or resolved candidate identities changed"
                )
            export_state_sha256 = _sha256(export_state_path)
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("schema_version") != PROGRESS_SCHEMA or (
                _payload_sha256(progress) != progress.get("payload_sha256")
            ):
                raise ValueError("resume progress schema or payload checksum mismatch")
            if progress.get("export_state_sha256") != export_state_sha256:
                raise ValueError("resume progress references a different export state")
            recovery_event = _recover_partial_output(
                output=run_output,
                progress=progress,
                reset_rows=shard_rows,
                task=task,
                split=args.split,
            )
            started = float(progress["started_unix_s"])
            accepted = int(progress["accepted_count"])
            attempted_resets = int(progress["next_reset_index"])
            attempt_count = int(progress["candidate_attempt_count"])
            budget_histogram = {
                str(key): int(value)
                for key, value in progress["budget_histogram"].items()
            }
            if set(budget_histogram) != {str(budget) for budget in budgets}:
                raise ValueError("resume budget histogram keys changed")
            resume_count = int(progress["resume_count"]) + 1
            recovery_events = list(progress["recovery_events"])
            if recovery_event is not None:
                recovery_events.append(recovery_event)
            expected_line_counts = {
                attempts_path: attempt_count,
                reset_results_path: attempted_resets,
                winners_path: accepted,
            }
            for path, expected_count in expected_line_counts.items():
                with path.open("r", encoding="utf-8") as stream:
                    actual_count = sum(1 for line in stream if line.strip())
                if actual_count != expected_count:
                    raise ValueError(
                        f"resume committed line count mismatch for {path.name}"
                    )
            _atomic_json(
                progress_path,
                _progress_payload(
                    export_state_sha256=export_state_sha256,
                    started_unix_s=started,
                    next_reset_index=attempted_resets,
                    accepted_count=accepted,
                    candidate_attempt_count=attempt_count,
                    budget_histogram=budget_histogram,
                    attempts_path=attempts_path,
                    reset_results_path=reset_results_path,
                    winners_path=winners_path,
                    resume_count=resume_count,
                    recovery_events=recovery_events,
                ),
            )
        else:
            run_output.mkdir(parents=True)
            shutil.copyfile(
                args.candidate_manifest, run_output / "candidate_manifest.json"
            )
            shutil.copyfile(
                args.quality_v2_thresholds,
                run_output / "quality_v2_thresholds.json",
            )
            _copy_provenance_file(
                run_output,
                quality_v2_calibration_receipt.source_path,
                quality_v2_calibration_receipt.relative_path,
                quality_v2_calibration_receipt.sha256,
            )
            _materialize_compatibility_evidence(run_output, compatibility_evidence)
            _materialize_additional_provenance(
                run_output,
                calibration=calibration_evidence,
                release_files=candidate_release_provenance_files,
            )
            reset_manifest_path.write_text(reset_manifest_text, encoding="utf-8")
            for path in (attempts_path, winners_path, reset_results_path):
                path.write_text("", encoding="utf-8")
            _atomic_json(export_state_path, export_state)
            export_state_sha256 = _sha256(export_state_path)
            started = time.time()
            accepted = 0
            attempted_resets = 0
            attempt_count = 0
            budget_histogram = {str(budget): 0 for budget in budgets}
            resume_count = 0
            recovery_events: list[str] = []
            _atomic_json(
                progress_path,
                _progress_payload(
                    export_state_sha256=export_state_sha256,
                    started_unix_s=started,
                    next_reset_index=0,
                    accepted_count=0,
                    candidate_attempt_count=0,
                    budget_histogram=budget_histogram,
                    attempts_path=attempts_path,
                    reset_results_path=reset_results_path,
                    winners_path=winners_path,
                    resume_count=0,
                    recovery_events=[],
                ),
            )
        for local_index, row in enumerate(shard_rows):
            reset_index = start + local_index
            if accepted >= args.accepted_episodes:
                break
            if local_index < attempted_resets:
                if local_index + 1 < len(shard_rows):
                    reset_all_envs()
                continue
            for variant_light, variant_render in env_pairs.values():
                light_request = variant_light._requests[0]
                render_request = variant_render._requests[0]
                if (
                    light_request is None
                    or render_request is None
                    or light_request.episode_id != row.request.episode_id
                    or render_request.episode_id != row.request.episode_id
                ):
                    raise RuntimeError(
                        "rollout order diverged from the frozen reset manifest"
                    )
            initial_states = {
                key: pair[0].checkpoint_state() for key, pair in env_pairs.items()
            }
            reset_attempts: list[dict[str, Any]] = []
            winner = None
            budget_used = budgets[-1]
            for budget in budgets:
                for candidate in candidates[len(reset_attempts) : budget]:
                    env_key = candidate_env_keys[candidate.index]
                    candidate_env = env_pairs[env_key][0]
                    _restore_candidate_start(candidate_env, initial_states[env_key])
                    record, arrays, _ = _rollout(
                        env=candidate_env,
                        candidate=candidate,
                        device=device,
                        capture_trace=False,
                        quality_v2_thresholds=quality_v2_thresholds,
                        quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
                    )
                    relative, tape_sha256 = _write_attempt_tape(
                        run_output,
                        episode_id=record["episode_id"],
                        candidate_index=candidate.index,
                        arrays=arrays,
                    )
                    record["attempt_tape"] = relative
                    record["attempt_tape_sha256"] = tape_sha256
                    record["quality_score"] = list(_quality_score(record))
                    record["eligible"] = _eligible(record)
                    _append_jsonl(attempts_path, record)
                    reset_attempts.append(record)
                    attempt_count += 1
                winner = _select_winner(
                    reset_attempts,
                    selection_mode=args.selection_mode,
                    planner_dominance=planner_dominance,
                    quality_v2_dominance=quality_v2_dominance,
                )
                budget_used = budget
                if (
                    winner is not None
                    and args.candidate_search_mode == FIRST_ELIGIBLE_SEARCH_MODE
                ):
                    break
            selection_result = _selection_result(
                reset_attempts,
                winner,
                selection_mode=args.selection_mode,
            )
            attempted_resets += 1
            budget_histogram[str(budget_used)] += 1
            reset_result = {
                "reset_index": reset_index,
                "episode_id": row.request.episode_id,
                "source_group_id": row.source_group_id,
                "candidate_count": len(reset_attempts),
                "budget_used": budget_used,
                "candidate_search_mode": args.candidate_search_mode,
                "selection_mode": args.selection_mode,
                "selection_result": selection_result,
                "accepted": winner is not None,
                "winner_candidate_id": None
                if winner is None
                else winner["candidate_id"],
                "winner_candidate_index": None
                if winner is None
                else winner["candidate_index"],
            }
            _append_jsonl(reset_results_path, reset_result)
            if winner is not None:
                candidate = candidates[int(winner["candidate_index"])]
                winner_env_key = candidate_env_keys[candidate.index]
                winner_render_env = env_pairs[winner_env_key][1]
                tape_path = run_output / winner["attempt_tape"]
                replay_actions_array = np.load(tape_path)["actions"]
                try:
                    render_record, _, trace = _rollout(
                        env=winner_render_env,
                        candidate=candidate,
                        device=device,
                        capture_trace=True,
                        replay_actions_array=replay_actions_array,
                        quality_v2_thresholds=quality_v2_thresholds,
                        quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
                        trace_metadata={
                            "candidate_manifest_sha256": candidate_manifest_sha256,
                            "budget_used": budget_used,
                            "candidate_search_mode": args.candidate_search_mode,
                            "selection_mode": args.selection_mode,
                            "selection_contract": selection_contract,
                            "planner_dominance": planner_dominance,
                            "evaluator_identity": evaluator_identity,
                            "compatibility_evidence": compatibility_evidence_rows,
                            "calibration_evidence": calibration_evidence_row,
                            "candidate_release_manifest_sha256": (
                                candidate_release_manifest_sha256
                            ),
                            "selection_result": selection_result,
                            "winner_quality_score": list(_quality_score(winner)),
                            "quality_v2": winner["quality_v2"],
                            "quality_v2_sha256": winner["quality_v2_sha256"],
                            "quality_v2_gate": winner.get("quality_v2_gate"),
                            "quality_v2_threshold_identity": quality_v2_threshold_identity,
                            "lightweight_action_sha256": winner["action_sha256"],
                            "source_identity": source_identity,
                        },
                    )
                    if trace is None:
                        raise RuntimeError(
                            "winner render did not return an episode trace"
                        )
                    parity_keys = (
                        "episode_id",
                        "success",
                        "safety_failure",
                        "termination_reason",
                        "trajectory_completion",
                        "completion_time_s",
                        "return",
                        "control_steps",
                        "action_l2_sum",
                        "action_sha256",
                        "quality_v2_sha256",
                        "quality_v2_gate",
                    )
                    if planner_dominance is not None:
                        parity_keys += ("task_quality",)
                    for key in parity_keys:
                        if render_record[key] != winner[key]:
                            raise RuntimeError(f"winner render parity failed for {key}")
                    episode_record = write_episode_atomic(run_output, trace)
                    winner_row = {
                        **episode_record,
                        "candidate_id": candidate.spec.candidate_id,
                        "candidate_index": candidate.index,
                        "candidate_count": len(reset_attempts),
                        "budget_used": budget_used,
                        "candidate_search_mode": args.candidate_search_mode,
                        "selection_mode": args.selection_mode,
                        "selection_contract": selection_contract,
                        "planner_dominance": planner_dominance,
                        "evaluator_identity": evaluator_identity,
                        "compatibility_evidence": compatibility_evidence_rows,
                        "calibration_evidence": calibration_evidence_row,
                        "candidate_release_manifest_sha256": (
                            candidate_release_manifest_sha256
                        ),
                        "selection_result": selection_result,
                        "quality_score": list(_quality_score(winner)),
                        "quality_v2": winner["quality_v2"],
                        "quality_v2_sha256": winner["quality_v2_sha256"],
                        "quality_v2_gate": winner.get("quality_v2_gate"),
                        "lightweight_attempt_tape": winner["attempt_tape"],
                        "lightweight_attempt_tape_sha256": winner[
                            "attempt_tape_sha256"
                        ],
                    }
                    _append_jsonl(winners_path, winner_row)
                    accepted += 1
                    print(
                        json.dumps(
                            {
                                "accepted": accepted,
                                "episode_id": winner["episode_id"],
                                "candidate_id": winner["candidate_id"],
                                "budget_used": budget_used,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except (RuntimeError, ValueError) as exc:
                    if _render_parity_failure_reason(exc) is None:
                        raise
                    render_parity_skip = _render_parity_skip(winner, exc)
                    _rewrite_last_jsonl(
                        reset_results_path,
                        {
                            **reset_result,
                            "accepted": False,
                            "winner_candidate_id": None,
                            "winner_candidate_index": None,
                            "render_parity_skip": render_parity_skip,
                        },
                    )
                    winner = None
                    recovery_events.append(
                        f"render_parity_skip:reset:{reset_index}:"
                        f"{row.request.episode_id}:{str(exc)}"
                    )
            _atomic_json(
                progress_path,
                _progress_payload(
                    export_state_sha256=export_state_sha256,
                    started_unix_s=started,
                    next_reset_index=local_index + 1,
                    accepted_count=accepted,
                    candidate_attempt_count=attempt_count,
                    budget_histogram=budget_histogram,
                    attempts_path=attempts_path,
                    reset_results_path=reset_results_path,
                    winners_path=winners_path,
                    resume_count=resume_count,
                    recovery_events=recovery_events,
                ),
            )
            if local_index + 1 < len(shard_rows):
                reset_all_envs()

        if _sha256(args.candidate_manifest) != candidate_manifest_sha256:
            raise RuntimeError("candidate manifest changed during export")
        for candidate in candidates:
            if candidate.spec.kind != "policy":
                continue
            assert candidate.spec.policy_path is not None
            assert candidate.spec.policy_sha256 is not None
            if _sha256(candidate.spec.policy_path) != candidate.spec.policy_sha256:
                raise RuntimeError(
                    f"candidate policy changed during export: {candidate.spec.candidate_id}"
                )
        if sharded:
            _atomic_json(
                run_output / "shard_complete.json",
                {
                    "schema_version": "rlinf-dynamic-benchmark-optimal-shard-v0.1",
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "accepted_count": accepted,
                    "attempted_reset_count": attempted_resets,
                    "candidate_attempt_count": attempt_count,
                    "candidate_search_mode": args.candidate_search_mode,
                    "selection_mode": args.selection_mode,
                    "budget_histogram": dict(budget_histogram),
                },
            )
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "accepted": accepted,
                        "attempted_resets": attempted_resets,
                        "candidate_attempts": attempt_count,
                        "budget_histogram": budget_histogram,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        status = "complete" if accepted == args.accepted_episodes else "incomplete"
        card = {
            "schema_version": EXPORT_SCHEMA,
            "status": status,
            "training_eligible": False,
            "training_eligibility_reason": "independent audit has not yet passed",
            "optimality_claim": "best-known under the frozen candidate/reset/budget contract",
            "task": task,
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "accepted_target": args.accepted_episodes,
            "accepted_count": accepted,
            "attempted_reset_count": attempted_resets,
            "candidate_attempt_count": attempt_count,
            "candidate_search_mode": args.candidate_search_mode,
            "candidate_pool_size": len(candidates),
            "initial_k": budgets[0],
            "max_k": budgets[-1],
            "budget_sequence": list(budgets),
            "budget_histogram": budget_histogram,
            "selection_mode": args.selection_mode,
            "selection_contract": selection_contract,
            "planner_dominance": planner_dominance,
            "candidate_schema_version": candidate_schema_version,
            "evaluator_identity": evaluator_identity,
            "compatibility_evidence": compatibility_evidence_rows,
            "calibration_evidence": calibration_evidence_row,
            "candidate_release_manifest_sha256": candidate_release_manifest_sha256,
            "candidate_release_provenance": candidate_release_provenance,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "reset_manifest_sha256": _sha256(reset_manifest_path),
            "export_state_sha256": export_state_sha256,
            "progress_sha256": _sha256(progress_path),
            "resume_count": resume_count,
            "recovery_events": recovery_events,
            "source_identity": source_identity,
            "quality_v2_threshold_identity": quality_v2_threshold_identity,
            "image_size": args.image_size,
            "device": str(device),
            "started_unix_s": started,
            "finished_unix_s": time.time(),
        }
        card["payload_sha256"] = _payload_sha256(card)
        _atomic_json(args.output / "dataset_card.json", card)
        checksum_count = _root_checksums(args.output)
        print(
            json.dumps(
                {
                    "status": status,
                    "accepted": accepted,
                    "attempted_resets": attempted_resets,
                    "candidate_attempts": attempt_count,
                    "checksum_entries": checksum_count,
                    "dataset_card_payload_sha256": card["payload_sha256"],
                },
                sort_keys=True,
            )
        )
        if status != "complete":
            raise RuntimeError(
                f"accepted {accepted}/{args.accepted_episodes} winners within {attempted_resets} resets"
            )
    finally:
        for lightweight, rendered in env_pairs.values():
            lightweight.close()
            rendered.close()


if __name__ == "__main__":
    main()
