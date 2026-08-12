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

"""Qv4 attempt, full-export replay, and paired-Pareto contracts.

This module is intentionally parallel to the released Qv3 ``quality_v2``
pipeline.  It never mutates or aliases a Qv3 schema, threshold path, or SHA.
The lightweight attempt is useful for every candidate; only the selected
winner needs the complete HDF5 source tree and its independent recomputation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

QUALITY_V4_ATTEMPT_SCHEMA = "rlinf-dynamic-benchmark-quality-v4-attempt-v0.1"
QUALITY_V4_FULL_EXPORT_SCHEMA = "rlinf-dynamic-benchmark-quality-v4-full-export-v0.1"
QUALITY_V4_FULL_EXPORT_GATE_SCHEMA = (
    "rlinf-dynamic-benchmark-quality-v4-full-export-gate-v0.1"
)
QUALITY_V4_LIGHTWEIGHT_SOURCE_SCHEMA = (
    "rlinf-dynamic-benchmark-quality-v4-lightweight-source-v0.1"
)
QUALITY_V4_PAIRED_PARETO_SCHEMA = (
    "rlinf-dynamic-benchmark-quality-v4-paired-pareto-v0.1"
)
QUALITY_V4_DATASET_VALIDATION_SCHEMA = (
    "se3-wam-trajectory-quality-v4-export-validation-v0.1"
)
QUALITY_V4_VISION_TOLERANCE_SCHEMA = (
    "rlinf-dynamic-benchmark-quality-v4-vision-tolerance-v0.1"
)
QUALITY_V4_SEGMENTATION_CONTRACT_SCHEMA = (
    "rlinf-dynamic-benchmark-quality-v4-segmentation-contract-v0.1"
)
QUALITY_V4_ROLLOUT_REFERENCE_SCHEMA = (
    "rlinf-dynamic-benchmark-quality-v4-rollout-reference-v0.1"
)
QUALITY_V4_T5_ACTION_HISTORY_SCHEMA = (
    "rlinf-dynamic-benchmark-quality-v4-t5-action-history-v0.1"
)
QUALITY_V4_ARTIFACT_SUBDIRECTORY = Path("quality_v4")
QUALITY_V4_ATTEMPT_SUBDIRECTORY = QUALITY_V4_ARTIFACT_SUBDIRECTORY / "attempts"
QUALITY_V4_LIGHTWEIGHT_SUBDIRECTORY = (
    QUALITY_V4_ARTIFACT_SUBDIRECTORY / "lightweight_sources"
)
QUALITY_V4_FULL_EXPORT_SUBDIRECTORY = QUALITY_V4_ARTIFACT_SUBDIRECTORY / "full_exports"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CALIBRATION_SOURCE_KEYS = frozenset(
    {
        "exact14x20_planner",
        "known_good_bad_trajectories",
        "fresh_deterministic_rl_pilot",
    }
)
_FORBIDDEN_TUNING_SPLITS = frozenset({"test_id", "test_ood"})
_QUALITY_V4_CHECK_VALUE_EVIDENCE_KEYS = frozenset(
    {
        "hard_bound_sha256",
        "good_bad_discriminability_sha256",
        "paired_non_worse_tolerance_sha256",
        "strict_improvement_margin_sha256",
    }
)
_QUALITY_V4_FULL_EPISODE_METRICS = (
    "action.issued.continuous_first_difference_l2_sum",
    "action.issued.continuous_first_difference_l2_mean",
    "action.issued.continuous_first_difference_l2_max",
    "action.issued.continuous_second_difference_l2_sum",
    "action.issued.continuous_second_difference_l2_mean",
    "action.issued.continuous_second_difference_l2_max",
    "action.issued.continuous_direction_reversal_count",
    "action.issued.continuous_oscillation_magnitude",
    "action.applied.continuous_first_difference_l2_sum",
    "action.applied.continuous_first_difference_l2_mean",
    "action.applied.continuous_first_difference_l2_max",
    "action.applied.continuous_second_difference_l2_sum",
    "action.applied.continuous_second_difference_l2_mean",
    "action.applied.continuous_second_difference_l2_max",
    "action.applied.continuous_direction_reversal_count",
    "action.applied.continuous_oscillation_magnitude",
    "eef_control_rate.eef_linear_acceleration_max_m_s2",
    "eef_control_rate.eef_angular_acceleration_max_rad_s2",
    "eef_control_rate.eef_linear_jerk_max_m_s3",
    "eef_control_rate.eef_angular_jerk_max_rad_s3",
    "eef_physics_rate.eef_linear_velocity_rms_m_s",
    "eef_physics_rate.eef_linear_velocity_max_m_s",
    "eef_physics_rate.eef_angular_velocity_rms_rad_s",
    "eef_physics_rate.eef_angular_velocity_max_rad_s",
    "eef_physics_rate.eef_linear_acceleration_rms_m_s2",
    "eef_physics_rate.eef_linear_acceleration_max_m_s2",
    "eef_physics_rate.eef_angular_acceleration_rms_rad_s2",
    "eef_physics_rate.eef_angular_acceleration_max_rad_s2",
    "eef_physics_rate.eef_linear_jerk_rms_m_s3",
    "eef_physics_rate.eef_linear_jerk_max_m_s3",
    "eef_physics_rate.eef_angular_jerk_rms_rad_s3",
    "eef_physics_rate.eef_angular_jerk_max_rad_s3",
)
_QUALITY_V4_PHASE_BASE_METRICS = (
    "action_issued.continuous_first_difference_l2_max",
    "action_issued.continuous_second_difference_l2_max",
    "action_issued.continuous_direction_reversal_count",
    "action_applied.continuous_first_difference_l2_max",
    "action_applied.continuous_second_difference_l2_max",
    "action_applied.continuous_direction_reversal_count",
    "eef_control_rate.eef_linear_acceleration_max_m_s2",
    "eef_control_rate.eef_angular_acceleration_max_rad_s2",
    "eef_control_rate.eef_linear_jerk_max_m_s3",
    "eef_control_rate.eef_angular_jerk_max_rad_s3",
    "path_efficiency.excess_path_ratio",
    "path_efficiency.maximum_corridor_deviation_m",
    "path_efficiency.corridor_violation_duration_samples",
    "path_efficiency.progress_backtrack_sum",
    "path_efficiency.progress_backtrack_max",
    "path_efficiency.direction_reversal_count",
)


def _se3_quality() -> Any:
    """Import the Qv4 contract lazily from the pinned benchmark checkout."""

    from se3_wam.benchmark import trajectory_quality_v4

    return trajectory_quality_v4


def _quality_v4_orientation_metric_paths(row: Mapping[str, Any]) -> tuple[str, ...]:
    checks = {str(value) for value in row.get("checks", ())}
    metrics: list[str] = []
    if "tool_axis" in checks:
        metrics.extend(
            (
                "orientation.tool_axis.max_rad",
                "orientation.tool_axis.p95_rad",
                "orientation.tool_axis.duration_over_limit_s",
            )
        )
    if "jaw_axis" in checks:
        metrics.extend(
            (
                "orientation.jaw_axis.max_rad",
                "orientation.jaw_axis.p95_rad",
                "orientation.jaw_axis.duration_over_limit_s",
            )
        )
    if checks & {"full_orientation", "full_orientation_stability", "release_pose"}:
        metrics.extend(
            (
                "orientation.full_orientation.max_rad",
                "orientation.full_orientation.p95_rad",
                "orientation.full_orientation.duration_over_limit_s",
            )
        )
    if checks & {"relative_pose", "object_gripper_slip"}:
        metrics.extend(
            (
                "orientation.gripper_object_relative_so3_drift.max_rad",
                "orientation.gripper_object_relative_so3_drift.p95_rad",
                "orientation.gripper_object_relative_so3_drift.duration_over_limit_s",
            )
        )
    if checks & {"object_tilt", "roll_pitch"}:
        metrics.extend(
            (
                "orientation.object_tilt.max_rad",
                "orientation.object_tilt.p95_rad",
                "orientation.object_tilt.duration_over_limit_s",
            )
        )
    if "angular_dynamics" in checks:
        metrics.extend(
            (
                "orientation.object_angular_dynamics.eef_angular_velocity_max_rad_s",
                "orientation.object_angular_dynamics.eef_angular_acceleration_max_rad_s2",
                "orientation.object_angular_dynamics.eef_angular_jerk_max_rad_s3",
            )
        )
    if "close_instant" in checks:
        metrics.append("orientation.close_instant_tool_axis_error_rad")
        if "jaw_axis" in checks:
            metrics.append("orientation.close_instant_jaw_axis_error_rad")
    return tuple(dict.fromkeys(metrics))


def quality_v4_threshold_check_inventory() -> dict[str, tuple[tuple[str, str], ...]]:
    """Return the exact task/phase/metric inventory accepted by Qv4 thresholds."""

    quality = _se3_quality()
    inventory: dict[str, tuple[tuple[str, str], ...]] = {}
    for task_id, phases in quality.EXACT14_ORIENTATION_CONTRACT.items():
        checks = [
            ("full_episode", metric) for metric in _QUALITY_V4_FULL_EPISODE_METRICS
        ]
        for phase, phase_contract in phases.items():
            if phase_contract.get("applicable") is not True:
                continue
            checks.extend(
                (phase, metric)
                for metric in (
                    *_QUALITY_V4_PHASE_BASE_METRICS,
                    *_quality_v4_orientation_metric_paths(phase_contract),
                )
            )
        inventory[task_id] = tuple(checks)
    return inventory


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    from se3_wam.benchmark.contracts import stable_sha256

    return str(stable_sha256(dict(payload)))


def quality_v4_segmentation_contract() -> dict[str, Any]:
    """Return the frozen exact-segmentation replay contract.

    Segmentation remains part of the structured observation component digest, so
    replay requires byte-derived component identity with no tolerance or label remap.
    """

    quality = _se3_quality()
    evidence_sha256 = _stable_sha256(
        {
            "field_contract_schema": quality.QUALITY_V4_FIELD_CONTRACT_SCHEMA,
            "field": "observation.segmentation",
            "comparison": "structured_component_sha256_exact",
        }
    )
    payload: dict[str, Any] = {
        "schema_version": QUALITY_V4_SEGMENTATION_CONTRACT_SCHEMA,
        "calibration_status": "frozen",
        "comparison": "component_sha256_exact",
        "exact": True,
        "shape_exact": True,
        "dtype_exact": True,
        "label_remap_allowed": False,
        "task_ids": list(quality.EXACT14_ORIENTATION_CONTRACT),
        "evidence_sha256": evidence_sha256,
    }
    payload["contract_sha256"] = _stable_sha256(payload)
    return payload


def _canonical_json(payload: Any) -> str:
    from se3_wam.benchmark.contracts import canonical_json

    return str(canonical_json(payload))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} is not a safe portable identifier")
    return value


def _event_rows(raw: Any) -> tuple[Any, ...]:
    from se3_wam.benchmark.contracts import EventRecord

    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("Qv4 source events must be a sequence")
    events = []
    for index, row in enumerate(raw):
        if isinstance(row, EventRecord):
            event = row
        elif isinstance(row, Mapping):
            try:
                event = EventRecord(**dict(row))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Qv4 source event {index} is invalid") from error
        else:
            raise ValueError(f"Qv4 source event {index} is invalid")
        events.append(event)
    return tuple(events)


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Qv4 source metadata cannot contain NaN or Inf")
        return value
    raise TypeError(f"unsupported Qv4 source scalar {type(value).__name__}")


def _encode_source_tree(
    value: Any,
    arrays: dict[str, np.ndarray],
    *,
    path: str = "root",
) -> Any:
    """Replace arrays with content-addressed HDF5 dataset descriptors."""

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind in {"O", "U"}:
            raise ValueError(f"Qv4 source array {path} has an unsupported dtype")
        key = f"array_{len(arrays):06d}"
        arrays[key] = array
        return {
            "__qv4_array__": key,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _encode_source_tree(
                item,
                arrays,
                path=f"{path}.{key}",
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _encode_source_tree(item, arrays, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return _jsonable_scalar(value)


def _decode_source_tree(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, Mapping) and "__qv4_array__" in value:
        key = value["__qv4_array__"]
        if not isinstance(key, str) or key not in arrays:
            raise ValueError("Qv4 full export references a missing array")
        array = np.ascontiguousarray(arrays[key])
        if str(array.dtype) != value.get("dtype") or list(array.shape) != value.get(
            "shape"
        ):
            raise ValueError("Qv4 full-export array dtype/shape drift")
        if hashlib.sha256(array.tobytes()).hexdigest() != value.get("sha256"):
            raise ValueError("Qv4 full-export array checksum drift")
        return array
    if isinstance(value, Mapping):
        return {
            str(key): _decode_source_tree(item, arrays) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_source_tree(item, arrays) for item in value]
    return value


def quality_v4_source_manifest(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic no-pixels-in-JSON manifest of all source bytes."""

    arrays: dict[str, np.ndarray] = {}
    tree = _encode_source_tree(dict(source), arrays)
    payload: dict[str, Any] = {
        "tree": tree,
        "array_count": len(arrays),
    }
    payload["source_sha256"] = _stable_sha256(payload)
    return payload


def _sample_value(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _stack_observation_group(
    observations: Sequence[Any], name: str, *, dtype: np.dtype[Any]
) -> np.ndarray:
    groups = [getattr(observation, name) for observation in observations]
    camera_order = tuple(groups[0])
    if not camera_order or any(tuple(group) != camera_order for group in groups):
        raise ValueError(f"Qv4 observation {name} camera inventory drift")
    return np.stack(
        [
            np.stack(
                [np.asarray(group[camera], dtype=dtype) for camera in camera_order],
                axis=0,
            )
            for group in groups
        ],
        axis=0,
    )


def _physics_eef_pose(sample: Any) -> np.ndarray:
    direct = _sample_value(sample, "eef_pose_xyzw")
    if direct is not None:
        return np.asarray(direct, dtype=np.float64)
    position = _sample_value(sample, "eef_position_m")
    quaternion = _sample_value(sample, "eef_quaternion_xyzw")
    if position is None or quaternion is None:
        raise ValueError("Qv4 physics sample has no EEF pose")
    return np.asarray((*position, *quaternion), dtype=np.float64)


def _plain_t5_action_row(
    raw: Any, *, action_dimension: int, applied: bool
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Qv4 T5 action-history row must be a mapping")
    base_keys = {
        "policy_step",
        "issue_time_s",
        "apply_policy_step",
        "apply_time_s",
        "values",
    }
    required = (
        base_keys | {"actual_apply_policy_step", "actual_apply_time_s"}
        if applied
        else base_keys
    )
    if set(raw) != required:
        raise ValueError("Qv4 T5 action-history row inventory mismatch")
    integer_names = ["policy_step", "apply_policy_step"]
    if applied:
        integer_names.append("actual_apply_policy_step")
    result: dict[str, Any] = {}
    for name in integer_names:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Qv4 T5 action-history {name} must be an integer")
        result[name] = int(value)
    time_names = ["issue_time_s", "apply_time_s"]
    if applied:
        time_names.append("actual_apply_time_s")
    for name in time_names:
        value = raw[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Qv4 T5 action-history {name} must be finite")
        result[name] = float(value)
    values = np.asarray(raw["values"], dtype=np.float64)
    if values.shape != (action_dimension,) or not np.all(np.isfinite(values)):
        raise ValueError("Qv4 T5 action-history values are invalid")
    result["values"] = values.tolist()
    return result


def _normalize_quality_v4_t5_action_history(
    history: Any,
    *,
    issued: np.ndarray,
    issue_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not isinstance(history, Mapping):
        raise ValueError("Qv4 T5 source has no canonical issued/applied history")
    wrapped = "schema_version" in history
    if wrapped:
        if (
            set(history)
            != {
                "schema_version",
                "issued_actions",
                "applied_actions",
                "controller_applied_source_policy_step",
                "history_sha256",
            }
            or history.get("schema_version") != QUALITY_V4_T5_ACTION_HISTORY_SCHEMA
        ):
            raise ValueError("Qv4 T5 action-history schema or inventory mismatch")
        unsigned = dict(history)
        recorded_hash = unsigned.pop("history_sha256", None)
        if recorded_hash != _stable_sha256(unsigned):
            raise ValueError("Qv4 T5 action-history hash mismatch")
    elif set(history) != {"issued_actions", "applied_actions"}:
        raise ValueError("Qv4 T5 canonical action-history inventory mismatch")
    if issued.ndim != 2 or issue_time_s.shape != (issued.shape[0],):
        raise ValueError("Qv4 T5 issued action/time tape is misaligned")
    action_count, action_dimension = issued.shape
    raw_issued = history["issued_actions"]
    raw_applied = history["applied_actions"]
    if (
        isinstance(raw_issued, (str, bytes))
        or not isinstance(raw_issued, Sequence)
        or isinstance(raw_applied, (str, bytes))
        or not isinstance(raw_applied, Sequence)
    ):
        raise ValueError("Qv4 T5 action histories must be sequences")
    issued_rows = [
        _plain_t5_action_row(row, action_dimension=action_dimension, applied=False)
        for row in raw_issued
    ]
    applied_rows = [
        _plain_t5_action_row(row, action_dimension=action_dimension, applied=True)
        for row in raw_applied
    ]
    if len(issued_rows) != action_count:
        raise ValueError("Qv4 T5 issued action-history count mismatch")
    for index, row in enumerate(issued_rows):
        if (
            row["policy_step"] != index
            or not math.isclose(
                row["issue_time_s"],
                float(issue_time_s[index]),
                abs_tol=1.0e-12,
                rel_tol=0.0,
            )
            or not np.array_equal(np.asarray(row["values"]), issued[index])
        ):
            raise ValueError("Qv4 T5 issued history differs from the rollout tape")
        delay_steps = row["apply_policy_step"] - row["policy_step"]
        if delay_steps <= 0 or not math.isclose(
            row["apply_time_s"],
            row["issue_time_s"] + delay_steps / 20.0,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError("Qv4 T5 issued history has an invalid delay schedule")

    applied = np.zeros_like(issued)
    applied_time = issue_time_s.copy()
    applied_source_step = np.full(action_count, -1, dtype=np.int64)
    seen_source_steps: set[int] = set()
    seen_applied_steps: set[int] = set()
    for row in applied_rows:
        source_step = row["policy_step"]
        target_step = row["actual_apply_policy_step"]
        if (
            source_step < 0
            or source_step >= action_count
            or target_step < 0
            or target_step >= action_count
            or source_step in seen_source_steps
            or target_step in seen_applied_steps
        ):
            raise ValueError("Qv4 T5 applied history has an invalid queue mapping")
        issued_row = issued_rows[source_step]
        if (
            any(row[name] != issued_row[name] for name in issued_row)
            or target_step != row["apply_policy_step"]
            or not math.isclose(
                row["actual_apply_time_s"],
                float(issue_time_s[target_step]),
                abs_tol=1.0e-12,
                rel_tol=0.0,
            )
        ):
            raise ValueError("Qv4 T5 applied history differs from its queued issue")
        seen_source_steps.add(source_step)
        seen_applied_steps.add(target_step)
        applied[target_step] = issued[source_step]
        applied_source_step[target_step] = source_step
    expected_applied_sources = {
        int(row["policy_step"])
        for row in issued_rows
        if int(row["apply_policy_step"]) < action_count
    }
    if seen_source_steps != expected_applied_sources:
        raise ValueError("Qv4 T5 applied history omits or invents a scheduled action")
    payload: dict[str, Any] = {
        "schema_version": QUALITY_V4_T5_ACTION_HISTORY_SCHEMA,
        "issued_actions": issued_rows,
        "applied_actions": applied_rows,
        "controller_applied_source_policy_step": applied_source_step.tolist(),
    }
    payload["history_sha256"] = _stable_sha256(payload)
    if wrapped and dict(history) != payload:
        raise ValueError("Qv4 T5 stored action history does not normalize exactly")
    return applied, applied_time, applied_source_step, payload


def _quality_v4_applied_history(
    task_id: str,
    raw_env: Any,
    issued: np.ndarray,
    issue_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    if task_id != "t5_replan":
        return (
            issued.copy(),
            issue_time_s.copy(),
            np.arange(issued.shape[0], dtype=np.int64),
            None,
        )
    return _normalize_quality_v4_t5_action_history(
        getattr(raw_env, "canonical_action_history", None),
        issued=issued,
        issue_time_s=issue_time_s,
    )


def build_quality_v4_rollout_source(
    *,
    record: Mapping[str, Any],
    raw_env: Any,
    observations: Sequence[Any],
    issued_actions: Sequence[Sequence[float]] | np.ndarray,
    rewards: Sequence[float] | np.ndarray,
    outcomes: Sequence[Sequence[Any]],
    physics_samples: Sequence[Any],
    events: Sequence[Any],
    thresholds: Mapping[str, Any],
    reference_contract: Mapping[str, Any],
    replay_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the raw Qv4 source tape from one serial evaluator rollout.

    Reset-dependent phase/path/orientation references and the field manifest are
    explicit inputs.  Missing references fail closed; this function never falls
    back to an episode-initial quaternion or a start/end chord.
    """

    task_id = _safe_id("Qv4 rollout task_id", record.get("task_id"))
    episode_id = _safe_id("Qv4 rollout episode_id", record.get("episode_id"))
    if (
        reference_contract.get("schema_version") != QUALITY_V4_ROLLOUT_REFERENCE_SCHEMA
        or reference_contract.get("task_id") != task_id
        or reference_contract.get("episode_id") != episode_id
    ):
        raise ValueError("Qv4 rollout reference identity mismatch")
    reference_unsigned = dict(reference_contract)
    reference_sha256 = reference_unsigned.pop("reference_sha256", None)
    _sha256_value("Qv4 rollout reference_sha256", reference_sha256)
    if reference_sha256 != _stable_sha256(reference_unsigned):
        raise ValueError("Qv4 rollout reference SHA-256 mismatch")
    field_contract = reference_contract.get("field_contract")
    if (
        not isinstance(field_contract, Mapping)
        or field_contract.get("task_id") != task_id
    ):
        raise ValueError("Qv4 rollout reference has no task-bound field contract")
    if isinstance(observations, (str, bytes)) or len(observations) < 2:
        raise ValueError("Qv4 rollout requires at least two observations")
    issued = np.asarray(issued_actions, dtype=np.float64)
    action_count = len(observations) - 1
    if (
        issued.ndim != 2
        or issued.shape[0] != action_count
        or issued.shape[1] < 1
        or not np.all(np.isfinite(issued))
    ):
        raise ValueError("Qv4 rollout issued action tape does not align")
    issue_time = np.asarray(
        [float(observation.time_s) for observation in observations[:-1]],
        dtype=np.float64,
    )
    applied, applied_time, applied_source_step, t5_action_history = (
        _quality_v4_applied_history(task_id, raw_env, issued, issue_time)
    )
    reward = np.asarray(rewards, dtype=np.float64)
    if reward.shape != (action_count,) or len(outcomes) != action_count:
        raise ValueError("Qv4 rollout result tape does not align")
    progress = np.asarray([float(row[4]) for row in outcomes], dtype=np.float64)

    normalized_physics: list[Any] = []
    previous_time = -math.inf
    for sample in physics_samples:
        time_value = _sample_value(sample, "time_s")
        qpos_value = _sample_value(sample, "simulator_qpos")
        qvel_value = _sample_value(sample, "simulator_qvel")
        if time_value is None or qpos_value is None or qvel_value is None:
            raise ValueError("Qv4 physics source sample is missing time/qpos/qvel")
        time_s = float(time_value)
        if not math.isfinite(time_s):
            raise ValueError("Qv4 physics source tape contains a non-finite time")
        if math.isclose(time_s, previous_time, abs_tol=1.0e-12, rel_tol=0.0):
            continue
        if time_s < previous_time:
            raise ValueError("Qv4 physics source tape is not monotonic")
        normalized_physics.append(sample)
        previous_time = time_s
    if len(normalized_physics) < 4:
        raise ValueError("Qv4 physics source tape is too short for jerk recomputation")
    qpos = np.stack(
        [
            np.asarray(_sample_value(sample, "simulator_qpos"), dtype=np.float64)
            for sample in normalized_physics
        ]
    )
    qvel = np.stack(
        [
            np.asarray(_sample_value(sample, "simulator_qvel"), dtype=np.float64)
            for sample in normalized_physics
        ]
    )
    physics_eef = np.stack([_physics_eef_pose(sample) for sample in normalized_physics])
    physics_time = np.asarray(
        [float(_sample_value(sample, "time_s")) for sample in normalized_physics],
        dtype=np.float64,
    )
    bounds = field_contract.get("bounds")
    contact_bounds = (
        bounds.get("contact_impulse_max_by_name")
        if isinstance(bounds, Mapping)
        else None
    )
    if not isinstance(contact_bounds, Mapping) or not contact_bounds:
        raise ValueError("Qv4 field contract has no named contact bounds")
    contact_names = tuple(str(name) for name in contact_bounds)
    contact_rows: list[list[float]] = []
    for sample_index, sample in enumerate(normalized_physics):
        row: list[float] = []
        for name in contact_names:
            value = _sample_value(sample, name)
            if value is None:
                if sample_index == 0:
                    value = 0.0
                else:
                    raise ValueError(
                        f"Qv4 physics sample has no registered contact field {name!r}"
                    )
            row.append(float(value))
        contact_rows.append(row)
    contact = np.asarray(contact_rows, dtype=np.float64)

    eef = np.stack(
        [
            np.asarray(observation.privileged["eef_pose_xyzw"], dtype=np.float64)
            for observation in observations
        ]
    )
    obj = np.stack(
        [
            np.asarray(observation.privileged["object_pose_wxyz"], dtype=np.float64)
            for observation in observations
        ]
    )
    twist = np.stack(
        [
            np.asarray(observation.privileged["object_twist_world"], dtype=np.float64)
            for observation in observations
        ]
    )
    closing = np.stack(
        [
            np.asarray(
                observation.privileged["fingerpad_closing_axis_world"],
                dtype=np.float64,
            )
            for observation in observations
        ]
    )
    rgb = _stack_observation_group(observations, "rgb", dtype=np.dtype(np.uint8))
    depth = _stack_observation_group(
        observations, "depth_m", dtype=np.dtype(np.float32)
    )
    segmentation = _stack_observation_group(
        observations, "segmentation", dtype=np.dtype(np.int32)
    )
    depth_valid_mask = np.isfinite(depth) & (depth > 0.0)
    history_valid_mask = np.asarray(reference_contract.get("history_valid_mask"))
    if history_valid_mask.ndim != 2 or history_valid_mask.shape[0] != len(observations):
        raise ValueError("Qv4 rollout history-valid mask is missing or misaligned")
    history_valid_mask = history_valid_mask.astype(np.bool_, copy=False)
    observation_time = np.asarray(
        [float(observation.time_s) for observation in observations],
        dtype=np.float64,
    )
    policy_steps = np.asarray(
        [int(observation.policy_step) for observation in observations],
        dtype=np.int64,
    )
    final_outcome = outcomes[-1]
    raw_safety_failures = record.get("reward_schema_safety_failures")
    if isinstance(raw_safety_failures, (str, bytes)) or not isinstance(
        raw_safety_failures, Sequence
    ):
        raise ValueError("Qv4 rollout has no reward-schema safety-failure inventory")
    safety_failures = sorted({str(name) for name in raw_safety_failures})
    if not safety_failures or any(not name for name in safety_failures):
        raise ValueError("Qv4 reward-schema safety-failure inventory is invalid")
    source = {
        "episode_id": episode_id,
        "task_id": task_id,
        "reset_pair_key": _safe_id(
            "Qv4 rollout reset_pair_key",
            reference_contract.get("reset_pair_key", episode_id),
        ),
        "rollout_reference_sha256": reference_sha256,
        "field_contract": dict(field_contract),
        "field_tape": {
            "simulator": {"qpos": qpos, "qvel": qvel},
            "observation": {
                "eef_pose_xyzw": eef,
                "object_pose_wxyz": obj,
                "object_twist_world": twist,
                "fingerpad_closing_axis_world": closing,
                "rgb": rgb,
                "depth_m": depth,
                "depth_valid_mask": depth_valid_mask,
                "segmentation": segmentation,
            },
            "action": {"issued": issued, "applied": applied},
            "physics": {
                "eef_pose_xyzw": physics_eef,
                "contact_impulse_n_s": contact,
                "contact_names": list(contact_names),
            },
            "result": {"reward": reward, "progress": progress},
            "mask": {"history_valid_mask": history_valid_mask},
            "clock": {
                "physics_time_s": physics_time,
                "observation_time_s": observation_time,
                "action_issue_time_s": issue_time,
                "action_applied_time_s": applied_time,
                "action_applied_source_policy_step": applied_source_step,
                "observation_policy_step": policy_steps,
            },
        },
        "events": list(events),
        "t5_action_history": t5_action_history,
        "safety_contract": {
            "source": "rollout_reward_schema.safety_failures",
            "safety_failures": safety_failures,
        },
        "final": {
            "success": bool(final_outcome[2]),
            "termination_reason": final_outcome[3],
            "active_stage_progress": float(final_outcome[4]),
        },
        "replay_validation": dict(replay_validation),
        "summary_inputs": {
            "issued_actions": issued,
            "applied_actions": applied,
            "control_eef_pose_xyzw": eef,
            "control_object_pose_wxyz": obj,
            "closing_axis_world": closing,
            "progress": np.concatenate(([0.0], progress)),
            "phase_slices": reference_contract.get("phase_slices"),
            "path_references": reference_contract.get("path_references"),
            "orientation_references": reference_contract.get("orientation_references"),
            "continuous_dimensions": max(1, issued.shape[1] - 1),
            "reversal_deadband": reference_contract.get("reversal_deadband", 0.02),
        },
        "thresholds": dict(thresholds),
        "return_diagnostic": record.get("return"),
    }
    return source


def validate_quality_v4_thresholds(
    payload: Mapping[str, Any],
    *,
    expected_thresholds_sha256: str | None = None,
    require_formal_freeze: bool = False,
) -> dict[str, Any]:
    """Validate coverage, provenance, split isolation, and freeze status."""

    quality = _se3_quality()
    if payload.get("schema_version") != quality.QUALITY_V4_THRESHOLDS_SCHEMA:
        raise ValueError("Qv4 threshold schema mismatch")
    unsigned = dict(payload)
    recorded_sha256 = unsigned.pop("thresholds_sha256", None)
    _sha256_value("Qv4 thresholds_sha256", recorded_sha256)
    if recorded_sha256 != _stable_sha256(unsigned):
        raise ValueError("Qv4 threshold canonical SHA-256 mismatch")
    if expected_thresholds_sha256 is not None and recorded_sha256 != _sha256_value(
        "expected Qv4 thresholds_sha256", expected_thresholds_sha256
    ):
        raise ValueError("Qv4 threshold identity differs from the expected SHA-256")

    orientation = quality.orientation_contract_manifest()
    if payload.get("orientation_contract_sha256") != orientation.get("contract_sha256"):
        raise ValueError("Qv4 thresholds bind a different exact-14 orientation table")
    sources = payload.get("calibration_sources")
    if not isinstance(sources, Mapping) or set(sources) != _CALIBRATION_SOURCE_KEYS:
        raise ValueError("Qv4 calibration source inventory mismatch")
    planner_source = sources["exact14x20_planner"]
    if not isinstance(planner_source, Mapping) or (
        planner_source.get("split") != "metric_calibration"
        or planner_source.get("task_count") != 14
        or planner_source.get("episodes_per_task") != 20
        or planner_source.get("total_reset_count") != 280
    ):
        raise ValueError("Qv4 planner calibration source is not exact14x20")
    splits_read = payload.get("splits_read")
    if isinstance(splits_read, (str, bytes)) or not isinstance(splits_read, Sequence):
        raise ValueError("Qv4 thresholds must declare splits_read")
    normalized_splits = {str(value) for value in splits_read}
    if normalized_splits & _FORBIDDEN_TUNING_SPLITS:
        raise ValueError("Qv4 thresholds used a test split for tuning")
    allowed_tuning_splits = payload.get("allowed_tuning_splits")
    if allowed_tuning_splits != ["metric_calibration"]:
        raise ValueError("Qv4 thresholds must isolate tuning to metric_calibration")
    if (
        payload.get("calibration_status") == "frozen"
        and "metric_calibration" not in normalized_splits
    ):
        raise ValueError("frozen Qv4 thresholds did not use metric_calibration")

    tasks = payload.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(
        quality.EXACT14_ORIENTATION_CONTRACT
    ):
        raise ValueError("Qv4 threshold task inventory is not exact-14")
    task_check_count = 0
    all_checks_numeric = True
    all_check_statuses_frozen = True
    all_check_evidence_bound = True
    all_check_values_evidence_bound = True
    all_vision_tolerances_frozen = True
    all_vision_evidence_bound = True
    vision_segmentation_contract_sha256s: set[str] = set()
    expected_check_inventory = quality_v4_threshold_check_inventory()
    for task_id, orientation_table in quality.EXACT14_ORIENTATION_CONTRACT.items():
        task_contract = tasks[task_id]
        if not isinstance(task_contract, Mapping):
            raise ValueError(f"Qv4 threshold task {task_id} is not a mapping")
        vision_tolerance = task_contract.get("vision_tolerance")
        if not isinstance(vision_tolerance, Mapping):
            raise ValueError(f"Qv4 threshold task {task_id} has no vision tolerance")
        vision_unsigned = dict(vision_tolerance)
        vision_sha256 = vision_unsigned.pop("tolerance_sha256", None)
        _sha256_value(f"Qv4 {task_id} vision tolerance SHA-256", vision_sha256)
        if (
            vision_tolerance.get("schema_version") != QUALITY_V4_VISION_TOLERANCE_SCHEMA
            or vision_tolerance.get("task_id") != task_id
            or vision_sha256 != _stable_sha256(vision_unsigned)
        ):
            raise ValueError(f"Qv4 threshold task {task_id} vision tolerance mismatch")
        all_vision_tolerances_frozen = bool(
            all_vision_tolerances_frozen
            and vision_tolerance.get("calibration_status") == "frozen"
        )
        all_vision_evidence_bound = bool(
            all_vision_evidence_bound
            and isinstance(vision_tolerance.get("evidence_sha256"), str)
            and _SHA256.fullmatch(str(vision_tolerance["evidence_sha256"])) is not None
        )
        vision_segmentation_contract_sha256 = vision_tolerance.get(
            "segmentation_contract_sha256"
        )
        if isinstance(vision_segmentation_contract_sha256, str):
            vision_segmentation_contract_sha256s.add(
                vision_segmentation_contract_sha256
            )
        for name in (
            "rgb_max_abs_lsb",
            "rgb_max_changed_fraction_per_frame",
            "depth_abs_tolerance_m",
            "depth_relative_tolerance",
        ):
            value = vision_tolerance.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"Qv4 threshold task {task_id} {name} is invalid")
        if vision_tolerance.get("segmentation_exact") is not True:
            raise ValueError(f"Qv4 threshold task {task_id} segmentation must be exact")
        checks = task_contract.get("checks")
        if isinstance(checks, (str, bytes)) or not isinstance(checks, Sequence):
            raise ValueError(f"Qv4 threshold task {task_id} has no checks")
        covered_phases: set[str] = set()
        identities: set[tuple[str, str]] = set()
        for index, check in enumerate(checks):
            if not isinstance(check, Mapping):
                raise ValueError(f"Qv4 threshold {task_id} check {index} is invalid")
            phase = check.get("phase")
            metric = check.get("metric")
            direction = check.get("direction")
            if not isinstance(phase, str) or not isinstance(metric, str):
                raise ValueError("Qv4 threshold phase/metric must be strings")
            if direction not in {"minimize", "maximize"}:
                raise ValueError("Qv4 threshold direction is invalid")
            identity = (phase, metric)
            if identity in identities:
                raise ValueError(f"duplicate Qv4 threshold check {task_id}:{identity}")
            identities.add(identity)
            covered_phases.add(phase)
            bound = check.get("max" if direction == "minimize" else "min")
            margin = check.get("strict_improvement_margin")
            tolerance = check.get("paired_non_worse_tolerance", 0.0)
            all_checks_numeric = bool(
                all_checks_numeric
                and bound is not None
                and margin is not None
                and tolerance is not None
            )
            all_check_statuses_frozen = bool(
                all_check_statuses_frozen
                and check.get("calibration_status") == "frozen"
            )
            all_check_evidence_bound = bool(
                all_check_evidence_bound
                and isinstance(check.get("evidence_sha256"), str)
                and _SHA256.fullmatch(str(check["evidence_sha256"])) is not None
            )
            value_evidence = check.get("value_evidence")
            check_values_evidence_bound = bool(
                isinstance(value_evidence, Mapping)
                and set(value_evidence) == _QUALITY_V4_CHECK_VALUE_EVIDENCE_KEYS
                and all(
                    isinstance(value, str) and _SHA256.fullmatch(value) is not None
                    for value in value_evidence.values()
                )
                and check.get("evidence_sha256") == _stable_sha256(value_evidence)
            )
            all_check_values_evidence_bound = bool(
                all_check_values_evidence_bound and check_values_evidence_bound
            )
            for label, value in (
                ("bound", bound),
                ("strict_improvement_margin", margin),
                ("paired_non_worse_tolerance", tolerance),
            ):
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(
                        f"Qv4 threshold {task_id}:{phase}:{metric} {label} is invalid"
                    )
        required_phases = {
            phase
            for phase, row in orientation_table.items()
            if row.get("applicable") is True
        }
        missing = sorted(required_phases - covered_phases)
        if missing:
            raise ValueError(
                f"Qv4 threshold task {task_id} lacks phase coverage: "
                + ", ".join(missing)
            )
        if "full_episode" not in covered_phases:
            raise ValueError(f"Qv4 threshold task {task_id} lacks full-episode checks")
        expected_identities = set(expected_check_inventory[task_id])
        if identities != expected_identities:
            missing_identities = sorted(expected_identities - identities)
            unexpected_identities = sorted(identities - expected_identities)
            raise ValueError(
                f"Qv4 threshold task {task_id} metric inventory mismatch: "
                f"missing={missing_identities!r}, unexpected={unexpected_identities!r}"
            )
        task_check_count += len(checks)

    segmentation_contract = payload.get("segmentation_contract")
    segmentation_contract_frozen = False
    segmentation_contract_sha256: str | None = None
    if isinstance(segmentation_contract, Mapping):
        segmentation_unsigned = dict(segmentation_contract)
        segmentation_contract_sha256 = segmentation_unsigned.pop(
            "contract_sha256", None
        )
        _sha256_value(
            "Qv4 exact segmentation contract SHA-256",
            segmentation_contract_sha256,
        )
        segmentation_contract_frozen = bool(
            segmentation_contract.get("schema_version")
            == QUALITY_V4_SEGMENTATION_CONTRACT_SCHEMA
            and segmentation_contract.get("calibration_status") == "frozen"
            and segmentation_contract.get("comparison") == "component_sha256_exact"
            and segmentation_contract.get("exact") is True
            and segmentation_contract.get("shape_exact") is True
            and segmentation_contract.get("dtype_exact") is True
            and segmentation_contract.get("label_remap_allowed") is False
            and segmentation_contract.get("task_ids")
            == list(quality.EXACT14_ORIENTATION_CONTRACT)
            and isinstance(segmentation_contract.get("evidence_sha256"), str)
            and _SHA256.fullmatch(str(segmentation_contract["evidence_sha256"]))
            is not None
            and segmentation_contract_sha256 == _stable_sha256(segmentation_unsigned)
            and vision_segmentation_contract_sha256s == {segmentation_contract_sha256}
        )

    owner_review = payload.get("owner_review")
    if not isinstance(owner_review, Mapping):
        raise ValueError("Qv4 thresholds have no owner-review record")
    source_evidence_complete = all(
        isinstance(source, Mapping)
        and source.get("status") == "complete"
        and isinstance(source.get("evidence_sha256"), str)
        and _SHA256.fullmatch(str(source["evidence_sha256"])) is not None
        for source in sources.values()
    )
    owner_review_complete = bool(
        owner_review.get("approved") is True
        and all(
            isinstance(owner_review.get(name), str)
            and bool(str(owner_review[name]).strip())
            for name in ("reviewer", "reviewed_at", "decision_record")
        )
    )
    owner_review_pending = bool(
        owner_review.get("approved") is False
        and all(
            owner_review.get(name) is None
            for name in ("reviewer", "reviewed_at", "decision_record")
        )
    )
    numeric_evidence_candidate = bool(
        payload.get("calibration_status") in {"formal_candidate", "frozen"}
        and "metric_calibration" in normalized_splits
        and source_evidence_complete
        and all_checks_numeric
        and all_check_statuses_frozen
        and all_check_evidence_bound
        and all_check_values_evidence_bound
        and all_vision_tolerances_frozen
        and all_vision_evidence_bound
        and segmentation_contract_frozen
    )
    formal_candidate = bool(
        numeric_evidence_candidate
        and payload.get("calibration_status") == "formal_candidate"
        and payload.get("formal_freeze_eligible") is False
        and owner_review_pending
    )
    frozen = bool(
        numeric_evidence_candidate
        and payload.get("calibration_status") == "frozen"
        and payload.get("formal_freeze_eligible") is True
        and owner_review_complete
    )
    if (
        payload.get("calibration_status") == "provisional"
        and payload.get("formal_freeze_eligible") is not False
    ):
        raise ValueError("provisional Qv4 thresholds cannot be formal-freeze eligible")
    if payload.get("calibration_status") == "formal_candidate" and not formal_candidate:
        raise ValueError(
            "Qv4 thresholds claim a formal candidate without numeric value-level "
            "evidence, frozen vision/segmentation contracts, complete calibration "
            "receipts, split isolation, or a pending owner review"
        )
    if (
        payload.get("calibration_status") == "frozen"
        or payload.get("formal_freeze_eligible") is True
    ) and not frozen:
        raise ValueError(
            "Qv4 thresholds claim formal freeze without numeric checks, frozen vision "
            "tolerances, per-check evidence, evidence receipts, split isolation, and "
            "owner review"
        )
    if require_formal_freeze and not frozen:
        raise ValueError("Qv4 thresholds are not owner-reviewed and formally frozen")
    return {
        "schema_version": payload["schema_version"],
        "thresholds_sha256": recorded_sha256,
        "orientation_contract_sha256": orientation["contract_sha256"],
        "task_count": 14,
        "check_count": task_check_count,
        "formal_candidate_validated": bool(formal_candidate or frozen),
        "formal_freeze_eligible": frozen,
        "owner_review_complete": owner_review_complete,
        "calibration_status": payload.get("calibration_status"),
        "test_splits_read": sorted(normalized_splits & _FORBIDDEN_TUNING_SPLITS),
    }


def validate_quality_v4_threshold_candidate(
    payload: Mapping[str, Any],
    *,
    expected_thresholds_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a numeric evidence-bound candidate without approving release.

    This validation is deliberately separate from ``require_formal_freeze=True``.
    A candidate may pass here with a pending Owner review, while every production
    export remains blocked until the same artifact is Owner-approved, marked
    ``calibration_status=frozen``, and passes the formal-freeze validator.
    """

    validation = validate_quality_v4_thresholds(
        payload,
        expected_thresholds_sha256=expected_thresholds_sha256,
    )
    if not validation["formal_candidate_validated"]:
        raise ValueError("Qv4 thresholds are not a numeric evidence-bound candidate")
    return validation


def load_quality_v4_thresholds(
    path: Path,
    *,
    expected_file_sha256: str,
    require_formal_freeze: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a regular threshold artifact and bind both file and canonical hashes."""

    raw_path = Path(path)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    actual_file_sha256 = _file_sha256(raw_path)
    if actual_file_sha256 != _sha256_value(
        "expected Qv4 threshold file SHA-256", expected_file_sha256
    ):
        raise ValueError("Qv4 threshold file SHA-256 mismatch")
    loaded = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("Qv4 threshold artifact must be a JSON object")
    thresholds = dict(loaded)
    validation = validate_quality_v4_thresholds(
        thresholds,
        require_formal_freeze=require_formal_freeze,
    )
    validation["file_sha256"] = actual_file_sha256
    return thresholds, validation


def _validate_attempt(attempt: Mapping[str, Any]) -> None:
    if attempt.get("schema_version") != QUALITY_V4_ATTEMPT_SCHEMA:
        raise ValueError("Qv4 attempt schema mismatch")
    unsigned = dict(attempt)
    recorded = unsigned.pop("attempt_sha256", None)
    _sha256_value("Qv4 attempt_sha256", recorded)
    if recorded != _stable_sha256(unsigned):
        raise ValueError("Qv4 attempt SHA-256 mismatch")


def build_quality_v4_attempt(source: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute one candidate's Qv4 summary and two independent gates."""

    quality = _se3_quality()
    episode_id = _safe_id("Qv4 episode_id", source.get("episode_id"))
    task_id = _safe_id("Qv4 task_id", source.get("task_id"))
    reset_pair_key = _safe_id(
        "Qv4 reset_pair_key", source.get("reset_pair_key", episode_id)
    )
    field_tape = source.get("field_tape")
    field_contract = source.get("field_contract")
    summary_inputs = source.get("summary_inputs")
    final = source.get("final")
    safety_contract = source.get("safety_contract")
    replay_validation = source.get("replay_validation")
    thresholds = source.get("thresholds")
    for label, value in (
        ("field_tape", field_tape),
        ("field_contract", field_contract),
        ("summary_inputs", summary_inputs),
        ("final", final),
        ("safety_contract", safety_contract),
        ("replay_validation", replay_validation),
        ("thresholds", thresholds),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"Qv4 source {label} must be a mapping")
    events = _event_rows(source.get("events"))
    if safety_contract.get("source") != "rollout_reward_schema.safety_failures":
        raise ValueError("Qv4 source safety contract is not reward-schema bound")
    threshold_validation = validate_quality_v4_thresholds(thresholds)
    task_thresholds = thresholds["tasks"][task_id]
    vision_tolerance = task_thresholds["vision_tolerance"]
    if replay_validation.get("vision_tolerance_sha256") != vision_tolerance.get(
        "tolerance_sha256"
    ):
        raise ValueError("Qv4 replay used a different task-specific vision tolerance")
    action_fields = field_tape.get("action")
    observation_fields = field_tape.get("observation")
    result_fields = field_tape.get("result")
    clock_fields = field_tape.get("clock")
    if not all(
        isinstance(value, Mapping)
        for value in (action_fields, observation_fields, result_fields, clock_fields)
    ):
        raise ValueError(
            "Qv4 field tape lacks action/observation/result/clock mappings"
        )
    if not {"issued", "applied"}.issubset(action_fields):
        raise ValueError("Qv4 action field tape lacks issued/applied actions")
    required_action_clocks = {
        "action_issue_time_s",
        "action_applied_time_s",
        "action_applied_source_policy_step",
    }
    if not required_action_clocks.issubset(clock_fields):
        raise ValueError("Qv4 action field tape lacks issued/applied clocks")
    field_issued = np.asarray(action_fields.get("issued"), dtype=np.float64)
    field_applied = np.asarray(action_fields.get("applied"), dtype=np.float64)
    field_issue_time = np.asarray(
        clock_fields.get("action_issue_time_s"), dtype=np.float64
    )
    field_applied_time = np.asarray(
        clock_fields.get("action_applied_time_s"), dtype=np.float64
    )
    field_applied_source_step = np.asarray(
        clock_fields.get("action_applied_source_policy_step")
    )
    if task_id == "t5_replan":
        (
            recomputed_applied,
            recomputed_applied_time,
            recomputed_source_step,
            normalized_history,
        ) = _normalize_quality_v4_t5_action_history(
            source.get("t5_action_history"),
            issued=field_issued,
            issue_time_s=field_issue_time,
        )
        if (
            source.get("t5_action_history") != normalized_history
            or not np.array_equal(field_applied, recomputed_applied)
            or not np.array_equal(field_applied_time, recomputed_applied_time)
            or not np.array_equal(field_applied_source_step, recomputed_source_step)
        ):
            raise ValueError("Qv4 T5 field tape is not derived from its queue ledger")
    elif source.get("t5_action_history") is not None:
        raise ValueError("non-T5 Qv4 source cannot contain a T5 action history")

    issued = np.asarray(summary_inputs.get("issued_actions"), dtype=np.float64)
    applied = np.asarray(summary_inputs.get("applied_actions"), dtype=np.float64)
    expected_progress = np.concatenate(
        ([0.0], np.asarray(result_fields.get("progress"), dtype=np.float64))
    )
    summary_bindings = (
        (issued, field_issued, "issued actions"),
        (applied, field_applied, "applied actions"),
        (
            np.asarray(summary_inputs.get("control_eef_pose_xyzw")),
            np.asarray(observation_fields.get("eef_pose_xyzw")),
            "control-rate EEF pose",
        ),
        (
            np.asarray(summary_inputs.get("control_object_pose_wxyz")),
            np.asarray(observation_fields.get("object_pose_wxyz")),
            "control-rate object pose",
        ),
        (
            np.asarray(summary_inputs.get("closing_axis_world")),
            np.asarray(observation_fields.get("fingerpad_closing_axis_world")),
            "fingerpad closing axis",
        ),
        (
            np.asarray(summary_inputs.get("progress"), dtype=np.float64),
            expected_progress,
            "progress",
        ),
    )
    for recorded, raw, label in summary_bindings:
        if not np.array_equal(recorded, raw):
            raise ValueError(f"Qv4 summary {label} differs from the raw field tape")
    field_validation = quality.validate_trajectory_field_tape(
        field_tape, field_contract
    )
    physics = field_tape.get("physics")
    clock = field_tape.get("clock")
    if not isinstance(physics, Mapping) or not isinstance(clock, Mapping):
        raise ValueError("Qv4 field tape has no physics/clock source mappings")
    required_physics = {
        "eef_pose_xyzw",
        "contact_impulse_n_s",
        "contact_names",
    }
    if not required_physics.issubset(physics) or "physics_time_s" not in clock:
        raise ValueError("Qv4 field tape lacks physics/contact source fields")
    physics_reducer = quality.PhysicsRateEEFReducer()
    physics_reducer.update_many(
        clock.get("physics_time_s"),
        physics.get("eef_pose_xyzw"),
    )
    recomputed_physics_rate_eef = physics_reducer.summary()
    recorded_physics_rate_eef = summary_inputs.get("physics_rate_eef")
    if (
        recorded_physics_rate_eef is not None
        and recorded_physics_rate_eef != recomputed_physics_rate_eef
    ):
        raise ValueError("Qv4 recorded physics-rate reducer summary mismatch")
    summary = quality.trajectory_quality_v4(
        task_id=task_id,
        issued_actions=issued,
        applied_actions=applied,
        control_eef_pose_xyzw=summary_inputs.get("control_eef_pose_xyzw"),
        control_object_pose_wxyz=summary_inputs.get("control_object_pose_wxyz"),
        closing_axis_world=summary_inputs.get("closing_axis_world"),
        progress=summary_inputs.get("progress"),
        phase_slices=summary_inputs.get("phase_slices"),
        path_references=summary_inputs.get("path_references"),
        orientation_references=summary_inputs.get("orientation_references"),
        physics_rate_eef=recomputed_physics_rate_eef,
        continuous_dimensions=int(summary_inputs.get("continuous_dimensions")),
        reversal_deadband=float(summary_inputs.get("reversal_deadband", 0.02)),
    )
    if not isinstance(final.get("success"), bool):
        raise ValueError("Qv4 final success must be boolean")
    active_stage_progress = final.get("active_stage_progress")
    if isinstance(active_stage_progress, bool) or not isinstance(
        active_stage_progress, (int, float)
    ):
        raise ValueError("Qv4 final active_stage_progress must be numeric")
    layer1 = quality.evaluate_quality_v4_layer1(
        task_id=task_id,
        final_success=final["success"],
        termination_reason=final.get("termination_reason"),
        active_stage_progress=float(active_stage_progress),
        events=events,
        safety_failures=safety_contract.get("safety_failures"),
        field_validation=field_validation,
        replay_validation=replay_validation,
        issued_actions=issued,
        applied_actions=applied,
    )
    layer2 = quality.evaluate_quality_v4_layer2(
        summary,
        thresholds,
        task_id=task_id,
    )
    source_manifest = quality_v4_source_manifest(source)
    result: dict[str, Any] = {
        "schema_version": QUALITY_V4_ATTEMPT_SCHEMA,
        "episode_id": episode_id,
        "task_id": task_id,
        "reset_pair_key": reset_pair_key,
        "source_sha256": source_manifest["source_sha256"],
        "field_contract_sha256": field_contract.get("contract_sha256"),
        "orientation_contract_sha256": summary.get("orientation_contract_sha256"),
        "thresholds_sha256": threshold_validation["thresholds_sha256"],
        "summary": summary,
        "layer1_gate": layer1,
        "layer2_gate": layer2,
        "eligible": bool(layer1["passed"] and layer2["passed"]),
        "formal_thresholds_frozen": threshold_validation["formal_freeze_eligible"],
        "return_diagnostic": source.get("return_diagnostic"),
    }
    result["attempt_sha256"] = _stable_sha256(result)
    return result


def finalize_quality_v4_fresh_replay(
    *,
    original_source: Mapping[str, Any],
    replayed_source: Mapping[str, Any],
    base_replay_validation: Mapping[str, Any],
    observation_comparison: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive Qv4 replay flags by independently rebuilding both source tapes."""

    receipt: dict[str, Any] = {
        "structured_observations_exact": (
            observation_comparison.get("structured_observations_exact") is True
        ),
        "event_ledger_exact": (
            observation_comparison.get("event_ledger_exact") is True
        ),
        "outcomes_exact": base_replay_validation.get("outcomes_exact") is True,
        "final_state_exact": base_replay_validation.get("final_state_exact") is True,
        "terminal_task_quality_exact": (
            base_replay_validation.get("task_quality_exact") is True
            or base_replay_validation.get("terminal_task_quality_exact") is True
        ),
        "quality_v4_summary_exact": False,
        "quality_v4_layer1_exact": False,
        "quality_v4_layer2_exact": False,
        "rgb_within_tolerance": (
            observation_comparison.get("rgb_within_tolerance") is True
        ),
        "depth_within_tolerance": (
            observation_comparison.get("depth_within_tolerance") is True
        ),
        "vision_tolerance_sha256": observation_comparison.get(
            "vision_tolerance_sha256"
        ),
    }
    original_pre = copy.deepcopy(dict(original_source))
    replayed_pre = copy.deepcopy(dict(replayed_source))
    original_pre["replay_validation"] = dict(receipt)
    replayed_pre["replay_validation"] = dict(receipt)
    original_pre_attempt = build_quality_v4_attempt(original_pre)
    replayed_pre_attempt = build_quality_v4_attempt(replayed_pre)
    receipt["quality_v4_summary_exact"] = (
        original_pre_attempt["summary"] == replayed_pre_attempt["summary"]
    )
    receipt["quality_v4_layer1_exact"] = (
        original_pre_attempt["layer1_gate"] == replayed_pre_attempt["layer1_gate"]
    )
    receipt["quality_v4_layer2_exact"] = (
        original_pre_attempt["layer2_gate"] == replayed_pre_attempt["layer2_gate"]
    )
    original_final = copy.deepcopy(dict(original_source))
    replayed_final = copy.deepcopy(dict(replayed_source))
    original_final["replay_validation"] = dict(receipt)
    replayed_final["replay_validation"] = dict(receipt)
    original_attempt = build_quality_v4_attempt(original_final)
    replayed_attempt = build_quality_v4_attempt(replayed_final)
    if all(
        receipt[name]
        for name in (
            "quality_v4_summary_exact",
            "quality_v4_layer1_exact",
            "quality_v4_layer2_exact",
        )
    ):
        quality = _se3_quality()
        quality.assert_quality_v4_replay_equal(
            original_attempt["summary"],
            replayed_attempt["summary"],
            original_attempt["layer1_gate"],
            replayed_attempt["layer1_gate"],
            original_attempt["layer2_gate"],
            replayed_attempt["layer2_gate"],
        )
    original_final["replay_validation"] = dict(receipt)
    return original_final, original_attempt


def write_quality_v4_attempt(output_root: Path, attempt: Mapping[str, Any]) -> Path:
    """Publish one lightweight attempt JSON under the new Qv4 artifact path."""

    _validate_attempt(attempt)
    episode_id = _safe_id("Qv4 attempt episode_id", attempt.get("episode_id"))
    directory = Path(output_root) / QUALITY_V4_ATTEMPT_SUBDIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{episode_id}.json"
    if destination.exists():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{episode_id}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(dict(attempt), indent=2, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _vision_array_descriptor(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype.kind in {"O", "U"}:
        raise ValueError("Qv4 vision source has an unsupported dtype")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _quality_v4_lightweight_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the independently useful non-vision tape and hash omitted pixels."""

    field_tape = source.get("field_tape")
    if not isinstance(field_tape, Mapping):
        raise ValueError("Qv4 source has no field tape")
    observation = field_tape.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("Qv4 source has no observation field tape")
    vision_names = ("rgb", "depth_m", "depth_valid_mask", "segmentation")
    if any(name not in observation for name in vision_names):
        raise ValueError("Qv4 source has an incomplete vision field tape")
    nonvision_observation = {
        str(name): value
        for name, value in observation.items()
        if name not in vision_names
    }
    nonvision_field_tape = {
        str(name): value for name, value in field_tape.items() if name != "observation"
    }
    nonvision_field_tape["observation"] = nonvision_observation
    thresholds = source.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("Qv4 source has no threshold contract")
    payload = {
        "episode_id": source.get("episode_id"),
        "task_id": source.get("task_id"),
        "reset_pair_key": source.get("reset_pair_key"),
        "rollout_reference_sha256": source.get("rollout_reference_sha256"),
        "field_contract": source.get("field_contract"),
        "field_tape": nonvision_field_tape,
        "events": source.get("events"),
        "t5_action_history": source.get("t5_action_history"),
        "safety_contract": source.get("safety_contract"),
        "final": source.get("final"),
        "replay_validation": source.get("replay_validation"),
        "summary_inputs": source.get("summary_inputs"),
        "thresholds_sha256": thresholds.get("thresholds_sha256"),
        "return_diagnostic": source.get("return_diagnostic"),
        "omitted_vision_arrays": {
            name: _vision_array_descriptor(observation[name]) for name in vision_names
        },
    }
    return payload


def write_quality_v4_lightweight_source(
    output_root: Path,
    *,
    source: Mapping[str, Any],
    recorded_attempt: Mapping[str, Any],
) -> Path:
    """Write one non-vision candidate tape; raw RGB-D stays winner-only."""

    _validate_attempt(recorded_attempt)
    episode_id = _safe_id(
        "Qv4 lightweight episode_id", recorded_attempt.get("episode_id")
    )
    source_manifest = quality_v4_source_manifest(source)
    if source_manifest["source_sha256"] != recorded_attempt.get("source_sha256"):
        raise ValueError("Qv4 lightweight source does not match its attempt")
    lightweight = _quality_v4_lightweight_payload(source)
    arrays: dict[str, np.ndarray] = {}
    tree = _encode_source_tree(lightweight, arrays)
    destination = (
        Path(output_root) / QUALITY_V4_LIGHTWEIGHT_SUBDIRECTORY / f"{episode_id}.h5"
    )
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = QUALITY_V4_LIGHTWEIGHT_SOURCE_SCHEMA
            handle.attrs["source_sha256"] = source_manifest["source_sha256"]
            handle.attrs["attempt_sha256"] = recorded_attempt["attempt_sha256"]
            handle.attrs["lightweight_tree_json"] = _canonical_json(tree)
            handle.attrs["source_manifest_json"] = _canonical_json(source_manifest)
            group = handle.create_group("arrays")
            for key, array in sorted(arrays.items()):
                compressible = bool(array.size and array.ndim > 0)
                group.create_dataset(
                    key,
                    data=array,
                    compression="gzip" if compressible else None,
                    compression_opts=4 if compressible else None,
                    shuffle=bool(compressible and array.dtype.kind not in {"S"}),
                )
            handle.flush()
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def audit_quality_v4_lightweight_source(path: Path) -> dict[str, Any]:
    """Verify the candidate tape and prove that it contains no raw vision arrays."""

    source_path = Path(path)
    if source_path.is_symlink() or not source_path.is_file():
        raise FileNotFoundError(source_path)
    with h5py.File(source_path, "r") as handle:
        if handle.attrs.get("schema_version") != QUALITY_V4_LIGHTWEIGHT_SOURCE_SCHEMA:
            raise ValueError("Qv4 lightweight-source schema mismatch")
        tree = json.loads(str(handle.attrs["lightweight_tree_json"]))
        source_manifest = json.loads(str(handle.attrs["source_manifest_json"]))
        arrays = {key: np.asarray(handle[f"arrays/{key}"]) for key in handle["arrays"]}
        recorded_source_sha256 = str(handle.attrs["source_sha256"])
        recorded_attempt_sha256 = str(handle.attrs["attempt_sha256"])
    decoded = _decode_source_tree(tree, arrays)
    observation = decoded.get("field_tape", {}).get("observation", {})
    forbidden = {"rgb", "depth_m", "depth_valid_mask", "segmentation"}
    if not isinstance(observation, Mapping) or forbidden & set(observation):
        raise ValueError("Qv4 lightweight source contains raw vision fields")
    omitted = decoded.get("omitted_vision_arrays")
    if not isinstance(omitted, Mapping) or set(omitted) != forbidden:
        raise ValueError(
            "Qv4 lightweight source has no complete vision digest inventory"
        )
    for name, descriptor in omitted.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Qv4 lightweight {name} descriptor is invalid")
        _sha256_value(f"Qv4 lightweight {name} SHA-256", descriptor.get("sha256"))
        if not isinstance(descriptor.get("dtype"), str) or not isinstance(
            descriptor.get("shape"), list
        ):
            raise ValueError(f"Qv4 lightweight {name} descriptor is invalid")
    if source_manifest.get("source_sha256") != recorded_source_sha256:
        raise ValueError("Qv4 lightweight source manifest identity mismatch")
    _sha256_value("Qv4 lightweight attempt SHA-256", recorded_attempt_sha256)
    receipt: dict[str, Any] = {
        "schema_version": QUALITY_V4_LIGHTWEIGHT_SOURCE_SCHEMA,
        "episode_id": decoded.get("episode_id"),
        "task_id": decoded.get("task_id"),
        "source_sha256": recorded_source_sha256,
        "attempt_sha256": recorded_attempt_sha256,
        "raw_vision_arrays_present": False,
        "array_count": len(arrays),
        "file_sha256": _file_sha256(source_path),
    }
    receipt["receipt_sha256"] = _stable_sha256(receipt)
    return receipt


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(dict(payload), indent=2, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_quality_v4_full_export(
    path: Path,
    *,
    source: Mapping[str, Any],
    recorded_attempt: Mapping[str, Any],
) -> Path:
    """Write the selected winner's complete Qv4 source tree to atomic HDF5."""

    _validate_attempt(recorded_attempt)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    tree = _encode_source_tree(dict(source), arrays)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = QUALITY_V4_FULL_EXPORT_SCHEMA
            handle.attrs["source_tree_json"] = _canonical_json(tree)
            handle.attrs["recorded_attempt_json"] = _canonical_json(
                dict(recorded_attempt)
            )
            group = handle.create_group("arrays")
            for key, array in sorted(arrays.items()):
                compressible = bool(array.size and array.ndim > 0)
                group.create_dataset(
                    key,
                    data=array,
                    compression="gzip" if compressible else None,
                    compression_opts=4 if compressible else None,
                    shuffle=bool(compressible and array.dtype.kind not in {"S"}),
                )
            handle.flush()
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def audit_quality_v4_full_export(path: Path) -> dict[str, Any]:
    """Read complete HDF5 sources and independently rebuild both Qv4 gates."""

    quality = _se3_quality()
    export_path = Path(path)
    if export_path.is_symlink() or not export_path.is_file():
        raise FileNotFoundError(export_path)
    with h5py.File(export_path, "r") as handle:
        if handle.attrs.get("schema_version") != QUALITY_V4_FULL_EXPORT_SCHEMA:
            raise ValueError("Qv4 full-export schema mismatch")
        tree = json.loads(str(handle.attrs["source_tree_json"]))
        recorded_attempt = json.loads(str(handle.attrs["recorded_attempt_json"]))
        arrays = {key: np.asarray(handle[f"arrays/{key}"]) for key in handle["arrays"]}
    source = _decode_source_tree(tree, arrays)
    if not isinstance(source, Mapping) or not isinstance(recorded_attempt, Mapping):
        raise ValueError("Qv4 full export does not decode to mappings")
    _validate_attempt(recorded_attempt)
    recomputed = build_quality_v4_attempt(source)
    quality.assert_quality_v4_replay_equal(
        recorded_attempt["summary"],
        recomputed["summary"],
        recorded_attempt["layer1_gate"],
        recomputed["layer1_gate"],
        recorded_attempt["layer2_gate"],
        recomputed["layer2_gate"],
    )
    if dict(recorded_attempt) != recomputed:
        raise RuntimeError("Qv4 full-export attempt mismatch")
    layer1 = recomputed["layer1_gate"]
    layer2 = recomputed["layer2_gate"]
    replay = layer1["replay"]
    owner_review_complete = bool(recomputed["formal_thresholds_frozen"])
    passed = bool(recomputed["eligible"] and replay["passed"] and owner_review_complete)
    reason_codes = [
        *([] if layer1["passed"] else layer1["reason_codes"]),
        *([] if layer2["passed"] else layer2["reason_codes"]),
    ]
    if not owner_review_complete:
        reason_codes.append("FULL_EXPORT:THRESHOLDS_NOT_OWNER_REVIEWED_FROZEN")
    gate: dict[str, Any] = {
        "schema_version": QUALITY_V4_FULL_EXPORT_GATE_SCHEMA,
        "episode_id": recomputed["episode_id"],
        "task_id": recomputed["task_id"],
        "reset_pair_key": recomputed["reset_pair_key"],
        "export_file_sha256": _file_sha256(export_path),
        "attempt_sha256": recomputed["attempt_sha256"],
        "summary_sha256": recomputed["summary"]["summary_sha256"],
        "thresholds_sha256": recomputed["thresholds_sha256"],
        "orientation_contract_sha256": recomputed["orientation_contract_sha256"],
        "field_contract_sha256": recomputed["field_contract_sha256"],
        "layer1_gate": layer1,
        "layer2_gate": layer2,
        "replay_gate": replay,
        "full_export_recomputed": True,
        "formal_thresholds_frozen": recomputed["formal_thresholds_frozen"],
        "owner_review_complete": owner_review_complete,
        "passed": passed,
        "eligible_for_behavior_cloning": passed,
        "reason_codes": reason_codes,
    }
    gate["gate_sha256"] = _stable_sha256(gate)
    return gate


def _validate_full_export_gate(full_export_gate: Mapping[str, Any]) -> None:
    if full_export_gate.get("schema_version") != QUALITY_V4_FULL_EXPORT_GATE_SCHEMA:
        raise ValueError("Qv4 full-export gate schema mismatch")
    unsigned = dict(full_export_gate)
    recorded_sha256 = unsigned.pop("gate_sha256", None)
    _sha256_value("Qv4 full-export gate_sha256", recorded_sha256)
    if recorded_sha256 != _stable_sha256(unsigned):
        raise ValueError("Qv4 full-export gate SHA-256 mismatch")


def write_quality_v4_full_export_gate(
    output_root: Path, full_export_gate: Mapping[str, Any]
) -> Path:
    """Publish the independent winner re-gate next to its full HDF5 source."""

    _validate_full_export_gate(full_export_gate)
    episode_id = _safe_id(
        "Qv4 full-export gate episode_id", full_export_gate.get("episode_id")
    )
    destination = (
        Path(output_root)
        / QUALITY_V4_FULL_EXPORT_SUBDIRECTORY
        / f"{episode_id}.gate.json"
    )
    return _write_json_exclusive(destination, full_export_gate)


def dataset_quality_v4_validation(
    full_export_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate a verified full-export gate into the benchmark dataset contract.

    The dataset accepts only this independently recomputed gate.  It must never
    be populated from a lightweight attempt, a success flag, or a caller's
    precomputed summary.
    """

    _validate_full_export_gate(full_export_gate)
    recorded_sha256 = full_export_gate["gate_sha256"]
    if full_export_gate.get("full_export_recomputed") is not True:
        raise ValueError("Qv4 dataset validation requires a full-export recomputation")
    layer1 = full_export_gate.get("layer1_gate")
    layer2 = full_export_gate.get("layer2_gate")
    replay = full_export_gate.get("replay_gate")
    if not all(isinstance(gate, Mapping) for gate in (layer1, layer2, replay)):
        raise ValueError("Qv4 full-export gate is missing a component gate")
    expected_passed = bool(
        layer1.get("passed")
        and layer2.get("passed")
        and replay.get("passed")
        and full_export_gate.get("formal_thresholds_frozen") is True
        and full_export_gate.get("owner_review_complete") is True
    )
    if full_export_gate.get("passed") is not expected_passed:
        raise ValueError("Qv4 full-export passed flag is not gate-derived")
    result = {
        "schema_version": QUALITY_V4_DATASET_VALIDATION_SCHEMA,
        "full_export_gate_sha256": recorded_sha256,
        "export_file_sha256": full_export_gate.get("export_file_sha256"),
        "attempt_sha256": full_export_gate.get("attempt_sha256"),
        "summary_sha256": full_export_gate.get("summary_sha256"),
        "thresholds_sha256": full_export_gate.get("thresholds_sha256"),
        "orientation_contract_sha256": full_export_gate.get(
            "orientation_contract_sha256"
        ),
        "field_contract_sha256": full_export_gate.get("field_contract_sha256"),
        "layer1_gate": dict(layer1),
        "layer2_gate": dict(layer2),
        "replay_gate": dict(replay),
        "full_export_recomputed": True,
        "formal_thresholds_frozen": (
            full_export_gate.get("formal_thresholds_frozen") is True
        ),
        "owner_review_complete": (
            full_export_gate.get("owner_review_complete") is True
        ),
        "passed": expected_passed,
    }
    return result


def _layer2_actuals(attempt: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    checks = attempt.get("layer2_gate", {}).get("checks")
    if isinstance(checks, (str, bytes)) or not isinstance(checks, Sequence):
        raise ValueError("Qv4 attempt has no Layer-2 checks")
    result: dict[tuple[str, str], float] = {}
    for row in checks:
        if not isinstance(row, Mapping):
            raise ValueError("Qv4 Layer-2 check is invalid")
        identity = (str(row.get("phase")), str(row.get("metric")))
        actual = row.get("actual")
        if (
            identity in result
            or isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
        ):
            raise ValueError("Qv4 Layer-2 actual inventory is invalid")
        result[identity] = float(actual)
    return result


def paired_pareto_winner(
    *,
    planner_attempt: Mapping[str, Any],
    rl_attempt: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Select independently gated attempts; return never affects the decision."""

    _validate_attempt(planner_attempt)
    _validate_attempt(rl_attempt)
    threshold_validation = validate_quality_v4_thresholds(
        thresholds, require_formal_freeze=True
    )
    if planner_attempt.get("task_id") != rl_attempt.get(
        "task_id"
    ) or planner_attempt.get("reset_pair_key") != rl_attempt.get("reset_pair_key"):
        raise ValueError("Qv4 Pareto attempts are not a same-reset pair")
    task_id = str(planner_attempt["task_id"])
    if any(
        attempt.get("thresholds_sha256") != threshold_validation["thresholds_sha256"]
        for attempt in (planner_attempt, rl_attempt)
    ):
        raise ValueError("Qv4 Pareto attempts do not use the selected thresholds")
    planner_eligible = planner_attempt.get("eligible") is True
    rl_eligible = rl_attempt.get("eligible") is True
    comparisons: list[dict[str, Any]] = []
    rl_non_worse = False
    rl_strictly_better = False
    if planner_eligible and rl_eligible:
        planner_actuals = _layer2_actuals(planner_attempt)
        rl_actuals = _layer2_actuals(rl_attempt)
        raw_checks = thresholds["tasks"][task_id]["checks"]
        for contract in raw_checks:
            identity = (str(contract["phase"]), str(contract["metric"]))
            if identity not in planner_actuals or identity not in rl_actuals:
                raise ValueError(f"Qv4 Pareto metric {identity} is missing")
            direction = str(contract["direction"])
            tolerance = float(contract.get("paired_non_worse_tolerance", 0.0))
            margin = contract.get("strict_improvement_margin")
            if isinstance(margin, bool) or not isinstance(margin, (int, float)):
                raise ValueError(f"Qv4 Pareto metric {identity} has no strict margin")
            margin_value = float(margin)
            planner_value = planner_actuals[identity]
            rl_value = rl_actuals[identity]
            if direction == "minimize":
                non_worse = rl_value <= planner_value + tolerance
                strictly_better = rl_value < planner_value - margin_value
            else:
                non_worse = rl_value >= planner_value - tolerance
                strictly_better = rl_value > planner_value + margin_value
            comparisons.append(
                {
                    "phase": identity[0],
                    "metric": identity[1],
                    "direction": direction,
                    "planner": planner_value,
                    "rl": rl_value,
                    "paired_non_worse_tolerance": tolerance,
                    "strict_improvement_margin": margin_value,
                    "rl_non_worse": non_worse,
                    "rl_strictly_better": strictly_better,
                }
            )
        rl_non_worse = all(row["rl_non_worse"] for row in comparisons)
        rl_strictly_better = any(row["rl_strictly_better"] for row in comparisons)

    if rl_eligible and not planner_eligible:
        winner = "rl"
        reason = "planner_ineligible_rl_eligible"
    elif planner_eligible and not rl_eligible:
        winner = "planner"
        reason = "planner_eligible_rl_ineligible"
    elif not planner_eligible and not rl_eligible:
        winner = "reject"
        reason = "both_ineligible"
    elif rl_non_worse and rl_strictly_better:
        winner = "rl"
        reason = "rl_pareto_non_worse_with_strict_improvement"
    else:
        winner = "planner"
        reason = "rl_lacks_paired_pareto_improvement"
    result: dict[str, Any] = {
        "schema_version": QUALITY_V4_PAIRED_PARETO_SCHEMA,
        "task_id": task_id,
        "reset_pair_key": planner_attempt["reset_pair_key"],
        "thresholds_sha256": threshold_validation["thresholds_sha256"],
        "planner_attempt_sha256": planner_attempt["attempt_sha256"],
        "rl_attempt_sha256": rl_attempt["attempt_sha256"],
        "planner_eligible": planner_eligible,
        "rl_eligible": rl_eligible,
        "rl_non_worse_on_all_dimensions": rl_non_worse,
        "rl_strictly_better_on_any_dimension": rl_strictly_better,
        "comparisons": comparisons,
        "winner": winner,
        "reason": reason,
        "return_diagnostic_only": True,
        "diagnostic_return": {
            "planner": planner_attempt.get("return_diagnostic"),
            "rl": rl_attempt.get("return_diagnostic"),
        },
    }
    result["selection_sha256"] = _stable_sha256(result)
    return result


__all__ = [
    "QUALITY_V4_ARTIFACT_SUBDIRECTORY",
    "QUALITY_V4_ATTEMPT_SCHEMA",
    "QUALITY_V4_ATTEMPT_SUBDIRECTORY",
    "QUALITY_V4_DATASET_VALIDATION_SCHEMA",
    "QUALITY_V4_FULL_EXPORT_GATE_SCHEMA",
    "QUALITY_V4_FULL_EXPORT_SCHEMA",
    "QUALITY_V4_FULL_EXPORT_SUBDIRECTORY",
    "QUALITY_V4_LIGHTWEIGHT_SOURCE_SCHEMA",
    "QUALITY_V4_LIGHTWEIGHT_SUBDIRECTORY",
    "QUALITY_V4_PAIRED_PARETO_SCHEMA",
    "QUALITY_V4_ROLLOUT_REFERENCE_SCHEMA",
    "QUALITY_V4_T5_ACTION_HISTORY_SCHEMA",
    "QUALITY_V4_VISION_TOLERANCE_SCHEMA",
    "audit_quality_v4_full_export",
    "audit_quality_v4_lightweight_source",
    "build_quality_v4_attempt",
    "build_quality_v4_rollout_source",
    "dataset_quality_v4_validation",
    "finalize_quality_v4_fresh_replay",
    "load_quality_v4_thresholds",
    "paired_pareto_winner",
    "quality_v4_source_manifest",
    "quality_v4_threshold_check_inventory",
    "validate_quality_v4_thresholds",
    "write_quality_v4_attempt",
    "write_quality_v4_full_export",
    "write_quality_v4_full_export_gate",
    "write_quality_v4_lightweight_source",
]
