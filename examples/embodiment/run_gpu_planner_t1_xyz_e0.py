#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run one fail-closed ``t1_xyz`` GPU Planner evidence row.

The result path accepts only a sealed E0/D32 manifest row.  It validates the
complete export-bound ResetRequest and the clean five-repository source tuple
before constructing CUDA, plans from the current observation at every control
step, and emits a countable result only after semantic fresh-backend replay,
quality-v2, exact-once terminal ledger, and evidence-file gates all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


def _load_strict_contract() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/envs/dynamic_benchmark/t1_xyz_strict_evidence.py"
    )
    name = "_t1_xyz_strict_evidence_contract"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load review evidence contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_STRICT = _load_strict_contract()
BACKEND_ID = _STRICT.BACKEND_ID
EXECUTION_CONTRACT = _STRICT.EXECUTION_CONTRACT
QUALITY_EVALUATOR_ID = _STRICT.QUALITY_EVALUATOR_ID
QUALITY_SCHEMA_VERSION = _STRICT.QUALITY_SCHEMA_VERSION
RESULT_SCHEMA_VERSION = _STRICT.RESULT_SCHEMA_VERSION
TASK_ID = _STRICT.TASK_ID
load_frozen_manifest = _STRICT.load_frozen_manifest
preflight_export_request = _STRICT.preflight_export_request
request_identity = _STRICT.request_identity
validate_repository_tuple = _STRICT.validate_repository_tuple
validate_result_for_row = _STRICT.validate_result_for_row

_REVIEW_CAMERAS = ("agentview", "robot0_eye_in_hand")
_VISUAL_PRIVILEGED_SUFFIXES = (
    "target_visible_pixels",
    "target_image_fraction",
    "occluder_visible_pixels",
)
_FLOAT_REPORT_ATOL = 1.0e-5
_FLOAT_REPORT_RTOL = 1.0e-5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--phase", choices=("e0", "d32"), default="e0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tape-output",
        type=Path,
        help="Complete observation/action tape; defaults beside --output.",
    )
    parser.add_argument(
        "--visual-gif",
        type=Path,
        help="GPU-rendered scene/wrist GIF; defaults beside --output.",
    )
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--research-source-root", type=Path, required=True)
    parser.add_argument("--se3-source-root", type=Path, required=True)
    parser.add_argument("--mjwarp-source-root", type=Path, required=True)
    parser.add_argument("--rlinf-source-root", type=Path, required=True)
    parser.add_argument("--dynamic-source-root", type=Path, required=True)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _observation_payload(observation: Any) -> dict[str, Any]:
    """Return a compact, field-addressable observation fingerprint payload."""

    return {
        "identity": {
            "episode_id": observation.episode_id,
            "task_id": observation.task_id,
            "physics_step": int(observation.physics_step),
            "control_step": int(observation.control_step),
            "policy_step": int(observation.policy_step),
            "time_s": float(observation.time_s),
        },
        "rgb": {name: _array_digest(value) for name, value in observation.rgb.items()},
        "depth_m": {
            name: _array_digest(value) for name, value in observation.depth_m.items()
        },
        "segmentation": {
            name: _array_digest(value)
            for name, value in observation.segmentation.items()
        },
        "proprio": {
            name: _array_digest(value) for name, value in observation.proprio.items()
        },
        "privileged": {
            name: _array_digest(value) for name, value in observation.privileged.items()
        },
        "events": [
            {
                "name": event.name,
                "physics_step": int(event.physics_step),
                "time_s": float(event.time_s),
            }
            for event in observation.events_since_last_observation
        ],
    }


def _observation_digest(observation: Any) -> str:
    return _json_sha256(_observation_payload(observation))


def _command_payload(command: Any) -> dict[str, Any]:
    return {
        "mode": getattr(command.mode, "value", command.mode),
        "policy_step": int(command.policy_step),
        "values": np.asarray(command.values, dtype=np.float64).tolist(),
    }


def _first_sequence_mismatch(
    expected: Sequence[Any],
    actual: Sequence[Any],
) -> dict[str, Any] | None:
    """Describe the first unequal item without weakening exact equality."""

    expected_values = tuple(expected)
    actual_values = tuple(actual)
    shared_length = min(len(expected_values), len(actual_values))
    mismatch_index = next(
        (
            index
            for index in range(shared_length)
            if expected_values[index] != actual_values[index]
        ),
        None,
    )
    if mismatch_index is None:
        if len(expected_values) == len(actual_values):
            return None
        mismatch_index = shared_length
    return {
        "sequence_index": mismatch_index,
        "expected_length": len(expected_values),
        "actual_length": len(actual_values),
        "expected_item": (
            expected_values[mismatch_index]
            if mismatch_index < len(expected_values)
            else None
        ),
        "actual_item": (
            actual_values[mismatch_index]
            if mismatch_index < len(actual_values)
            else None
        ),
        "expected_sequence_sha256": _json_sha256(expected_values),
        "actual_sequence_sha256": _json_sha256(actual_values),
    }


def _first_divergence(
    *,
    fresh_backend_distinct: bool,
    backend_identity_exact: bool,
    reset_identity_exact: bool,
    action_tape_exact: bool,
    observation_semantic_mismatch: Mapping[str, Any] | None,
    review_semantic_mismatch: Mapping[str, Any] | None,
    outcome_mismatch: Mapping[str, Any] | None,
    terminal_ledger_semantic_exact: bool,
    terminal_ledger_exact_once: bool,
    replay_stop: Mapping[str, Any] | None,
    commands: Sequence[Any],
) -> dict[str, Any] | None:
    """Locate the first blocking semantic replay failure."""

    if not fresh_backend_distinct:
        return {"channel": "fresh_backend_distinct", "control_step": None}
    if not backend_identity_exact:
        return {"channel": "backend_identity", "control_step": None}
    if not reset_identity_exact:
        return {"channel": "reset_identity", "control_step": None}
    if not action_tape_exact:
        return {"channel": "action_tape", "control_step": None}
    if replay_stop is not None:
        control_step = int(replay_stop["policy_step"])
        preceding_action = None
        if control_step > 0 and control_step - 1 < len(commands):
            preceding_action = _command_payload(commands[control_step - 1])
        return {
            "channel": "replay_terminated_before_action_tape_end",
            "control_step": control_step,
            "transition_action": preceding_action,
            "details": dict(replay_stop),
        }

    candidates: list[tuple[int, int, str, Mapping[str, Any]]] = []

    def semantic_step(mismatch: Mapping[str, Any]) -> int:
        path = str(mismatch.get("path", ""))
        _, bracket, suffix = path.partition("[")
        token, closing, _ = suffix.partition("]")
        if bracket and closing:
            try:
                return int(token)
            except ValueError:
                pass
        return 0

    for priority, (channel, mismatch, outcome_offset) in enumerate(
        (
            ("observation_semantics", observation_semantic_mismatch, 0),
            ("review_semantics", review_semantic_mismatch, 0),
            ("outcome", outcome_mismatch, 1),
        )
    ):
        if mismatch is not None:
            sequence_index = mismatch.get("sequence_index")
            if isinstance(sequence_index, bool) or not isinstance(sequence_index, int):
                sequence_index = semantic_step(mismatch)
            candidates.append(
                (
                    sequence_index + outcome_offset,
                    priority,
                    channel,
                    mismatch,
                )
            )
    if candidates:
        control_step, _, channel, mismatch = min(candidates)
        preceding_action = None
        if control_step > 0 and control_step - 1 < len(commands):
            preceding_action = _command_payload(commands[control_step - 1])
        return {
            "channel": channel,
            "control_step": control_step,
            "transition_action": preceding_action,
            "mismatch": dict(mismatch),
        }
    if not terminal_ledger_semantic_exact:
        return {
            "channel": "terminal_ledger_semantics",
            "control_step": None,
            "transition_action": (_command_payload(commands[-1]) if commands else None),
        }
    if not terminal_ledger_exact_once:
        return {
            "channel": "terminal_ledger_exact_once",
            "control_step": None,
            "transition_action": (_command_payload(commands[-1]) if commands else None),
        }
    return None


def _new_numeric_drift_report() -> dict[str, Any]:
    return {
        "blocking": False,
        "float_atol": _FLOAT_REPORT_ATOL,
        "float_rtol": _FLOAT_REPORT_RTOL,
        "compared_arrays": 0,
        "exact_arrays": 0,
        "within_reporting_tolerance_arrays": 0,
        "compared_values": 0,
        "absolute_error_sum": 0.0,
        "max_absolute_error": 0.0,
        "first_exact_mismatch": None,
        "first_reporting_tolerance_mismatch": None,
    }


def _record_array_drift(
    report: dict[str, Any],
    *,
    path: str,
    expected: Any,
    actual: Any,
) -> dict[str, Any] | None:
    left = np.asarray(expected)
    right = np.asarray(actual)
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "path": path,
            "expected_shape": list(left.shape),
            "actual_shape": list(right.shape),
            "expected_dtype": str(left.dtype),
            "actual_dtype": str(right.dtype),
        }
    if np.issubdtype(left.dtype, np.number) and (
        not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
    ):
        return {"path": path, "error": "non-finite numeric value"}

    exact = np.array_equal(left, right)
    if np.issubdtype(left.dtype, np.floating):
        within_tolerance = np.allclose(
            left,
            right,
            rtol=_FLOAT_REPORT_RTOL,
            atol=_FLOAT_REPORT_ATOL,
            equal_nan=False,
        )
    else:
        within_tolerance = exact
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    value_count = int(difference.size)
    absolute_error_sum = float(np.sum(difference, dtype=np.float64))
    max_absolute_error = float(np.max(difference)) if value_count else 0.0
    report["compared_arrays"] += 1
    report["exact_arrays"] += int(exact)
    report["within_reporting_tolerance_arrays"] += int(within_tolerance)
    report["compared_values"] += value_count
    report["absolute_error_sum"] += absolute_error_sum
    report["max_absolute_error"] = max(report["max_absolute_error"], max_absolute_error)
    mismatch = {
        "path": path,
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": absolute_error_sum / value_count if value_count else 0.0,
    }
    if not exact and report["first_exact_mismatch"] is None:
        report["first_exact_mismatch"] = mismatch
    if not within_tolerance and report["first_reporting_tolerance_mismatch"] is None:
        report["first_reporting_tolerance_mismatch"] = mismatch
    return None


def _finish_numeric_drift_report(report: dict[str, Any]) -> dict[str, Any]:
    compared_values = int(report.pop("compared_values"))
    absolute_error_sum = float(report.pop("absolute_error_sum"))
    report["mean_absolute_error"] = (
        absolute_error_sum / compared_values if compared_values else 0.0
    )
    report["exact"] = report["exact_arrays"] == report["compared_arrays"]
    report["within_reporting_tolerance"] = (
        report["within_reporting_tolerance_arrays"] == report["compared_arrays"]
    )
    return report


def _compare_observation_sequences(
    expected: Sequence[Any],
    actual: Sequence[Any],
) -> dict[str, Any]:
    left_values = tuple(expected)
    right_values = tuple(actual)
    report = _new_numeric_drift_report()
    semantic_mismatch: dict[str, Any] | None = None
    first_event_mismatch: dict[str, Any] | None = None
    if len(left_values) != len(right_values):
        semantic_mismatch = {
            "path": "observations",
            "expected_length": len(left_values),
            "actual_length": len(right_values),
        }
    for index, (left, right) in enumerate(zip(left_values, right_values, strict=False)):
        left_identity = (
            left.episode_id,
            left.task_id,
            int(left.physics_step),
            int(left.control_step),
            int(left.policy_step),
        )
        right_identity = (
            right.episode_id,
            right.task_id,
            int(right.physics_step),
            int(right.control_step),
            int(right.policy_step),
        )
        if semantic_mismatch is None and left_identity != right_identity:
            semantic_mismatch = {
                "path": f"observations[{index}].identity",
                "expected": list(left_identity),
                "actual": list(right_identity),
            }
        mismatch = _record_array_drift(
            report,
            path=f"observations[{index}].time_s",
            expected=np.asarray([left.time_s], dtype=np.float64),
            actual=np.asarray([right.time_s], dtype=np.float64),
        )
        if semantic_mismatch is None and mismatch is not None:
            semantic_mismatch = mismatch
        left_events = tuple(event.name for event in left.events_since_last_observation)
        right_events = tuple(
            event.name for event in right.events_since_last_observation
        )
        if first_event_mismatch is None and left_events != right_events:
            first_event_mismatch = {
                "path": f"observations[{index}].events",
                "expected": list(left_events),
                "actual": list(right_events),
            }
        for group_name in ("rgb", "depth_m", "segmentation", "proprio", "privileged"):
            left_group = getattr(left, group_name)
            right_group = getattr(right, group_name)
            if semantic_mismatch is None and tuple(sorted(left_group)) != tuple(
                sorted(right_group)
            ):
                semantic_mismatch = {
                    "path": f"observations[{index}].{group_name}",
                    "expected_keys": sorted(left_group),
                    "actual_keys": sorted(right_group),
                }
                continue
            for name in sorted(set(left_group) & set(right_group)):
                mismatch = _record_array_drift(
                    report,
                    path=f"observations[{index}].{group_name}.{name}",
                    expected=left_group[name],
                    actual=right_group[name],
                )
                if semantic_mismatch is None and mismatch is not None:
                    semantic_mismatch = mismatch
    return {
        "semantic_structure_exact": semantic_mismatch is None,
        "first_semantic_mismatch": semantic_mismatch,
        "event_drift": {
            "blocking": False,
            "exact": first_event_mismatch is None,
            "first_mismatch": first_event_mismatch,
        },
        "numeric_drift": _finish_numeric_drift_report(report),
    }


def _compare_review_sequences(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left_values = tuple(expected)
    right_values = tuple(actual)
    report = _new_numeric_drift_report()
    semantic_mismatch: dict[str, Any] | None = None
    if len(left_values) != len(right_values):
        semantic_mismatch = {
            "path": "review",
            "expected_length": len(left_values),
            "actual_length": len(right_values),
        }
    for index, (left, right) in enumerate(zip(left_values, right_values, strict=False)):
        if semantic_mismatch is None and set(left) != set(right):
            semantic_mismatch = {
                "path": f"review[{index}]",
                "expected_cameras": sorted(left),
                "actual_cameras": sorted(right),
            }
        for camera in sorted(set(left) & set(right)):
            mismatch = _record_array_drift(
                report,
                path=f"review[{index}].{camera}",
                expected=left[camera],
                actual=right[camera],
            )
            if semantic_mismatch is None and mismatch is not None:
                semantic_mismatch = mismatch
    return {
        "semantic_structure_exact": semantic_mismatch is None,
        "first_semantic_mismatch": semantic_mismatch,
        "numeric_drift": _finish_numeric_drift_report(report),
    }


def _task_quality_semantics(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return {"invalid": type(value).__name__}
    components = value.get("components")
    if not isinstance(components, Mapping):
        return {"invalid_components": type(components).__name__}
    return {
        "episode_id": value.get("episode_id"),
        "task_id": value.get("task_id"),
        "evaluator_backend_id": value.get("evaluator_backend_id"),
        "schema_version": value.get("schema_version"),
        "schema_sha256": value.get("schema_sha256"),
        "terminal": value.get("terminal"),
        "components": {
            name: {
                key: component.get(key)
                for key in ("direction", "unit", "scientific_resolution", "reducer")
            }
            for name, component in components.items()
            if isinstance(component, Mapping)
        },
    }


def _terminal_ledger_semantics(
    ledger: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "lane": row.get("lane"),
            "episode_id": row.get("episode_id"),
            "task_id": row.get("task_id"),
            "outcome": row.get("outcome"),
            "terminated": row.get("terminated"),
            "truncated": row.get("truncated"),
            "success": row.get("success"),
            "termination_reason": row.get("termination_reason"),
            "event_names": [event.get("name") for event in row.get("events", ())],
            "task_quality": _task_quality_semantics(row.get("task_quality")),
        }
        for row in ledger
    ]


def _compare_terminal_numeric_drift(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = _new_numeric_drift_report()
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        for name in ("physics_step", "control_step", "policy_step", "completion"):
            _record_array_drift(
                report,
                path=f"terminal[{index}].{name}",
                expected=np.asarray([left.get(name)], dtype=np.float64),
                actual=np.asarray([right.get(name)], dtype=np.float64),
            )
        for event_index, (left_event, right_event) in enumerate(
            zip(left.get("events", ()), right.get("events", ()), strict=False)
        ):
            for name in ("physics_step", "time_s"):
                _record_array_drift(
                    report,
                    path=f"terminal[{index}].events[{event_index}].{name}",
                    expected=np.asarray([left_event.get(name)], dtype=np.float64),
                    actual=np.asarray([right_event.get(name)], dtype=np.float64),
                )
        left_quality = left.get("task_quality")
        right_quality = right.get("task_quality")
        if isinstance(left_quality, Mapping) and isinstance(right_quality, Mapping):
            _record_array_drift(
                report,
                path=f"terminal[{index}].task_quality.physics_sample_count",
                expected=np.asarray(
                    [left_quality.get("physics_sample_count")], dtype=np.float64
                ),
                actual=np.asarray(
                    [right_quality.get("physics_sample_count")], dtype=np.float64
                ),
            )
            left_components = left_quality.get("components", {})
            right_components = right_quality.get("components", {})
            for name in sorted(set(left_components) & set(right_components)):
                _record_array_drift(
                    report,
                    path=f"terminal[{index}].task_quality.components.{name}.value",
                    expected=np.asarray(
                        [left_components[name].get("value")], dtype=np.float64
                    ),
                    actual=np.asarray(
                        [right_components[name].get("value")], dtype=np.float64
                    ),
                )
    return _finish_numeric_drift_report(report)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _readonly_copy(value: Any) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _state_and_review_observation(
    observation: Any,
) -> tuple[Any, dict[str, np.ndarray]]:
    """Split one rendered audit packet into STATE Planner and review material."""

    groups = {
        "rgb": observation.rgb,
        "depth_m": observation.depth_m,
        "segmentation": observation.segmentation,
    }
    for group_name, group in groups.items():
        if not isinstance(group, Mapping) or set(group) != set(_REVIEW_CAMERAS):
            raise RuntimeError(
                f"t1_xyz {group_name} must contain the exact scene/wrist cameras"
            )

    review: dict[str, np.ndarray] = {}
    state_rgb: dict[str, np.ndarray] = {}
    state_depth: dict[str, np.ndarray] = {}
    state_segmentation: dict[str, np.ndarray] = {}
    for camera in _REVIEW_CAMERAS:
        rgb = np.asarray(observation.rgb[camera])
        depth = np.asarray(observation.depth_m[camera])
        segmentation = np.asarray(observation.segmentation[camera])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise RuntimeError(f"t1_xyz review RGB for {camera} is not uint8 HxWx3")
        scalar_image_shape = (*rgb.shape[:2], 1)
        if (
            depth.shape != scalar_image_shape
            or segmentation.shape != scalar_image_shape
        ):
            raise RuntimeError(f"t1_xyz review material shapes differ for {camera}")
        if not np.all(np.isfinite(depth)):
            raise RuntimeError(f"t1_xyz review depth for {camera} is non-finite")
        review[camera] = _readonly_copy(rgb)
        state_rgb[camera] = _readonly_copy(np.zeros_like(rgb))
        state_depth[camera] = _readonly_copy(np.ones_like(depth))
        state_segmentation[camera] = _readonly_copy(np.zeros_like(segmentation))

    if not isinstance(observation.privileged, Mapping):
        raise RuntimeError("t1_xyz observation privileged payload must be a mapping")
    privileged = {
        name: _readonly_copy(value) for name, value in observation.privileged.items()
    }
    required_visual_keys = {
        f"{camera}_{suffix}"
        for camera in _REVIEW_CAMERAS
        for suffix in _VISUAL_PRIVILEGED_SUFFIXES
    }
    missing = sorted(required_visual_keys - set(privileged))
    if missing:
        raise RuntimeError(
            f"t1_xyz rendered audit lacks visual privileged fields: {missing}"
        )
    for name in required_visual_keys:
        privileged[name] = _readonly_copy(np.zeros_like(privileged[name]))

    state_observation = replace(
        observation,
        rgb=MappingProxyType(state_rgb),
        depth_m=MappingProxyType(state_depth),
        segmentation=MappingProxyType(state_segmentation),
        privileged=MappingProxyType(privileged),
    )
    return state_observation, review


def _review_digest(review: Mapping[str, Any]) -> str:
    if set(review) != set(_REVIEW_CAMERAS):
        raise RuntimeError("t1_xyz review packet changed the scene/wrist camera set")
    return _json_sha256(
        {camera: _array_digest(review[camera]) for camera in _REVIEW_CAMERAS}
    )


def _ledger_payload(ledger: Any) -> list[dict[str, Any]]:
    if ledger is None:
        return []
    return [
        {
            "lane": int(row.lane),
            "episode_id": row.episode_id,
            "task_id": row.task_id,
            "outcome": row.outcome.value,
            "terminated": bool(row.terminated),
            "truncated": bool(row.truncated),
            "success": bool(row.success),
            "termination_reason": row.termination_reason,
            "physics_step": int(row.physics_step),
            "control_step": int(row.control_step),
            "policy_step": int(row.policy_step),
            "completion": float(row.completion),
            "events": [
                {
                    "name": event.name,
                    "physics_step": int(event.physics_step),
                    "time_s": float(event.time_s),
                }
                for event in row.events
            ],
            "task_quality": None
            if row.task_quality is None
            else row.task_quality.to_dict(),
        }
        for row in ledger.rows
    ]


def _provenance_payload(provenance: Any) -> dict[str, Any]:
    names = (
        "backend_id",
        "implementation_version",
        "device_platform",
        "device_name",
        "device_ordinal",
        "git_commit",
        "git_tree",
        "physical_device_uuid",
        "physical_device_pci_bus_id",
        "physical_device_identity_source",
        "precision",
    )
    return {
        name: getattr(provenance, name)
        for name in names
        if getattr(provenance, name, None) is not None
    } | {"runtime_versions": dict(getattr(provenance, "runtime_versions", {}))}


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_gpu_uuid: str,
    expected_commit: str,
    expected_tree: str,
) -> None:
    if provenance.get("backend_id") != BACKEND_ID:
        raise RuntimeError("t1_xyz result backend is not mjwarp_gpu_v1")
    if provenance.get("device_platform") not in {"cuda", "gpu"}:
        raise RuntimeError("t1_xyz result did not use CUDA/GPU physics")
    if provenance.get("physical_device_uuid") != expected_gpu_uuid:
        raise RuntimeError("t1_xyz result used a different physical GPU")
    if (
        provenance.get("git_commit") != expected_commit
        or provenance.get("git_tree") != expected_tree
    ):
        raise RuntimeError("t1_xyz result SE3 source provenance drifted")


def _validate_module_root(module: Any, expected_root: Path, name: str) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError(f"{name} import has no filesystem identity")
    resolved = Path(module_file).resolve(strict=True)
    try:
        resolved.relative_to(expected_root.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"{name} imported outside the sealed source root") from exc


def _make_command(request: Any, values: Sequence[float], policy_step: int) -> Any:
    from se3_wam.benchmark.api import ActionCommand

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise ValueError("Planner E7 action must be a finite 7-vector")
    return ActionCommand(
        mode=request.action_mode,
        values=np.clip(array, -1.0, 1.0),
        policy_step=int(policy_step),
    )


def _replay(
    *,
    backend: Any,
    request: Any,
    observations: tuple[Any, ...],
    reviews: tuple[Mapping[str, Any], ...],
    commands: tuple[Any, ...],
    outcomes: tuple[tuple[bool, bool, bool, str | None], ...],
    terminal_ledger: Any,
) -> dict[str, Any]:
    """Run a fresh replay with exact identities and semantic outcome gates."""

    replay_backend = backend.new_replay_backend()
    try:
        fresh_backend_distinct = replay_backend is not backend
        primary_provenance = _provenance_payload(backend.provenance)
        replay_provenance = _provenance_payload(replay_backend.provenance)
        backend_identity_exact = replay_provenance == primary_provenance
        frozen_requests = tuple(replay_backend.frozen_requests)
        reset_identity_exact = len(frozen_requests) == 1 and request_identity(
            frozen_requests[0]
        ) == request_identity(request)
        replay_raw_observation = replay_backend.reset((request,))[0]
        replay_observation, replay_review = _state_and_review_observation(
            replay_raw_observation
        )
        replay_observations = [replay_observation]
        observation_digests = [_observation_digest(replay_observation)]
        replay_reviews = [replay_review]
        actual_review_digests = [_review_digest(replay_review)]
        replay_action_payloads: list[dict[str, Any]] = []
        replay_outcomes: list[tuple[bool, bool, bool, str | None]] = []
        replay_stop: dict[str, Any] | None = None
        for command_index, command in enumerate(commands):
            replay_policy_step = int(replay_backend.policy_steps()[0])
            replay_command = _make_command(
                request,
                command.values,
                replay_policy_step,
            )
            replay_action_payloads.append(_command_payload(replay_command))
            result = replay_backend.step((replay_command,))[0]
            if result is None:
                replay_stop = {
                    "reason": "backend_returned_none_after_terminal",
                    "command_index": command_index,
                    "policy_step": replay_policy_step,
                    "submitted_action_count": len(replay_action_payloads),
                    "expected_action_count": len(commands),
                }
                break
            replay_observation, replay_review = _state_and_review_observation(
                result.observation
            )
            replay_observations.append(replay_observation)
            replay_reviews.append(replay_review)
            observation_digests.append(_observation_digest(replay_observation))
            actual_review_digests.append(_review_digest(replay_review))
            replay_outcomes.append(
                (
                    bool(result.terminated),
                    bool(result.truncated),
                    bool(result.success),
                    result.termination_reason,
                )
            )
        replay_task_quality_margin = _task_quality_margin_diagnostic(replay_backend)
        replay_ledger_object = replay_backend.last_terminal_ledger
        replay_ledger = _ledger_payload(replay_ledger_object)
        expected_ledger = _ledger_payload(terminal_ledger)
        exact_once_error = None
        exact_once_witnesses: tuple[str, ...] = ()
        try:
            from rlinf.envs.dynamic_benchmark.gpu_backend import (
                assert_terminal_ledger_exact_once,
            )

            exact_once_witnesses = assert_terminal_ledger_exact_once(
                replay_backend,
                () if replay_ledger_object is None else replay_ledger_object.rows,
            )
        except Exception as exc:
            exact_once_error = {"error_type": type(exc).__name__, "error": str(exc)}
        terminal_ledger_exact_once = exact_once_error is None
        if len(replay_ledger) == 1:
            _validate_terminal_quality(
                replay_ledger[0],
                episode_id=str(request.episode_id),
            )
        expected_observation_payloads = [
            _observation_payload(value) for value in observations
        ]
        replay_observation_payloads = [
            _observation_payload(value) for value in replay_observations
        ]
        observation_mismatch = _first_sequence_mismatch(
            expected_observation_payloads,
            replay_observation_payloads,
        )
        outcome_mismatch = _first_sequence_mismatch(outcomes, replay_outcomes)
        terminal_ledger_mismatch = _first_sequence_mismatch(
            expected_ledger,
            replay_ledger,
        )
        observation_tape_exact = observation_mismatch is None
        expected_review_digests = tuple(_review_digest(value) for value in reviews)
        review_mismatch = _first_sequence_mismatch(
            expected_review_digests,
            actual_review_digests,
        )
        review_tape_exact = review_mismatch is None
        expected_action_payloads = [_command_payload(command) for command in commands]
        action_tape_exact = replay_action_payloads == expected_action_payloads
        outcomes_exact = outcome_mismatch is None
        terminal_ledger_exact = terminal_ledger_mismatch is None
        observation_comparison = _compare_observation_sequences(
            observations,
            replay_observations,
        )
        review_comparison = _compare_review_sequences(reviews, replay_reviews)
        terminal_ledger_semantic_exact = _terminal_ledger_semantics(
            expected_ledger
        ) == _terminal_ledger_semantics(replay_ledger)
        first_divergence = _first_divergence(
            fresh_backend_distinct=fresh_backend_distinct,
            backend_identity_exact=backend_identity_exact,
            reset_identity_exact=reset_identity_exact,
            action_tape_exact=action_tape_exact,
            observation_semantic_mismatch=observation_comparison[
                "first_semantic_mismatch"
            ],
            review_semantic_mismatch=review_comparison["first_semantic_mismatch"],
            outcome_mismatch=outcome_mismatch,
            terminal_ledger_semantic_exact=terminal_ledger_semantic_exact,
            terminal_ledger_exact_once=terminal_ledger_exact_once,
            replay_stop=replay_stop,
            commands=commands,
        )
        return {
            "mode": "semantic_fresh_backend_v1",
            "passed": bool(
                fresh_backend_distinct
                and backend_identity_exact
                and reset_identity_exact
                and action_tape_exact
                and observation_comparison["semantic_structure_exact"]
                and review_comparison["semantic_structure_exact"]
                and outcomes_exact
                and terminal_ledger_semantic_exact
                and terminal_ledger_exact_once
            ),
            "fresh_backend_distinct": fresh_backend_distinct,
            "backend_identity_exact": backend_identity_exact,
            "source_identity_exact": backend_identity_exact,
            "reset_identity_exact": reset_identity_exact,
            "action_tape_exact": action_tape_exact,
            "observation_semantic_structure_exact": observation_comparison[
                "semantic_structure_exact"
            ],
            "observation_tape_exact": observation_tape_exact,
            "observation_event_sequence_exact": observation_comparison["event_drift"][
                "exact"
            ],
            "observation_event_drift": observation_comparison["event_drift"],
            "observation_numeric_drift": observation_comparison["numeric_drift"],
            "review_semantic_structure_exact": review_comparison[
                "semantic_structure_exact"
            ],
            "review_tape_exact": review_tape_exact,
            "review_numeric_drift": review_comparison["numeric_drift"],
            "semantic_outcomes_exact": outcomes_exact,
            "outcomes_exact": outcomes_exact,
            "terminal_ledger_semantic_exact": terminal_ledger_semantic_exact,
            "terminal_ledger_exact": terminal_ledger_exact,
            "terminal_numeric_drift": _compare_terminal_numeric_drift(
                expected_ledger,
                replay_ledger,
            ),
            "terminal_ledger_exact_once": terminal_ledger_exact_once,
            "primary_provenance": primary_provenance,
            "replay_provenance": replay_provenance,
            "exact_once_negative_witnesses": list(exact_once_witnesses),
            "exact_once_error": exact_once_error,
            "replay_stop": replay_stop,
            "first_divergence": first_divergence,
            "replay_observation_sha256": _json_sha256(observation_digests),
            "replay_review_sha256": _json_sha256(actual_review_digests),
            "replay_ledger_sha256": _json_sha256(replay_ledger),
            "task_quality_margin_diagnostic": replay_task_quality_margin,
        }
    finally:
        replay_backend.close()


def _write_visual_gif(path: Path, reviews: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("review evidence export requires imageio") from exc
    frames = []
    for review in reviews:
        left = np.asarray(review["agentview"], dtype=np.uint8)
        right = np.asarray(
            review["robot0_eye_in_hand"],
            dtype=np.uint8,
        )
        if left.shape[0] != right.shape[0]:
            raise RuntimeError("GPU scene and wrist frames have different heights")
        frames.append(np.concatenate((left, right), axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), extension=".gif", duration=0.05, loop=0)


def _write_tape_npz(
    path: Path,
    observations: Sequence[Any],
    commands: Sequence[Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if not observations or len(observations) != len(commands) + 1:
        raise ValueError(
            "trajectory tape must contain one more observation than actions"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "episode_id": np.asarray([value.episode_id for value in observations]),
        "task_id": np.asarray([value.task_id for value in observations]),
        "physics_step": np.asarray(
            [value.physics_step for value in observations], dtype=np.int64
        ),
        "control_step": np.asarray(
            [value.control_step for value in observations], dtype=np.int64
        ),
        "policy_step": np.asarray(
            [value.policy_step for value in observations], dtype=np.int64
        ),
        "time_s": np.asarray(
            [value.time_s for value in observations], dtype=np.float64
        ),
        "action_values": np.stack(
            [np.asarray(value.values, dtype=np.float64) for value in commands]
        ),
        "action_policy_step": np.asarray(
            [value.policy_step for value in commands], dtype=np.int64
        ),
        "observation_digest": np.asarray(
            [_observation_digest(value) for value in observations]
        ),
    }
    for group_name in ("rgb", "depth_m", "segmentation", "proprio", "privileged"):
        keys = sorted(getattr(observations[0], group_name))
        for key in keys:
            payload[f"{group_name}/{key}"] = np.stack(
                [np.asarray(getattr(value, group_name)[key]) for value in observations]
            )
    np.savez_compressed(path, **payload)


def _validate_terminal_quality(
    terminal: Mapping[str, Any],
    *,
    episode_id: str,
) -> None:
    quality = terminal.get("task_quality")
    if terminal.get("success") is True and not isinstance(quality, Mapping):
        raise RuntimeError("successful t1_xyz result lacks task quality")
    if quality is None:
        return
    if (
        quality.get("episode_id") != episode_id
        or quality.get("task_id") != TASK_ID
        or quality.get("schema_version") != QUALITY_SCHEMA_VERSION
        or quality.get("evaluator_backend_id") != QUALITY_EVALUATOR_ID
        or quality.get("terminal") is not True
    ):
        raise RuntimeError("t1_xyz terminal quality-v2 identity drifted")


def _task_quality_margin_diagnostic(backend: Any) -> dict[str, Any]:
    """Materialize bounded lane-zero state without changing replay gates."""

    materializer = getattr(backend, "materialize_task_quality_audit", None)
    if not callable(materializer):
        return {"available": False, "reason": "adapter_method_unavailable"}
    audit = materializer()

    def scalar(name: str, cast: type[int] | type[float]) -> int | float:
        values = np.asarray(audit[name]).reshape(-1)
        if values.size < 1:
            raise RuntimeError(f"task-quality audit field {name} has no lane-zero value")
        return cast(values[0])

    return {
        "available": True,
        "physics_step": scalar("physics_step", int),
        "stage_index": scalar("stage_index", int),
        "bilateral_steps": scalar("bilateral_steps", int),
        "max_bilateral_steps": scalar("max_bilateral_steps", int),
        "success": bool(scalar("success", int)),
        "terminated": bool(scalar("terminated", int)),
        "truncated": bool(scalar("truncated", int)),
        "event_mask": scalar("event_mask", int),
        "event_physics_step": np.asarray(audit["event_physics_step"])
        .reshape(-1)
        .astype(np.int64)
        .tolist(),
        "quality_physics_sample_count": scalar(
            "quality_physics_sample_count", int
        ),
        "quality_has_post_hold_sample": bool(
            scalar("quality_has_post_hold_sample", int)
        ),
        "quality_maximum_lift_clearance_m": scalar(
            "quality_maximum_lift_clearance_m", float
        ),
        "quality_maximum_axis_error_rad": scalar(
            "quality_maximum_axis_error_rad", float
        ),
        "quality_error": scalar("quality_error", int),
        "quality_has_bilateral_hold_margin": bool(
            scalar("quality_has_bilateral_hold_margin", int)
        ),
        "quality_bilateral_hold_downstream_margin_m": scalar(
            "quality_bilateral_hold_downstream_margin_m", float
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.image_size < 224:
        raise ValueError("--image-size must be at least 224 for policy/review RGB evidence")
    if args.device_ordinal < 0:
        raise ValueError("--device-ordinal must be nonnegative")
    if not args.expected_gpu_uuid.strip():
        raise ValueError("--expected-gpu-uuid must be non-empty")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    tape_output = args.tape_output or args.output.with_name(
        f"{args.output.stem}.tape.npz"
    )
    visual_gif = args.visual_gif or args.output.with_name(
        f"{args.output.stem}.scene-wrist.gif"
    )
    if tape_output.suffix.lower() != ".npz":
        raise ValueError("--tape-output must use the .npz suffix")
    if tape_output.exists() or visual_gif.exists():
        raise FileExistsError("review evidence output paths must be new")
    replay_failure_output = args.output.with_name(
        f"{args.output.stem}.semantic-replay-failure.json"
    )
    if replay_failure_output.exists():
        raise FileExistsError(
            f"refusing to overwrite semantic replay failure evidence: {replay_failure_output}"
        )

    # All host/source/export checks complete before any CUDA backend exists.
    manifest = load_frozen_manifest(
        args.manifest,
        expected_phase=args.phase,
        verify_exports=True,
    )
    row = manifest.row(args.row_index)
    repositories = validate_repository_tuple(
        manifest,
        research_root=args.research_source_root,
        se3_root=args.se3_source_root,
        mjwarp_root=args.mjwarp_source_root,
        rlinf_root=args.rlinf_source_root,
        dynamic_root=args.dynamic_source_root,
    )
    request, export_identity = preflight_export_request(manifest, row)
    expected_request = dict(row["request"])
    if request_identity(request) != expected_request:
        raise RuntimeError("preflight ResetRequest identity drifted")
    if Path(__file__).resolve().parents[2] != args.rlinf_source_root.resolve(
        strict=True
    ):
        raise RuntimeError(
            "strict row runner is not executing from the sealed RLinf root"
        )

    import se3_wam
    from se3_wam.benchmark.config import load_task_config
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    _validate_module_root(se3_wam, args.se3_source_root, "SE3-WAM")

    from rlinf.envs.dynamic_benchmark.gpu_backend import (
        GpuNativeBackendEnv,
        assert_terminal_ledger_exact_once,
    )

    task_config = load_task_config(TASK_ID)
    horizon = int(task_config["clock"]["horizon_steps"])
    if horizon != EXECUTION_CONTRACT["horizon_control_steps"]:
        raise RuntimeError("t1_xyz task horizon differs from the sealed contract")
    teacher, preparation = make_privileged_teacher(TASK_ID, request=request)
    teacher.reset()
    se3_identity = manifest.payload["repositories"]["se3_wam"]
    backend = GpuNativeBackendEnv(
        task_id=TASK_ID,
        num_envs=1,
        export_dir=str(manifest.export_dir(row)),
        device_ordinal=args.device_ordinal,
        image_size=args.image_size,
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_se3_source_commit=se3_identity["commit"],
        expected_se3_source_tree=se3_identity["tree"],
        task_quality_schema_version=QUALITY_SCHEMA_VERSION,
        task_quality_evaluator_backend_id=QUALITY_EVALUATOR_ID,
        observation_track=EXECUTION_CONTRACT["observation_track"],
        require_exact_export_identity=True,
        render_observations=True,
    )
    try:
        if (
            getattr(backend.observation_track, "value", None) != "state"
            or backend.require_exact_export_identity is not True
            or backend.render_observations is not True
        ):
            raise RuntimeError("CUDA backend did not establish STATE plus review mode")
        provenance = _provenance_payload(backend.provenance)
        _validate_provenance(
            provenance,
            expected_gpu_uuid=args.expected_gpu_uuid,
            expected_commit=se3_identity["commit"],
            expected_tree=se3_identity["tree"],
        )
        frozen_requests = backend.frozen_requests
        if (
            len(frozen_requests) != 1
            or request_identity(frozen_requests[0]) != expected_request
        ):
            raise RuntimeError(
                "CUDA backend request differs from the sealed manifest row"
            )
        raw_observations = tuple(backend.reset((request,)))
        if (
            len(raw_observations) != 1
            or raw_observations[0].episode_id != request.episode_id
            or raw_observations[0].task_id != TASK_ID
        ):
            raise RuntimeError("CUDA reset changed the sealed row identity")
        observation, review = _state_and_review_observation(raw_observations[0])
        observations = [observation]
        reviews = [review]
        commands: list[Any] = []
        outcomes: list[tuple[bool, bool, bool, str | None]] = []
        latencies_s: list[float] = []
        result = None
        for _ in range(horizon):
            started = time.perf_counter()
            planner_action = teacher.act(observation)
            latencies_s.append(time.perf_counter() - started)
            command = _make_command(
                request,
                planner_action.values,
                observation.policy_step,
            )
            result = backend.step((command,))[0]
            observation, review = _state_and_review_observation(result.observation)
            commands.append(command)
            observations.append(observation)
            reviews.append(review)
            outcomes.append(
                (
                    bool(result.terminated),
                    bool(result.truncated),
                    bool(result.success),
                    result.termination_reason,
                )
            )
            if result.terminated or result.truncated:
                break
        if result is None or not (result.terminated or result.truncated):
            raise RuntimeError("t1_xyz row did not reach natural termination")
        terminal_ledger = backend.last_terminal_ledger
        if terminal_ledger is None or len(terminal_ledger.rows) != 1:
            raise RuntimeError("t1_xyz row lacks one exact terminal ledger row")
        terminal_payload = _ledger_payload(terminal_ledger)
        _validate_terminal_quality(
            terminal_payload[0],
            episode_id=str(request.episode_id),
        )
        primary_task_quality_margin = _task_quality_margin_diagnostic(backend)
        primary_exact_once = assert_terminal_ledger_exact_once(
            backend,
            terminal_ledger.rows,
        )
        replay = _replay(
            backend=backend,
            request=request,
            observations=tuple(observations),
            reviews=tuple(reviews),
            commands=tuple(commands),
            outcomes=tuple(outcomes),
            terminal_ledger=terminal_ledger,
        )
        if replay.get("passed") is not True:
            action_tape = [_command_payload(command) for command in commands]
            failure_payload = {
                "schema_version": "gpu-planner-t1-xyz-semantic-replay-failure-v1",
                "status": "blocked_semantic_fresh_replay",
                "evidence_passed": False,
                "qualification_completed": 0,
                "review_candidate": False,
                "countable_result_written": False,
                "task_id": TASK_ID,
                "phase": args.phase,
                "manifest": {
                    "candidate_index": row["candidate_index"],
                    "episode_id": row["request"]["episode_id"],
                    "manifest_index": row["manifest_index"],
                    "manifest_sha256": manifest.manifest_sha256,
                    "source_identity_sha256": manifest.source_identity_sha256,
                },
                "reset_request": expected_request,
                "source_gate": {
                    "passed": True,
                    "repositories_exact": True,
                    "source_identity_sha256": manifest.source_identity_sha256,
                    "repositories": repositories,
                },
                "export": export_identity,
                "provenance": provenance,
                "primary": {
                    "action_count": len(action_tape),
                    "action_tape_sha256": _json_sha256(action_tape),
                    "observation_tape_sha256": _json_sha256(
                        [_observation_digest(value) for value in observations]
                    ),
                    "review_tape_sha256": _json_sha256(
                        [_review_digest(value) for value in reviews]
                    ),
                    "outcomes_sha256": _json_sha256(outcomes),
                    "terminal_ledger_sha256": _json_sha256(terminal_payload),
                    "terminal_ledger": terminal_payload,
                    "task_quality_margin_diagnostic": primary_task_quality_margin,
                },
                "task_quality_thresholds": {
                    "bilateral_contact_hold_s": float(
                        task_config["capture"]["bilateral_contact_hold_s"]
                    ),
                    "clearance_m": float(task_config["capture"]["clearance_m"]),
                    "stable_dwell_s": float(task_config["capture"]["stable_dwell_s"]),
                },
                "replay": replay,
            }
            _write_json_exclusive(replay_failure_output, failure_payload)
            raise RuntimeError(
                "semantic fresh replay failed; fail-closed evidence written to "
                f"{replay_failure_output.resolve()}: "
                f"{json.dumps(replay, sort_keys=True)}"
            )

        tape_output = tape_output.resolve()
        visual_gif = visual_gif.resolve()
        _write_tape_npz(tape_output, observations, commands)
        _write_visual_gif(visual_gif, reviews)
        action_tape = [_command_payload(command) for command in commands]
        trajectory_tape = [_observation_digest(value) for value in observations]
        evidence_export = {
            "passed": True,
            "action_tape_sha256": _json_sha256(action_tape),
            "trajectory_tape_sha256": _json_sha256(trajectory_tape),
            "tape_file": str(tape_output),
            "tape_file_sha256": _file_sha256(tape_output),
            "visual_file": str(visual_gif),
            "visual_sha256": _file_sha256(visual_gif),
        }
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "completed_review_evidence",
            "evidence_passed": True,
            "task_id": TASK_ID,
            "backend_id": BACKEND_ID,
            "phase": args.phase,
            "manifest_index": int(row["manifest_index"]),
            "manifest": {
                "candidate_index": row["candidate_index"],
                "episode_id": row["request"]["episode_id"],
                "manifest_index": row["manifest_index"],
                "manifest_sha256": manifest.manifest_sha256,
                "source_identity_sha256": manifest.source_identity_sha256,
            },
            "reset_request": expected_request,
            "online_planner": True,
            "planner_observation_source": EXECUTION_CONTRACT[
                "planner_observation_source"
            ],
            "planner_observation_track": EXECUTION_CONTRACT["observation_track"],
            "review_materialization": EXECUTION_CONTRACT["review_materialization"],
            "frozen_action_replay": False,
            "cpu_physics_or_env_fallback": False,
            "quality": {
                "schema_version": QUALITY_SCHEMA_VERSION,
                "evaluator_backend_id": QUALITY_EVALUATOR_ID,
            },
            "source_gate": {
                "passed": True,
                "repositories_exact": True,
                "source_identity_sha256": manifest.source_identity_sha256,
                "repositories": repositories,
            },
            "export": export_identity,
            "provenance": provenance,
            "success": bool(result.success),
            "termination_reason": result.termination_reason,
            "control_steps": len(commands),
            "physics_steps": int(observations[-1].physics_step),
            "terminal_ledger_gate": {
                "passed": True,
                "exact_once_second_consumption_rejected": True,
                "exact_once_negative_witnesses": list(primary_exact_once),
            },
            "terminal_ledger": terminal_payload,
            "replay": replay,
            "action_tape": action_tape,
            "trajectory_tape": trajectory_tape,
            "evidence_export": evidence_export,
            "teacher_preparation": preparation,
            "planner_latency_s": {
                "count": len(latencies_s),
                "max": max(latencies_s),
                "mean": float(np.mean(latencies_s)),
            },
        }
        validate_result_for_row(payload, manifest=manifest, row=row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "manifest_index": payload["manifest_index"],
                    "success": payload["success"],
                },
                sort_keys=True,
            )
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
