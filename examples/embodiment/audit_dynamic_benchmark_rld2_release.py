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

"""Build and independently audit the exact-14 RLD2 trajectory release.

The per-task trajectory auditor establishes that one export is internally
sound.  This program supplies the deliberately separate release boundary: it
requires exactly the frozen fourteen tasks, binds every new RLD2 data root to
its candidate/input records and independent audit, rejects mixed execution or
utility contracts, and recomputes the published source labels and paired
planner deltas from the sealed raw records.

``release_manifest.json`` is intentionally not self-authorizing.  It is first
written with ``release_eligible=false`` and sealed by ``SHA256SUMS``.  A second
pass reopens every referenced input and writes ``release_audit.json``.  Only
that independent audit payload can carry ``release_eligible=true``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from examples.embodiment import (
        audit_dynamic_benchmark_optimal_trajectories as _optimal_auditor,
    )
except ModuleNotFoundError:
    import audit_dynamic_benchmark_optimal_trajectories as _optimal_auditor

EXACT_TASKS = (
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

LEGACY_RELEASE_INPUT_SCHEMA = "rlinf-dynamic-benchmark-rld2-release-inputs-v0.1"
RELEASE_INPUT_SCHEMA = "rlinf-dynamic-benchmark-rld2-release-inputs-v0.2"
HISTORICAL_RELEASE_MANIFEST_SCHEMAS = (
    "rlinf-dynamic-benchmark-rld2-release-v0.1",
    "rlinf-dynamic-benchmark-rld2-release-v0.2",
)
RELEASE_MANIFEST_SCHEMA = "rlinf-dynamic-benchmark-rld2-release-v0.3"
HISTORICAL_RELEASE_AUDIT_SCHEMAS = (
    "rlinf-dynamic-benchmark-rld2-release-audit-v0.1",
    "rlinf-dynamic-benchmark-rld2-release-audit-v0.2",
)
RELEASE_AUDIT_SCHEMA = "rlinf-dynamic-benchmark-rld2-release-audit-v0.3"
QUALITY_V4_RELEASE_READINESS_SCHEMA = (
    "rlinf-dynamic-benchmark-rld2-quality-v4-release-readiness-v0.1"
)
CANDIDATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-candidates-v0.2"
CANDIDATE_RELEASE_SCHEMA = "rlinf-dynamic-benchmark-rld2-candidate-release-v0.2"
EVALUATOR_IDENTITY_SCHEMA = "rlinf-dynamic-benchmark-quality-evaluator-identity-v0.1"
COMPATIBILITY_EVIDENCE_SCHEMA = (
    "rlinf-dynamic-benchmark-checkpoint-compatibility-evidence-v0.1"
)
CALIBRATION_EVIDENCE_SCHEMA = (
    "rlinf-dynamic-benchmark-planner-calibration-evidence-v0.1"
)
INPUT_INVENTORY_SCHEMA = "rlinf-dynamic-benchmark-rld2-input-inventory-v0.1"
DATASET_CARD_SCHEMA = "rlinf-dynamic-benchmark-optimal-export-v0.1"
TASK_AUDIT_SCHEMA = "rlinf-dynamic-benchmark-optimal-audit-v0.1"
PLANNER_DOMINANCE_SCHEMA = _optimal_auditor.PLANNER_DOMINANCE_SCHEMA
ATTEMPT_SCHEMA = _optimal_auditor.ATTEMPT_SCHEMA
QUALITY_V2_THRESHOLDS_SCHEMA = _optimal_auditor.QUALITY_V2_THRESHOLDS_SCHEMA
QUALITY_V2_SUMMARY_SCHEMA = _optimal_auditor.QUALITY_V2_SUMMARY_SCHEMA
QUALITY_V2_GATE_SCHEMA = _optimal_auditor.QUALITY_V2_GATE_SCHEMA
QUALITY_V2_DOMINANCE_SCHEMA = _optimal_auditor.QUALITY_V2_DOMINANCE_SCHEMA
T5_ACTION_HISTORY_SCHEMA = _optimal_auditor.T5_ACTION_HISTORY_SCHEMA
FULL_POOL_SEARCH_MODE = "full-pool"
PLANNER_PARETO_SELECTION_MODE = "planner-pareto"
PLANNER_PARETO_SELECTION_CONTRACT = _optimal_auditor.PLANNER_PARETO_SELECTION_CONTRACT
ACCEPTED_PER_TASK = 100
RELEASE_ID = "RLD2"

_TASK_INPUT_KEYS = {
    "task",
    "dataset_root",
    "dataset_card_sha256",
    "checksums_sha256",
    "candidate_manifest_sha256",
    "audit_path",
    "audit_sha256",
    "input_inventory_path",
    "input_inventory_sha256",
    "quality_v2_thresholds_path",
    "quality_v2_thresholds_sha256",
}
_CANDIDATE_RELEASE_KEYS = {
    "schema_version",
    "release_id",
    "candidate_schema_version",
    "evaluator_identity",
    "policy_rlinf_commits",
    "policy_benchmark_commits",
    "evaluator_evidence",
    "calibration_evidence",
    "tasks",
    "task_manifest_sha256",
    "candidate_count",
    "deduplicated",
    "input_spec_sha256",
    "input_inventory_sha256",
    "inputs_sha256_sha256",
    "production_validated",
    "payload_sha256",
}
_COMMON_METRICS = (
    "trajectory_completion",
    "completion_time_s",
    "control_steps",
    "action_l2_sum",
)
_T5_CAUSAL_LATENCY_METRIC = (
    "t5_replan.impact_end_to_first_qualifying_applied_correction_s"
)


class ReleaseAuditError(ValueError):
    """Raised when an RLD2 release invariant fails closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReleaseAuditError(f"{label} must be a full lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReleaseAuditError(f"{label} must be a full lowercase Git commit")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseAuditError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseAuditError(f"{label} must be a JSON mapping")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReleaseAuditError(f"cannot read {label}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ReleaseAuditError(f"{label}:{line_number} is blank")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReleaseAuditError(f"{label}:{line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ReleaseAuditError(f"{label}:{line_number} is not a mapping")
        rows.append(row)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _rld2_path(value: Any, label: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseAuditError(f"{label} path is missing")
    path = Path(value).resolve()
    folded = {part.casefold() for part in path.parts}
    if "rld1" in folded or "rld2" not in folded:
        raise ReleaseAuditError(f"{label} is not a new RLD2 path: {path}")
    if directory and not path.is_dir():
        raise ReleaseAuditError(f"{label} directory is missing: {path}")
    if not directory and not path.is_file():
        raise ReleaseAuditError(f"{label} file is missing: {path}")
    return path


def _safe_root_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ReleaseAuditError(f"unsafe root-relative path {relative!r}")
    path = root.joinpath(*pure.parts)
    if path.is_symlink():
        raise ReleaseAuditError(
            f"RLD2 roots must not contain symlinked files: {relative}"
        )
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ReleaseAuditError(f"root-relative path escapes RLD2 root: {relative!r}")
    return path


def _verify_root_checksums(root: Path, expected_sha256: str) -> int:
    checksum_path = root / "SHA256SUMS"
    if _sha256(checksum_path) != expected_sha256:
        raise ReleaseAuditError(f"{root.name} SHA256SUMS identity mismatch")
    declared: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ReleaseAuditError(
                f"{root.name} SHA256SUMS:{line_number} is malformed"
            )
        digest = _require_sha256(parts[0], f"SHA256SUMS:{line_number}")
        relative = parts[1]
        if relative in declared:
            raise ReleaseAuditError(f"{root.name} SHA256SUMS duplicates {relative!r}")
        path = _safe_root_file(root, relative)
        if not path.is_file() or path.name == "SHA256SUMS":
            raise ReleaseAuditError(
                f"{root.name} checksum target is missing: {relative!r}"
            )
        if _sha256(path) != digest:
            raise ReleaseAuditError(f"{root.name} file hash tamper: {relative!r}")
        declared[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and ".staging" not in path.parts
    }
    if actual != set(declared):
        raise ReleaseAuditError(
            f"{root.name} checksum inventory mismatch: "
            f"missing={sorted(actual - set(declared))}, "
            f"extra={sorted(set(declared) - actual)}"
        )
    return len(declared)


def _candidate_semantics(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("kind") == "planner":
        return {"kind": "planner"}
    if candidate.get("kind") != "policy":
        raise ReleaseAuditError("candidate kind must be planner or policy")
    residual = candidate.get("residual_scale")
    if residual is not None:
        residual = _number(residual, "candidate residual_scale")
        if not math.isfinite(residual) or not 0.0 < residual <= 1.0:
            raise ReleaseAuditError("candidate residual_scale is invalid")
    stochastic = candidate.get("stochastic", False)
    offset = candidate.get("exploration_seed_offset", 0)
    if (
        not isinstance(stochastic, bool)
        or isinstance(offset, bool)
        or not isinstance(offset, int)
    ):
        raise ReleaseAuditError("candidate rollout semantics are invalid")
    return {
        "kind": "policy",
        "policy_sha256": _require_sha256(
            candidate.get("policy_sha256"), "candidate policy SHA-256"
        ),
        "stochastic": stochastic,
        "exploration_seed_offset": offset,
        "residual_scale": residual,
    }


def _metric_contract(value: Any, *, metric_name: str, direction: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseAuditError(f"planner metric {metric_name!r} must be a mapping")
    common = {"direction", "max_observed_replay_drift", "scientific_resolution"}
    expected = (
        common | {"numeric_floor_absolute", "numeric_floor_relative"}
        if metric_name == "action_l2_sum"
        else common | {"numeric_floor"}
    )
    if set(value) != expected or value.get("direction") != direction:
        raise ReleaseAuditError(f"planner metric {metric_name!r} contract mismatch")
    drift = _number(
        value["max_observed_replay_drift"],
        f"planner metric {metric_name!r} max_observed_replay_drift",
    )
    resolution = _number(
        value["scientific_resolution"],
        f"planner metric {metric_name!r} scientific_resolution",
    )
    if not math.isfinite(drift) or drift < 0.0:
        raise ReleaseAuditError(f"planner metric {metric_name!r} drift is invalid")
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ReleaseAuditError(f"planner metric {metric_name!r} resolution is invalid")
    normalized = {
        "direction": direction,
        "max_observed_replay_drift": drift,
        "scientific_resolution": resolution,
    }
    if metric_name == "action_l2_sum":
        absolute = _number(
            value["numeric_floor_absolute"], "action_l2_sum numeric_floor_absolute"
        )
        relative = _number(
            value["numeric_floor_relative"], "action_l2_sum numeric_floor_relative"
        )
        if absolute != 1.0e-6 or relative != 1.0e-6:
            raise ReleaseAuditError("action_l2_sum numeric floor contract mismatch")
        normalized.update(
            numeric_floor_absolute=absolute,
            numeric_floor_relative=relative,
        )
    else:
        floor = _number(value["numeric_floor"], f"planner metric {metric_name!r} floor")
        expected_floor = 0.0 if metric_name == "control_steps" else 1.0e-6
        if floor != expected_floor:
            raise ReleaseAuditError(f"planner metric {metric_name!r} floor mismatch")
        if metric_name == "control_steps" and resolution < 1.0:
            raise ReleaseAuditError(
                "control_steps scientific resolution must be at least one"
            )
        if metric_name == "completion_time_s" and resolution != 0.002:
            raise ReleaseAuditError(
                "completion_time_s scientific resolution must be one 0.002 s physics step"
            )
        normalized["numeric_floor"] = floor
    return normalized


def _normalize_contract(
    candidate_payload: Mapping[str, Any], task: str
) -> dict[str, Any]:
    """Independently normalize the frozen utility and calibration contract."""

    raw = candidate_payload.get("planner_dominance")
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "task",
        "backend_id",
        "quality_schema",
        "calibration",
        "metrics",
        "tie_break_order",
    }:
        raise ReleaseAuditError(f"{task} planner-dominance field inventory mismatch")
    if raw.get("schema_version") != PLANNER_DOMINANCE_SCHEMA or raw.get("task") != task:
        raise ReleaseAuditError(f"{task} planner-dominance schema/task mismatch")
    backend_id = raw.get("backend_id")
    if (
        not isinstance(backend_id, str)
        or not backend_id
        or backend_id.strip() != backend_id
    ):
        raise ReleaseAuditError(f"{task} planner backend identity is missing")
    quality = raw.get("quality_schema")
    if not isinstance(quality, Mapping) or set(quality) != {
        "schema_version",
        "task_id",
        "task_config_sha256",
        "components",
        "schema_sha256",
    }:
        raise ReleaseAuditError(f"{task} quality schema inventory mismatch")
    quality_version = quality.get("schema_version")
    if not isinstance(quality_version, str) or not quality_version:
        raise ReleaseAuditError(f"{task} quality schema version is missing")
    if quality.get("task_id") != task:
        raise ReleaseAuditError(f"{task} quality schema task mismatch")
    task_config_sha256 = _require_sha256(
        quality.get("task_config_sha256"), f"{task} task config SHA-256"
    )
    quality_sha256 = _require_sha256(
        quality.get("schema_sha256"), f"{task} quality schema SHA-256"
    )
    components = quality.get("components")
    if not isinstance(components, list) or not components:
        raise ReleaseAuditError(f"{task} quality component inventory is empty")
    normalized_components: list[dict[str, Any]] = []
    component_names: set[str] = set()
    for component in components:
        if not isinstance(component, Mapping) or set(component) != {
            "name",
            "direction",
            "unit",
            "scientific_resolution",
            "reducer",
            "source",
            "description",
        }:
            raise ReleaseAuditError(f"{task} quality component metadata is invalid")
        name = component.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or "." in name
            or name in component_names
        ):
            raise ReleaseAuditError(f"{task} quality component name is invalid")
        if component.get("direction") not in {"minimize", "maximize"}:
            raise ReleaseAuditError(f"{task}/{name} quality direction is invalid")
        if component.get("reducer") not in {"minimum", "maximum", "terminal"}:
            raise ReleaseAuditError(f"{task}/{name} quality reducer is invalid")
        for key in ("unit", "source", "description"):
            value = component.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ReleaseAuditError(f"{task}/{name} quality {key} is missing")
        resolution = _number(
            component.get("scientific_resolution"), f"{task}/{name} quality resolution"
        )
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ReleaseAuditError(f"{task}/{name} quality resolution is invalid")
        component_names.add(name)
        normalized_components.append(dict(component))
    if quality_sha256 != _payload_sha256(
        {
            "schema_version": quality_version,
            "task_id": task,
            "task_config_sha256": task_config_sha256,
            "components": list(components),
        }
    ):
        raise ReleaseAuditError(f"{task} quality schema SHA-256 does not recompute")
    normalized_quality = {
        "schema_version": quality_version,
        "task_id": task,
        "task_config_sha256": task_config_sha256,
        "components": normalized_components,
        "schema_sha256": quality_sha256,
    }
    calibration = raw.get("calibration")
    if not isinstance(calibration, Mapping) or set(calibration) != {
        "replay_count",
        "reset_episode_id",
        "reset_manifest_sha256",
        "evidence_path",
        "evidence_sha256",
    }:
        raise ReleaseAuditError(f"{task} planner calibration inventory mismatch")
    replay_count = calibration.get("replay_count")
    reset_episode_id = calibration.get("reset_episode_id")
    if (
        isinstance(replay_count, bool)
        or not isinstance(replay_count, int)
        or replay_count < 3
    ):
        raise ReleaseAuditError(
            f"{task} planner calibration requires at least three replays"
        )
    if not isinstance(reset_episode_id, str) or not reset_episode_id.strip():
        raise ReleaseAuditError(f"{task} planner calibration reset identity is missing")
    evidence_path = calibration.get("evidence_path")
    if not isinstance(evidence_path, str) or not evidence_path.strip():
        raise ReleaseAuditError(f"{task} planner calibration evidence path is missing")
    normalized_calibration = {
        "replay_count": replay_count,
        "reset_episode_id": reset_episode_id,
        "reset_manifest_sha256": _require_sha256(
            calibration.get("reset_manifest_sha256"),
            f"{task} calibration reset manifest",
        ),
        "evidence_path": evidence_path,
        "evidence_sha256": _require_sha256(
            calibration.get("evidence_sha256"), f"{task} calibration evidence"
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
        raise ReleaseAuditError(f"{task} planner metric inventory mismatch")
    quality_metrics = metrics.get("task_quality")
    component_index = {
        component["name"]: component for component in normalized_components
    }
    if not isinstance(quality_metrics, Mapping) or set(quality_metrics) != set(
        component_index
    ):
        raise ReleaseAuditError(f"{task} quality metric mapping is incomplete")
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
            metrics["control_steps"], metric_name="control_steps", direction="min"
        ),
        "action_l2_sum": _metric_contract(
            metrics["action_l2_sum"], metric_name="action_l2_sum", direction="min"
        ),
    }
    for name, component in component_index.items():
        if (
            normalized_metrics["task_quality"][name]["scientific_resolution"]
            != component["scientific_resolution"]
        ):
            raise ReleaseAuditError(
                f"{task}/{name} calibration resolution differs from schema"
            )
    metric_names = [
        "trajectory_completion",
        *(f"task_quality.{name}" for name in component_index),
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    ]
    tie_break_order = raw.get("tie_break_order")
    if (
        not isinstance(tie_break_order, list)
        or len(tie_break_order) != len(metric_names)
        or set(tie_break_order) != set(metric_names)
    ):
        raise ReleaseAuditError(f"{task} tie-break order is incomplete")
    normalized = {
        "schema_version": PLANNER_DOMINANCE_SCHEMA,
        "task": task,
        "backend_id": backend_id,
        "quality_schema": normalized_quality,
        "calibration": normalized_calibration,
        "metrics": normalized_metrics,
        "tie_break_order": list(tie_break_order),
    }
    normalized["payload_sha256"] = _payload_sha256(normalized)
    return normalized


def _evidence_path(
    candidate_path: Path,
    value: Any,
    label: str,
    *,
    release_root: Path,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseAuditError(f"{label} evidence path is missing")
    path = Path(value)
    if path.is_absolute():
        raise ReleaseAuditError(f"{label} evidence path must be portable and relative")
    path = (candidate_path.parent / path).resolve()
    root = release_root.resolve()
    if path != root and root not in path.parents:
        raise ReleaseAuditError(f"{label} evidence path escapes candidate release root")
    if not path.is_file():
        raise ReleaseAuditError(f"{label} evidence file is missing: {path}")
    return path


def _compatibility_inventory_sha256(probes: Sequence[Mapping[str, Any]]) -> str:
    """Hash the frozen policy inventory represented by compatibility probes."""

    projected = [
        {
            "task": probe["task"],
            "policy_sha256": probe["policy_sha256"],
            "policy_rlinf_commit": probe["policy_rlinf_commit"],
            "policy_benchmark_commit": probe["policy_benchmark_commit"],
            "policy_state_schema_sha256": probe["policy_state_schema_sha256"],
            "policy_state_dim": probe["policy_state_dim"],
            "policy_mask_dim": probe["policy_mask_dim"],
        }
        for probe in probes
    ]
    return hashlib.sha256(_canonical_json(projected).encode("utf-8")).hexdigest()


def _validate_compatibility_evidence(
    path: Path,
    *,
    policy_benchmark_commit: str,
    evaluator_identity: Mapping[str, Any],
    expected_inventory: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Independently validate one checkpoint-compatibility proof."""

    evidence = _load_json(path, "benchmark compatibility evidence")
    expected_top_keys = {
        "schema_version",
        "policy_benchmark_commit",
        "evaluator_rlinf_commit",
        "evaluator_benchmark_commit",
        "backend_id",
        "split",
        "test_exposure",
        "probe_count",
        "policy_inventory_sha256",
        "probes",
        "payload_sha256",
    }
    if set(evidence) != expected_top_keys:
        raise ReleaseAuditError("compatibility evidence field inventory mismatch")
    if (
        evidence.get("schema_version") != COMPATIBILITY_EVIDENCE_SCHEMA
        or evidence.get("policy_benchmark_commit") != policy_benchmark_commit
        or evidence.get("evaluator_rlinf_commit")
        != evaluator_identity["evaluator_rlinf_commit"]
        or evidence.get("evaluator_benchmark_commit")
        != evaluator_identity["evaluator_benchmark_commit"]
        or evidence.get("backend_id") != evaluator_identity["backend_id"]
        or evidence.get("split") not in {"train", "validation"}
        or evidence.get("test_exposure") != {"test_id": False, "test_ood": False}
        or evidence.get("payload_sha256") != _payload_sha256(evidence)
    ):
        raise ReleaseAuditError(
            "compatibility evidence identity, split, or payload mismatch"
        )
    _require_commit(policy_benchmark_commit, "compatibility policy benchmark commit")
    _require_commit(
        evidence.get("evaluator_rlinf_commit"),
        "compatibility evaluator RLinf commit",
    )
    _require_commit(
        evidence.get("evaluator_benchmark_commit"),
        "compatibility evaluator benchmark commit",
    )
    probes = evidence.get("probes")
    probe_count = evidence.get("probe_count")
    if (
        isinstance(probe_count, bool)
        or not isinstance(probe_count, int)
        or probe_count < 1
        or not isinstance(probes, list)
        or len(probes) != probe_count
    ):
        raise ReleaseAuditError("compatibility evidence probe count mismatch")
    expected_probe_keys = {
        "task",
        "policy_sha256",
        "policy_rlinf_commit",
        "policy_state_schema_sha256",
        "policy_state_dim",
        "policy_mask_dim",
        "evaluator_state_schema_sha256",
        "evaluator_state_dim",
        "evaluator_mask_dim",
        "policy_action_dim",
        "evaluator_action_dim",
        "evaluator_task_config_sha256",
        "environment_instance_id",
        "episode_id",
        "reset_request_sha256",
        "observation_sha256",
        "action_sha256",
        "load_success",
        "reset_success",
        "inference_success",
        "step_success",
        "finite_observation",
        "finite_action",
        "finite_reward",
    }
    normalized_probes: list[dict[str, Any]] = []
    previous_key: tuple[str, str] | None = None
    for index, raw_probe in enumerate(probes):
        if not isinstance(raw_probe, Mapping) or set(raw_probe) != expected_probe_keys:
            raise ReleaseAuditError(
                f"compatibility probe {index} field inventory mismatch"
            )
        probe = dict(raw_probe)
        task = probe.get("task")
        if not isinstance(task, str) or task not in EXACT_TASKS:
            raise ReleaseAuditError(f"compatibility probe {index} task is invalid")
        policy_sha256 = _require_sha256(
            probe.get("policy_sha256"), f"compatibility probe {index} policy"
        )
        _require_commit(
            probe.get("policy_rlinf_commit"),
            f"compatibility probe {index} policy RLinf commit",
        )
        policy_schema = _require_sha256(
            probe.get("policy_state_schema_sha256"),
            f"compatibility probe {index} policy state schema",
        )
        evaluator_schema = _require_sha256(
            probe.get("evaluator_state_schema_sha256"),
            f"compatibility probe {index} evaluator state schema",
        )
        _require_sha256(
            probe.get("evaluator_task_config_sha256"),
            f"compatibility probe {index} evaluator task config",
        )
        for key in ("reset_request_sha256", "observation_sha256", "action_sha256"):
            _require_sha256(probe.get(key), f"compatibility probe {index} {key}")
        for key in ("environment_instance_id", "episode_id"):
            value = probe.get(key)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ReleaseAuditError(f"compatibility probe {index} {key} is invalid")
        dimensions: dict[str, int] = {}
        for key in (
            "policy_state_dim",
            "policy_mask_dim",
            "evaluator_state_dim",
            "evaluator_mask_dim",
            "policy_action_dim",
            "evaluator_action_dim",
        ):
            value = probe.get(key)
            minimum = 0 if key.endswith("mask_dim") else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ReleaseAuditError(f"compatibility probe {index} {key} is invalid")
            dimensions[key] = value
        for key in (
            "load_success",
            "reset_success",
            "inference_success",
            "step_success",
            "finite_observation",
            "finite_action",
            "finite_reward",
        ):
            if probe.get(key) is not True:
                raise ReleaseAuditError(f"compatibility probe {index} failed {key}")
        if (
            policy_schema != evaluator_schema
            or dimensions["policy_state_dim"] != dimensions["evaluator_state_dim"]
            or dimensions["policy_mask_dim"] != dimensions["evaluator_mask_dim"]
            or dimensions["policy_action_dim"] != dimensions["evaluator_action_dim"]
        ):
            raise ReleaseAuditError(
                f"compatibility probe {index} schema/dimension mismatch"
            )
        key = (task, policy_sha256)
        if previous_key is not None and key <= previous_key:
            raise ReleaseAuditError("compatibility probes must be sorted and unique")
        previous_key = key
        probe["policy_benchmark_commit"] = policy_benchmark_commit
        normalized_probes.append(probe)
    inventory_sha256 = _compatibility_inventory_sha256(normalized_probes)
    if evidence.get("policy_inventory_sha256") != inventory_sha256:
        raise ReleaseAuditError("compatibility policy inventory SHA-256 mismatch")
    if expected_inventory is not None:
        normalized_expected = [dict(row) for row in expected_inventory]
        actual_projection = [
            {
                key: probe[key]
                for key in (
                    "task",
                    "policy_sha256",
                    "policy_rlinf_commit",
                    "policy_benchmark_commit",
                    "policy_state_schema_sha256",
                    "policy_state_dim",
                    "policy_mask_dim",
                )
            }
            for probe in normalized_probes
        ]
        if actual_projection != normalized_expected:
            raise ReleaseAuditError(
                "compatibility probes do not exactly cover the candidate release policies"
            )
    return evidence


def _policy_authority(
    candidates: Sequence[Mapping[str, Any]], *, task: str
) -> tuple[list[str], list[str]]:
    policy_rlinf_commits: set[str] = set()
    policy_benchmark_commits: set[str] = set()
    for candidate in candidates:
        if candidate.get("kind") != "policy":
            continue
        provenance = candidate.get("provenance")
        source = provenance.get("source") if isinstance(provenance, Mapping) else None
        benchmark = (
            provenance.get("benchmark") if isinstance(provenance, Mapping) else None
        )
        if not isinstance(source, Mapping) or not isinstance(benchmark, Mapping):
            raise ReleaseAuditError(
                f"{task}/{candidate.get('candidate_id')} policy authority is missing"
            )
        policy_rlinf_commits.add(
            _require_commit(
                source.get("rlinf_commit"),
                f"{task}/{candidate.get('candidate_id')} policy RLinf authority",
            )
        )
        policy_benchmark_commits.add(
            _require_commit(
                benchmark.get("commit"),
                f"{task}/{candidate.get('candidate_id')} policy benchmark authority",
            )
        )
    if not policy_rlinf_commits or not policy_benchmark_commits:
        raise ReleaseAuditError(f"{task} has no policy authority inventory")
    return sorted(policy_rlinf_commits), sorted(policy_benchmark_commits)


def _compatibility_inventory_rows(
    candidates: Sequence[Mapping[str, Any]], *, task: str
) -> list[dict[str, Any]]:
    """Project unique task/checkpoint identities needed by the compatibility proof."""

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        if candidate.get("kind") != "policy":
            continue
        policy_sha256 = _require_sha256(
            candidate.get("policy_sha256"), f"{task} compatibility policy"
        )
        provenance = candidate.get("provenance")
        source = provenance.get("source") if isinstance(provenance, Mapping) else None
        benchmark = (
            provenance.get("benchmark") if isinstance(provenance, Mapping) else None
        )
        state_schema = (
            provenance.get("state_schema") if isinstance(provenance, Mapping) else None
        )
        if (
            not isinstance(source, Mapping)
            or not isinstance(benchmark, Mapping)
            or not isinstance(state_schema, Mapping)
        ):
            raise ReleaseAuditError(f"{task} compatibility provenance is incomplete")
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
            raise ReleaseAuditError(
                f"{task} compatibility state dimensions are invalid"
            )
        row = {
            "task": task,
            "policy_sha256": policy_sha256,
            "policy_rlinf_commit": _require_commit(
                source.get("rlinf_commit"), f"{task} compatibility policy RLinf commit"
            ),
            "policy_benchmark_commit": _require_commit(
                benchmark.get("commit"), f"{task} compatibility policy benchmark commit"
            ),
            "policy_state_schema_sha256": _require_sha256(
                state_schema.get("sha256"), f"{task} compatibility state schema"
            ),
            "policy_state_dim": state_dim,
            "policy_mask_dim": mask_dim,
        }
        key = (task, policy_sha256)
        previous = rows.get(key)
        if previous is not None and previous != row:
            raise ReleaseAuditError(
                f"{task}/{policy_sha256} has mixed provenance across rollout expansions"
            )
        rows[key] = row
    if not rows:
        raise ReleaseAuditError(
            f"{task} compatibility inventory has no policy checkpoints"
        )
    return [rows[key] for key in sorted(rows)]


def _validate_evaluator_identity(
    value: Any,
    *,
    candidate_path: Path,
    task: str,
    policy_benchmark_commits: Sequence[str],
    release_root: Path,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "evaluator_rlinf_commit",
        "evaluator_benchmark_commit",
        "backend_id",
        "policy_benchmark_relations",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ReleaseAuditError(f"{task} evaluator identity field inventory mismatch")
    schema_version = value.get("schema_version")
    backend_id = value.get("backend_id")
    if schema_version != EVALUATOR_IDENTITY_SCHEMA:
        raise ReleaseAuditError(f"{task} evaluator identity schema mismatch")
    if not isinstance(backend_id, str) or not backend_id.strip():
        raise ReleaseAuditError(f"{task} evaluator backend identity is missing")
    evaluator_rlinf_commit = _require_commit(
        value.get("evaluator_rlinf_commit"), f"{task} evaluator RLinf commit"
    )
    evaluator_benchmark_commit = _require_commit(
        value.get("evaluator_benchmark_commit"), f"{task} evaluator benchmark commit"
    )
    relations = value.get("policy_benchmark_relations")
    if not isinstance(relations, list):
        raise ReleaseAuditError(
            f"{task} policy/evaluator benchmark relations are missing"
        )
    commits = [
        row.get("policy_benchmark_commit") if isinstance(row, Mapping) else None
        for row in relations
    ]
    if commits != list(policy_benchmark_commits):
        raise ReleaseAuditError(
            f"{task} ordered policy/evaluator benchmark relation inventory mismatch"
        )
    normalized_relations: list[dict[str, Any]] = []
    for row in relations:
        if not isinstance(row, Mapping) or set(row) != {
            "policy_benchmark_commit",
            "relation",
            "evidence_path",
            "evidence_sha256",
        }:
            raise ReleaseAuditError(
                f"{task} benchmark relation field inventory mismatch"
            )
        policy_commit = _require_commit(
            row.get("policy_benchmark_commit"), f"{task} policy benchmark relation"
        )
        relation = row.get("relation")
        if relation == "identical":
            if (
                policy_commit != evaluator_benchmark_commit
                or row.get("evidence_path") is not None
                or row.get("evidence_sha256") is not None
            ):
                raise ReleaseAuditError(
                    f"{task} identical benchmark relation is invalid"
                )
            evidence_path = None
            evidence_sha256 = None
        elif relation == "checkpoint-compatible":
            if policy_commit == evaluator_benchmark_commit:
                raise ReleaseAuditError(
                    f"{task} identical benchmark was mislabeled checkpoint-compatible"
                )
            evidence_sha256 = _require_sha256(
                row.get("evidence_sha256"), f"{task} compatibility evidence"
            )
            evidence = _evidence_path(
                candidate_path,
                row.get("evidence_path"),
                f"{task}/{policy_commit}",
                release_root=release_root,
            )
            if _sha256(evidence) != evidence_sha256:
                raise ReleaseAuditError(f"{task} compatibility evidence hash tamper")
            _validate_compatibility_evidence(
                evidence,
                policy_benchmark_commit=policy_commit,
                evaluator_identity=value,
            )
            evidence_path = str(evidence)
        else:
            raise ReleaseAuditError(
                f"{task} benchmark relation {relation!r} is unsupported"
            )
        normalized_relations.append(
            {
                "policy_benchmark_commit": policy_commit,
                "relation": relation,
                "evidence_path": evidence_path,
                "evidence_sha256": evidence_sha256,
            }
        )
    return {
        "schema_version": schema_version,
        "evaluator_rlinf_commit": evaluator_rlinf_commit,
        "evaluator_benchmark_commit": evaluator_benchmark_commit,
        "backend_id": backend_id,
        "policy_benchmark_relations": normalized_relations,
    }


def _validate_candidate_manifest(
    path: Path,
    *,
    task: str,
    expected_sha256: str,
    evidence_manifest_path: Path,
    candidate_release_root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[str],
]:
    if _sha256(path) != expected_sha256:
        raise ReleaseAuditError(f"{task} candidate manifest hash mismatch")
    payload = _load_json(path, f"{task} candidate manifest")
    if set(payload) != {
        "schema_version",
        "task",
        "evaluator_identity",
        "policy_rlinf_commits",
        "policy_benchmark_commits",
        "planner_dominance",
        "candidates",
    }:
        raise ReleaseAuditError(f"{task} production candidate field inventory mismatch")
    if payload.get("schema_version") != CANDIDATE_SCHEMA or payload.get("task") != task:
        raise ReleaseAuditError(f"{task} candidate schema/task mismatch")
    if "rlinf_commit" in payload or "benchmark_commit" in payload:
        raise ReleaseAuditError(
            f"{task} singular top-level policy identity is forbidden in production v0.2"
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ReleaseAuditError(
            f"{task} frozen pool must contain planner and RL candidates"
        )
    if candidates[0].get("kind") != "planner":
        raise ReleaseAuditError(f"{task} planner must be candidate index zero")
    ids: list[str] = []
    semantics: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ReleaseAuditError(f"{task} candidate {index} is not a mapping")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ReleaseAuditError(f"{task} candidate {index} ID is missing")
        if not isinstance(candidate.get("provenance"), Mapping):
            raise ReleaseAuditError(
                f"{task} candidate {candidate_id} provenance is missing"
            )
        ids.append(candidate_id)
        semantics.append(_canonical_json(_candidate_semantics(candidate)))
    if len(ids) != len(set(ids)) or len(semantics) != len(set(semantics)):
        raise ReleaseAuditError(
            f"{task} candidate IDs or rollout semantics are duplicated"
        )
    if sum(candidate.get("kind") == "planner" for candidate in candidates) != 1:
        raise ReleaseAuditError(f"{task} must contain exactly one planner")
    policy_rlinf_commits, policy_benchmark_commits = _policy_authority(
        candidates, task=task
    )
    if payload.get("policy_rlinf_commits") != policy_rlinf_commits:
        raise ReleaseAuditError(
            f"{task} top-level policy RLinf authority is not the sorted candidate-derived set"
        )
    if payload.get("policy_benchmark_commits") != policy_benchmark_commits:
        raise ReleaseAuditError(
            f"{task} top-level policy benchmark authority is not the sorted candidate-derived set"
        )
    evaluator_identity = _validate_evaluator_identity(
        payload.get("evaluator_identity"),
        candidate_path=evidence_manifest_path,
        task=task,
        policy_benchmark_commits=policy_benchmark_commits,
        release_root=candidate_release_root,
    )
    contract = _normalize_contract(payload, task)
    if contract.get("backend_id") != evaluator_identity["backend_id"]:
        raise ReleaseAuditError(
            f"{task} planner-dominance backend is not bound to evaluator identity"
        )
    return (
        payload,
        candidates,
        contract,
        evaluator_identity,
        policy_rlinf_commits,
        policy_benchmark_commits,
    )


def _input_subset_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    content = "".join(_canonical_json(row) + "\n" for row in rows)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_inventory_task(
    inventory_rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    rows = [row for row in inventory_rows if row.get("task") == task]
    if len(rows) != len(candidates):
        raise ReleaseAuditError(f"{task} input inventory candidate count mismatch")
    for row in rows:
        index = row.get("candidate_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ReleaseAuditError(f"{task} input inventory index is not an integer")
    rows.sort(key=lambda row: row["candidate_index"])
    if [row.get("candidate_index") for row in rows] != list(range(len(candidates))):
        raise ReleaseAuditError(
            f"{task} input inventory indices are missing or duplicated"
        )
    for index, (row, candidate) in enumerate(zip(rows, candidates, strict=True)):
        if row.get("schema_version") != INPUT_INVENTORY_SCHEMA:
            raise ReleaseAuditError(f"{task} input record {index} schema mismatch")
        semantics = _candidate_semantics(candidate)
        if (
            row.get("candidate_id") != candidate.get("candidate_id")
            or row.get("kind") != candidate.get("kind")
            or row.get("semantics") != semantics
            or row.get("semantics_sha256") != _payload_sha256(semantics)
            or row.get("provenance") != candidate.get("provenance")
        ):
            raise ReleaseAuditError(
                f"{task} input record {index} differs from candidate pool"
            )
    return _input_subset_hash(rows)


def _sorted_commits(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ReleaseAuditError(f"{label} must be an ordered list")
    commits = [_require_commit(item, label) for item in value]
    if commits != sorted(commits) or len(commits) != len(set(commits)):
        raise ReleaseAuditError(f"{label} must be sorted and unique")
    return commits


def _candidate_release_file(
    root: Path,
    value: Any,
    *,
    label: str,
    expected_sha256: str,
) -> Path:
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ReleaseAuditError(f"{label} must be a portable release-relative path")
    pure = PurePosixPath(value)
    if not pure.parts or ".." in pure.parts:
        raise ReleaseAuditError(f"{label} is unsafe")
    path = root.joinpath(*pure.parts).resolve()
    if path != root and root not in path.parents:
        raise ReleaseAuditError(f"{label} escapes the candidate release root")
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ReleaseAuditError(f"{label} is missing or hash-tampered")
    return path


def _verify_candidate_inputs(path: Path, expected_sha256: str) -> None:
    if _sha256(path) != expected_sha256:
        raise ReleaseAuditError("candidate release INPUTS.sha256 identity mismatch")
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ReleaseAuditError(
                f"candidate INPUTS.sha256:{line_number} is malformed"
            )
        digest = _require_sha256(parts[0], f"candidate INPUTS.sha256:{line_number}")
        fields = parts[1].split("\t")
        if len(fields) != 3:
            raise ReleaseAuditError(
                f"candidate INPUTS.sha256:{line_number} label is malformed"
            )
        input_path = Path(fields[2])
        if not input_path.is_file() or _sha256(input_path) != digest:
            raise ReleaseAuditError(
                f"candidate release input hash mismatch for {fields[0]}:{fields[1]}"
            )


def _validate_candidate_release(
    root: Path,
    expected_manifest_sha256: str,
    expected_checksums_sha256: str,
) -> dict[str, Any]:
    root = _rld2_path(str(root.resolve()), "candidate release root", directory=True)
    checksum_entry_count = _verify_root_checksums(root, expected_checksums_sha256)
    manifest_path = root / "release_manifest.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ReleaseAuditError("candidate release-manifest hash mismatch")
    release = _load_json(manifest_path, "RLD2 candidate release manifest")
    if set(release) != _CANDIDATE_RELEASE_KEYS:
        raise ReleaseAuditError("candidate release-manifest field inventory mismatch")
    if (
        release.get("schema_version") != CANDIDATE_RELEASE_SCHEMA
        or release.get("release_id") != RELEASE_ID
        or release.get("candidate_schema_version") != CANDIDATE_SCHEMA
        or release.get("production_validated") is not True
        or release.get("payload_sha256") != _payload_sha256(release)
        or tuple(release.get("tasks", [])) != EXACT_TASKS
    ):
        raise ReleaseAuditError(
            "candidate release schema, production status, payload, or exact14 mismatch"
        )
    if "rlinf_commit" in release or "benchmark_commit" in release:
        raise ReleaseAuditError(
            "candidate release must not use singular policy authority"
        )
    policy_rlinf_commits = _sorted_commits(
        release.get("policy_rlinf_commits"), "candidate release policy RLinf commits"
    )
    policy_benchmark_commits = _sorted_commits(
        release.get("policy_benchmark_commits"),
        "candidate release policy benchmark commits",
    )
    evaluator_identity = _validate_evaluator_identity(
        release.get("evaluator_identity"),
        candidate_path=manifest_path,
        task="candidate-release",
        policy_benchmark_commits=policy_benchmark_commits,
        release_root=root,
    )
    raw_evaluator = release["evaluator_identity"]
    expected_evaluator_evidence = [
        {
            "path": relation["evidence_path"],
            "sha256": relation["evidence_sha256"],
        }
        for relation in raw_evaluator["policy_benchmark_relations"]
        if relation["relation"] == "checkpoint-compatible"
    ]
    if release.get("evaluator_evidence") != expected_evaluator_evidence:
        raise ReleaseAuditError(
            "candidate release evaluator evidence inventory mismatch"
        )
    calibration_rows = release.get("calibration_evidence")
    if not isinstance(calibration_rows, list) or [
        row.get("task") if isinstance(row, Mapping) else None
        for row in calibration_rows
    ] != list(EXACT_TASKS):
        raise ReleaseAuditError(
            "candidate release calibration evidence is not exact14-ordered"
        )
    calibration_paths: dict[str, Path] = {}
    for row in calibration_rows:
        if not isinstance(row, Mapping) or set(row) != {"task", "path", "sha256"}:
            raise ReleaseAuditError(
                "candidate release calibration evidence row is invalid"
            )
        task_value = row["task"]
        if not isinstance(task_value, str) or task_value not in EXACT_TASKS:
            raise ReleaseAuditError(
                "candidate release calibration task identity is invalid"
            )
        task = task_value
        digest = _require_sha256(row["sha256"], f"{task} calibration evidence")
        calibration_paths[task] = _candidate_release_file(
            root,
            row["path"],
            label=f"{task} calibration evidence",
            expected_sha256=digest,
        )
    hashes = release.get("task_manifest_sha256")
    counts = release.get("candidate_count")
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != set(EXACT_TASKS)
        or not isinstance(counts, Mapping)
        or set(counts) != set(EXACT_TASKS)
    ):
        raise ReleaseAuditError("candidate release task hash/count inventory mismatch")
    discovered: dict[str, Path] = {}
    for path in root.rglob("candidate_manifest.json"):
        task = path.parent.name
        if task in discovered:
            raise ReleaseAuditError(
                f"candidate release duplicates task manifest {task}"
            )
        discovered[task] = path.resolve()
    if set(discovered) != set(EXACT_TASKS):
        raise ReleaseAuditError(
            "candidate release contains missing, extra, or orphan task manifests"
        )
    for task in EXACT_TASKS:
        digest = _require_sha256(
            hashes[task], f"{task} candidate release task manifest"
        )
        if _sha256(discovered[task]) != digest:
            raise ReleaseAuditError(
                f"{task} candidate release task-manifest hash mismatch"
            )
        count = counts[task]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ReleaseAuditError(f"{task} candidate release pool size is invalid")
    inventory_path = root / "input_inventory.jsonl"
    inventory_sha256 = _require_sha256(
        release.get("input_inventory_sha256"), "candidate release input inventory"
    )
    if _sha256(inventory_path) != inventory_sha256:
        raise ReleaseAuditError("candidate release input inventory hash mismatch")
    inventory_rows = _read_jsonl(inventory_path, "candidate release input inventory")
    if {row.get("task") for row in inventory_rows} != set(EXACT_TASKS):
        raise ReleaseAuditError("candidate release input inventory is not exact14")
    _verify_candidate_inputs(
        root / "INPUTS.sha256",
        _require_sha256(
            release.get("inputs_sha256_sha256"), "candidate release INPUTS.sha256"
        ),
    )
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest_sha256": expected_manifest_sha256,
        "checksums_sha256": expected_checksums_sha256,
        "checksum_entry_count": checksum_entry_count,
        "manifest": release,
        "evaluator_identity": evaluator_identity,
        "policy_rlinf_commits": policy_rlinf_commits,
        "policy_benchmark_commits": policy_benchmark_commits,
        "task_paths": discovered,
        "task_hashes": dict(hashes),
        "candidate_counts": dict(counts),
        "inventory_path": inventory_path,
        "inventory_sha256": inventory_sha256,
        "inventory_rows": inventory_rows,
        "calibration_paths": calibration_paths,
    }


def _contract_from_audit_summary(
    summary: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    task: str,
) -> None:
    if "planner_dominance" in summary:
        if summary.get("planner_dominance") != expected:
            raise ReleaseAuditError(
                f"{task} task audit carries mixed planner calibration"
            )
        return
    if summary.get("planner_dominance_payload_sha256") == expected.get(
        "payload_sha256"
    ):
        return
    raise ReleaseAuditError(
        f"{task} task audit does not bind the planner-dominance contract"
    )


def _evaluator_from_audit_summary(
    summary: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    task: str,
) -> None:
    if "evaluator_identity" in summary:
        if summary.get("evaluator_identity") != expected:
            raise ReleaseAuditError(
                f"{task} task audit carries swapped evaluator identity"
            )
        return
    expected_hash = hashlib.sha256(
        _canonical_json(expected).encode("utf-8")
    ).hexdigest()
    if summary.get("evaluator_identity_sha256") == expected_hash:
        return
    raise ReleaseAuditError(f"{task} task audit does not bind evaluator identity")


def _validate_task_audit(
    path: Path,
    *,
    expected_sha256: str,
    root: Path,
    task: str,
    card_sha256: str,
    checksums_sha256: str,
    candidate_sha256: str,
    candidate_release_sha256: str,
    card: Mapping[str, Any],
    contract: Mapping[str, Any],
    quality_v2_threshold_identity: Mapping[str, str],
    quality_v2_calibration_receipt_identity: Mapping[str, str],
    evaluator_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ReleaseAuditError(f"{task} independent audit hash mismatch")
    audit = _load_json(path, f"{task} independent audit")
    if audit.get("schema_version") != TASK_AUDIT_SCHEMA:
        raise ReleaseAuditError(f"{task} independent audit schema mismatch")
    if audit.get("payload_sha256") != _payload_sha256(audit):
        raise ReleaseAuditError(f"{task} independent audit payload hash mismatch")
    audit_root_value = audit.get("dataset_root")
    if not isinstance(audit_root_value, str) or not audit_root_value:
        raise ReleaseAuditError(f"{task} independent audit root is invalid")
    audited_root = Path(audit_root_value).resolve()
    summary = audit.get("summary")
    if not isinstance(summary, Mapping):
        raise ReleaseAuditError(f"{task} independent audit summary is missing")
    audited_release_sha256 = audit.get("candidate_release_manifest_sha256")
    if audited_release_sha256 is None:
        audited_release_sha256 = summary.get("candidate_release_manifest_sha256")
    if (
        audited_root != root
        or audit.get("dataset_card_sha256") != card_sha256
        or audit.get("checksums_sha256") != checksums_sha256
        or audit.get("candidate_manifest_sha256") != candidate_sha256
        or audited_release_sha256 != candidate_release_sha256
        or audit.get("quality_v2_thresholds_sha256")
        != quality_v2_threshold_identity["sha256"]
    ):
        raise ReleaseAuditError(f"{task} independent audit input identity mismatch")
    if audit.get("status") != "passed" or audit.get("training_eligible") is not True:
        raise ReleaseAuditError(
            f"{task} independent audit did not grant training eligibility"
        )
    _require_commit(audit.get("auditor_commit"), f"{task} per-task auditor commit")
    accepted_count = summary.get("accepted_count")
    if (
        summary.get("task") != task
        or isinstance(accepted_count, bool)
        or not isinstance(accepted_count, int)
        or accepted_count != ACCEPTED_PER_TASK
        or summary.get("candidate_search_mode") != FULL_POOL_SEARCH_MODE
        or summary.get("selection_mode") != PLANNER_PARETO_SELECTION_MODE
        or summary.get("source_identity") != card.get("source_identity")
        or summary.get("dataset_card_payload_sha256") != card.get("payload_sha256")
        or summary.get("quality_v2_threshold_identity")
        != dict(quality_v2_threshold_identity)
        or summary.get("quality_v2_calibration_wave_receipt_identity")
        != dict(quality_v2_calibration_receipt_identity)
    ):
        raise ReleaseAuditError(f"{task} independent audit summary contract mismatch")
    _contract_from_audit_summary(summary, contract, task=task)
    _evaluator_from_audit_summary(summary, evaluator_identity, task=task)
    quality_v4 = summary.get("quality_v4_full_exports")
    if quality_v4 is not None:
        if not isinstance(quality_v4, Mapping) or set(quality_v4) != {
            "enabled",
            "audited_count",
            "thresholds_sha256",
            "orientation_contract_sha256",
            "gate_sha256",
        }:
            raise ReleaseAuditError(f"{task} Qv4 audit summary inventory mismatch")
        enabled = quality_v4.get("enabled")
        count = quality_v4.get("audited_count")
        gate_hashes = quality_v4.get("gate_sha256")
        if (
            not isinstance(enabled, bool)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not isinstance(gate_hashes, list)
        ):
            raise ReleaseAuditError(f"{task} Qv4 audit summary types are invalid")
        if enabled:
            _require_sha256(
                quality_v4.get("thresholds_sha256"), f"{task} Qv4 thresholds"
            )
            _require_sha256(
                quality_v4.get("orientation_contract_sha256"),
                f"{task} Qv4 orientation contract",
            )
            if count != accepted_count or len(gate_hashes) != accepted_count:
                raise ReleaseAuditError(f"{task} Qv4 winner audit count mismatch")
            for value in gate_hashes:
                _require_sha256(value, f"{task} Qv4 full-export gate")
        elif (
            count != 0
            or quality_v4.get("thresholds_sha256") is not None
            or quality_v4.get("orientation_contract_sha256") is not None
            or gate_hashes
        ):
            raise ReleaseAuditError(f"{task} disabled Qv4 audit summary is non-empty")
    return audit, dict(summary)


def quality_v4_release_readiness(
    task_audit_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the parallel exact-14 Qv4 release boundary without mutating Qv3."""

    if set(task_audit_summaries) != set(EXACT_TASKS):
        raise ReleaseAuditError("Qv4 release readiness requires the exact-14 task set")
    threshold_hashes = set()
    orientation_hashes = set()
    gate_count = 0
    for task in EXACT_TASKS:
        summary = task_audit_summaries[task]
        quality_v4 = summary.get("quality_v4_full_exports")
        if not isinstance(quality_v4, Mapping) or quality_v4.get("enabled") is not True:
            raise ReleaseAuditError(f"{task} Qv4 full-export audit is not enabled")
        count = quality_v4.get("audited_count")
        hashes = quality_v4.get("gate_sha256")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count != ACCEPTED_PER_TASK
            or not isinstance(hashes, list)
            or len(hashes) != count
        ):
            raise ReleaseAuditError(f"{task} Qv4 full-export count mismatch")
        threshold_hashes.add(
            _require_sha256(
                quality_v4.get("thresholds_sha256"), f"{task} Qv4 thresholds"
            )
        )
        orientation_hashes.add(
            _require_sha256(
                quality_v4.get("orientation_contract_sha256"),
                f"{task} Qv4 orientation contract",
            )
        )
        for value in hashes:
            _require_sha256(value, f"{task} Qv4 full-export gate")
        gate_count += count
    if len(threshold_hashes) != 1 or len(orientation_hashes) != 1:
        raise ReleaseAuditError("Qv4 release mixes threshold or orientation contracts")
    result: dict[str, Any] = {
        "schema_version": QUALITY_V4_RELEASE_READINESS_SCHEMA,
        "task_count": len(EXACT_TASKS),
        "full_export_gate_count": gate_count,
        "thresholds_sha256": next(iter(threshold_hashes)),
        "orientation_contract_sha256": next(iter(orientation_hashes)),
        "release_ready": True,
    }
    result["payload_sha256"] = _payload_sha256(result)
    return result


def _eligible(record: Mapping[str, Any]) -> bool:
    try:
        return _optimal_auditor._eligible(record)
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseAuditError(str(error)) from error


def _metric_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
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


def _metric_spec(contract: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name.startswith("task_quality."):
        return contract["metrics"]["task_quality"][name.split(".", 1)[1]]
    return contract["metrics"][name]


def _metric_value(record: Mapping[str, Any], name: str) -> float:
    if name.startswith("task_quality."):
        component_name = name.split(".", 1)[1]
        summary = record.get("task_quality")
        components = summary.get("components") if isinstance(summary, Mapping) else None
        component = (
            components.get(component_name) if isinstance(components, Mapping) else None
        )
        if not isinstance(component, Mapping) or "value" not in component:
            raise ReleaseAuditError(
                f"task quality mapping is missing {component_name!r}"
            )
        value = _number(component["value"], f"task quality {component_name!r} value")
    else:
        raw_value = record[name]
        if name == "control_steps":
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 1
            ):
                raise ReleaseAuditError(
                    "control_steps must be a positive native integer"
                )
            value = float(raw_value)
        else:
            value = _number(raw_value, f"planner metric {name!r}")
            if name == "trajectory_completion" and not 0.0 <= value <= 1.0:
                raise ReleaseAuditError("trajectory_completion must be in [0, 1]")
            if name in {"completion_time_s", "action_l2_sum"} and value < 0.0:
                raise ReleaseAuditError(f"planner metric {name!r} must be non-negative")
    if not math.isfinite(value):
        raise ReleaseAuditError(f"planner metric {name!r} is not finite")
    return value


def _metric_thresholds(
    name: str, reference_value: float, spec: Mapping[str, Any]
) -> tuple[float, float]:
    if name == "action_l2_sum":
        floor = max(
            _number(spec["numeric_floor_absolute"], "action_l2_sum absolute floor"),
            _number(spec["numeric_floor_relative"], "action_l2_sum relative floor")
            * abs(reference_value),
        )
    else:
        floor = _number(spec["numeric_floor"], f"planner metric {name!r} floor")
    epsilon = max(
        floor,
        2.0
        * _number(
            spec["max_observed_replay_drift"],
            f"planner metric {name!r} replay drift",
        ),
    )
    strict_margin = max(
        _number(spec["scientific_resolution"], f"planner metric {name!r} resolution"),
        2.0 * epsilon,
    )
    if name == "control_steps":
        strict_margin = max(1.0, strict_margin)
    return epsilon, strict_margin


def _quality_v2_dominance_contract(
    payload: Mapping[str, Any],
    *,
    task: str,
    thresholds_sha256: str,
) -> dict[str, Any]:
    """Use the canonical per-task auditor to derive dynamic Qv3 dimensions.

    The canonical helper intentionally derives the inventory from the frozen
    threshold checks, so this release boundary neither assumes a fixed 10/11
    count nor hard-codes task phases such as ``acquisition_window``.
    """

    try:
        return _optimal_auditor._quality_v2_dominance_contract(
            payload,
            task=task,
            thresholds_sha256=thresholds_sha256,
            require_formal_freeze=True,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseAuditError(
            f"{task} quality-v2 threshold contract is invalid: {error}"
        ) from error


def _quality_v2_metric_value(
    record: Mapping[str, Any], spec: Mapping[str, Any]
) -> float:
    try:
        return _optimal_auditor._quality_v2_metric_value(record, spec)
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseAuditError(str(error)) from error


def _validate_quality_v2_attempt(
    record: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, float]:
    try:
        return _optimal_auditor._validate_quality_v2_attempt(record, contract)
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseAuditError(str(error)) from error


def _quality_v2_metric_thresholds(
    reference_value: float, spec: Mapping[str, Any]
) -> tuple[float, float]:
    try:
        return _optimal_auditor._quality_v2_metric_thresholds(reference_value, spec)
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseAuditError(str(error)) from error


def _t5_causal_latency(record: Mapping[str, Any]) -> float | None:
    helper = getattr(_optimal_auditor, "_t5_causal_latency", None)
    if helper is None:
        return None
    try:
        return helper(record)
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseAuditError(str(error)) from error


def _validate_attempt_quality(
    record: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    try:
        _optimal_auditor._validate_attempt_quality(record, contract)
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseAuditError(str(error)) from error


def _planner_pareto_dominates(
    candidate: Mapping[str, Any],
    planner: Mapping[str, Any],
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
) -> bool:
    """Reproduce the canonical same-reset task+Qv3 Pareto predicate."""

    try:
        return _optimal_auditor._planner_pareto_dominates(
            candidate,
            planner,
            contract,
            quality_v2_contract,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseAuditError(str(error)) from error


def _selected(
    records: list[dict[str, Any]],
    *,
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recompute the canonical full-pool winner, including planner-on-tie."""

    try:
        return _optimal_auditor._selected(
            records,
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=contract,
            quality_v2_dominance=quality_v2_contract,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseAuditError(str(error)) from error


def _audit_attempt_tape_binding(
    root: Path,
    record: Mapping[str, Any],
    *,
    task: str,
    quality_v2_thresholds: Mapping[str, Any],
    quality_v2_thresholds_sha256: str,
) -> None:
    """Reopen the lightweight tape and independently recompute Qv3 and hashes."""

    try:
        _optimal_auditor._audit_attempt_tape(
            root,
            record,
            expected_task=task,
            quality_v2_thresholds=quality_v2_thresholds,
            quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
        )
    except Exception as error:
        raise ReleaseAuditError(
            f"{task}/{record.get('episode_id')} attempt tape mismatch: {error}"
        ) from error


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseAuditError(f"{label} must be a native JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ReleaseAuditError(f"{label} must be finite")
    return result


def _validate_calibration_evidence(
    *,
    task: str,
    contract: Mapping[str, Any],
    evaluator_identity_raw: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidate_release_root: Path,
    release_evidence_path: Path,
) -> dict[str, Any]:
    calibration = contract["calibration"]
    evidence_path = _evidence_path(
        candidate_manifest_path,
        calibration["evidence_path"],
        f"{task} calibration",
        release_root=candidate_release_root,
    )
    if evidence_path != release_evidence_path.resolve():
        raise ReleaseAuditError(
            f"{task} calibration path differs from release inventory"
        )
    if _sha256(evidence_path) != calibration["evidence_sha256"]:
        raise ReleaseAuditError(f"{task} calibration evidence hash tamper")
    evidence = _load_json(evidence_path, f"{task} calibration evidence")
    if set(evidence) != {
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
        raise ReleaseAuditError(f"{task} calibration evidence field inventory mismatch")
    calibration_evaluator_identity = {
        "evaluator_rlinf_commit": evaluator_identity_raw["evaluator_rlinf_commit"],
        "evaluator_benchmark_commit": evaluator_identity_raw[
            "evaluator_benchmark_commit"
        ],
        "backend_id": evaluator_identity_raw["backend_id"],
    }
    expected_evaluator_sha256 = hashlib.sha256(
        _canonical_json(calibration_evaluator_identity).encode("utf-8")
    ).hexdigest()
    if (
        evidence.get("schema_version") != CALIBRATION_EVIDENCE_SCHEMA
        or evidence.get("task") != task
        or evidence.get("backend_id") != contract["backend_id"]
        or evidence.get("evaluator_identity_sha256") != expected_evaluator_sha256
        or evidence.get("split") not in {"train", "validation"}
        or evidence.get("test_exposure") != {"test_id": False, "test_ood": False}
        or evidence.get("reset_manifest_sha256") != calibration["reset_manifest_sha256"]
        or evidence.get("payload_sha256") != _payload_sha256(evidence)
    ):
        raise ReleaseAuditError(f"{task} calibration evidence identity mismatch")
    replay_count = evidence.get("replay_count")
    if (
        isinstance(replay_count, bool)
        or not isinstance(replay_count, int)
        or replay_count < 3
        or replay_count != calibration["replay_count"]
    ):
        raise ReleaseAuditError(f"{task} calibration replay_count is invalid")
    replays = evidence.get("replays")
    if not isinstance(replays, list) or len(replays) != replay_count:
        raise ReleaseAuditError(f"{task} calibration replay inventory mismatch")
    expected_replay_keys = {
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
    stable_identity: tuple[str, str, str, int] | None = None
    metric_values = {name: [] for name in _metric_names(contract)}
    for index, replay in enumerate(replays):
        if not isinstance(replay, Mapping) or set(replay) != expected_replay_keys:
            raise ReleaseAuditError(
                f"{task} calibration replay {index} inventory mismatch"
            )
        replay_index = replay.get("replay_index")
        environment_id = replay.get("environment_instance_id")
        control_steps = replay.get("control_steps")
        if (
            isinstance(replay_index, bool)
            or not isinstance(replay_index, int)
            or replay_index != index
            or not isinstance(environment_id, str)
            or not environment_id.strip()
            or environment_id in environment_ids
            or isinstance(control_steps, bool)
            or not isinstance(control_steps, int)
            or control_steps < 1
        ):
            raise ReleaseAuditError(
                f"{task} calibration replay {index} index/env/steps invalid"
            )
        environment_ids.add(environment_id)
        reset_request_sha256 = _require_sha256(
            replay.get("reset_request_sha256"), f"{task} calibration reset request"
        )
        action_sha256 = _require_sha256(
            replay.get("action_sha256"), f"{task} calibration action"
        )
        termination_reason = replay.get("termination_reason")
        if not isinstance(termination_reason, str) or not termination_reason.strip():
            raise ReleaseAuditError(f"{task} calibration termination reason is missing")
        identity = (
            reset_request_sha256,
            action_sha256,
            termination_reason,
            control_steps,
        )
        if stable_identity is None:
            stable_identity = identity
        elif identity != stable_identity:
            raise ReleaseAuditError(
                f"{task} calibration reset/action/termination/steps identity drifted"
            )
        if (
            replay.get("episode_id") != calibration["reset_episode_id"]
            or replay.get("success") is not True
            or replay.get("safety_failure") is not False
            or replay.get("finite_and_bounded") is not True
        ):
            raise ReleaseAuditError(
                f"{task} calibration replay {index} hard gates failed"
            )
        _number(replay.get("trajectory_completion"), f"{task} trajectory_completion")
        _number(replay.get("completion_time_s"), f"{task} completion_time_s")
        _number(replay.get("action_l2_sum"), f"{task} action_l2_sum")
        _validate_attempt_quality(replay, contract)
        for name in metric_values:
            metric_values[name].append(_metric_value(replay, name))
    tolerances: dict[str, dict[str, float]] = {}
    for name, values in metric_values.items():
        observed_drift = max(values) - min(values)
        spec = _metric_spec(contract, name)
        frozen_drift = _number(
            spec["max_observed_replay_drift"],
            f"{task}/{name} frozen replay drift",
        )
        if not math.isclose(observed_drift, frozen_drift, rel_tol=0.0, abs_tol=1.0e-15):
            raise ReleaseAuditError(
                f"{task} calibration max drift does not recompute for {name}"
            )
        epsilon, strict_margin = _metric_thresholds(name, values[0], spec)
        tolerances[name] = {
            "max_observed_replay_drift": observed_drift,
            "epsilon": epsilon,
            "strict_margin": strict_margin,
        }
    return {
        "evidence_path": str(evidence_path),
        "evidence_sha256": calibration["evidence_sha256"],
        "evidence_payload_sha256": evidence["payload_sha256"],
        "replay_count": replay_count,
        "tolerances": tolerances,
    }


@dataclass
class _DeltaAccumulator:
    paired_count: int = 0
    unavailable_count: int = 0
    better_count: int = 0
    within_tolerance_count: int = 0
    worse_count: int = 0
    values: list[float] = field(default_factory=list)

    def unavailable(self) -> None:
        self.unavailable_count += 1

    def add(self, value: float, *, epsilon: float, strict_margin: float) -> None:
        self.paired_count += 1
        self.values.append(0.0 if value == 0.0 else value)
        if value < -epsilon:
            self.worse_count += 1
        elif value > strict_margin:
            self.better_count += 1
        else:
            self.within_tolerance_count += 1

    def merge(self, other: "_DeltaAccumulator") -> None:
        self.paired_count += other.paired_count
        self.unavailable_count += other.unavailable_count
        self.better_count += other.better_count
        self.within_tolerance_count += other.within_tolerance_count
        self.worse_count += other.worse_count
        self.values.extend(other.values)

    def payload(self) -> dict[str, Any]:
        return {
            "paired_count": self.paired_count,
            "unavailable_count": self.unavailable_count,
            "better_count": self.better_count,
            "within_tolerance_count": self.within_tolerance_count,
            "worse_count": self.worse_count,
            "minimum_improvement": min(self.values) if self.values else None,
            "maximum_improvement": max(self.values) if self.values else None,
            "mean_improvement": (
                math.fsum(self.values) / len(self.values) if self.values else None
            ),
        }


def _attempt_index(
    attempts: Sequence[Mapping[str, Any]],
    *,
    task: str,
    pool_size: int,
) -> dict[str, dict[int, Mapping[str, Any]]]:
    grouped: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in attempts:
        episode_id = row.get("episode_id")
        index = row.get("candidate_index")
        if (
            not isinstance(episode_id, str)
            or not episode_id
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < pool_size
            or row.get("task_id") != task
        ):
            raise ReleaseAuditError(f"{task} attempt identity is invalid")
        if index in grouped[episode_id]:
            raise ReleaseAuditError(f"{task} repeats attempt {episode_id}/{index}")
        grouped[episode_id][index] = row
    return grouped


def _audit_source_labels_and_deltas(
    root: Path,
    *,
    task: str,
    candidates: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
    quality_v2_thresholds: Mapping[str, Any],
    quality_v2_thresholds_sha256: str,
) -> tuple[dict[str, int], dict[str, dict[str, Any]], int]:
    attempts = _read_jsonl(root / "attempts.jsonl", f"{task} attempts")
    results = _read_jsonl(root / "reset_results.jsonl", f"{task} reset results")
    referenced_tapes: set[str] = set()
    for attempt in attempts:
        candidate_index = attempt.get("candidate_index")
        if (
            attempt.get("schema_version") != ATTEMPT_SCHEMA
            or isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < len(candidates)
            or attempt.get("candidate_id")
            != candidates[candidate_index].get("candidate_id")
            or attempt.get("candidate_kind") != candidates[candidate_index].get("kind")
        ):
            raise ReleaseAuditError(
                f"{task} attempt candidate/schema identity mismatch"
            )
        tape = attempt.get("attempt_tape")
        if not isinstance(tape, str) or not tape or tape in referenced_tapes:
            raise ReleaseAuditError(
                f"{task} attempt tape paths must be present and unique"
            )
        referenced_tapes.add(tape)
        _validate_attempt_quality(attempt, contract)
        _audit_attempt_tape_binding(
            root,
            attempt,
            task=task,
            quality_v2_thresholds=quality_v2_thresholds,
            quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
        )
        eligible = attempt.get("eligible")
        if not isinstance(eligible, bool) or eligible is not _eligible(attempt):
            raise ReleaseAuditError(f"{task} attempt eligibility does not recompute")
    grouped = _attempt_index(attempts, task=task, pool_size=len(candidates))
    source_counts: Counter[str] = Counter()
    delta_names = [
        *_metric_names(contract),
        *(str(spec["name"]) for spec in quality_v2_contract["metrics"]),
    ]
    causal_selection = "t5-replan-causal-timing" in PLANNER_PARETO_SELECTION_CONTRACT
    if task == "t5_replan" and causal_selection:
        delta_names.append(_T5_CAUSAL_LATENCY_METRIC)
    deltas = {name: _DeltaAccumulator() for name in delta_names}
    accepted = 0
    seen_episodes: set[str] = set()
    for result_position, result in enumerate(results):
        episode_id = result.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in seen_episodes:
            raise ReleaseAuditError(f"{task} reset-result episode IDs are invalid")
        seen_episodes.add(episode_id)
        records = grouped.get(episode_id)
        if records is None or set(records) != set(range(len(candidates))):
            raise ReleaseAuditError(
                f"{task}/{episode_id} did not run the full frozen pool"
            )
        candidate_count = result.get("candidate_count")
        budget_used = result.get("budget_used")
        if (
            result.get("candidate_search_mode") != FULL_POOL_SEARCH_MODE
            or result.get("selection_mode") != PLANNER_PARETO_SELECTION_MODE
            or isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count != len(candidates)
            or isinstance(budget_used, bool)
            or not isinstance(budget_used, int)
            or budget_used != len(candidates)
        ):
            raise ReleaseAuditError(f"{task}/{episode_id} is not a full-pool reset")
        if "reset_index" in result:
            reset_index = result["reset_index"]
            if (
                isinstance(reset_index, bool)
                or not isinstance(reset_index, int)
                or reset_index != result_position
            ):
                raise ReleaseAuditError(f"{task} reset-result order is not contiguous")
        selection = result.get("selection_result")
        if not isinstance(selection, Mapping):
            raise ReleaseAuditError(f"{task}/{episode_id} selection result is missing")
        ordered_records = [dict(records[index]) for index in range(len(candidates))]
        selected = _selected(
            ordered_records,
            contract=contract,
            quality_v2_contract=quality_v2_contract,
        )
        selected_index = None if selected is None else int(selected["candidate_index"])
        expected_source = (
            "rejected"
            if selected_index is None
            else "planner_fallback"
            if selected_index == 0
            else "expert_dominant"
        )
        expected_selection = {
            "source_kind": expected_source,
            "planner_eligible": _eligible(records[0]),
            "winner_candidate_id": (
                None if selected is None else selected["candidate_id"]
            ),
            "winner_candidate_index": selected_index,
        }
        if dict(selection) != expected_selection:
            raise ReleaseAuditError(
                f"{task}/{episode_id} selection result does not reproduce from "
                "the full same-reset pool"
            )
        planner = records[0]
        if selected_index is None:
            pass
        else:
            selected = records[selected_index]
            if not _eligible(selected):
                raise ReleaseAuditError(
                    f"{task}/{episode_id} selected attempt is ineligible"
                )
            if selected_index == 0 and not _eligible(planner):
                raise ReleaseAuditError(
                    f"{task}/{episode_id} ineligible planner was used as fallback"
                )
            if (
                selected_index > 0
                and _eligible(planner)
                and not _planner_pareto_dominates(
                    selected,
                    planner,
                    contract,
                    quality_v2_contract,
                )
            ):
                raise ReleaseAuditError(
                    f"{task}/{episode_id} expert_dominant winner does not dominate planner"
                )
        published = result.get("accepted")
        if not isinstance(published, bool):
            raise ReleaseAuditError(
                f"{task}/{episode_id} accepted must be a native boolean"
            )
        published_index = result.get("winner_candidate_index")
        if published:
            if (
                selected_index is None
                or isinstance(published_index, bool)
                or not isinstance(published_index, int)
                or published_index != selected_index
                or result.get("winner_candidate_id")
                != selection.get("winner_candidate_id")
            ):
                raise ReleaseAuditError(
                    f"{task}/{episode_id} published winner identity mismatch"
                )
            accepted += 1
            source_counts[expected_source] += 1
            selected = records[selected_index]
            for name in _metric_names(contract):
                planner_value = _metric_value(planner, name)
                winner_value = _metric_value(selected, name)
                spec = _metric_spec(contract, name)
                epsilon, strict_margin = _metric_thresholds(name, planner_value, spec)
                improvement = (
                    winner_value - planner_value
                    if spec["direction"] == "max"
                    else planner_value - winner_value
                )
                deltas[name].add(
                    improvement,
                    epsilon=epsilon,
                    strict_margin=strict_margin,
                )
            for spec in quality_v2_contract["metrics"]:
                name = str(spec["name"])
                planner_value = _quality_v2_metric_value(planner, spec)
                winner_value = _quality_v2_metric_value(selected, spec)
                epsilon, strict_margin = _quality_v2_metric_thresholds(
                    planner_value, spec
                )
                deltas[name].add(
                    planner_value - winner_value,
                    epsilon=epsilon,
                    strict_margin=strict_margin,
                )
            if task == "t5_replan" and causal_selection:
                if _eligible(planner):
                    planner_latency = _t5_causal_latency(planner)
                    winner_latency = _t5_causal_latency(selected)
                    if planner_latency is None or winner_latency is None:
                        raise ReleaseAuditError(
                            f"{task}/{episode_id} is missing canonical causal latency"
                        )
                    deltas[_T5_CAUSAL_LATENCY_METRIC].add(
                        planner_latency - winner_latency,
                        epsilon=1.0e-9,
                        strict_margin=1.0e-9,
                    )
                else:
                    deltas[_T5_CAUSAL_LATENCY_METRIC].unavailable()
        else:
            if (
                result.get("winner_candidate_id") is not None
                or published_index is not None
            ):
                raise ReleaseAuditError(
                    f"{task}/{episode_id} rejected publication names a winner"
                )
            if result.get("render_parity_skip") is None and selected_index is not None:
                raise ReleaseAuditError(
                    f"{task}/{episode_id} unpublished selection lacks render-parity evidence"
                )
            source_counts["reject"] += 1
    if set(grouped) != seen_episodes:
        raise ReleaseAuditError(
            f"{task} attempts include resets absent from reset results"
        )
    if accepted != ACCEPTED_PER_TASK:
        raise ReleaseAuditError(
            f"{task} has {accepted} accepted winners, expected {ACCEPTED_PER_TASK}"
        )
    if source_counts["expert_dominant"] + source_counts["planner_fallback"] != accepted:
        raise ReleaseAuditError(f"{task} accepted source-kind counts do not sum to 100")
    return (
        dict(source_counts),
        {name: value.payload() for name, value in deltas.items()},
        len(results),
    )


def _validate_card(
    root: Path,
    *,
    task: str,
    expected_sha256: str,
    expected_candidate_sha256: str,
    expected_candidate_release_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    quality_v2_threshold_identity: Mapping[str, str],
    evaluator_identity: Mapping[str, Any],
    policy_rlinf_commits: Sequence[str],
    policy_benchmark_commits: Sequence[str],
) -> dict[str, Any]:
    path = root / "dataset_card.json"
    if _sha256(path) != expected_sha256:
        raise ReleaseAuditError(f"{task} dataset-card hash mismatch")
    card = _load_json(path, f"{task} dataset card")
    if card.get("schema_version") != DATASET_CARD_SCHEMA:
        raise ReleaseAuditError(f"{task} dataset-card schema mismatch")
    if card.get("payload_sha256") != _payload_sha256(card):
        raise ReleaseAuditError(f"{task} dataset-card payload hash mismatch")
    accepted_count = card.get("accepted_count")
    accepted_target = card.get("accepted_target")
    pool_size = card.get("candidate_pool_size")
    if (
        card.get("task") != task
        or card.get("status") != "complete"
        or card.get("training_eligible") is not False
        or isinstance(accepted_count, bool)
        or not isinstance(accepted_count, int)
        or accepted_count != ACCEPTED_PER_TASK
        or isinstance(accepted_target, bool)
        or not isinstance(accepted_target, int)
        or accepted_target != ACCEPTED_PER_TASK
        or card.get("candidate_search_mode") != FULL_POOL_SEARCH_MODE
        or card.get("selection_mode") != PLANNER_PARETO_SELECTION_MODE
        or card.get("selection_contract") != PLANNER_PARETO_SELECTION_CONTRACT
        or isinstance(pool_size, bool)
        or not isinstance(pool_size, int)
        or pool_size != len(candidates)
        or card.get("budget_sequence") != [len(candidates)]
        or card.get("candidate_manifest_sha256") != expected_candidate_sha256
        or card.get("candidate_release_manifest_sha256")
        != expected_candidate_release_sha256
        or card.get("planner_dominance") != contract
        or card.get("quality_v2_threshold_identity")
        != dict(quality_v2_threshold_identity)
    ):
        raise ReleaseAuditError(f"{task} dataset-card release contract mismatch")
    source = card.get("source_identity")
    if not isinstance(source, Mapping) or set(source) != {
        "evaluator_rlinf_commit",
        "evaluator_benchmark_commit",
        "policy_rlinf_commits",
        "policy_benchmark_commits",
    }:
        raise ReleaseAuditError(f"{task} runtime/benchmark source identity is invalid")
    if (
        source.get("evaluator_rlinf_commit")
        != evaluator_identity["evaluator_rlinf_commit"]
        or source.get("evaluator_benchmark_commit")
        != evaluator_identity["evaluator_benchmark_commit"]
        or source.get("policy_rlinf_commits") != list(policy_rlinf_commits)
        or source.get("policy_benchmark_commits") != list(policy_benchmark_commits)
    ):
        raise ReleaseAuditError(
            f"{task} candidate/card runtime or benchmark identity mismatch"
        )
    _require_commit(
        source["evaluator_rlinf_commit"], f"{task} evaluator RLinf identity"
    )
    _require_commit(
        source["evaluator_benchmark_commit"], f"{task} evaluator benchmark identity"
    )
    return card


def _validate_quality_v2_thresholds(
    path: Path,
    *,
    root: Path,
    task: str,
    expected_sha256: str,
    expected_benchmark_commit: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, str],
]:
    expected_path = root / "quality_v2_thresholds.json"
    if path.resolve() != expected_path.resolve() or path.is_symlink():
        raise ReleaseAuditError(
            f"{task} threshold path must name dataset-local quality_v2_thresholds.json"
        )
    digest = _require_sha256(expected_sha256, f"{task} quality-v2 threshold")
    if not path.is_file() or _sha256(path) != digest:
        raise ReleaseAuditError(f"{task} quality-v2 threshold file hash mismatch")
    payload = _load_json(path, f"{task} quality-v2 thresholds")
    contract = _quality_v2_dominance_contract(
        payload,
        task=task,
        thresholds_sha256=digest,
    )
    identity = {
        "schema_version": QUALITY_V2_THRESHOLDS_SCHEMA,
        "sha256": digest,
    }
    try:
        calibration_receipt_identity = (
            _optimal_auditor._audit_quality_v2_calibration_receipt_artifact(
                root,
                payload,
                expected_benchmark_commit=expected_benchmark_commit,
            )
        )
    except Exception as error:
        raise ReleaseAuditError(
            f"{task} quality-v2 calibration receipt sidecar is invalid: {error}"
        ) from error
    return payload, contract, identity, calibration_receipt_identity


def _load_release_inputs(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path, "RLD2 release inputs")
    if payload.get("schema_version") == LEGACY_RELEASE_INPUT_SCHEMA:
        raise ReleaseAuditError(
            "release-input schema v0.1 is historical and lacks authoritative "
            "quality-v2 threshold identity"
        )
    if set(payload) != {
        "schema_version",
        "release_id",
        "candidate_release_root",
        "candidate_release_manifest_sha256",
        "candidate_release_checksums_sha256",
        "tasks",
        "payload_sha256",
    }:
        raise ReleaseAuditError("release-input field inventory mismatch")
    if (
        payload.get("schema_version") != RELEASE_INPUT_SCHEMA
        or payload.get("release_id") != RELEASE_ID
        or payload.get("payload_sha256") != _payload_sha256(payload)
    ):
        raise ReleaseAuditError(
            "release-input schema, release ID, or payload hash mismatch"
        )
    _require_sha256(
        payload.get("candidate_release_manifest_sha256"),
        "release-input candidate release manifest",
    )
    _require_sha256(
        payload.get("candidate_release_checksums_sha256"),
        "release-input candidate release SHA256SUMS",
    )
    records = payload.get("tasks")
    if not isinstance(records, list):
        raise ReleaseAuditError("release-input tasks must be a list")
    task_ids = [
        record.get("task") if isinstance(record, Mapping) else None
        for record in records
    ]
    if tuple(task_ids) != EXACT_TASKS:
        missing = sorted(set(EXACT_TASKS) - set(task_ids))
        extra = sorted(set(task_ids) - set(EXACT_TASKS), key=str)
        duplicates = sorted(
            task for task, count in Counter(task_ids).items() if count > 1
        )
        raise ReleaseAuditError(
            "release-input task inventory is not ordered exact14: "
            f"missing={missing}, extra={extra}, duplicates={duplicates}"
        )
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _TASK_INPUT_KEYS:
            raise ReleaseAuditError(
                "release-input task record field inventory mismatch"
            )
        row = dict(record)
        for key in (
            "dataset_card_sha256",
            "checksums_sha256",
            "candidate_manifest_sha256",
            "audit_sha256",
            "input_inventory_sha256",
            "quality_v2_thresholds_sha256",
        ):
            row[key] = _require_sha256(row[key], f"{row['task']} {key}")
        normalized.append(row)
    return payload, normalized


def _collect_release(
    input_manifest: Path, *, release_auditor_commit: str
) -> dict[str, Any]:
    input_manifest = _rld2_path(str(input_manifest.resolve()), "release input manifest")
    input_payload, inputs = _load_release_inputs(input_manifest)
    candidate_release_root = _rld2_path(
        input_payload["candidate_release_root"],
        "candidate release root",
        directory=True,
    )
    candidate_release_sha256 = _require_sha256(
        input_payload["candidate_release_manifest_sha256"],
        "candidate release-manifest SHA-256",
    )
    candidate_release_checksums_sha256 = _require_sha256(
        input_payload["candidate_release_checksums_sha256"],
        "candidate release SHA256SUMS SHA-256",
    )
    candidate_release = _validate_candidate_release(
        candidate_release_root,
        candidate_release_sha256,
        candidate_release_checksums_sha256,
    )
    inventory_path = candidate_release["inventory_path"]
    inventory_sha256 = candidate_release["inventory_sha256"]
    inventory_rows = candidate_release["inventory_rows"]
    seen_roots: set[Path] = set()
    seen_audits: set[Path] = set()
    common_static_identity: dict[str, Any] | None = None
    release_policy_rlinf_commits: set[str] = set()
    release_policy_benchmark_commits: set[str] = set()
    release_relations: dict[str, dict[str, Any]] = {}
    compatibility_inventory: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    task_records: dict[str, dict[str, Any]] = {}
    aggregate_sources: Counter[str] = Counter()
    aggregate_common = {name: _DeltaAccumulator() for name in _COMMON_METRICS}
    aggregate_accepted = 0
    aggregate_attempted = 0
    common_quality_v2_threshold_identity: dict[str, str] | None = None
    common_quality_v2_calibration_receipt_identity: dict[str, str] | None = None

    for row in inputs:
        task_value = row["task"]
        if not isinstance(task_value, str) or task_value not in EXACT_TASKS:
            raise ReleaseAuditError("release-input task identity is invalid")
        task = task_value
        root = _rld2_path(row["dataset_root"], f"{task} dataset root", directory=True)
        audit_path = _rld2_path(row["audit_path"], f"{task} audit")
        task_inventory_path = _rld2_path(
            row["input_inventory_path"], f"{task} input inventory"
        )
        quality_v2_thresholds_path = _rld2_path(
            row["quality_v2_thresholds_path"], f"{task} quality-v2 thresholds"
        )
        if root in seen_roots or audit_path in seen_audits:
            raise ReleaseAuditError(
                "release-input dataset roots and audits must be unique"
            )
        seen_roots.add(root)
        seen_audits.add(audit_path)
        if (
            task_inventory_path != inventory_path
            or row["input_inventory_sha256"] != inventory_sha256
        ):
            raise ReleaseAuditError(
                "mixed input inventories are forbidden in one release"
            )

        checksum_entries = _verify_root_checksums(root, row["checksums_sha256"])
        (
            quality_v2_thresholds,
            quality_v2_contract,
            quality_v2_threshold_identity,
            quality_v2_calibration_receipt_identity,
        ) = _validate_quality_v2_thresholds(
            quality_v2_thresholds_path,
            root=root,
            task=task,
            expected_sha256=row["quality_v2_thresholds_sha256"],
            expected_benchmark_commit=candidate_release["evaluator_identity"][
                "evaluator_benchmark_commit"
            ],
        )
        if common_quality_v2_threshold_identity is None:
            common_quality_v2_threshold_identity = quality_v2_threshold_identity
        elif quality_v2_threshold_identity != common_quality_v2_threshold_identity:
            raise ReleaseAuditError(
                f"{task} mixes a different quality-v2 threshold contract into RLD2"
            )
        if common_quality_v2_calibration_receipt_identity is None:
            common_quality_v2_calibration_receipt_identity = (
                quality_v2_calibration_receipt_identity
            )
        elif (
            quality_v2_calibration_receipt_identity
            != common_quality_v2_calibration_receipt_identity
        ):
            raise ReleaseAuditError(
                f"{task} mixes a different quality-v2 calibration receipt into RLD2"
            )
        candidate_path = root / "candidate_manifest.json"
        if row["candidate_manifest_sha256"] != candidate_release["task_hashes"][task]:
            raise ReleaseAuditError(
                f"{task} is orphaned from the pinned candidate release"
            )
        (
            candidate_payload,
            candidates,
            contract,
            evaluator_identity,
            policy_rlinf_commits,
            policy_benchmark_commits,
        ) = _validate_candidate_manifest(
            candidate_path,
            task=task,
            expected_sha256=row["candidate_manifest_sha256"],
            evidence_manifest_path=candidate_release["task_paths"][task],
            candidate_release_root=candidate_release_root,
        )
        if len(candidates) != candidate_release["candidate_counts"][task]:
            raise ReleaseAuditError(
                f"{task} candidate count differs from candidate release"
            )
        if evaluator_identity != candidate_release["evaluator_identity"]:
            raise ReleaseAuditError(
                f"{task} evaluator identity differs from pinned candidate release"
            )
        calibration_audit = _validate_calibration_evidence(
            task=task,
            contract=contract,
            evaluator_identity_raw=candidate_payload["evaluator_identity"],
            candidate_manifest_path=candidate_release["task_paths"][task],
            candidate_release_root=candidate_release_root,
            release_evidence_path=candidate_release["calibration_paths"][task],
        )
        input_subset_sha256 = _validate_inventory_task(
            inventory_rows, task=task, candidates=candidates
        )
        card = _validate_card(
            root,
            task=task,
            expected_sha256=row["dataset_card_sha256"],
            expected_candidate_sha256=row["candidate_manifest_sha256"],
            expected_candidate_release_sha256=candidate_release_sha256,
            candidates=candidates,
            contract=contract,
            quality_v2_threshold_identity=quality_v2_threshold_identity,
            evaluator_identity=evaluator_identity,
            policy_rlinf_commits=policy_rlinf_commits,
            policy_benchmark_commits=policy_benchmark_commits,
        )
        audit, audit_summary = _validate_task_audit(
            audit_path,
            expected_sha256=row["audit_sha256"],
            root=root,
            task=task,
            card_sha256=row["dataset_card_sha256"],
            checksums_sha256=row["checksums_sha256"],
            candidate_sha256=row["candidate_manifest_sha256"],
            candidate_release_sha256=candidate_release_sha256,
            card=card,
            contract=contract,
            quality_v2_threshold_identity=quality_v2_threshold_identity,
            quality_v2_calibration_receipt_identity=(
                quality_v2_calibration_receipt_identity
            ),
            evaluator_identity=candidate_payload["evaluator_identity"],
        )
        sources, paired_deltas, attempted_count = _audit_source_labels_and_deltas(
            root,
            task=task,
            candidates=candidates,
            contract=contract,
            quality_v2_contract=quality_v2_contract,
            quality_v2_thresholds=quality_v2_thresholds,
            quality_v2_thresholds_sha256=quality_v2_threshold_identity["sha256"],
        )
        static_identity = {
            "evaluator_identity_schema_version": evaluator_identity["schema_version"],
            "evaluator_rlinf_commit": evaluator_identity["evaluator_rlinf_commit"],
            "evaluator_benchmark_commit": evaluator_identity[
                "evaluator_benchmark_commit"
            ],
            "candidate_schema_version": candidate_payload["schema_version"],
            "attempt_schema_version": ATTEMPT_SCHEMA,
            "t5_action_history_schema_version": T5_ACTION_HISTORY_SCHEMA,
            "task_audit_schema_version": audit["schema_version"],
            "task_auditor_commit": audit["auditor_commit"],
            "planner_dominance_schema_version": contract["schema_version"],
            "quality_v2_threshold_schema_version": quality_v2_threshold_identity[
                "schema_version"
            ],
            "quality_v2_dominance_schema_version": quality_v2_contract[
                "schema_version"
            ],
            "quality_schema_version": contract["quality_schema"]["schema_version"],
            "backend_id": contract["backend_id"],
            "selection_contract": card["selection_contract"],
            "state_schema_version": card.get("state_schema", {}).get("schema_version"),
        }
        if common_static_identity is None:
            common_static_identity = static_identity
        elif static_identity != common_static_identity:
            raise ReleaseAuditError(
                f"{task} has mixed runtime, benchmark, schema, backend, or auditor identity"
            )
        release_policy_rlinf_commits.update(policy_rlinf_commits)
        release_policy_benchmark_commits.update(policy_benchmark_commits)
        for inventory_row in _compatibility_inventory_rows(candidates, task=task):
            compatibility_inventory[inventory_row["policy_benchmark_commit"]].append(
                inventory_row
            )
        for relation in evaluator_identity["policy_benchmark_relations"]:
            policy_commit = relation["policy_benchmark_commit"]
            previous = release_relations.get(policy_commit)
            if previous is not None and previous != relation:
                raise ReleaseAuditError(
                    f"{task} mixes compatibility evidence for policy benchmark {policy_commit}"
                )
            release_relations[policy_commit] = dict(relation)
        task_records[task] = {
            "dataset_root": str(root),
            "dataset_card_sha256": row["dataset_card_sha256"],
            "dataset_card_payload_sha256": card["payload_sha256"],
            "checksums_sha256": row["checksums_sha256"],
            "checksum_entry_count": checksum_entries,
            "candidate_manifest_sha256": row["candidate_manifest_sha256"],
            "candidate_release_manifest_sha256": candidate_release_sha256,
            "candidate_pool_size": len(candidates),
            "attempt_schema_version": ATTEMPT_SCHEMA,
            "input_records_sha256": input_subset_sha256,
            "audit_path": str(audit_path),
            "audit_sha256": row["audit_sha256"],
            "audit_payload_sha256": audit["payload_sha256"],
            "planner_dominance_payload_sha256": contract["payload_sha256"],
            "quality_v2_thresholds_path": str(quality_v2_thresholds_path),
            "quality_v2_thresholds_sha256": quality_v2_threshold_identity["sha256"],
            "quality_v2_calibration_wave_receipt_identity": (
                quality_v2_calibration_receipt_identity
            ),
            "quality_v2_dominance_payload_sha256": quality_v2_contract[
                "payload_sha256"
            ],
            "quality_v2_check_count": len(quality_v2_contract["metrics"]),
            "quality_schema_sha256": contract["quality_schema"]["schema_sha256"],
            "calibration_evidence_sha256": contract["calibration"]["evidence_sha256"],
            "calibration_audit": calibration_audit,
            "evaluator_identity_sha256": hashlib.sha256(
                _canonical_json(evaluator_identity).encode("utf-8")
            ).hexdigest(),
            "policy_rlinf_commits": list(policy_rlinf_commits),
            "policy_benchmark_commits": list(policy_benchmark_commits),
            "accepted_count": ACCEPTED_PER_TASK,
            "attempted_reset_count": attempted_count,
            "source_counts": sources,
            "paired_deltas": paired_deltas,
            "task_audit_summary_sha256": hashlib.sha256(
                _canonical_json(audit_summary).encode("utf-8")
            ).hexdigest(),
        }
        aggregate_sources.update(sources)
        aggregate_accepted += ACCEPTED_PER_TASK
        aggregate_attempted += attempted_count
        for name in _COMMON_METRICS:
            payload = paired_deltas[name]
            accumulator = _DeltaAccumulator(
                paired_count=payload["paired_count"],
                unavailable_count=payload["unavailable_count"],
                better_count=payload["better_count"],
                within_tolerance_count=payload["within_tolerance_count"],
                worse_count=payload["worse_count"],
            )
            # The per-task payload intentionally does not retain all raw values.
            # Re-read them is unnecessary for release identity, so global extrema
            # and mean are represented by per-task summaries below.
            aggregate_common[name].paired_count += accumulator.paired_count
            aggregate_common[name].unavailable_count += accumulator.unavailable_count
            aggregate_common[name].better_count += accumulator.better_count
            aggregate_common[
                name
            ].within_tolerance_count += accumulator.within_tolerance_count
            aggregate_common[name].worse_count += accumulator.worse_count

    assert (
        common_static_identity is not None
        and common_quality_v2_threshold_identity is not None
        and common_quality_v2_calibration_receipt_identity is not None
        and inventory_path is not None
        and inventory_sha256 is not None
    )
    if aggregate_sources["expert_dominant"] + aggregate_sources["planner_fallback"] != (
        ACCEPTED_PER_TASK * len(EXACT_TASKS)
    ):
        raise ReleaseAuditError("release accepted source counts do not sum to 1400")
    if (
        sorted(release_policy_rlinf_commits)
        != candidate_release["policy_rlinf_commits"]
    ):
        raise ReleaseAuditError(
            "policy RLinf commit union differs from candidate release"
        )
    if (
        sorted(release_policy_benchmark_commits)
        != candidate_release["policy_benchmark_commits"]
    ):
        raise ReleaseAuditError(
            "policy benchmark commit union differs from candidate release"
        )
    if set(release_relations) != set(release_policy_benchmark_commits):
        raise ReleaseAuditError(
            "compatibility relation inventory differs from release policies"
        )
    for policy_commit in sorted(release_policy_benchmark_commits):
        relation = release_relations[policy_commit]
        if relation["relation"] == "identical":
            continue
        evidence_path = relation.get("evidence_path")
        if not isinstance(evidence_path, str):
            raise ReleaseAuditError("compatibility relation evidence path is missing")
        _validate_compatibility_evidence(
            Path(evidence_path),
            policy_benchmark_commit=policy_commit,
            evaluator_identity=candidate_release["evaluator_identity"],
            expected_inventory=sorted(
                compatibility_inventory[policy_commit],
                key=lambda row: (row["task"], row["policy_sha256"]),
            ),
        )
    manifest: dict[str, Any] = {
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        "release_id": RELEASE_ID,
        "status": "complete",
        "release_eligible": False,
        "release_eligibility_reason": "independent release audit required",
        "tasks": list(EXACT_TASKS),
        "accepted_per_task": ACCEPTED_PER_TASK,
        "accepted_count": aggregate_accepted,
        "attempted_reset_count": aggregate_attempted,
        "input_manifest_path": str(input_manifest),
        "input_manifest_sha256": _sha256(input_manifest),
        "input_manifest_payload_sha256": input_payload["payload_sha256"],
        "input_inventory_path": str(inventory_path),
        "input_inventory_sha256": inventory_sha256,
        "candidate_release_root": str(candidate_release_root),
        "candidate_release_manifest_sha256": candidate_release_sha256,
        "candidate_release_checksums_sha256": candidate_release_checksums_sha256,
        "candidate_release_payload_sha256": candidate_release["manifest"][
            "payload_sha256"
        ],
        "quality_v2_threshold_identity": common_quality_v2_threshold_identity,
        "quality_v2_calibration_wave_receipt_identity": (
            common_quality_v2_calibration_receipt_identity
        ),
        "release_auditor_commit": release_auditor_commit,
        "common_identity": {
            **common_static_identity,
            "policy_rlinf_commits": sorted(release_policy_rlinf_commits),
            "policy_benchmark_commits": sorted(release_policy_benchmark_commits),
            "policy_benchmark_relations": [
                release_relations[commit]
                for commit in sorted(release_policy_benchmark_commits)
            ],
        },
        "task_records": task_records,
        "aggregate": {
            "source_counts": dict(aggregate_sources),
            "paired_common_metric_counts": {
                name: value.payload() for name, value in aggregate_common.items()
            },
            "paired_deltas_by_task": {
                task: record["paired_deltas"] for task, record in task_records.items()
            },
        },
    }
    manifest["payload_sha256"] = _payload_sha256(manifest)
    return manifest


def _write_release_checksums(root: Path) -> str:
    manifest_path = root / "release_manifest.json"
    content = f"{_sha256(manifest_path)}  release_manifest.json\n"
    path = root / "SHA256SUMS"
    path.write_text(content, encoding="utf-8")
    return _sha256(path)


def _verify_release_checksums(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_checksums_sha256: str,
) -> None:
    checksum_path = root / "SHA256SUMS"
    if _sha256(checksum_path) != expected_checksums_sha256:
        raise ReleaseAuditError("release SHA256SUMS identity mismatch")
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    expected_line = f"{expected_manifest_sha256}  release_manifest.json"
    if lines != [expected_line]:
        raise ReleaseAuditError("release SHA256SUMS inventory mismatch")
    if _sha256(root / "release_manifest.json") != expected_manifest_sha256:
        raise ReleaseAuditError("release-manifest file hash mismatch")


def audit_release(
    *,
    input_manifest: Path,
    release_root: Path,
    expected_release_manifest_sha256: str,
    expected_checksums_sha256: str,
    auditor_commit: str,
    output: Path | None = None,
) -> dict[str, Any]:
    """Independently reopen and audit a sealed release manifest."""

    release_root = _rld2_path(
        str(release_root.resolve()), "release root", directory=True
    )
    expected_manifest = _require_sha256(
        expected_release_manifest_sha256, "expected release-manifest SHA-256"
    )
    expected_checksums = _require_sha256(
        expected_checksums_sha256, "expected release SHA256SUMS SHA-256"
    )
    auditor_commit = _require_commit(auditor_commit, "release auditor commit")
    output = output or release_root / "release_audit.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite release audit: {output}")
    report: dict[str, Any] = {
        "schema_version": RELEASE_AUDIT_SCHEMA,
        "release_id": RELEASE_ID,
        "release_root": str(release_root),
        "release_manifest_sha256": expected_manifest,
        "checksums_sha256": expected_checksums,
        "auditor_commit": auditor_commit,
        "started_unix_s": time.time(),
    }
    try:
        _verify_release_checksums(
            release_root,
            expected_manifest_sha256=expected_manifest,
            expected_checksums_sha256=expected_checksums,
        )
        sealed = _load_json(release_root / "release_manifest.json", "release manifest")
        if sealed.get("schema_version") in HISTORICAL_RELEASE_MANIFEST_SCHEMAS:
            raise ReleaseAuditError(
                "historical release-manifest schema lacks calibration receipt identity"
            )
        if (
            sealed.get("schema_version") != RELEASE_MANIFEST_SCHEMA
            or sealed.get("release_id") != RELEASE_ID
            or sealed.get("release_eligible") is not False
            or sealed.get("payload_sha256") != _payload_sha256(sealed)
        ):
            raise ReleaseAuditError(
                "sealed release-manifest identity or payload mismatch"
            )
        recomputed = _collect_release(
            input_manifest,
            release_auditor_commit=_require_commit(
                sealed.get("release_auditor_commit"), "sealed release auditor commit"
            ),
        )
        if recomputed != sealed:
            raise ReleaseAuditError(
                "release manifest does not reproduce from sealed inputs"
            )
        report.update(
            status="passed",
            release_eligible=True,
            release_eligibility_reason="independent exact-14 release audit passed",
            quality_v2_threshold_identity=sealed["quality_v2_threshold_identity"],
            quality_v2_calibration_wave_receipt_identity=sealed[
                "quality_v2_calibration_wave_receipt_identity"
            ],
            summary={
                "task_count": len(EXACT_TASKS),
                "accepted_count": sealed["accepted_count"],
                "attempted_reset_count": sealed["attempted_reset_count"],
                "source_counts": sealed["aggregate"]["source_counts"],
                "quality_v2_threshold_identity": sealed[
                    "quality_v2_threshold_identity"
                ],
                "quality_v2_calibration_wave_receipt_identity": sealed[
                    "quality_v2_calibration_wave_receipt_identity"
                ],
                "release_manifest_payload_sha256": sealed["payload_sha256"],
            },
        )
    except Exception as error:
        report.update(
            status="failed",
            release_eligible=False,
            release_eligibility_reason="independent exact-14 release audit failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        report["finished_unix_s"] = time.time()
        report["payload_sha256"] = _payload_sha256(report)
        _write_json(output, report)
        raise
    report["finished_unix_s"] = time.time()
    report["payload_sha256"] = _payload_sha256(report)
    _write_json(output, report)
    return report


def build_and_audit_release(
    *,
    input_manifest: Path,
    output_root: Path,
    auditor_commit: str,
) -> dict[str, Any]:
    """Create a new sealed release root, then independently audit it."""

    output_root = output_root.resolve()
    folded = {part.casefold() for part in output_root.parts}
    if "rld1" in folded or "rld2" not in folded:
        raise ReleaseAuditError("release output must be a new RLD2 path")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite release root: {output_root}")
    auditor_commit = _require_commit(auditor_commit, "release auditor commit")
    manifest = _collect_release(input_manifest, release_auditor_commit=auditor_commit)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        _write_json(staging / "release_manifest.json", manifest)
        checksums_sha256 = _write_release_checksums(staging)
        manifest_sha256 = _sha256(staging / "release_manifest.json")
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit_release(
        input_manifest=input_manifest,
        release_root=output_root,
        expected_release_manifest_sha256=manifest_sha256,
        expected_checksums_sha256=checksums_sha256,
        auditor_commit=auditor_commit,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build, seal, and independently audit")
    build.add_argument("--input-manifest", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--auditor-commit", required=True)
    audit = subparsers.add_parser("audit", help="audit an existing sealed release")
    audit.add_argument("--input-manifest", type=Path, required=True)
    audit.add_argument("--release-root", type=Path, required=True)
    audit.add_argument("--expected-release-manifest-sha256", required=True)
    audit.add_argument("--expected-checksums-sha256", required=True)
    audit.add_argument("--auditor-commit", required=True)
    audit.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        report = build_and_audit_release(
            input_manifest=args.input_manifest,
            output_root=args.output_root,
            auditor_commit=args.auditor_commit,
        )
    else:
        report = audit_release(
            input_manifest=args.input_manifest,
            release_root=args.release_root,
            expected_release_manifest_sha256=args.expected_release_manifest_sha256,
            expected_checksums_sha256=args.expected_checksums_sha256,
            auditor_commit=args.auditor_commit,
            output=args.output,
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
