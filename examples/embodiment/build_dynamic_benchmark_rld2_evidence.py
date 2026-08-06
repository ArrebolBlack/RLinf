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

"""Build and validate RLD2 compatibility and planner-calibration evidence.

The utility consumes measured probe/replay JSON.  It never synthesizes rollout
outcomes.  Compatibility probes must show a real checkpoint load, reset,
inference, and environment step.  Calibration rows must come from at least
three distinct environment instances executing the same reset and action.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

COMPATIBILITY_EVIDENCE_SCHEMA = (
    "rlinf-dynamic-benchmark-checkpoint-compatibility-evidence-v0.1"
)
CALIBRATION_EVIDENCE_SCHEMA = (
    "rlinf-dynamic-benchmark-planner-calibration-evidence-v0.1"
)
PLANNER_DOMINANCE_SCHEMA = "rlinf-dynamic-benchmark-planner-dominance-v0.1"

COMPATIBILITY_PROBE_KEYS = {
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
CALIBRATION_REPLAY_KEYS = {
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


class EvidenceError(ValueError):
    """Raised when measured RLD2 evidence fails closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_artifact_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceError(f"{label} must be a full lowercase Git commit")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EvidenceError(f"{label} must be a non-empty trimmed string")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a native JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_artifact_bytes(value))
    os.replace(temporary, path)


def _compatibility_projection(
    probes: Sequence[Mapping[str, Any]], policy_benchmark_commit: str
) -> list[dict[str, Any]]:
    return [
        {
            "task": probe["task"],
            "policy_sha256": probe["policy_sha256"],
            "policy_rlinf_commit": probe["policy_rlinf_commit"],
            "policy_benchmark_commit": policy_benchmark_commit,
            "policy_state_schema_sha256": probe["policy_state_schema_sha256"],
            "policy_state_dim": probe["policy_state_dim"],
            "policy_mask_dim": probe["policy_mask_dim"],
        }
        for probe in probes
    ]


def _validate_compatibility_probe(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != COMPATIBILITY_PROBE_KEYS:
        raise EvidenceError(f"compatibility probe {index} field inventory mismatch")
    probe = dict(raw)
    _require_string(probe.get("task"), f"compatibility probe {index} task")
    _require_sha256(probe.get("policy_sha256"), f"compatibility probe {index} policy")
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
        _require_string(probe.get(key), f"compatibility probe {index} {key}")
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
            raise EvidenceError(f"compatibility probe {index} {key} is invalid")
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
            raise EvidenceError(f"compatibility probe {index} failed {key}")
    if (
        policy_schema != evaluator_schema
        or dimensions["policy_state_dim"] != dimensions["evaluator_state_dim"]
        or dimensions["policy_mask_dim"] != dimensions["evaluator_mask_dim"]
        or dimensions["policy_action_dim"] != dimensions["evaluator_action_dim"]
    ):
        raise EvidenceError(f"compatibility probe {index} schema/dimension mismatch")
    return probe


def build_compatibility_evidence(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Build a self-bound compatibility proof from measured probe rows."""

    expected = {
        "policy_benchmark_commit",
        "evaluator_rlinf_commit",
        "evaluator_benchmark_commit",
        "backend_id",
        "split",
        "test_exposure",
        "probes",
    }
    if set(raw) != expected:
        raise EvidenceError("compatibility input field inventory mismatch")
    policy_commit = _require_commit(
        raw.get("policy_benchmark_commit"), "policy benchmark commit"
    )
    evaluator_rlinf_commit = _require_commit(
        raw.get("evaluator_rlinf_commit"), "evaluator RLinf commit"
    )
    evaluator_benchmark_commit = _require_commit(
        raw.get("evaluator_benchmark_commit"), "evaluator benchmark commit"
    )
    if policy_commit == evaluator_benchmark_commit:
        raise EvidenceError("identical commits do not need compatibility evidence")
    backend_id = _require_string(raw.get("backend_id"), "evaluator backend ID")
    if raw.get("split") not in {"train", "validation"} or raw.get(
        "test_exposure"
    ) != {"test_id": False, "test_ood": False}:
        raise EvidenceError("compatibility probes used a formal test split")
    raw_probes = raw.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise EvidenceError("compatibility probes must be a non-empty list")
    probes = [_validate_compatibility_probe(row, index) for index, row in enumerate(raw_probes)]
    keys = [(row["task"], row["policy_sha256"]) for row in probes]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise EvidenceError("compatibility probes must be sorted and unique by task/policy")
    projection = _compatibility_projection(probes, policy_commit)
    evidence = {
        "schema_version": COMPATIBILITY_EVIDENCE_SCHEMA,
        "policy_benchmark_commit": policy_commit,
        "evaluator_rlinf_commit": evaluator_rlinf_commit,
        "evaluator_benchmark_commit": evaluator_benchmark_commit,
        "backend_id": backend_id,
        "split": raw["split"],
        "test_exposure": dict(raw["test_exposure"]),
        "probe_count": len(probes),
        "policy_inventory_sha256": hashlib.sha256(
            _canonical_json(projection).encode("utf-8")
        ).hexdigest(),
        "probes": probes,
    }
    evidence["payload_sha256"] = _payload_sha256(evidence)
    return evidence


def validate_compatibility_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_inventory: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a compatibility proof and optional exact policy coverage."""

    input_payload = {
        key: copy.deepcopy(evidence.get(key))
        for key in (
            "policy_benchmark_commit",
            "evaluator_rlinf_commit",
            "evaluator_benchmark_commit",
            "backend_id",
            "split",
            "test_exposure",
            "probes",
        )
    }
    rebuilt = build_compatibility_evidence(input_payload)
    expected_keys = set(rebuilt) | {"payload_sha256"}
    if set(evidence) != expected_keys or dict(evidence) != rebuilt:
        raise EvidenceError("compatibility evidence schema, payload, or derived fields mismatch")
    if expected_inventory is not None:
        projection = _compatibility_projection(
            rebuilt["probes"], rebuilt["policy_benchmark_commit"]
        )
        if projection != [dict(row) for row in expected_inventory]:
            raise EvidenceError("compatibility probes do not exactly cover expected policies")
    return rebuilt


def _quality_schema(contract: Mapping[str, Any]) -> dict[str, Any]:
    schema = contract.get("quality_schema")
    expected = {
        "schema_version",
        "task_id",
        "task_config_sha256",
        "components",
        "schema_sha256",
    }
    if not isinstance(schema, Mapping) or set(schema) != expected:
        raise EvidenceError("planner contract quality schema inventory mismatch")
    _require_string(schema.get("schema_version"), "quality schema version")
    _require_string(schema.get("task_id"), "quality schema task")
    _require_sha256(schema.get("task_config_sha256"), "quality task config")
    _require_sha256(schema.get("schema_sha256"), "quality schema")
    components = schema.get("components")
    if not isinstance(components, list) or not components:
        raise EvidenceError("quality component inventory is empty")
    names: set[str] = set()
    for row in components:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "direction",
            "unit",
            "scientific_resolution",
            "reducer",
            "source",
            "description",
        }:
            raise EvidenceError("quality component metadata inventory mismatch")
        name = _require_string(row.get("name"), "quality component name")
        if name in names or "." in name:
            raise EvidenceError("quality component names must be unique and non-dotted")
        names.add(name)
        if row.get("direction") not in {"minimize", "maximize"}:
            raise EvidenceError(f"quality component {name} direction is invalid")
        if row.get("reducer") not in {"minimum", "maximum", "terminal"}:
            raise EvidenceError(f"quality component {name} reducer is invalid")
        for key in ("unit", "source", "description"):
            _require_string(row.get(key), f"quality component {name} {key}")
        if _number(row.get("scientific_resolution"), f"quality component {name} resolution") <= 0:
            raise EvidenceError(f"quality component {name} resolution is invalid")
    unhashed = dict(schema)
    stored = unhashed.pop("schema_sha256")
    if stored != hashlib.sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest():
        raise EvidenceError("quality schema SHA-256 does not recompute")
    return copy.deepcopy(dict(schema))


def _metric_rows(contract: Mapping[str, Any]) -> tuple[list[str], Mapping[str, Any]]:
    schema = _quality_schema(contract)
    names = [
        "trajectory_completion",
        *(f"task_quality.{row['name']}" for row in schema["components"]),
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    ]
    metrics = contract.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "trajectory_completion",
        "task_quality",
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    }:
        raise EvidenceError("planner metric inventory mismatch")
    quality_metrics = metrics.get("task_quality")
    quality_names = [row["name"] for row in schema["components"]]
    if not isinstance(quality_metrics, Mapping) or set(quality_metrics) != set(quality_names):
        raise EvidenceError("planner quality metric inventory mismatch")
    tie_break = contract.get("tie_break_order")
    if not isinstance(tie_break, list) or tie_break != names:
        raise EvidenceError("planner tie-break order must match the canonical metric order")
    return names, metrics


def _metric_spec(metrics: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return (
        metrics["task_quality"][name.split(".", 1)[1]]
        if name.startswith("task_quality.")
        else metrics[name]
    )


def _metric_value(row: Mapping[str, Any], name: str) -> float:
    if name.startswith("task_quality."):
        summary = row["task_quality"]
        component = summary["components"][name.split(".", 1)[1]]
        return _number(component["value"], f"calibration {name}")
    value = row[name]
    if name == "control_steps":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EvidenceError("calibration control_steps must be a positive integer")
        return float(value)
    result = _number(value, f"calibration {name}")
    if name == "trajectory_completion" and not 0.0 <= result <= 1.0:
        raise EvidenceError("calibration trajectory_completion must be in [0, 1]")
    if name in {"completion_time_s", "action_l2_sum"} and result < 0.0:
        raise EvidenceError(f"calibration {name} must be non-negative")
    return result


def _validate_quality_summary(
    summary: Any,
    *,
    contract: Mapping[str, Any],
    episode_id: str,
) -> None:
    schema = contract["quality_schema"]
    if not isinstance(summary, Mapping) or set(summary) != {
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
        raise EvidenceError("calibration task-quality summary inventory mismatch")
    payload = dict(summary)
    stored_sha256 = payload.pop("summary_sha256")
    if (
        summary.get("schema_version") != schema["schema_version"]
        or summary.get("episode_id") != episode_id
        or summary.get("task_id") != contract["task"]
        or summary.get("evaluator_backend_id") != contract["backend_id"]
        or summary.get("schema_sha256") != schema["schema_sha256"]
        or summary.get("terminal") is not True
        or stored_sha256 != hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    ):
        raise EvidenceError("calibration task-quality identity or hash mismatch")
    count = summary.get("physics_sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise EvidenceError("calibration task-quality sample count is invalid")
    components = summary.get("components")
    expected_names = [row["name"] for row in schema["components"]]
    if not isinstance(components, Mapping) or list(components) != expected_names:
        raise EvidenceError("calibration task-quality component order/inventory mismatch")
    for metadata in schema["components"]:
        name = metadata["name"]
        component = components[name]
        if not isinstance(component, Mapping) or set(component) != {
            "value",
            "direction",
            "unit",
            "scientific_resolution",
            "reducer",
        }:
            raise EvidenceError(f"calibration task-quality component {name} is invalid")
        for key in ("direction", "unit", "scientific_resolution", "reducer"):
            if component.get(key) != metadata[key]:
                raise EvidenceError(f"calibration task-quality component {name} metadata drifted")
        _number(component.get("value"), f"calibration task-quality component {name}")


def _calibration_drifts(
    replays: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, float]:
    names, metrics = _metric_rows(contract)
    environment_ids: set[str] = set()
    stable_identity: tuple[str, str, str, str, int] | None = None
    values = {name: [] for name in names}
    for index, raw in enumerate(replays):
        if not isinstance(raw, Mapping) or set(raw) != CALIBRATION_REPLAY_KEYS:
            raise EvidenceError(f"calibration replay {index} field inventory mismatch")
        if raw.get("replay_index") != index or isinstance(raw.get("replay_index"), bool):
            raise EvidenceError("calibration replay indices must be contiguous integers")
        environment_id = _require_string(
            raw.get("environment_instance_id"), f"calibration replay {index} environment"
        )
        if environment_id in environment_ids:
            raise EvidenceError("calibration requires unique fresh environment instances")
        environment_ids.add(environment_id)
        episode_id = _require_string(raw.get("episode_id"), "calibration episode ID")
        reset_sha256 = _require_sha256(
            raw.get("reset_request_sha256"), "calibration reset request"
        )
        action_sha256 = _require_sha256(raw.get("action_sha256"), "calibration action")
        termination_reason = _require_string(
            raw.get("termination_reason"), "calibration termination reason"
        )
        control_steps = raw.get("control_steps")
        if isinstance(control_steps, bool) or not isinstance(control_steps, int) or control_steps < 1:
            raise EvidenceError("calibration control_steps must be a positive integer")
        identity = (
            episode_id,
            reset_sha256,
            action_sha256,
            termination_reason,
            control_steps,
        )
        if stable_identity is None:
            stable_identity = identity
        elif identity != stable_identity:
            raise EvidenceError("calibration reset/action/discrete outcome identity drifted")
        if (
            raw.get("success") is not True
            or raw.get("safety_failure") is not False
            or raw.get("finite_and_bounded") is not True
        ):
            raise EvidenceError("calibration replay is not successful, safe, and finite")
        _validate_quality_summary(raw.get("task_quality"), contract=contract, episode_id=episode_id)
        for name in names:
            values[name].append(_metric_value(raw, name))
    if len(environment_ids) < 3:
        raise EvidenceError("calibration requires at least three fresh environments")
    drifts = {name: max(rows) - min(rows) for name, rows in values.items()}
    for name in names:
        spec = _metric_spec(metrics, name)
        if not isinstance(spec, Mapping):
            raise EvidenceError(f"planner metric {name} is not a mapping")
    return drifts


def build_calibration_evidence(
    raw: Mapping[str, Any],
    *,
    contract_template: Mapping[str, Any],
    evidence_reference: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build evidence and a drift-populated planner-dominance contract."""

    expected = {
        "task",
        "backend_id",
        "evaluator_identity",
        "split",
        "test_exposure",
        "reset_manifest_sha256",
        "replays",
    }
    if set(raw) != expected:
        raise EvidenceError("calibration input field inventory mismatch")
    if not isinstance(contract_template, Mapping) or set(contract_template) != {
        "schema_version",
        "task",
        "backend_id",
        "quality_schema",
        "calibration",
        "metrics",
        "tie_break_order",
    }:
        raise EvidenceError("planner contract template inventory mismatch")
    if contract_template.get("schema_version") != PLANNER_DOMINANCE_SCHEMA:
        raise EvidenceError("planner contract template schema mismatch")
    task = _require_string(raw.get("task"), "calibration task")
    backend_id = _require_string(raw.get("backend_id"), "calibration backend")
    if contract_template.get("task") != task or contract_template.get("backend_id") != backend_id:
        raise EvidenceError("calibration input differs from planner contract identity")
    evaluator_identity = raw.get("evaluator_identity")
    if not isinstance(evaluator_identity, Mapping):
        raise EvidenceError("calibration evaluator identity is missing")
    _require_commit(
        evaluator_identity.get("evaluator_rlinf_commit"), "calibration evaluator RLinf commit"
    )
    _require_commit(
        evaluator_identity.get("evaluator_benchmark_commit"),
        "calibration evaluator benchmark commit",
    )
    if evaluator_identity.get("backend_id") != backend_id:
        raise EvidenceError("calibration evaluator backend mismatch")
    if raw.get("split") not in {"train", "validation"} or raw.get(
        "test_exposure"
    ) != {"test_id": False, "test_ood": False}:
        raise EvidenceError("calibration used a formal test split")
    reset_manifest_sha256 = _require_sha256(
        raw.get("reset_manifest_sha256"), "calibration reset manifest"
    )
    replays = raw.get("replays")
    if not isinstance(replays, list) or len(replays) < 3:
        raise EvidenceError("calibration requires at least three replay rows")
    contract = copy.deepcopy(dict(contract_template))
    contract["quality_schema"] = _quality_schema(contract)
    drifts = _calibration_drifts(replays, contract)
    evidence = {
        "schema_version": CALIBRATION_EVIDENCE_SCHEMA,
        "task": task,
        "backend_id": backend_id,
        "evaluator_identity_sha256": _payload_sha256(evaluator_identity),
        "split": raw["split"],
        "test_exposure": dict(raw["test_exposure"]),
        "reset_manifest_sha256": reset_manifest_sha256,
        "replay_count": len(replays),
        "replays": copy.deepcopy(replays),
    }
    evidence["payload_sha256"] = _payload_sha256(evidence)
    reference = _require_string(evidence_reference, "calibration evidence reference")
    episode_id = replays[0]["episode_id"]
    contract["calibration"] = {
        "replay_count": len(replays),
        "reset_episode_id": episode_id,
        "reset_manifest_sha256": reset_manifest_sha256,
        "evidence_path": reference,
        "evidence_sha256": _artifact_sha256(evidence),
    }
    for name, drift in drifts.items():
        spec = _metric_spec(contract["metrics"], name)
        spec["max_observed_replay_drift"] = drift
    return evidence, contract


def validate_calibration_evidence(
    evidence: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    evaluator_identity: Mapping[str, Any],
) -> dict[str, float]:
    """Independently recompute calibration identities and observed drifts."""

    if not isinstance(contract, Mapping) or contract.get("schema_version") != (
        PLANNER_DOMINANCE_SCHEMA
    ):
        raise EvidenceError("planner dominance contract is invalid")
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
        raise EvidenceError("calibration evidence field inventory mismatch")
    calibration = contract.get("calibration")
    if not isinstance(calibration, Mapping):
        raise EvidenceError("planner calibration contract is missing")
    if (
        evidence.get("schema_version") != CALIBRATION_EVIDENCE_SCHEMA
        or evidence.get("task") != contract.get("task")
        or evidence.get("backend_id") != contract.get("backend_id")
        or evidence.get("evaluator_identity_sha256") != _payload_sha256(evaluator_identity)
        or evidence.get("split") not in {"train", "validation"}
        or evidence.get("test_exposure") != {"test_id": False, "test_ood": False}
        or evidence.get("reset_manifest_sha256") != calibration.get("reset_manifest_sha256")
        or evidence.get("payload_sha256") != _payload_sha256(evidence)
    ):
        raise EvidenceError("calibration evidence identity or payload mismatch")
    replays = evidence.get("replays")
    replay_count = evidence.get("replay_count")
    if (
        isinstance(replay_count, bool)
        or not isinstance(replay_count, int)
        or not isinstance(replays, list)
        or replay_count != len(replays)
        or replay_count != calibration.get("replay_count")
        or replay_count < 3
    ):
        raise EvidenceError("calibration replay count mismatch")
    drifts = _calibration_drifts(replays, contract)
    if replays[0]["episode_id"] != calibration.get("reset_episode_id"):
        raise EvidenceError("calibration reset episode identity mismatch")
    for name, drift in drifts.items():
        frozen = _number(
            _metric_spec(contract["metrics"], name).get("max_observed_replay_drift"),
            f"planner metric {name} frozen drift",
        )
        if not math.isclose(drift, frozen, rel_tol=0.0, abs_tol=1.0e-15):
            raise EvidenceError(f"planner metric {name} drift does not recompute")
    return drifts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compatibility = subparsers.add_parser("compatibility")
    compatibility.add_argument("--input", type=Path, required=True)
    compatibility.add_argument("--output", type=Path, required=True)

    validate_compatibility = subparsers.add_parser("validate-compatibility")
    validate_compatibility.add_argument("--evidence", type=Path, required=True)

    calibration = subparsers.add_parser("calibration")
    calibration.add_argument("--input", type=Path, required=True)
    calibration.add_argument("--contract-template", type=Path, required=True)
    calibration.add_argument("--evidence-output", type=Path, required=True)
    calibration.add_argument("--contract-output", type=Path, required=True)
    calibration.add_argument("--evidence-reference", required=True)

    validate_calibration = subparsers.add_parser("validate-calibration")
    validate_calibration.add_argument("--evidence", type=Path, required=True)
    validate_calibration.add_argument("--contract", type=Path, required=True)
    validate_calibration.add_argument("--evaluator-identity", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "compatibility":
        result = build_compatibility_evidence(_load_json(args.input, "compatibility input"))
        _write_json(args.output, result)
        report = {
            "status": "built",
            "kind": "compatibility",
            "output": str(args.output.resolve()),
            "sha256": _artifact_sha256(result),
            "probe_count": result["probe_count"],
        }
    elif args.command == "validate-compatibility":
        result = validate_compatibility_evidence(
            _load_json(args.evidence, "compatibility evidence")
        )
        report = {
            "status": "validated",
            "kind": "compatibility",
            "probe_count": result["probe_count"],
        }
    elif args.command == "calibration":
        evidence, contract = build_calibration_evidence(
            _load_json(args.input, "calibration input"),
            contract_template=_load_json(args.contract_template, "planner contract template"),
            evidence_reference=args.evidence_reference,
        )
        _write_json(args.evidence_output, evidence)
        _write_json(args.contract_output, contract)
        report = {
            "status": "built",
            "kind": "calibration",
            "evidence": str(args.evidence_output.resolve()),
            "evidence_sha256": _artifact_sha256(evidence),
            "contract": str(args.contract_output.resolve()),
            "replay_count": evidence["replay_count"],
        }
    else:
        evidence = _load_json(args.evidence, "calibration evidence")
        contract = _load_json(args.contract, "planner contract")
        evaluator_identity = _load_json(args.evaluator_identity, "evaluator identity")
        drifts = validate_calibration_evidence(
            evidence,
            contract=contract,
            evaluator_identity=evaluator_identity,
        )
        report = {
            "status": "validated",
            "kind": "calibration",
            "metric_count": len(drifts),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
