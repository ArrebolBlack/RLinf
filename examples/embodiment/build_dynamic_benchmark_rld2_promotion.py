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

"""Build an evidence-derived RLD2-QA policy promotion receipt.

The receipt emitted by this entrypoint is deliberately small.  Its decision is
not an assertion supplied on the command line: it is derived from a separate,
canonical selection-evidence JSON after reopening the policy checkpoint, the
training summary, both matched evaluators, the exact twenty-reset manifest, and
the frozen selector and trajectory-quality contracts.

The pure :func:`validate_selection_evidence_artifacts` entrypoint is also used
by the paired-review exporter.  Consequently a receipt cannot enter review by
carrying plausible booleans while its bound evaluator artifacts disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


EVIDENCE_SCHEMA = "rld2-qa-policy-selection-evidence-v0.1"
PROMOTION_SCHEMA = "rld2-qa-policy-promotion-v0.2"
HISTORICAL_PROMOTION_SCHEMA = "rld2-qa-policy-promotion-v0.1"
POLICY_SCHEMA = "rlinf-dynamic-benchmark-expert-policy-v0.1"
POLICY_METADATA_SCHEMA = "rlinf-dynamic-benchmark-expert-summary-v0.1"
POLICY_EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-expert-evaluation-v0.3"
PLANNER_EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-planner-evaluation-v0.2"
QUALITY_V2_THRESHOLD_SCHEMA = "se3-wam-trajectory-quality-v2-thresholds-v0.3"
QUALITY_V2_SUMMARY_SCHEMA = "se3-wam-trajectory-quality-v2"
QUALITY_V2_GATE_SCHEMA = "se3-wam-trajectory-quality-v2-gate-v0.1"
SELECTOR_SCHEMA = "rlinf-dynamic-benchmark-planner-dominance-v0.1"
RESET_COUNT = 20
REVIEW_MANIFEST_SEED = 20261250
CALIBRATION_MANIFEST_SEED = 20261350
TEST_ID_MANIFEST_SEED = 20262040
TEST_OOD_MANIFEST_SEED = 20262041
FORBIDDEN_PROMOTION_MANIFEST_SEEDS = frozenset(
    {
        REVIEW_MANIFEST_SEED,
        CALIBRATION_MANIFEST_SEED,
        TEST_ID_MANIFEST_SEED,
        TEST_OOD_MANIFEST_SEED,
    }
)
WILSON_Z_95 = 1.959963984540054
T5_CAUSAL_LATENCY_TOLERANCE_S = 1.0e-9

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_RESET_IDENTITY_FIELDS = (
    "task_id",
    "episode_id",
    "seed",
    "action_mode",
    "observation_track",
    "object_mode",
    "reset_mode",
    "factors",
    "source_group_id",
    "pair_id",
    "pair_member_id",
    "candidate_index",
)
_RECORD_RESET_FIELDS = (
    "task_id",
    "episode_id",
    "seed",
    "factors",
    "source_group_id",
    "pair_id",
    "pair_member_id",
    "candidate_index",
)


def _canonical_json(value: Any) -> str:
    """Return the one accepted JSON representation and reject NaN/Infinity."""

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


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (_canonical_json(dict(value)) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_sha256(value: Any, label: str) -> str:
    result = _require_string(value, label)
    if _HEX_64.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _require_commit(value: Any, label: str) -> str:
    result = _require_string(value, label)
    if _HEX_40.fullmatch(result) is None:
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    _canonical_json(payload)
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{label} contains a blank row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid {label} JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} must be an object")
        _canonical_json(row)
        rows.append(row)
    return rows


def _verify_file(path: Path, expected_sha256: str | None, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if expected_sha256 is not None and actual != _require_sha256(
        expected_sha256, f"expected {label} SHA-256"
    ):
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


def _verify_payload_hash(payload: Mapping[str, Any], label: str) -> str:
    claimed = _require_sha256(payload.get("payload_sha256"), f"{label} payload SHA-256")
    if claimed != _payload_sha256(payload):
        raise ValueError(f"{label} payload SHA-256 does not recompute")
    return claimed


def _json_safe_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not finite canonical JSON") from error


def _policy_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Load only immutable metadata from the trusted local policy checkpoint."""

    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"cannot load policy checkpoint metadata: {path}") from error
    checkpoint = _require_mapping(payload, "policy checkpoint")
    if checkpoint.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported policy checkpoint schema")
    config = _json_safe_copy(
        _require_mapping(checkpoint.get("config"), "policy config"), "policy config"
    )
    state_schema = _json_safe_copy(
        _require_mapping(checkpoint.get("state_schema"), "policy state schema"),
        "policy state schema",
    )
    validation = _json_safe_copy(
        _require_mapping(checkpoint.get("validation"), "policy validation metadata"),
        "policy validation metadata",
    )
    if "infra_identity" not in checkpoint:
        raise ValueError("policy infra identity is missing")
    raw_infra_identity = checkpoint["infra_identity"]
    if raw_infra_identity is not None and not isinstance(raw_infra_identity, Mapping):
        raise ValueError("policy infra identity must be a mapping or null")
    infra_identity = _json_safe_copy(raw_infra_identity, "policy infra identity")
    env_steps = _require_int(checkpoint.get("env_steps"), "policy env_steps", minimum=1)
    metadata = {
        "schema_version": POLICY_SCHEMA,
        "config": config,
        "state_schema": state_schema,
        "infra_identity": infra_identity,
        "validation": validation,
        "env_steps": env_steps,
    }
    metadata["payload_sha256"] = _payload_sha256(metadata)
    return metadata


def _validate_training_metadata(
    payload: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    summary = dict(payload)
    if summary.get("schema_version") != POLICY_METADATA_SCHEMA:
        raise ValueError("policy metadata must be a complete trainer summary")
    _verify_payload_hash(summary, "policy metadata")
    if summary.get("status") != "complete":
        raise ValueError("only a naturally complete training run can promote a policy")
    if summary.get("config") != checkpoint["config"]:
        raise ValueError("policy metadata/config does not match the checkpoint")
    if summary.get("infra_identity") != checkpoint["infra_identity"]:
        raise ValueError("policy metadata infra identity does not match the checkpoint")
    if summary.get("best_validation") != checkpoint["validation"]:
        raise ValueError("policy checkpoint is not the trainer-selected best policy")
    if summary.get("best_validation") is None or summary.get("best_score") is None:
        raise ValueError("policy metadata has no learned best checkpoint")
    summary_steps = _require_int(
        summary.get("env_steps"), "trainer env_steps", minimum=1
    )
    if summary_steps < int(checkpoint["env_steps"]):
        raise ValueError("policy checkpoint env_steps exceeds its training summary")
    config_sha = _require_sha256(summary.get("config_sha256"), "policy config SHA-256")
    if config_sha != _value_sha256(checkpoint["config"]):
        raise ValueError("policy metadata config SHA-256 does not recompute")
    return summary


def _selector_contract(payload: Mapping[str, Any], *, task: str) -> dict[str, Any]:
    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    raw = payload.get("planner_dominance", payload)
    raw = _require_mapping(raw, "selector contract")
    if raw.get("schema_version") != SELECTOR_SCHEMA:
        raise ValueError("canonical selector contract schema mismatch")
    normalized = optimal._validate_planner_dominance_contract(
        {"planner_dominance": dict(raw)},
        task=task,
        selection_mode=optimal.PLANNER_PARETO_SELECTION_MODE,
    )
    if normalized is None:
        raise ValueError("selector contract did not produce planner dominance")
    return normalized


def _quality_contract(
    payload: Mapping[str, Any], *, task: str, sha256: str
) -> dict[str, Any]:
    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    if payload.get("schema_version") != QUALITY_V2_THRESHOLD_SCHEMA:
        raise ValueError(
            "production promotion requires quality-v2 threshold schema v0.3"
        )
    normalized = optimal._quality_v2_dominance_contract(
        payload,
        task=task,
        thresholds_sha256=sha256,
        require_formal_freeze=True,
    )
    metric_count = len(normalized["metrics"])
    if metric_count not in {10, 11}:
        raise ValueError(
            "production Qv3 comparison requires exactly 10 or 11 task checks"
        )
    return normalized


def _reset_identity(row: Mapping[str, Any]) -> str:
    missing = [field for field in _RESET_IDENTITY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"reset manifest row is missing identity fields: {missing}")
    return _value_sha256({field: row[field] for field in _RESET_IDENTITY_FIELDS})


def _validate_reset_manifest(
    rows: Sequence[Mapping[str, Any]], *, task: str
) -> tuple[int, list[str]]:
    if len(rows) != RESET_COUNT:
        raise ValueError(f"promotion requires exactly {RESET_COUNT} validation resets")
    episode_ids: set[str] = set()
    reset_identities: set[str] = set()
    seeds: set[int] = set()
    manifest_seed: int | None = None
    identities: list[str] = []
    for index, row in enumerate(rows):
        if row.get("task_id") != task or row.get("split") != "validation":
            raise ValueError(
                "promotion reset manifest must contain only task validation rows"
            )
        episode_id = _require_string(row.get("episode_id"), f"reset {index} episode_id")
        seed = _require_int(row.get("seed"), f"reset {index} seed")
        if episode_id in episode_ids or seed in seeds:
            raise ValueError(
                "promotion reset manifest contains duplicate episode/seed identities"
            )
        identity = _reset_identity(row)
        if identity in reset_identities:
            raise ValueError(
                "promotion reset manifest contains duplicate physical resets"
            )
        if "manifest_seed" in row:
            current_manifest_seed = _require_int(
                row.get("manifest_seed"), f"reset {index} manifest_seed"
            )
            if manifest_seed is None:
                manifest_seed = current_manifest_seed
            elif manifest_seed != current_manifest_seed:
                raise ValueError("reset manifest rows disagree on manifest_seed")
        episode_ids.add(episode_id)
        seeds.add(seed)
        reset_identities.add(identity)
        identities.append(identity)
    return (-1 if manifest_seed is None else manifest_seed), identities


def _validate_evaluation(
    payload: Mapping[str, Any],
    *,
    label: str,
    schema: str,
    task: str,
    reset_manifest_sha256: str,
) -> tuple[int, list[Mapping[str, Any]]]:
    if payload.get("schema_version") != schema:
        raise ValueError(f"{label} evaluator schema mismatch")
    _verify_payload_hash(payload, label)
    if payload.get("split") != "validation":
        raise ValueError(f"{label} must use the validation split without test exposure")
    manifest_seed = _require_int(payload.get("manifest_seed"), f"{label} manifest_seed")
    if manifest_seed in FORBIDDEN_PROMOTION_MANIFEST_SEEDS:
        raise ValueError(f"{label} reused a review/calibration/test manifest seed")
    if payload.get("reset_manifest_sha256") != reset_manifest_sha256:
        raise ValueError(f"{label} reset-manifest SHA-256 mismatch")
    if (
        _require_int(payload.get("episodes"), f"{label} episodes", minimum=1)
        != RESET_COUNT
    ):
        raise ValueError(f"{label} must evaluate exactly {RESET_COUNT} resets")
    records = list(_require_sequence(payload.get("records"), f"{label} records"))
    if len(records) != RESET_COUNT or any(
        not isinstance(row, Mapping) for row in records
    ):
        raise ValueError(f"{label} record coverage is not exactly twenty objects")
    if (
        _require_bool(payload.get("all_replays_passed"), f"{label} all replays")
        is not True
    ):
        raise ValueError(f"{label} did not pass every exact replay")
    if label == "policy evaluation":
        _require_bool(
            payload.get("all_successful_quality_v2_gates_passed"),
            "policy successful quality gates",
        )
        identity = _require_mapping(payload.get("policy_identity"), "policy identity")
        if identity.get("task") != task:
            raise ValueError("policy evaluation task identity mismatch")
    else:
        identity = _require_mapping(payload.get("planner_identity"), "planner identity")
        if identity != {"task": task, "kind": "privileged_teacher"}:
            raise ValueError("planner evaluation identity mismatch")
    return manifest_seed, records


def _validate_record_reset_alignment(
    record: Mapping[str, Any],
    reset: Mapping[str, Any],
    *,
    label: str,
    index: int,
) -> None:
    for field in _RECORD_RESET_FIELDS:
        if field not in reset or record.get(field) != reset[field]:
            raise ValueError(
                f"{label} record {index} reset identity mismatch for {field}"
            )


def _validate_replay_and_actions(record: Mapping[str, Any], *, label: str) -> None:
    import numpy as np

    for key in ("success", "safety_failure"):
        _require_bool(record.get(key), f"{label} {key}")
    replay = _require_mapping(record.get("replay_validation"), f"{label} replay")
    if _require_bool(replay.get("passed"), f"{label} replay passed") is not True:
        raise ValueError(f"{label} replay did not pass")
    _canonical_json(replay)
    actions = np.asarray(record.get("actions"), dtype=np.float64)
    if (
        actions.ndim != 2
        or actions.shape[0] < 1
        or actions.shape[1] < 1
        or not np.all(np.isfinite(actions))
        or np.any(actions < -1.0)
        or np.any(actions > 1.0)
    ):
        raise ValueError(f"{label} actions must be a finite bounded matrix")
    steps = _require_int(
        record.get("control_steps"), f"{label} control_steps", minimum=1
    )
    if steps != int(actions.shape[0]):
        raise ValueError(f"{label} control_steps/action coverage mismatch")
    action_sha = _require_sha256(record.get("action_sha256"), f"{label} action SHA-256")
    if (
        action_sha
        != hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
    ):
        raise ValueError(f"{label} action SHA-256 does not recompute")
    l2 = _require_number(record.get("action_l2_sum"), f"{label} action_l2_sum")
    recomputed_l2 = float(np.square(actions).sum(dtype=np.float64))
    if l2 < 0.0 or not math.isclose(
        l2, recomputed_l2, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise ValueError(f"{label} action_l2_sum does not recompute")
    completion = _require_number(
        record.get("trajectory_completion"), f"{label} trajectory_completion"
    )
    if not 0.0 <= completion <= 1.0:
        raise ValueError(f"{label} trajectory_completion must be in [0, 1]")
    _require_number(record.get("return"), f"{label} return")
    completion_time = record.get("completion_time_s")
    if record["success"]:
        if _require_number(completion_time, f"{label} completion_time_s") < 0.0:
            raise ValueError(f"{label} completion_time_s must be non-negative")
    elif completion_time is not None:
        raise ValueError(f"{label} failed record must not claim completion_time_s")


def _attempt_tape_payload_sha256(path: Path) -> str:
    """Hash the semantic NPZ payload independently of ZIP container metadata."""

    import numpy as np

    arrays: list[dict[str, Any]] = []
    try:
        with np.load(path, allow_pickle=False) as tape:
            for name in sorted(tape.files):
                value = np.asarray(tape[name])
                if value.dtype.hasobject:
                    raise ValueError("attempt tape cannot contain object arrays")
                arrays.append(
                    {
                        "name": name,
                        "dtype": value.dtype.str,
                        "shape": list(value.shape),
                        "sha256": hashlib.sha256(
                            np.ascontiguousarray(value).tobytes()
                        ).hexdigest(),
                    }
                )
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load attempt tape payload {path}: {error}") from error
    if not arrays:
        raise ValueError("attempt tape contains no arrays")
    return _value_sha256({"arrays": arrays})


def _audit_evaluation_attempt(
    record: Mapping[str, Any],
    *,
    evaluation_path: Path,
    task: str,
    quality_v2_thresholds: Mapping[str, Any],
    quality_v2_thresholds_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Reopen one evaluation-relative v0.3 tape and run the canonical auditor."""

    from examples.embodiment import (
        audit_dynamic_benchmark_optimal_trajectories as optimal_auditor,
    )

    attempt_schema = _require_string(
        record.get("attempt_schema_version"), f"{label} attempt schema"
    )
    if attempt_schema != optimal_auditor.ATTEMPT_SCHEMA:
        raise ValueError(f"{label} requires canonical attempt schema v0.3")
    relative = _require_string(record.get("attempt_tape"), f"{label} attempt tape path")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or relative != relative_path.as_posix()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(
            f"{label} attempt tape must be an evaluation-relative canonical path"
        )
    tape_path = (evaluation_path.parent / relative_path).resolve()
    evaluation_root = evaluation_path.parent.resolve()
    if tape_path == evaluation_root or evaluation_root not in tape_path.parents:
        raise ValueError(f"{label} attempt tape escapes its evaluation root")
    tape_sha = _require_sha256(
        record.get("attempt_tape_sha256"), f"{label} attempt tape SHA-256"
    )
    if not tape_path.is_file() or _sha256(tape_path) != tape_sha:
        raise ValueError(f"{label} attempt tape is missing or has a SHA-256 mismatch")
    audit_record = dict(record)
    audit_record["schema_version"] = attempt_schema
    optimal_auditor._audit_attempt_tape(
        evaluation_root,
        audit_record,
        expected_task=task,
        quality_v2_thresholds=quality_v2_thresholds,
        quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
    )
    payload_sha = _attempt_tape_payload_sha256(tape_path)
    if _sha256(tape_path) != tape_sha:
        raise ValueError(f"{label} attempt tape changed during validation")
    return {
        "attempt_schema_version": attempt_schema,
        "path": relative_path.as_posix(),
        "sha256": tape_sha,
        "payload_sha256": payload_sha,
    }


def _validate_task_quality(
    record: Mapping[str, Any], selector: Mapping[str, Any], *, label: str
) -> None:
    """Validate the canonical task-quality value without assigning meaning to key order."""

    summary = _require_mapping(record.get("task_quality"), f"{label} task_quality")
    expected_keys = {
        "schema_version",
        "episode_id",
        "task_id",
        "evaluator_backend_id",
        "schema_sha256",
        "physics_sample_count",
        "terminal",
        "components",
        "summary_sha256",
    }
    if set(summary) != expected_keys:
        raise ValueError(f"{label} task-quality field inventory mismatch")
    schema = _require_mapping(selector.get("quality_schema"), "selector quality schema")
    if (
        summary.get("schema_version") != schema.get("schema_version")
        or summary.get("episode_id") != record.get("episode_id")
        or summary.get("task_id") != selector.get("task")
        or summary.get("evaluator_backend_id") != selector.get("backend_id")
        or summary.get("schema_sha256") != schema.get("schema_sha256")
        or summary.get("terminal") is not True
    ):
        raise ValueError(f"{label} task-quality identity mismatch")
    _require_int(
        summary.get("physics_sample_count"),
        f"{label} task-quality physics_sample_count",
        minimum=1,
    )
    summary_sha = _require_sha256(
        summary.get("summary_sha256"), f"{label} task-quality summary SHA-256"
    )
    unsigned = dict(summary)
    unsigned.pop("summary_sha256", None)
    if summary_sha != _value_sha256(unsigned):
        raise ValueError(f"{label} task-quality summary SHA-256 does not recompute")
    values = _require_mapping(
        summary.get("components"), f"{label} task-quality components"
    )
    specifications = {
        str(component["name"]): component for component in schema["components"]
    }
    if set(values) != set(specifications):
        raise ValueError(f"{label} task-quality component inventory mismatch")
    for name, specification in specifications.items():
        component = _require_mapping(values[name], f"{label} task-quality {name}")
        if set(component) != {
            "value",
            "direction",
            "unit",
            "scientific_resolution",
            "reducer",
        }:
            raise ValueError(f"{label} task-quality {name} field inventory mismatch")
        for key in ("direction", "unit", "scientific_resolution", "reducer"):
            if component.get(key) != specification.get(key):
                raise ValueError(f"{label} task-quality {name} metadata mismatch")
        _require_number(component.get("value"), f"{label} task-quality {name} value")


def _quality_values_and_gate(
    record: Mapping[str, Any],
    quality_contract: Mapping[str, Any],
    *,
    label: str,
    require_recorded_gate: bool,
) -> tuple[dict[str, float], bool, dict[str, Any]]:
    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    quality = _require_mapping(record.get("quality_v2"), f"{label} quality_v2")
    if quality.get("schema_version") != QUALITY_V2_SUMMARY_SCHEMA:
        raise ValueError(f"{label} quality-v2 summary schema mismatch")
    quality_sha = _require_sha256(
        record.get("quality_v2_sha256"), f"{label} quality-v2 SHA-256"
    )
    if quality_sha != _payload_sha256(quality):
        raise ValueError(f"{label} quality-v2 SHA-256 does not recompute")
    values: dict[str, float] = {}
    checks: list[dict[str, Any]] = []
    for spec in quality_contract["metrics"]:
        name = str(spec["name"])
        actual = optimal._quality_v2_metric_value(record, spec)
        maximum = float(spec["maximum"])
        passed = actual <= maximum
        values[name] = actual
        checks.append(
            {
                "phase": spec["phase"],
                "metric": spec["metric"],
                "actual": actual,
                "max": maximum,
                "passed": passed,
            }
        )
    recomputed = {
        "schema_version": QUALITY_V2_GATE_SCHEMA,
        "contract_schema_version": quality_contract["threshold_schema_version"],
        "contract_sha256": quality_contract["threshold_sha256"],
        "task_id": quality_contract["task"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    recorded = record.get("quality_v2_gate")
    if require_recorded_gate and not isinstance(recorded, Mapping):
        raise ValueError(f"{label} is missing its recorded Qv3 gate")
    if isinstance(recorded, Mapping):
        recorded_checks = recorded.get("checks")
        if (
            recorded.get("schema_version") != QUALITY_V2_GATE_SCHEMA
            or recorded.get("contract_schema_version")
            != quality_contract["threshold_schema_version"]
            or recorded.get("task_id") != quality_contract["task"]
            or recorded.get("passed") != recomputed["passed"]
            or recorded_checks != checks
        ):
            raise ValueError(f"{label} recorded Qv3 gate does not recompute")
        recorded_contract_sha = recorded.get("contract_sha256")
        if (
            recorded_contract_sha is not None
            and recorded_contract_sha != quality_contract["threshold_sha256"]
        ):
            raise ValueError(f"{label} recorded Qv3 threshold identity mismatch")
    return values, bool(recomputed["passed"]), recomputed


def _causal_timing_evidence(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Validate the issued/applied declaration independently of stored eligibility."""

    issued_equals_applied = _require_bool(
        record.get("issued_equals_applied"), f"{label} issued_equals_applied"
    )
    raw_gate = record.get("t5_replan_causal_timing_passed")
    raw_latency = record.get("impact_end_to_first_qualifying_applied_correction_s")
    if record.get("task_id") != "t5_replan":
        if not issued_equals_applied:
            raise ValueError(
                f"{label} non-T5 attempt must declare issued_equals_applied=true"
            )
        if raw_gate is not None or raw_latency is not None:
            raise ValueError(
                f"{label} non-T5 attempt cannot declare T5 causal evidence"
            )
        return {
            "issued_equals_applied": True,
            "t5_replan_causal_timing_passed": None,
            "impact_end_to_first_qualifying_applied_correction_s": None,
        }
    if issued_equals_applied:
        raise ValueError(
            f"{label} T5 attempt must preserve distinct issued/applied histories"
        )
    gate = _require_bool(raw_gate, f"{label} T5 causal-timing gate")
    if gate:
        latency = _require_number(raw_latency, f"{label} T5 causal latency")
        if latency < 0.0:
            raise ValueError(f"{label} T5 causal latency must be non-negative")
    else:
        if raw_latency is not None:
            raise ValueError(f"{label} failed T5 causal gate cannot declare latency")
        latency = None
    return {
        "issued_equals_applied": False,
        "t5_replan_causal_timing_passed": gate,
        "impact_end_to_first_qualifying_applied_correction_s": latency,
    }


def _compare_both_success(
    policy: Mapping[str, Any],
    planner: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    quality_contract: Mapping[str, Any],
    policy_quality_values: Mapping[str, float],
    planner_quality_values: Mapping[str, float],
    policy_gate: bool,
    planner_gate: bool,
    policy_causal_timing: Mapping[str, Any],
    planner_causal_timing: Mapping[str, Any],
) -> dict[str, Any]:
    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    dimensions: list[dict[str, Any]] = []
    strict: list[str] = []
    nonworse = policy_gate
    dimensions.append(
        {
            "name": "quality_v3.absolute_gate",
            "direction": "max",
            "planner": planner_gate,
            "policy": policy_gate,
            "nonworse": policy_gate,
            "strictly_better": policy_gate and not planner_gate,
        }
    )
    if policy_gate and not planner_gate:
        strict.append("quality_v3.absolute_gate")

    for metric_name in optimal._dominance_metric_keys(selector):
        policy_value = optimal._metric_value(policy, metric_name)
        planner_value = optimal._metric_value(planner, metric_name)
        spec = optimal._metric_spec(selector, metric_name)
        tolerance, strict_margin = optimal._metric_thresholds(
            metric_name, planner_value, spec
        )
        direction = str(spec["direction"])
        if direction == "max":
            dimension_nonworse = policy_value >= planner_value - tolerance
            dimension_strict = policy_value > planner_value + strict_margin
        else:
            dimension_nonworse = policy_value <= planner_value + tolerance
            dimension_strict = policy_value < planner_value - strict_margin
        nonworse &= dimension_nonworse
        if dimension_strict:
            strict.append(metric_name)
        dimensions.append(
            {
                "name": metric_name,
                "direction": direction,
                "planner": planner_value,
                "policy": policy_value,
                "tolerance": tolerance,
                "strict_margin": strict_margin,
                "nonworse": dimension_nonworse,
                "strictly_better": dimension_strict,
            }
        )

    for spec in quality_contract["metrics"]:
        metric_name = str(spec["name"])
        policy_value = float(policy_quality_values[metric_name])
        planner_value = float(planner_quality_values[metric_name])
        tolerance, strict_margin = optimal._quality_v2_metric_thresholds(
            planner_value, spec
        )
        dimension_nonworse = policy_value <= planner_value + tolerance
        dimension_strict = policy_value < planner_value - strict_margin
        nonworse &= dimension_nonworse
        if dimension_strict:
            strict.append(metric_name.replace("quality_v2.", "quality_v3.", 1))
        dimensions.append(
            {
                "name": metric_name.replace("quality_v2.", "quality_v3.", 1),
                "direction": "min",
                "paired_comparison_family": spec["paired_comparison_family"],
                "planner": planner_value,
                "policy": policy_value,
                "tolerance": tolerance,
                "strict_margin": strict_margin,
                "nonworse": dimension_nonworse,
                "strictly_better": dimension_strict,
            }
        )

    if policy["task_id"] == "t5_replan":
        policy_causal_gate = bool(
            policy_causal_timing["t5_replan_causal_timing_passed"]
        )
        planner_causal_gate = bool(
            planner_causal_timing["t5_replan_causal_timing_passed"]
        )
        gate_dimension = "causal.t5_replan_causal_timing_gate"
        gate_nonworse = policy_causal_gate
        gate_strict = policy_causal_gate and not planner_causal_gate
        nonworse &= gate_nonworse
        if gate_strict:
            strict.append(gate_dimension)
        dimensions.append(
            {
                "name": gate_dimension,
                "direction": "pass",
                "planner": planner_causal_gate,
                "policy": policy_causal_gate,
                "nonworse": gate_nonworse,
                "strictly_better": gate_strict,
            }
        )

    if (
        policy_causal_timing["t5_replan_causal_timing_passed"] is True
        and planner_causal_timing["t5_replan_causal_timing_passed"] is True
    ):
        policy_causal_latency = float(
            policy_causal_timing["impact_end_to_first_qualifying_applied_correction_s"]
        )
        planner_causal_latency = float(
            planner_causal_timing["impact_end_to_first_qualifying_applied_correction_s"]
        )
        dimension_name = "causal.impact_end_to_first_qualifying_applied_correction_s"
        dimension_nonworse = (
            policy_causal_latency
            <= planner_causal_latency + T5_CAUSAL_LATENCY_TOLERANCE_S
        )
        dimension_strict = (
            policy_causal_latency
            < planner_causal_latency - T5_CAUSAL_LATENCY_TOLERANCE_S
        )
        nonworse &= dimension_nonworse
        if dimension_strict:
            strict.append(dimension_name)
        dimensions.append(
            {
                "name": dimension_name,
                "direction": "min",
                "planner": planner_causal_latency,
                "policy": policy_causal_latency,
                "tolerance": T5_CAUSAL_LATENCY_TOLERANCE_S,
                "strict_margin": T5_CAUSAL_LATENCY_TOLERANCE_S,
                "nonworse": dimension_nonworse,
                "strictly_better": dimension_strict,
            }
        )
    return {
        "planner_nonworse_all_dimensions": bool(nonworse),
        "strict_improvement_dimensions": sorted(set(strict)),
        "exact_tie": bool(nonworse and not strict),
        "dimensions": dimensions,
    }


def _wilson_interval(successes: int, total: int) -> dict[str, Any]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    rate = successes / total
    z2 = WILSON_Z_95 * WILSON_Z_95
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    radius = (
        WILSON_Z_95
        * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return {
        "count": successes,
        "total": total,
        "rate": rate,
        "wilson_95": {
            "low": max(0.0, center - radius),
            "high": min(1.0, center + radius),
        },
    }


def _artifact_identity(
    path: Path,
    *,
    sha256: str,
    schema_version: str | None = None,
    payload_sha256: str | None = None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {"path": str(path.resolve()), "sha256": sha256}
    if schema_version is not None:
        identity["schema_version"] = schema_version
    if payload_sha256 is not None:
        identity["payload_sha256"] = payload_sha256
    return identity


def _expected_sha(expected_sha256: Mapping[str, str] | None, name: str) -> str | None:
    if expected_sha256 is None:
        return None
    if name not in expected_sha256:
        raise ValueError(f"missing expected SHA-256 for {name}")
    return expected_sha256[name]


def build_selection_evidence(
    *,
    candidate_id: str,
    run_tag: str,
    policy_path: Path,
    policy_metadata_path: Path,
    checkpoint_selection_path: Path,
    policy_evaluation_path: Path,
    planner_evaluation_path: Path,
    reset_manifest_path: Path,
    quality_v2_thresholds_path: Path,
    quality_v2_calibration_wave_receipt_path: Path,
    selector_contract_path: Path,
    policy_evaluator_source_path: Path,
    planner_evaluator_source_path: Path,
    image_reference: str,
    image_sha256: str,
    expected_sha256: Mapping[str, str] | None = None,
    expected_residual_scale: float | None = None,
) -> dict[str, Any]:
    """Reopen exact artifacts and return deterministic promotion evidence."""

    from examples.embodiment.dynamic_benchmark_checkpoint_admission import (
        validate_selected_learned_policy,
    )

    candidate_id = _require_string(candidate_id, "candidate_id")
    run_tag = _require_string(run_tag, "run_tag")
    image_reference = _require_string(image_reference, "image reference")
    image_sha256 = _require_sha256(image_sha256, "image SHA-256")
    paths = {
        "policy": policy_path,
        "policy_metadata": policy_metadata_path,
        "checkpoint_selection": checkpoint_selection_path,
        "policy_evaluation": policy_evaluation_path,
        "planner_evaluation": planner_evaluation_path,
        "reset_manifest": reset_manifest_path,
        "quality_v2_thresholds": quality_v2_thresholds_path,
        "quality_v2_calibration_wave_receipt": (
            quality_v2_calibration_wave_receipt_path
        ),
        "selector_contract": selector_contract_path,
        "policy_evaluator_source": policy_evaluator_source_path,
        "planner_evaluator_source": planner_evaluator_source_path,
    }
    hashes = {
        name: _verify_file(path, _expected_sha(expected_sha256, name), name)
        for name, path in paths.items()
    }
    if (
        policy_path.name.lower() != "best_policy.pt"
        or "final" in policy_path.name.lower()
    ):
        raise ValueError(
            "production promotion requires best_policy.pt, never final policy"
        )

    checkpoint = _policy_checkpoint_metadata(policy_path)
    _validate_training_metadata(
        _read_json(policy_metadata_path, "policy metadata"), checkpoint=checkpoint
    )
    config = _require_mapping(checkpoint["config"], "policy config")
    task = _require_string(config.get("task"), "policy task")
    training_seed = _require_int(config.get("seed"), "policy training seed")
    training_validation_seed = _require_int(
        config.get("validation_manifest_seed"),
        "policy training validation manifest seed",
    )
    if training_validation_seed in FORBIDDEN_PROMOTION_MANIFEST_SEEDS:
        raise ValueError(
            "policy checkpoint selection reused a review/calibration/test manifest seed"
        )
    learned_policy_admission = validate_selected_learned_policy(
        policy_path=policy_path,
        trainer_summary_path=policy_metadata_path,
        checkpoint_selection_path=checkpoint_selection_path,
        expected_policy_sha256=hashes["policy"],
        expected_trainer_summary_sha256=hashes["policy_metadata"],
        expected_checkpoint_selection_sha256=hashes["checkpoint_selection"],
    )
    env_steps = _require_int(checkpoint.get("env_steps"), "policy env_steps", minimum=1)
    rlinf_commit = _require_commit(config.get("rlinf_commit"), "policy RLinf commit")
    benchmark_commit = _require_commit(
        config.get("benchmark_commit"), "policy benchmark commit"
    )
    algorithm = _require_string(config.get("algorithm"), "policy algorithm")
    residual_scale = (
        _require_number(config.get("residual_scale", 0.25), "policy residual_scale")
        if algorithm == "residual_rlpd"
        else None
    )
    if residual_scale is not None and not 0.0 < residual_scale <= 1.0:
        raise ValueError("policy residual_scale must be in (0, 1]")
    if expected_residual_scale != residual_scale:
        raise ValueError("requested residual_scale does not match checkpoint metadata")

    threshold_payload = _read_json(quality_v2_thresholds_path, "Qv3 thresholds")
    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    calibration_receipt = optimal._validate_quality_v2_calibration_receipt_artifact(
        threshold_payload,
        quality_v2_calibration_wave_receipt_path,
        expected_sha256=hashes["quality_v2_calibration_wave_receipt"],
        expected_benchmark_commit=benchmark_commit,
    )
    if calibration_receipt.sha256 != hashes["quality_v2_calibration_wave_receipt"]:
        raise ValueError(
            "Qv3 calibration wave receipt identity changed during validation"
        )
    selector_payload = _read_json(selector_contract_path, "selector contract")
    selector = _selector_contract(selector_payload, task=task)
    quality_contract = _quality_contract(
        threshold_payload, task=task, sha256=hashes["quality_v2_thresholds"]
    )
    policy_evaluation = _read_json(policy_evaluation_path, "policy evaluation")
    planner_evaluation = _read_json(planner_evaluation_path, "planner evaluation")
    reset_rows = _read_jsonl(reset_manifest_path, "promotion reset manifest")
    row_manifest_seed, reset_identities = _validate_reset_manifest(
        reset_rows, task=task
    )
    policy_manifest_seed, policy_records = _validate_evaluation(
        policy_evaluation,
        label="policy evaluation",
        schema=POLICY_EVALUATION_SCHEMA,
        task=task,
        reset_manifest_sha256=hashes["reset_manifest"],
    )
    if policy_manifest_seed == training_validation_seed:
        raise ValueError(
            "policy evaluation reused the checkpoint-selection validation manifest seed"
        )
    planner_manifest_seed, planner_records = _validate_evaluation(
        planner_evaluation,
        label="planner evaluation",
        schema=PLANNER_EVALUATION_SCHEMA,
        task=task,
        reset_manifest_sha256=hashes["reset_manifest"],
    )
    if policy_manifest_seed != planner_manifest_seed:
        raise ValueError("policy and planner evaluations use different manifest seeds")
    if row_manifest_seed not in {-1, policy_manifest_seed}:
        raise ValueError("reset rows/evaluations disagree on manifest_seed")

    reset_sha = hashes["reset_manifest"]
    calibration_sha = selector["calibration"]["reset_manifest_sha256"]
    if reset_sha == calibration_sha:
        raise ValueError(
            "promotion validation resets overlap selector calibration evidence"
        )
    serialized_thresholds = _canonical_json(threshold_payload)
    if reset_sha in serialized_thresholds:
        raise ValueError("promotion validation resets overlap Qv3 calibration evidence")

    policy_identity = _require_mapping(
        policy_evaluation.get("policy_identity"), "policy evaluation identity"
    )
    accepted_policy_paths = {str(policy_path), str(policy_path.resolve())}
    if (
        policy_identity.get("path") not in accepted_policy_paths
        or policy_identity.get("sha256") != hashes["policy"]
        or policy_identity.get("schema_version") != POLICY_SCHEMA
        or policy_identity.get("task") != task
        or policy_identity.get("algorithm") != algorithm
        or policy_identity.get("training_seed") != training_seed
        or policy_identity.get("training_env_steps") != env_steps
        or policy_identity.get("validation") != checkpoint["validation"]
        or policy_identity.get("checkpoint_role") != "best"
        or policy_evaluation.get("state_schema") != checkpoint["state_schema"]
    ):
        raise ValueError("policy evaluation/checkpoint metadata identity mismatch")
    if policy_evaluation.get("learned_policy_admission") != learned_policy_admission:
        raise ValueError("policy evaluation learned-policy admission identity mismatch")
    threshold_identity = _require_mapping(
        policy_evaluation.get("quality_v2_threshold_identity"),
        "policy evaluation Qv3 threshold identity",
    )
    if threshold_identity != {
        "schema_version": QUALITY_V2_THRESHOLD_SCHEMA,
        "sha256": hashes["quality_v2_thresholds"],
    }:
        raise ValueError("policy evaluation Qv3 threshold identity mismatch")

    policy_source = _require_mapping(
        policy_evaluation.get("source_identity"), "policy evaluation source identity"
    )
    planner_source = _require_mapping(
        planner_evaluation.get("source_identity"), "planner evaluation source identity"
    )
    evaluator_commit = _require_commit(
        policy_source.get("evaluator_rlinf_commit"), "policy evaluator commit"
    )
    if (
        policy_source.get("policy_rlinf_commit") != rlinf_commit
        or policy_source.get("benchmark_commit") != benchmark_commit
        or planner_source.get("evaluator_rlinf_commit") != evaluator_commit
        or planner_source.get("benchmark_commit") != benchmark_commit
    ):
        raise ValueError("policy/planner evaluator source commit identity mismatch")
    if policy_evaluator_source_path.name != "evaluate_dynamic_benchmark_expert.py":
        raise ValueError("policy evaluator source path is not canonical")
    if planner_evaluator_source_path.name != "evaluate_dynamic_benchmark_planner.py":
        raise ValueError("planner evaluator source path is not canonical")

    per_reset: list[dict[str, Any]] = []
    strict_dimensions: set[str] = set()
    rejection_reasons: list[dict[str, Any]] = []
    planner_nonworse_all_both_success = True
    counts = {
        "both_success": 0,
        "policy_only_success": 0,
        "planner_only_success": 0,
        "neither_success": 0,
        "policy_success": 0,
        "planner_success": 0,
        "policy_safety_failure": 0,
        "planner_safety_failure": 0,
        "policy_t5_causal_timing_passed": 0,
        "planner_t5_causal_timing_passed": 0,
    }
    seen_policy: set[str] = set()
    seen_planner: set[str] = set()
    for index, (reset, policy, planner) in enumerate(
        zip(reset_rows, policy_records, planner_records, strict=True)
    ):
        policy = _require_mapping(policy, f"policy record {index}")
        planner = _require_mapping(planner, f"planner record {index}")
        _validate_record_reset_alignment(
            policy, reset, label="policy evaluation", index=index
        )
        _validate_record_reset_alignment(
            planner, reset, label="planner evaluation", index=index
        )
        episode_id = str(reset["episode_id"])
        if episode_id in seen_policy or episode_id in seen_planner:
            raise ValueError("evaluation records contain duplicate episode identities")
        seen_policy.add(episode_id)
        seen_planner.add(episode_id)
        _validate_replay_and_actions(policy, label=f"policy record {index}")
        _validate_replay_and_actions(planner, label=f"planner record {index}")
        policy_causal = _causal_timing_evidence(policy, label=f"policy record {index}")
        planner_causal = _causal_timing_evidence(
            planner, label=f"planner record {index}"
        )
        policy_tape = _audit_evaluation_attempt(
            policy,
            evaluation_path=policy_evaluation_path,
            task=task,
            quality_v2_thresholds=threshold_payload,
            quality_v2_thresholds_sha256=hashes["quality_v2_thresholds"],
            label=f"policy record {index}",
        )
        planner_tape = _audit_evaluation_attempt(
            planner,
            evaluation_path=planner_evaluation_path,
            task=task,
            quality_v2_thresholds=threshold_payload,
            quality_v2_thresholds_sha256=hashes["quality_v2_thresholds"],
            label=f"planner record {index}",
        )
        _validate_task_quality(policy, selector, label=f"policy record {index}")
        _validate_task_quality(planner, selector, label=f"planner record {index}")
        policy_quality, policy_gate, policy_gate_payload = _quality_values_and_gate(
            policy,
            quality_contract,
            label=f"policy record {index}",
            require_recorded_gate=True,
        )
        planner_quality, planner_gate, planner_gate_payload = _quality_values_and_gate(
            planner,
            quality_contract,
            label=f"planner record {index}",
            require_recorded_gate=False,
        )
        policy_success = bool(policy["success"])
        planner_success = bool(planner["success"])
        policy_safety = bool(policy["safety_failure"])
        planner_safety = bool(planner["safety_failure"])
        counts["policy_success"] += int(policy_success)
        counts["planner_success"] += int(planner_success)
        counts["policy_safety_failure"] += int(policy_safety)
        counts["planner_safety_failure"] += int(planner_safety)
        policy_causal_gate = policy_causal["t5_replan_causal_timing_passed"]
        counts["policy_t5_causal_timing_passed"] += int(policy_causal_gate is True)
        counts["planner_t5_causal_timing_passed"] += int(
            planner_causal["t5_replan_causal_timing_passed"] is True
        )
        per_reset_rejections: list[dict[str, Any]] = []
        if policy_success:
            failed_gates = [
                gate_name
                for gate_name, failed in (
                    ("safety", policy_safety),
                    ("quality_v3", not policy_gate),
                    ("t5_causal_timing", policy_causal_gate is False),
                )
                if failed
            ]
            if failed_gates:
                per_reset_rejections.append(
                    {
                        "code": "successful_policy_gate_failure",
                        "scope": "reset",
                        "reset_index": index,
                        "episode_id": episode_id,
                        "failed_gates": failed_gates,
                    }
                )
        if planner_success and not policy_success:
            per_reset_rejections.append(
                {
                    "code": "planner_success_policy_failure",
                    "scope": "reset",
                    "reset_index": index,
                    "episode_id": episode_id,
                }
            )
        if policy_success and planner_success:
            counts["both_success"] += 1
            comparison = _compare_both_success(
                policy,
                planner,
                selector=selector,
                quality_contract=quality_contract,
                policy_quality_values=policy_quality,
                planner_quality_values=planner_quality,
                policy_gate=policy_gate,
                planner_gate=planner_gate,
                policy_causal_timing=policy_causal,
                planner_causal_timing=planner_causal,
            )
            planner_nonworse_all_both_success &= comparison[
                "planner_nonworse_all_dimensions"
            ]
            if comparison["planner_nonworse_all_dimensions"] is not True:
                failed_dimensions = sorted(
                    row["name"]
                    for row in comparison["dimensions"]
                    if row.get("nonworse") is False
                    and row["name"]
                    not in {
                        "quality_v3.absolute_gate",
                        "causal.t5_replan_causal_timing_gate",
                    }
                )
                if failed_dimensions:
                    per_reset_rejections.append(
                        {
                            "code": "both_success_metric_nonworse_failure",
                            "scope": "reset",
                            "reset_index": index,
                            "episode_id": episode_id,
                            "failed_dimensions": failed_dimensions,
                        }
                    )
            strict_dimensions.update(comparison["strict_improvement_dimensions"])
            outcome = "both_success"
        elif policy_success:
            counts["policy_only_success"] += 1
            if per_reset_rejections:
                comparison = {
                    "planner_nonworse_all_dimensions": False,
                    "strict_improvement_dimensions": [],
                    "exact_tie": False,
                    "dimensions": [],
                }
                outcome = "policy_only_success_formal_gate_rejection"
            else:
                rescue = f"success.safe_planner_failure_rescue.reset_{index:02d}"
                strict_dimensions.add(rescue)
                comparison = {
                    "planner_nonworse_all_dimensions": True,
                    "strict_improvement_dimensions": [rescue],
                    "exact_tie": False,
                    "dimensions": [],
                }
                outcome = "safe_policy_rescue"
        elif planner_success:
            counts["planner_only_success"] += 1
            comparison = {
                "planner_nonworse_all_dimensions": False,
                "strict_improvement_dimensions": [],
                "exact_tie": False,
                "dimensions": [],
            }
            outcome = "planner_only_success"
        else:
            counts["neither_success"] += 1
            comparison = {
                "planner_nonworse_all_dimensions": True,
                "strict_improvement_dimensions": [],
                "exact_tie": True,
                "dimensions": [],
            }
            outcome = "neither_success"
        per_reset.append(
            {
                "reset_index": index,
                "episode_id": episode_id,
                "reset_identity_sha256": reset_identities[index],
                "policy_record_sha256": _value_sha256(policy),
                "planner_record_sha256": _value_sha256(planner),
                "outcome": outcome,
                "policy": {
                    "success": policy_success,
                    "safety_failure": policy_safety,
                    "quality_v3_gate": policy_gate_payload,
                    "action_application": policy_causal,
                    "attempt_tape": policy_tape,
                },
                "planner": {
                    "success": planner_success,
                    "safety_failure": planner_safety,
                    "quality_v3_gate": planner_gate_payload,
                    "action_application": planner_causal,
                    "attempt_tape": planner_tape,
                },
                "comparison": comparison,
                "rejection_reasons": per_reset_rejections,
            }
        )
        rejection_reasons.extend(per_reset_rejections)

    all_successful_policy_quality_v3_gates_passed = all(
        row["policy"]["quality_v3_gate"]["passed"] is True
        for row in per_reset
        if row["policy"]["success"]
    )
    if (
        policy_evaluation.get("all_successful_quality_v2_gates_passed")
        is not all_successful_policy_quality_v3_gates_passed
    ):
        raise ValueError(
            "policy evaluation successful-Qv3 aggregate disagrees with audited records"
        )
    if counts["policy_safety_failure"] > counts["planner_safety_failure"]:
        rejection_reasons.append(
            {
                "code": "aggregate_policy_safety_count_worse",
                "scope": "aggregate",
                "policy_failure_count": counts["policy_safety_failure"],
                "planner_failure_count": counts["planner_safety_failure"],
            }
        )
    if counts["policy_safety_failure"] < counts["planner_safety_failure"]:
        strict_dimensions.add("safety.failure_count")
    if rejection_reasons:
        decision = "keep_planner"
        selection_reason = "formal_gate_rejection"
        planner_nonworse_all_dimensions = False
        exact_tie = False
    elif strict_dimensions:
        decision = "promote"
        selection_reason = "strict_planner_nonworse_improvement"
        planner_nonworse_all_dimensions = True
        exact_tie = False
    else:
        decision = "keep_planner"
        selection_reason = "no_strict_improvement"
        planner_nonworse_all_dimensions = True
        exact_tie = True

    from examples.embodiment import (
        audit_dynamic_benchmark_optimal_trajectories as optimal_auditor,
    )

    source_files = {
        "policy_evaluator": {
            "path": str(policy_evaluator_source_path.resolve()),
            "sha256": hashes["policy_evaluator_source"],
        },
        "planner_evaluator": {
            "path": str(planner_evaluator_source_path.resolve()),
            "sha256": hashes["planner_evaluator_source"],
        },
        "qv3_comparator": {
            "path": str(Path(optimal.__file__).resolve()),
            "sha256": _sha256(Path(optimal.__file__).resolve()),
        },
        "attempt_auditor": {
            "path": str(Path(optimal_auditor.__file__).resolve()),
            "sha256": _sha256(Path(optimal_auditor.__file__).resolve()),
        },
        "promotion_builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    source_identity = {
        "rlinf_commit": rlinf_commit,
        "benchmark_commit": benchmark_commit,
        "evaluator_rlinf_commit": evaluator_commit,
        "files": source_files,
    }
    source_identity["sha256"] = _value_sha256(source_identity)
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA,
        "task_id": task,
        "candidate_id": candidate_id,
        "inference": {
            "mode": "deterministic_mean",
            "deterministic": True,
            "action_noise": False,
            "residual_scale": residual_scale,
            "basis": {
                "evaluator_schema": POLICY_EVALUATION_SCHEMA,
                "evaluator_source_sha256": hashes["policy_evaluator_source"],
            },
        },
        "inputs": {
            "policy": {
                **_artifact_identity(policy_path, sha256=hashes["policy"]),
                "metadata_payload_sha256": checkpoint["payload_sha256"],
                "checkpoint_role": "best",
                "training_seed": training_seed,
                "env_steps": env_steps,
                "run_tag": run_tag,
            },
            "policy_metadata": dict(learned_policy_admission["trainer_summary"]),
            "checkpoint_selection": dict(
                learned_policy_admission["checkpoint_selection"]
            ),
            "policy_evaluation": _artifact_identity(
                policy_evaluation_path,
                sha256=hashes["policy_evaluation"],
                schema_version=POLICY_EVALUATION_SCHEMA,
                payload_sha256=policy_evaluation["payload_sha256"],
            ),
            "planner_evaluation": _artifact_identity(
                planner_evaluation_path,
                sha256=hashes["planner_evaluation"],
                schema_version=PLANNER_EVALUATION_SCHEMA,
                payload_sha256=planner_evaluation["payload_sha256"],
            ),
            "reset_manifest": {
                **_artifact_identity(reset_manifest_path, sha256=reset_sha),
                "payload_sha256": _value_sha256(reset_rows),
                "split": "validation",
                "manifest_seed": policy_manifest_seed,
                "reset_count": RESET_COUNT,
            },
            "quality_v3_thresholds": _artifact_identity(
                quality_v2_thresholds_path,
                sha256=hashes["quality_v2_thresholds"],
                schema_version=QUALITY_V2_THRESHOLD_SCHEMA,
                payload_sha256=_payload_sha256(threshold_payload),
            ),
            "quality_v2_calibration_wave_receipt": {
                **_artifact_identity(
                    quality_v2_calibration_wave_receipt_path,
                    sha256=hashes["quality_v2_calibration_wave_receipt"],
                    schema_version=(optimal.QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA),
                    payload_sha256=hashes["quality_v2_calibration_wave_receipt"],
                ),
                "dataset_relative_path": calibration_receipt.relative_path,
            },
            "selector_contract": _artifact_identity(
                selector_contract_path,
                sha256=hashes["selector_contract"],
                schema_version=SELECTOR_SCHEMA,
                payload_sha256=_payload_sha256(selector_payload),
            ),
            "attempt_artifacts": {
                "schema_version": optimal_auditor.ATTEMPT_SCHEMA,
                "path_base": "evaluation_parent",
                "policy_tape_count": len(per_reset),
                "planner_tape_count": len(per_reset),
                "policy_payload_sha256": _value_sha256(
                    [row["policy"]["attempt_tape"] for row in per_reset]
                ),
                "planner_payload_sha256": _value_sha256(
                    [row["planner"]["attempt_tape"] for row in per_reset]
                ),
            },
        },
        "source_identity": source_identity,
        "image_identity": {"reference": image_reference, "sha256": image_sha256},
        "per_reset": per_reset,
        "aggregate": {
            "counts": counts,
            "success": {
                "policy": _wilson_interval(counts["policy_success"], RESET_COUNT),
                "planner": _wilson_interval(counts["planner_success"], RESET_COUNT),
            },
            "safety_failure": {
                "policy": _wilson_interval(
                    counts["policy_safety_failure"], RESET_COUNT
                ),
                "planner": _wilson_interval(
                    counts["planner_safety_failure"], RESET_COUNT
                ),
            },
            "planner_nonworse_all_both_success": (planner_nonworse_all_both_success),
            "all_successful_policy_quality_v3_gates_passed": (
                all_successful_policy_quality_v3_gates_passed
            ),
            "all_successful_policy_t5_causal_gates_passed": all(
                row["policy"]["action_application"]["t5_replan_causal_timing_passed"]
                is not False
                for row in per_reset
                if row["policy"]["success"]
            ),
            "strict_improvement_dimensions": sorted(strict_dimensions),
        },
        "selection": {
            "decision": decision,
            "reason": selection_reason,
            "planner_nonworse_all_dimensions": planner_nonworse_all_dimensions,
            "strict_improvement_dimensions": sorted(strict_dimensions),
            "rejection_reasons": rejection_reasons,
            "exact_aggregate_tie": exact_tie,
            "rule": (
                "all integrity evidence must validate; any formal safety, success, Qv3, T5 "
                "causal, or paired nonworse regression keeps the planner; otherwise at least "
                "one strict planner-nonworse improvement is required for promotion"
            ),
        },
    }
    evidence["payload_sha256"] = _payload_sha256(evidence)
    return evidence


def _input_path(evidence: Mapping[str, Any], name: str) -> Path:
    inputs = _require_mapping(evidence.get("inputs"), "selection evidence inputs")
    identity = _require_mapping(inputs.get(name), f"selection evidence {name}")
    return Path(
        _require_string(identity.get("path"), f"selection evidence {name} path")
    )


def _input_hashes(evidence: Mapping[str, Any]) -> dict[str, str]:
    inputs = _require_mapping(evidence.get("inputs"), "selection evidence inputs")
    source = _require_mapping(
        evidence.get("source_identity"), "evidence source identity"
    )
    files = _require_mapping(source.get("files"), "evidence source files")
    return {
        "policy": _require_sha256(
            _require_mapping(inputs.get("policy"), "evidence policy").get("sha256"),
            "evidence policy SHA-256",
        ),
        "policy_metadata": _require_sha256(
            _require_mapping(
                inputs.get("policy_metadata"), "evidence policy metadata"
            ).get("sha256"),
            "evidence policy metadata SHA-256",
        ),
        "checkpoint_selection": _require_sha256(
            _require_mapping(
                inputs.get("checkpoint_selection"),
                "evidence checkpoint-selection manifest",
            ).get("sha256"),
            "evidence checkpoint-selection SHA-256",
        ),
        "policy_evaluation": _require_sha256(
            _require_mapping(
                inputs.get("policy_evaluation"), "evidence policy evaluation"
            ).get("sha256"),
            "evidence policy evaluation SHA-256",
        ),
        "planner_evaluation": _require_sha256(
            _require_mapping(
                inputs.get("planner_evaluation"), "evidence planner evaluation"
            ).get("sha256"),
            "evidence planner evaluation SHA-256",
        ),
        "reset_manifest": _require_sha256(
            _require_mapping(
                inputs.get("reset_manifest"), "evidence reset manifest"
            ).get("sha256"),
            "evidence reset manifest SHA-256",
        ),
        "quality_v2_thresholds": _require_sha256(
            _require_mapping(
                inputs.get("quality_v3_thresholds"), "evidence thresholds"
            ).get("sha256"),
            "evidence thresholds SHA-256",
        ),
        "quality_v2_calibration_wave_receipt": _require_sha256(
            _require_mapping(
                inputs.get("quality_v2_calibration_wave_receipt"),
                "evidence calibration wave receipt",
            ).get("sha256"),
            "evidence calibration wave receipt SHA-256",
        ),
        "selector_contract": _require_sha256(
            _require_mapping(inputs.get("selector_contract"), "evidence selector").get(
                "sha256"
            ),
            "evidence selector SHA-256",
        ),
        "policy_evaluator_source": _require_sha256(
            _require_mapping(
                files.get("policy_evaluator"), "policy evaluator source"
            ).get("sha256"),
            "policy evaluator source SHA-256",
        ),
        "planner_evaluator_source": _require_sha256(
            _require_mapping(
                files.get("planner_evaluator"), "planner evaluator source"
            ).get("sha256"),
            "planner evaluator source SHA-256",
        ),
    }


def validate_selection_evidence_artifacts(
    evidence: Mapping[str, Any],
    *,
    expected_task: str | None = None,
    expected_candidate_id: str | None = None,
    expected_policy_path: str | None = None,
    expected_policy_sha256: str | None = None,
    expected_threshold_sha256: str | None = None,
    expected_calibration_receipt_sha256: str | None = None,
    expected_selector_sha256: str | None = None,
    expected_rlinf_commit: str | None = None,
    expected_benchmark_commit: str | None = None,
    expected_policy_evaluator_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute a selection evidence payload from every path it binds."""

    evidence = dict(evidence)
    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("selection evidence schema mismatch")
    _verify_payload_hash(evidence, "selection evidence")
    inputs = _require_mapping(evidence.get("inputs"), "selection evidence inputs")
    policy = _require_mapping(inputs.get("policy"), "selection evidence policy")
    inference = _require_mapping(
        evidence.get("inference"), "selection evidence inference"
    )
    source = _require_mapping(
        evidence.get("source_identity"), "evidence source identity"
    )
    files = _require_mapping(source.get("files"), "evidence source files")
    image = _require_mapping(evidence.get("image_identity"), "evidence image identity")
    hashes = _input_hashes(evidence)
    recomputed = build_selection_evidence(
        candidate_id=str(evidence.get("candidate_id")),
        run_tag=str(policy.get("run_tag")),
        policy_path=_input_path(evidence, "policy"),
        policy_metadata_path=_input_path(evidence, "policy_metadata"),
        checkpoint_selection_path=_input_path(evidence, "checkpoint_selection"),
        policy_evaluation_path=_input_path(evidence, "policy_evaluation"),
        planner_evaluation_path=_input_path(evidence, "planner_evaluation"),
        reset_manifest_path=_input_path(evidence, "reset_manifest"),
        quality_v2_thresholds_path=_input_path(evidence, "quality_v3_thresholds"),
        quality_v2_calibration_wave_receipt_path=_input_path(
            evidence, "quality_v2_calibration_wave_receipt"
        ),
        selector_contract_path=_input_path(evidence, "selector_contract"),
        policy_evaluator_source_path=Path(
            str(
                _require_mapping(files.get("policy_evaluator"), "policy evaluator")[
                    "path"
                ]
            )
        ),
        planner_evaluator_source_path=Path(
            str(
                _require_mapping(files.get("planner_evaluator"), "planner evaluator")[
                    "path"
                ]
            )
        ),
        image_reference=str(image.get("reference")),
        image_sha256=str(image.get("sha256")),
        expected_sha256=hashes,
        expected_residual_scale=inference.get("residual_scale"),
    )
    if recomputed != evidence:
        raise ValueError(
            "selection evidence does not equal artifact-level recomputation"
        )
    if expected_task is not None and evidence.get("task_id") != expected_task:
        raise ValueError("selection evidence task mismatch")
    if (
        expected_candidate_id is not None
        and evidence.get("candidate_id") != expected_candidate_id
    ):
        raise ValueError("selection evidence candidate mismatch")
    if expected_policy_path is not None and policy.get("path") != expected_policy_path:
        accepted = {expected_policy_path, str(Path(expected_policy_path).resolve())}
        if policy.get("path") not in accepted:
            raise ValueError("selection evidence policy path mismatch")
    if (
        expected_policy_sha256 is not None
        and hashes["policy"] != expected_policy_sha256
    ):
        raise ValueError("selection evidence policy SHA-256 mismatch")
    if (
        expected_threshold_sha256 is not None
        and hashes["quality_v2_thresholds"] != expected_threshold_sha256
    ):
        raise ValueError("selection evidence threshold SHA-256 mismatch")
    if (
        expected_calibration_receipt_sha256 is not None
        and hashes["quality_v2_calibration_wave_receipt"]
        != expected_calibration_receipt_sha256
    ):
        raise ValueError("selection evidence calibration receipt SHA-256 mismatch")
    if (
        expected_selector_sha256 is not None
        and hashes["selector_contract"] != expected_selector_sha256
    ):
        raise ValueError("selection evidence selector SHA-256 mismatch")
    if (
        expected_rlinf_commit is not None
        and source.get("rlinf_commit") != expected_rlinf_commit
    ):
        raise ValueError("selection evidence RLinf commit mismatch")
    if (
        expected_benchmark_commit is not None
        and source.get("benchmark_commit") != expected_benchmark_commit
    ):
        raise ValueError("selection evidence benchmark commit mismatch")
    if (
        expected_policy_evaluator_source_sha256 is not None
        and hashes["policy_evaluator_source"] != expected_policy_evaluator_source_sha256
    ):
        raise ValueError("selection evidence evaluator source SHA-256 mismatch")
    return evidence


def build_promotion_receipt(
    evidence: Mapping[str, Any],
    *,
    evidence_path: Path,
    evidence_file_sha256: str,
) -> dict[str, Any]:
    """Build the v0.2 receipt exclusively from validated selection evidence."""

    validated = validate_selection_evidence_artifacts(evidence)
    inputs = _require_mapping(validated["inputs"], "selection evidence inputs")
    policy = _require_mapping(inputs["policy"], "selection evidence policy")
    policy_evaluation = _require_mapping(
        inputs["policy_evaluation"], "selection evidence policy evaluation"
    )
    planner_evaluation = _require_mapping(
        inputs["planner_evaluation"], "selection evidence planner evaluation"
    )
    attempt_artifacts = _require_mapping(
        inputs["attempt_artifacts"], "selection evidence attempt artifacts"
    )
    reset = _require_mapping(
        inputs["reset_manifest"], "selection evidence reset manifest"
    )
    thresholds = _require_mapping(
        inputs["quality_v3_thresholds"], "selection evidence thresholds"
    )
    calibration_receipt = _require_mapping(
        inputs["quality_v2_calibration_wave_receipt"],
        "selection evidence calibration wave receipt",
    )
    selector = _require_mapping(
        inputs["selector_contract"], "selection evidence selector"
    )
    inference = _require_mapping(validated["inference"], "selection evidence inference")
    source = _require_mapping(validated["source_identity"], "selection evidence source")
    selection = _require_mapping(validated["selection"], "selection evidence decision")
    basis = _require_mapping(inference.get("basis"), "selection inference basis")
    receipt: dict[str, Any] = {
        "schema_version": PROMOTION_SCHEMA,
        "task_id": validated["task_id"],
        "candidate_id": validated["candidate_id"],
        "policy": {
            "path": policy["path"],
            "sha256": policy["sha256"],
            "seed": policy["training_seed"],
            "run_tag": policy["run_tag"],
            "checkpoint_role": "best",
            "env_steps": policy["env_steps"],
            "metadata_path": inputs["policy_metadata"]["path"],
            "metadata_sha256": inputs["policy_metadata"]["sha256"],
            "metadata_payload_sha256": inputs["policy_metadata"]["payload_sha256"],
        },
        "inference": {
            "residual_scale": inference["residual_scale"],
            "deterministic": True,
            "action_noise": False,
        },
        "validation_receipt": {
            "path": policy_evaluation["path"],
            "sha256": policy_evaluation["sha256"],
            "payload_sha256": policy_evaluation["payload_sha256"],
            "evaluator_schema": policy_evaluation["schema_version"],
            "evaluator_source_sha256": basis["evaluator_source_sha256"],
            "partition": "validation",
            "reset_manifest_path": reset["path"],
            "reset_manifest_sha256": reset["sha256"],
            "test_exposure": False,
            "review_exposure": False,
            "calibration_exposure": False,
            "quality_threshold_schema": thresholds["schema_version"],
            "quality_threshold_sha256": thresholds["sha256"],
            "attempt_schema_version": attempt_artifacts["schema_version"],
            "all_successful_quality_gates_passed": validated["aggregate"][
                "all_successful_policy_quality_v3_gates_passed"
            ],
            "all_successful_t5_causal_gates_passed": validated["aggregate"][
                "all_successful_policy_t5_causal_gates_passed"
            ],
        },
        "quality_v2_calibration_wave_receipt": dict(calibration_receipt),
        "selection": {
            "decision": selection["decision"],
            "reason": selection["reason"],
            "planner_nonworse_all_dimensions": selection[
                "planner_nonworse_all_dimensions"
            ],
            "strict_improvement_dimensions": selection["strict_improvement_dimensions"],
            "rejection_reasons": selection["rejection_reasons"],
            "selector_contract_path": selector["path"],
            "selector_contract_sha256": selector["sha256"],
            "planner_evaluation_path": planner_evaluation["path"],
            "planner_evaluation_sha256": planner_evaluation["sha256"],
            "planner_evaluation_payload_sha256": planner_evaluation["payload_sha256"],
            "attempt_artifacts_payload_sha256": _value_sha256(attempt_artifacts),
            "evidence_path": str(evidence_path.resolve()),
            "evidence_sha256": _require_sha256(
                evidence_file_sha256, "selection evidence file SHA-256"
            ),
            "evidence_payload_sha256": validated["payload_sha256"],
        },
        "source_identity": dict(source),
        "image_identity": dict(validated["image_identity"]),
    }
    receipt["payload_sha256"] = _payload_sha256(receipt)
    return receipt


def _write_new(path: Path, body: bytes) -> None:
    """Create ``path`` atomically without ever replacing an existing target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_promotion_artifacts(
    *,
    evidence: Mapping[str, Any],
    evidence_path: Path,
    receipt_path: Path,
    require_promote: bool,
) -> dict[str, Any]:
    """Seal the evidence and receipt with exclusive, canonical writes."""

    if evidence_path.resolve() == receipt_path.resolve():
        raise ValueError("evidence and receipt paths must be different")
    if evidence_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite promotion artifacts")
    evidence_body = _json_bytes(evidence)
    evidence_file_sha = hashlib.sha256(evidence_body).hexdigest()
    receipt = build_promotion_receipt(
        evidence,
        evidence_path=evidence_path,
        evidence_file_sha256=evidence_file_sha,
    )
    selection = _require_mapping(receipt.get("selection"), "promotion selection")
    if require_promote and selection.get("decision") != "promote":
        raise RuntimeError(
            "--require-promote was requested but selection keeps the planner: "
            + str(selection.get("reason"))
        )
    receipt_body = _json_bytes(receipt)
    _write_new(evidence_path, evidence_body)
    try:
        _write_new(receipt_path, receipt_body)
    except BaseException:
        # This path was created by this call and has not been handed off yet.
        evidence_path.unlink(missing_ok=True)
        raise
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--policy-metadata", type=Path, required=True)
    parser.add_argument("--expected-policy-metadata-sha256", required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-selection-sha256", required=True)
    parser.add_argument("--policy-evaluation", type=Path, required=True)
    parser.add_argument("--expected-policy-evaluation-sha256", required=True)
    parser.add_argument("--planner-evaluation", type=Path, required=True)
    parser.add_argument("--expected-planner-evaluation-sha256", required=True)
    parser.add_argument("--reset-manifest", type=Path, required=True)
    parser.add_argument("--expected-reset-manifest-sha256", required=True)
    parser.add_argument("--quality-v2-thresholds", type=Path, required=True)
    parser.add_argument("--expected-quality-v2-thresholds-sha256", required=True)
    parser.add_argument(
        "--quality-v2-calibration-wave-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--expected-quality-v2-calibration-wave-receipt-sha256", required=True
    )
    parser.add_argument("--selector-contract", type=Path, required=True)
    parser.add_argument("--expected-selector-contract-sha256", required=True)
    parser.add_argument("--policy-evaluator-source", type=Path, required=True)
    parser.add_argument("--expected-policy-evaluator-source-sha256", required=True)
    parser.add_argument("--planner-evaluator-source", type=Path, required=True)
    parser.add_argument("--expected-planner-evaluator-source-sha256", required=True)
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--expected-residual-scale", type=float)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--require-promote", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    evidence = build_selection_evidence(
        candidate_id=args.candidate_id,
        run_tag=args.run_tag,
        policy_path=args.policy,
        policy_metadata_path=args.policy_metadata,
        checkpoint_selection_path=args.checkpoint_selection,
        policy_evaluation_path=args.policy_evaluation,
        planner_evaluation_path=args.planner_evaluation,
        reset_manifest_path=args.reset_manifest,
        quality_v2_thresholds_path=args.quality_v2_thresholds,
        quality_v2_calibration_wave_receipt_path=(
            args.quality_v2_calibration_wave_receipt
        ),
        selector_contract_path=args.selector_contract,
        policy_evaluator_source_path=args.policy_evaluator_source,
        planner_evaluator_source_path=args.planner_evaluator_source,
        image_reference=args.image_reference,
        image_sha256=args.image_sha256,
        expected_sha256={
            "policy": args.expected_policy_sha256,
            "policy_metadata": args.expected_policy_metadata_sha256,
            "checkpoint_selection": args.expected_checkpoint_selection_sha256,
            "policy_evaluation": args.expected_policy_evaluation_sha256,
            "planner_evaluation": args.expected_planner_evaluation_sha256,
            "reset_manifest": args.expected_reset_manifest_sha256,
            "quality_v2_thresholds": args.expected_quality_v2_thresholds_sha256,
            "quality_v2_calibration_wave_receipt": (
                args.expected_quality_v2_calibration_wave_receipt_sha256
            ),
            "selector_contract": args.expected_selector_contract_sha256,
            "policy_evaluator_source": args.expected_policy_evaluator_source_sha256,
            "planner_evaluator_source": args.expected_planner_evaluator_source_sha256,
        },
        expected_residual_scale=args.expected_residual_scale,
    )
    receipt = write_promotion_artifacts(
        evidence=evidence,
        evidence_path=args.evidence_output,
        receipt_path=args.receipt_output,
        require_promote=args.require_promote,
    )
    print(
        _canonical_json(
            {
                "decision": receipt["selection"]["decision"],
                "evidence_path": str(args.evidence_output.resolve()),
                "evidence_sha256": receipt["selection"]["evidence_sha256"],
                "evidence_payload_sha256": receipt["selection"][
                    "evidence_payload_sha256"
                ],
                "receipt_path": str(args.receipt_output.resolve()),
                "receipt_sha256": _sha256(args.receipt_output),
                "receipt_payload_sha256": receipt["payload_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
