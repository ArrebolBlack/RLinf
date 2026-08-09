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

"""Fail-closed admission of a trainer-selected learned policy checkpoint."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "rlinf-dynamic-benchmark-expert-policy-v0.1"
TRAINER_SUMMARY_SCHEMA = "rlinf-dynamic-benchmark-expert-summary-v0.1"
CHECKPOINT_SELECTION_SCHEMA = "rlinf-dynamic-benchmark-checkpoint-selection-v0.1"
CHECKPOINT_SELECTION_RUN_SCHEMA = (
    "rlinf-dynamic-benchmark-checkpoint-selection-run-v0.1"
)

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    try:
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not finite canonical JSON") from error
    return payload


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


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


def _json_safe_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not finite canonical JSON") from error


def _canonical_artifact(path: Path, name: str, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = path.resolve(strict=True)
    if resolved.name != name:
        raise ValueError(f"{label} must be the canonical trainer artifact {name}")
    return resolved


def _verify_payload_hash(
    payload: Mapping[str, Any],
    *,
    label: str,
    canonical_sha256: Any,
) -> str:
    claimed = _require_sha256(payload.get("payload_sha256"), f"{label} payload SHA-256")
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("payload_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError(f"{label} payload SHA-256 does not recompute")
    return claimed


def validate_selected_learned_policy(
    *,
    policy_path: Path,
    trainer_summary_path: Path,
    checkpoint_selection_path: Path,
    expected_policy_sha256: str | None = None,
    expected_trainer_summary_sha256: str | None = None,
    expected_checkpoint_selection_sha256: str | None = None,
    expected_rlinf_commit: str | None = None,
    expected_benchmark_commit: str | None = None,
) -> dict[str, Any]:
    """Reopen a trainer run and admit only its selected eligible snapshot.

    The trainer's own selection ledger is replayed so a re-sealed manifest cannot
    change eligibility, ordering, snapshot contents, or the selected best-policy
    copy. The returned identity is JSON-safe and can be embedded in an evaluator
    artifact without treating the planner fallback as a learned checkpoint.
    """

    from examples.embodiment import train_dynamic_benchmark_expert as trainer

    summary_path = _canonical_artifact(
        trainer_summary_path, "summary.json", "trainer summary"
    )
    selection_path = _canonical_artifact(
        checkpoint_selection_path,
        "checkpoint_selection.json",
        "checkpoint-selection manifest",
    )
    output = summary_path.parent
    if selection_path.parent != output:
        raise ValueError(
            "trainer summary and checkpoint-selection manifest must be siblings"
        )

    # Resolve without requiring existence so the no-eligible fallback produces an
    # explicit selection error instead of being mistaken for a missing learned file.
    resolved_policy = policy_path.resolve(strict=False)
    if resolved_policy != output / "best_policy.pt":
        raise ValueError(
            "formal learned-policy admission requires checkpoint_role=best at "
            "best_policy.pt"
        )

    summary = _read_json(summary_path, "trainer summary")
    manifest = _read_json(selection_path, "checkpoint-selection manifest")
    summary_file_sha256 = _sha256(summary_path)
    selection_file_sha256 = _sha256(selection_path)
    if expected_trainer_summary_sha256 is not None and summary_file_sha256 != (
        _require_sha256(
            expected_trainer_summary_sha256, "expected trainer summary SHA-256"
        )
    ):
        raise ValueError("trainer summary SHA-256 mismatch")
    if expected_checkpoint_selection_sha256 is not None and selection_file_sha256 != (
        _require_sha256(
            expected_checkpoint_selection_sha256,
            "expected checkpoint-selection SHA-256",
        )
    ):
        raise ValueError("checkpoint-selection manifest SHA-256 mismatch")

    if summary.get("schema_version") != TRAINER_SUMMARY_SCHEMA:
        raise ValueError("trainer summary schema does not match")
    summary_payload_sha256 = _verify_payload_hash(
        summary,
        label="trainer summary",
        canonical_sha256=trainer._canonical_json_sha256,
    )
    if summary.get("status") != "complete":
        raise ValueError("formal learned-policy admission requires complete training")
    if manifest.get("schema_version") != CHECKPOINT_SELECTION_SCHEMA:
        raise ValueError("checkpoint-selection manifest schema does not match")
    selection_payload_sha256 = _verify_payload_hash(
        manifest,
        label="checkpoint-selection manifest",
        canonical_sha256=trainer._canonical_json_sha256,
    )

    selection = _require_mapping(
        manifest.get("selection"), "checkpoint-selection result"
    )
    status = selection.get("status")
    if status == "planner_fallback_no_eligible":
        raise ValueError(
            "planner_fallback_no_eligible is not a learned policy checkpoint"
        )
    if status != "selected_eligible_snapshot":
        raise ValueError("checkpoint-selection status is not an eligible snapshot")
    eligible_count = _require_int(
        selection.get("eligible_snapshot_count"),
        "checkpoint-selection eligible snapshot count",
        minimum=1,
    )
    selected_identity = selection.get("selected_snapshot_identity")
    best_policy = selection.get("best_policy")
    if selected_identity is None:
        raise ValueError("checkpoint-selection selected snapshot is null")
    if best_policy is None:
        raise ValueError("checkpoint-selection best policy is null")
    selected_identity = _require_mapping(
        selected_identity, "checkpoint-selection selected snapshot"
    )
    best_policy = _require_mapping(best_policy, "checkpoint-selection best policy")
    selected_env_steps = _require_int(
        selected_identity.get("env_steps"), "selected snapshot env_steps", minimum=1
    )
    if (
        best_policy.get("path") != "best_policy.pt"
        or _require_int(
            best_policy.get("env_steps"), "best policy env_steps", minimum=1
        )
        != selected_env_steps
    ):
        raise ValueError("checkpoint-selection best policy identity is not canonical")

    rows = manifest.get("evaluated_snapshots")
    if not isinstance(rows, list) or not rows:
        raise ValueError("checkpoint-selection snapshot ledger is empty")
    eligible_rows = 0
    selected_rows: list[Mapping[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"checkpoint-selection row {index}")
        _require_int(
            row.get("env_steps"),
            f"checkpoint-selection row {index} env_steps",
            minimum=1,
        )
        if not isinstance(row.get("eligible"), bool):
            raise ValueError(
                f"checkpoint-selection row {index} eligible must be boolean"
            )
        if not isinstance(row.get("selected"), bool):
            raise ValueError(
                f"checkpoint-selection row {index} selected must be boolean"
            )
        eligible_rows += int(row["eligible"])
        if row["selected"]:
            selected_rows.append(row)
    if eligible_rows != eligible_count:
        raise ValueError("checkpoint-selection eligible snapshot count was tampered")
    if len(selected_rows) != 1 or selected_rows[0].get("eligible") is not True:
        raise ValueError(
            "checkpoint-selection must select exactly one eligible snapshot"
        )
    selected_row = selected_rows[0]
    selected_policy = _require_mapping(
        selected_row.get("policy"), "selected checkpoint policy"
    )
    expected_selected_identity = {
        "env_steps": selected_row["env_steps"],
        "policy_path": selected_policy.get("path"),
        "policy_sha256": selected_policy.get("sha256"),
        "validation_metrics_sha256": selected_row.get("validation_metrics_sha256"),
    }
    if dict(selected_identity) != expected_selected_identity:
        raise ValueError("checkpoint-selection selected snapshot identity was tampered")
    expected_best_policy = {**dict(selected_policy), "path": "best_policy.pt"}
    if dict(best_policy) != expected_best_policy:
        raise ValueError("checkpoint-selection best policy identity was tampered")

    if not resolved_policy.is_file():
        raise FileNotFoundError(resolved_policy)
    policy_sha256 = _sha256(resolved_policy)
    if expected_policy_sha256 is not None and policy_sha256 != _require_sha256(
        expected_policy_sha256, "expected policy SHA-256"
    ):
        raise ValueError("policy SHA-256 mismatch")
    try:
        import torch

        raw_checkpoint = torch.load(
            resolved_policy, map_location="cpu", weights_only=False
        )
    except Exception as error:
        raise ValueError(
            f"cannot load selected policy checkpoint {resolved_policy}"
        ) from error
    checkpoint = _require_mapping(raw_checkpoint, "selected policy checkpoint")
    if checkpoint.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("selected policy checkpoint schema does not match")
    checkpoint_env_steps = _require_int(
        checkpoint.get("env_steps"), "selected policy env_steps", minimum=1
    )
    if checkpoint_env_steps != selected_env_steps:
        raise ValueError("selected policy env_steps do not match checkpoint selection")
    if policy_sha256 != _require_sha256(
        best_policy.get("sha256"), "checkpoint-selection best policy SHA-256"
    ):
        raise ValueError("best_policy.pt does not match the selected eligible snapshot")
    config = _json_safe_copy(
        _require_mapping(checkpoint.get("config"), "selected policy config"),
        "selected policy config",
    )
    state_schema = _json_safe_copy(
        _require_mapping(
            checkpoint.get("state_schema"), "selected policy state schema"
        ),
        "selected policy state schema",
    )
    validation = _json_safe_copy(
        _require_mapping(checkpoint.get("validation"), "selected policy validation"),
        "selected policy validation",
    )
    if "infra_identity" not in checkpoint:
        raise ValueError("selected policy infra identity is missing")
    raw_infra_identity = checkpoint["infra_identity"]
    if raw_infra_identity is not None and not isinstance(raw_infra_identity, Mapping):
        raise ValueError("selected policy infra identity must be a mapping or null")
    infra_identity = _json_safe_copy(
        raw_infra_identity, "selected policy infra identity"
    )

    summary_config = _require_mapping(summary.get("config"), "trainer summary config")
    if "infra_identity" not in summary:
        raise ValueError("trainer summary infra identity is missing")
    if dict(summary_config) != config:
        raise ValueError("trainer summary config does not match selected policy")
    if summary["infra_identity"] != infra_identity:
        raise ValueError(
            "trainer summary infra identity does not match selected policy"
        )
    if summary.get("best_validation") != validation:
        raise ValueError(
            "trainer summary best validation does not match selected policy"
        )
    if summary.get("best_validation") != selected_row.get("validation_metrics"):
        raise ValueError("trainer summary best validation is not the selected snapshot")
    if summary.get("best_score") is None:
        raise ValueError("trainer summary best score is null")
    summary_best_score = _require_sequence(
        summary["best_score"], "trainer summary best score"
    )
    selected_score = _require_sequence(
        selected_row.get("selection_score"), "selected snapshot score"
    )
    if list(summary_best_score) != list(selected_score):
        raise ValueError(
            "trainer summary best score is not the selected snapshot score"
        )
    summary_env_steps = _require_int(
        summary.get("env_steps"), "trainer summary env_steps", minimum=1
    )
    if summary_env_steps < checkpoint_env_steps:
        raise ValueError("selected policy env_steps exceed trainer summary env_steps")

    config_payload_sha256 = trainer._canonical_json_sha256(config)
    if summary.get("config_sha256") != config_payload_sha256:
        raise ValueError("trainer summary config SHA-256 does not recompute")
    config_path = _canonical_artifact(
        output / "config.json", "config.json", "trainer config"
    )
    if _read_json(config_path, "trainer config") != config:
        raise ValueError("trainer config artifact does not match selected policy")
    config_file_sha256 = _sha256(config_path)
    if summary.get("config_file_sha256") != config_file_sha256:
        raise ValueError("trainer summary config-file SHA-256 does not recompute")

    task = _require_string(config.get("task"), "policy task")
    algorithm = _require_string(config.get("algorithm"), "policy algorithm")
    if algorithm != "residual_rlpd":
        raise ValueError("checkpoint-selected learned policy must use residual_rlpd")
    rlinf_commit = _require_commit(config.get("rlinf_commit"), "policy RLinf commit")
    benchmark_commit = _require_commit(
        config.get("benchmark_commit"), "policy benchmark commit"
    )
    if expected_rlinf_commit is not None and rlinf_commit != _require_commit(
        expected_rlinf_commit, "expected RLinf commit"
    ):
        raise ValueError("selected policy RLinf source identity mismatch")
    if expected_benchmark_commit is not None and benchmark_commit != _require_commit(
        expected_benchmark_commit, "expected benchmark commit"
    ):
        raise ValueError("selected policy benchmark source identity mismatch")
    training_seed = _require_int(config.get("seed"), "policy training seed")
    validation_manifest_seed = _require_int(
        config.get("validation_manifest_seed"),
        "checkpoint-selection validation manifest seed",
    )
    expected_run_identity = {
        "schema_version": CHECKPOINT_SELECTION_RUN_SCHEMA,
        "task": task,
        "algorithm": algorithm,
        "rlinf_commit": rlinf_commit,
        "benchmark_commit": benchmark_commit,
        "seed": training_seed,
        "validation_manifest_seed": validation_manifest_seed,
        "eval_episodes": _require_int(
            config.get("eval_episodes"), "checkpoint-selection eval_episodes", minimum=1
        ),
        "eval_num_envs": _require_int(
            config.get("eval_num_envs"), "checkpoint-selection eval_num_envs", minimum=1
        ),
        "config_sha256": config_file_sha256,
        "config_payload_sha256": config_payload_sha256,
        "state_schema_sha256": trainer._canonical_json_sha256(state_schema),
    }
    if manifest.get("run_identity") != expected_run_identity:
        raise ValueError("checkpoint-selection config/source identity does not match")

    expected_summary_reference = {
        "manifest_path": "checkpoint_selection.json",
        "manifest_payload_sha256": selection_payload_sha256,
        "status": "selected_eligible_snapshot",
        "eligible_snapshot_count": eligible_count,
        "selected_snapshot_identity": dict(selected_identity),
        "planner_fallback_policy": selection.get("planner_fallback_policy"),
    }
    if summary.get("checkpoint_selection") != expected_summary_reference:
        raise ValueError(
            "trainer summary and checkpoint-selection manifest identities diverged"
        )

    validation_metrics_sha256 = trainer._canonical_json_sha256(validation)
    if selected_row.get("validation_metrics_sha256") != validation_metrics_sha256:
        raise ValueError("selected policy validation identity does not match")
    if selected_policy.get("config_payload_sha256") != config_payload_sha256:
        raise ValueError("selected policy config identity does not match")
    if (
        selected_policy.get("state_schema_sha256")
        != expected_run_identity["state_schema_sha256"]
    ):
        raise ValueError("selected policy state-schema identity does not match")

    # Replay the trainer's complete selection verifier. This reopens the planner
    # baseline, every immutable snapshot, and best_policy.pt, then recomputes the
    # safety eligibility and lexicographic winner.
    try:
        ledger = trainer._CheckpointSelectionLedger(
            output, expected_run_identity, copy.deepcopy(manifest)
        )
        ledger._verify_manifest()
    except Exception as error:
        raise ValueError("checkpoint-selection ledger verification failed") from error

    if (
        _sha256(resolved_policy) != policy_sha256
        or _sha256(summary_path) != summary_file_sha256
        or _sha256(selection_path) != selection_file_sha256
        or _sha256(config_path) != config_file_sha256
    ):
        raise RuntimeError("trainer admission artifacts changed during validation")

    return {
        "checkpoint_role": "best",
        "policy": {
            "path": str(resolved_policy),
            "sha256": policy_sha256,
            "schema_version": POLICY_SCHEMA,
            "env_steps": checkpoint_env_steps,
            "config_sha256": config_payload_sha256,
            "state_schema_sha256": expected_run_identity["state_schema_sha256"],
            "validation_metrics_sha256": validation_metrics_sha256,
        },
        "trainer_summary": {
            "path": str(summary_path),
            "sha256": summary_file_sha256,
            "schema_version": TRAINER_SUMMARY_SCHEMA,
            "payload_sha256": summary_payload_sha256,
        },
        "checkpoint_selection": {
            "path": str(selection_path),
            "sha256": selection_file_sha256,
            "schema_version": CHECKPOINT_SELECTION_SCHEMA,
            "payload_sha256": selection_payload_sha256,
            "status": "selected_eligible_snapshot",
            "eligible_snapshot_count": eligible_count,
            "selected_snapshot_identity": copy.deepcopy(dict(selected_identity)),
        },
        "config": {
            "path": str(config_path),
            "sha256": config_file_sha256,
            "payload_sha256": config_payload_sha256,
        },
        "source_identity": {
            "task": task,
            "algorithm": algorithm,
            "training_seed": training_seed,
            "validation_manifest_seed": validation_manifest_seed,
            "rlinf_commit": rlinf_commit,
            "benchmark_commit": benchmark_commit,
        },
    }
