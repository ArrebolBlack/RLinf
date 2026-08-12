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

"""Strict validation for dataset-local calibration binding projections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

CALIBRATION_BINDING_PROJECTION_SCHEMA = (
    "rlinf-dynamic-benchmark-calibration-binding-projection-v0.2"
)
REVIEW_SCHEMA = "se3wam-rld2-cpu-planner-trajectory-review-v1"
PRODUCTION_AUTHORIZATION_RECEIPT_SCHEMA = (
    "se3wam-cpu-planner-production-authorization-receipt-v0.1"
)
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


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def projection_payload(
    binding: Mapping[str, Any],
    evaluator_benchmark_commit: str,
    approved_review_manifest: Mapping[str, Any],
    approved_review_file_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": CALIBRATION_BINDING_PROJECTION_SCHEMA,
        "evaluator_benchmark_commit": evaluator_benchmark_commit,
        "calibration_wave_receipt_binding": dict(binding),
        "production_authorization_receipt": production_authorization_receipt(
            approved_review_manifest, approved_review_file_sha256
        ),
    }


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return value


def production_authorization_receipt(
    review: Mapping[str, Any], review_file_sha256: str
) -> dict[str, Any]:
    """Project the large review artifact to the exact production authorization facts."""

    if review.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("projection approved review schema mismatch")
    units = review.get("units")
    if not isinstance(units, list):
        raise ValueError("projection approved review unit inventory is missing")
    unit_by_task = {
        unit.get("task"): unit for unit in units if isinstance(unit, Mapping)
    }
    if len(unit_by_task) != len(units):
        raise ValueError("projection approved review has duplicate or invalid units")
    source = review.get("source_identity")
    authorization = (
        source.get("production_authorization") if isinstance(source, Mapping) else None
    )
    if not isinstance(authorization, Mapping):
        raise ValueError("projection approved review has no production authorization")
    return {
        "schema_version": PRODUCTION_AUTHORIZATION_RECEIPT_SCHEMA,
        "review_schema_version": REVIEW_SCHEMA,
        "review_file_sha256": _sha256(
            review_file_sha256, "projection approved review file SHA-256"
        ),
        "review_payload_sha256": _sha256(
            review.get("payload_sha256"), "projection approved review payload SHA-256"
        ),
        "counts": dict(review.get("counts", {})),
        "owner_review_complete": review.get("owner_review_complete"),
        "training_data_authorized": review.get("training_data_authorized"),
        "full_trajectory_generation": review.get("full_trajectory_generation"),
        "promotion_authorized": review.get("promotion_authorized"),
        "production_authorization": dict(authorization),
        "units": [
            {
                "task": task,
                "review_status": unit_by_task.get(task, {}).get("review_status"),
                "owner_decision": (
                    unit_by_task.get(task, {}).get("owner_review", {}).get("decision")
                    if isinstance(unit_by_task.get(task, {}).get("owner_review"), Mapping)
                    else None
                ),
            }
            for task in EXACT_TASKS
        ],
    }


def _validate_authorization_receipt(
    receipt: Any, evaluator_benchmark_commit: str
) -> None:
    expected_fields = {
        "schema_version",
        "review_schema_version",
        "review_file_sha256",
        "review_payload_sha256",
        "counts",
        "owner_review_complete",
        "training_data_authorized",
        "full_trajectory_generation",
        "promotion_authorized",
        "production_authorization",
        "units",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
        raise ValueError("projection production authorization receipt field inventory mismatch")
    if (
        receipt.get("schema_version") != PRODUCTION_AUTHORIZATION_RECEIPT_SCHEMA
        or receipt.get("review_schema_version") != REVIEW_SCHEMA
    ):
        raise ValueError("projection production authorization receipt schema mismatch")
    _sha256(receipt.get("review_file_sha256"), "approved review file SHA-256")
    _sha256(receipt.get("review_payload_sha256"), "approved review payload SHA-256")
    if receipt.get("counts") != {
        "approved": 14,
        "pending_owner_review": 0,
        "rejected": 0,
        "revise": 0,
        "tasks": 14,
    }:
        raise ValueError("projection authorization receipt is not exact 14/14")
    for flag in (
        "owner_review_complete",
        "training_data_authorized",
        "full_trajectory_generation",
        "promotion_authorized",
    ):
        if receipt.get(flag) is not True:
            raise ValueError(f"projection authorization receipt has no {flag}")
    authorization = receipt.get("production_authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("projection authorization receipt has no production authorization")
    if set(authorization) != {
        "approved_planner_master_commit",
        "authorized_on",
        "authorized_tasks",
        "decision",
        "scope",
    }:
        raise ValueError("projection production authorization field inventory mismatch")
    authorized_tasks = authorization.get("authorized_tasks")
    if (
        authorization.get("decision") != "approved"
        or authorization.get("scope")
        != "cpu_planner_full_trajectory_generation_and_training_data"
        or authorization.get("approved_planner_master_commit")
        != evaluator_benchmark_commit
        or not isinstance(authorized_tasks, list)
        or len(authorized_tasks) != len(EXACT_TASKS)
        or set(authorized_tasks) != set(EXACT_TASKS)
        or not isinstance(authorization.get("authorized_on"), str)
        or len(authorization["authorized_on"]) != 10
    ):
        raise ValueError("projection production authorization mismatch")
    units = receipt.get("units")
    if not isinstance(units, list) or len(units) != len(EXACT_TASKS):
        raise ValueError("projection approved review unit inventory is not exact14")
    unit_tasks = []
    for unit in units:
        if (
            not isinstance(unit, Mapping)
            or set(unit) != {"task", "review_status", "owner_decision"}
            or unit.get("review_status") != "approved"
            or unit.get("owner_decision") != "approved"
        ):
            raise ValueError("projection authorization unit is not owner-approved")
        unit_tasks.append(unit.get("task"))
    if unit_tasks != list(EXACT_TASKS):
        raise ValueError("projection authorization unit tasks/order mismatch")


def _validate_frozen_binding(binding: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "rld2-qa-planner-calibration-wave-receipt-v0.1",
        "binding_status": "bound",
        "scientific_partition": "metric_calibration",
        "transport_split": "validation",
        "task_count": len(EXACT_TASKS),
        "episodes_per_task": 20,
        "total_reset_count": 20 * len(EXACT_TASKS),
        "task_order": list(EXACT_TASKS),
    }
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError(f"frozen calibration binding {name} mismatch")
    source = binding.get("source_identity")
    if not isinstance(source, Mapping):
        raise ValueError("frozen calibration binding source identity is missing")
    _commit(source.get("benchmark_commit"), "frozen calibration benchmark commit")
    tasks = binding.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(EXACT_TASKS):
        raise ValueError("frozen calibration binding task inventory is not exact14")
    required_hashes = (
        "task_contract_sha256",
        "task_receipt_sha256",
        "task_config_sha256",
        "task_quality_schema_sha256",
        "reset_manifest_sha256",
        "reset_identity_set_sha256",
        "reset_row_set_sha256",
        "evaluation_sha256",
        "evaluation_payload_sha256",
    )
    for ordinal, (task_id, row) in enumerate(zip(EXACT_TASKS, tasks, strict=True)):
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != ordinal
            or row.get("task_id") != task_id
            or row.get("reset_identity_count") != 20
        ):
            raise ValueError("frozen calibration binding task order/count mismatch")
        for name in required_hashes:
            _sha256(row.get(name), f"frozen calibration task {task_id} {name}")


def validate_projection_artifact(
    thresholds: Mapping[str, Any],
    artifact_path: Path,
    *,
    expected_sha256: str | None,
    expected_benchmark_commit: str | None,
) -> dict[str, Any] | None:
    """Validate a projection, returning None when the artifact is canonical receipt data."""

    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValueError("quality-v2 calibration receipt artifact is missing or symlinked")
    artifact_bytes = artifact_path.read_bytes()
    try:
        payload = json.loads(artifact_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("quality-v2 calibration receipt is not UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError("quality-v2 calibration receipt must be a mapping")
    if payload.get("schema_version") != CALIBRATION_BINDING_PROJECTION_SCHEMA:
        return None
    if set(payload) != {
        "schema_version",
        "evaluator_benchmark_commit",
        "calibration_wave_receipt_binding",
        "production_authorization_receipt",
    }:
        raise ValueError("calibration binding projection field inventory mismatch")
    canonical = canonical_json_bytes(payload)
    if artifact_bytes != canonical:
        raise ValueError("calibration binding projection is not canonical JSON")
    actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if expected_sha256 is not None and actual_sha256 != _sha256(
        expected_sha256, "expected calibration binding projection SHA-256"
    ):
        raise ValueError("calibration binding projection file SHA-256 mismatch")
    frozen = thresholds.get("calibration_wave_receipt")
    projected = payload.get("calibration_wave_receipt_binding")
    if not isinstance(frozen, Mapping) or frozen.get("binding_status") != "bound":
        raise ValueError("quality-v2 thresholds have no bound calibration receipt")
    _validate_frozen_binding(frozen)
    if not isinstance(projected, Mapping) or dict(projected) != dict(frozen):
        raise ValueError("calibration binding projection differs from frozen thresholds")
    historical_sha256 = _sha256(
        frozen.get("file_sha256"), "historical canonical receipt file SHA-256"
    )
    if expected_sha256 is not None and expected_sha256 == historical_sha256:
        raise ValueError(
            "projection SHA-256 must be its actual file identity, not the historical receipt SHA-256"
        )
    evaluator_benchmark_commit = _commit(
        payload.get("evaluator_benchmark_commit"),
        "calibration binding projection evaluator benchmark commit",
    )
    if expected_benchmark_commit is not None and evaluator_benchmark_commit != _commit(
        expected_benchmark_commit,
        "expected calibration binding projection evaluator benchmark commit",
    ):
        raise ValueError(
            "calibration binding projection evaluator benchmark commit mismatch"
        )
    _validate_authorization_receipt(
        payload.get("production_authorization_receipt"), evaluator_benchmark_commit
    )
    return {
        "binding": dict(projected),
        "evaluator_benchmark_commit": evaluator_benchmark_commit,
        "file_sha256": actual_sha256,
        "payload_sha256": actual_sha256,
    }
