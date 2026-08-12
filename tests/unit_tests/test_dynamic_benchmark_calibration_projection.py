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

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from examples.embodiment.dynamic_benchmark_calibration_projection import (
    CALIBRATION_BINDING_PROJECTION_SCHEMA,
    EXACT_TASKS,
    REVIEW_SCHEMA,
    canonical_json_bytes,
    projection_payload,
    validate_projection_artifact,
)


def _review(commit: str = "c" * 40) -> dict:
    review = {
        "schema_version": REVIEW_SCHEMA,
        "counts": {
            "approved": 14,
            "pending_owner_review": 0,
            "rejected": 0,
            "revise": 0,
            "tasks": 14,
        },
        "owner_review_complete": True,
        "training_data_authorized": True,
        "full_trajectory_generation": True,
        "promotion_authorized": True,
        "source_identity": {
            "production_authorization": {
                "approved_planner_master_commit": commit,
                "authorized_on": "2026-08-12",
                "authorized_tasks": list(EXACT_TASKS),
                "decision": "approved",
                "scope": "cpu_planner_full_trajectory_generation_and_training_data",
            }
        },
        "units": [
            {
                "task": task,
                "review_status": "approved",
                "owner_review": {"decision": "approved"},
            }
            for task in EXACT_TASKS
        ],
    }
    review["payload_sha256"] = hashlib.sha256(canonical_json_bytes(review)).hexdigest()
    return review


def _fixture(tmp_path: Path) -> tuple[dict, Path, str]:
    tasks = []
    for ordinal, task in enumerate(EXACT_TASKS):
        row = {
            "ordinal": ordinal,
            "task_id": task,
            "reset_identity_count": 20,
        }
        for offset, name in enumerate(
            (
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
        ):
            row[name] = f"{ordinal * 20 + offset + 1:064x}"
        tasks.append(row)
    thresholds = {"calibration_wave_receipt": {
        "schema_version": "rld2-qa-planner-calibration-wave-receipt-v0.1",
        "binding_status": "bound",
        "scientific_partition": "metric_calibration",
        "transport_split": "validation",
        "file_sha256": "a" * 64,
        "source_identity": {"benchmark_commit": "b" * 40},
        "task_count": 14,
        "episodes_per_task": 20,
        "total_reset_count": 280,
        "task_order": list(EXACT_TASKS),
        "tasks": tasks,
    }}
    path = tmp_path / "projection.json"
    path.write_bytes(
        canonical_json_bytes(
            projection_payload(
                thresholds["calibration_wave_receipt"], "c" * 40, _review(), "d" * 64
            )
        )
    )
    return thresholds, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_projection_uses_actual_file_identity_and_exact_frozen_binding(
    tmp_path: Path,
) -> None:
    thresholds, path, actual = _fixture(tmp_path)
    result = validate_projection_artifact(
        thresholds,
        path,
        expected_sha256=actual,
        expected_benchmark_commit="c" * 40,
    )
    assert result is not None
    assert result["file_sha256"] == actual
    assert result["file_sha256"] != thresholds["calibration_wave_receipt"]["file_sha256"]


@pytest.mark.parametrize("mutation", ["field", "delete"])
def test_projection_rejects_changed_or_deleted_binding_field(
    tmp_path: Path, mutation: str
) -> None:
    thresholds, path, _ = _fixture(tmp_path)
    payload = projection_payload(
        deepcopy(thresholds["calibration_wave_receipt"]),
        "c" * 40,
        _review(),
        "d" * 64,
    )
    binding = payload["calibration_wave_receipt_binding"]
    if mutation == "field":
        binding["task_count"] = 13
    else:
        binding.pop("task_count")
    path.write_bytes(canonical_json_bytes(payload))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="differs from frozen thresholds"):
        validate_projection_artifact(
            thresholds,
            path,
            expected_sha256=actual,
            expected_benchmark_commit="c" * 40,
        )


def test_projection_rejects_historical_receipt_sha_as_its_file_identity(
    tmp_path: Path,
) -> None:
    thresholds, path, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="projection file SHA-256 mismatch"):
        validate_projection_artifact(
            thresholds,
            path,
            expected_sha256=thresholds["calibration_wave_receipt"]["file_sha256"],
            expected_benchmark_commit="c" * 40,
        )


def test_projection_schema_inventory_is_exact(tmp_path: Path) -> None:
    thresholds, path, _ = _fixture(tmp_path)
    payload = projection_payload(
        thresholds["calibration_wave_receipt"], "c" * 40, _review(), "d" * 64
    )
    payload["unexpected"] = CALIBRATION_BINDING_PROJECTION_SCHEMA
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="field inventory"):
        validate_projection_artifact(
            thresholds,
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_benchmark_commit="c" * 40,
        )


def test_projection_rejects_benchmark_or_authorization_tamper(tmp_path: Path) -> None:
    thresholds, path, actual = _fixture(tmp_path)
    with pytest.raises(ValueError, match="evaluator benchmark commit mismatch"):
        validate_projection_artifact(
            thresholds,
            path,
            expected_sha256=actual,
            expected_benchmark_commit="d" * 40,
        )
    payload = projection_payload(
        thresholds["calibration_wave_receipt"], "c" * 40, _review(), "d" * 64
    )
    payload["production_authorization_receipt"]["training_data_authorized"] = False
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="has no training_data_authorized"):
        validate_projection_artifact(
            thresholds,
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_benchmark_commit="c" * 40,
        )


def test_projection_rejects_authorization_hash_unit_or_field_tamper(tmp_path: Path) -> None:
    thresholds, path, _ = _fixture(tmp_path)
    base = projection_payload(
        thresholds["calibration_wave_receipt"], "c" * 40, _review(), "d" * 64
    )
    mutations = []
    bad_hash = deepcopy(base)
    bad_hash["production_authorization_receipt"]["review_file_sha256"] = "bad"
    mutations.append((bad_hash, "review file SHA-256"))
    bad_unit = deepcopy(base)
    bad_unit["production_authorization_receipt"]["units"][0]["owner_decision"] = "pending"
    mutations.append((bad_unit, "unit is not owner-approved"))
    missing = deepcopy(base)
    missing["production_authorization_receipt"].pop("review_payload_sha256")
    mutations.append((missing, "field inventory"))
    for payload, message in mutations:
        path.write_bytes(canonical_json_bytes(payload))
        with pytest.raises(ValueError, match=message):
            validate_projection_artifact(
                thresholds,
                path,
                expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_benchmark_commit="c" * 40,
            )
