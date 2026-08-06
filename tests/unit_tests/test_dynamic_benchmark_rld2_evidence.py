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

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
    EvidenceError,
    _payload_sha256,
    build_calibration_evidence,
    build_compatibility_evidence,
    validate_calibration_evidence,
    validate_compatibility_evidence,
)

POLICY_BENCHMARK = "a" * 40
EVALUATOR_RLinf = "b" * 40
EVALUATOR_BENCHMARK = "c" * 40
POLICY_RLinf = "d" * 40
BACKEND_ID = "se3-wam-quality-test"


def _probe(task: str = "t1_xyz", policy_sha256: str = "1" * 64) -> dict[str, Any]:
    return {
        "task": task,
        "policy_sha256": policy_sha256,
        "policy_rlinf_commit": POLICY_RLinf,
        "policy_state_schema_sha256": "2" * 64,
        "policy_state_dim": 128,
        "policy_mask_dim": 14,
        "evaluator_state_schema_sha256": "2" * 64,
        "evaluator_state_dim": 128,
        "evaluator_mask_dim": 14,
        "policy_action_dim": 7,
        "evaluator_action_dim": 7,
        "evaluator_task_config_sha256": "3" * 64,
        "environment_instance_id": f"env-{task}",
        "episode_id": f"{task}-compatibility",
        "reset_request_sha256": "4" * 64,
        "observation_sha256": "5" * 64,
        "action_sha256": "6" * 64,
        "load_success": True,
        "reset_success": True,
        "inference_success": True,
        "step_success": True,
        "finite_observation": True,
        "finite_action": True,
        "finite_reward": True,
    }


def _compatibility_input() -> dict[str, Any]:
    return {
        "policy_benchmark_commit": POLICY_BENCHMARK,
        "evaluator_rlinf_commit": EVALUATOR_RLinf,
        "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
        "backend_id": BACKEND_ID,
        "split": "validation",
        "test_exposure": {"test_id": False, "test_ood": False},
        "probes": [_probe()],
    }


def _quality_schema() -> dict[str, Any]:
    schema = {
        "schema_version": "db0-episode-task-quality-v1",
        "task_id": "t1_xyz",
        "task_config_sha256": "7" * 64,
        "components": [
            {
                "name": "lift_clearance_m",
                "direction": "maximize",
                "unit": "m",
                "scientific_resolution": 0.002,
                "reducer": "maximum",
                "source": "post-bilateral-contact COM clearance",
                "description": "Maximum post-capture object lift clearance.",
            }
        ],
    }
    schema["schema_sha256"] = _payload_sha256(schema)
    return schema


def _metric(direction: str, resolution: float, *, steps: bool = False) -> dict[str, Any]:
    return {
        "direction": direction,
        "max_observed_replay_drift": 999.0,
        "scientific_resolution": resolution,
        "numeric_floor": 0.0 if steps else 1.0e-6,
    }


def _contract_template() -> dict[str, Any]:
    return {
        "schema_version": "rlinf-dynamic-benchmark-planner-dominance-v0.1",
        "task": "t1_xyz",
        "backend_id": BACKEND_ID,
        "quality_schema": _quality_schema(),
        "calibration": {
            "replay_count": 0,
            "reset_episode_id": "placeholder",
            "reset_manifest_sha256": "8" * 64,
            "evidence_path": "placeholder.json",
            "evidence_sha256": "9" * 64,
        },
        "metrics": {
            "trajectory_completion": _metric("max", 1.0e-6),
            "task_quality": {"lift_clearance_m": _metric("max", 0.002)},
            "completion_time_s": _metric("min", 0.002),
            "control_steps": _metric("min", 1.0, steps=True),
            "action_l2_sum": {
                "direction": "min",
                "max_observed_replay_drift": 999.0,
                "scientific_resolution": 1.0e-6,
                "numeric_floor_absolute": 1.0e-6,
                "numeric_floor_relative": 1.0e-6,
            },
        },
        "tie_break_order": [
            "trajectory_completion",
            "task_quality.lift_clearance_m",
            "completion_time_s",
            "control_steps",
            "action_l2_sum",
        ],
    }


def _evaluator_identity() -> dict[str, Any]:
    return {
        "schema_version": "rlinf-dynamic-benchmark-quality-evaluator-identity-v0.1",
        "evaluator_rlinf_commit": EVALUATOR_RLinf,
        "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
        "backend_id": BACKEND_ID,
        "policy_benchmark_relations": [],
    }


def _quality_summary(value: float) -> dict[str, Any]:
    schema = _quality_schema()
    summary = {
        "schema_version": schema["schema_version"],
        "episode_id": "t1_xyz-calibration",
        "task_id": "t1_xyz",
        "evaluator_backend_id": BACKEND_ID,
        "schema_sha256": schema["schema_sha256"],
        "physics_sample_count": 20,
        "terminal": True,
        "components": {
            "lift_clearance_m": {
                "value": value,
                "direction": "maximize",
                "unit": "m",
                "scientific_resolution": 0.002,
                "reducer": "maximum",
            }
        },
    }
    summary["summary_sha256"] = _payload_sha256(summary)
    return summary


def _replay(index: int) -> dict[str, Any]:
    return {
        "replay_index": index,
        "environment_instance_id": f"fresh-env-{index}",
        "episode_id": "t1_xyz-calibration",
        "reset_request_sha256": "a" * 64,
        "action_sha256": "b" * 64,
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "termination_reason": "success",
        "trajectory_completion": 1.0,
        "completion_time_s": 1.0 + 0.001 * index,
        "control_steps": 500,
        "action_l2_sum": 2.0,
        "task_quality": _quality_summary(0.1 + 0.001 * index),
    }


def _calibration_input() -> dict[str, Any]:
    return {
        "task": "t1_xyz",
        "backend_id": BACKEND_ID,
        "evaluator_identity": _evaluator_identity(),
        "split": "validation",
        "test_exposure": {"test_id": False, "test_ood": False},
        "reset_manifest_sha256": "c" * 64,
        "replays": [_replay(index) for index in range(3)],
    }


def test_compatibility_evidence_is_self_bound_and_exactly_covered() -> None:
    evidence = build_compatibility_evidence(_compatibility_input())
    expected_inventory = [
        {
            "task": "t1_xyz",
            "policy_sha256": "1" * 64,
            "policy_rlinf_commit": POLICY_RLinf,
            "policy_benchmark_commit": POLICY_BENCHMARK,
            "policy_state_schema_sha256": "2" * 64,
            "policy_state_dim": 128,
            "policy_mask_dim": 14,
        }
    ]

    assert validate_compatibility_evidence(
        evidence, expected_inventory=expected_inventory
    ) == evidence

    tampered = copy.deepcopy(evidence)
    tampered["probes"][0]["load_success"] = "true"
    with pytest.raises(EvidenceError, match="failed load_success"):
        validate_compatibility_evidence(tampered)

    with pytest.raises(EvidenceError, match="exactly cover"):
        validate_compatibility_evidence(evidence, expected_inventory=[])


def test_compatibility_rejects_noncanonical_types_and_order() -> None:
    raw = _compatibility_input()
    raw["probes"][0]["policy_state_dim"] = "128"
    with pytest.raises(EvidenceError, match="policy_state_dim"):
        build_compatibility_evidence(raw)

    raw = _compatibility_input()
    raw["probes"] = [_probe("t2_se3", "2" * 64), _probe("t1_xyz", "1" * 64)]
    with pytest.raises(EvidenceError, match="sorted and unique"):
        build_compatibility_evidence(raw)


def test_calibration_builds_fresh_replay_drifts_and_validates() -> None:
    raw = _calibration_input()
    evidence, contract = build_calibration_evidence(
        raw,
        contract_template=_contract_template(),
        evidence_reference="calibration/t1_xyz.json",
    )

    drifts = validate_calibration_evidence(
        evidence,
        contract=contract,
        evaluator_identity=raw["evaluator_identity"],
    )

    assert drifts["task_quality.lift_clearance_m"] == pytest.approx(0.002)
    assert drifts["completion_time_s"] == pytest.approx(0.002)
    assert drifts["control_steps"] == 0.0
    assert contract["calibration"]["replay_count"] == 3
    assert len(contract["calibration"]["evidence_sha256"]) == 64


def test_calibration_rejects_reused_environment_and_test_exposure() -> None:
    raw = _calibration_input()
    raw["replays"][1]["environment_instance_id"] = "fresh-env-0"
    with pytest.raises(EvidenceError, match="unique fresh"):
        build_calibration_evidence(
            raw,
            contract_template=_contract_template(),
            evidence_reference="calibration/t1_xyz.json",
        )

    raw = _calibration_input()
    raw["test_exposure"]["test_id"] = True
    with pytest.raises(EvidenceError, match="formal test"):
        build_calibration_evidence(
            raw,
            contract_template=_contract_template(),
            evidence_reference="calibration/t1_xyz.json",
        )


def test_direct_cli_builds_and_validates_compatibility(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "embodiment"
        / "build_dynamic_benchmark_rld2_evidence.py"
    )
    input_path = tmp_path / "compatibility-input.json"
    evidence_path = tmp_path / "compatibility-evidence.json"
    input_path.write_text(json.dumps(_compatibility_input()), encoding="utf-8")

    built = subprocess.run(
        [
            sys.executable,
            str(script),
            "compatibility",
            "--input",
            str(input_path),
            "--output",
            str(evidence_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    validated = subprocess.run(
        [
            sys.executable,
            str(script),
            "validate-compatibility",
            "--evidence",
            str(evidence_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["status"] == "validated"
