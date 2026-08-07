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

"""Regression coverage for the optimal-trajectories evaluator-identity audit.

The independent trajectory auditor builds ``expected_evidence`` rows that carry
``policy_benchmark_commit`` in addition to ``relative_path``/``sha256``; the
provenance-file gate requires the exact two-field projection.  This test
guards that caller-side projection so a real checkpoint-compatible dataset can
pass the independent audit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    EVALUATOR_IDENTITY_SCHEMA,
    _audit_evaluator_identity,
    _payload_sha256,
)
from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
    build_compatibility_evidence,
)

POLICY_BENCHMARK = "a" * 40
EVALUATOR_RLinf = "b" * 40
EVALUATOR_BENCHMARK = "c" * 40
POLICY_RLinf = "d" * 40
BACKEND_ID = "se3-wam-quality-test"


def _compatibility_probe(task: str) -> dict[str, Any]:
    return {
        "task": task,
        "policy_sha256": "1" * 64,
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
        "environment_instance_id": "compat-env",
        "episode_id": "compat",
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


def _candidate_payload(task: str = "t1_xyz") -> dict[str, Any]:
    return {
        "schema_version": "rlinf-dynamic-benchmark-optimal-candidates-v0.2",
        "task": task,
        "candidates": [
            {
                "kind": "policy",
                "candidate_id": "policy-1",
                "candidate_index": 1,
                "policy_sha256": "1" * 64,
                "provenance": {
                    "source": {"rlinf_commit": POLICY_RLinf},
                    "runtime": {"evaluator_rlinf_commit": EVALUATOR_RLinf},
                    "benchmark": {"commit": POLICY_BENCHMARK},
                    "state_schema": {
                        "sha256": "2" * 64,
                        "state_dim": 128,
                        "mask_dim": 14,
                    },
                },
            }
        ],
        "policy_benchmark_commits": [POLICY_BENCHMARK],
        "policy_rlinf_commits": [POLICY_RLinf],
        "evaluator_identity": {
            "schema_version": EVALUATOR_IDENTITY_SCHEMA,
            "evaluator_rlinf_commit": EVALUATOR_RLinf,
            "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
            "backend_id": BACKEND_ID,
            "policy_benchmark_relations": [
                {
                    "policy_benchmark_commit": POLICY_BENCHMARK,
                    "relation": "checkpoint-compatible",
                    "evidence_path": "../evidence/benchmark-compatibility/proof.json",
                    "evidence_sha256": None,
                }
            ],
        },
    }


def test_checkpoint_compatible_evaluator_identity_passes_provenance_gate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    evidence_root = root / "provenance" / "evidence"
    evidence_root.mkdir(parents=True)

    candidate = _candidate_payload()
    raw = candidate["evaluator_identity"]
    relations = raw["policy_benchmark_relations"]
    relation = relations[0]
    proof = build_compatibility_evidence(
        {
            "policy_benchmark_commit": POLICY_BENCHMARK,
            "evaluator_rlinf_commit": EVALUATOR_RLinf,
            "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
            "backend_id": BACKEND_ID,
            "split": "validation",
            "test_exposure": {"test_id": False, "test_ood": False},
            "probes": [_compatibility_probe("t1_xyz")],
        }
    )
    canonical_proof = json.dumps(
        proof,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_proof.encode("utf-8")).hexdigest()
    relation["evidence_sha256"] = digest
    proof_path = evidence_root / f"{digest}.json"
    proof_path.write_text(canonical_proof, encoding="utf-8")

    card = {
        "source_identity": {
            "evaluator_rlinf_commit": EVALUATOR_RLinf,
            "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
        },
        "evaluator_identity": raw,
        "compatibility_evidence": [
            {
                "policy_benchmark_commit": POLICY_BENCHMARK,
                "relative_path": f"provenance/evidence/{digest}.json",
                "sha256": digest,
            }
        ],
    }
    planner_dominance = {"backend_id": BACKEND_ID}

    result = _audit_evaluator_identity(
        root,
        card=card,
        candidate_payload=candidate,
        planner_dominance=planner_dominance,
    )
    assert result == raw


def test_legacy_candidate_schema_short_circuits_to_none(tmp_path: Path) -> None:
    candidate = {
        "schema_version": "rlinf-dynamic-benchmark-optimal-candidates-v0.1",
        "task": "t1_xyz",
    }
    card: dict[str, Any] = {}
    result = _audit_evaluator_identity(
        tmp_path,
        card=card,
        candidate_payload=candidate,
        planner_dominance=None,
    )
    assert result is None


def test_extra_provenance_fields_still_fail_closed(tmp_path: Path) -> None:
    candidate = _candidate_payload()
    raw = candidate["evaluator_identity"]
    relation = raw["policy_benchmark_relations"][0]
    relation["evidence_sha256"] = "f" * 64
    card = {
        "source_identity": {
            "evaluator_rlinf_commit": EVALUATOR_RLinf,
            "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
        },
        "evaluator_identity": raw,
        "compatibility_evidence": [
            {
                "policy_benchmark_commit": POLICY_BENCHMARK,
                "relative_path": "provenance/evidence/ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff.json",
                "sha256": "f" * 64,
            }
        ],
    }
    with pytest.raises(ValueError, match="checksum mismatch"):
        _audit_evaluator_identity(
            tmp_path,
            card=card,
            candidate_payload=candidate,
            planner_dominance={"backend_id": BACKEND_ID},
        )


def _quality_summary(component_order: list[str]) -> dict[str, Any]:
    resolutions = {
        "terminal_goal_planar_error_m": 0.0005,
        "maximum_rim_impulse_n_s": 0.001,
    }
    summary = {
        "schema_version": "db0-episode-task-quality-v1",
        "task_id": "t2_trans",
        "evaluator_backend_id": BACKEND_ID,
        "schema_sha256": "0" * 64,
        "episode_id": "ep",
        "physics_sample_count": 500,
        "terminal": True,
        "components": {
            name: {
                "value": 0.0,
                "direction": "minimize",
                "unit": "N*s" if name.endswith("impulse_n_s") else "m",
                "scientific_resolution": resolutions[name],
                "reducer": "maximum" if name.endswith("impulse_n_s") else "terminal",
            }
            for name in component_order
        },
    }
    summary["summary_sha256"] = _payload_sha256(summary)
    return summary


def _quality_contract() -> dict[str, Any]:
    return {
        "task": "t2_trans",
        "backend_id": BACKEND_ID,
        "quality_schema": {
            "schema_version": "db0-episode-task-quality-v1",
            "task_id": "t2_trans",
            "schema_sha256": "0" * 64,
            "components": [
                {
                    "name": "terminal_goal_planar_error_m",
                    "direction": "minimize",
                    "unit": "m",
                    "scientific_resolution": 0.0005,
                    "reducer": "terminal",
                },
                {
                    "name": "maximum_rim_impulse_n_s",
                    "direction": "minimize",
                    "unit": "N*s",
                    "scientific_resolution": 0.001,
                    "reducer": "maximum",
                },
            ],
        },
    }


def test_attempt_quality_accepts_non_canonical_component_order() -> None:
    from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
        _validate_attempt_quality,
    )

    record = {
        "episode_id": "ep",
        "task_quality": _quality_summary(
            ["maximum_rim_impulse_n_s", "terminal_goal_planar_error_m"]
        ),
    }
    _validate_attempt_quality(record, _quality_contract())


def test_attempt_quality_still_rejects_missing_component() -> None:
    from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
        _validate_attempt_quality,
    )

    record = {
        "episode_id": "ep",
        "task_quality": _quality_summary(["terminal_goal_planar_error_m"]),
    }
    with pytest.raises(ValueError, match="mapping gap"):
        _validate_attempt_quality(record, _quality_contract())
