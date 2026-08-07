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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from examples.embodiment.audit_dynamic_benchmark_rld2_release import (
    ACCEPTED_PER_TASK,
    CANDIDATE_RELEASE_SCHEMA,
    CANDIDATE_SCHEMA,
    DATASET_CARD_SCHEMA,
    EXACT_TASKS,
    FULL_POOL_SEARCH_MODE,
    INPUT_INVENTORY_SCHEMA,
    PLANNER_PARETO_SELECTION_CONTRACT,
    PLANNER_PARETO_SELECTION_MODE,
    RELEASE_ID,
    RELEASE_INPUT_SCHEMA,
    TASK_AUDIT_SCHEMA,
    ReleaseAuditError,
    _collect_release,
    _normalize_contract,
    _payload_sha256,
    _sha256,
    build_and_audit_release,
)
from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
    build_compatibility_evidence,
)

EVALUATOR_RLinf = "a" * 40
POLICY_RLinf = "b" * 40
EVALUATOR_BENCHMARK = "c" * 40
POLICY_BENCHMARK = "d" * 40
AUDITOR_COMMIT = "e" * 40
BACKEND_ID = "se3-wam-quality-test"


@dataclass
class _Fixture:
    root: Path
    candidate_root: Path
    input_manifest: Path
    records: dict[str, dict[str, Any]]
    candidate_release_sha256: str
    candidate_checksums_sha256: str


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _seal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def _seal_root(root: Path) -> str:
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    content = "".join(
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
    )
    (root / "SHA256SUMS").write_text(content, encoding="utf-8")
    return _sha256(root / "SHA256SUMS")


def _quality_schema(task: str) -> dict[str, Any]:
    schema = {
        "schema_version": "db0-episode-task-quality-v1",
        "task_id": task,
        "task_config_sha256": "1" * 64,
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


def _metric(
    direction: str, resolution: float, *, steps: bool = False
) -> dict[str, Any]:
    return {
        "direction": direction,
        "max_observed_replay_drift": 0.0,
        "scientific_resolution": resolution,
        "numeric_floor": 0.0 if steps else 1.0e-6,
    }


def _contract(task: str, evidence_path: str, evidence_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "rlinf-dynamic-benchmark-planner-dominance-v0.1",
        "task": task,
        "backend_id": BACKEND_ID,
        "quality_schema": _quality_schema(task),
        "calibration": {
            "replay_count": 3,
            "reset_episode_id": f"{task}-calibration",
            "reset_manifest_sha256": "2" * 64,
            "evidence_path": evidence_path,
            "evidence_sha256": evidence_sha256,
        },
        "metrics": {
            "trajectory_completion": _metric("max", 1.0e-6),
            "task_quality": {"lift_clearance_m": _metric("max", 0.002)},
            "completion_time_s": _metric("min", 0.002),
            "control_steps": _metric("min", 1.0, steps=True),
            "action_l2_sum": {
                "direction": "min",
                "max_observed_replay_drift": 0.0,
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


def _quality_summary(task: str, episode_id: str, value: float) -> dict[str, Any]:
    schema = _quality_schema(task)
    summary = {
        "schema_version": schema["schema_version"],
        "episode_id": episode_id,
        "task_id": task,
        "evaluator_backend_id": BACKEND_ID,
        "schema_sha256": schema["schema_sha256"],
        "physics_sample_count": 10,
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


def _evaluator_identity(path: str) -> dict[str, Any]:
    return {
        "schema_version": "rlinf-dynamic-benchmark-quality-evaluator-identity-v0.1",
        "evaluator_rlinf_commit": EVALUATOR_RLinf,
        "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
        "backend_id": BACKEND_ID,
        "policy_benchmark_relations": [
            {
                "policy_benchmark_commit": POLICY_BENCHMARK,
                "relation": "checkpoint-compatible",
                "evidence_path": path,
                "evidence_sha256": None,
            }
        ],
    }


def _provenance(*, policy: bool) -> dict[str, Any]:
    commit = POLICY_RLinf if policy else EVALUATOR_RLinf
    benchmark = POLICY_BENCHMARK if policy else EVALUATOR_BENCHMARK
    return {
        "source": {"rlinf_commit": commit},
        "runtime": {"evaluator_rlinf_commit": EVALUATOR_RLinf},
        "benchmark": {"commit": benchmark},
        "state_schema": {
            "sha256": "3" * 64,
            "state_dim": 128,
            "mask_dim": 14,
        },
    }


def _compatibility_probe(task: str) -> dict[str, Any]:
    return {
        "task": task,
        "policy_sha256": "4" * 64,
        "policy_rlinf_commit": POLICY_RLinf,
        "policy_state_schema_sha256": "3" * 64,
        "policy_state_dim": 128,
        "policy_mask_dim": 14,
        "evaluator_state_schema_sha256": "3" * 64,
        "evaluator_state_dim": 128,
        "evaluator_mask_dim": 14,
        "policy_action_dim": 7,
        "evaluator_action_dim": 7,
        "evaluator_task_config_sha256": "1" * 64,
        "environment_instance_id": f"compat-env-{task}",
        "episode_id": f"compat-{task}",
        "reset_request_sha256": "5" * 64,
        "observation_sha256": "6" * 64,
        "action_sha256": "7" * 64,
        "load_success": True,
        "reset_success": True,
        "inference_success": True,
        "step_success": True,
        "finite_observation": True,
        "finite_action": True,
        "finite_reward": True,
    }


def _calibration_replay(task: str, index: int) -> dict[str, Any]:
    episode_id = f"{task}-calibration"
    return {
        "replay_index": index,
        "environment_instance_id": f"fresh-{task}-{index}",
        "episode_id": episode_id,
        "reset_request_sha256": "8" * 64,
        "action_sha256": "9" * 64,
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "termination_reason": "success",
        "trajectory_completion": 1.0,
        "completion_time_s": 1.0,
        "control_steps": 500,
        "action_l2_sum": 2.0,
        "task_quality": _quality_summary(task, episode_id, 1.0),
    }


def _candidate_manifest(
    task: str,
    *,
    compatibility_path: str,
    compatibility_sha256: str,
    calibration_path: str,
    calibration_sha256: str,
) -> dict[str, Any]:
    identity = _evaluator_identity(compatibility_path)
    identity["policy_benchmark_relations"][0]["evidence_sha256"] = compatibility_sha256
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "task": task,
        "evaluator_identity": identity,
        "policy_rlinf_commits": [POLICY_RLinf],
        "policy_benchmark_commits": [POLICY_BENCHMARK],
        "planner_dominance": _contract(task, calibration_path, calibration_sha256),
        "candidates": [
            {
                "candidate_id": "planner",
                "kind": "planner",
                "provenance": _provenance(policy=False),
            },
            {
                "candidate_id": "policy-1",
                "kind": "policy",
                "policy_path": f"/policies/{task}.pt",
                "policy_sha256": "4" * 64,
                "stochastic": False,
                "exploration_seed_offset": 0,
                "provenance": _provenance(policy=True),
            },
        ],
    }


def _attempt(task: str, episode_id: str, index: int) -> dict[str, Any]:
    return {
        "task_id": task,
        "episode_id": episode_id,
        "candidate_index": index,
        "candidate_id": "planner" if index == 0 else "policy-1",
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "replay_validation": {"passed": True, "outcomes_exact": True},
        "trajectory_completion": 1.0,
        "task_quality": _quality_summary(task, episode_id, 1.0 if index == 0 else 1.01),
        "completion_time_s": 1.0,
        "return": 1.0,
        "control_steps": 500,
        "action_l2_sum": 2.0,
        "eligible": True,
    }


def _inventory_row(task: str, index: int, candidate: dict[str, Any]) -> dict[str, Any]:
    semantics = (
        {"kind": "planner"}
        if index == 0
        else {
            "kind": "policy",
            "policy_sha256": candidate["policy_sha256"],
            "stochastic": False,
            "exploration_seed_offset": 0,
            "residual_scale": None,
        }
    )
    return {
        "schema_version": INPUT_INVENTORY_SCHEMA,
        "task": task,
        "candidate_index": index,
        "candidate_id": candidate["candidate_id"],
        "kind": candidate["kind"],
        "source_kind": "synthetic-test",
        "semantics": semantics,
        "semantics_sha256": _payload_sha256(semantics),
        "provenance": candidate["provenance"],
    }


def _source_identity() -> dict[str, Any]:
    return {
        "evaluator_rlinf_commit": EVALUATOR_RLinf,
        "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
        "policy_rlinf_commits": [POLICY_RLinf],
        "policy_benchmark_commits": [POLICY_BENCHMARK],
    }


def _write_input_manifest(fixture: _Fixture) -> None:
    payload = _seal_payload(
        {
            "schema_version": RELEASE_INPUT_SCHEMA,
            "release_id": RELEASE_ID,
            "candidate_release_root": str(fixture.candidate_root.resolve()),
            "candidate_release_manifest_sha256": fixture.candidate_release_sha256,
            "candidate_release_checksums_sha256": fixture.candidate_checksums_sha256,
            "tasks": [
                fixture.records[task] for task in EXACT_TASKS if task in fixture.records
            ],
        }
    )
    _write_json(fixture.input_manifest, payload)


def _make_fixture(tmp_path: Path) -> _Fixture:
    root = tmp_path / "RLD2"
    candidate_root = root / "candidate-release"
    evidence_root = candidate_root / "evidence"
    compatibility = build_compatibility_evidence(
        {
            "policy_benchmark_commit": POLICY_BENCHMARK,
            "evaluator_rlinf_commit": EVALUATOR_RLinf,
            "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
            "backend_id": BACKEND_ID,
            "split": "validation",
            "test_exposure": {"test_id": False, "test_ood": False},
            "probes": [_compatibility_probe(task) for task in sorted(EXACT_TASKS)],
        }
    )
    compatibility_path = evidence_root / "benchmark-compatibility" / "proof.json"
    _write_json(compatibility_path, compatibility)
    compatibility_sha256 = _sha256(compatibility_path)

    manifests: dict[str, dict[str, Any]] = {}
    inventory_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for task in EXACT_TASKS:
        task_identity = _evaluator_identity(
            "../evidence/benchmark-compatibility/proof.json"
        )
        task_identity["policy_benchmark_relations"][0]["evidence_sha256"] = (
            compatibility_sha256
        )
        calibration = {
            "schema_version": "rlinf-dynamic-benchmark-planner-calibration-evidence-v0.1",
            "task": task,
            "backend_id": BACKEND_ID,
            "evaluator_identity_sha256": _payload_sha256(
                {
                    "evaluator_rlinf_commit": EVALUATOR_RLinf,
                    "evaluator_benchmark_commit": EVALUATOR_BENCHMARK,
                    "backend_id": BACKEND_ID,
                }
            ),
            "split": "validation",
            "test_exposure": {"test_id": False, "test_ood": False},
            "reset_manifest_sha256": "2" * 64,
            "replay_count": 3,
            "replays": [_calibration_replay(task, index) for index in range(3)],
        }
        _seal_payload(calibration)
        calibration_path = evidence_root / "calibration" / f"{task}.json"
        _write_json(calibration_path, calibration)
        calibration_sha256 = _sha256(calibration_path)
        manifest = _candidate_manifest(
            task,
            compatibility_path="../evidence/benchmark-compatibility/proof.json",
            compatibility_sha256=compatibility_sha256,
            calibration_path=f"../evidence/calibration/{task}.json",
            calibration_sha256=calibration_sha256,
        )
        manifests[task] = manifest
        path = candidate_root / task / "candidate_manifest.json"
        _write_json(path, manifest)
        inventory_rows.extend(
            _inventory_row(task, index, candidate)
            for index, candidate in enumerate(manifest["candidates"])
        )
        calibration_rows.append(
            {
                "task": task,
                "path": f"evidence/calibration/{task}.json",
                "sha256": calibration_sha256,
            }
        )
    inventory_path = candidate_root / "input_inventory.jsonl"
    _write_jsonl(inventory_path, inventory_rows)
    source = root / "source-input.json"
    _write_json(source, {"source": "synthetic-test"})
    inputs_path = candidate_root / "INPUTS.sha256"
    inputs_path.write_text(
        f"{_sha256(source)}  source\tsynthetic\t{source.resolve()}\n",
        encoding="utf-8",
    )
    release_identity = _evaluator_identity(
        "evidence/benchmark-compatibility/proof.json"
    )
    release_identity["policy_benchmark_relations"][0]["evidence_sha256"] = (
        compatibility_sha256
    )
    release_manifest = _seal_payload(
        {
            "schema_version": CANDIDATE_RELEASE_SCHEMA,
            "release_id": RELEASE_ID,
            "candidate_schema_version": CANDIDATE_SCHEMA,
            "evaluator_identity": release_identity,
            "policy_rlinf_commits": [POLICY_RLinf],
            "policy_benchmark_commits": [POLICY_BENCHMARK],
            "evaluator_evidence": [
                {
                    "path": "evidence/benchmark-compatibility/proof.json",
                    "sha256": compatibility_sha256,
                }
            ],
            "calibration_evidence": list(calibration_rows),
            "tasks": list(EXACT_TASKS),
            "task_manifest_sha256": {
                task: _sha256(candidate_root / task / "candidate_manifest.json")
                for task in EXACT_TASKS
            },
            "candidate_count": dict.fromkeys(EXACT_TASKS, 2),
            "deduplicated": [],
            "input_spec_sha256": _sha256(source),
            "input_inventory_sha256": _sha256(inventory_path),
            "inputs_sha256_sha256": _sha256(inputs_path),
            "production_validated": True,
        }
    )
    release_path = candidate_root / "release_manifest.json"
    _write_json(release_path, release_manifest)
    candidate_release_sha256 = _sha256(release_path)
    candidate_checksums_sha256 = _seal_root(candidate_root)

    records: dict[str, dict[str, Any]] = {}
    for task in EXACT_TASKS:
        dataset_root = root / "datasets" / task
        dataset_root.mkdir(parents=True)
        candidate_path = dataset_root / "candidate_manifest.json"
        _write_json(candidate_path, manifests[task])
        candidate_sha256 = _sha256(candidate_path)
        contract = _normalize_contract(manifests[task], task)
        card = _seal_payload(
            {
                "schema_version": DATASET_CARD_SCHEMA,
                "task": task,
                "status": "complete",
                "training_eligible": False,
                "accepted_target": ACCEPTED_PER_TASK,
                "accepted_count": ACCEPTED_PER_TASK,
                "attempted_reset_count": ACCEPTED_PER_TASK,
                "candidate_search_mode": FULL_POOL_SEARCH_MODE,
                "candidate_pool_size": 2,
                "selection_mode": PLANNER_PARETO_SELECTION_MODE,
                "selection_contract": PLANNER_PARETO_SELECTION_CONTRACT,
                "budget_sequence": [2],
                "candidate_manifest_sha256": candidate_sha256,
                "candidate_release_manifest_sha256": candidate_release_sha256,
                "planner_dominance": contract,
                "evaluator_identity": manifests[task]["evaluator_identity"],
                "source_identity": _source_identity(),
                "state_schema": {"schema_version": "rlinf-dynamic-state-v1"},
            }
        )
        _write_json(dataset_root / "dataset_card.json", card)
        attempts: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for index in range(ACCEPTED_PER_TASK):
            episode_id = f"{task}-{index:03d}"
            attempts.extend(
                (_attempt(task, episode_id, 0), _attempt(task, episode_id, 1))
            )
            results.append(
                {
                    "reset_index": index,
                    "episode_id": episode_id,
                    "candidate_count": 2,
                    "budget_used": 2,
                    "candidate_search_mode": FULL_POOL_SEARCH_MODE,
                    "selection_mode": PLANNER_PARETO_SELECTION_MODE,
                    "selection_result": {
                        "source_kind": "expert_dominant",
                        "planner_eligible": True,
                        "winner_candidate_id": "policy-1",
                        "winner_candidate_index": 1,
                    },
                    "accepted": True,
                    "winner_candidate_id": "policy-1",
                    "winner_candidate_index": 1,
                }
            )
        _write_jsonl(dataset_root / "attempts.jsonl", attempts)
        _write_jsonl(dataset_root / "reset_results.jsonl", results)
        checksums_sha256 = _seal_root(dataset_root)
        card_sha256 = _sha256(dataset_root / "dataset_card.json")
        audit_path = root / "audits" / f"{task}.json"
        audit = _seal_payload(
            {
                "schema_version": TASK_AUDIT_SCHEMA,
                "dataset_root": str(dataset_root.resolve()),
                "dataset_card_sha256": card_sha256,
                "checksums_sha256": checksums_sha256,
                "candidate_manifest_sha256": candidate_sha256,
                "candidate_release_manifest_sha256": candidate_release_sha256,
                "auditor_commit": AUDITOR_COMMIT,
                "status": "passed",
                "training_eligible": True,
                "summary": {
                    "task": task,
                    "accepted_count": ACCEPTED_PER_TASK,
                    "candidate_pool_size": 2,
                    "candidate_search_mode": FULL_POOL_SEARCH_MODE,
                    "selection_mode": PLANNER_PARETO_SELECTION_MODE,
                    "source_identity": _source_identity(),
                    "dataset_card_payload_sha256": card["payload_sha256"],
                    "planner_dominance": contract,
                    "evaluator_identity": manifests[task]["evaluator_identity"],
                },
            }
        )
        _write_json(audit_path, audit)
        records[task] = {
            "task": task,
            "dataset_root": str(dataset_root.resolve()),
            "dataset_card_sha256": card_sha256,
            "checksums_sha256": checksums_sha256,
            "candidate_manifest_sha256": candidate_sha256,
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": _sha256(audit_path),
            "input_inventory_path": str(inventory_path.resolve()),
            "input_inventory_sha256": _sha256(inventory_path),
        }
    fixture = _Fixture(
        root=root,
        candidate_root=candidate_root,
        input_manifest=root / "release_inputs.json",
        records=records,
        candidate_release_sha256=candidate_release_sha256,
        candidate_checksums_sha256=candidate_checksums_sha256,
    )
    _write_input_manifest(fixture)
    return fixture


def _refresh_dataset(fixture: _Fixture, task: str) -> None:
    record = fixture.records[task]
    dataset_root = Path(record["dataset_root"])
    record["checksums_sha256"] = _seal_root(dataset_root)
    audit_path = Path(record["audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["checksums_sha256"] = record["checksums_sha256"]
    _seal_payload(audit)
    _write_json(audit_path, audit)
    record["audit_sha256"] = _sha256(audit_path)
    _write_input_manifest(fixture)


def test_exact14_release_builds_then_independent_audit_grants_eligibility(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    output = fixture.root / "release" / "unified14"

    report = build_and_audit_release(
        input_manifest=fixture.input_manifest,
        output_root=output,
        auditor_commit=AUDITOR_COMMIT,
    )

    assert report["status"] == "passed"
    assert report["release_eligible"] is True
    assert report["summary"]["accepted_count"] == 1400
    manifest = json.loads(
        (output / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tasks"] == list(EXACT_TASKS)
    assert manifest["aggregate"]["source_counts"] == {"expert_dominant": 1400}


def test_release_input_rejects_missing_task(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.records.pop("t1_xyz")
    _write_input_manifest(fixture)

    with pytest.raises(ReleaseAuditError, match="ordered exact14"):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)


def test_release_rejects_string_accepted_even_when_dataset_is_resealed(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    task = "t1_xyz"
    path = Path(fixture.records[task]["dataset_root"]) / "reset_results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["accepted"] = "false"
    _write_jsonl(path, rows)
    _refresh_dataset(fixture, task)

    with pytest.raises(ReleaseAuditError, match="accepted must be a native boolean"):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)


def test_release_rejects_non_dominant_expert_even_when_dataset_is_resealed(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    task = "t1_xyz"
    path = Path(fixture.records[task]["dataset_root"]) / "attempts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["task_quality"] = copy.deepcopy(rows[0]["task_quality"])
    rows[1]["task_quality"]["episode_id"] = rows[1]["episode_id"]
    summary = rows[1]["task_quality"]
    summary.pop("summary_sha256")
    summary["summary_sha256"] = _payload_sha256(summary)
    _write_jsonl(path, rows)
    _refresh_dataset(fixture, task)

    with pytest.raises(ReleaseAuditError, match="does not dominate planner"):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)


def test_release_rejects_candidate_release_hash_tamper(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    path = fixture.candidate_root / "release_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["release_id"] = "tampered"
    _write_json(path, payload)

    with pytest.raises(
        ReleaseAuditError, match="file hash tamper|release-manifest hash mismatch"
    ):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)
