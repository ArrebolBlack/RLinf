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
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from examples.embodiment import audit_dynamic_benchmark_rld2_release as release_auditor
from examples.embodiment.audit_dynamic_benchmark_rld2_release import (
    ACCEPTED_PER_TASK,
    ATTEMPT_SCHEMA,
    CANDIDATE_RELEASE_SCHEMA,
    CANDIDATE_SCHEMA,
    DATASET_CARD_SCHEMA,
    EXACT_TASKS,
    FULL_POOL_SEARCH_MODE,
    HISTORICAL_RELEASE_AUDIT_SCHEMAS,
    HISTORICAL_RELEASE_MANIFEST_SCHEMAS,
    INPUT_INVENTORY_SCHEMA,
    LEGACY_RELEASE_INPUT_SCHEMA,
    PLANNER_PARETO_SELECTION_CONTRACT,
    PLANNER_PARETO_SELECTION_MODE,
    QUALITY_V2_GATE_SCHEMA,
    QUALITY_V2_SUMMARY_SCHEMA,
    QUALITY_V2_THRESHOLDS_SCHEMA,
    RELEASE_AUDIT_SCHEMA,
    RELEASE_ID,
    RELEASE_INPUT_SCHEMA,
    RELEASE_MANIFEST_SCHEMA,
    TASK_AUDIT_SCHEMA,
    ReleaseAuditError,
    _audit_attempt_tape_binding,
    _collect_release,
    _load_release_inputs,
    _normalize_contract,
    _payload_sha256,
    _planner_pareto_dominates,
    _quality_v2_dominance_contract,
    _selected,
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

_QUALITY_V2_FULL_EPISODE_METRICS = (
    "action.action_second_difference_l2_mean_per_transition",
    "action.action_max_second_difference_l2",
    "action.action_total_variation_l2_mean_per_transition",
    "eef_motion.eef_translation_path_length_m",
    "eef_motion.eef_rotation_path_length_rad",
    "eef_motion.eef_angular_jerk_max_rad_s3",
    "eef_motion.eef_linear_jerk_max_m_s3",
    "eef_motion.eef_angular_jerk_rms_rad_s3",
    "eef_motion.eef_linear_jerk_rms_m_s3",
)
_CAPTURE_APPROACH_TASKS = frozenset(
    {
        "p0_grasp",
        "t1_xyz",
        "t1_belt",
        "t1_so3",
        "t1_occ",
        "t3_full",
        "t4_sphere",
        "t4_sphere_tabletop",
        "t4_slider",
        "t4_can",
        "t5_replan",
    }
)


def _quality_v2_check(phase: str, metric: str) -> dict[str, Any]:
    if metric.startswith("action."):
        family = "action_l2"
    elif metric == "eef_motion.eef_translation_path_length_m":
        family = "translation_path_m"
    elif "linear_jerk" in metric:
        family = "linear_jerk_m_s3"
    elif "angular_jerk" in metric:
        family = "angular_jerk_rad_s3"
    else:
        family = "rotation_or_orientation_rad"
    return {
        "phase": phase,
        "metric": metric,
        "max": 10.0,
        "direction": "minimize",
        "paired_comparison_family": family,
        "paired_nonworse_absolute_tolerance": 0.01,
        "paired_nonworse_relative_tolerance": 0.0,
        "paired_strict_improvement_absolute": 0.05,
        "paired_strict_improvement_relative": 0.0,
    }


def _quality_v2_checks(*, task: str, include_jaw: bool) -> list[dict[str, Any]]:
    capture = task in _CAPTURE_APPROACH_TASKS
    checks = [
        _quality_v2_check(
            "acquisition_window" if capture else "full_episode",
            (
                "approach_axis.approach_axis_error_max_rad"
                if capture
                else "orientation_reference.orientation_reference_error_max_rad"
            ),
        )
    ]
    if include_jaw:
        checks.append(
            _quality_v2_check(
                "acquisition_window" if capture else "post_hold",
                "jaw_axis.jaw_axis_error_max_rad",
            )
        )
    checks.extend(
        _quality_v2_check("full_episode", metric)
        for metric in _QUALITY_V2_FULL_EPISODE_METRICS
    )
    return checks


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _calibration_wave_receipt() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_tasks = []
    binding_tasks = []
    for ordinal, task in enumerate(EXACT_TASKS):
        hashes = {
            "task_contract_sha256": f"{ordinal + 1:064x}",
            "task_receipt_sha256": f"{ordinal + 21:064x}",
            "task_config_sha256": f"{ordinal + 41:064x}",
            "task_quality_schema_sha256": f"{ordinal + 61:064x}",
            "reset_manifest_sha256": f"{ordinal + 81:064x}",
            "reset_identity_set_sha256": f"{ordinal + 101:064x}",
            "reset_row_set_sha256": f"{ordinal + 121:064x}",
            "evaluation_sha256": f"{ordinal + 141:064x}",
            "evaluation_payload_sha256": f"{ordinal + 161:064x}",
        }
        receipt_task = {
            "ordinal": ordinal,
            "task_id": task,
            "reset_count": 20,
            "task_quality_schema_version": "db0-episode-task-quality-v2",
            "reset_manifest_relative_path": f"tasks/{task}/reset_manifest.jsonl",
            "evaluation_relative_path": f"tasks/{task}/evaluation.json",
            **hashes,
        }
        receipt_tasks.append(receipt_task)
        binding_task = dict(receipt_task)
        binding_task["reset_identity_count"] = binding_task.pop("reset_count")
        binding_tasks.append(binding_task)
    receipt = {
        "schema_version": (
            release_auditor._optimal_auditor.QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA
        ),
        "scientific_partition": "metric_calibration",
        "transport_split": "validation",
        "manifest_seed": 20261350,
        "task_count": len(EXACT_TASKS),
        "episodes_per_task": 20,
        "total_reset_count": len(EXACT_TASKS) * 20,
        "task_order": list(EXACT_TASKS),
        "wave_contract_sha256": "a" * 64,
        "predeclaration_receipt_sha256": "b" * 64,
        "source_identity": {"wave_id": "release-unit-test"},
        "disjointness": {"verified": True},
        "tasks": receipt_tasks,
    }
    return receipt, binding_tasks


def _quality_v2_thresholds() -> dict[str, Any]:
    receipt, binding_tasks = _calibration_wave_receipt()
    receipt_sha256 = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    return {
        "schema_version": QUALITY_V2_THRESHOLDS_SCHEMA,
        "calibration_status": "frozen",
        "formal_freeze_eligible": True,
        "minimum_attempted_episodes": 20,
        "minimum_successful_episodes": 8,
        "calibration_wave_receipt": {
            "binding_status": "bound",
            **{key: value for key, value in receipt.items() if key != "tasks"},
            "tasks": binding_tasks,
            "relative_path": "provenance/calibration_wave/wave_receipt.json",
            "sha256": receipt_sha256,
            "file_sha256": receipt_sha256,
            "payload_sha256": receipt_sha256,
        },
        "tasks": {
            task: {
                "checks": _quality_v2_checks(task=task, include_jaw=task == "p0_grasp"),
                "orientation_mode": (
                    "world_down_tool_axis"
                    if task in _CAPTURE_APPROACH_TASKS
                    else "reset_frozen_full_orientation"
                ),
                "jaw_axis_mode": (
                    "object_xy_teacher_offset_mod_pi"
                    if task == "p0_grasp"
                    else "unconstrained"
                ),
                "provenance": {
                    "formal_freeze_eligible": True,
                    "attempted_episode_count": 20,
                    "successful_episode_count": 8,
                },
            }
            for task in EXACT_TASKS
        },
    }


@dataclass
class _Fixture:
    root: Path
    candidate_root: Path
    input_manifest: Path
    records: dict[str, dict[str, Any]]
    candidate_release_sha256: str
    candidate_checksums_sha256: str
    calibration_receipt_identity: dict[str, str]


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


def _quality_v2_summary(
    task: str,
    *,
    values: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    metric_values = dict.fromkeys(_QUALITY_V2_FULL_EPISODE_METRICS, 1.0)
    capture = task in _CAPTURE_APPROACH_TASKS
    orientation_metric = (
        "approach_axis.approach_axis_error_max_rad"
        if capture
        else "orientation_reference.orientation_reference_error_max_rad"
    )
    metric_values[orientation_metric] = 1.0
    if task == "p0_grasp":
        metric_values["jaw_axis.jaw_axis_error_max_rad"] = 1.0
    if values is not None:
        metric_values.update(values)
    summary: dict[str, Any] = {
        "schema_version": QUALITY_V2_SUMMARY_SCHEMA,
        "phases": {},
        "action": {},
        "eef_motion": {},
    }
    if capture:
        summary["phases"]["acquisition_window"] = {
            "approach_axis": {
                "approach_axis_error_max_rad": metric_values[orientation_metric]
            }
        }
    else:
        summary["orientation_reference"] = {
            "orientation_reference_error_max_rad": metric_values[orientation_metric]
        }
    if task == "p0_grasp":
        summary["phases"]["acquisition_window"]["jaw_axis"] = {
            "jaw_axis_error_max_rad": metric_values["jaw_axis.jaw_axis_error_max_rad"]
        }
    for metric in _QUALITY_V2_FULL_EPISODE_METRICS:
        namespace, name = metric.split(".", 1)
        summary[namespace][name] = metric_values[metric]
    return summary


def _quality_v2_value(summary: Mapping[str, Any], check: Mapping[str, Any]) -> float:
    value: Any = summary
    if check["phase"] != "full_episode":
        value = summary["phases"][check["phase"]]
    for part in check["metric"].split("."):
        value = value[part]
    return float(value)


def _quality_v2_gate(
    task: str,
    summary: Mapping[str, Any],
    threshold_sha256: str,
) -> dict[str, Any]:
    checks = _quality_v2_checks(task=task, include_jaw=task == "p0_grasp")
    gate_checks = [
        {
            "phase": check["phase"],
            "metric": check["metric"],
            "actual": _quality_v2_value(summary, check),
            "max": check["max"],
            "passed": _quality_v2_value(summary, check) <= float(check["max"]),
        }
        for check in checks
    ]
    return {
        "schema_version": QUALITY_V2_GATE_SCHEMA,
        "contract_schema_version": QUALITY_V2_THRESHOLDS_SCHEMA,
        "contract_sha256": threshold_sha256,
        "task_id": task,
        "passed": all(check["passed"] for check in gate_checks),
        "checks": gate_checks,
    }


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


def _attempt(
    task: str,
    episode_id: str,
    index: int,
    *,
    threshold_sha256: str,
    task_quality_value: float | None = None,
    quality_v2_values: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    quality_v2 = _quality_v2_summary(task, values=quality_v2_values)
    record = {
        "schema_version": ATTEMPT_SCHEMA,
        "task_id": task,
        "episode_id": episode_id,
        "candidate_index": index,
        "candidate_id": "planner" if index == 0 else "policy-1",
        "candidate_kind": "planner" if index == 0 else "policy",
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "replay_validation": {"passed": True, "outcomes_exact": True},
        "trajectory_completion": 1.0,
        "task_quality": _quality_summary(
            task,
            episode_id,
            (
                task_quality_value
                if task_quality_value is not None
                else 1.0
                if index == 0
                else 1.01
            ),
        ),
        "quality_v2": quality_v2,
        "quality_v2_sha256": _payload_sha256(quality_v2),
        "quality_v2_gate": _quality_v2_gate(task, quality_v2, threshold_sha256),
        "completion_time_s": 1.0,
        "return": 1.0,
        "control_steps": 500,
        "action_l2_sum": 2.0,
        "issued_equals_applied": task != "t5_replan",
        "t5_replan_causal_timing_passed": True if task == "t5_replan" else None,
        "impact_end_to_first_qualifying_applied_correction_s": (
            0.1 if task == "t5_replan" else None
        ),
        "attempt_tape": f"lightweight/{episode_id}/candidate-{index:03d}.npz",
        "attempt_tape_sha256": "a" * 64,
        "eligible": True,
    }
    return record


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

    thresholds = _quality_v2_thresholds()
    receipt, _ = _calibration_wave_receipt()
    wave_binding = thresholds["calibration_wave_receipt"]
    calibration_receipt_identity = {
        "relative_path": wave_binding["relative_path"],
        "file_sha256": wave_binding["file_sha256"],
        "payload_sha256": wave_binding["payload_sha256"],
    }
    records: dict[str, dict[str, Any]] = {}
    for task in EXACT_TASKS:
        dataset_root = root / "datasets" / task
        dataset_root.mkdir(parents=True)
        receipt_path = dataset_root.joinpath(
            *Path(calibration_receipt_identity["relative_path"]).parts
        )
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_bytes(_canonical_bytes(receipt))
        thresholds_path = dataset_root / "quality_v2_thresholds.json"
        _write_json(thresholds_path, thresholds)
        thresholds_sha256 = _sha256(thresholds_path)
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
                "quality_v2_threshold_identity": {
                    "schema_version": QUALITY_V2_THRESHOLDS_SCHEMA,
                    "sha256": thresholds_sha256,
                },
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
                (
                    _attempt(
                        task,
                        episode_id,
                        0,
                        threshold_sha256=thresholds_sha256,
                    ),
                    _attempt(
                        task,
                        episode_id,
                        1,
                        threshold_sha256=thresholds_sha256,
                    ),
                )
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
                "quality_v2_thresholds_sha256": thresholds_sha256,
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
                    "quality_v2_threshold_identity": {
                        "schema_version": QUALITY_V2_THRESHOLDS_SCHEMA,
                        "sha256": thresholds_sha256,
                    },
                    "quality_v2_calibration_wave_receipt_identity": (
                        calibration_receipt_identity
                    ),
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
            "quality_v2_thresholds_path": str(thresholds_path.resolve()),
            "quality_v2_thresholds_sha256": thresholds_sha256,
        }
    fixture = _Fixture(
        root=root,
        candidate_root=candidate_root,
        input_manifest=root / "release_inputs.json",
        records=records,
        candidate_release_sha256=candidate_release_sha256,
        candidate_checksums_sha256=candidate_checksums_sha256,
        calibration_receipt_identity=calibration_receipt_identity,
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


@pytest.fixture
def synthetic_tape_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep exact-14 release fixtures small; tape replay has a focused test below."""

    monkeypatch.setattr(
        release_auditor,
        "_audit_attempt_tape_binding",
        lambda *args, **kwargs: None,
    )


def test_exact14_release_builds_then_independent_audit_grants_eligibility(
    tmp_path: Path,
    synthetic_tape_replay: None,
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
    assert report["schema_version"] == RELEASE_AUDIT_SCHEMA
    assert report["summary"]["accepted_count"] == 1400
    assert report["quality_v2_calibration_wave_receipt_identity"] == (
        fixture.calibration_receipt_identity
    )
    manifest = json.loads(
        (output / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == RELEASE_MANIFEST_SCHEMA
    assert manifest["tasks"] == list(EXACT_TASKS)
    assert manifest["aggregate"]["source_counts"] == {"expert_dominant": 1400}
    assert manifest["quality_v2_calibration_wave_receipt_identity"] == (
        fixture.calibration_receipt_identity
    )


def test_release_accepts_audit_with_release_sha_in_summary(
    tmp_path: Path,
    synthetic_tape_replay: None,
) -> None:
    fixture = _make_fixture(tmp_path)
    record = fixture.records["p0_grasp"]
    audit_path = Path(record["audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    release_sha = audit.pop("candidate_release_manifest_sha256")
    audit["summary"]["candidate_release_manifest_sha256"] = release_sha
    _seal_payload(audit)
    _write_json(audit_path, audit)
    record["audit_sha256"] = _sha256(audit_path)
    _write_input_manifest(fixture)

    output = fixture.root / "release" / "unified14-summary-sha"
    report = build_and_audit_release(
        input_manifest=fixture.input_manifest,
        output_root=output,
        auditor_commit=AUDITOR_COMMIT,
    )

    assert report["status"] == "passed"
    assert report["release_eligible"] is True


def test_release_input_rejects_missing_task(
    tmp_path: Path,
    synthetic_tape_replay: None,
) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.records.pop("t1_xyz")
    _write_input_manifest(fixture)

    with pytest.raises(ReleaseAuditError, match="ordered exact14"):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)


def test_release_rejects_string_accepted_even_when_dataset_is_resealed(
    tmp_path: Path,
    synthetic_tape_replay: None,
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
    synthetic_tape_replay: None,
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

    with pytest.raises(
        ReleaseAuditError,
        match="selection result does not reproduce|does not dominate planner",
    ):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)


def test_release_rejects_candidate_release_hash_tamper(
    tmp_path: Path,
    synthetic_tape_replay: None,
) -> None:
    fixture = _make_fixture(tmp_path)
    path = fixture.candidate_root / "release_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["release_id"] = "tampered"
    _write_json(path, payload)

    with pytest.raises(
        ReleaseAuditError, match="file hash tamper|release-manifest hash mismatch"
    ):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)


def _pareto_fixture(
    task: str = "t1_xyz",
) -> tuple[dict[str, Any], dict[str, Any], str]:
    threshold_sha256 = "c" * 64
    thresholds = _quality_v2_thresholds()
    quality_v2_contract = _quality_v2_dominance_contract(
        thresholds,
        task=task,
        thresholds_sha256=threshold_sha256,
    )
    planner_contract = _normalize_contract(
        {
            "planner_dominance": _contract(
                task,
                "calibration.json",
                "d" * 64,
            )
        },
        task,
    )
    return planner_contract, quality_v2_contract, threshold_sha256


@pytest.mark.parametrize(
    "metric",
    [
        "action.action_total_variation_l2_mean_per_transition",
        "action.action_max_second_difference_l2",
        "eef_motion.eef_translation_path_length_m",
        "eef_motion.eef_rotation_path_length_rad",
        "approach_axis.approach_axis_error_max_rad",
        "eef_motion.eef_angular_jerk_max_rad_s3",
        "eef_motion.eef_linear_jerk_rms_m_s3",
    ],
)
def test_qv3_degradation_blocks_learned_candidate(metric: str) -> None:
    contract, quality_v2_contract, threshold_sha256 = _pareto_fixture()
    planner = _attempt(
        "t1_xyz",
        "same-reset",
        0,
        threshold_sha256=threshold_sha256,
    )
    learned = _attempt(
        "t1_xyz",
        "same-reset",
        1,
        threshold_sha256=threshold_sha256,
        quality_v2_values={metric: 1.2},
    )

    assert not _planner_pareto_dominates(
        learned,
        planner,
        contract,
        quality_v2_contract,
    )


def test_qv3_strict_improvement_selects_learned_and_exact_tie_selects_planner() -> None:
    contract, quality_v2_contract, threshold_sha256 = _pareto_fixture()
    planner = _attempt(
        "t1_xyz",
        "same-reset",
        0,
        threshold_sha256=threshold_sha256,
    )
    tied = _attempt(
        "t1_xyz",
        "same-reset",
        1,
        threshold_sha256=threshold_sha256,
        task_quality_value=1.0,
    )
    assert (
        _selected(
            [planner, tied],
            contract=contract,
            quality_v2_contract=quality_v2_contract,
        )["candidate_index"]
        == 0
    )

    improved = _attempt(
        "t1_xyz",
        "same-reset",
        1,
        threshold_sha256=threshold_sha256,
        task_quality_value=1.0,
        quality_v2_values={"eef_motion.eef_translation_path_length_m": 0.8},
    )
    assert _planner_pareto_dominates(
        improved,
        planner,
        contract,
        quality_v2_contract,
    )
    assert (
        _selected(
            [planner, improved],
            contract=contract,
            quality_v2_contract=quality_v2_contract,
        )["candidate_index"]
        == 1
    )


def test_qv3_absolute_gate_rejects_learned_before_pareto() -> None:
    contract, quality_v2_contract, threshold_sha256 = _pareto_fixture()
    planner = _attempt(
        "t1_xyz",
        "same-reset",
        0,
        threshold_sha256=threshold_sha256,
    )
    learned = _attempt(
        "t1_xyz",
        "same-reset",
        1,
        threshold_sha256=threshold_sha256,
    )
    learned["quality_v2_gate"]["passed"] = False
    learned["eligible"] = False

    assert (
        _selected(
            [planner, learned],
            contract=contract,
            quality_v2_contract=quality_v2_contract,
        )["candidate_index"]
        == 0
    )


def test_t5_causal_latency_is_same_reset_nonworse_and_strict_dimension() -> None:
    contract, quality_v2_contract, threshold_sha256 = _pareto_fixture("t5_replan")
    planner = _attempt(
        "t5_replan",
        "same-reset",
        0,
        threshold_sha256=threshold_sha256,
        task_quality_value=1.0,
    )
    degraded = _attempt(
        "t5_replan",
        "same-reset",
        1,
        threshold_sha256=threshold_sha256,
    )
    degraded["impact_end_to_first_qualifying_applied_correction_s"] = 0.2
    assert not _planner_pareto_dominates(
        degraded,
        planner,
        contract,
        quality_v2_contract,
    )

    improved = _attempt(
        "t5_replan",
        "same-reset",
        1,
        threshold_sha256=threshold_sha256,
        task_quality_value=1.0,
    )
    improved["impact_end_to_first_qualifying_applied_correction_s"] = 0.05
    assert _planner_pareto_dominates(
        improved,
        planner,
        contract,
        quality_v2_contract,
    )
    assert (
        _selected(
            [planner, improved],
            contract=contract,
            quality_v2_contract=quality_v2_contract,
        )["candidate_index"]
        == 1
    )


def test_qv3_formal_dynamic_inventory_and_threshold_sha_fail_closed() -> None:
    thresholds = _quality_v2_thresholds()
    ten = _quality_v2_dominance_contract(
        thresholds,
        task="t1_xyz",
        thresholds_sha256="c" * 64,
    )
    eleven = _quality_v2_dominance_contract(
        thresholds,
        task="p0_grasp",
        thresholds_sha256="c" * 64,
    )
    assert len(ten["metrics"]) == 10
    assert len(eleven["metrics"]) == 11
    assert any(
        spec["phase"] == "acquisition_window"
        and spec["metric"] == "approach_axis.approach_axis_error_max_rad"
        and spec["key"] == "approach_verticality"
        and spec["group"] == "grasp_geometry"
        for spec in ten["metrics"]
    )

    wrong_phase = copy.deepcopy(thresholds)
    next(
        check
        for check in wrong_phase["tasks"]["t1_xyz"]["checks"]
        if check["metric"] == "eef_motion.eef_angular_jerk_rms_rad_s3"
    )["phase"] = "acquisition_window"
    with pytest.raises(ReleaseAuditError, match="inventory mismatch"):
        _quality_v2_dominance_contract(
            wrong_phase,
            task="t1_xyz",
            thresholds_sha256="c" * 64,
        )

    provisional = copy.deepcopy(thresholds)
    provisional["formal_freeze_eligible"] = False
    provisional["calibration_status"] = "provisional"
    with pytest.raises(ReleaseAuditError, match="formal freeze"):
        _quality_v2_dominance_contract(
            provisional,
            task="t1_xyz",
            thresholds_sha256="c" * 64,
        )
    with pytest.raises(ReleaseAuditError, match="SHA-256"):
        _quality_v2_dominance_contract(
            thresholds,
            task="t1_xyz",
            thresholds_sha256="",
        )


def test_qv3_missing_metric_and_gate_threshold_hash_fail_closed() -> None:
    contract, quality_v2_contract, threshold_sha256 = _pareto_fixture()
    planner = _attempt(
        "t1_xyz",
        "same-reset",
        0,
        threshold_sha256=threshold_sha256,
    )
    missing = _attempt(
        "t1_xyz",
        "same-reset",
        1,
        threshold_sha256=threshold_sha256,
    )
    del missing["quality_v2"]["action"]["action_total_variation_l2_mean_per_transition"]
    missing["quality_v2_sha256"] = _payload_sha256(missing["quality_v2"])
    with pytest.raises(ReleaseAuditError, match="missing metric"):
        _planner_pareto_dominates(
            missing,
            planner,
            contract,
            quality_v2_contract,
        )

    wrong_hash = _attempt(
        "t1_xyz",
        "same-reset",
        1,
        threshold_sha256=threshold_sha256,
    )
    wrong_hash["quality_v2_gate"]["contract_sha256"] = "e" * 64
    with pytest.raises(ReleaseAuditError, match="gate identity"):
        _planner_pareto_dominates(
            wrong_hash,
            planner,
            contract,
            quality_v2_contract,
        )


@pytest.mark.parametrize("missing", ["quality_v2", "quality_v2_sha256"])
def test_qv3_missing_summary_or_summary_hash_fails_closed(missing: str) -> None:
    contract, quality_v2_contract, threshold_sha256 = _pareto_fixture()
    planner = _attempt(
        "t1_xyz",
        "same-reset",
        0,
        threshold_sha256=threshold_sha256,
    )
    learned = _attempt(
        "t1_xyz",
        "same-reset",
        1,
        threshold_sha256=threshold_sha256,
    )
    del learned[missing]

    with pytest.raises(ReleaseAuditError, match="summary schema|summary SHA-256"):
        _planner_pareto_dominates(
            learned,
            planner,
            contract,
            quality_v2_contract,
        )


def test_release_input_v01_is_historical_and_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "RLD2" / "legacy-release-inputs.json"
    _write_json(path, {"schema_version": LEGACY_RELEASE_INPUT_SCHEMA})

    with pytest.raises(ReleaseAuditError, match="historical"):
        _load_release_inputs(path)


@pytest.mark.parametrize(
    "missing",
    ["quality_v2_thresholds_path", "quality_v2_thresholds_sha256"],
)
def test_release_input_v02_requires_threshold_path_and_sha(
    tmp_path: Path,
    missing: str,
) -> None:
    task_rows = []
    for task in EXACT_TASKS:
        row = {
            "task": task,
            "dataset_root": "unused",
            "dataset_card_sha256": "0" * 64,
            "checksums_sha256": "1" * 64,
            "candidate_manifest_sha256": "2" * 64,
            "audit_path": "unused",
            "audit_sha256": "3" * 64,
            "input_inventory_path": "unused",
            "input_inventory_sha256": "4" * 64,
            "quality_v2_thresholds_path": "unused",
            "quality_v2_thresholds_sha256": "5" * 64,
        }
        if task == EXACT_TASKS[0]:
            del row[missing]
        task_rows.append(row)
    payload = _seal_payload(
        {
            "schema_version": RELEASE_INPUT_SCHEMA,
            "release_id": RELEASE_ID,
            "candidate_release_root": "unused",
            "candidate_release_manifest_sha256": "6" * 64,
            "candidate_release_checksums_sha256": "7" * 64,
            "tasks": task_rows,
        }
    )
    path = tmp_path / "RLD2" / "release-inputs.json"
    _write_json(path, payload)

    with pytest.raises(ReleaseAuditError, match="field inventory"):
        _load_release_inputs(path)


def test_attempt_tape_hash_mismatch_fails_before_summary_acceptance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "RLD2" / "dataset"
    tape = root / "lightweight" / "attempt.npz"
    tape.parent.mkdir(parents=True)
    tape.write_bytes(b"not-the-declared-tape")
    record = {
        "schema_version": ATTEMPT_SCHEMA,
        "task_id": "t1_xyz",
        "episode_id": "same-reset",
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "trajectory_completion": 1.0,
        "completion_time_s": 1.0,
        "issued_equals_applied": True,
        "t5_replan_causal_timing_passed": None,
        "impact_end_to_first_qualifying_applied_correction_s": None,
        "attempt_tape": "lightweight/attempt.npz",
        "attempt_tape_sha256": "0" * 64,
    }

    with pytest.raises(ReleaseAuditError, match="tape.*checksum mismatch"):
        _audit_attempt_tape_binding(
            root,
            record,
            task="t1_xyz",
            quality_v2_thresholds=_quality_v2_thresholds(),
            quality_v2_thresholds_sha256="c" * 64,
        )


def test_release_reopens_calibration_receipt_and_rejects_missing_sidecar(
    tmp_path: Path,
    synthetic_tape_replay: None,
) -> None:
    fixture = _make_fixture(tmp_path)
    task = "t1_xyz"
    dataset_root = Path(fixture.records[task]["dataset_root"])
    sidecar = dataset_root.joinpath(
        *Path(fixture.calibration_receipt_identity["relative_path"]).parts
    )
    sidecar.unlink()
    _refresh_dataset(fixture, task)

    with pytest.raises(ReleaseAuditError, match="calibration receipt.*missing"):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)


def test_release_reopens_calibration_receipt_and_rejects_tampered_sidecar(
    tmp_path: Path,
    synthetic_tape_replay: None,
) -> None:
    fixture = _make_fixture(tmp_path)
    task = "t1_xyz"
    dataset_root = Path(fixture.records[task]["dataset_root"])
    sidecar = dataset_root.joinpath(
        *Path(fixture.calibration_receipt_identity["relative_path"]).parts
    )
    sidecar.write_bytes(sidecar.read_bytes() + b"\n")
    _refresh_dataset(fixture, task)

    with pytest.raises(ReleaseAuditError, match="calibration receipt.*SHA-256"):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)


def test_release_rejects_old_passed_task_audit_without_receipt_identity(
    tmp_path: Path,
    synthetic_tape_replay: None,
) -> None:
    fixture = _make_fixture(tmp_path)
    task = "t1_xyz"
    record = fixture.records[task]
    audit_path = Path(record["audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["summary"].pop("quality_v2_calibration_wave_receipt_identity")
    _seal_payload(audit)
    _write_json(audit_path, audit)
    record["audit_sha256"] = _sha256(audit_path)
    _write_input_manifest(fixture)

    with pytest.raises(ReleaseAuditError, match="audit summary contract"):
        _collect_release(fixture.input_manifest, release_auditor_commit=AUDITOR_COMMIT)


def test_release_v02_manifest_and_audit_schemas_are_historical(tmp_path: Path) -> None:
    assert "rlinf-dynamic-benchmark-rld2-release-v0.2" in (
        HISTORICAL_RELEASE_MANIFEST_SCHEMAS
    )
    assert "rlinf-dynamic-benchmark-rld2-release-audit-v0.2" in (
        HISTORICAL_RELEASE_AUDIT_SCHEMAS
    )
    assert RELEASE_MANIFEST_SCHEMA.endswith("-v0.3")
    assert RELEASE_AUDIT_SCHEMA.endswith("-v0.3")

    root = tmp_path / "RLD2" / "historical-release"
    root.mkdir(parents=True)
    _write_json(
        root / "release_manifest.json",
        {"schema_version": "rlinf-dynamic-benchmark-rld2-release-v0.2"},
    )
    manifest_sha256 = _sha256(root / "release_manifest.json")
    (root / "SHA256SUMS").write_text(
        f"{manifest_sha256}  release_manifest.json\n", encoding="utf-8"
    )
    failed_audit = root / "failed_release_audit.json"

    with pytest.raises(ReleaseAuditError, match="historical release-manifest"):
        release_auditor.audit_release(
            input_manifest=tmp_path / "RLD2" / "unused-inputs.json",
            release_root=root,
            expected_release_manifest_sha256=manifest_sha256,
            expected_checksums_sha256=_sha256(root / "SHA256SUMS"),
            auditor_commit=AUDITOR_COMMIT,
            output=failed_audit,
        )
    failed = json.loads(failed_audit.read_text(encoding="utf-8"))
    assert failed["schema_version"] == RELEASE_AUDIT_SCHEMA
    assert failed["release_eligible"] is False
