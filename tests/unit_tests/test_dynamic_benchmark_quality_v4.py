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

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from se3_wam.benchmark.contracts import EventRecord, stable_sha256
from se3_wam.benchmark.dataset import _validate_quality_v4_validation
from se3_wam.benchmark.registry import get_task_spec
from se3_wam.benchmark.trajectory_quality_v4 import (
    EXACT14_ORIENTATION_CONTRACT,
    PhysicsRateEEFReducer,
    compare_replayed_observations_v4,
    trajectory_field_contract_manifest,
)

from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _audit_quality_v4_contract,
    _audit_quality_v4_full_exports,
    _audit_quality_v4_source_inventory,
)
from examples.embodiment.audit_dynamic_benchmark_rld2_release import (
    ACCEPTED_PER_TASK,
    EXACT_TASKS,
    ReleaseAuditError,
    quality_v4_release_readiness,
)
from examples.embodiment.build_dynamic_benchmark_quality_v4_candidate import (
    assess_candidate_evidence,
)
from examples.embodiment.build_dynamic_benchmark_quality_v4_provisional import (
    build_provisional_thresholds,
)
from examples.embodiment.dynamic_benchmark_evaluation_attempt import (
    load_quality_v4_rollout_reference,
    materialize_quality_v4_evaluation_attempt,
    materialize_quality_v4_fresh_replay_attempt,
    materialize_quality_v4_winner_export,
)
from examples.embodiment.dynamic_benchmark_quality_v4 import (
    QUALITY_V4_ARTIFACT_SUBDIRECTORY,
    QUALITY_V4_ATTEMPT_SUBDIRECTORY,
    QUALITY_V4_FULL_EXPORT_SUBDIRECTORY,
    QUALITY_V4_LIGHTWEIGHT_SUBDIRECTORY,
    QUALITY_V4_ROLLOUT_REFERENCE_SCHEMA,
    QUALITY_V4_SEGMENTATION_CONTRACT_SCHEMA,
    QUALITY_V4_T5_ACTION_HISTORY_SCHEMA,
    _quality_v4_applied_history,
    audit_quality_v4_full_export,
    audit_quality_v4_lightweight_source,
    build_quality_v4_attempt,
    build_quality_v4_rollout_source,
    dataset_quality_v4_validation,
    finalize_quality_v4_fresh_replay,
    load_quality_v4_thresholds,
    paired_pareto_winner,
    quality_v4_segmentation_contract,
    validate_quality_v4_threshold_candidate,
    validate_quality_v4_thresholds,
    write_quality_v4_attempt,
    write_quality_v4_full_export,
    write_quality_v4_lightweight_source,
)
from examples.embodiment.export_dynamic_benchmark_optimal_trajectories import (
    attach_quality_v4_winner_validation,
)
from examples.embodiment.merge_optimal_export_shards import (
    _copy_quality_v4_artifacts,
    _validate_quality_v4_artifacts,
    _validate_quality_v4_shard_contract,
)

_REPRESENTATIVE_TASKS = (
    "p0_grasp",
    "t2_se3",
    "t3_full",
    "t4_sphere",
    "t4_can",
    "t5_replan",
)


def _field_contract(task_id: str) -> dict[str, object]:
    return trajectory_field_contract_manifest(
        task_id=task_id,
        state_schema_sha256="a" * 64,
        joint_lower=(-1.0, -2.0),
        joint_upper=(1.0, 2.0),
        joint_limited=(True, True),
        joint_limit_tolerance=(1.0e-6, 1.0e-6),
        qvel_abs_max=(2.0, 3.0),
        workspace_bounds_m=((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)),
        object_twist_abs_max=(3.0, 3.0, 3.0, 4.0, 4.0, 4.0),
        reward_bounds=(-1.0, 2.0),
        contact_impulse_max_by_name={"object_robot": 1.0},
    )


def _physics_pose(count: int) -> np.ndarray:
    poses = np.zeros((count, 7), dtype=np.float64)
    poses[:, 0] = np.linspace(0.0, 0.1, count)
    poses[:, 6] = 1.0
    return poses


def _field_tape(
    task_id: str, issued: np.ndarray, applied: np.ndarray
) -> dict[str, object]:
    action_count = issued.shape[0]
    observation_count = action_count + 1
    physics_count = action_count * 25 + 1
    eef = _physics_pose(observation_count)
    obj = np.zeros((observation_count, 7), dtype=np.float64)
    obj[:, 0] = eef[:, 0]
    obj[:, 3] = 1.0
    issue_time = np.arange(action_count, dtype=np.float64) / 20.0
    applied_time = issue_time.copy()
    applied_source_step = np.arange(action_count, dtype=np.int64)
    if task_id == "t5_replan":
        applied_source_step = np.arange(action_count, dtype=np.int64) - 1
    return {
        "simulator": {
            "qpos": np.zeros((physics_count, 2), dtype=np.float64),
            "qvel": np.zeros((physics_count, 2), dtype=np.float64),
        },
        "observation": {
            "eef_pose_xyzw": eef,
            "object_pose_wxyz": obj,
            "object_twist_world": np.zeros((observation_count, 6), dtype=np.float64),
            "fingerpad_closing_axis_world": np.tile(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                (observation_count, 1),
            ),
            "rgb": np.zeros((observation_count, 2, 2, 2, 3), dtype=np.uint8),
            "depth_m": np.ones((observation_count, 2, 2, 2, 1), dtype=np.float32),
            "depth_valid_mask": np.ones(
                (observation_count, 2, 2, 2, 1), dtype=np.bool_
            ),
            "segmentation": np.zeros((observation_count, 2, 2, 2, 1), dtype=np.int32),
        },
        "action": {"issued": issued, "applied": applied},
        "physics": {
            "eef_pose_xyzw": _physics_pose(physics_count),
            "contact_impulse_n_s": np.zeros((physics_count, 1), dtype=np.float64),
            "contact_names": ["object_robot"],
        },
        "result": {
            "reward": np.zeros(action_count, dtype=np.float64),
            "progress": np.linspace(
                1.0 / action_count,
                1.0,
                action_count,
                dtype=np.float64,
            ),
        },
        "mask": {"history_valid_mask": np.ones((observation_count, 2), dtype=np.bool_)},
        "clock": {
            "physics_time_s": np.arange(physics_count, dtype=np.float64) / 500.0,
            "observation_time_s": np.arange(observation_count, dtype=np.float64) / 20.0,
            "action_issue_time_s": issue_time,
            "action_applied_time_s": applied_time,
            "action_applied_source_policy_step": applied_source_step,
            "observation_policy_step": np.arange(observation_count, dtype=np.int64),
        },
    }


def _t5_action_history(issued: np.ndarray) -> dict[str, object]:
    issued_rows = []
    applied_rows = []
    for policy_step, values in enumerate(issued):
        row = {
            "policy_step": policy_step,
            "issue_time_s": policy_step / 20.0,
            "apply_policy_step": policy_step + 1,
            "apply_time_s": (policy_step + 1) / 20.0,
            "values": values.tolist(),
        }
        issued_rows.append(row)
        if policy_step + 1 < len(issued):
            applied_rows.append(
                {
                    **row,
                    "actual_apply_policy_step": policy_step + 1,
                    "actual_apply_time_s": (policy_step + 1) / 20.0,
                }
            )
    payload: dict[str, object] = {
        "schema_version": QUALITY_V4_T5_ACTION_HISTORY_SCHEMA,
        "issued_actions": issued_rows,
        "applied_actions": applied_rows,
        "controller_applied_source_policy_step": (
            np.arange(len(issued), dtype=np.int64) - 1
        ).tolist(),
    }
    payload["history_sha256"] = stable_sha256(payload)
    return payload


def _events(task_id: str) -> tuple[EventRecord, ...]:
    task = get_task_spec(task_id)
    names = (task.task_start_event, *task.success_stages, "success")
    return tuple(
        EventRecord(name=name, physics_step=index * 5, time_s=index * 0.01)
        for index, name in enumerate(names)
    )


def _source(
    task_id: str,
    thresholds: dict[str, object],
    *,
    oscillation: float = 0.0,
    return_diagnostic: float = 0.0,
) -> dict[str, object]:
    issued = np.zeros((4, 7), dtype=np.float64)
    if oscillation:
        issued[:, 0] = (0.0, oscillation, 0.0, oscillation)
    applied = issued.copy()
    if task_id == "t5_replan":
        issued[:, 1] = (0.1, 0.0, -0.1, 0.0)
        applied = np.zeros_like(issued)
        applied[1:] = issued[:-1]
    observations = _physics_pose(5)
    obj = np.zeros((5, 7), dtype=np.float64)
    obj[:, 0] = observations[:, 0]
    obj[:, 3] = 1.0
    applicable = [
        phase
        for phase, row in EXACT14_ORIENTATION_CONTRACT[task_id].items()
        if row["applicable"]
    ]
    paths = {
        phase: {
            "reference_path": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            "reference_kind": "teacher_reference",
            "corridor_half_width_m": 0.5,
        }
        for phase in applicable
    }
    orientations: dict[str, dict[str, object]] = {}
    for phase in applicable:
        phase_contract = EXACT14_ORIENTATION_CONTRACT[task_id][phase]
        reference: dict[str, object] = {
            "reference_mode": phase_contract["reference_mode"],
            "tool_axis_reference_world": [0.0, 0.0, 1.0],
            "jaw_axis_reference_world": [1.0, 0.0, 0.0],
            "full_orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
            "error_limit_rad": 0.2,
        }
        checks = set(phase_contract["checks"])
        if checks & {"object_tilt", "roll_pitch"}:
            reference["object_orientation_reference_wxyz"] = [1.0, 0.0, 0.0, 0.0]
        if phase_contract["reference_time_mode"] == "per_sample":
            for name in (
                "tool_axis_reference_world",
                "jaw_axis_reference_world",
                "full_orientation_reference_xyzw",
                "object_orientation_reference_wxyz",
            ):
                if name in reference:
                    reference[name] = np.tile(
                        np.asarray(reference[name], dtype=np.float64),
                        (5, 1),
                    ).tolist()
        orientations[phase] = reference
    physics = PhysicsRateEEFReducer()
    physics.update_many(np.arange(101, dtype=np.float64) / 500.0, _physics_pose(101))
    tolerance_sha256 = thresholds["tasks"][task_id]["vision_tolerance"][
        "tolerance_sha256"
    ]
    return {
        "episode_id": f"{task_id}-episode",
        "task_id": task_id,
        "reset_pair_key": f"{task_id}-reset-0001",
        "field_contract": _field_contract(task_id),
        "field_tape": _field_tape(task_id, issued, applied),
        "events": _events(task_id),
        "t5_action_history": (
            _t5_action_history(issued) if task_id == "t5_replan" else None
        ),
        "safety_contract": {
            "source": "rollout_reward_schema.safety_failures",
            "safety_failures": ["drop", "workspace_exit", "unsafe_contact"],
        },
        "final": {
            "success": True,
            "termination_reason": "success",
            "active_stage_progress": 1.0,
        },
        "replay_validation": {
            "structured_observations_exact": True,
            "event_ledger_exact": True,
            "outcomes_exact": True,
            "final_state_exact": True,
            "terminal_task_quality_exact": True,
            "quality_v4_summary_exact": True,
            "quality_v4_layer1_exact": True,
            "quality_v4_layer2_exact": True,
            "rgb_within_tolerance": True,
            "depth_within_tolerance": True,
            "vision_tolerance_sha256": tolerance_sha256,
        },
        "summary_inputs": {
            "issued_actions": issued,
            "applied_actions": applied,
            "control_eef_pose_xyzw": observations,
            "control_object_pose_wxyz": obj,
            "closing_axis_world": np.tile([1.0, 0.0, 0.0], (5, 1)),
            "progress": np.linspace(0.0, 1.0, 5, dtype=np.float64),
            "phase_slices": dict.fromkeys(applicable, (0, 4)),
            "path_references": paths,
            "orientation_references": orientations,
            "physics_rate_eef": physics.summary(),
            "continuous_dimensions": 6,
            "reversal_deadband": 0.02,
        },
        "thresholds": thresholds,
        "return_diagnostic": return_diagnostic,
    }


def _frozen_thresholds(provisional: dict[str, object]) -> dict[str, object]:
    frozen = copy.deepcopy(provisional)
    frozen["calibration_status"] = "frozen"
    frozen["formal_freeze_eligible"] = True
    frozen["splits_read"] = ["metric_calibration"]
    frozen["owner_review"] = {
        "approved": True,
        "reviewer": "unit-test-owner",
        "reviewed_at": "2026-08-11T00:00:00Z",
        "decision_record": "unit-test-fixture",
    }
    for source_name, source in frozen["calibration_sources"].items():
        source["status"] = "complete"
        source["evidence_sha256"] = stable_sha256({"source": source_name})
    segmentation_contract = quality_v4_segmentation_contract()
    frozen["segmentation_contract"] = segmentation_contract
    for task in frozen["tasks"].values():
        vision_tolerance = task["vision_tolerance"]
        vision_tolerance.pop("tolerance_sha256")
        vision_tolerance["calibration_status"] = "frozen"
        vision_tolerance["evidence_sha256"] = stable_sha256(
            {"vision": vision_tolerance["task_id"]}
        )
        vision_tolerance["segmentation_contract_sha256"] = segmentation_contract[
            "contract_sha256"
        ]
        vision_tolerance["tolerance_sha256"] = stable_sha256(vision_tolerance)
        for check in task["checks"]:
            check["max"] = 1.0e20
            check["paired_non_worse_tolerance"] = 0.0
            check["strict_improvement_margin"] = 0.0
            check["calibration_status"] = "frozen"
            identity = {"phase": check["phase"], "metric": check["metric"]}
            check["value_evidence"] = {
                "hard_bound_sha256": stable_sha256({**identity, "value": "bound"}),
                "good_bad_discriminability_sha256": stable_sha256(
                    {**identity, "value": "good_bad"}
                ),
                "paired_non_worse_tolerance_sha256": stable_sha256(
                    {**identity, "value": "tolerance"}
                ),
                "strict_improvement_margin_sha256": stable_sha256(
                    {**identity, "value": "margin"}
                ),
            }
            check["evidence_sha256"] = stable_sha256(check["value_evidence"])
    frozen.pop("thresholds_sha256")
    frozen["thresholds_sha256"] = stable_sha256(frozen)
    validate_quality_v4_thresholds(frozen, require_formal_freeze=True)
    return frozen


def _formal_candidate_thresholds(
    provisional: dict[str, object],
) -> dict[str, object]:
    candidate = _frozen_thresholds(provisional)
    candidate["calibration_status"] = "formal_candidate"
    candidate["formal_freeze_eligible"] = False
    candidate["owner_review"] = {
        "approved": False,
        "reviewer": None,
        "reviewed_at": None,
        "decision_record": None,
    }
    candidate.pop("thresholds_sha256")
    candidate["thresholds_sha256"] = stable_sha256(candidate)
    return candidate


def test_qv4_provisional_inventory_is_exact14_unfrozen_and_not_test_tuned(
    tmp_path: Path,
) -> None:
    thresholds = build_provisional_thresholds()
    validation = validate_quality_v4_thresholds(thresholds)
    assert validation["task_count"] == 14
    assert not validation["formal_freeze_eligible"]
    assert validation["test_splits_read"] == []
    assert thresholds["splits_read"] == []
    assert set(thresholds["tasks"]) == set(EXACT14_ORIENTATION_CONTRACT)
    with pytest.raises(ValueError, match="formally frozen"):
        validate_quality_v4_thresholds(thresholds, require_formal_freeze=True)

    forged = _frozen_thresholds(thresholds)
    forged["calibration_sources"]["fresh_deterministic_rl_pilot"].pop("evidence_sha256")
    forged.pop("thresholds_sha256")
    forged["thresholds_sha256"] = stable_sha256(forged)
    with pytest.raises(ValueError, match="claim formal freeze"):
        validate_quality_v4_thresholds(forged)

    incomplete = copy.deepcopy(thresholds)
    incomplete["tasks"]["p0_grasp"]["checks"].pop()
    incomplete.pop("thresholds_sha256")
    incomplete["thresholds_sha256"] = stable_sha256(incomplete)
    with pytest.raises(ValueError, match="metric inventory mismatch"):
        validate_quality_v4_thresholds(incomplete)

    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(thresholds), encoding="utf-8")
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, receipt = load_quality_v4_thresholds(path, expected_file_sha256=file_sha256)
    assert loaded == thresholds
    assert receipt["file_sha256"] == file_sha256


def test_qv4_formal_candidate_is_validated_but_production_freeze_stays_blocked() -> (
    None
):
    candidate = _formal_candidate_thresholds(build_provisional_thresholds())
    validation = validate_quality_v4_threshold_candidate(candidate)

    assert validation["formal_candidate_validated"]
    assert not validation["formal_freeze_eligible"]
    assert not validation["owner_review_complete"]
    with pytest.raises(ValueError, match="formally frozen"):
        validate_quality_v4_thresholds(candidate, require_formal_freeze=True)

    missing_margin_evidence = copy.deepcopy(candidate)
    first_check = missing_margin_evidence["tasks"]["p0_grasp"]["checks"][0]
    first_check["value_evidence"].pop("strict_improvement_margin_sha256")
    first_check["evidence_sha256"] = stable_sha256(first_check["value_evidence"])
    missing_margin_evidence.pop("thresholds_sha256")
    missing_margin_evidence["thresholds_sha256"] = stable_sha256(
        missing_margin_evidence
    )
    with pytest.raises(ValueError, match="formal candidate"):
        validate_quality_v4_threshold_candidate(missing_margin_evidence)


def test_qv4_exact_segmentation_contract_is_frozen_and_hash_bound() -> None:
    contract = quality_v4_segmentation_contract()
    unsigned = dict(contract)
    recorded = unsigned.pop("contract_sha256")

    assert contract["schema_version"] == QUALITY_V4_SEGMENTATION_CONTRACT_SCHEMA
    assert contract["calibration_status"] == "frozen"
    assert contract["comparison"] == "component_sha256_exact"
    assert contract["exact"] is True
    assert contract["label_remap_allowed"] is False
    assert contract["task_ids"] == list(EXACT14_ORIENTATION_CONTRACT)
    assert recorded == stable_sha256(unsigned)


def test_qv4_candidate_evidence_assessment_reports_current_blockers() -> None:
    planner = {
        "schema_version": "se3-wam-qv4-planner-calibration-aggregator-manifest-v0.1",
        "status": "complete_calibration_evidence",
        "scientific_partition": "metric_calibration",
        "task_count": 14,
        "episodes_per_task": 20,
        "expected_reset_count": 280,
        "complete_reset_count": 280,
        "source_tape_count": 280,
    }
    good_bad = {
        "schema_version": "se3-wam-qv4-known-good-bad-evidence-v0.1",
        "classification": [
            {"expected_label": label}
            for label in (
                "known_good",
                "issued_action_jitter",
                "applied_action_jitter",
                "physics_rate_jerk",
                "path_detour",
                "path_backtrack",
                "orientation_drift",
                "object_tilt_or_slip",
            )
        ],
        "acceptance": {
            "known_good_retained": True,
            "all_corresponding_bad_rejected": True,
        },
        "scientific_boundary": {
            "source_runtime_split": "validation",
            "may_set_or_freeze_absolute_qv4_thresholds": False,
        },
    }
    readiness = assess_candidate_evidence(
        planner=planner,
        good_bad=good_bad,
        rl_pairs=None,
        evidence_files={
            "exact14x20_planner": "1" * 64,
            "known_good_bad_trajectories": "2" * 64,
            "fresh_deterministic_rl_pilot": None,
        },
    )

    assert not readiness["formal_candidate_ready"]
    assert not readiness["formal_freeze_eligible"]
    assert readiness["owner_review"]["approved"] is False
    assert "RL_PAIR_EVIDENCE_MISSING" in readiness["blockers"]
    assert (
        "GOOD_BAD_NOT_FORMAL_EXACT14_METRIC_CALIBRATION_EVIDENCE"
        in readiness["blockers"]
    )
    assert "PLANNER_TASK_CHECK_HARD_BOUND_EVIDENCE_MISSING" in readiness["blockers"]


@pytest.mark.parametrize("task_id", _REPRESENTATIVE_TASKS)
def test_qv4_six_family_canary_recomputes_layer1_and_rejects_provisional_layer2(
    task_id: str,
) -> None:
    thresholds = build_provisional_thresholds()
    attempt = build_quality_v4_attempt(_source(task_id, thresholds))
    assert attempt["layer1_gate"]["passed"]
    assert not attempt["layer2_gate"]["passed"]
    assert not attempt["eligible"]
    assert not attempt["formal_thresholds_frozen"]


@pytest.mark.parametrize("task_id", tuple(EXACT14_ORIENTATION_CONTRACT))
def test_qv4_exact14_threshold_metric_paths_resolve(task_id: str) -> None:
    thresholds = build_provisional_thresholds()
    attempt = build_quality_v4_attempt(_source(task_id, thresholds))

    assert attempt["layer1_gate"]["passed"]
    assert not attempt["layer2_gate"]["passed"]
    assert all(
        str(code).startswith("L2:") and str(code).endswith(":uncalibrated_threshold")
        for code in attempt["layer2_gate"]["reason_codes"]
    )


def test_qv4_t5_queue_ledger_builds_aligned_applied_control_trace() -> None:
    issued = np.zeros((4, 7), dtype=np.float64)
    issued[:, 0] = (0.1, 0.2, -0.1, 0.0)
    issue_time = np.arange(4, dtype=np.float64) / 20.0
    wrapped = _t5_action_history(issued)
    raw_history = {
        "issued_actions": wrapped["issued_actions"],
        "applied_actions": wrapped["applied_actions"],
    }

    applied, applied_time, source_steps, stored = _quality_v4_applied_history(
        "t5_replan",
        SimpleNamespace(canonical_action_history=raw_history),
        issued,
        issue_time,
    )

    np.testing.assert_array_equal(applied[0], np.zeros(7))
    np.testing.assert_array_equal(applied[1:], issued[:-1])
    np.testing.assert_array_equal(applied_time, issue_time)
    np.testing.assert_array_equal(source_steps, (-1, 0, 1, 2))
    assert stored["schema_version"] == QUALITY_V4_T5_ACTION_HISTORY_SCHEMA
    assert stored["history_sha256"] == stable_sha256(
        {key: value for key, value in stored.items() if key != "history_sha256"}
    )

    incomplete = copy.deepcopy(raw_history)
    incomplete["applied_actions"].pop()
    with pytest.raises(ValueError, match="omits or invents"):
        _quality_v4_applied_history(
            "t5_replan",
            SimpleNamespace(canonical_action_history=incomplete),
            issued,
            issue_time,
        )


def test_qv4_artifacts_use_parallel_paths_and_full_hdf5_is_regated(
    tmp_path: Path,
) -> None:
    thresholds = build_provisional_thresholds()
    source = _source("p0_grasp", thresholds)
    attempt = build_quality_v4_attempt(source)
    attempt_path = write_quality_v4_attempt(tmp_path, attempt)
    assert attempt_path.parent == tmp_path / QUALITY_V4_ATTEMPT_SUBDIRECTORY
    assert QUALITY_V4_ARTIFACT_SUBDIRECTORY == Path("quality_v4")
    lightweight_path = write_quality_v4_lightweight_source(
        tmp_path,
        source=source,
        recorded_attempt=attempt,
    )
    assert lightweight_path.parent == tmp_path / QUALITY_V4_LIGHTWEIGHT_SUBDIRECTORY
    lightweight_audit = audit_quality_v4_lightweight_source(lightweight_path)
    assert not lightweight_audit["raw_vision_arrays_present"]
    with h5py.File(lightweight_path, "r") as handle:
        assert all(
            "rgb" not in name and "depth_m" not in name for name in handle["arrays"]
        )

    export_path = tmp_path / QUALITY_V4_FULL_EXPORT_SUBDIRECTORY / "winner.h5"
    write_quality_v4_full_export(export_path, source=source, recorded_attempt=attempt)
    full_gate = audit_quality_v4_full_export(export_path)
    assert full_gate["full_export_recomputed"]
    assert not full_gate["passed"]
    assert not full_gate["eligible_for_behavior_cloning"]
    dataset_gate = dataset_quality_v4_validation(full_gate)
    assert not dataset_gate["passed"]
    assert not _validate_quality_v4_validation(dataset_gate)

    with h5py.File(export_path, "r+") as handle:
        first = sorted(handle["arrays"])[0]
        dataset = handle[f"arrays/{first}"]
        dataset[...] = np.zeros_like(dataset[...]) + 1
    with pytest.raises(ValueError, match="checksum drift"):
        audit_quality_v4_full_export(export_path)

    adapter_root = tmp_path / "evaluation-adapter"
    materialized = materialize_quality_v4_evaluation_attempt(adapter_root, source)
    assert materialized["lightweight_source_path"].startswith(
        "quality_v4/lightweight_sources/"
    )
    assert not materialized["lightweight_source_audit"]["raw_vision_arrays_present"]
    winner = materialize_quality_v4_winner_export(
        adapter_root,
        source=source,
        attempt=materialized["attempt"],
    )
    assert winner["full_export_path"].startswith("quality_v4/full_exports/")
    assert winner["full_export_gate_path"].endswith(".gate.json")
    assert not winner["dataset_quality_v4_validation"]["passed"]


@pytest.mark.parametrize(
    ("group", "field", "message"),
    (
        ("field_tape", "physics", "physics/clock source mappings"),
        ("physics", "contact_impulse_n_s", "physics/contact source fields"),
        ("action", "applied", "issued/applied actions"),
    ),
)
def test_qv4_full_export_fails_closed_on_missing_production_source_fields(
    tmp_path: Path,
    group: str,
    field: str,
    message: str,
) -> None:
    source = _source("p0_grasp", build_provisional_thresholds())
    attempt = build_quality_v4_attempt(source)
    tampered = copy.deepcopy(source)
    field_tape = tampered["field_tape"]
    if group == "field_tape":
        field_tape.pop(field)
    else:
        field_tape[group].pop(field)
    export_path = tmp_path / f"missing-{group}-{field}.h5"
    write_quality_v4_full_export(
        export_path,
        source=tampered,
        recorded_attempt=attempt,
    )
    with pytest.raises(ValueError, match=message):
        audit_quality_v4_full_export(export_path)


def test_qv4_attempt_rejects_task_vision_tolerance_identity_drift() -> None:
    thresholds = build_provisional_thresholds()
    source = _source("t4_can", thresholds)
    source["replay_validation"]["vision_tolerance_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="task-specific vision tolerance"):
        build_quality_v4_attempt(source)


def test_qv4_fresh_replay_recomputes_both_sources_before_issuing_exact_receipt() -> (
    None
):
    thresholds = build_provisional_thresholds()
    original_source = _source("t4_can", thresholds)
    replayed_source = copy.deepcopy(original_source)
    original_observation = SimpleNamespace(
        component_sha256={"privileged": "a" * 64},
        events_since_last_observation=(),
        rgb={"agentview": np.zeros((2, 2, 3), dtype=np.uint8)},
        depth_m={"agentview": np.ones((2, 2, 1), dtype=np.float32)},
    )
    replayed_observation = copy.deepcopy(original_observation)
    comparison = compare_replayed_observations_v4(
        [original_observation],
        [replayed_observation],
        vision_tolerance=thresholds["tasks"]["t4_can"]["vision_tolerance"],
    )
    finalized_source, attempt = finalize_quality_v4_fresh_replay(
        original_source=original_source,
        replayed_source=replayed_source,
        base_replay_validation={
            "outcomes_exact": True,
            "final_state_exact": True,
            "task_quality_exact": True,
        },
        observation_comparison=comparison,
    )

    receipt = finalized_source["replay_validation"]
    assert receipt["quality_v4_summary_exact"]
    assert receipt["quality_v4_layer1_exact"]
    assert receipt["quality_v4_layer2_exact"]
    assert attempt["layer1_gate"]["replay"]["passed"]

    changed_replay = copy.deepcopy(replayed_source)
    changed_replay["summary_inputs"]["issued_actions"][1, 0] = 0.25
    _, changed_attempt = finalize_quality_v4_fresh_replay(
        original_source=original_source,
        replayed_source=changed_replay,
        base_replay_validation={
            "outcomes_exact": True,
            "final_state_exact": True,
            "task_quality_exact": True,
        },
        observation_comparison=comparison,
    )
    assert not changed_attempt["layer1_gate"]["replay"]["passed"]
    assert not changed_attempt["eligible"]


def test_qv4_serial_rollout_source_is_built_from_raw_observation_and_physics_tapes(
    tmp_path: Path,
) -> None:
    thresholds = build_provisional_thresholds()
    direct = _source("t1_belt", thresholds)
    tape = direct["field_tape"]
    observations = []
    for index in range(5):
        observations.append(
            SimpleNamespace(
                time_s=float(index) / 20.0,
                policy_step=index,
                rgb={
                    "agentview": tape["observation"]["rgb"][index, 0],
                    "wrist": tape["observation"]["rgb"][index, 1],
                },
                depth_m={
                    "agentview": tape["observation"]["depth_m"][index, 0],
                    "wrist": tape["observation"]["depth_m"][index, 1],
                },
                segmentation={
                    "agentview": tape["observation"]["segmentation"][index, 0],
                    "wrist": tape["observation"]["segmentation"][index, 1],
                },
                privileged={
                    "eef_pose_xyzw": tape["observation"]["eef_pose_xyzw"][index],
                    "object_pose_wxyz": tape["observation"]["object_pose_wxyz"][index],
                    "object_twist_world": tape["observation"]["object_twist_world"][
                        index
                    ],
                    "fingerpad_closing_axis_world": np.asarray(
                        [1.0, 0.0, 0.0], dtype=np.float64
                    ),
                },
                component_sha256={
                    "metadata": f"metadata-{index}",
                    "privileged": f"privileged-{index}",
                    "rgb/agentview": f"rgb-{index}",
                    "depth_m/agentview": f"depth-{index}",
                },
                events_since_last_observation=(),
            )
        )
    physics_samples = [
        {
            "time_s": tape["clock"]["physics_time_s"][index],
            "simulator_qpos": tape["simulator"]["qpos"][index],
            "simulator_qvel": tape["simulator"]["qvel"][index],
            "eef_pose_xyzw": tape["physics"]["eef_pose_xyzw"][index],
            "object_robot": tape["physics"]["contact_impulse_n_s"][index, 0],
        }
        for index in range(len(tape["clock"]["physics_time_s"]))
    ]
    summary_inputs = direct["summary_inputs"]
    reference = {
        "schema_version": QUALITY_V4_ROLLOUT_REFERENCE_SCHEMA,
        "task_id": "t1_belt",
        "episode_id": "t1_belt-episode",
        "reset_pair_key": "t1_belt-reset-0001",
        "field_contract": direct["field_contract"],
        "history_valid_mask": np.ones((5, 2), dtype=np.bool_).tolist(),
        "phase_slices": summary_inputs["phase_slices"],
        "path_references": summary_inputs["path_references"],
        "orientation_references": summary_inputs["orientation_references"],
        "reversal_deadband": 0.02,
    }
    reference["reference_sha256"] = stable_sha256(reference)
    outcomes = [
        (False, False, False, None, 0.25),
        (False, False, False, None, 0.50),
        (False, False, False, None, 0.75),
        (True, False, True, "success", 1.0),
    ]

    def build_source(samples: list[dict[str, object]]) -> dict[str, object]:
        return build_quality_v4_rollout_source(
            record={
                "task_id": "t1_belt",
                "episode_id": "t1_belt-episode",
                "return": 1.0,
                "reward_schema_safety_failures": [
                    "drop",
                    "workspace_exit",
                    "unsafe_contact",
                ],
            },
            raw_env=SimpleNamespace(),
            observations=observations,
            issued_actions=tape["action"]["issued"],
            rewards=tape["result"]["reward"],
            outcomes=outcomes,
            physics_samples=samples,
            events=direct["events"],
            thresholds=thresholds,
            reference_contract=reference,
            replay_validation=direct["replay_validation"],
        )

    source = build_source(physics_samples)
    attempt = build_quality_v4_attempt(source)
    assert attempt["layer1_gate"]["passed"]
    assert not attempt["layer2_gate"]["passed"]
    missing_contact = copy.deepcopy(physics_samples)
    missing_contact[1].pop("object_robot")
    with pytest.raises(ValueError, match="registered contact field"):
        build_source(missing_contact)

    reference_root = tmp_path / "references"
    reference_path = reference_root / "t1_belt" / "t1_belt-episode.json"
    reference_path.parent.mkdir(parents=True)
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    loaded_reference = load_quality_v4_rollout_reference(
        reference_root,
        task_id="t1_belt",
        episode_id="t1_belt-episode",
        expected_state_schema_sha256="a" * 64,
    )
    assert loaded_reference == json.loads(json.dumps(reference))
    with pytest.raises(ValueError, match="state-schema identity mismatch"):
        load_quality_v4_rollout_reference(
            reference_root,
            task_id="t1_belt",
            episode_id="t1_belt-episode",
            expected_state_schema_sha256="b" * 64,
        )
    materialized = materialize_quality_v4_fresh_replay_attempt(
        output=tmp_path / "evaluation",
        record={
            "task_id": "t1_belt",
            "episode_id": "t1_belt-episode",
            "return": 1.0,
            "reward_schema_safety_failures": [
                "drop",
                "workspace_exit",
                "unsafe_contact",
            ],
        },
        raw_env=SimpleNamespace(),
        observations=observations,
        actions=tape["action"]["issued"],
        rewards=tape["result"]["reward"].tolist(),
        outcomes=outcomes,
        physics_samples=physics_samples,
        events=direct["events"],
        thresholds=thresholds,
        reference_contract=loaded_reference,
        base_replay_validation={
            "outcomes_exact": True,
            "final_state_exact": True,
            "task_quality_exact": True,
        },
        replay_capture={
            "raw_env": SimpleNamespace(),
            "observations": tuple(copy.deepcopy(observations)),
            "rewards": tuple(tape["result"]["reward"].tolist()),
            "outcomes": tuple(outcomes),
            "physics_samples": tuple(copy.deepcopy(physics_samples)),
            "events": tuple(direct["events"]),
        },
    )
    assert materialized["layer1_passed"]
    assert not materialized["layer2_passed"]
    assert materialized["fresh_replay_observation_comparison"]["passed"]
    assert materialized["attempt_path"].startswith("quality_v4/attempts/")
    assert materialized["lightweight_source_path"].startswith(
        "quality_v4/lightweight_sources/"
    )


def test_qv4_same_reset_pareto_uses_absolute_gates_and_ignores_return() -> None:
    thresholds = _frozen_thresholds(build_provisional_thresholds())
    planner = build_quality_v4_attempt(
        _source("t1_belt", thresholds, oscillation=0.2, return_diagnostic=100.0)
    )
    rl = build_quality_v4_attempt(
        _source("t1_belt", thresholds, oscillation=0.0, return_diagnostic=-100.0)
    )
    assert planner["eligible"] and rl["eligible"]
    selection = paired_pareto_winner(
        planner_attempt=planner,
        rl_attempt=rl,
        thresholds=thresholds,
    )
    assert selection["winner"] == "rl"
    assert selection["return_diagnostic_only"]
    assert selection["rl_non_worse_on_all_dimensions"]
    assert selection["rl_strictly_better_on_any_dimension"]

    failed_planner = copy.deepcopy(planner)
    failed_planner["eligible"] = False
    failed_planner["attempt_sha256"] = stable_sha256(
        {key: value for key, value in failed_planner.items() if key != "attempt_sha256"}
    )
    assert (
        paired_pareto_winner(
            planner_attempt=failed_planner,
            rl_attempt=rl,
            thresholds=thresholds,
        )["winner"]
        == "rl"
    )
    failed_rl = copy.deepcopy(rl)
    failed_rl["eligible"] = False
    failed_rl["attempt_sha256"] = stable_sha256(
        {key: value for key, value in failed_rl.items() if key != "attempt_sha256"}
    )
    assert (
        paired_pareto_winner(
            planner_attempt=failed_planner,
            rl_attempt=failed_rl,
            thresholds=thresholds,
        )["winner"]
        == "reject"
    )


@dataclass(frozen=True)
class _TraceFixture:
    request: object
    quality_v4_validation: dict[str, object] | None = None


def test_qv4_optimal_audit_and_exact14_release_readiness_recompute_full_gates(
    tmp_path: Path,
) -> None:
    thresholds = _frozen_thresholds(build_provisional_thresholds())
    source = _source("p0_grasp", thresholds)
    attempt = build_quality_v4_attempt(source)
    assert attempt["eligible"]
    attached_trace, production_export = attach_quality_v4_winner_validation(
        tmp_path / "production",
        trace=_TraceFixture(
            request=SimpleNamespace(episode_id="p0_grasp-episode"),
        ),
        source=source,
        attempt=attempt,
    )
    assert attached_trace.quality_v4_validation["passed"]
    assert production_export["full_export_path"] == (
        "quality_v4/full_exports/p0_grasp-episode.h5"
    )
    exported = materialize_quality_v4_winner_export(
        tmp_path,
        source=source,
        attempt=attempt,
    )
    assert exported["dataset_quality_v4_validation"]["passed"]
    assert _validate_quality_v4_validation(exported["dataset_quality_v4_validation"])
    task_audit = _audit_quality_v4_full_exports(
        tmp_path,
        [{"request": {"episode_id": "p0_grasp-episode"}}],
        threshold_identity={"payload_sha256": thresholds["thresholds_sha256"]},
    )
    assert task_audit["enabled"]
    assert task_audit["audited_count"] == 1

    summaries = {
        task_id: {
            "quality_v4_full_exports": {
                "enabled": True,
                "audited_count": ACCEPTED_PER_TASK,
                "thresholds_sha256": task_audit["thresholds_sha256"],
                "orientation_contract_sha256": task_audit[
                    "orientation_contract_sha256"
                ],
                "gate_sha256": ["f" * 64] * ACCEPTED_PER_TASK,
            }
        }
        for task_id in EXACT_TASKS
    }
    readiness = quality_v4_release_readiness(summaries)
    assert readiness["task_count"] == 14
    assert readiness["full_export_gate_count"] == 14 * ACCEPTED_PER_TASK
    assert readiness["release_ready"]

    drifted = copy.deepcopy(summaries)
    drifted[EXACT_TASKS[0]]["quality_v4_full_exports"]["thresholds_sha256"] = "0" * 64
    with pytest.raises(ReleaseAuditError, match="mixes threshold"):
        quality_v4_release_readiness(drifted)


def test_qv4_optimal_audit_rejects_missing_directory_and_tampered_gate(
    tmp_path: Path,
) -> None:
    thresholds = _frozen_thresholds(build_provisional_thresholds())
    identity = {"payload_sha256": thresholds["thresholds_sha256"]}
    winners = [{"request": {"episode_id": "p0_grasp-episode"}}]
    with pytest.raises(ValueError, match="full-export directory is missing"):
        _audit_quality_v4_full_exports(
            tmp_path,
            winners,
            threshold_identity=identity,
        )
    source = _source("p0_grasp", thresholds)
    attempt = build_quality_v4_attempt(source)
    exported = materialize_quality_v4_winner_export(
        tmp_path,
        source=source,
        attempt=attempt,
    )
    gate_path = tmp_path / exported["full_export_gate_path"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["passed"] = False
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="does not recompute"):
        _audit_quality_v4_full_exports(
            tmp_path,
            winners,
            threshold_identity=identity,
        )


def test_qv4_shard_merge_independent_audit_end_to_end_fixture(
    tmp_path: Path,
) -> None:
    thresholds = _frozen_thresholds(build_provisional_thresholds())
    threshold_bytes = (json.dumps(thresholds, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    threshold_file_sha256 = hashlib.sha256(threshold_bytes).hexdigest()
    receipt = {
        "schema_version": "se3wam-quality-v4-owner-review-receipt-v0.1",
        "threshold_schema_version": thresholds["schema_version"],
        "threshold_file_sha256": threshold_file_sha256,
        "threshold_payload_sha256": thresholds["thresholds_sha256"],
        "owner_review": thresholds["owner_review"],
    }
    receipt["payload_sha256"] = stable_sha256(receipt)
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    identity = {
        "schema_version": thresholds["schema_version"],
        "file_sha256": threshold_file_sha256,
        "payload_sha256": thresholds["thresholds_sha256"],
        "owner_review_receipt": {
            "relative_path": "provenance/quality_v4/owner_review_receipt.json",
            "file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "payload_sha256": receipt["payload_sha256"],
        },
    }
    shard = tmp_path / "shard-00"
    shard.mkdir()
    (shard / "quality_v4_thresholds.json").write_bytes(threshold_bytes)
    receipt_path = shard / identity["owner_review_receipt"]["relative_path"]
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt_bytes)
    source = _source("p0_grasp", thresholds)
    attempt = build_quality_v4_attempt(source)
    write_quality_v4_attempt(shard, attempt)
    write_quality_v4_lightweight_source(
        shard,
        source=source,
        recorded_attempt=attempt,
    )
    materialize_quality_v4_winner_export(
        shard,
        source=source,
        attempt=attempt,
    )
    episode_id = str(attempt["episode_id"])
    state = {"quality_v4_threshold_identity": identity}
    assert _validate_quality_v4_shard_contract(shard, state) == identity
    _validate_quality_v4_artifacts(
        shard,
        task="p0_grasp",
        threshold_identity=identity,
        reset_episode_ids=[episode_id],
        winner_episode_ids=[episode_id],
    )

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "quality_v4_thresholds.json").write_bytes(threshold_bytes)
    merged_receipt = merged / identity["owner_review_receipt"]["relative_path"]
    merged_receipt.parent.mkdir(parents=True)
    merged_receipt.write_bytes(receipt_bytes)
    _copy_quality_v4_artifacts(
        shard_dirs=[shard],
        output=merged,
        reset_episode_ids=[episode_id],
        winner_episode_ids=[episode_id],
    )
    _, audited_identity = _audit_quality_v4_contract(
        merged,
        expected_threshold_file_sha256=identity["file_sha256"],
        expected_threshold_payload_sha256=identity["payload_sha256"],
        expected_receipt_file_sha256=identity["owner_review_receipt"]["file_sha256"],
        expected_receipt_payload_sha256=identity["owner_review_receipt"][
            "payload_sha256"
        ],
    )
    assert audited_identity == identity
    _audit_quality_v4_source_inventory(
        merged,
        task="p0_grasp",
        reset_episode_ids=[episode_id],
        winner_episode_ids=[episode_id],
        threshold_identity=identity,
    )
    summary = _audit_quality_v4_full_exports(
        merged,
        [{"request": {"episode_id": episode_id}}],
        threshold_identity=identity,
    )
    assert summary["audited_count"] == 1
    assert summary["thresholds_sha256"] == thresholds["thresholds_sha256"]
