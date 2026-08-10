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
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from se3_wam.benchmark.trajectory_quality import (
    evaluate_quality_v2_gate,
    trajectory_quality_v2_from_observations,
)

from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _audit_attempt_tape,
    _audit_quality_v2_calibration_receipt_artifact,
    _audit_render_parity_skip,
    _audit_t5_replan_causal_history,
    _render_parity_skip_events,
)
from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _eligible as _audit_eligible,
)
from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _payload_sha256 as _audit_payload_sha256,
)
from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _planner_metric_relations as _audit_planner_metric_relations,
)
from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _quality_v2_dominance_contract as _audit_quality_v2_dominance_contract,
)
from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _quality_v2_metric_thresholds as _audit_quality_v2_metric_thresholds,
)
from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _selected as _audit_selected,
)
from examples.embodiment.export_dynamic_benchmark_optimal_trajectories import (
    CANDIDATE_SCHEMA,
    PLANNER_PARETO_SELECTION_MODE,
    _ArmedResetReplayEnv,
    _budget_sequence,
    _candidate_identity,
    _copy_provenance_file,
    _eligible,
    _file_boundary,
    _progress_payload,
    _quality_score,
    _quality_v2_dominance_contract,
    _quality_v2_metric_thresholds,
    _recover_partial_output,
    _render_parity_skip,
    _restore_candidate_start,
    _select_winner,
    _t5_replan_causal_evidence,
    _task_quality_from_infos,
    _validate_candidate_manifest,
    _validate_quality_v2_calibration_receipt_artifact,
    _write_attempt_tape,
)
from examples.embodiment.merge_optimal_export_shards import (
    _expected_shard_indices,
    _kept_recovery_events,
    _load_shard_receipts,
    _validate_shard_records,
)
from examples.embodiment.merge_optimal_export_shards import (
    main as _merge_shards_main,
)


def _record(candidate_index: int, *, value: float = 1.0) -> dict:
    replay = {"passed": True, "outcomes_exact": True}
    record = {
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "replay_validation": replay,
        "quality_v2_gate": {"passed": True},
        "trajectory_completion": 1.0,
        "return": value,
        "control_steps": 4,
        "action_l2_sum": 2.0,
        "candidate_index": candidate_index,
    }
    record["quality_score"] = list(_quality_score(record))
    record["eligible"] = _eligible(record)
    return record


_DYNAMIC_QUALITY_METRICS = (
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
_QUALITY_V2_THRESHOLD_SHA256 = "c" * 64
_CALIBRATION_TASKS = (
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


def _quality_v2_check(phase: str, metric: str) -> dict:
    family_by_metric = {
        "action.action_second_difference_l2_mean_per_transition": "action_l2",
        "action.action_max_second_difference_l2": "action_l2",
        "action.action_total_variation_l2_mean_per_transition": "action_l2",
        "eef_motion.eef_translation_path_length_m": "translation_path_m",
        "eef_motion.eef_rotation_path_length_rad": "rotation_or_orientation_rad",
        "eef_motion.eef_angular_jerk_max_rad_s3": "angular_jerk_rad_s3",
        "eef_motion.eef_linear_jerk_max_m_s3": "linear_jerk_m_s3",
        "eef_motion.eef_angular_jerk_rms_rad_s3": "angular_jerk_rad_s3",
        "eef_motion.eef_linear_jerk_rms_m_s3": "linear_jerk_m_s3",
        "orientation_reference.orientation_reference_error_max_rad": (
            "rotation_or_orientation_rad"
        ),
        "approach_axis.approach_axis_error_max_rad": ("rotation_or_orientation_rad"),
        "jaw_axis.jaw_axis_error_max_rad": "rotation_or_orientation_rad",
    }
    return {
        "phase": phase,
        "metric": metric,
        "max": 2.0,
        "direction": "minimize",
        "paired_comparison_family": family_by_metric[metric],
        "paired_nonworse_absolute_tolerance": 0.01,
        "paired_nonworse_relative_tolerance": 0.0,
        "paired_strict_improvement_absolute": 0.02,
        "paired_strict_improvement_relative": 0.0,
    }


def _quality_v2_thresholds(*, include_jaw: bool = False) -> dict:
    checks = [
        _quality_v2_check("full_episode", metric) for metric in _DYNAMIC_QUALITY_METRICS
    ]
    checks.append(
        _quality_v2_check(
            "full_episode",
            "orientation_reference.orientation_reference_error_max_rad",
        )
    )
    if include_jaw:
        checks.append(_quality_v2_check("post_hold", "jaw_axis.jaw_axis_error_max_rad"))
    return {
        "schema_version": "se3-wam-trajectory-quality-v2-thresholds-v0.3",
        "calibration_status": "frozen",
        "formal_freeze_eligible": True,
        "minimum_attempted_episodes": 20,
        "minimum_successful_episodes": 8,
        "calibration_wave_receipt": {
            "schema_version": "rld2-qa-planner-calibration-wave-receipt-v0.1",
            "binding_status": "bound",
            "scientific_partition": "metric_calibration",
            "transport_split": "validation",
            "task_count": 14,
            "episodes_per_task": 20,
            "total_reset_count": 280,
            "sha256": "e" * 64,
            "file_sha256": "e" * 64,
            "payload_sha256": "e" * 64,
            "relative_path": "provenance/calibration_wave/wave_receipt.json",
        },
        "tasks": {
            "t2_trans": {
                "checks": checks,
                "orientation_mode": "reset_frozen_full_orientation",
                "jaw_axis_mode": (
                    "object_local_x_mod_pi" if include_jaw else "unconstrained"
                ),
                "provenance": {
                    "formal_freeze_eligible": True,
                    "attempted_episode_count": 20,
                    "successful_episode_count": 8,
                },
            }
        },
    }


def _calibration_receipt_fixture(tmp_path: Path) -> tuple[dict, Path, str]:
    receipt_tasks = []
    binding_tasks = []
    for ordinal, task_id in enumerate(_CALIBRATION_TASKS):
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
            "task_id": task_id,
            "reset_count": 20,
            "task_quality_schema_version": "db0-episode-task-quality-v2",
            "reset_manifest_relative_path": f"tasks/{task_id}/reset_manifest.jsonl",
            "evaluation_relative_path": f"tasks/{task_id}/evaluation.json",
            **hashes,
        }
        receipt_tasks.append(receipt_task)
        binding_tasks.append(
            {key: value for key, value in receipt_task.items() if key != "reset_count"}
            | {"reset_identity_count": 20}
        )
    receipt = {
        "schema_version": "rld2-qa-planner-calibration-wave-receipt-v0.1",
        "scientific_partition": "metric_calibration",
        "transport_split": "validation",
        "manifest_seed": 20261350,
        "task_count": 14,
        "episodes_per_task": 20,
        "total_reset_count": 280,
        "task_order": list(_CALIBRATION_TASKS),
        "wave_contract_sha256": "a" * 64,
        "predeclaration_receipt_sha256": "b" * 64,
        "source_identity": {"wave_id": "synthetic-wave"},
        "disjointness": {"verified": True},
        "tasks": receipt_tasks,
    }
    receipt_bytes = json.dumps(
        receipt,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    receipt_path = tmp_path / "source_wave_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    thresholds = _quality_v2_thresholds()
    thresholds["calibration_wave_receipt"] = {
        "binding_status": "bound",
        "schema_version": receipt["schema_version"],
        "scientific_partition": receipt["scientific_partition"],
        "transport_split": receipt["transport_split"],
        "manifest_seed": receipt["manifest_seed"],
        "task_count": receipt["task_count"],
        "episodes_per_task": receipt["episodes_per_task"],
        "total_reset_count": receipt["total_reset_count"],
        "task_order": receipt["task_order"],
        "wave_contract_sha256": receipt["wave_contract_sha256"],
        "predeclaration_receipt_sha256": receipt["predeclaration_receipt_sha256"],
        "source_identity": receipt["source_identity"],
        "disjointness": receipt["disjointness"],
        "tasks": binding_tasks,
        "relative_path": "provenance/calibration_wave/wave_receipt.json",
        "sha256": receipt_sha256,
        "file_sha256": receipt_sha256,
        "payload_sha256": receipt_sha256,
    }
    return thresholds, receipt_path, receipt_sha256


def _planner_dominance_contract() -> dict:
    def metric(direction: str, resolution: float, *, steps: bool = False) -> dict:
        return {
            "direction": direction,
            "max_observed_replay_drift": 0.0,
            "scientific_resolution": resolution,
            "numeric_floor": 0.0 if steps else 1.0e-6,
        }

    component = {
        "name": "utility",
        "direction": "maximize",
        "unit": "score",
        "scientific_resolution": 0.01,
        "reducer": "maximum",
        "source": "unit-test",
        "description": "Synthetic task utility used by selector tests.",
    }
    return {
        "task": "t2_trans",
        "backend_id": "unit-test-backend",
        "quality_schema": {
            "schema_version": "unit-test-task-quality-v1",
            "task_id": "t2_trans",
            "task_config_sha256": "a" * 64,
            "components": [component],
            "schema_sha256": "b" * 64,
        },
        "metrics": {
            "trajectory_completion": metric("max", 1.0e-6),
            "task_quality": {"utility": metric("max", 0.01)},
            "completion_time_s": metric("min", 0.002),
            "control_steps": metric("min", 1.0, steps=True),
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
            "task_quality.utility",
            "completion_time_s",
            "control_steps",
            "action_l2_sum",
        ],
    }


def _set_nested(root: dict, dotted: str, value: float) -> None:
    current = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _pareto_record(
    candidate_index: int,
    *,
    quality_values: dict[str, float] | None = None,
    utility: float = 1.0,
) -> dict:
    values = dict.fromkeys(_DYNAMIC_QUALITY_METRICS, 0.5)
    if quality_values is not None:
        values.update(quality_values)
    episode_id = "same-reset"
    task_quality = {
        "schema_version": "unit-test-task-quality-v1",
        "episode_id": episode_id,
        "task_id": "t2_trans",
        "evaluator_backend_id": "unit-test-backend",
        "schema_sha256": "b" * 64,
        "physics_sample_count": 10,
        "terminal": True,
        "components": {
            "utility": {
                "value": utility,
                "direction": "maximize",
                "unit": "score",
                "scientific_resolution": 0.01,
                "reducer": "maximum",
            }
        },
    }
    task_quality["summary_sha256"] = _audit_payload_sha256(task_quality)
    quality_v2: dict = {
        "schema_version": "se3-wam-trajectory-quality-v2",
        "phases": {},
    }
    for metric, value in values.items():
        _set_nested(quality_v2, metric, value)
    _set_nested(
        quality_v2,
        "orientation_reference.orientation_reference_error_max_rad",
        0.1,
    )
    thresholds = _quality_v2_thresholds()
    quality_gate = evaluate_quality_v2_gate(
        quality_v2,
        thresholds,
        task_id="t2_trans",
    )
    quality_gate["contract_sha256"] = _QUALITY_V2_THRESHOLD_SHA256
    record = {
        "schema_version": "rlinf-dynamic-benchmark-optimal-attempt-v0.3",
        "task_id": "t2_trans",
        "episode_id": episode_id,
        "candidate_id": "planner"
        if candidate_index == 0
        else f"policy-{candidate_index}",
        "candidate_index": candidate_index,
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "replay_validation": {"passed": True, "outcomes_exact": True},
        "trajectory_completion": 1.0,
        "task_quality": task_quality,
        "quality_v2": quality_v2,
        "quality_v2_sha256": _audit_payload_sha256(quality_v2),
        "quality_v2_gate": quality_gate,
        "completion_time_s": 1.0,
        "return": 1.0,
        "control_steps": 500,
        "action_l2_sum": 2.0,
        "issued_equals_applied": True,
        "t5_replan_causal_timing_passed": None,
        "impact_end_to_first_qualifying_applied_correction_s": None,
    }
    record["quality_score"] = list(_quality_score(record))
    record["eligible"] = _eligible(record)
    return record


def test_budget_sequence_doubles_to_frozen_maximum() -> None:
    assert _budget_sequence(8, 32) == (8, 16, 32)
    assert _budget_sequence(12, 32) == (12, 24, 32)
    with pytest.raises(ValueError, match="candidate budgets"):
        _budget_sequence(16, 8)


def test_missing_vector_task_quality_remains_an_absent_summary() -> None:
    assert _task_quality_from_infos({"task_quality": [None]}) is None
    with pytest.raises(ValueError, match="non-empty mapping"):
        _task_quality_from_infos({"task_quality": [{}]})


def test_candidate_manifest_is_commit_bound_and_resolves_relative_paths(
    tmp_path: Path,
) -> None:
    candidates = [{"candidate_id": "planner", "kind": "planner"}]
    candidates.extend(
        {
            "candidate_id": f"policy-{index}",
            "kind": "policy",
            "policy_path": f"policies/{index}.pt",
            "policy_sha256": f"{index:x}" * 64,
            "stochastic": True,
            "exploration_seed_offset": index,
        }
        for index in range(1, 8)
    )
    payload = {
        "schema_version": CANDIDATE_SCHEMA,
        "task": "t2_trans",
        "rlinf_commit": "a" * 40,
        "benchmark_commit": "b" * 40,
        "candidates": candidates,
    }

    task, specs = _validate_candidate_manifest(
        payload,
        manifest_path=tmp_path / "candidates.json",
        rlinf_commit="a" * 40,
        benchmark_commit="b" * 40,
        max_k=8,
    )

    assert task == "t2_trans"
    assert specs[1].policy_path == (tmp_path / "policies/1.pt").resolve()
    assert _candidate_identity(specs[1])["policy_path"] == str(
        (tmp_path / "policies/1.pt").resolve()
    )
    with pytest.raises(ValueError, match="benchmark commit"):
        _validate_candidate_manifest(
            payload,
            manifest_path=tmp_path / "candidates.json",
            rlinf_commit="a" * 40,
            benchmark_commit="c" * 40,
            max_k=8,
        )
    payload["candidates"] = [*payload["candidates"][1:], payload["candidates"][0]]
    with pytest.raises(ValueError, match="planner at index zero"):
        _validate_candidate_manifest(
            payload,
            manifest_path=tmp_path / "candidates.json",
            rlinf_commit="a" * 40,
            benchmark_commit="b" * 40,
            max_k=8,
        )


def test_winner_selection_is_quality_first_then_stable_candidate_index() -> None:
    first = _record(0)
    same_quality_later = _record(1)
    better_return = _record(2, value=2.0)

    assert _select_winner([same_quality_later, first]) is first
    assert _select_winner([first, better_return]) is better_return
    better_return["safety_failure"] = True
    better_return["eligible"] = _eligible(better_return)
    assert _select_winner([first, better_return]) is first


def _quality_v2_dominance_pair() -> tuple[dict, dict]:
    thresholds = _quality_v2_thresholds()
    exporter_contract = _quality_v2_dominance_contract(
        thresholds,
        task="t2_trans",
        thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
    )
    auditor_contract = _audit_quality_v2_dominance_contract(
        thresholds,
        task="t2_trans",
        thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
    )
    assert auditor_contract == exporter_contract
    return exporter_contract, auditor_contract


@pytest.mark.parametrize("metric_name", _DYNAMIC_QUALITY_METRICS)
def test_planner_pareto_blocks_regression_on_every_frozen_dynamic_metric(
    metric_name: str,
) -> None:
    quality_v2_contract, audit_quality_v2_contract = _quality_v2_dominance_pair()
    dominance = _planner_dominance_contract()
    planner = _pareto_record(0)
    policy = _pareto_record(
        1,
        utility=1.1,
        quality_values={metric_name: 0.55},
    )

    assert (
        _select_winner(
            [planner, policy],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=quality_v2_contract,
        )
        is planner
    )
    assert (
        _audit_selected(
            [planner, policy],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=audit_quality_v2_contract,
        )
        is planner
    )


def test_planner_pareto_accepts_quality_v2_strict_improvement_and_reports_it() -> None:
    quality_v2_contract, audit_quality_v2_contract = _quality_v2_dominance_pair()
    dominance = _planner_dominance_contract()
    planner = _pareto_record(0)
    metric_name = "eef_motion.eef_translation_path_length_m"
    policy = _pareto_record(1, quality_values={metric_name: 0.4})

    assert (
        _select_winner(
            [planner, policy],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=quality_v2_contract,
        )
        is policy
    )
    assert (
        _audit_selected(
            [planner, policy],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=audit_quality_v2_contract,
        )
        is policy
    )
    relations = _audit_planner_metric_relations(
        policy,
        planner,
        dominance,
        audit_quality_v2_contract,
    )
    assert relations[f"quality_v2.full_episode.{metric_name}"] == "strictly_improved"


def test_planner_pareto_exact_union_tie_keeps_planner() -> None:
    quality_v2_contract, audit_quality_v2_contract = _quality_v2_dominance_pair()
    dominance = _planner_dominance_contract()
    planner = _pareto_record(0)
    policy = _pareto_record(1)

    assert (
        _select_winner(
            [planner, policy],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=quality_v2_contract,
        )
        is planner
    )
    assert (
        _audit_selected(
            [planner, policy],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=audit_quality_v2_contract,
        )
        is planner
    )


def test_planner_pareto_quality_v2_metric_and_threshold_identity_fail_closed() -> None:
    quality_v2_contract, audit_quality_v2_contract = _quality_v2_dominance_pair()
    dominance = _planner_dominance_contract()
    planner = _pareto_record(0)
    missing_metric = copy.deepcopy(_pareto_record(1, utility=1.1))
    del missing_metric["quality_v2"]["eef_motion"]["eef_rotation_path_length_rad"]
    missing_metric["quality_v2_sha256"] = _audit_payload_sha256(
        missing_metric["quality_v2"]
    )
    with pytest.raises(ValueError, match="mapping gap"):
        _select_winner(
            [planner, missing_metric],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=quality_v2_contract,
        )
    with pytest.raises(ValueError, match="mapping gap"):
        _audit_selected(
            [planner, missing_metric],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=audit_quality_v2_contract,
        )

    missing_orientation = copy.deepcopy(_pareto_record(1, utility=1.1))
    del missing_orientation["quality_v2"]["orientation_reference"]
    missing_orientation["quality_v2_sha256"] = _audit_payload_sha256(
        missing_orientation["quality_v2"]
    )
    with pytest.raises(ValueError, match="mapping gap"):
        _select_winner(
            [planner, missing_orientation],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=quality_v2_contract,
        )
    with pytest.raises(ValueError, match="mapping gap"):
        _audit_selected(
            [planner, missing_orientation],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=audit_quality_v2_contract,
        )

    wrong_threshold = copy.deepcopy(_pareto_record(1, utility=1.1))
    wrong_threshold["quality_v2_gate"]["contract_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="gate identity"):
        _select_winner(
            [planner, wrong_threshold],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=quality_v2_contract,
        )
    with pytest.raises(ValueError, match="gate identity"):
        _audit_selected(
            [planner, wrong_threshold],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=audit_quality_v2_contract,
        )


def test_planner_pareto_quality_v2_absolute_gate_rejects_before_dominance() -> None:
    quality_v2_contract, audit_quality_v2_contract = _quality_v2_dominance_pair()
    dominance = _planner_dominance_contract()
    planner = _pareto_record(0)
    policy = _pareto_record(
        1,
        utility=1.1,
        quality_values={"action.action_max_second_difference_l2": 2.1},
    )
    assert policy["quality_v2_gate"]["passed"] is False
    assert (
        _select_winner(
            [planner, policy],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=quality_v2_contract,
        )
        is planner
    )
    assert (
        _audit_selected(
            [planner, policy],
            selection_mode=PLANNER_PARETO_SELECTION_MODE,
            planner_dominance=dominance,
            quality_v2_dominance=audit_quality_v2_contract,
        )
        is planner
    )


def test_quality_v2_dominance_contract_rejects_missing_metadata_and_wrong_direction() -> (
    None
):
    thresholds = _quality_v2_thresholds()
    del thresholds["tasks"]["t2_trans"]["checks"][0][
        "paired_nonworse_absolute_tolerance"
    ]
    for builder in (
        _quality_v2_dominance_contract,
        _audit_quality_v2_dominance_contract,
    ):
        with pytest.raises(
            ValueError, match="missing frozen paired-comparison metadata"
        ):
            builder(
                thresholds,
                task="t2_trans",
                thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
            )

    thresholds = _quality_v2_thresholds()
    thresholds["tasks"]["t2_trans"]["checks"][0]["direction"] = "maximize"
    for builder in (
        _quality_v2_dominance_contract,
        _audit_quality_v2_dominance_contract,
    ):
        with pytest.raises(ValueError, match="direction='minimize'"):
            builder(
                thresholds,
                task="t2_trans",
                thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
            )


@pytest.mark.parametrize(
    "builder",
    (_quality_v2_dominance_contract, _audit_quality_v2_dominance_contract),
)
def test_quality_v2_exact_task_derived_inventories_accept_valid_ten_and_eleven(
    builder,
) -> None:
    ten = builder(
        _quality_v2_thresholds(),
        task="t2_trans",
        thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
    )
    assert len(ten["metrics"]) == 10
    assert ten["orientation_mode"] == "reset_frozen_full_orientation"
    assert ten["jaw_axis_mode"] == "unconstrained"
    by_identity = {(spec["phase"], spec["metric"]): spec for spec in ten["metrics"]}
    assert (
        by_identity[("full_episode", "eef_motion.eef_angular_jerk_rms_rad_s3")]["key"]
        == "eef_angular_jerk_rms"
    )
    assert (
        by_identity[("full_episode", "eef_motion.eef_angular_jerk_rms_rad_s3")]["group"]
        == "eef_motion"
    )
    assert (
        by_identity[
            (
                "full_episode",
                "orientation_reference.orientation_reference_error_max_rad",
            )
        ]["group"]
        == "grasp_geometry"
    )

    eleven = builder(
        _quality_v2_thresholds(include_jaw=True),
        task="t2_trans",
        thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
    )
    assert len(eleven["metrics"]) == 11
    assert eleven["jaw_axis_mode"] == "object_local_x_mod_pi"
    assert any(
        spec["phase"] == "post_hold"
        and spec["metric"] == "jaw_axis.jaw_axis_error_max_rad"
        and spec["key"] == "jaw_angle"
        and spec["group"] == "grasp_geometry"
        for spec in eleven["metrics"]
    )


@pytest.mark.parametrize(
    "builder",
    (_quality_v2_dominance_contract, _audit_quality_v2_dominance_contract),
)
def test_quality_v2_orientation_mode_selects_exact_approach_identity(builder) -> None:
    thresholds = _quality_v2_thresholds()
    task_contract = thresholds["tasks"]["t2_trans"]
    task_contract["orientation_mode"] = "world_down_tool_axis"
    task_contract["checks"] = [
        check
        for check in task_contract["checks"]
        if check["metric"]
        != "orientation_reference.orientation_reference_error_max_rad"
    ]
    task_contract["checks"].append(
        _quality_v2_check(
            "acquisition_window",
            "approach_axis.approach_axis_error_max_rad",
        )
    )

    contract = builder(
        thresholds,
        task="t2_trans",
        thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
    )

    assert len(contract["metrics"]) == 10
    assert any(
        spec["phase"] == "acquisition_window"
        and spec["metric"] == "approach_axis.approach_axis_error_max_rad"
        and spec["key"] == "approach_verticality"
        and spec["group"] == "grasp_geometry"
        for spec in contract["metrics"]
    )


@pytest.mark.parametrize(
    "builder",
    (_quality_v2_dominance_contract, _audit_quality_v2_dominance_contract),
)
def test_quality_v2_exact_inventory_rejects_missing_rms_wrong_phase_and_extra(
    builder,
) -> None:
    missing_rms = _quality_v2_thresholds()
    checks = missing_rms["tasks"]["t2_trans"]["checks"]
    replacement = next(
        check
        for check in checks
        if check["metric"] == "eef_motion.eef_angular_jerk_rms_rad_s3"
    )
    replacement["metric"] = "eef_motion.eef_angular_jerk_peak_rad_s3"
    with pytest.raises(ValueError, match="inventory mismatch"):
        builder(
            missing_rms,
            task="t2_trans",
            thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
        )

    wrong_phase = _quality_v2_thresholds()
    next(
        check
        for check in wrong_phase["tasks"]["t2_trans"]["checks"]
        if check["metric"] == "eef_motion.eef_linear_jerk_rms_m_s3"
    )["phase"] = "post_hold"
    with pytest.raises(ValueError, match="inventory mismatch"):
        builder(
            wrong_phase,
            task="t2_trans",
            thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
        )

    extra = _quality_v2_thresholds()
    extra["tasks"]["t2_trans"]["checks"].append(
        {
            **extra["tasks"]["t2_trans"]["checks"][0],
            "metric": "action.unexpected_extra_metric",
        }
    )
    with pytest.raises(ValueError, match="exactly 10"):
        builder(
            extra,
            task="t2_trans",
            thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
        )


@pytest.mark.parametrize(
    "builder",
    (_quality_v2_dominance_contract, _audit_quality_v2_dominance_contract),
)
def test_quality_v2_exact_inventory_rejects_wrong_metric_group(builder) -> None:
    thresholds = _quality_v2_thresholds()
    next(
        check
        for check in thresholds["tasks"]["t2_trans"]["checks"]
        if check["metric"] == "eef_motion.eef_linear_jerk_rms_m_s3"
    )["paired_comparison_family"] = "angular_jerk_rad_s3"

    with pytest.raises(ValueError, match="comparison family"):
        builder(
            thresholds,
            task="t2_trans",
            thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
        )


def test_quality_v2_provisional_contract_is_validation_only() -> None:
    thresholds = _quality_v2_thresholds()
    thresholds["formal_freeze_eligible"] = False
    for builder in (
        _quality_v2_dominance_contract,
        _audit_quality_v2_dominance_contract,
    ):
        with pytest.raises(ValueError, match="not eligible for formal freeze"):
            builder(
                thresholds,
                task="t2_trans",
                thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
            )
        validation_contract = builder(
            thresholds,
            task="t2_trans",
            thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
            require_formal_freeze=False,
        )
        assert validation_contract["formal_freeze_eligible"] is False


def test_quality_v2_formal_freeze_requires_bound_wave_and_task_counts() -> None:
    for builder in (
        _quality_v2_dominance_contract,
        _audit_quality_v2_dominance_contract,
    ):
        unfrozen = _quality_v2_thresholds()
        unfrozen["calibration_status"] = "provisional"
        with pytest.raises(ValueError, match="invalid formal calibration provenance"):
            builder(
                unfrozen,
                task="t2_trans",
                thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
            )

        under_sampled = _quality_v2_thresholds()
        under_sampled["tasks"]["t2_trans"]["provenance"]["attempted_episode_count"] = 19
        with pytest.raises(
            ValueError, match="task .* invalid formal calibration provenance"
        ):
            builder(
                under_sampled,
                task="t2_trans",
                thresholds_sha256=_QUALITY_V2_THRESHOLD_SHA256,
            )


def test_quality_v2_calibration_receipt_artifact_round_trip(tmp_path: Path) -> None:
    thresholds, receipt_path, receipt_sha256 = _calibration_receipt_fixture(tmp_path)

    provenance = _validate_quality_v2_calibration_receipt_artifact(
        thresholds,
        receipt_path,
        expected_sha256=receipt_sha256,
    )
    assert provenance.relative_path == ("provenance/calibration_wave/wave_receipt.json")
    assert provenance.sha256 == receipt_sha256

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _copy_provenance_file(
        dataset_root,
        provenance.source_path,
        provenance.relative_path,
        provenance.sha256,
    )
    assert _audit_quality_v2_calibration_receipt_artifact(
        dataset_root,
        thresholds,
    ) == {
        "relative_path": provenance.relative_path,
        "file_sha256": receipt_sha256,
        "payload_sha256": receipt_sha256,
    }


def test_quality_v2_calibration_receipt_artifact_is_required(tmp_path: Path) -> None:
    thresholds, _, receipt_sha256 = _calibration_receipt_fixture(tmp_path)
    missing = tmp_path / "missing_wave_receipt.json"

    with pytest.raises(ValueError, match="missing or symlinked"):
        _validate_quality_v2_calibration_receipt_artifact(
            thresholds,
            missing,
            expected_sha256=receipt_sha256,
        )
    with pytest.raises(ValueError, match="missing or symlinked"):
        _audit_quality_v2_calibration_receipt_artifact(
            tmp_path / "missing-dataset",
            thresholds,
        )


def test_quality_v2_calibration_receipt_rejects_threshold_self_report(
    tmp_path: Path,
) -> None:
    thresholds, _, _ = _calibration_receipt_fixture(tmp_path)
    forged_bytes = b"{}"
    forged_sha256 = hashlib.sha256(forged_bytes).hexdigest()
    for key in ("sha256", "file_sha256", "payload_sha256"):
        thresholds["calibration_wave_receipt"][key] = forged_sha256
    forged_source = tmp_path / "forged_wave_receipt.json"
    forged_source.write_bytes(forged_bytes)

    with pytest.raises(ValueError, match="schema_version mismatch"):
        _validate_quality_v2_calibration_receipt_artifact(
            thresholds,
            forged_source,
            expected_sha256=forged_sha256,
        )
    dataset_root = tmp_path / "dataset"
    dataset_receipt = (
        dataset_root / thresholds["calibration_wave_receipt"]["relative_path"]
    )
    dataset_receipt.parent.mkdir(parents=True)
    dataset_receipt.write_bytes(forged_bytes)
    with pytest.raises(ValueError, match="schema_version mismatch"):
        _audit_quality_v2_calibration_receipt_artifact(dataset_root, thresholds)


def test_quality_v2_calibration_receipt_rejects_tampered_bytes(
    tmp_path: Path,
) -> None:
    thresholds, receipt_path, receipt_sha256 = _calibration_receipt_fixture(tmp_path)
    tampered = tmp_path / "tampered_wave_receipt.json"
    tampered.write_bytes(receipt_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        _validate_quality_v2_calibration_receipt_artifact(
            thresholds,
            tampered,
            expected_sha256=receipt_sha256,
        )
    dataset_root = tmp_path / "dataset"
    dataset_receipt = (
        dataset_root / thresholds["calibration_wave_receipt"]["relative_path"]
    )
    dataset_receipt.parent.mkdir(parents=True)
    dataset_receipt.write_bytes(tampered.read_bytes())
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        _audit_quality_v2_calibration_receipt_artifact(dataset_root, thresholds)


def test_quality_v2_calibration_receipt_rejects_path_escape(tmp_path: Path) -> None:
    thresholds, receipt_path, receipt_sha256 = _calibration_receipt_fixture(tmp_path)
    thresholds["calibration_wave_receipt"]["relative_path"] = "../wave_receipt.json"

    with pytest.raises(ValueError, match="unsafe"):
        _validate_quality_v2_calibration_receipt_artifact(
            thresholds,
            receipt_path,
            expected_sha256=receipt_sha256,
        )
    with pytest.raises(ValueError, match="unsafe"):
        _audit_quality_v2_calibration_receipt_artifact(tmp_path, thresholds)


def test_quality_v2_paired_tolerance_and_strict_margin_use_planner_scale() -> None:
    spec = {
        "paired_nonworse_absolute_tolerance": 0.5,
        "paired_nonworse_relative_tolerance": 0.02,
        "paired_strict_improvement_absolute": 2.0,
        "paired_strict_improvement_relative": 0.05,
    }
    assert _quality_v2_metric_thresholds(100.0, spec) == (2.0, 5.0)
    assert _audit_quality_v2_metric_thresholds(100.0, spec) == (2.0, 5.0)


def test_render_parity_skip_binds_the_selected_attempt_and_recovery_event() -> None:
    selected = {
        **_record(3),
        "candidate_id": "policy-3",
        "attempt_tape": "lightweight/episode-1/candidate-03.npz",
        "attempt_tape_sha256": "a" * 64,
        "action_sha256": "b" * 64,
    }
    error = RuntimeError("winner render parity failed for return")
    result = {
        "reset_index": 7,
        "episode_id": "episode-1",
        "accepted": False,
        "winner_candidate_id": None,
        "winner_candidate_index": None,
        "render_parity_skip": _render_parity_skip(selected, error),
    }
    events = _render_parity_skip_events(
        ["render_parity_skip:reset:7:episode-1:winner render parity failed for return"]
    )

    assert _audit_render_parity_skip(result, selected, events[7]) == "structured-v0.1"
    result["render_parity_skip"]["candidate_id"] = "policy-tampered"
    with pytest.raises(ValueError, match="selected attempt"):
        _audit_render_parity_skip(result, selected, events[7])


def test_legacy_render_parity_skip_requires_one_recognized_recovery_event() -> None:
    selected = {
        **_record(0),
        "candidate_id": "planner",
        "attempt_tape": "lightweight/episode-1/candidate-00.npz",
        "attempt_tape_sha256": "a" * 64,
        "action_sha256": "b" * 64,
    }
    result = {"reset_index": 1, "episode_id": "episode-1", "accepted": False}
    events = _render_parity_skip_events(
        [
            "resume.recovery-1",
            "render_parity_skip:reset:1:episode-1:canonical replay contract mismatch",
        ]
    )

    assert _audit_render_parity_skip(result, selected, events[1]) == "legacy-v0.1"
    with pytest.raises(ValueError, match="no matching recovery event"):
        _audit_render_parity_skip(result, selected, None)
    with pytest.raises(ValueError, match="invalid"):
        _render_parity_skip_events(
            ["render_parity_skip:reset:1:episode-1:unrecognized failure"]
        )


def test_shard_merge_keeps_only_render_skip_events_in_the_sealed_prefix() -> None:
    events = [
        "shard-00.recovery-1",
        "render_parity_skip:reset:3:episode-3:winner render parity failed for return",
        "render_parity_skip:reset:9:episode-9:winner render parity failed for return",
    ]

    assert _kept_recovery_events(events, max_reset=5) == events[:2]
    with pytest.raises(ValueError, match="malformed"):
        _kept_recovery_events(["render_parity_skip:bad"], max_reset=5)


def _shard_receipt(*, shard_index: int = 0, shard_count: int = 1) -> dict:
    return {
        "schema_version": "rlinf-dynamic-benchmark-optimal-shard-v0.1",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "accepted_count": 2,
        "attempted_reset_count": 2,
        "candidate_attempt_count": 4,
        "candidate_search_mode": "full-pool",
        "selection_mode": "planner-pareto",
        "budget_histogram": {"2": 2},
    }


def _shard_records() -> tuple[list[dict], list[dict], list[dict]]:
    results = [
        {
            "reset_index": index,
            "episode_id": f"episode-{index}",
            "candidate_count": 2,
            "budget_used": 2,
            "candidate_search_mode": "full-pool",
            "selection_mode": "planner-pareto",
            "accepted": True,
        }
        for index in range(2)
    ]
    attempts = [
        {"episode_id": f"episode-{reset_index}", "candidate_index": candidate_index}
        for reset_index in range(2)
        for candidate_index in range(2)
    ]
    winners = [{"request": {"episode_id": f"episode-{index}"}} for index in range(2)]
    return results, attempts, winners


def test_shard_merge_requires_exact_directory_and_receipt_inventory(
    tmp_path: Path,
) -> None:
    for index in (0, 2):
        shard = tmp_path / f"shard-{index:02d}"
        shard.mkdir()
        receipt = _shard_receipt(shard_index=index, shard_count=3)
        (shard / "shard_complete.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="not exact and gap-free"):
        _load_shard_receipts(tmp_path)


def test_shard_merge_proves_exact_reset_and_full_pool_coverage(tmp_path: Path) -> None:
    export_state = {
        "max_resets": 2,
        "candidate_pool_size": 2,
        "candidate_search_mode": "full-pool",
        "selection_mode": "planner-pareto",
    }
    reset_manifest = [{"episode_id": f"episode-{index}"} for index in range(2)]
    results, attempts, winners = _shard_records()

    _validate_shard_records(
        shard=tmp_path / "shard-00",
        receipt=_shard_receipt(),
        export_state=export_state,
        reset_manifest=reset_manifest,
        results=results,
        attempts=attempts,
        winners=winners,
    )
    assert _expected_shard_indices(
        max_resets=200, shard_count=8, shard_index=0
    ) == list(range(25))
    assert _expected_shard_indices(
        max_resets=200, shard_count=8, shard_index=7
    ) == list(range(175, 200))

    duplicate_results = [dict(results[0]), {**results[1], "reset_index": 0}]
    with pytest.raises(ValueError, match="duplicate, gap, or order mismatch"):
        _validate_shard_records(
            shard=tmp_path / "shard-00",
            receipt=_shard_receipt(),
            export_state=export_state,
            reset_manifest=reset_manifest,
            results=duplicate_results,
            attempts=attempts,
            winners=winners,
        )

    duplicate_attempts = [*attempts[:3], {**attempts[3], "candidate_index": 0}]
    with pytest.raises(ValueError, match="candidate coverage"):
        _validate_shard_records(
            shard=tmp_path / "shard-00",
            receipt=_shard_receipt(),
            export_state=export_state,
            reset_manifest=reset_manifest,
            results=results,
            attempts=duplicate_attempts,
            winners=winners,
        )


def _write_sharded_quality_merge_fixture(
    tmp_path: Path,
) -> tuple[Path, dict, str, str]:
    thresholds, receipt_path, receipt_sha256 = _calibration_receipt_fixture(tmp_path)
    threshold_bytes = json.dumps(
        thresholds,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    threshold_sha256 = hashlib.sha256(threshold_bytes).hexdigest()
    candidate_bytes = b'{"schema_version":"unit-test-candidates"}'
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    reset_rows = [{"episode_id": f"fixture-{index:05d}-s1"} for index in range(2)]
    reset_bytes = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in reset_rows
    ).encode("utf-8")
    reset_sha256 = hashlib.sha256(reset_bytes).hexdigest()
    root = tmp_path / "shards"
    root.mkdir()

    for shard_index, reset in enumerate(reset_rows):
        shard = root / f"shard-{shard_index:02d}"
        shard.mkdir()
        quality_identity = {
            "schema_version": thresholds["schema_version"],
            "sha256": threshold_sha256,
        }
        export_state = {
            "schema_version": "rlinf-dynamic-benchmark-optimal-export-state-v0.1",
            "task": "t2_trans",
            "split": "validation",
            "manifest_seed": 20261150,
            "max_resets": 2,
            "accepted_target": 2,
            "candidate_search_mode": "full-pool",
            "candidate_pool_size": 1,
            "initial_k": 1,
            "max_k": 1,
            "budget_sequence": [1],
            "selection_mode": "planner-pareto",
            "planner_dominance": None,
            "candidate_schema_version": "unit-test-candidates",
            "evaluator_identity": None,
            "compatibility_evidence": None,
            "calibration_evidence": None,
            "candidate_release_manifest_sha256": None,
            "candidate_release_provenance": None,
            "image_size": 64,
            "device": "cpu",
            "candidate_manifest_sha256": candidate_sha256,
            "reset_manifest_sha256": reset_sha256,
            "source_identity": {"fixture": "shard-merge"},
            "quality_v2_threshold_identity": quality_identity,
            "state_schema": {"fixture": True},
            "candidates": [{"candidate_id": "planner"}],
        }
        export_state["payload_sha256"] = _audit_payload_sha256(export_state)
        (shard / "candidate_manifest.json").write_bytes(candidate_bytes)
        (shard / "quality_v2_thresholds.json").write_bytes(threshold_bytes)
        (shard / "reset_manifest.jsonl").write_bytes(reset_bytes)
        (shard / "export_state.json").write_text(
            json.dumps(export_state, sort_keys=True), encoding="utf-8"
        )
        receipt_target = shard / thresholds["calibration_wave_receipt"]["relative_path"]
        receipt_target.parent.mkdir(parents=True)
        receipt_target.write_bytes(receipt_path.read_bytes())

        episode_id = reset["episode_id"]
        result = {
            "reset_index": shard_index,
            "episode_id": episode_id,
            "candidate_count": 1,
            "budget_used": 1,
            "candidate_search_mode": "full-pool",
            "selection_mode": "planner-pareto",
            "accepted": True,
        }
        attempt = {"episode_id": episode_id, "candidate_index": 0}
        episode_relative = f"episodes/t2_trans/validation/{episode_id}"
        winner = {
            "request": {"episode_id": episode_id},
            "relative_episode_dir": episode_relative,
        }
        for name, rows in (
            ("reset_results.jsonl", [result]),
            ("attempts.jsonl", [attempt]),
            ("winner_manifest.jsonl", [winner]),
        ):
            (shard / name).write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
        (shard / "progress.json").write_text(
            json.dumps(
                {
                    "started_unix_s": float(100 + shard_index),
                    "resume_count": 0,
                    "recovery_events": [],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (shard / "shard_complete.json").write_text(
            json.dumps(
                {
                    "schema_version": "rlinf-dynamic-benchmark-optimal-shard-v0.1",
                    "shard_index": shard_index,
                    "shard_count": 2,
                    "accepted_count": 1,
                    "attempted_reset_count": 1,
                    "candidate_attempt_count": 1,
                    "candidate_search_mode": "full-pool",
                    "selection_mode": "planner-pareto",
                    "budget_histogram": {"1": 1},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        episode_dir = shard / episode_relative
        episode_dir.mkdir(parents=True)
        (episode_dir / "episode.json").write_text("{}", encoding="utf-8")
        lightweight_dir = shard / "lightweight" / episode_id
        lightweight_dir.mkdir(parents=True)
        (lightweight_dir / "candidate-00.npz").write_bytes(b"fixture-tape")
    return root, thresholds, threshold_sha256, receipt_sha256


def test_shard_merge_preserves_qv3_receipt_for_independent_auditor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, thresholds, threshold_sha256, receipt_sha256 = (
        _write_sharded_quality_merge_fixture(tmp_path)
    )
    output = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_optimal_export_shards.py",
            "--root",
            str(root),
            "--output",
            str(output),
            "--accepted-episodes",
            "2",
        ],
    )

    _merge_shards_main()

    expected_identity = {
        "schema_version": thresholds["schema_version"],
        "sha256": threshold_sha256,
    }
    card = json.loads((output / "dataset_card.json").read_text(encoding="utf-8"))
    state = json.loads((output / "export_state.json").read_text(encoding="utf-8"))
    merged_thresholds = json.loads(
        (output / "quality_v2_thresholds.json").read_text(encoding="utf-8")
    )
    assert card["quality_v2_threshold_identity"] == expected_identity
    assert state["quality_v2_threshold_identity"] == expected_identity
    assert _audit_quality_v2_calibration_receipt_artifact(
        output,
        merged_thresholds,
    ) == {
        "relative_path": "provenance/calibration_wave/wave_receipt.json",
        "file_sha256": receipt_sha256,
        "payload_sha256": receipt_sha256,
    }
    checksum_inventory = (output / "SHA256SUMS").read_text(encoding="utf-8")
    assert "quality_v2_thresholds.json" in checksum_inventory
    assert "provenance/calibration_wave/wave_receipt.json" in checksum_inventory


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("missing_identity", "exact quality-v2 threshold identity"),
        ("different_identity", "different quality-v2 threshold or receipt identity"),
        ("missing_receipt", "missing or symlinked"),
    ),
)
def test_shard_merge_rejects_qv3_identity_or_receipt_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    root, _, _, _ = _write_sharded_quality_merge_fixture(tmp_path)
    shard = root / "shard-01"
    state_path = shard / "export_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if tamper == "missing_identity":
        del state["quality_v2_threshold_identity"]
    elif tamper == "different_identity":
        threshold_path = shard / "quality_v2_thresholds.json"
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold["fixture_variant"] = "different-shard"
        threshold_bytes = json.dumps(
            threshold,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        threshold_path.write_bytes(threshold_bytes)
        state["quality_v2_threshold_identity"]["sha256"] = hashlib.sha256(
            threshold_bytes
        ).hexdigest()
    else:
        receipt_relative = "provenance/calibration_wave/wave_receipt.json"
        (shard / receipt_relative).unlink()
    state["payload_sha256"] = _audit_payload_sha256(state)
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    output = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_optimal_export_shards.py",
            "--root",
            str(root),
            "--output",
            str(output),
            "--accepted-episodes",
            "2",
        ],
    )

    with pytest.raises(ValueError, match=message):
        _merge_shards_main()


def test_attempt_tape_round_trip_recomputes_shapes_hashes_and_score(
    tmp_path: Path,
) -> None:
    steps = 3
    arrays = {
        "states": np.arange((steps + 1) * 5, dtype=np.float32).reshape(steps + 1, 5),
        "policy_actions": np.full((steps, 7), 0.25, dtype=np.float32),
        "actions": np.full((steps, 7), 0.5, dtype=np.float64),
        "rewards": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "terminated": np.asarray([False, False, True], dtype=np.bool_),
        "truncated": np.zeros(steps, dtype=np.bool_),
    }
    relative, tape_sha256 = _write_attempt_tape(
        tmp_path,
        episode_id="episode-1",
        candidate_index=0,
        arrays=arrays,
    )
    replay = {"passed": True, "outcomes_exact": True}
    record = {
        "schema_version": "rlinf-dynamic-benchmark-optimal-attempt-v0.1",
        "task_id": "t2_trans",
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "trajectory_completion": 1.0,
        "completion_time_s": steps * 0.002,
        "return": float(arrays["rewards"].sum(dtype=np.float64)),
        "control_steps": steps,
        "action_l2_sum": float(np.square(arrays["actions"]).sum()),
        "candidate_index": 0,
        "attempt_tape": relative,
        "attempt_tape_sha256": tape_sha256,
        "state_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["states"]).tobytes()
        ).hexdigest(),
        "policy_action_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["policy_actions"]).tobytes()
        ).hexdigest(),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["actions"]).tobytes()
        ).hexdigest(),
        "reward_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["rewards"]).tobytes()
        ).hexdigest(),
        "replay_validation": replay,
        "replay_validation_sha256": _audit_payload_sha256(replay),
        "quality_v2_gate": {"passed": True},
    }
    record["quality_score"] = list(_quality_score(record))
    record["eligible"] = _eligible(record)

    _audit_attempt_tape(tmp_path, record, expected_task="t2_trans")

    record["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content checksum"):
        _audit_attempt_tape(tmp_path, record, expected_task="t2_trans")


def test_attempt_v02_quality_tapes_are_replay_auditable(tmp_path: Path) -> None:
    steps = 3
    actions = np.full((steps, 7), 0.2, dtype=np.float64)
    poses = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    closing = np.tile(np.asarray([1.0, 0.0, 0.0]), (steps + 1, 1))
    object_poses = np.tile(
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        (steps + 1, 1),
    )
    contact_flags = np.zeros((steps + 1, 2), dtype=np.float64)
    observations = [
        SimpleNamespace(
            privileged={
                "eef_pose_xyzw": poses[index],
                "fingerpad_closing_axis_world": closing[index],
                "object_pose_wxyz": object_poses[index],
                "fingerpad_contact_flags": contact_flags[index],
            },
            events_since_last_observation=(),
        )
        for index in range(steps + 1)
    ]
    quality_v2 = trajectory_quality_v2_from_observations(
        observations,
        actions,
        task_id="t2_trans",
        continuous_dimensions=6,
    )
    thresholds = {
        "schema_version": "test-thresholds",
        "tasks": {
            "t2_trans": {
                "checks": [
                    {
                        "phase": "full_episode",
                        "metric": "orientation_reference.orientation_reference_error_max_rad",
                        "max": 0.1,
                    }
                ]
            }
        },
    }
    threshold_sha = "a" * 64
    quality_gate = evaluate_quality_v2_gate(quality_v2, thresholds, task_id="t2_trans")
    quality_gate["contract_sha256"] = threshold_sha
    arrays = {
        "states": np.zeros((steps + 1, 5), dtype=np.float32),
        "policy_actions": actions.astype(np.float32),
        "actions": actions,
        "rewards": np.ones(steps, dtype=np.float32),
        "terminated": np.asarray([False, False, True], dtype=np.bool_),
        "truncated": np.zeros(steps, dtype=np.bool_),
        "eef_pose_xyzw": poses,
        "fingerpad_closing_axis_world": closing,
        "object_pose_wxyz": object_poses,
        "fingerpad_contact_flags": contact_flags,
    }
    relative, tape_sha256 = _write_attempt_tape(
        tmp_path,
        episode_id="episode-v02",
        candidate_index=0,
        arrays=arrays,
    )
    replay = {"passed": True, "outcomes_exact": True}
    record = {
        "schema_version": "rlinf-dynamic-benchmark-optimal-attempt-v0.2",
        "task_id": "t2_trans",
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "trajectory_completion": 1.0,
        "completion_time_s": steps * 0.002,
        "return": float(arrays["rewards"].sum(dtype=np.float64)),
        "control_steps": steps,
        "action_l2_sum": float(np.square(actions).sum()),
        "candidate_index": 0,
        "attempt_tape": relative,
        "attempt_tape_sha256": tape_sha256,
        "state_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["states"]).tobytes()
        ).hexdigest(),
        "policy_action_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["policy_actions"]).tobytes()
        ).hexdigest(),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest(),
        "reward_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["rewards"]).tobytes()
        ).hexdigest(),
        "replay_validation": replay,
        "replay_validation_sha256": _audit_payload_sha256(replay),
        "quality_v2": quality_v2,
        "quality_v2_sha256": _audit_payload_sha256(quality_v2),
        "quality_v2_gate": quality_gate,
        "quality_v2_events_by_observation": [[] for _ in range(steps + 1)],
    }
    record["quality_score"] = list(_quality_score(record))
    record["eligible"] = _eligible(record)

    _audit_attempt_tape(
        tmp_path,
        record,
        expected_task="t2_trans",
        quality_v2_thresholds=thresholds,
        quality_v2_thresholds_sha256=threshold_sha,
    )

    v03_record = {
        **record,
        "schema_version": "rlinf-dynamic-benchmark-optimal-attempt-v0.3",
        "issued_equals_applied": True,
        "t5_replan_causal_timing_passed": None,
        "impact_end_to_first_qualifying_applied_correction_s": None,
    }
    _audit_attempt_tape(
        tmp_path,
        v03_record,
        expected_task="t2_trans",
        quality_v2_thresholds=thresholds,
        quality_v2_thresholds_sha256=threshold_sha,
    )
    v03_record["issued_equals_applied"] = False
    with pytest.raises(ValueError, match="issued_equals_applied=true"):
        _audit_attempt_tape(
            tmp_path,
            v03_record,
            expected_task="t2_trans",
            quality_v2_thresholds=thresholds,
            quality_v2_thresholds_sha256=threshold_sha,
        )


def _t5_causal_fixture() -> tuple[dict, dict[str, np.ndarray]]:
    control_hz = 20.0
    delay_steps = 1
    action_values = (
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        (0.1, 0.2, 0.0, 0.0, 0.0, 0.0, -1.0),
        (0.1, 0.2, 0.0, 0.0, 0.0, 0.0, -1.0),
    )
    issued = [
        {
            "policy_step": step,
            "issue_time_s": step / control_hz,
            "apply_policy_step": step + delay_steps,
            "apply_time_s": (step + delay_steps) / control_hz,
            "values": values,
        }
        for step, values in enumerate(action_values)
    ]
    applied = [
        {
            **issued[step],
            "actual_apply_policy_step": step + delay_steps,
            "actual_apply_time_s": (step + delay_steps) / control_hz,
        }
        for step in range(len(issued) - delay_steps)
    ]
    raw_env = SimpleNamespace(
        canonical_action_history={
            "issued_actions": tuple(issued),
            "applied_actions": tuple(applied),
        },
        timing_summary=lambda: {
            "action_delay_steps": delay_steps,
            "impact_end_time_s": 0.10,
            "first_contact_time_s": 0.20,
        },
    )
    fields, arrays = _t5_replan_causal_evidence(
        raw_env,
        control_steps=len(issued),
    )
    return {"task_id": "t5_replan", **fields}, arrays


def test_t5_causal_tape_round_trip_reconstructs_issued_applied_history() -> None:
    record, arrays = _t5_causal_fixture()

    _audit_t5_replan_causal_history(record, arrays, control_steps=4)

    assert record["t5_replan_causal_timing_passed"] is True
    assert record[
        "impact_end_to_first_qualifying_applied_correction_s"
    ] == pytest.approx(0.05)
    assert arrays["issued_action_values"].dtype == np.float64
    assert arrays["issued_policy_step"].dtype == np.int64
    assert arrays["action_value_semantic_labels"].dtype.kind == "U"
    assert all(array.dtype.kind != "O" for array in arrays.values())


def test_t5_causal_timing_gate_is_a_hard_eligibility_requirement() -> None:
    record = {
        "schema_version": "rlinf-dynamic-benchmark-optimal-attempt-v0.3",
        "task_id": "t5_replan",
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "replay_validation": {"passed": True},
        "quality_v2_gate": {"passed": True},
        "issued_equals_applied": False,
        "t5_replan_causal_timing_passed": False,
        "impact_end_to_first_qualifying_applied_correction_s": None,
    }

    assert not _eligible(record)
    assert not _audit_eligible(record)


@pytest.mark.parametrize(
    "tamper",
    ("missing", "linkage", "delay", "repeated", "gripper_only", "late", "non_applied"),
)
def test_t5_causal_tape_rejects_missing_tampered_or_noncausal_history(
    tamper: str,
) -> None:
    record, original_arrays = _t5_causal_fixture()
    arrays = {name: value.copy() for name, value in original_arrays.items()}
    if tamper == "missing":
        del arrays["issued_time_s"]
    elif tamper == "linkage":
        arrays["applied_action_values"][2, 0] += 0.1
    elif tamper == "delay":
        arrays["scheduled_apply_policy_step"][2] += 1
    elif tamper == "repeated":
        arrays["issued_action_values"][2] = arrays["issued_action_values"][1]
        arrays["applied_action_values"][2] = arrays["applied_action_values"][1]
    elif tamper == "gripper_only":
        arrays["issued_action_values"][2] = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )
        arrays["applied_action_values"][2] = arrays["issued_action_values"][2]
    elif tamper == "late":
        arrays["t5_timing_values"][1] = 0.15
    else:
        arrays["applied_action_values"] = arrays["applied_action_values"][:-1]

    with pytest.raises(ValueError):
        _audit_t5_replan_causal_history(record, arrays, control_steps=4)


def test_resume_preserves_dirty_tail_and_restores_committed_boundary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "export"
    output.mkdir()
    paths = {
        name: output / name
        for name in ("attempts.jsonl", "reset_results.jsonl", "winner_manifest.jsonl")
    }
    committed = {
        "attempts.jsonl": b'{"attempt":0}\n',
        "reset_results.jsonl": b'{"reset":0}\n',
        "winner_manifest.jsonl": b'{"winner":0}\n',
    }
    for name, path in paths.items():
        path.write_bytes(committed[name])
    progress = _progress_payload(
        export_state_sha256="a" * 64,
        started_unix_s=1.0,
        next_reset_index=1,
        accepted_count=1,
        candidate_attempt_count=1,
        budget_histogram={"8": 1},
        attempts_path=paths["attempts.jsonl"],
        reset_results_path=paths["reset_results.jsonl"],
        winners_path=paths["winner_manifest.jsonl"],
        resume_count=0,
        recovery_events=[],
    )
    for path in paths.values():
        with path.open("ab") as stream:
            stream.write(b'{"dirty":true}\n')
    dirty_episode = "episode-1"
    lightweight = output / "lightweight" / dirty_episode
    published = output / "episodes" / "t2_trans" / "train" / dirty_episode
    staging = output / ".staging" / "partial"
    for directory in (lightweight, published, staging):
        directory.mkdir(parents=True)
        (directory / "evidence.bin").write_bytes(b"dirty")
    rows = [
        SimpleNamespace(request=SimpleNamespace(episode_id="episode-0")),
        SimpleNamespace(request=SimpleNamespace(episode_id=dirty_episode)),
    ]

    recovery_name = _recover_partial_output(
        output=output,
        progress=progress,
        reset_rows=rows,
        task="t2_trans",
        split="train",
    )

    assert recovery_name is not None
    recovery = output.parent / recovery_name
    assert recovery.is_dir()
    for name, path in paths.items():
        assert path.read_bytes() == committed[name]
        assert _file_boundary(path) == progress["file_boundaries"][name]
        assert (recovery / name).is_file()
    assert not lightweight.exists()
    assert not published.exists()
    assert not (output / ".staging").exists()
    assert (recovery / "lightweight" / dirty_episode / "evidence.bin").is_file()
    assert (
        recovery / "episodes" / "t2_trans" / "train" / dirty_episode / "evidence.bin"
    ).is_file()


def test_candidate_restore_uses_canonical_request_reset_and_rearms_hidden_event() -> (
    None
):
    request = SimpleNamespace(episode_id="episode-1")

    class RawEnv:
        def __init__(self) -> None:
            self.reset_requests = []

        def reset(self, value):
            self.reset_requests.append(value)
            return "canonical-observation"

    class VectorEnv:
        def __init__(self) -> None:
            self.envs = [RawEnv()]
            self._requests = [None]
            self._raw_observations = [None]
            self._last_obs = None
            self.armed = []

        def load_checkpoint_state(self, state):
            assert state == {"checkpoint": True}
            self._requests[0] = request
            self._raw_observations[0] = "loaded-state-observation"

        def _arm_hidden_t5_event(self, raw_env, value):
            self.armed.append((raw_env, value))

        def _encode(self, observation, value):
            assert observation == "canonical-observation"
            assert value is request
            return np.asarray([1.0, 2.0], dtype=np.float32)

    env = VectorEnv()
    _restore_candidate_start(env, {"checkpoint": True})

    assert env.envs[0].reset_requests == [request]
    assert env.armed == [(env.envs[0], request)]
    assert env._raw_observations == ["canonical-observation"]
    np.testing.assert_array_equal(
        env._last_obs["states"].numpy(),
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    )


def test_replay_proxy_rearms_after_canonical_reset() -> None:
    request = object()

    class RawEnv:
        def reset(self, value):
            assert value is request
            return "observation"

        def step(self, action):
            return ("step", action)

        def save_state(self):
            return b"state"

    class VectorEnv:
        def __init__(self) -> None:
            self.armed = []

        def _arm_hidden_t5_event(self, raw_env, value):
            self.armed.append((raw_env, value))

    raw_env = RawEnv()
    vector_env = VectorEnv()
    proxy = _ArmedResetReplayEnv(vector_env, raw_env)

    assert proxy.reset(request) == "observation"
    assert vector_env.armed == [(raw_env, request)]
    assert proxy.step("action") == ("step", "action")
    assert proxy.save_state() == b"state"
