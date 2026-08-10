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

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from examples.embodiment import (
    audit_dynamic_benchmark_optimal_trajectories as optimal_auditor,
)
from examples.embodiment import (
    export_dynamic_benchmark_optimal_trajectories as optimal_exporter,
)
from examples.embodiment import export_dynamic_benchmark_rld2_review as review_exporter
from examples.embodiment.export_dynamic_benchmark_rld2_review import (
    CATEGORIES,
    HISTORICAL_REVIEW_SCHEMA,
    RELEASE_HANDOFF_BLOCKERS,
    RESET_COUNT,
    REVIEW_SCHEMA,
    _absolute_gate_eligible,
    _assert_disjoint_reset_rows,
    _audit_review_attempt_tape,
    _copy_quality_v2_calibration_wave_receipt,
    _decorate_trajectory_decisions,
    _load_thresholds,
    _mark_review_selected,
    _payload_sha256,
    _planner_comparison,
    _quality_v2_selection_manifest_provenance,
    _select_review_pairs,
    _threshold_check_inventory,
    _validate_attempt_coverage,
    _validate_promotion_receipt_payload,
    _validate_quality_v2_calibration_wave_receipt,
)

_CALIBRATION_BENCHMARK_COMMIT = "4" * 40


def _gate(smooth: float, orientation: float) -> dict:
    checks = [
        {
            "phase": "full_episode",
            "metric": "action.action_max_second_difference_l2",
            "actual": smooth,
            "max": 1.0,
            "passed": smooth <= 1.0,
        },
        {
            "phase": "post_contact",
            "metric": "approach_axis.approach_axis_error_max_rad",
            "actual": orientation,
            "max": 1.0,
            "passed": orientation <= 1.0,
        },
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _attempt(
    reset_index: int,
    candidate_index: int,
    *,
    smooth: float,
    orientation: float,
    success: bool = True,
    trajectory_eligible: bool = True,
    task: str = "t1_xyz",
    causal_gate: bool = True,
    causal_latency_s: float = 0.05,
) -> dict:
    kind = "planner" if candidate_index == 0 else "policy"
    return {
        "reset_index": reset_index,
        "episode_id": f"review-{reset_index:02d}",
        "task_id": task,
        "candidate_id": "planner"
        if kind == "planner"
        else f"learned-{candidate_index}",
        "candidate_index": candidate_index,
        "candidate_kind": kind,
        "success": success,
        "safety_failure": False,
        "finite_and_bounded": True,
        "replay_validation": {"passed": True},
        "quality_v2_gate": _gate(smooth, orientation),
        "trajectory_completion": 1.0 if success else 0.5,
        "completion_time_s": 2.0,
        "control_steps": 40,
        "action_l2_sum": 10.0,
        "absolute_gate_eligible": bool(
            success and smooth <= 1.0 and orientation <= 1.0
        ),
        "trajectory_eligible": trajectory_eligible,
        "promotion_validated": kind == "policy",
        "review_selected": False,
        "issued_equals_applied": task != "t5_replan",
        "t5_replan_causal_timing_passed": (
            causal_gate if task == "t5_replan" else None
        ),
        "impact_end_to_first_qualifying_applied_correction_s": (
            causal_latency_s if task == "t5_replan" and causal_gate else None
        ),
    }


def _selection_fixture(*, include_rejection: bool = True) -> list[dict]:
    attempts: list[dict] = []
    for reset_index in range(6):
        attempts.append(
            _attempt(
                reset_index,
                0,
                smooth=0.25 + 0.02 * reset_index,
                orientation=0.20 + 0.01 * reset_index,
            )
        )
        if reset_index == 4 and include_rejection:
            attempts.append(
                _attempt(
                    reset_index,
                    1,
                    smooth=1.35,
                    orientation=0.3,
                    trajectory_eligible=False,
                )
            )
        elif reset_index == 5:
            attempts.append(
                _attempt(
                    reset_index,
                    1,
                    smooth=0.45,
                    orientation=0.45,
                    success=False,
                    trajectory_eligible=False,
                )
            )
        else:
            attempts.append(
                _attempt(
                    reset_index,
                    1,
                    smooth=0.35 + 0.12 * reset_index,
                    orientation=0.30 + 0.14 * reset_index,
                )
            )
    return attempts


def test_category_selector_builds_six_same_reset_pairs_and_keeps_rejection() -> None:
    attempts = _selection_fixture()

    cards, missing = _select_review_pairs(attempts)
    marked = _mark_review_selected(attempts, cards)

    assert missing == ()
    assert tuple(card["category"] for card in cards) == CATEGORIES
    for card in cards:
        planner = next(
            row
            for row in marked
            if row["reset_index"] == card["reset_index"]
            and row["candidate_index"] == card["planner_candidate_index"]
        )
        comparison = next(
            row
            for row in marked
            if row["reset_index"] == card["reset_index"]
            and row["candidate_index"] == card["comparison_candidate_index"]
        )
        assert planner["episode_id"] == comparison["episode_id"] == card["episode_id"]
        assert planner["review_selected"] is True
        assert comparison["review_selected"] is True
        assert card["category"] in planner["review_categories"]
        assert card["category"] in comparison["review_categories"]
    rejected = next(
        card for card in cards if card["category"] == "rejected_jitter_or_tilt"
    )
    rejected_row = next(
        row
        for row in marked
        if row["reset_index"] == rejected["reset_index"]
        and row["candidate_index"] == rejected["comparison_candidate_index"]
    )
    assert rejected_row["quality_v2_gate"]["passed"] is False
    assert rejected_row["trajectory_eligible"] is False
    assert rejected_row["review_selected"] is True


def test_category_selector_reports_incomplete_without_replacement_sampling() -> None:
    cards, missing = _select_review_pairs(_selection_fixture(include_rejection=False))

    assert "rejected_jitter_or_tilt" in missing
    assert len(cards) < len(CATEGORIES)


def test_categories_do_not_become_missing_only_because_they_share_a_reset() -> None:
    attempts = [
        _attempt(0, 0, smooth=0.3, orientation=0.3),
        _attempt(0, 1, smooth=0.8, orientation=0.8),
        _attempt(
            0,
            2,
            smooth=1.4,
            orientation=0.4,
            trajectory_eligible=False,
        ),
    ]

    cards, missing = _select_review_pairs(attempts)

    assert missing == ()
    assert len(cards) == len(CATEGORIES)
    assert {card["reset_index"] for card in cards} == {0}


def _comparison_attempt(
    candidate_index: int,
    *,
    path_value: float,
    control_steps: int,
    task_quality: dict | None = None,
    task: str = "t1_xyz",
    causal_gate: bool = True,
    causal_latency_s: float = 0.05,
) -> dict:
    path_passed = path_value <= 2.0
    gate = {
        "passed": path_passed,
        "checks": [
            {
                "phase": "full_episode",
                "metric": "eef_motion.eef_translation_path_length_m",
                "actual": path_value,
                "max": 2.0,
                "passed": path_passed,
            },
            {
                "phase": "full_episode",
                "metric": "action.action_tv_l2_mean_per_transition",
                "actual": 0.5,
                "max": 1.0,
                "passed": True,
            },
        ],
    }
    return {
        "reset_index": 0,
        "episode_id": "review-00",
        "task_id": task,
        "candidate_id": "planner"
        if candidate_index == 0
        else f"learned-{candidate_index}",
        "candidate_index": candidate_index,
        "candidate_kind": "planner" if candidate_index == 0 else "policy",
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "replay_validation": {"passed": True},
        "quality_v2_gate": gate,
        "trajectory_completion": 1.0,
        "completion_time_s": 2.0,
        "control_steps": control_steps,
        "action_l2_sum": 10.0,
        "task_quality": task_quality,
        "promotion_validated": candidate_index != 0,
        "issued_equals_applied": task != "t5_replan",
        "t5_replan_causal_timing_passed": (
            causal_gate if task == "t5_replan" else None
        ),
        "impact_end_to_first_qualifying_applied_correction_s": (
            causal_latency_s if task == "t5_replan" and causal_gate else None
        ),
    }


def test_same_reset_eligibility_requires_nonworse_and_strict_improvement() -> None:
    threshold_checks = [
        {
            "phase": "full_episode",
            "metric": "eef_motion.eef_translation_path_length_m",
            "max": 2.0,
            "direction": "minimize",
            "paired_comparison_family": "translation_path_m",
            "paired_nonworse_absolute_tolerance": 0.001,
            "paired_nonworse_relative_tolerance": 0.02,
            "paired_strict_improvement_absolute": 0.01,
            "paired_strict_improvement_relative": 0.05,
        },
        {
            "phase": "full_episode",
            "metric": "action.action_tv_l2_mean_per_transition",
            "max": 1.0,
            "direction": "minimize",
            "paired_comparison_family": "action_l2",
            "paired_nonworse_absolute_tolerance": 0.001,
            "paired_nonworse_relative_tolerance": 0.02,
            "paired_strict_improvement_absolute": 0.01,
            "paired_strict_improvement_relative": 0.05,
        },
    ]
    planner = _comparison_attempt(0, path_value=1.0, control_steps=40)
    improved = _comparison_attempt(1, path_value=0.8, control_steps=40)
    exact_tie = _comparison_attempt(2, path_value=1.0, control_steps=40)

    rows = _decorate_trajectory_decisions(
        [planner, improved, exact_tie], threshold_checks
    )

    learned_improved = next(row for row in rows if row["candidate_index"] == 1)
    learned_tie = next(row for row in rows if row["candidate_index"] == 2)
    assert learned_improved["trajectory_eligible"] is True
    assert any(
        "translation_path_length" in name
        for name in learned_improved["planner_comparison"][
            "strict_improvement_dimensions"
        ]
    )
    assert learned_tie["trajectory_eligible"] is False
    assert learned_tie["absolute_gate_eligible"] is True
    assert learned_tie["planner_comparison"]["planner_exact_tie"] is True


def test_same_reset_eligibility_understands_task_quality_schema_direction() -> None:
    threshold_checks = [
        {
            "phase": "full_episode",
            "metric": "eef_motion.eef_translation_path_length_m",
            "max": 2.0,
            "direction": "minimize",
            "paired_comparison_family": "translation_path_m",
            "paired_nonworse_absolute_tolerance": 0.001,
            "paired_nonworse_relative_tolerance": 0.02,
            "paired_strict_improvement_absolute": 0.01,
            "paired_strict_improvement_relative": 0.05,
        },
        {
            "phase": "full_episode",
            "metric": "action.action_tv_l2_mean_per_transition",
            "max": 1.0,
            "direction": "minimize",
            "paired_comparison_family": "action_l2",
            "paired_nonworse_absolute_tolerance": 0.001,
            "paired_nonworse_relative_tolerance": 0.02,
            "paired_strict_improvement_absolute": 0.01,
            "paired_strict_improvement_relative": 0.05,
        },
    ]
    planner_quality = {
        "components": {
            "placement_score": {
                "direction": "maximize",
                "value": 0.8,
                "scientific_resolution": 0.01,
            }
        }
    }
    learned_quality = deepcopy(planner_quality)
    learned_quality["components"]["placement_score"]["value"] = 0.9
    planner = _comparison_attempt(
        0, path_value=1.0, control_steps=40, task_quality=planner_quality
    )
    learned = _comparison_attempt(
        1, path_value=1.0, control_steps=40, task_quality=learned_quality
    )

    rows = _decorate_trajectory_decisions([planner, learned], threshold_checks)

    learned_row = next(row for row in rows if row["candidate_index"] == 1)
    assert learned_row["trajectory_eligible"] is True
    assert (
        "utility.task_quality.placement_score"
        in learned_row["planner_comparison"]["strict_improvement_dimensions"]
    )


def test_failed_planner_gate_does_not_bypass_remaining_nonworse_dimensions() -> None:
    threshold_checks = [
        {
            "phase": "full_episode",
            "metric": "eef_motion.eef_translation_path_length_m",
            "max": 2.0,
            "direction": "minimize",
            "paired_comparison_family": "translation_path_m",
            "paired_nonworse_absolute_tolerance": 0.001,
            "paired_nonworse_relative_tolerance": 0.02,
            "paired_strict_improvement_absolute": 0.01,
            "paired_strict_improvement_relative": 0.05,
        },
        {
            "phase": "full_episode",
            "metric": "action.action_tv_l2_mean_per_transition",
            "max": 1.0,
            "direction": "minimize",
            "paired_comparison_family": "action_l2",
            "paired_nonworse_absolute_tolerance": 0.001,
            "paired_nonworse_relative_tolerance": 0.02,
            "paired_strict_improvement_absolute": 0.01,
            "paired_strict_improvement_relative": 0.05,
        },
    ]
    planner = _comparison_attempt(0, path_value=2.1, control_steps=40)
    learned = _comparison_attempt(1, path_value=1.0, control_steps=50)

    rows = _decorate_trajectory_decisions([planner, learned], threshold_checks)

    learned_row = next(row for row in rows if row["candidate_index"] == 1)
    assert learned_row["absolute_gate_eligible"] is True
    assert learned_row["planner_comparison"]["planner_nonworse_all_dimensions"] is False
    assert learned_row["trajectory_eligible"] is False


def test_t5_causal_gate_and_issued_applied_identity_are_absolute_requirements() -> None:
    failed_gate = _attempt(
        0,
        1,
        smooth=0.5,
        orientation=0.5,
        task="t5_replan",
        causal_gate=False,
    )
    assert _absolute_gate_eligible(failed_gate) is False

    wrong_history = _attempt(0, 1, smooth=0.5, orientation=0.5, task="t5_replan")
    wrong_history["issued_equals_applied"] = True
    with pytest.raises(ValueError, match="distinct issued/applied"):
        _absolute_gate_eligible(wrong_history)

    non_t5 = _attempt(0, 1, smooth=0.5, orientation=0.5)
    assert non_t5["issued_equals_applied"] is True
    assert _absolute_gate_eligible(non_t5) is True


def test_t5_causal_latency_is_a_nonworse_and_strict_comparison_dimension() -> None:
    thresholds = [
        {
            "phase": "full_episode",
            "metric": "eef_motion.eef_translation_path_length_m",
            "max": 2.0,
            "direction": "minimize",
            "paired_comparison_family": "translation_path_m",
            "paired_nonworse_absolute_tolerance": 0.001,
            "paired_nonworse_relative_tolerance": 0.02,
            "paired_strict_improvement_absolute": 0.01,
            "paired_strict_improvement_relative": 0.05,
        },
        {
            "phase": "full_episode",
            "metric": "action.action_tv_l2_mean_per_transition",
            "max": 1.0,
            "direction": "minimize",
            "paired_comparison_family": "action_l2",
            "paired_nonworse_absolute_tolerance": 0.001,
            "paired_nonworse_relative_tolerance": 0.02,
            "paired_strict_improvement_absolute": 0.01,
            "paired_strict_improvement_relative": 0.05,
        },
    ]
    planner = _comparison_attempt(
        0,
        path_value=1.0,
        control_steps=40,
        task="t5_replan",
        causal_latency_s=0.05,
    )
    regressed = _comparison_attempt(
        1,
        path_value=1.0,
        control_steps=40,
        task="t5_replan",
        causal_latency_s=0.06,
    )
    improved = _comparison_attempt(
        2,
        path_value=1.0,
        control_steps=40,
        task="t5_replan",
        causal_latency_s=0.04,
    )

    regression = _planner_comparison(regressed, planner, thresholds)
    strict = _planner_comparison(improved, planner, thresholds)

    assert regression["planner_nonworse_all_dimensions"] is False
    assert any(
        row["name"] == "causal.impact_end_to_first_qualifying_applied_correction_s"
        and row["nonworse"] is False
        for row in regression["dimensions"]
    )
    assert (
        "causal.impact_end_to_first_qualifying_applied_correction_s"
        in strict["strict_improvement_dimensions"]
    )


def test_review_attempt_tape_uses_canonical_auditor_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tape = tmp_path / "attempt.npz"
    np.savez(tape, actions=np.zeros((1, 7), dtype=np.float64))
    record = {
        "schema_version": optimal_auditor.ATTEMPT_SCHEMA,
        "task_id": "t1_xyz",
        "attempt_tape": tape.name,
        "attempt_tape_sha256": hashlib.sha256(tape.read_bytes()).hexdigest(),
        "issued_equals_applied": True,
        "t5_replan_causal_timing_passed": None,
        "impact_end_to_first_qualifying_applied_correction_s": None,
    }
    calls = 0

    def audit(root: Path, payload: dict, **_: object) -> None:
        nonlocal calls
        calls += 1
        path = root / payload["attempt_tape"]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != payload["attempt_tape_sha256"]
        ):
            raise ValueError("attempt tape checksum mismatch")

    monkeypatch.setattr(optimal_auditor, "_audit_attempt_tape", audit)
    _audit_review_attempt_tape(
        tmp_path,
        record,
        task="t1_xyz",
        quality_v2_thresholds={},
        quality_v2_thresholds_sha256="a" * 64,
    )
    assert calls == 1

    tape.write_bytes(tape.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        _audit_review_attempt_tape(
            tmp_path,
            record,
            task="t1_xyz",
            quality_v2_thresholds={},
            quality_v2_thresholds_sha256="a" * 64,
        )


def _promotion_receipt(
    *,
    env_steps: int = 12_000,
    role: str = "best",
    decision: str = "promote",
    reason: str = "strict_planner_nonworse_improvement",
    rejection_reasons: list[dict] | None = None,
) -> dict:
    payload = {
        "schema_version": "rld2-qa-policy-promotion-v0.2",
        "task_id": "t1_xyz",
        "candidate_id": "learned-s1",
        "policy": {
            "path": "/runs/t1_xyz/best_policy.pt",
            "sha256": "a" * 64,
            "seed": 1,
            "run_tag": "cycle1-seed1",
            "checkpoint_role": role,
            "env_steps": env_steps,
            "metadata_path": "/runs/t1_xyz/summary.json",
            "metadata_sha256": "5" * 64,
            "metadata_payload_sha256": "6" * 64,
        },
        "inference": {
            "residual_scale": 0.25,
            "deterministic": True,
            "action_noise": False,
        },
        "validation_receipt": {
            "path": "validation/evaluation.json",
            "sha256": "b" * 64,
            "payload_sha256": "7" * 64,
            "evaluator_schema": "rlinf-dynamic-benchmark-expert-evaluation-v0.1",
            "evaluator_source_sha256": "c" * 64,
            "partition": "validation",
            "reset_manifest_path": "reset_manifest.jsonl",
            "reset_manifest_sha256": "d" * 64,
            "test_exposure": False,
            "review_exposure": False,
            "calibration_exposure": False,
            "quality_threshold_schema": "threshold-v0.3",
            "quality_threshold_sha256": "e" * 64,
            "attempt_schema_version": optimal_auditor.ATTEMPT_SCHEMA,
            "all_successful_quality_gates_passed": True,
            "all_successful_t5_causal_gates_passed": True,
        },
        "quality_v2_calibration_wave_receipt": {
            "path": "/calibration/wave_receipt.json",
            "sha256": "2" * 64,
            "schema_version": (
                optimal_exporter.QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA
            ),
            "payload_sha256": "2" * 64,
            "dataset_relative_path": ("provenance/calibration_wave/wave_receipt.json"),
        },
        "selection": {
            "decision": decision,
            "reason": reason,
            "planner_nonworse_all_dimensions": True,
            "strict_improvement_dimensions": ["path.translation"],
            "rejection_reasons": (
                [] if rejection_reasons is None else rejection_reasons
            ),
            "selector_contract_sha256": "f" * 64,
            "selector_contract_path": "validation/selector.json",
            "planner_evaluation_path": "validation/planner_evaluation.json",
            "planner_evaluation_sha256": "0" * 64,
            "planner_evaluation_payload_sha256": "3" * 64,
            "attempt_artifacts_payload_sha256": "7" * 64,
            "evidence_path": "validation/selection_evidence.json",
            "evidence_sha256": "8" * 64,
            "evidence_payload_sha256": "9" * 64,
        },
        "source_identity": {
            "rlinf_commit": "1" * 40,
            "benchmark_commit": "2" * 40,
            "evaluator_rlinf_commit": "3" * 40,
            "files": {
                "policy_evaluator": {
                    "path": "/src/evaluate_dynamic_benchmark_expert.py",
                    "sha256": "c" * 64,
                },
                "planner_evaluator": {
                    "path": "/src/evaluate_dynamic_benchmark_planner.py",
                    "sha256": "d" * 64,
                },
                "qv3_comparator": {
                    "path": "/src/export_dynamic_benchmark_optimal_trajectories.py",
                    "sha256": "e" * 64,
                },
                "attempt_auditor": {
                    "path": "/src/audit_dynamic_benchmark_optimal_trajectories.py",
                    "sha256": "f" * 64,
                },
                "promotion_builder": {
                    "path": "/src/build_dynamic_benchmark_rld2_promotion.py",
                    "sha256": "0" * 64,
                },
            },
        },
        "image_identity": {"reference": "runtime:test", "sha256": "4" * 64},
    }
    payload["source_identity"]["sha256"] = hashlib.sha256(
        json.dumps(
            payload["source_identity"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def _validate_receipt(receipt: dict) -> dict:
    return _validate_promotion_receipt_payload(
        receipt,
        task="t1_xyz",
        candidate_id="learned-s1",
        policy_path="/runs/t1_xyz/best_policy.pt",
        policy_sha256="a" * 64,
        residual_scale=0.25,
        rlinf_commit="1" * 40,
        benchmark_commit="2" * 40,
        threshold_schema="threshold-v0.3",
        threshold_sha256="e" * 64,
        calibration_receipt_identity={
            "relative_path": "provenance/calibration_wave/wave_receipt.json",
            "file_sha256": "2" * 64,
            "payload_sha256": "2" * 64,
        },
        evaluator_rlinf_commit="3" * 40,
        evaluator_source_sha256="c" * 64,
    )


def _reseal_promotion_receipt(receipt: dict, *, reseal_source: bool = True) -> None:
    if reseal_source:
        unsigned_source = dict(receipt["source_identity"])
        unsigned_source.pop("sha256", None)
        receipt["source_identity"]["sha256"] = hashlib.sha256(
            json.dumps(
                unsigned_source,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    receipt["payload_sha256"] = _payload_sha256(receipt)


def test_promotion_receipt_rejects_zero_step_final_unpromoted_and_test_exposure() -> (
    None
):
    assert _validate_receipt(_promotion_receipt())["selection"]["decision"] == "promote"

    for mutation, message in (
        (("policy", "env_steps", 0), "env_steps"),
        (("policy", "checkpoint_role", "final"), "checkpoint_role=best"),
        (("selection", "decision", "hold"), "promote decision"),
        (("selection", "reason", "no_strict_improvement"), "noncanonical"),
        (
            (
                "selection",
                "rejection_reasons",
                [{"code": "planner_success_policy_failure"}],
            ),
            "formal rejection reasons",
        ),
        (("validation_receipt", "test_exposure", True), "test-exposed"),
        (
            ("validation_receipt", "all_successful_t5_causal_gates_passed", False),
            "T5 causal gates",
        ),
        (
            ("quality_v2_calibration_wave_receipt", "sha256", "3" * 64),
            "calibration receipt identity",
        ),
    ):
        receipt = deepcopy(_promotion_receipt())
        section, key, value = mutation
        receipt[section][key] = value
        receipt["payload_sha256"] = _payload_sha256(receipt)
        with pytest.raises(ValueError, match=message):
            _validate_receipt(receipt)

    historical = _promotion_receipt()
    historical["schema_version"] = "rld2-qa-policy-promotion-v0.1"
    historical["payload_sha256"] = _payload_sha256(historical)
    with pytest.raises(ValueError, match="historical promotion schema"):
        _validate_receipt(historical)


def test_promotion_v02_selection_is_exact_and_keep_planner_never_enters_pool() -> None:
    validated = _validate_receipt(_promotion_receipt())
    assert set(validated["selection"]) == review_exporter._SELECTION_KEYS
    assert validated["selection"]["reason"] == ("strict_planner_nonworse_improvement")
    assert validated["selection"]["rejection_reasons"] == []

    for field_inventory_mutation in ("missing", "extra"):
        receipt = _promotion_receipt()
        if field_inventory_mutation == "missing":
            receipt["selection"].pop("reason")
        else:
            receipt["selection"]["decision_reason"] = receipt["selection"]["reason"]
        _reseal_promotion_receipt(receipt)
        with pytest.raises(ValueError, match="selection field inventory"):
            _validate_receipt(receipt)

    formal_rejection = _promotion_receipt(
        decision="keep_planner",
        reason="formal_gate_rejection",
        rejection_reasons=[
            {
                "code": "planner_success_policy_failure",
                "scope": "reset",
                "reset_index": 0,
                "episode_id": "validation-00",
            }
        ],
    )
    formal_rejection["selection"]["planner_nonworse_all_dimensions"] = False
    formal_rejection["selection"]["strict_improvement_dimensions"] = []
    _reseal_promotion_receipt(formal_rejection)
    with pytest.raises(ValueError, match="no promote decision"):
        _validate_receipt(formal_rejection)

    exact_tie = _promotion_receipt(
        decision="keep_planner",
        reason="no_strict_improvement",
    )
    exact_tie["selection"]["strict_improvement_dimensions"] = []
    _reseal_promotion_receipt(exact_tie)
    with pytest.raises(ValueError, match="no promote decision"):
        _validate_receipt(exact_tie)


def test_promotion_v02_source_identity_is_exact_and_hash_bound() -> None:
    validated = _validate_receipt(_promotion_receipt())
    assert set(validated["source_identity"]) == {
        "rlinf_commit",
        "benchmark_commit",
        "evaluator_rlinf_commit",
        "files",
        "sha256",
    }
    assert set(validated["source_identity"]["files"]) == {
        "policy_evaluator",
        "planner_evaluator",
        "qv3_comparator",
        "attempt_auditor",
        "promotion_builder",
    }

    old_alias = _promotion_receipt()
    source = old_alias["source_identity"]
    source["evaluator_commit"] = source.pop("evaluator_rlinf_commit")
    _reseal_promotion_receipt(old_alias)
    with pytest.raises(ValueError, match="source identity field inventory"):
        _validate_receipt(old_alias)

    extra_alias = _promotion_receipt()
    extra_alias["source_identity"]["policy_rlinf_commit"] = "1" * 40
    _reseal_promotion_receipt(extra_alias)
    with pytest.raises(ValueError, match="source identity field inventory"):
        _validate_receipt(extra_alias)

    missing_files = _promotion_receipt()
    missing_files["source_identity"].pop("files")
    _reseal_promotion_receipt(missing_files)
    with pytest.raises(ValueError, match="source identity field inventory"):
        _validate_receipt(missing_files)

    missing_file_field = _promotion_receipt()
    missing_file_field["source_identity"]["files"]["attempt_auditor"].pop("path")
    _reseal_promotion_receipt(missing_file_field)
    with pytest.raises(ValueError, match="attempt_auditor field inventory"):
        _validate_receipt(missing_file_field)

    tampered = _promotion_receipt()
    tampered["source_identity"]["files"]["planner_evaluator"]["sha256"] = "1" * 64
    _reseal_promotion_receipt(tampered, reseal_source=False)
    with pytest.raises(ValueError, match="source SHA-256 does not recompute"):
        _validate_receipt(tampered)

    wrong_evaluator_commit = _promotion_receipt()
    wrong_evaluator_commit["source_identity"]["evaluator_rlinf_commit"] = "4" * 40
    _reseal_promotion_receipt(wrong_evaluator_commit)
    with pytest.raises(ValueError, match="source commit mismatch"):
        _validate_receipt(wrong_evaluator_commit)

    wrong_policy_evaluator = _promotion_receipt()
    wrong_policy_evaluator["source_identity"]["files"]["policy_evaluator"]["sha256"] = (
        "1" * 64
    )
    _reseal_promotion_receipt(wrong_policy_evaluator)
    with pytest.raises(ValueError, match="policy evaluator source SHA-256 mismatch"):
        _validate_receipt(wrong_policy_evaluator)


def test_promotion_calibration_receipt_rejects_invented_file_sha256_alias() -> None:
    receipt = _promotion_receipt()
    receipt["quality_v2_calibration_wave_receipt"]["file_sha256"] = "2" * 64
    _reseal_promotion_receipt(receipt)

    with pytest.raises(ValueError, match="calibration wave receipt field inventory"):
        _validate_receipt(receipt)


def _reset_row(index: int, *, episode_id: str | None = None) -> dict:
    return {
        "task_id": "t1_xyz",
        "split": "validation",
        "episode_id": episode_id or f"review-{index:02d}",
        "seed": index,
        "action_mode": "E7",
        "observation_track": "hybrid",
        "object_mode": "asym_t",
        "reset_mode": "default",
        "factors": {"initial_x_m": index / 100.0},
    }


def test_review_resets_are_exact_twenty_and_disjoint_from_checkpoint_selection() -> (
    None
):
    review = [_reset_row(index) for index in range(RESET_COUNT)]
    selection = [
        _reset_row(100 + index, episode_id=f"selection-{index}") for index in range(8)
    ]
    _assert_disjoint_reset_rows(review, [selection], task="t1_xyz")

    overlapping = deepcopy(selection)
    overlapping[0] = {**review[3], "episode_id": "different-label"}
    with pytest.raises(ValueError, match="overlap"):
        _assert_disjoint_reset_rows(review, [overlapping], task="t1_xyz")


def test_full_pool_coverage_rejects_one_missing_candidate_attempt() -> None:
    review = [_reset_row(index) for index in range(RESET_COUNT)]
    attempts = [
        {
            "reset_index": reset_index,
            "candidate_index": candidate_index,
            "episode_id": review[reset_index]["episode_id"],
        }
        for reset_index in range(RESET_COUNT)
        for candidate_index in range(3)
    ]
    _validate_attempt_coverage(attempts, review_rows=review, candidate_count=3)
    with pytest.raises(ValueError, match="complete reset x candidate pool"):
        _validate_attempt_coverage(attempts[:-1], review_rows=review, candidate_count=3)


@pytest.mark.parametrize("check_count", [10, 11])
def test_threshold_inventory_is_task_dynamic_and_phase_driven(check_count: int) -> None:
    checks = [
        {
            "phase": "acquisition_window" if index == 0 else "full_episode",
            "metric": f"dynamic.metric_{index}",
            "max": 1.0 + index,
            "direction": "minimize",
            "paired_comparison_family": "action_l2",
            "paired_nonworse_absolute_tolerance": 0.001,
            "paired_nonworse_relative_tolerance": 0.02,
            "paired_strict_improvement_absolute": 0.01,
            "paired_strict_improvement_relative": 0.05,
        }
        for index in range(check_count)
    ]

    inventory = _threshold_check_inventory(checks)

    assert len(inventory) == check_count
    assert ("acquisition_window", "dynamic.metric_0") in inventory


def test_provisional_v03_thresholds_cannot_authorize_review(tmp_path) -> None:
    path = tmp_path / "quality_v2_thresholds.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "se3-wam-trajectory-quality-v2-thresholds-v0.3",
                "formal_freeze_eligible": False,
                "tasks": {"t1_xyz": {}},
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="provisional"):
        _load_thresholds(path, digest)


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _calibration_receipt_fixture(tmp_path: Path) -> tuple[dict, Path, str]:
    reset_count = optimal_exporter.QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES
    receipt_tasks = []
    binding_tasks = []
    for ordinal, task_id in enumerate(optimal_exporter.QUALITY_V2_CALIBRATION_TASKS):
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
            "reset_count": reset_count,
            "task_quality_schema_version": "db0-episode-task-quality-v2",
            "reset_manifest_relative_path": f"tasks/{task_id}/reset_manifest.jsonl",
            "evaluation_relative_path": f"tasks/{task_id}/evaluation.json",
            **hashes,
        }
        receipt_tasks.append(receipt_task)
        binding_task = dict(receipt_task)
        binding_task["reset_identity_count"] = binding_task.pop("reset_count")
        binding_tasks.append(binding_task)
    task_order = list(optimal_exporter.QUALITY_V2_CALIBRATION_TASKS)
    receipt = {
        "schema_version": (optimal_exporter.QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA),
        "scientific_partition": "metric_calibration",
        "transport_split": "validation",
        "manifest_seed": 20261350,
        "task_count": len(task_order),
        "episodes_per_task": reset_count,
        "total_reset_count": len(task_order) * reset_count,
        "task_order": task_order,
        "wave_contract_sha256": "a" * 64,
        "predeclaration_receipt_sha256": "b" * 64,
        "source_identity": {
            "wave_id": "review-unit-test",
            "benchmark_commit": _CALIBRATION_BENCHMARK_COMMIT,
        },
        "disjointness": {"verified": True},
        "tasks": receipt_tasks,
    }
    receipt_bytes = _canonical_bytes(receipt)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    receipt_path = tmp_path / "authoritative_wave_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    wave_binding = {
        "binding_status": "bound",
        **{key: value for key, value in receipt.items() if key != "tasks"},
        "tasks": binding_tasks,
        "relative_path": "provenance/calibration_wave/wave_receipt.json",
        "sha256": receipt_sha256,
        "file_sha256": receipt_sha256,
        "payload_sha256": receipt_sha256,
    }
    thresholds = {
        "schema_version": "se3-wam-trajectory-quality-v2-thresholds-v0.3",
        "formal_freeze_eligible": True,
        "calibration_wave_receipt": wave_binding,
        "tasks": {"t1_xyz": {"checks": []}},
    }
    return thresholds, receipt_path, receipt_sha256


def test_review_calibration_receipt_is_copied_bound_and_root_checksummed(
    tmp_path: Path,
) -> None:
    thresholds, receipt_path, receipt_sha256 = _calibration_receipt_fixture(tmp_path)

    provenance, identity = _validate_quality_v2_calibration_wave_receipt(
        thresholds,
        receipt_path,
        receipt_sha256,
        expected_benchmark_commit=_CALIBRATION_BENCHMARK_COMMIT,
    )
    output = tmp_path / "review"
    output.mkdir()
    _copy_quality_v2_calibration_wave_receipt(output, provenance)
    optimal_exporter._root_checksums(output)
    manifest_provenance = _quality_v2_selection_manifest_provenance(
        threshold_schema=thresholds["schema_version"],
        threshold_sha256="c" * 64,
        paired_dimension_count=10,
        calibration_receipt_identity=identity,
    )

    assert identity == {
        "relative_path": "provenance/calibration_wave/wave_receipt.json",
        "file_sha256": receipt_sha256,
        "payload_sha256": receipt_sha256,
    }
    assert (
        manifest_provenance["quality_v2_calibration_wave_receipt_identity"] == identity
    )
    copied = output.joinpath(*Path(identity["relative_path"]).parts)
    assert copied.read_bytes() == receipt_path.read_bytes()
    assert f"{receipt_sha256}  {identity['relative_path']}\n" in (
        output / "SHA256SUMS"
    ).read_text(encoding="utf-8")


def test_review_calibration_receipt_binds_evaluator_benchmark_commit(
    tmp_path: Path,
) -> None:
    thresholds, receipt_path, receipt_sha256 = _calibration_receipt_fixture(tmp_path)

    with pytest.raises(ValueError, match="authenticated evaluator benchmark commit"):
        _validate_quality_v2_calibration_wave_receipt(
            thresholds,
            receipt_path,
            receipt_sha256,
            expected_benchmark_commit="5" * 40,
        )


def test_review_calibration_receipt_missing_fails_closed(tmp_path: Path) -> None:
    thresholds, _, receipt_sha256 = _calibration_receipt_fixture(tmp_path)

    with pytest.raises(ValueError, match="missing or symlinked"):
        _validate_quality_v2_calibration_wave_receipt(
            thresholds,
            tmp_path / "missing.json",
            receipt_sha256,
            expected_benchmark_commit=_CALIBRATION_BENCHMARK_COMMIT,
        )


def test_review_calibration_receipt_noncanonical_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    thresholds, receipt_path, _ = _calibration_receipt_fixture(tmp_path)
    tampered_path = tmp_path / "tampered_wave_receipt.json"
    tampered_path.write_bytes(receipt_path.read_bytes() + b"\n")
    tampered_sha256 = hashlib.sha256(tampered_path.read_bytes()).hexdigest()
    for key in ("sha256", "file_sha256", "payload_sha256"):
        thresholds["calibration_wave_receipt"][key] = tampered_sha256

    with pytest.raises(ValueError, match="not canonical JSON"):
        _validate_quality_v2_calibration_wave_receipt(
            thresholds,
            tampered_path,
            tampered_sha256,
            expected_benchmark_commit=_CALIBRATION_BENCHMARK_COMMIT,
        )


def test_review_calibration_receipt_task_hash_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    thresholds, receipt_path, _ = _calibration_receipt_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["tasks"][0]["task_contract_sha256"] = "f" * 64
    tampered_path = tmp_path / "task_tampered_wave_receipt.json"
    tampered_path.write_bytes(_canonical_bytes(receipt))
    tampered_sha256 = hashlib.sha256(tampered_path.read_bytes()).hexdigest()
    for key in ("sha256", "file_sha256", "payload_sha256"):
        thresholds["calibration_wave_receipt"][key] = tampered_sha256

    with pytest.raises(ValueError, match="task .* task_contract_sha256 mismatch"):
        _validate_quality_v2_calibration_wave_receipt(
            thresholds,
            tampered_path,
            tampered_sha256,
            expected_benchmark_commit=_CALIBRATION_BENCHMARK_COMMIT,
        )


def test_review_calibration_receipt_path_traversal_fails_closed(
    tmp_path: Path,
) -> None:
    thresholds, receipt_path, receipt_sha256 = _calibration_receipt_fixture(tmp_path)
    thresholds["calibration_wave_receipt"]["relative_path"] = "../wave_receipt.json"

    with pytest.raises(ValueError, match="unsafe .*calibration receipt path"):
        _validate_quality_v2_calibration_wave_receipt(
            thresholds,
            receipt_path,
            receipt_sha256,
            expected_benchmark_commit=_CALIBRATION_BENCHMARK_COMMIT,
        )


def test_review_output_preexistence_stays_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing-review"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("do-not-touch", encoding="utf-8")
    monkeypatch.setattr(
        review_exporter.sys,
        "argv",
        [
            "export_dynamic_benchmark_rld2_review.py",
            "--candidate-manifest",
            str(tmp_path / "candidate.json"),
            "--expected-candidate-manifest-sha256",
            "a" * 64,
            "--quality-v2-thresholds",
            str(tmp_path / "thresholds.json"),
            "--expected-quality-v2-thresholds-sha256",
            "b" * 64,
            "--quality-v2-calibration-wave-receipt",
            str(tmp_path / "wave_receipt.json"),
            "--expected-quality-v2-calibration-wave-receipt-sha256",
            "c" * 64,
            "--evaluator-commit",
            "1" * 40,
            "--evaluator-benchmark-commit",
            "2" * 40,
            "--output",
            str(output),
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        review_exporter.main()
    assert sentinel.read_text(encoding="utf-8") == "do-not-touch"
    assert list(output.iterdir()) == [sentinel]


def test_review_schema_v02_has_no_stale_release_auditor_blocker() -> None:
    assert REVIEW_SCHEMA == "rlinf-dynamic-benchmark-rld2-paired-review-v0.2"
    assert HISTORICAL_REVIEW_SCHEMA == (
        "rlinf-dynamic-benchmark-rld2-paired-review-v0.1"
    )
    assert RELEASE_HANDOFF_BLOCKERS == ()
