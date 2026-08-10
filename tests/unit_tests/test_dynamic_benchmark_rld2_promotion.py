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
import torch

from examples.embodiment import (
    audit_dynamic_benchmark_optimal_trajectories as optimal_auditor,
)
from examples.embodiment import build_dynamic_benchmark_rld2_promotion as promotion
from examples.embodiment import export_dynamic_benchmark_optimal_trajectories as optimal

TASK = "t1_xyz"
RLINF_COMMIT = "1" * 40
BENCHMARK_COMMIT = "2" * 40
EVALUATOR_COMMIT = "3" * 40
IMAGE_SHA256 = "4" * 64
FORMAL_POLICY_VALIDATION_MANIFEST_SEED = 20261150
CHECKPOINT_SELECTION_VALIDATION_MANIFEST_SEED = 20261450


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _seal(value: dict) -> dict:
    value["payload_sha256"] = promotion._payload_sha256(value)
    return value


def _task_quality(
    episode_id: str, schema: dict, value: float = 1.0, *, task: str = TASK
) -> dict:
    summary = {
        "schema_version": schema["schema_version"],
        "episode_id": episode_id,
        "task_id": task,
        "evaluator_backend_id": "mujoco311-rs140-v1-rld2-quality",
        "schema_sha256": schema["schema_sha256"],
        "physics_sample_count": 10,
        "terminal": True,
        "components": {
            component["name"]: {
                "value": value,
                "direction": component["direction"],
                "unit": component["unit"],
                "scientific_resolution": component["scientific_resolution"],
                "reducer": component["reducer"],
            }
            for component in schema["components"]
        },
    }
    summary["summary_sha256"] = promotion._value_sha256(summary)
    return summary


Q_METRICS = (
    (
        "full_episode",
        "action.action_second_difference_l2_mean_per_transition",
        "action_l2",
    ),
    ("full_episode", "action.action_max_second_difference_l2", "action_l2"),
    (
        "full_episode",
        "action.action_total_variation_l2_mean_per_transition",
        "action_l2",
    ),
    (
        "full_episode",
        "eef_motion.eef_translation_path_length_m",
        "translation_path_m",
    ),
    (
        "full_episode",
        "eef_motion.eef_rotation_path_length_rad",
        "rotation_or_orientation_rad",
    ),
    (
        "full_episode",
        "eef_motion.eef_angular_jerk_max_rad_s3",
        "angular_jerk_rad_s3",
    ),
    (
        "full_episode",
        "eef_motion.eef_linear_jerk_max_m_s3",
        "linear_jerk_m_s3",
    ),
    (
        "full_episode",
        "eef_motion.eef_angular_jerk_rms_rad_s3",
        "angular_jerk_rad_s3",
    ),
    (
        "full_episode",
        "eef_motion.eef_linear_jerk_rms_m_s3",
        "linear_jerk_m_s3",
    ),
    (
        "acquisition_window",
        "approach_axis.approach_axis_error_max_rad",
        "rotation_or_orientation_rad",
    ),
    (
        "acquisition_window",
        "jaw_axis.jaw_axis_error_max_rad",
        "rotation_or_orientation_rad",
    ),
)


def _set_nested(root: dict, dotted: str, value: float) -> None:
    current = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _quality(values: dict[tuple[str, str], float] | None = None) -> dict:
    values = values or {}
    summary: dict = {
        "schema_version": promotion.QUALITY_V2_SUMMARY_SCHEMA,
        "phases": {},
    }
    for phase, metric, _ in Q_METRICS:
        target = (
            summary
            if phase == "full_episode"
            else summary["phases"].setdefault(phase, {})
        )
        _set_nested(target, metric, values.get((phase, metric), 1.0))
    return summary


def _gate(
    quality: dict, checks: list[dict], threshold_sha: str, *, task: str = TASK
) -> dict:
    rows = []
    for check in checks:
        target = (
            quality
            if check["phase"] == "full_episode"
            else quality["phases"][check["phase"]]
        )
        current = target
        for part in check["metric"].split("."):
            current = current[part]
        rows.append(
            {
                "phase": check["phase"],
                "metric": check["metric"],
                "actual": current,
                "max": check["max"],
                "passed": current <= check["max"],
            }
        )
    return {
        "schema_version": promotion.QUALITY_V2_GATE_SCHEMA,
        "contract_schema_version": promotion.QUALITY_V2_THRESHOLD_SCHEMA,
        "contract_sha256": threshold_sha,
        "task_id": task,
        "passed": all(row["passed"] for row in rows),
        "checks": rows,
    }


def _record(
    reset: dict,
    *,
    schema: dict,
    checks: list[dict],
    threshold_sha: str,
    quality_values: dict[tuple[str, str], float] | None = None,
    success: bool = True,
    safety_failure: bool = False,
) -> dict:
    actions = np.zeros((2, 7), dtype=np.float64)
    quality = _quality(quality_values)
    return {
        "episode_id": reset["episode_id"],
        "task_id": reset["task_id"],
        "seed": reset["seed"],
        "factors": reset["factors"],
        "source_group_id": reset["source_group_id"],
        "pair_id": reset["pair_id"],
        "pair_member_id": reset["pair_member_id"],
        "candidate_index": reset["candidate_index"],
        "success": success,
        "safety_failure": safety_failure,
        "termination_reason": "success" if success else "timeout",
        "trajectory_completion": 1.0 if success else 0.5,
        "completion_time_s": 1.0 if success else None,
        "return": 1.0,
        "control_steps": len(actions),
        "action_l2_sum": 0.0,
        "task_quality": _task_quality(
            reset["episode_id"], schema, task=str(reset["task_id"])
        ),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest(),
        "actions": actions.tolist(),
        "quality_v2": quality,
        "quality_v2_sha256": promotion._payload_sha256(quality),
        "quality_v2_gate": _gate(
            quality, checks, threshold_sha, task=str(reset["task_id"])
        ),
        "replay_validation": {
            "passed": True,
            "final_state_exact": True,
            "outcomes_exact": True,
            "task_quality_exact": True,
        },
        "events": ["success"] if success else [],
    }


def _selector(schema: dict, *, task: str = TASK) -> dict:
    components = {row["name"]: row for row in schema["components"]}

    def metric(direction: str, resolution: float, *, control: bool = False) -> dict:
        return {
            "direction": direction,
            "max_observed_replay_drift": 0.0,
            "scientific_resolution": resolution,
            "numeric_floor": 0.0 if control else 1.0e-6,
        }

    return {
        "schema_version": promotion.SELECTOR_SCHEMA,
        "task": task,
        "backend_id": "mujoco311-rs140-v1-rld2-quality",
        "quality_schema": schema,
        "calibration": {
            "replay_count": 3,
            "reset_episode_id": "selector-calibration",
            "reset_manifest_sha256": "a" * 64,
            "evidence_path": "selector_calibration.json",
            "evidence_sha256": "b" * 64,
        },
        "metrics": {
            "trajectory_completion": metric("max", 1.0e-6),
            "task_quality": {
                name: metric(
                    "max" if component["direction"] == "maximize" else "min",
                    component["scientific_resolution"],
                )
                for name, component in components.items()
            },
            "completion_time_s": metric("min", 0.002),
            "control_steps": metric("min", 1.0, control=True),
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
            *(f"task_quality.{name}" for name in components),
            "completion_time_s",
            "control_steps",
            "action_l2_sum",
        ],
    }


def _thresholds(*, task: str = TASK) -> dict:
    checks = [
        {
            "phase": phase,
            "metric": metric,
            "max": 2.0,
            "direction": "minimize",
            "paired_comparison_family": family,
            "paired_nonworse_absolute_tolerance": 0.01,
            "paired_nonworse_relative_tolerance": 0.0,
            "paired_strict_improvement_absolute": 0.02,
            "paired_strict_improvement_relative": 0.0,
        }
        for phase, metric, family in Q_METRICS
    ]
    return {
        "schema_version": promotion.QUALITY_V2_THRESHOLD_SCHEMA,
        "formal_freeze_eligible": True,
        "calibration_status": "frozen",
        "minimum_attempted_episodes": 20,
        "minimum_successful_episodes": 8,
        "calibration_wave_receipt": {
            "binding_status": "bound",
            "schema_version": "rld2-qa-planner-calibration-wave-receipt-v0.1",
            "scientific_partition": "metric_calibration",
            "task_count": 14,
            "episodes_per_task": 20,
            "total_reset_count": 280,
            "sha256": "c" * 64,
            "file_sha256": "c" * 64,
            "payload_sha256": "c" * 64,
            "relative_path": "provenance/calibration_wave/wave_receipt.json",
        },
        "tasks": {
            task: {
                "checks": checks,
                "orientation_mode": "world_down_tool_axis",
                "jaw_axis_mode": "object_xy_teacher_offset_mod_pi",
                "provenance": {
                    "formal_freeze_eligible": True,
                    "attempted_episode_count": 20,
                    "successful_episode_count": 20,
                },
            }
        },
    }


class Fixture:
    def __init__(self, root: Path, *, tie: bool = False, task: str = TASK) -> None:
        from se3_wam.benchmark.task_quality import task_quality_schema_manifest

        self.root = root
        self.task = task
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = task_quality_schema_manifest(task)
        self.thresholds = _thresholds(task=task)
        self.calibration_receipt_path = root / "wave_receipt.json"
        self.calibration_receipt_path.write_bytes(
            promotion._canonical_json(
                {
                    "schema_version": ("rld2-qa-planner-calibration-wave-receipt-v0.1"),
                    "task_count": 14,
                    "episodes_per_task": 20,
                    "total_reset_count": 280,
                }
            ).encode("utf-8")
        )
        self.calibration_receipt_sha = promotion._sha256(self.calibration_receipt_path)
        for key in ("sha256", "file_sha256", "payload_sha256"):
            self.thresholds["calibration_wave_receipt"][key] = (
                self.calibration_receipt_sha
            )
        self.threshold_path = root / "quality_v2_thresholds.json"
        _write_json(self.threshold_path, self.thresholds)
        self.threshold_sha = promotion._sha256(self.threshold_path)
        self.checks = self.thresholds["tasks"][task]["checks"]
        self.selector_path = root / "selector.json"
        _write_json(self.selector_path, _selector(self.schema, task=task))
        self.reset_path = root / "reset_manifest.jsonl"
        self.resets = [
            {
                "task_id": task,
                "split": "validation",
                "episode_id": f"validation-{index:02d}",
                "seed": 1000 + index,
                "action_mode": "E7",
                "observation_track": "hybrid",
                "object_mode": "asym_t",
                "reset_mode": "default",
                "factors": {"initial_x_m": index / 100.0},
                "source_group_id": f"group-{index // 2}",
                "pair_id": f"pair-{index // 2}",
                "pair_member_id": index % 2,
                "candidate_index": index,
            }
            for index in range(promotion.RESET_COUNT)
        ]
        self.reset_path.write_text(
            "".join(promotion._canonical_json(row) + "\n" for row in self.resets),
            encoding="utf-8",
        )
        self.reset_sha = promotion._sha256(self.reset_path)
        self.config = {
            "task": task,
            "seed": 7,
            "algorithm": "rlpd",
            "validation_manifest_seed": CHECKPOINT_SELECTION_VALIDATION_MANIFEST_SEED,
            "rlinf_commit": RLINF_COMMIT,
            "benchmark_commit": BENCHMARK_COMMIT,
        }
        self.validation = {"episodes": 8, "success_rate": 1.0}
        self.policy_path = root / "best_policy.pt"
        torch.save(
            {
                "schema_version": promotion.POLICY_SCHEMA,
                "config": self.config,
                "model": {"fixture": torch.tensor([1.0])},
                "normalizer": {"fixture": torch.tensor([1.0])},
                "state_schema": {"state_dim": 2, "mask_dim": 0},
                "infra_identity": {"image": "fixture"},
                "validation": self.validation,
                "env_steps": 100,
            },
            self.policy_path,
        )
        self.policy_sha = promotion._sha256(self.policy_path)
        self.metadata_path = root / "summary.json"
        _write_json(
            self.metadata_path,
            _seal(
                {
                    "schema_version": promotion.POLICY_METADATA_SCHEMA,
                    "status": "complete",
                    "config": self.config,
                    "infra_identity": {"image": "fixture"},
                    "best_validation": self.validation,
                    "best_score": [1.0],
                    "final_validation": self.validation,
                    "env_steps": 200,
                    "config_sha256": promotion._value_sha256(self.config),
                }
            ),
        )
        improved = (
            {}
            if tie
            else {("full_episode", "eef_motion.eef_translation_path_length_m"): 0.5}
        )
        self.policy_records = [
            _record(
                reset,
                schema=self.schema,
                checks=self.checks,
                threshold_sha=self.threshold_sha,
                quality_values=improved,
            )
            for reset in self.resets
        ]
        self.planner_records = [
            _record(
                reset,
                schema=self.schema,
                checks=self.checks,
                threshold_sha=self.threshold_sha,
            )
            for reset in self.resets
        ]
        for index, record in enumerate(self.policy_records):
            self._bind_attempt_tape(record, role="policy", index=index)
        for index, record in enumerate(self.planner_records):
            self._bind_attempt_tape(record, role="planner", index=index)
        self.policy_evaluation_path = root / "policy_evaluation.json"
        self.planner_evaluation_path = root / "planner_evaluation.json"
        self._write_evaluations()
        examples = Path(__file__).resolve().parents[2] / "examples" / "embodiment"
        self.policy_evaluator_source = examples / "evaluate_dynamic_benchmark_expert.py"
        self.planner_evaluator_source = (
            examples / "evaluate_dynamic_benchmark_planner.py"
        )

    def _bind_attempt_tape(self, record: dict, *, role: str, index: int) -> None:
        tape_dir = self.root / "tapes"
        tape_dir.mkdir(exist_ok=True)
        actions = np.asarray(record["actions"], dtype=np.float64)
        steps = len(actions)
        arrays = {
            "states": np.zeros((steps + 1, 2), dtype=np.float64),
            "policy_actions": actions.copy(),
            "actions": actions.copy(),
            "rewards": np.asarray([0.5, 0.5], dtype=np.float64),
            "terminated": np.asarray([False, True], dtype=np.bool_),
            "truncated": np.zeros(steps, dtype=np.bool_),
            "eef_pose_xyzw": np.zeros((steps + 1, 7), dtype=np.float64),
            "fingerpad_closing_axis_world": np.zeros((steps + 1, 3), dtype=np.float64),
            "object_pose_wxyz": np.zeros((steps + 1, 7), dtype=np.float64),
            "fingerpad_contact_flags": np.zeros((steps + 1, 2), dtype=np.bool_),
        }
        tape_path = tape_dir / f"{role}-{index:02d}.npz"
        np.savez(tape_path, **arrays)
        record.update(
            {
                "attempt_schema_version": optimal_auditor.ATTEMPT_SCHEMA,
                "attempt_tape": tape_path.relative_to(self.root).as_posix(),
                "attempt_tape_sha256": promotion._sha256(tape_path),
                "finite_and_bounded": True,
                "state_sha256": hashlib.sha256(
                    np.ascontiguousarray(arrays["states"]).tobytes()
                ).hexdigest(),
                "policy_action_sha256": hashlib.sha256(
                    np.ascontiguousarray(arrays["policy_actions"]).tobytes()
                ).hexdigest(),
                "reward_sha256": hashlib.sha256(
                    np.ascontiguousarray(arrays["rewards"]).tobytes()
                ).hexdigest(),
                "replay_validation_sha256": optimal._payload_sha256(
                    record["replay_validation"]
                ),
                "quality_v2_events_by_observation": [[] for _ in range(steps + 1)],
                "issued_equals_applied": self.task != "t5_replan",
                "t5_replan_causal_timing_passed": (
                    True if self.task == "t5_replan" else None
                ),
                "impact_end_to_first_qualifying_applied_correction_s": (
                    0.05 if self.task == "t5_replan" else None
                ),
            }
        )
        audit_record = {**record, "schema_version": optimal_auditor.ATTEMPT_SCHEMA}
        record["quality_score"] = list(optimal._quality_score(audit_record))
        record["eligible"] = optimal._eligible(audit_record)

    def _write_evaluations(self) -> None:
        policy = _seal(
            {
                "schema_version": promotion.POLICY_EVALUATION_SCHEMA,
                "policy_identity": {
                    "path": str(self.policy_path.resolve()),
                    "sha256": self.policy_sha,
                    "schema_version": promotion.POLICY_SCHEMA,
                    "task": self.task,
                    "algorithm": self.config["algorithm"],
                    "training_seed": self.config["seed"],
                    "training_env_steps": 100,
                    "validation": self.validation,
                },
                "source_identity": {
                    "evaluator_rlinf_commit": EVALUATOR_COMMIT,
                    "policy_rlinf_commit": RLINF_COMMIT,
                    "benchmark_commit": BENCHMARK_COMMIT,
                },
                "task_quality_identity": {
                    "evaluator_backend_id": "mujoco311-rs140-v1-rld2-quality",
                    "task_quality_schema": self.schema,
                    "task_quality_schema_sha256": self.schema["schema_sha256"],
                },
                "quality_v2_threshold_identity": {
                    "schema_version": promotion.QUALITY_V2_THRESHOLD_SCHEMA,
                    "sha256": self.threshold_sha,
                },
                "split": "validation",
                "manifest_seed": FORMAL_POLICY_VALIDATION_MANIFEST_SEED,
                "reset_manifest_sha256": self.reset_sha,
                "episodes": promotion.RESET_COUNT,
                "device": "cpu",
                "state_schema": {"state_dim": 2, "mask_dim": 0},
                "records": self.policy_records,
                "task_summary": {self.task: {}},
                "decision_latency": {"p95_meets_20hz": True},
                "all_replays_passed": True,
                "all_successful_quality_v2_gates_passed": True,
                "started_unix_s": 1.0,
                "finished_unix_s": 2.0,
            }
        )
        planner = _seal(
            {
                "schema_version": promotion.PLANNER_EVALUATION_SCHEMA,
                "planner_identity": {
                    "task": self.task,
                    "kind": "privileged_teacher",
                },
                "source_identity": {
                    "evaluator_rlinf_commit": EVALUATOR_COMMIT,
                    "benchmark_commit": BENCHMARK_COMMIT,
                },
                "split": "validation",
                "manifest_seed": FORMAL_POLICY_VALIDATION_MANIFEST_SEED,
                "reset_manifest_sha256": self.reset_sha,
                "episodes": promotion.RESET_COUNT,
                "records": self.planner_records,
                "task_summary": {self.task: {}},
                "decision_latency": {"p95_meets_20hz": True},
                "all_replays_passed": True,
                "started_unix_s": 1.0,
                "finished_unix_s": 2.0,
            }
        )
        _write_json(self.policy_evaluation_path, policy)
        _write_json(self.planner_evaluation_path, planner)

    def build(self) -> dict:
        return promotion.build_selection_evidence(
            candidate_id="learned-s7",
            run_tag="cycle1-s7",
            policy_path=self.policy_path,
            policy_metadata_path=self.metadata_path,
            policy_evaluation_path=self.policy_evaluation_path,
            planner_evaluation_path=self.planner_evaluation_path,
            reset_manifest_path=self.reset_path,
            quality_v2_thresholds_path=self.threshold_path,
            quality_v2_calibration_wave_receipt_path=(self.calibration_receipt_path),
            selector_contract_path=self.selector_path,
            policy_evaluator_source_path=self.policy_evaluator_source,
            planner_evaluator_source_path=self.planner_evaluator_source,
            image_reference="runtime:test",
            image_sha256=IMAGE_SHA256,
        )


@pytest.fixture(autouse=True)
def _canonical_attempt_auditor_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep selection tests focused while proving every tape reaches the auditor."""

    def audit(
        root: Path,
        record: dict,
        *,
        expected_task: str,
        quality_v2_thresholds: dict,
        quality_v2_thresholds_sha256: str,
    ) -> None:
        assert root.is_dir()
        assert record["schema_version"] == optimal_auditor.ATTEMPT_SCHEMA
        assert record["task_id"] == expected_task
        assert quality_v2_thresholds["schema_version"] == (
            promotion.QUALITY_V2_THRESHOLD_SCHEMA
        )
        assert quality_v2_thresholds_sha256

    monkeypatch.setattr(optimal_auditor, "_audit_attempt_tape", audit)

    def validate_receipt(
        thresholds: dict,
        receipt_path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> optimal.ProvenanceFile:
        actual = promotion._sha256(receipt_path)
        binding = thresholds["calibration_wave_receipt"]
        if (
            expected_sha256 != actual
            or binding["file_sha256"] != actual
            or binding["payload_sha256"] != actual
        ):
            raise ValueError("calibration wave receipt SHA-256 mismatch")
        return optimal.ProvenanceFile(
            source_path=receipt_path.resolve(),
            relative_path=binding["relative_path"],
            sha256=actual,
        )

    monkeypatch.setattr(
        optimal, "_validate_quality_v2_calibration_receipt_artifact", validate_receipt
    )


def test_happy_path_promotes_and_review_recomputation_reopens_every_artifact(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    evidence = fixture.build()
    evidence_path = tmp_path / "selection_evidence.json"
    receipt_path = tmp_path / "promotion_receipt.json"

    receipt = promotion.write_promotion_artifacts(
        evidence=evidence,
        evidence_path=evidence_path,
        receipt_path=receipt_path,
        require_promote=True,
    )

    assert evidence["selection"]["decision"] == "promote"
    assert evidence["aggregate"]["counts"]["both_success"] == 20
    assert evidence["aggregate"]["success"]["policy"]["wilson_95"]["low"] > 0.8
    assert receipt["schema_version"] == promotion.PROMOTION_SCHEMA
    assert receipt["selection"]["evidence_sha256"] == promotion._sha256(evidence_path)
    assert (
        receipt["validation_receipt"]["attempt_schema_version"]
        == optimal_auditor.ATTEMPT_SCHEMA
    )
    assert receipt["selection"]["planner_evaluation_sha256"] == promotion._sha256(
        fixture.planner_evaluation_path
    )
    assert receipt["quality_v2_calibration_wave_receipt"]["sha256"] == (
        fixture.calibration_receipt_sha
    )
    assert (
        receipt["quality_v2_calibration_wave_receipt"]["dataset_relative_path"]
        == "provenance/calibration_wave/wave_receipt.json"
    )
    assert promotion.validate_selection_evidence_artifacts(evidence) == evidence
    tape = evidence["per_reset"][0]["policy"]["attempt_tape"]
    assert tape["path"] == "tapes/policy-00.npz"
    assert len(tape["sha256"]) == 64
    assert len(tape["payload_sha256"]) == 64
    with pytest.raises(FileExistsError, match="overwrite"):
        promotion.write_promotion_artifacts(
            evidence=evidence,
            evidence_path=evidence_path,
            receipt_path=receipt_path,
            require_promote=True,
        )


def test_formal_policy_validation_is_independent_from_checkpoint_selection(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)

    evidence = fixture.build()

    assert (
        fixture.config["validation_manifest_seed"]
        == CHECKPOINT_SELECTION_VALIDATION_MANIFEST_SEED
    )
    assert evidence["inputs"]["reset_manifest"]["manifest_seed"] == (
        FORMAL_POLICY_VALIDATION_MANIFEST_SEED
    )


def test_promotion_rejects_checkpoint_selection_manifest_reuse(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    for evaluation_path in (
        fixture.policy_evaluation_path,
        fixture.planner_evaluation_path,
    ):
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["manifest_seed"] = fixture.config["validation_manifest_seed"]
        evaluation["payload_sha256"] = promotion._payload_sha256(evaluation)
        _write_json(evaluation_path, evaluation)

    with pytest.raises(ValueError, match="checkpoint-selection validation manifest"):
        fixture.build()


def test_every_evaluator_tape_is_sent_to_the_canonical_auditor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path)
    calls: list[tuple[Path, str, str]] = []

    def audit(root: Path, record: dict, *, expected_task: str, **_: object) -> None:
        calls.append((root, record["schema_version"], expected_task))

    monkeypatch.setattr(optimal_auditor, "_audit_attempt_tape", audit)
    fixture.build()

    assert calls == [
        (tmp_path.resolve(), optimal_auditor.ATTEMPT_SCHEMA, TASK)
        for _ in range(2 * promotion.RESET_COUNT)
    ]


def test_t5_causal_gate_issued_history_and_latency_are_hard_requirements(
    tmp_path: Path,
) -> None:
    false_gate = Fixture(tmp_path / "false-gate", task="t5_replan")
    false_gate.policy_records[0]["t5_replan_causal_timing_passed"] = False
    false_gate.policy_records[0][
        "impact_end_to_first_qualifying_applied_correction_s"
    ] = None
    false_gate._write_evaluations()
    with pytest.raises(ValueError, match="T5 causal gate"):
        false_gate.build()

    wrong_history = Fixture(tmp_path / "issued", task="t5_replan")
    wrong_history.policy_records[0]["issued_equals_applied"] = True
    wrong_history._write_evaluations()
    with pytest.raises(ValueError, match="distinct issued/applied"):
        wrong_history.build()

    latency_regression = Fixture(tmp_path / "latency", task="t5_replan")
    latency_regression.policy_records[0][
        "impact_end_to_first_qualifying_applied_correction_s"
    ] = 0.06
    latency_regression._write_evaluations()
    with pytest.raises(ValueError, match="degradation"):
        latency_regression.build()


def test_t5_causal_latency_can_supply_the_required_strict_improvement(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path, tie=True, task="t5_replan")
    fixture.policy_records[0]["impact_end_to_first_qualifying_applied_correction_s"] = (
        0.04
    )
    fixture._write_evaluations()

    evidence = fixture.build()

    assert evidence["selection"]["decision"] == "promote"
    assert (
        "causal.impact_end_to_first_qualifying_applied_correction_s"
        in evidence["selection"]["strict_improvement_dimensions"]
    )


def test_tampered_attempt_tape_fails_before_selection(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    tape_path = fixture.root / fixture.policy_records[0]["attempt_tape"]
    tape_path.write_bytes(tape_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="attempt tape.*SHA-256 mismatch"):
        fixture.build()


def test_tampered_calibration_wave_receipt_fails_before_selection(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.calibration_receipt_path.write_bytes(
        fixture.calibration_receipt_path.read_bytes() + b"\n"
    )

    with pytest.raises(ValueError, match="calibration wave receipt SHA-256 mismatch"):
        fixture.build()


def test_exact_tie_keeps_planner_and_require_promote_writes_nothing(
    tmp_path: Path,
) -> None:
    evidence = Fixture(tmp_path, tie=True).build()
    evidence_path = tmp_path / "selection_evidence.json"
    receipt_path = tmp_path / "promotion_receipt.json"

    assert evidence["selection"] == {
        "decision": "keep_planner",
        "planner_nonworse_all_dimensions": True,
        "strict_improvement_dimensions": [],
        "exact_aggregate_tie": True,
        "rule": evidence["selection"]["rule"],
    }
    with pytest.raises(RuntimeError, match="require-promote"):
        promotion.write_promotion_artifacts(
            evidence=evidence,
            evidence_path=evidence_path,
            receipt_path=receipt_path,
            require_promote=True,
        )
    assert not evidence_path.exists()
    assert not receipt_path.exists()


def test_safety_and_success_regressions_fail_closed(tmp_path: Path) -> None:
    safety = Fixture(tmp_path / "safety")
    safety.root.mkdir(exist_ok=True)
    safety.policy_records[0]["success"] = False
    safety.policy_records[0]["safety_failure"] = True
    safety.policy_records[0]["completion_time_s"] = None
    safety.planner_records[0]["success"] = False
    safety.planner_records[0]["completion_time_s"] = None
    safety._write_evaluations()
    with pytest.raises(ValueError, match="safety failures exceed"):
        safety.build()

    regression = Fixture(tmp_path / "success")
    regression.policy_records[0]["success"] = False
    regression.policy_records[0]["completion_time_s"] = None
    regression._write_evaluations()
    with pytest.raises(ValueError, match="planner-success to policy-failure"):
        regression.build()


@pytest.mark.parametrize(
    "identity",
    [
        ("full_episode", "action.action_total_variation_l2_mean_per_transition"),
        ("full_episode", "eef_motion.eef_translation_path_length_m"),
        ("acquisition_window", "approach_axis.approach_axis_error_max_rad"),
        ("full_episode", "eef_motion.eef_linear_jerk_max_m_s3"),
    ],
)
def test_qv3_tv_path_orientation_and_jerk_degradation_fail_closed(
    tmp_path: Path, identity: tuple[str, str]
) -> None:
    fixture = Fixture(tmp_path)
    policy = fixture.policy_records[0]
    quality = policy["quality_v2"]
    target = (
        quality if identity[0] == "full_episode" else quality["phases"][identity[0]]
    )
    _set_nested(target, identity[1], 1.5)
    policy["quality_v2_sha256"] = promotion._payload_sha256(quality)
    policy["quality_v2_gate"] = _gate(quality, fixture.checks, fixture.threshold_sha)
    fixture._write_evaluations()

    with pytest.raises(ValueError, match="degradation"):
        fixture.build()


def test_missing_or_tampered_evidence_evaluation_reset_and_sha_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    evidence = fixture.build()
    evidence_path = tmp_path / "selection_evidence.json"
    receipt_path = tmp_path / "promotion_receipt.json"
    receipt = promotion.write_promotion_artifacts(
        evidence=evidence,
        evidence_path=evidence_path,
        receipt_path=receipt_path,
        require_promote=True,
    )
    assert receipt["selection"]["evidence_sha256"] == promotion._sha256(evidence_path)

    fixture.policy_evaluation_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        promotion.validate_selection_evidence_artifacts(evidence)
    fixture._write_evaluations()
    fixture.reset_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        promotion.validate_selection_evidence_artifacts(evidence)

    tampered = deepcopy(evidence)
    tampered["selection"]["decision"] = "keep_planner"
    with pytest.raises(ValueError, match="payload SHA-256"):
        promotion.validate_selection_evidence_artifacts(tampered)
    with pytest.raises(ValueError, match="policy SHA-256 mismatch"):
        promotion.build_selection_evidence(
            candidate_id="learned-s7",
            run_tag="cycle1-s7",
            policy_path=fixture.policy_path,
            policy_metadata_path=fixture.metadata_path,
            policy_evaluation_path=fixture.policy_evaluation_path,
            planner_evaluation_path=fixture.planner_evaluation_path,
            reset_manifest_path=fixture.reset_path,
            quality_v2_thresholds_path=fixture.threshold_path,
            quality_v2_calibration_wave_receipt_path=(fixture.calibration_receipt_path),
            selector_contract_path=fixture.selector_path,
            policy_evaluator_source_path=fixture.policy_evaluator_source,
            planner_evaluator_source_path=fixture.planner_evaluator_source,
            image_reference="runtime:test",
            image_sha256=IMAGE_SHA256,
            expected_sha256={
                "policy": "0" * 64,
                "policy_metadata": promotion._sha256(fixture.metadata_path),
                "policy_evaluation": promotion._sha256(fixture.policy_evaluation_path),
                "planner_evaluation": promotion._sha256(
                    fixture.planner_evaluation_path
                ),
                "reset_manifest": promotion._sha256(fixture.reset_path),
                "quality_v2_thresholds": promotion._sha256(fixture.threshold_path),
                "selector_contract": promotion._sha256(fixture.selector_path),
                "policy_evaluator_source": promotion._sha256(
                    fixture.policy_evaluator_source
                ),
                "planner_evaluator_source": promotion._sha256(
                    fixture.planner_evaluator_source
                ),
            },
        )


def test_provisional_thresholds_are_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.thresholds["formal_freeze_eligible"] = False
    _write_json(fixture.threshold_path, fixture.thresholds)
    with pytest.raises(ValueError, match="formal freeze"):
        fixture.build()


def test_promotion_rejects_same_count_qv3_identity_substitution() -> None:
    thresholds = _thresholds()
    next(
        check
        for check in thresholds["tasks"][TASK]["checks"]
        if check["metric"] == "eef_motion.eef_linear_jerk_rms_m_s3"
    )["metric"] = "eef_motion.eef_linear_jerk_peak_m_s3"

    with pytest.raises(ValueError, match="inventory mismatch"):
        promotion._quality_contract(thresholds, task=TASK, sha256="f" * 64)


def test_zero_step_final_and_test_exposure_are_rejected(tmp_path: Path) -> None:
    zero = Fixture(tmp_path / "zero")
    payload = torch.load(zero.policy_path, map_location="cpu", weights_only=False)
    payload["env_steps"] = 0
    torch.save(payload, zero.policy_path)
    zero.policy_sha = promotion._sha256(zero.policy_path)
    zero._write_evaluations()
    with pytest.raises(ValueError, match="env_steps"):
        zero.build()

    final = Fixture(tmp_path / "final")
    final_policy = final.root / "final_policy.pt"
    final.policy_path.rename(final_policy)
    final.policy_path = final_policy
    final.policy_sha = promotion._sha256(final_policy)
    final._write_evaluations()
    with pytest.raises(ValueError, match="best_policy"):
        final.build()

    exposed = Fixture(tmp_path / "test")
    evaluation = json.loads(exposed.policy_evaluation_path.read_text(encoding="utf-8"))
    evaluation["split"] = "test_id"
    evaluation["payload_sha256"] = promotion._payload_sha256(evaluation)
    _write_json(exposed.policy_evaluation_path, evaluation)
    with pytest.raises(ValueError, match="validation split"):
        exposed.build()

    calibration = Fixture(tmp_path / "calibration")
    evaluation = json.loads(
        calibration.policy_evaluation_path.read_text(encoding="utf-8")
    )
    evaluation["manifest_seed"] = promotion.CALIBRATION_MANIFEST_SEED
    evaluation["payload_sha256"] = promotion._payload_sha256(evaluation)
    _write_json(calibration.policy_evaluation_path, evaluation)
    with pytest.raises(ValueError, match="review/calibration/test manifest seed"):
        calibration.build()

    training_exposed = Fixture(tmp_path / "training-exposed")
    checkpoint = torch.load(
        training_exposed.policy_path, map_location="cpu", weights_only=False
    )
    checkpoint["config"]["validation_manifest_seed"] = promotion.REVIEW_MANIFEST_SEED
    torch.save(checkpoint, training_exposed.policy_path)
    training_exposed.config = checkpoint["config"]
    training_exposed.policy_sha = promotion._sha256(training_exposed.policy_path)
    summary = json.loads(training_exposed.metadata_path.read_text(encoding="utf-8"))
    summary["config"] = training_exposed.config
    summary["config_sha256"] = promotion._value_sha256(training_exposed.config)
    summary["payload_sha256"] = promotion._payload_sha256(summary)
    _write_json(training_exposed.metadata_path, summary)
    training_exposed._write_evaluations()
    with pytest.raises(ValueError, match="checkpoint selection reused"):
        training_exposed.build()
