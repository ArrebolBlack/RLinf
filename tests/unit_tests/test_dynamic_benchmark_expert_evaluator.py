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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.embodiment.dynamic_benchmark_evaluation_attempt import (
    attempt_schema_version,
    materialize_evaluation_attempt,
    recursive_output_checksums,
    validate_formal_quality_v2_thresholds,
)
from examples.embodiment.evaluate_dynamic_benchmark_expert import (
    EVALUATION_SCHEMA,
    FORMAL_EVALUATION_SCHEMA,
    TASK_QUALITY_BACKEND_ID,
    _aggregate_task_quality_values,
    _ArmedResetReplayEnv,
    _evaluation_schema,
    _expected_sha256,
    _latency_summary,
    _load_inference_policy,
    _model_kwargs,
    _replay_actions_on_fresh_env,
    _reset_rollout_on_fresh_env,
    _task_quality_aggregates,
    _task_quality_env_config,
    _task_quality_from_terminal_infos,
    _task_quality_identity,
    _task_summary,
    _validate_policy_payload,
)


def _payload(algorithm: str = "rlpd") -> dict:
    return {
        "schema_version": "rlinf-dynamic-benchmark-expert-policy-v0.1",
        "config": {
            "task": "t1_xyz",
            "algorithm": algorithm,
            "rlinf_commit": "a" * 40,
            "benchmark_commit": "b" * 40,
            "q_heads": 10,
        },
        "state_schema": {"state_dim": 173, "mask_dim": 35},
        "model": {"fixture": 1},
        "normalizer": {"fixture": 2},
    }


def _quality_summary(
    episode_id: str,
    *,
    values: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    task_id: str = "t1_xyz",
    backend_id: str = TASK_QUALITY_BACKEND_ID,
    terminal: bool = True,
) -> dict:
    from se3_wam.benchmark.task_quality import (
        EpisodeQualitySummary,
        TaskQualityComponentValue,
        get_task_quality_schema,
    )

    schema = get_task_quality_schema(task_id)
    assert len(values) == len(schema.components)
    summary = EpisodeQualitySummary(
        episode_id=episode_id,
        task_id=task_id,
        evaluator_backend_id=backend_id,
        schema_version=schema.schema_version,
        schema_sha256=schema.schema_sha256,
        physics_sample_count=10,
        terminal=terminal,
        components={
            spec.name: TaskQualityComponentValue(
                value=value,
                direction=spec.direction,
                unit=spec.unit,
                scientific_resolution=spec.scientific_resolution,
                reducer=spec.reducer,
            )
            for spec, value in zip(schema.components, values, strict=True)
        },
    )
    return summary.to_dict()


def test_task_quality_config_covers_canonical_exact_14_inventory() -> None:
    from se3_wam.benchmark.registry import RL_EXPERT_TASK_IDS

    assert len(RL_EXPERT_TASK_IDS) == 14
    for task_id in RL_EXPERT_TASK_IDS:
        identity = _task_quality_identity(task_id)

        config = _task_quality_env_config(identity)

        assert config == {
            "task_quality_schema_version": identity["task_quality_schema"][
                "schema_version"
            ],
            "task_quality_evaluator_backend_id": TASK_QUALITY_BACKEND_ID,
        }
        assert (
            identity["task_quality_schema_sha256"]
            == identity["task_quality_schema"]["schema_sha256"]
        )


def test_terminal_task_quality_is_validated_and_recorded_canonically() -> None:
    identity = _task_quality_identity("t1_xyz")
    expected = _quality_summary("episode-1")

    recorded = _task_quality_from_terminal_infos(
        {"task_quality": [expected]},
        identity=identity,
        task_id="t1_xyz",
        episode_id="episode-1",
    )

    assert recorded == expected
    assert recorded["terminal"] is True
    assert recorded["schema_sha256"] == identity["task_quality_schema_sha256"]


def test_task_quality_aggregates_all_and_successful_records() -> None:
    identity = _task_quality_identity("t1_xyz")
    records = [
        {
            "episode_id": "episode-1",
            "task_id": "t1_xyz",
            "success": True,
            "safety_failure": False,
            "termination_reason": "success",
            "trajectory_completion": 1.0,
            "completion_time_s": 1.0,
            "return": 2.0,
            "action_l2_sum": 3.0,
            "task_quality": _quality_summary("episode-1", values=(0.1, 0.2, 0.3, 0.4)),
        },
        {
            "episode_id": "episode-2",
            "task_id": "t1_xyz",
            "success": False,
            "safety_failure": True,
            "termination_reason": "collision",
            "trajectory_completion": 0.5,
            "completion_time_s": None,
            "return": 1.0,
            "action_l2_sum": 2.0,
            "task_quality": _quality_summary("episode-2", values=(0.3, 0.4, 0.5, 0.6)),
        },
    ]

    task_summary = _task_summary(
        "t1_xyz",
        records,
        task_quality_identity=identity,
        quality_v2_enabled=False,
    )
    aggregates = task_summary["t1_xyz"]["task_quality"]

    assert set(task_summary) == {"t1_xyz"}
    component_names = [
        component["name"] for component in identity["task_quality_schema"]["components"]
    ]
    first = aggregates["components"][component_names[0]]
    assert aggregates["record_count"] == 2
    assert aggregates["successful_record_count"] == 1
    assert first["all_records"] == {
        "count": 2,
        "mean": pytest.approx(0.2),
        "minimum": pytest.approx(0.1),
        "maximum": pytest.approx(0.3),
    }
    assert first["successful_records"] == {
        "count": 1,
        "mean": pytest.approx(0.1),
        "minimum": pytest.approx(0.1),
        "maximum": pytest.approx(0.1),
    }
    assert _aggregate_task_quality_values([]) == {
        "count": 0,
        "mean": None,
        "minimum": None,
        "maximum": None,
    }
    assert (
        _task_quality_aggregates(
            records,
            identity=identity,
            task_id="t1_xyz",
        )
        == aggregates
    )


def test_task_quality_fails_closed_on_missing_or_mismatched_terminal_summary() -> None:
    identity = _task_quality_identity("t1_xyz")
    with pytest.raises(ValueError, match="missing task_quality"):
        _task_quality_from_terminal_infos(
            {},
            identity=identity,
            task_id="t1_xyz",
            episode_id="episode-1",
        )
    with pytest.raises(ValueError, match="no task-quality"):
        _task_quality_from_terminal_infos(
            {"task_quality": [None]},
            identity=identity,
            task_id="t1_xyz",
            episode_id="episode-1",
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        _task_quality_from_terminal_infos(
            {"task_quality": [_quality_summary("other-episode")]},
            identity=identity,
            task_id="t1_xyz",
            episode_id="episode-1",
        )

    tampered_identity = copy.deepcopy(identity)
    tampered_identity["task_quality_schema"]["task_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schema is not canonical"):
        _task_quality_env_config(tampered_identity)


def test_model_kwargs_reconstruct_sac_and_ppo_heads() -> None:
    sac = _model_kwargs(_payload()["config"], 173)
    ppo = _model_kwargs(_payload("ppo")["config"], 173)

    assert sac["add_q_head"] is True
    assert sac["add_value_head"] is False
    assert sac["num_q_heads"] == 10
    assert ppo["add_q_head"] is False
    assert ppo["add_value_head"] is True


def test_policy_payload_validation_is_commit_and_schema_fail_closed() -> None:
    payload = _payload()

    config, schema = _validate_policy_payload(
        payload,
        rlinf_commit="a" * 40,
        benchmark_commit="b" * 40,
    )

    assert config["task"] == "t1_xyz"
    assert schema["state_dim"] == 173
    with pytest.raises(ValueError, match="RLinf commit"):
        _validate_policy_payload(
            payload,
            rlinf_commit="c" * 40,
            benchmark_commit="b" * 40,
        )
    payload["schema_version"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        _validate_policy_payload(
            payload,
            rlinf_commit="a" * 40,
            benchmark_commit="b" * 40,
        )


def test_latency_summary_reports_20hz_p95_gate() -> None:
    fast = _latency_summary([0.001, 0.002, 0.003])
    slow = _latency_summary([0.001, 0.002, 0.100])

    assert fast["p95_meets_20hz"] is True
    assert slow["p95_meets_20hz"] is False
    assert fast["sample_count"] == 3


def test_expected_sha256_requires_full_lowercase_hex() -> None:
    assert _expected_sha256("a" * 64) == "a" * 64
    with pytest.raises(ValueError, match="64 lowercase"):
        _expected_sha256("abc")


def _actor_state(state_dim: int, *, ppo: bool) -> dict[str, torch.Tensor]:
    state = {
        "backbone.0.weight": torch.randn(256, state_dim),
        "backbone.0.bias": torch.randn(256),
        "backbone.2.weight": torch.randn(256, 256),
        "backbone.2.bias": torch.randn(256),
        "backbone.4.weight": torch.randn(256, 256),
        "backbone.4.bias": torch.randn(256),
        "actor_mean.weight": torch.randn(7, 256),
        "actor_mean.bias": torch.randn(7),
    }
    if ppo:
        state["actor_logstd"] = torch.randn(1, 7)
        state["value_head.net.0.weight"] = torch.randn(256, state_dim)
    else:
        state["actor_logstd.weight"] = torch.randn(7, 256)
        state["actor_logstd.bias"] = torch.randn(7)
        state["action_scale"] = torch.tensor(1.0)
        state["action_bias"] = torch.tensor(0.0)
        for index in range(10):
            state[f"q_head.qs.{index}.net.0.weight"] = torch.randn(256, state_dim + 7)
    return state


@pytest.mark.parametrize("algorithm", ["rlpd", "ppo"])
def test_inference_policy_reconstructs_actor_without_training_stack(
    algorithm: str,
) -> None:
    config = _payload(algorithm)["config"]
    state = _actor_state(173, ppo=algorithm == "ppo")

    model = _load_inference_policy(config, 173, state, torch.device("cpu"))
    mean, log_std = model._sample_actions(torch.zeros(2, 173))

    assert mean.shape == (2, 7)
    assert log_std.shape == (2, 7)
    assert model.training is False


def test_inference_policy_rejects_unknown_or_incomplete_heads() -> None:
    config = _payload()["config"]
    state = _actor_state(173, ppo=False)
    state["mystery.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="unsupported"):
        _load_inference_policy(config, 173, state, torch.device("cpu"))

    state = _actor_state(173, ppo=False)
    del state["q_head.qs.9.net.0.weight"]
    with pytest.raises(ValueError, match="Q-head count"):
        _load_inference_policy(config, 173, state, torch.device("cpu"))


def test_expert_rollout_and_replay_use_separate_fresh_raw_envs() -> None:
    class Request:
        task_id = "t5_replan"

    request = Request()

    class RawEnv:
        horizon_steps = 120

        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False
            self.reset_calls = []

        def reset(self, value):
            self.reset_calls.append(value)
            return f"{self.name}-observation"

        def step(self, action):
            return (self.name, action)

        def save_state(self):
            return self.name.encode()

        def close(self):
            self.closed = True

    class VectorEnv:
        num_envs = 1
        task_id = "t5_replan"
        image_size = 64
        camera_observations = False
        task_quality_schema_version = "quality-v1"
        task_quality_evaluator_backend_id = "backend-v1"
        horizon_steps = 120

        def __init__(self) -> None:
            self.previous = RawEnv("previous")
            self.rollout = RawEnv("rollout")
            self.replay = RawEnv("replay")
            self.envs = [self.previous]
            self._raw_observations = ["stale"]
            self._requests = [object()]
            self._needs_reset = np.asarray([True])
            self._last_obs = None
            self._is_start = False
            self.make_calls = []
            self.arm_calls = []
            self.reset_metric_calls = []

        def _make_mujoco_env(self, task_id, **kwargs):
            self.make_calls.append((task_id, kwargs))
            return self.rollout if len(self.make_calls) == 1 else self.replay

        def _arm_hidden_t5_event(self, raw_env, value):
            self.arm_calls.append((raw_env, value))

        def _encode(self, observation, value):
            assert observation == "rollout-observation"
            assert value is request
            return np.asarray([1.0, 2.0], dtype=np.float32)

        def _reset_metrics(self, indices):
            self.reset_metric_calls.append(indices.copy())

    vector_env = VectorEnv()
    observation = _reset_rollout_on_fresh_env(
        vector_env=vector_env,
        request=request,
    )

    assert observation == "rollout-observation"
    assert vector_env.previous.closed
    assert vector_env.envs == [vector_env.rollout]
    assert vector_env.rollout.reset_calls == [request]
    assert not vector_env._needs_reset[0]
    assert vector_env._last_obs["states"].tolist() == [[1.0, 2.0]]

    def replay_fn(proxy, **kwargs):
        assert isinstance(proxy, _ArmedResetReplayEnv)
        assert proxy.reset(request) == "replay-observation"
        assert proxy.step("action") == ("replay", "action")
        assert proxy.save_state() == b"replay"
        assert kwargs["request"] is request
        return {"passed": True}

    result = _replay_actions_on_fresh_env(
        vector_env=vector_env,
        task_id="t5_replan",
        request=request,
        expected_observations=("expected",),
        actions=("action",),
        expected_outcomes=("outcome",),
        expected_final_state=b"expected",
        replay_fn=replay_fn,
    )

    assert result == {"passed": True}
    assert vector_env.make_calls == [
        (
            "t5_replan",
            {
                "image_size": 64,
                "camera_observations": False,
                "task_quality_schema_version": "quality-v1",
                "task_quality_evaluator_backend_id": "backend-v1",
            },
        ),
        (
            "t5_replan",
            {
                "image_size": 64,
                "camera_observations": False,
                "task_quality_schema_version": "quality-v1",
                "task_quality_evaluator_backend_id": "backend-v1",
            },
        ),
    ]
    assert vector_env.replay.closed
    assert vector_env.arm_calls == [
        (vector_env.rollout, request),
        (vector_env.replay, request),
    ]


@pytest.mark.parametrize(
    "replay_values, expected_passed",
    [
        ((0.1, 0.2, 0.3, 0.4), True),
        ((0.3, 0.4, 0.5, 0.6), False),
    ],
)
def test_expert_replay_binds_terminal_task_quality_exactly(
    replay_values: tuple[float, ...], expected_passed: bool
) -> None:
    identity = _task_quality_identity("t1_xyz")
    expected = _quality_summary("episode-1", values=(0.1, 0.2, 0.3, 0.4))

    class Request:
        episode_id = "episode-1"
        task_id = "t1_xyz"

    request = Request()

    class RawEnv:
        def reset(self, value):
            assert value is request
            return "observation"

        def step(self, action):
            assert action == "action"
            return SimpleNamespace(
                terminated=True,
                truncated=False,
                task_quality=_quality_summary(
                    "episode-1",
                    values=replay_values,
                ),
            )

        def save_state(self):
            return b"state"

        def close(self):
            pass

    class VectorEnv:
        image_size = 64
        camera_observations = False
        task_quality_schema_version = identity["task_quality_schema"]["schema_version"]
        task_quality_evaluator_backend_id = TASK_QUALITY_BACKEND_ID

        def _make_mujoco_env(self, task_id, **kwargs):
            assert task_id == "t1_xyz"
            return RawEnv()

        def _arm_hidden_t5_event(self, raw_env, value):
            pass

    def replay_fn(proxy, **kwargs):
        proxy.reset(request)
        proxy.step("action")
        return {"passed": True, "final_state_exact": True, "outcomes_exact": True}

    validation = _replay_actions_on_fresh_env(
        vector_env=VectorEnv(),
        task_id="t1_xyz",
        request=request,
        expected_observations=("observation",),
        actions=("action",),
        expected_outcomes=((True, False, False, None, 0.0),),
        expected_final_state=b"state",
        expected_task_quality=expected,
        task_quality_identity=identity,
        replay_fn=replay_fn,
    )

    assert validation["task_quality_exact"] is expected_passed
    assert validation["passed"] is expected_passed
    assert (
        validation["task_quality_summary_sha256"]
        == _quality_summary("episode-1", values=replay_values)["summary_sha256"]
    )


def test_formal_producer_tape_passes_promotion_validator(tmp_path: Path) -> None:
    from se3_wam.benchmark.config import load_task_config
    from se3_wam.benchmark.trajectory_quality import (
        evaluate_quality_v2_gate,
        trajectory_quality_v2_from_observations,
    )

    from examples.embodiment import build_dynamic_benchmark_rld2_promotion as promotion

    task_id = "t1_xyz"
    episode_id = "validation-00"
    steps = 4
    assert _evaluation_schema(formal_attempts=False) == EVALUATION_SCHEMA
    assert _evaluation_schema(formal_attempts=True) == FORMAL_EVALUATION_SCHEMA
    assert EVALUATION_SCHEMA == "rlinf-dynamic-benchmark-expert-evaluation-v0.2"
    assert FORMAL_EVALUATION_SCHEMA == "rlinf-dynamic-benchmark-expert-evaluation-v0.3"
    observations = [
        SimpleNamespace(
            privileged={
                "eef_pose_xyzw": np.asarray(
                    [0.01 * index, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float64,
                ),
                "fingerpad_closing_axis_world": np.asarray(
                    [1.0, 0.0, 0.0], dtype=np.float64
                ),
                "object_pose_wxyz": np.asarray(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                ),
                "fingerpad_contact_flags": np.asarray(
                    [1.0, 1.0] if index >= 1 else [0.0, 0.0],
                    dtype=np.float64,
                ),
            },
            events_since_last_observation=(),
        )
        for index in range(steps + 1)
    ]
    actions = np.zeros((steps, 7), dtype=np.float64)
    quality = trajectory_quality_v2_from_observations(
        observations,
        actions,
        task_id=task_id,
        task_config=load_task_config(task_id),
        sample_period_s=0.05,
        continuous_dimensions=6,
    )

    metric_specs = (
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

    def metric_value(phase: str, path: str) -> float:
        value = quality if phase == "full_episode" else quality["phases"][phase]
        for part in path.split("."):
            value = value[part]
        return float(value)

    checks = [
        {
            "phase": phase,
            "metric": path,
            "max": metric_value(phase, path) + 1.0,
            "direction": "minimize",
            "paired_comparison_family": family,
            "paired_nonworse_absolute_tolerance": 0.01,
            "paired_nonworse_relative_tolerance": 0.0,
            "paired_strict_improvement_absolute": 0.02,
            "paired_strict_improvement_relative": 0.0,
        }
        for phase, path, family in metric_specs
    ]
    thresholds = {
        "schema_version": "se3-wam-trajectory-quality-v2-thresholds-v0.3",
        "formal_freeze_eligible": True,
        "calibration_status": "frozen",
        "minimum_attempted_episodes": 20,
        "minimum_successful_episodes": 8,
        "calibration_wave_receipt": {
            "binding_status": "bound",
            "schema_version": "rld2-qa-planner-calibration-wave-receipt-v0.1",
            "scientific_partition": "metric_calibration",
            "transport_split": "validation",
            "task_count": 14,
            "episodes_per_task": 20,
            "total_reset_count": 280,
            "relative_path": "provenance/calibration/wave_receipt.json",
            "file_sha256": "b" * 64,
            "payload_sha256": "b" * 64,
            "sha256": "b" * 64,
        },
        "tasks": {
            task_id: {
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
    threshold_sha256 = "a" * 64
    contract = validate_formal_quality_v2_thresholds(
        thresholds,
        task_id=task_id,
        thresholds_sha256=threshold_sha256,
    )
    assert contract["formal_freeze_eligible"] is True
    wrong_phase = copy.deepcopy(thresholds)
    next(
        check
        for check in wrong_phase["tasks"][task_id]["checks"]
        if check["metric"] == "eef_motion.eef_angular_jerk_rms_rad_s3"
    )["phase"] = "acquisition_window"
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate_formal_quality_v2_thresholds(
            wrong_phase,
            task_id=task_id,
            thresholds_sha256=threshold_sha256,
        )
    provisional = copy.deepcopy(thresholds)
    provisional["formal_freeze_eligible"] = False
    with pytest.raises(ValueError, match="not eligible for formal freeze"):
        validate_formal_quality_v2_thresholds(
            provisional,
            task_id=task_id,
            thresholds_sha256=threshold_sha256,
        )
    gate = evaluate_quality_v2_gate(quality, thresholds, task_id=task_id)
    gate["contract_sha256"] = threshold_sha256
    replay = {
        "passed": True,
        "final_state_exact": True,
        "outcomes_exact": True,
        "task_quality_exact": True,
    }
    record = {
        "episode_id": episode_id,
        "task_id": task_id,
        "success": True,
        "safety_failure": False,
        "termination_reason": "success",
        "trajectory_completion": 1.0,
        "completion_time_s": 1.0,
        "return": 1.0,
        "control_steps": steps,
        "action_l2_sum": 0.0,
        "task_quality": _quality_summary(episode_id),
        "actions": actions.tolist(),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest(),
        "quality_v2": quality,
        "quality_v2_sha256": promotion._payload_sha256(quality),
        "quality_v2_gate": gate,
        "replay_validation": replay,
        "events": ["success"],
    }
    output = tmp_path / "evaluation"
    output.mkdir()

    produced = materialize_evaluation_attempt(
        output,
        record,
        candidate_index=0,
        raw_env=SimpleNamespace(),
        observations=observations,
        states=[np.zeros(2, dtype=np.float32) for _ in range(steps + 1)],
        policy_actions=[np.zeros(7, dtype=np.float32) for _ in range(steps)],
        rewards=[0.25] * steps,
        terminated=[False, False, False, True],
        truncated=[False] * steps,
        quality_v2_thresholds_sha256=threshold_sha256,
    )
    sealed = promotion._audit_evaluation_attempt(
        produced,
        evaluation_path=output / "evaluation.json",
        task=task_id,
        quality_v2_thresholds=thresholds,
        quality_v2_thresholds_sha256=threshold_sha256,
        label="producer fixture",
    )

    assert produced["attempt_schema_version"] == attempt_schema_version()
    assert produced["actions"] == actions.tolist()
    assert np.asarray(produced["actions"]).dtype == np.float64
    assert sealed["path"] == produced["attempt_tape"]
    assert sealed["sha256"] == produced["attempt_tape_sha256"]
    assert produced["eligible"] is True
    checksum_paths = [
        line.split("  ", 1)[1]
        for line in recursive_output_checksums(output).splitlines()
    ]
    assert produced["attempt_tape"] in checksum_paths
