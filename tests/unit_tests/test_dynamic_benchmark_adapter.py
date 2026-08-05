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

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs import SupportedEnvType
from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv
from rlinf.envs.dynamic_benchmark.reward import DynamicBenchmarkReward
from rlinf.envs.dynamic_benchmark.state_schema import (
    ALLOWED_PRIVILEGED_KEYS,
    DynamicBenchmarkStateSchema,
)
from rlinf.envs.dynamic_benchmark.t5_runtime import (
    arm_hidden_t5_event,
    t5_branch_for_episode,
)


def _observation() -> SimpleNamespace:
    return SimpleNamespace(
        task_id="t1_so3",
        policy_step=5,
        proprio={"robot0_proprio_state": np.asarray([0.1, -0.2])},
        privileged={
            "object_pose_wxyz": np.asarray([1.0, 2.0, 3.0, -1.0, 0.0, 0.0, 0.0]),
            "eef_pose_xyzw": np.asarray([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, -1.0]),
            "future_object_pose_wxyz": np.ones(7),
        },
    )


def test_dynamic_benchmark_env_type_is_registered() -> None:
    assert SupportedEnvType("dynamic_benchmark") is SupportedEnvType.DYNAMIC_BENCHMARK


def test_manifest_cache_round_trip_validates_identity_without_regeneration() -> None:
    class FakeSplit:
        def __init__(self, value: str) -> None:
            if value not in {"train", "validation"}:
                raise ValueError(value)
            self.value = value

    env = object.__new__(DynamicBenchmarkEnv)
    env.task_id = "t4_sphere"
    env.split_name = "train"
    env._split = FakeSplit("train")
    env._Split = FakeSplit
    env.base_manifest_seed = 17
    env.manifest_size = 2
    env._manifest_generation = 3
    env._manifest_cursor = 1
    rows = tuple(
        SimpleNamespace(
            request=SimpleNamespace(
                task_id="t4_sphere",
                split=SimpleNamespace(value="train"),
            )
        )
        for _ in range(2)
    )
    env._manifest_rows = rows

    cache = env.manifest_cache_state()
    env.split_name = "validation"
    env._split = FakeSplit("validation")
    env.base_manifest_seed = 23
    env._manifest_generation = 0
    env._manifest_rows = ()
    env._manifest_cursor = 99
    env.load_manifest_cache_state(cache)

    assert env.split_name == "train"
    assert env.base_manifest_seed == 17
    assert env._manifest_generation == 3
    assert env._manifest_rows is rows
    assert env._manifest_cursor == 0
    with pytest.raises(ValueError, match="generation"):
        env.load_manifest_cache_state(dict(cache, manifest_generation=-1))
    with pytest.raises(ValueError, match="task"):
        env.load_manifest_cache_state(dict(cache, task_id="t2_trans"))


def test_t5_hidden_event_branch_is_deterministic_balanced_and_not_a_reset_factor() -> None:
    episode_ids = [f"episode-{index:04d}" for index in range(128)]
    branches = [t5_branch_for_episode(episode_id) for episode_id in episode_ids]

    assert branches == [t5_branch_for_episode(episode_id) for episode_id in episode_ids]
    assert set(branches) == {"left", "right"}
    assert 40 <= branches.count("left") <= 88
    with pytest.raises(ValueError, match="non-empty trimmed"):
        t5_branch_for_episode(" episode-1")


def test_t5_reset_hook_arms_hidden_event_tape_before_the_first_step() -> None:
    armed: list[SimpleNamespace] = []

    class FakeEnv:
        def arm_event_tape(self, tape: SimpleNamespace) -> str:
            armed.append(tape)
            return "tape-sha"

    result = arm_hidden_t5_event(
        task_id="t5_replan",
        split_name="train",
        env=FakeEnv(),
        request=SimpleNamespace(episode_id="t5-episode-17", factors={}),
        load_task_config=lambda _task_id: {
            "event": {"default_trigger_gate_y_m": 0.42}
        },
        event_tape_type=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    assert result == "tape-sha"
    assert len(armed) == 1
    assert armed[0].branch in {"left", "right"}
    assert armed[0].trigger_gate_y_m == 0.42
    assert "t5-episode-17" in armed[0].event_id


def test_state_allowlist_contains_no_future_oracle_fields() -> None:
    assert not any(
        token in name.lower()
        for name in ALLOWED_PRIVILEGED_KEYS
        for token in ("future", "oracle", "next_state", "label")
    )


def test_state_schema_is_fixed_masked_and_quaternion_sign_safe() -> None:
    observation = _observation()
    factors = {"speed_class": "normal", "object_quat_wxyz": [-1.0, 0.0, 0.0, 0.0]}
    schema = DynamicBenchmarkStateSchema.from_observation(
        task_id="t1_so3",
        task_ids=("t1_xyz", "t1_so3"),
        observation=observation,
        factors=factors,
    )

    state = schema.encode(observation=observation, factors=factors, horizon_steps=10)

    assert state.shape == (schema.state_dim,)
    assert np.all(np.isfinite(state))
    assert "future_object_pose_wxyz" not in {
        field.name for field in schema.fields
    }
    # task one-hot + time + proprio precede object_pose_wxyz.
    assert np.array_equal(state[6:13], np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]))
    # eef_pose_xyzw is the next field and its scalar-last quaternion is canonicalized.
    assert np.array_equal(state[13:20], np.asarray([4.0, 5.0, 6.0, -0.0, -0.0, -0.0, 1.0]))
    assert np.all(state[-schema.mask_dim :] == 1.0)
    schema_payload = schema.to_dict()
    assert schema_payload["state_dim"] == schema.state_dim
    assert schema_payload["schema_version"] == "rlinf-dynamic-benchmark-state-v0.1"
    assert "derived_fields" not in schema_payload


def test_state_schema_rejects_shape_drift_and_unknown_categories() -> None:
    observation = _observation()
    schema = DynamicBenchmarkStateSchema.from_observation(
        task_id="t1_so3",
        task_ids=("t1_so3",),
        observation=observation,
        factors={"speed_class": "normal"},
    )
    changed = _observation()
    changed.proprio["robot0_proprio_state"] = np.zeros(3)

    with pytest.raises(ValueError, match="shape changed"):
        schema.encode(
            observation=changed,
            factors={"speed_class": "normal"},
            horizon_steps=10,
        )
    with pytest.raises(ValueError, match="outside vocabulary"):
        schema.encode(
            observation=observation,
            factors={"speed_class": "future-fast"},
            horizon_steps=10,
        )


def test_reward_is_independently_recomputable_from_current_transition() -> None:
    kwargs = {
        "action": np.zeros(7),
        "event_names": ("task_start", "approach"),
        "active_stage_progress": 0.5,
        "success": False,
        "terminated": False,
        "truncated": False,
        "termination_reason": None,
    }
    first = DynamicBenchmarkReward(success_stages=("approach", "grasp", "success_stage"))
    second = DynamicBenchmarkReward(success_stages=("approach", "grasp", "success_stage"))

    first_total, first_components = first.step(**kwargs)
    second_total, second_components = second.step(**kwargs)

    assert first_total == second_total
    assert first_components == second_components
    assert first_components["potential"] == pytest.approx(0.5)


def test_reward_applies_success_and_safety_as_distinct_terminal_terms() -> None:
    success_reward = DynamicBenchmarkReward(success_stages=("grasp",))
    safety_reward = DynamicBenchmarkReward(success_stages=("grasp",))

    _, success = success_reward.step(
        action=np.zeros(7),
        event_names=("grasp",),
        active_stage_progress=0.0,
        success=True,
        terminated=True,
        truncated=False,
        termination_reason="success",
    )
    _, safety = safety_reward.step(
        action=np.zeros(7),
        event_names=(),
        active_stage_progress=0.0,
        success=False,
        terminated=True,
        truncated=False,
        termination_reason="unsafe_contact",
    )

    assert success["success"] == 10.0
    assert success["safety"] == 0.0
    assert safety["success"] == 0.0
    assert safety["safety"] == -10.0


@pytest.mark.parametrize(
    "reason",
    (
        "object_goal_collision_unsafe",
        "forbidden_wire_contact",
        "release_impulse",
        "wrong_striker_contact",
    ),
)
def test_reward_recognizes_benchmark_specific_safety_failures(reason: str) -> None:
    reward = DynamicBenchmarkReward(success_stages=("grasp",))

    _, components = reward.step(
        action=np.zeros(7),
        event_names=(),
        active_stage_progress=0.0,
        success=False,
        terminated=True,
        truncated=False,
        termination_reason=reason,
    )

    assert components["safety"] == -10.0
    assert components["failure"] == 0.0


def test_reward_state_round_trip_and_validation() -> None:
    reward = DynamicBenchmarkReward(success_stages=("approach", "grasp"))
    reward.step(
        action=np.zeros(7),
        event_names=("approach",),
        active_stage_progress=0.25,
        success=False,
        terminated=False,
        truncated=False,
        termination_reason=None,
    )
    state = reward.state_dict()
    restored = DynamicBenchmarkReward(success_stages=("approach", "grasp"))
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    assert state == {
        "schema_version": "rlinf-dynamic-benchmark-reward-v0.2",
        "previous_potential": pytest.approx(0.625),
    }
    with pytest.raises(ValueError, match="schema"):
        restored.load_state_dict({"schema_version": "future", "previous_potential": 0.0})
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        restored.load_state_dict(
            {"schema_version": state["schema_version"], "previous_potential": 2.0}
        )


def _derived_observation() -> SimpleNamespace:
    return SimpleNamespace(
        task_id="t1_so3",
        policy_step=5,
        proprio={"robot0_proprio_state": np.asarray([0.1, -0.2])},
        privileged={
            "object_pose_wxyz": np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
            "object_twist_world": np.asarray([0.5, -0.25, 0.0, 0.0, 0.0, 0.75]),
            "eef_pose_xyzw": np.asarray([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]),
            "left_fingerpad_center_world": np.asarray([4.0, 4.9, 6.0]),
            "right_fingerpad_center_world": np.asarray([4.0, 5.1, 6.0]),
            "fingerpad_closing_axis_world": np.asarray([0.0, 1.0, 0.0]),
        },
    )


def test_derived_state_features_are_current_state_and_masked() -> None:
    observation = _derived_observation()
    features = (
        "eef_to_object_pose_wxyz",
        "eef_to_object_distance_m",
        "fingerpad_midpoint_world",
        "grasp_point_offset_world_m",
        "closing_axis_object_alignment_rad",
        "object_vertical_position_m",
        "eef_vertical_position_m",
        "object_yaw_rad",
        "object_xy_velocity_m_s",
        "object_angular_velocity_z_rad_s",
    )
    schema = DynamicBenchmarkStateSchema.from_observation(
        task_id="t1_so3",
        task_ids=("t1_xyz", "t1_so3"),
        observation=observation,
        factors={"speed_class": "normal"},
        derived_features=features,
    )

    state = schema.encode(
        observation=observation, factors={"speed_class": "normal"}, horizon_steps=10
    )

    assert state.shape == (schema.state_dim,)
    assert np.all(np.isfinite(state))
    assert {field.name for field in schema.derived_fields} == set(features)
    assert len(schema.derived_fields) == len(features)
    assert schema.to_dict()["schema_version"] == (
        "rlinf-dynamic-benchmark-state-v0.2"
    )
    assert len(schema.to_dict()["derived_fields"]) == len(features)
    # Derived masks follow the regular field masks and precede factor masks.
    derived_mask = state[-schema.mask_dim :][
        len(schema.fields) : len(schema.fields) + len(features)
    ]
    assert np.all(derived_mask == 1.0)
    # eef_to_object pose: object at (1,2,3) relative to eef at (4,5,6) with
    # identity orientation is (-3,-3,-3) and a canonical wxyz quaternion.
    base = (
        len(schema.task_ids)
        + 2
        + sum(field.size for field in schema.fields)
    )
    assert np.allclose(state[base : base + 7], [-3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0])
    assert state[base + 7 : base + 8] == pytest.approx(np.sqrt(27.0))


def test_derived_features_missing_prerequisites_are_zero_masked() -> None:
    observation = _derived_observation()
    del observation.privileged["fingerpad_closing_axis_world"]
    schema = DynamicBenchmarkStateSchema.from_observation(
        task_id="t1_so3",
        task_ids=("t1_so3",),
        observation=observation,
        factors={},
        derived_features=("closing_axis_object_alignment_rad",),
    )

    state = schema.encode(
        observation=observation, factors={}, horizon_steps=10
    )

    mask = state[-schema.mask_dim :]
    assert mask[-1] == 0.0
    assert state[schema.value_dim - 1] == 0.0


def test_dense_lift_shaping_reward_increases_after_bilateral_hold() -> None:
    reward = DynamicBenchmarkReward(
        success_stages=("approach", "bilateral_hold", "clearance", "stable_dwell"),
        lift_shaping_weight=2.0,
        lift_target_m=0.1,
    )
    kwargs = {
        "action": np.zeros(7),
        "active_stage_progress": 0.0,
        "success": False,
        "terminated": False,
        "truncated": False,
        "termination_reason": None,
    }
    _, before = reward.step(
        **kwargs,
        event_names=("approach", "bilateral_hold"),
        object_z_m=0.10,
    )
    _, after = reward.step(
        **kwargs,
        event_names=("approach", "bilateral_hold"),
        object_z_m=0.15,
    )

    assert before["lift_shaping"] == pytest.approx(0.0)
    assert after["lift_shaping"] == pytest.approx(2.0 * 0.5)
    # Second step has no progress delta, so total = step penalty + lift shaping.
    assert after["total"] == pytest.approx(after["step"] + after["lift_shaping"])


def test_dense_orientation_shaping_reward_tracks_alignment() -> None:
    reward = DynamicBenchmarkReward(
        success_stages=("approach", "bilateral_hold", "clearance", "stable_dwell"),
        orientation_shaping_weight=3.0,
    )
    kwargs = {
        "action": np.zeros(7),
        "event_names": (),
        "active_stage_progress": 0.0,
        "success": False,
        "terminated": False,
        "truncated": False,
        "termination_reason": None,
    }
    _, first = reward.step(**kwargs, alignment_error_rad=1.2)
    _, second = reward.step(**kwargs, alignment_error_rad=0.2)

    first_potential = 1.0 - 1.2 / (np.pi / 2.0)
    second_potential = 1.0 - 0.2 / (np.pi / 2.0)
    assert first["orientation_shaping"] == pytest.approx(
        3.0 * first_potential
    )
    assert second["orientation_shaping"] == pytest.approx(
        3.0 * (second_potential - first_potential)
    )


def test_dense_reward_missing_measurements_do_not_create_negative_deltas() -> None:
    reward = DynamicBenchmarkReward(
        success_stages=("approach", "bilateral_hold", "clearance"),
        lift_shaping_weight=2.0,
        orientation_shaping_weight=3.0,
        lift_target_m=0.1,
    )
    kwargs = {
        "action": np.zeros(7),
        "event_names": ("approach", "bilateral_hold"),
        "active_stage_progress": 0.0,
        "success": False,
        "terminated": False,
        "truncated": False,
        "termination_reason": None,
    }
    reward.step(**kwargs, object_z_m=0.10, alignment_error_rad=0.2)
    _, elevated = reward.step(
        **kwargs, object_z_m=0.15, alignment_error_rad=0.1
    )
    _, missing = reward.step(
        **kwargs, object_z_m=None, alignment_error_rad=None
    )

    assert elevated["lift_shaping"] > 0.0
    assert elevated["orientation_shaping"] > 0.0
    assert missing["lift_shaping"] == 0.0
    assert missing["orientation_shaping"] == 0.0


def test_dense_reward_defaults_preserve_baseline_components() -> None:
    kwargs = {
        "action": np.zeros(7),
        "event_names": ("task_start", "approach"),
        "active_stage_progress": 0.5,
        "success": False,
        "terminated": False,
        "truncated": False,
        "termination_reason": None,
    }
    plain = DynamicBenchmarkReward(success_stages=("approach", "grasp", "success_stage"))
    shaped = DynamicBenchmarkReward(success_stages=("approach", "grasp", "success_stage"))

    plain_total, plain_components = plain.step(**kwargs)
    shaped_total, shaped_components = shaped.step(**kwargs, object_z_m=1.0, alignment_error_rad=0.1)

    assert "lift_shaping" not in shaped_components
    assert "orientation_shaping" not in shaped_components
    assert shaped_total == plain_total
    for name, value in plain_components.items():
        assert shaped_components[name] == value


def test_dense_reward_state_round_trip_with_shaping() -> None:
    reward = DynamicBenchmarkReward(
        success_stages=("approach", "bilateral_hold", "clearance", "stable_dwell"),
        lift_shaping_weight=1.0,
        orientation_shaping_weight=2.0,
    )
    reward.step(
        action=np.zeros(7),
        event_names=("approach", "bilateral_hold"),
        active_stage_progress=0.0,
        success=False,
        terminated=False,
        truncated=False,
        termination_reason=None,
        object_z_m=0.12,
    )
    reward.step(
        action=np.zeros(7),
        event_names=("approach", "bilateral_hold"),
        active_stage_progress=0.0,
        success=False,
        terminated=False,
        truncated=False,
        termination_reason=None,
        object_z_m=0.14,
        alignment_error_rad=0.3,
    )
    state = reward.state_dict()
    restored = DynamicBenchmarkReward(
        success_stages=("approach", "bilateral_hold", "clearance", "stable_dwell"),
        lift_shaping_weight=1.0,
        orientation_shaping_weight=2.0,
    )
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    assert state["schema_version"] == "rlinf-dynamic-benchmark-reward-v0.3"
    legacy_state = {
        "schema_version": "rlinf-dynamic-benchmark-reward-v0.2",
        "previous_potential": 0.0,
    }
    with pytest.raises(ValueError, match="cannot resume enabled dense shaping"):
        restored.load_state_dict(legacy_state)
    broken_state = dict(state)
    broken_state.pop("previous_orientation_potential")
    with pytest.raises(ValueError, match="missing fields"):
        restored.load_state_dict(broken_state)
