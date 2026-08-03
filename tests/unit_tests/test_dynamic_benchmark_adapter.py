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
    assert schema.to_dict()["state_dim"] == schema.state_dim


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
    with pytest.raises(ValueError, match="schema"):
        restored.load_state_dict({"schema_version": "future", "previous_potential": 0.0})
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        restored.load_state_dict(
            {"schema_version": state["schema_version"], "previous_potential": 2.0}
        )
