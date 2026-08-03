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

import numpy as np
import pytest

from examples.embodiment.evaluate_dynamic_benchmark_planner import (
    _ArmedResetReplayEnv,
    _parser,
    _planner_action_values,
    _replay_actions_on_fresh_env,
)


def _arguments(split: str) -> list[str]:
    return [
        "--evaluator-commit",
        "a" * 40,
        "--benchmark-commit",
        "b" * 40,
        "--output",
        "planner-eval",
        "--task",
        "t1_xyz",
        "--split",
        split,
        "--manifest-seed",
        "20261150",
    ]


def test_planner_evaluator_accepts_validation_without_test_access() -> None:
    args = _parser().parse_args(_arguments("validation"))

    assert args.split == "validation"
    assert args.task == "t1_xyz"
    assert args.episodes == 20


def test_planner_evaluator_rejects_unknown_split() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(_arguments("train"))


def test_planner_action_record_uses_executed_float32_values() -> None:
    source = np.asarray([0.123456789012345, -2.0, 0.0, 0.5, -0.5, 1.0, 2.0])

    env_actions, recorded = _planner_action_values(source)

    assert env_actions.dtype.is_floating_point
    assert np.array_equal(recorded.astype(np.float32), env_actions.numpy()[0])
    assert recorded.tolist() == env_actions.numpy()[0].astype(np.float64).tolist()
    assert recorded[0] != source[0]
    assert recorded[1] == -1.0
    assert recorded[-1] == 1.0


def test_planner_replay_uses_fresh_raw_env_and_rearms_hidden_event() -> None:
    request = object()

    class RawEnv:
        def __init__(self) -> None:
            self.closed = False

        def reset(self, value):
            assert value is request
            return "fresh-observation"

        def step(self, action):
            return ("step", action)

        def save_state(self):
            return b"state"

        def close(self):
            self.closed = True

    class VectorEnv:
        image_size = 64
        camera_observations = False

        def __init__(self) -> None:
            self.raw_env = RawEnv()
            self.make_calls = []
            self.arm_calls = []

        def _make_mujoco_env(self, task_id, **kwargs):
            self.make_calls.append((task_id, kwargs))
            return self.raw_env

        def _arm_hidden_t5_event(self, raw_env, value):
            self.arm_calls.append((raw_env, value))

    vector_env = VectorEnv()

    def replay_fn(proxy, **kwargs):
        assert isinstance(proxy, _ArmedResetReplayEnv)
        assert proxy.reset(request) == "fresh-observation"
        assert proxy.step("action") == ("step", "action")
        assert proxy.save_state() == b"state"
        assert kwargs["request"] is request
        return {"passed": True}

    result = _replay_actions_on_fresh_env(
        vector_env=vector_env,
        task_id="t5_replan",
        request=request,
        expected_observations=("expected",),
        actions=("action",),
        expected_outcomes=("outcome",),
        expected_final_state=b"expected-state",
        replay_fn=replay_fn,
    )

    assert result == {"passed": True}
    assert vector_env.make_calls == [
        ("t5_replan", {"image_size": 64, "camera_observations": False})
    ]
    assert vector_env.arm_calls == [(vector_env.raw_env, request)]
    assert vector_env.raw_env.closed
