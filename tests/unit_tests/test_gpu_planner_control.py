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

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from se3_wam.benchmark.api import ActionCommand
from se3_wam.benchmark.contracts import ActionMode

_MODULE_PATH = (
    Path(__file__).parents[2]
    / "rlinf"
    / "envs"
    / "dynamic_benchmark"
    / "gpu_planner_control.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_gpu_planner_control_under_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
run_gpu_planner_cohort = _MODULE.run_gpu_planner_cohort


@dataclass(frozen=True)
class _Observation:
    episode_id: str
    task_id: str
    physics_step: int
    control_step: int
    policy_step: int


class _Backend:
    num_envs = 2
    cohort_horizon_steps = 4
    task_id = "p0_grasp"
    device = "cuda:0"
    provenance = SimpleNamespace(backend_id="mjwarp_gpu_v1")

    def __init__(self) -> None:
        self.reset_calls = 0
        self.observation_calls: list[tuple[int, ...]] = []
        self.action_calls: list[np.ndarray] = []
        self.terminal_calls: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        self.attest_calls = 0
        self.control_step = 0

    def reset(self) -> Any:
        self.reset_calls += 1
        return SimpleNamespace(episode_ids=("episode-0", "episode-1"))

    def materialize_teacher_observations(
        self, lanes: tuple[int, ...]
    ) -> tuple[_Observation, ...]:
        self.observation_calls.append(lanes)
        return tuple(
            _Observation(
                episode_id=f"episode-{lane}",
                task_id=self.task_id,
                physics_step=25 * self.control_step,
                control_step=self.control_step,
                policy_step=self.control_step,
            )
            for lane in lanes
        )

    def step_device(self, action: np.ndarray) -> Any:
        self.action_calls.append(action.copy())
        self.control_step += 1
        done = (
            np.asarray([True, False], dtype=np.bool_)
            if self.control_step == 1
            else np.asarray([True, True], dtype=np.bool_)
        )
        return SimpleNamespace(done=done)

    def materialize_terminal_ledger_once(
        self, lanes: tuple[int, ...], episode_ids: tuple[str, ...]
    ) -> tuple[Any, ...]:
        self.terminal_calls.append((lanes, episode_ids))
        return tuple(
            SimpleNamespace(lane=lane, episode_id=episode_id)
            for lane, episode_id in zip(lanes, episode_ids, strict=True)
        )

    def attest_end(self) -> str:
        self.attest_calls += 1
        return "attested"


class _Planner:
    def __init__(self, lane: int) -> None:
        self.lane = lane
        self.calls: list[_Observation] = []

    def act(self, observation: _Observation) -> ActionCommand:
        self.calls.append(observation)
        value = 0.1 * (self.lane + 1) + 0.1 * observation.policy_step
        return ActionCommand(
            mode=ActionMode.E7,
            values=np.full(7, value, dtype=np.float64),
            policy_step=observation.policy_step,
        )


def test_online_planner_uses_current_observations_and_fresh_cuda_actions() -> None:
    backend = _Backend()
    planners = (_Planner(0), _Planner(1))
    action_devices: list[Any] = []

    def action_factory(values: np.ndarray, device: Any) -> np.ndarray:
        action_devices.append(device)
        return values.copy()

    result = run_gpu_planner_cohort(
        backend,
        planners,
        action_tensor_factory=action_factory,
    )

    assert backend.reset_calls == 1
    assert backend.observation_calls == [(0, 1), (1,)]
    assert [observation.policy_step for observation in planners[0].calls] == [0]
    assert [observation.policy_step for observation in planners[1].calls] == [0, 1]
    assert len(backend.action_calls) == 2
    np.testing.assert_allclose(backend.action_calls[0][0], 0.1)
    np.testing.assert_allclose(backend.action_calls[0][1], 0.2)
    np.testing.assert_allclose(backend.action_calls[1][0], 0.0)
    np.testing.assert_allclose(backend.action_calls[1][1], 0.3)
    assert action_devices == ["cuda:0", "cuda:0"]
    assert backend.terminal_calls == [
        ((0,), ("episode-0",)),
        ((1,), ("episode-1",)),
    ]
    assert backend.attest_calls == 1
    assert result.control_steps == 2
    assert len(result.control_tape) == 3
    assert tuple(row.lane for row in result.terminal_rows) == (0, 1)
    assert all(
        record.policy_step == record.control_step for record in result.control_tape
    )


def test_gpu_planner_rejects_non_e7_commands_before_gpu_step() -> None:
    backend = _Backend()

    class _J8Planner:
        def act(self, observation: _Observation) -> ActionCommand:
            return ActionCommand(
                mode=ActionMode.J8,
                values=np.zeros(8, dtype=np.float64),
                policy_step=observation.policy_step,
            )

    with pytest.raises(ValueError, match="E7"):
        run_gpu_planner_cohort(
            backend,
            (_J8Planner(), _Planner(1)),
            action_tensor_factory=lambda values, device: values,
        )
    assert backend.action_calls == []
    assert backend.attest_calls == 0


def test_gpu_planner_rejects_cpu_backend_before_reset() -> None:
    backend = _Backend()
    backend.provenance = SimpleNamespace(backend_id="cpu")
    with pytest.raises(RuntimeError, match="backend=mjwarp_gpu_v1"):
        run_gpu_planner_cohort(
            backend,
            (_Planner(0), _Planner(1)),
            action_tensor_factory=lambda values, device: values,
        )
    assert backend.reset_calls == 0


def test_gpu_planner_does_not_return_partial_cohort() -> None:
    backend = _Backend()
    with pytest.raises(RuntimeError, match="remained active"):
        run_gpu_planner_cohort(
            backend,
            (_Planner(0), _Planner(1)),
            action_tensor_factory=lambda values, device: values,
            max_steps=1,
        )
    assert backend.attest_calls == 0
