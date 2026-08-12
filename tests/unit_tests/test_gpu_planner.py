# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only contract tests for the current-state GPU Planner seam."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.dynamic_benchmark.gpu_planner import (
    GpuCurrentStatePlanner,
    GpuPlannerUnavailableError,
)


class _Mode:
    value = "e7"


_MODE = _Mode()


class _Request:
    task_id = "t3_phase"
    episode_id = "planner-test-0"
    action_mode = _MODE


class _Observation:
    def __init__(self, policy_step: int) -> None:
        self.policy_step = policy_step
        self.fingerprint_sha256 = f"{policy_step + 1:064x}"


class _Action:
    def __init__(self, policy_step: int) -> None:
        self.mode = _MODE
        self.values = np.zeros(7, dtype=np.float64)
        self.policy_step = policy_step


class _Result:
    def __init__(self, policy_step: int) -> None:
        self.observation = _Observation(policy_step)
        self.terminated = policy_step >= 2
        self.truncated = False
        self.success = False
        self.termination_reason = "probe_failure" if self.terminated else None
        self.active_stage_progress = 0.25 * policy_step


class _Backend:
    backend_id = "mjwarp_gpu_v1"
    task_id = "t3_phase"
    num_envs = 1
    observation_track = SimpleNamespace(value="state")

    def __init__(self, platform: str = "cuda") -> None:
        self.provenance = SimpleNamespace(
            backend_id="mjwarp_gpu_v1",
            device_platform=platform,
            runtime_versions={},
        )
        self.step_count = 0
        self.last_terminal_rows = ()

    def reset(self, requests: tuple[_Request, ...]) -> tuple[_Observation, ...]:
        assert len(requests) == 1
        self.step_count = 0
        self.last_terminal_rows = ()
        return (_Observation(0),)

    def step(self, actions: tuple[_Action, ...]) -> tuple[_Result, ...]:
        assert len(actions) == 1
        self.step_count += 1
        result = _Result(self.step_count)
        if result.terminated:
            self.last_terminal_rows = (
                SimpleNamespace(
                    task_quality=None,
                    lane=0,
                    episode_id="planner-test-0",
                    task_id="t3_phase",
                    outcome=SimpleNamespace(value="failure"),
                    terminated=True,
                    truncated=False,
                    success=False,
                    termination_reason="probe_failure",
                    physics_step=50,
                    control_step=2,
                    policy_step=2,
                    completion=0.25,
                ),
            )
        return (result,)


class _Planner:
    def __init__(self) -> None:
        self.seen_policy_steps: list[int] = []

    def reset(self) -> None:
        self.seen_policy_steps.clear()

    def act(self, observation: _Observation) -> _Action:
        self.seen_policy_steps.append(observation.policy_step)
        return _Action(observation.policy_step)


def test_planner_consumes_current_observation_and_replays_tape() -> None:
    backend = _Backend()
    planners: list[_Planner] = []

    def factory(task_id: str, request: _Request) -> _Planner:
        assert task_id == request.task_id == "t3_phase"
        planner = _Planner()
        planners.append(planner)
        return planner

    runner = GpuCurrentStatePlanner(
        backend=backend,
        task_id="t3_phase",
        planner_factory=factory,
        max_control_steps=4,
    )
    tape = runner.rollout(_Request())

    assert planners[0].seen_policy_steps == [0, 1]
    assert len(tape.observations) == 3
    assert len(tape.actions) == len(tape.results) == 2
    assert tape.to_dict()["observation_fingerprints"] == [
        f"{index:064x}" for index in (1, 2, 3)
    ]
    replay = runner.replay(tape, backend=_Backend())
    assert replay["passed"] is True
    assert replay["action_count"] == 2


def test_planner_rejects_cpu_provenance() -> None:
    with pytest.raises(GpuPlannerUnavailableError, match="CUDA"):
        GpuCurrentStatePlanner(
            backend=_Backend(platform="cpu"),
            task_id="t3_phase",
        )
