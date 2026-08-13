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
    GpuPlannerReplayError,
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


class _ProbeRequest:
    task_id = "t3_full"
    episode_id = "t3-full-row-0"
    action_mode = _MODE


class _ProbeObservation:
    def __init__(self, control_step: int, *, drift: bool = False) -> None:
        self.physics_step = control_step * 25
        self.control_step = control_step
        self.policy_step = control_step
        self.time_s = control_step * 0.05
        self.privileged = {
            "object_pose": np.asarray(
                [float(control_step), 1.0 if drift else 0.0], dtype=np.float64
            )
        }
        self.proprio = {"joint": np.asarray([control_step], dtype=np.float64)}
        self.events_since_last_observation = ()
        self.fingerprint_sha256 = (
            f"{control_step + 1000:064x}" if drift else f"{control_step + 1:064x}"
        )


class _ProbeResult:
    def __init__(self, control_step: int, *, drift: bool = False) -> None:
        self.observation = _ProbeObservation(control_step, drift=drift)
        self.terminated = control_step >= 3
        self.truncated = False
        self.success = False
        self.termination_reason = "probe_failure" if self.terminated else None
        self.active_stage_progress = 0.0


class _ProbeBackend:
    backend_id = "mjwarp_gpu_v1"
    task_id = "t3_full"
    num_envs = 1
    observation_track = SimpleNamespace(value="state")

    def __init__(self, *, drift_step: int | None = None) -> None:
        self.provenance = SimpleNamespace(
            backend_id="mjwarp_gpu_v1",
            device_platform="cuda",
            runtime_versions={},
        )
        self.drift_step = drift_step
        self.step_count = 0
        self.last_terminal_rows = ()
        self._observation = _ProbeObservation(0)

    def reset(self, _requests: tuple[_ProbeRequest, ...]) -> tuple[_ProbeObservation, ...]:
        self.step_count = 0
        self.last_terminal_rows = ()
        self._observation = _ProbeObservation(0)
        return (self._observation,)

    @staticmethod
    def verify_replay_probe_snapshot_roundtrip() -> dict[str, object]:
        return {
            "exact": True,
            "before_payload_sha256": "a" * 64,
            "after_payload_sha256": "a" * 64,
        }

    def materialize_current_observations(self) -> tuple[_ProbeObservation, ...]:
        return (self._observation,)

    def materialize_replay_probe_audit(self) -> dict[str, object]:
        drift = self.step_count == self.drift_step
        stage = 1 if drift else 0
        health = {
            "stage_index": stage,
            "grasp_attachment_active": 0,
            "dock_attachment_active": 0,
            "overflow": 0,
        }
        health_sha = f"{stage + 1:064x}"
        return {
            "diagnostic_only": True,
            "sections": {
                "health": health,
                "snapshot_scope": {"digest": "stable"},
                "live_mutable_data": {"digest": "stable"},
                "active_contact": {"digest": "stable"},
                "active_efc": {"digest": "stable"},
                "bookkeeping": {"control_step": self.step_count},
            },
            "section_sha256": {
                "health": health_sha,
                "snapshot_scope": "b" * 64,
                "live_mutable_data": "c" * 64,
                "active_contact": "d" * 64,
                "active_efc": "e" * 64,
                "bookkeeping": f"{self.step_count + 10:064x}",
            },
            "payload_sha256": f"{self.step_count + stage + 20:064x}",
        }

    def step(self, _actions: tuple[_Action, ...]) -> tuple[_ProbeResult, ...]:
        self.step_count += 1
        drift = self.step_count == self.drift_step
        result = _ProbeResult(self.step_count, drift=drift)
        self._observation = result.observation
        if result.terminated:
            self.last_terminal_rows = (
                SimpleNamespace(
                    task_quality=None,
                    lane=0,
                    episode_id="t3-full-row-0",
                    task_id="t3_full",
                    outcome=SimpleNamespace(value="failure"),
                    terminated=True,
                    truncated=False,
                    success=False,
                    termination_reason="probe_failure",
                    physics_step=75,
                    control_step=3,
                    policy_step=3,
                    completion=0.0,
                ),
            )
        return (result,)


def test_replay_probe_reports_step3_components_state_and_call_order() -> None:
    source = _ProbeBackend()
    runner = GpuCurrentStatePlanner(
        backend=source,
        task_id="t3_full",
        planner_factory=lambda _task_id, _request: _Planner(),
        max_control_steps=4,
        capture_replay_probe=True,
    )
    tape = runner.rollout(_ProbeRequest())

    with pytest.raises(GpuPlannerReplayError) as raised:
        runner.replay(tape, backend=_ProbeBackend(drift_step=3))

    evidence = dict(raised.value.evidence)
    assert evidence["control_step"] == 3
    assert evidence["physics_step"] == 75
    assert "privileged.object_pose" in evidence["observation_component_mismatches"]
    assert evidence["backend_probe_mismatched_sections"] == ["health"]
    assert evidence["backend_probe"]["expected_health"]["stage_index"] == 0
    assert evidence["backend_probe"]["actual_health"]["stage_index"] == 1
    assert evidence["backend_call_order_aligned"] is True
    assert evidence["action"]["policy_step"] == 2
    assert evidence["source_planner_state_before_action"] is not None
    assert evidence["expected_snapshot_roundtrip"]["exact"] is True
    assert evidence["replay_gate_relaxed"] is False
