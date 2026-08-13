# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import enum
import hashlib
import math
import sys
import types
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rlinf.envs.dynamic_benchmark import t1_so3_planner as _MODULE


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _health(step: int, *, terminal: bool = False) -> dict[str, int]:
    return {
        "overflow": 0,
        "controller_valid": 1,
        "driver_valid": 1,
        "physics_step": 25 * step,
        "terminal": int(terminal),
    }


def _frame(step: int) -> dict[str, np.ndarray]:
    return {
        "agentview": np.full((4, 4, 3), step + 1, dtype=np.uint8),
        "robot0_eye_in_hand": np.full((4, 4, 3), step + 2, dtype=np.uint8),
    }


class _ActionMode(enum.Enum):
    E7 = "E7"


@dataclass(frozen=True)
class _ActionCommand:
    mode: _ActionMode
    values: np.ndarray
    policy_step: int


@dataclass(frozen=True)
class _Observation:
    episode_id: str
    task_id: str
    physics_step: int
    control_step: int
    policy_step: int
    component_sha256: MappingProxyType[str, str]
    privileged: MappingProxyType[str, np.ndarray]
    events_since_last_observation: tuple[Any, ...] = ()


def _observation(step: int) -> _Observation:
    return _Observation(
        episode_id="t1-so3-episode-cycle00000000",
        task_id="t1_so3",
        physics_step=25 * step,
        control_step=step,
        policy_step=step,
        component_sha256=MappingProxyType(
            {
                "metadata": _sha(f"metadata-{step}"),
                "privileged/state": _sha(f"state-{step}"),
                "rgb/agentview": _sha(f"rgb-{step}"),
            }
        ),
        privileged=MappingProxyType(
            {
                "object_pose_wxyz": np.asarray([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
                "eef_pose_xyzw": np.asarray([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
                "fingerpad_closing_axis_world": np.asarray([1.0, 0.0, 0.0]),
                "object_twist_world": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.2]),
            }
        ),
    )


class _Backend:
    backend_id = "mjwarp_gpu_v1"
    task_id = "t1_so3"
    num_envs = 1
    device = SimpleNamespace(type="cuda")
    observation_track = "state"
    render_observations = True
    cohort_horizon_steps = 3
    stable_identity = MappingProxyType(
        {
            "backend_id": "mjwarp_gpu_v1",
            "source_commit": "a" * 40,
            "render_observations": True,
            "task_id": "t1_so3",
        }
    )

    def __init__(self, *, terminate_at: int = 2) -> None:
        self.step = 0
        self.terminate_at = terminate_at
        self.terminal_calls = 0
        self.review_steps: list[int] = []

    def next_requests(self) -> tuple[Any, ...]:
        return (SimpleNamespace(episode_id="t1-so3-episode-cycle00000000"),)

    def reset(self) -> Any:
        self.step = 0
        self.terminal_calls = 0
        self.review_steps = []
        return SimpleNamespace(
            episode_ids=("t1-so3-episode-cycle00000000",),
            manifest_sha256=_sha("manifest"),
            manifest_ordinals=(0,),
            seeds=(20261040,),
        )

    def materialize_health_audit(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray([value], dtype=np.int64)
            for name, value in _health(
                self.step,
                terminal=self.step == self.terminate_at,
            ).items()
        }

    def materialize_review_rgb(
        self,
        lanes: tuple[int, ...],
    ) -> tuple[dict[str, np.ndarray], ...]:
        assert lanes == (0,)
        self.review_steps.append(self.step)
        return (_frame(self.step),)

    def materialize_teacher_observations(
        self,
        lanes: tuple[int, ...],
    ) -> tuple[_Observation, ...]:
        assert lanes == (0,)
        return (_observation(self.step),)

    def step_device(self, action: Any) -> Any:
        assert np.asarray(action).shape == (1, 7)
        self.step += 1
        return SimpleNamespace(
            done=np.asarray([self.step == self.terminate_at], dtype=np.bool_)
        )

    def materialize_terminal_ledger_once(
        self,
        lanes: tuple[int, ...],
        episode_ids: tuple[str, ...],
    ) -> tuple[Any, ...]:
        assert lanes == (0,)
        assert episode_ids == ("t1-so3-episode-cycle00000000",)
        self.terminal_calls += 1
        assert self.terminal_calls == 1
        return (
            SimpleNamespace(
                lane=0,
                episode_id=episode_ids[0],
                task_id="t1_so3",
                physics_step=25 * self.step,
                control_step=self.step,
                policy_step=self.step,
            ),
        )


class _Planner:
    phase = "track"
    orientation_mode = "track_object"
    lookahead_s = 0.1
    grasp_axis_offset_rad = math.pi

    def __init__(self) -> None:
        self.calls: list[int] = []

    def act(self, observation: _Observation) -> _ActionCommand:
        self.calls.append(observation.policy_step)
        return _ActionCommand(
            mode=_ActionMode.E7,
            values=np.full(7, 0.1 * (observation.policy_step + 1)),
            policy_step=observation.policy_step,
        )

    def planner_audit_snapshot(self) -> dict[str, Any]:
        return {"source": "current_observation", "calls": len(self.calls)}


@pytest.fixture
def action_api(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = types.ModuleType("torch")
    torch.float32 = np.float32
    torch.as_tensor = lambda value, dtype, device: np.asarray(value, dtype=dtype)
    monkeypatch.setitem(sys.modules, "torch", torch)

    api = types.ModuleType("se3_wam.benchmark.api")
    api.ActionCommand = _ActionCommand
    contracts = types.ModuleType("se3_wam.benchmark.contracts")
    contracts.ActionMode = _ActionMode
    monkeypatch.setitem(sys.modules, "se3_wam", types.ModuleType("se3_wam"))
    monkeypatch.setitem(
        sys.modules,
        "se3_wam.benchmark",
        types.ModuleType("se3_wam.benchmark"),
    )
    monkeypatch.setitem(sys.modules, "se3_wam.benchmark.api", api)
    monkeypatch.setitem(sys.modules, "se3_wam.benchmark.contracts", contracts)


def test_t1_so3_online_natural_tape_and_fresh_replay_are_strict(
    action_api: None,
) -> None:
    online = _Backend()
    planner = _Planner()
    adapter = _MODULE.CurrentStatePlannerAdapter(online, planner)
    adapter.reset()
    _result, online_rows = adapter.run_natural_termination()

    tape = adapter.tape
    assert planner.calls == [0, 1]
    assert tape.complete
    assert tape.identity.task_id == "t1_so3"
    assert tape.identity.horizon_steps == 2
    assert tape.as_dict()["schema_version"] == (
        _MODULE.T1_SO3_PLANNER_TAPE_SCHEMA_VERSION
    )
    assert [entry.done_after for entry in tape.entries] == [False, True]
    assert len(adapter.scene_frames) == 3
    assert online.review_steps == [0, 1, 2]
    assert online.terminal_calls == 1

    diagnostics = tape.entries[0].diagnostics
    assert diagnostics["orientation_mode"] == "track_object"
    assert diagnostics["future_yaw_rad"] == pytest.approx(0.02)
    assert diagnostics["modulo_pi_axis_error_rad"] == pytest.approx(0.0)
    assert diagnostics["planner_audit"]["source"] == "current_observation"

    fresh = _Backend()
    replay = _MODULE.replay_action_trajectory(fresh, tape)
    assert replay.passed
    assert len(replay.steps) == 2
    assert replay.terminal_rows[0].episode_id == online_rows[0].episode_id
    assert fresh.review_steps == online.review_steps
    assert fresh.terminal_calls == 1
    assert planner.calls == [0, 1]


def test_t1_so3_fixed_horizon_without_done_fails_closed(action_api: None) -> None:
    backend = _Backend(terminate_at=4)
    adapter = _MODULE.CurrentStatePlannerAdapter(backend, _Planner())
    adapter.reset()

    with pytest.raises(
        _MODULE.T1So3PlannerError,
        match="did not reach natural termination",
    ):
        adapter.run_natural_termination()

    assert not adapter.tape.complete
    assert len(adapter.tape.entries) == backend.cohort_horizon_steps
    assert backend.terminal_calls == 0


def test_t1_so3_adapter_rejects_wrong_task_before_any_step() -> None:
    backend = _Backend()
    backend.task_id = "t1_xyz"

    with pytest.raises(_MODULE.T1So3PlannerError, match="restricted to t1_so3"):
        _MODULE.CurrentStatePlannerAdapter(backend, _Planner())
