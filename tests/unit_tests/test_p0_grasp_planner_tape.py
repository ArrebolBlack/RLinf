# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import enum
import hashlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).parents[2]
    / "rlinf"
    / "envs"
    / "dynamic_benchmark"
    / "p0_grasp_planner.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_p0_grasp_planner_under_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity() -> Any:
    return _MODULE.PlannerTapeIdentity(
        task_id="p0_grasp",
        backend_id="mjwarp_gpu_v1",
        num_envs=1,
        manifest_sha256=_sha("manifest"),
        episode_ids=("p0-train-0000",),
        manifest_ordinals=(0,),
        seeds=(20261050,),
        horizon_steps=4,
        max_horizon_steps=4,
        backend_identity_sha256=_sha("backend"),
    )


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


def _entry(step: int, *, done: bool) -> Any:
    return _MODULE.PlannerTapeEntry(
        lane=0,
        episode_id="p0-train-0000",
        task_id="p0_grasp",
        physics_step=25 * step,
        control_step=step,
        policy_step=step,
        observation_fingerprint_sha256=_sha(f"observation-{step}"),
        observation_component_sha256={"metadata": _sha(f"metadata-{step}")},
        action=(0.0,) * 7,
        diagnostics={"phase": "track"},
        health_after=_health(step + 1, terminal=done),
        done_after=done,
    )


def test_p0_tape_covers_reset_through_natural_terminal_frame() -> None:
    tape = _MODULE.ActionTrajectoryTape(_identity(), reset_health=_health(0))
    tape.append_review_frame(physics_step=0, frame=_frame(0))
    for step in range(3):
        tape.append_review_frame(physics_step=25 * (step + 1), frame=_frame(step + 1))
        tape.append(_entry(step, done=step == 2))
    tape.finalize(3)

    assert tape.complete
    assert tape.identity.horizon_steps == 3
    assert len(tape.review_frames) == 4
    assert tape.as_dict()["sha256"] == tape.sha256


def test_p0_action_tape_is_distinct_from_full_trajectory_payload() -> None:
    tape = _MODULE.ActionTrajectoryTape(_identity(), reset_health=_health(0))
    tape.append_review_frame(physics_step=0, frame=_frame(0))
    tape.append_review_frame(physics_step=25, frame=_frame(1))
    tape.append(_entry(0, done=True))
    tape.finalize(1)

    action_payload = tape.action_dict()
    trajectory_payload = tape.as_dict()
    assert "diagnostics" not in action_payload["entries"][0]
    assert "observation_fingerprint_sha256" not in action_payload["entries"][0]
    assert trajectory_payload["entries"][0]["diagnostics"] == {"phase": "track"}
    assert action_payload["sha256"] != trajectory_payload["sha256"]


def test_p0_planner_fails_closed_without_state_plus_gpu_review_render() -> None:
    backend = SimpleNamespace(
        backend_id="mjwarp_gpu_v1",
        task_id="p0_grasp",
        num_envs=1,
        device="cuda:0",
        observation_track="hybrid",
        render_observations=True,
    )
    with pytest.raises(_MODULE.P0GraspPlannerError, match="STATE"):
        _MODULE.CurrentStatePlannerAdapter(
            backend,
            SimpleNamespace(act=lambda _: None),
        )

    backend.observation_track = "state"
    backend.render_observations = False
    with pytest.raises(_MODULE.P0GraspPlannerError, match="rendering"):
        _MODULE.CurrentStatePlannerAdapter(
            backend,
            SimpleNamespace(act=lambda _: None),
        )


def test_p0_causal_fingerprint_excludes_rendered_media() -> None:
    state = {
        "metadata": _sha("metadata-a"),
        "proprio/robot0": _sha("proprio-a"),
        "privileged/object": _sha("state-a"),
        "rgb/agentview": _sha("rgb-a"),
        "depth_m/agentview": _sha("depth-a"),
        "segmentation/agentview": _sha("seg-a"),
    }
    changed_media = {**state, "rgb/agentview": _sha("rgb-b")}
    assert _MODULE.causal_observation_fingerprint(
        SimpleNamespace(component_sha256=state)
    ) == _MODULE.causal_observation_fingerprint(
        SimpleNamespace(component_sha256=changed_media)
    )


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
    events_since_last_observation: tuple[Any, ...] = ()


class _Backend:
    backend_id = "mjwarp_gpu_v1"
    task_id = "p0_grasp"
    num_envs = 1
    device = SimpleNamespace(type="cuda")
    observation_track = "state"
    render_observations = True
    cohort_horizon_steps = 2
    stable_identity = MappingProxyType(
        {
            "backend_id": "mjwarp_gpu_v1",
            "source_commit": "a" * 40,
            "render_observations": True,
        }
    )

    def __init__(self) -> None:
        self.step = 0
        self.terminal_calls = 0

    def next_requests(self) -> tuple[Any, ...]:
        return (SimpleNamespace(episode_id="episode-0"),)

    def reset(self) -> Any:
        self.step = 0
        self.terminal_calls = 0
        return SimpleNamespace(
            episode_ids=("episode-0",),
            manifest_sha256=_sha("manifest"),
            manifest_ordinals=(0,),
            seeds=(20261050,),
        )

    def materialize_health_audit(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray([value], dtype=np.int64)
            for name, value in _health(self.step, terminal=self.step == 2).items()
        }

    def materialize_review_rgb(
        self, lanes: tuple[int, ...]
    ) -> tuple[dict[str, np.ndarray], ...]:
        assert lanes == (0,)
        return (_frame(self.step),)

    def materialize_teacher_observations(
        self, lanes: tuple[int, ...]
    ) -> tuple[_Observation, ...]:
        assert lanes == (0,)
        return (
            _Observation(
                episode_id="episode-0",
                task_id="p0_grasp",
                physics_step=25 * self.step,
                control_step=self.step,
                policy_step=self.step,
                component_sha256=MappingProxyType(
                    {
                        "metadata": _sha(f"metadata-{self.step}"),
                        "privileged/state": _sha(f"state-{self.step}"),
                        "rgb/agentview": _sha(f"rgb-{self.step}"),
                    }
                ),
            ),
        )

    def step_device(self, action: Any) -> Any:
        assert np.asarray(action).shape == (1, 7)
        self.step += 1
        return SimpleNamespace(done=np.asarray([self.step == 2], dtype=np.bool_))

    def materialize_terminal_ledger_once(
        self,
        lanes: tuple[int, ...],
        episode_ids: tuple[str, ...],
    ) -> tuple[Any, ...]:
        assert lanes == (0,) and episode_ids == ("episode-0",)
        self.terminal_calls += 1
        assert self.terminal_calls == 1
        return (
            SimpleNamespace(
                lane=0,
                episode_id="episode-0",
                task_id="p0_grasp",
                physics_step=25 * self.step,
                control_step=self.step,
                policy_step=self.step,
            ),
        )


class _Planner:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def act(self, observation: _Observation) -> _ActionCommand:
        self.calls.append(observation.policy_step)
        return _ActionCommand(
            mode=_ActionMode.E7,
            values=np.full(7, 0.1 * (observation.policy_step + 1)),
            policy_step=observation.policy_step,
        )


def test_online_adapter_and_fresh_replay_share_causal_state_not_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    online = _Backend()
    planner = _Planner()
    adapter = _MODULE.CurrentStatePlannerAdapter(online, planner)
    adapter.reset()
    _result, online_rows = adapter.run_natural_termination()
    assert planner.calls == [0, 1]
    assert adapter.tape.complete
    assert len(adapter.scene_frames) == 3
    assert online.terminal_calls == 1

    fresh = _Backend()
    replay = _MODULE.replay_action_trajectory(fresh, adapter.tape)
    assert replay.passed
    assert len(replay.steps) == 2
    assert replay.terminal_rows[0].episode_id == online_rows[0].episode_id
    assert fresh.terminal_calls == 1
