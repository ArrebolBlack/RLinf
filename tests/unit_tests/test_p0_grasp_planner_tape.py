from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


_MODULE_PATH = (
    Path(__file__).parents[2]
    / "rlinf"
    / "envs"
    / "dynamic_benchmark"
    / "p0_grasp_planner.py"
)
_SPEC = importlib.util.spec_from_file_location("_p0_grasp_planner_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class _Observation:
    fingerprint_sha256: str


def _identity() -> object:
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
        render_visual=True,
    )


def _entry(step: int) -> object:
    return _MODULE.PlannerTapeEntry(
        lane=0,
        episode_id="p0-train-0000",
        task_id="p0_grasp",
        policy_step=step,
        observation_fingerprint_sha256=_sha(f"observation-{step}"),
        action=(0.0,) * 7,
        diagnostics={"phase": "track"},
    )


def test_p0_tape_finalizes_at_natural_termination_and_round_trips() -> None:
    tape = _MODULE.ActionTrajectoryTape(_identity())
    for step in range(3):
        tape.append(_entry(step))
    tape.finalize(3)

    assert tape.complete
    assert tape.identity.horizon_steps == 3
    restored = _MODULE.ActionTrajectoryTape.from_dict(tape.as_dict())
    assert restored.complete
    assert restored.sha256 == tape.sha256
    assert restored.entries == tape.entries


def test_p0_action_tape_is_distinct_from_full_trajectory_payload() -> None:
    tape = _MODULE.ActionTrajectoryTape(_identity())
    tape.append(_entry(0))
    tape.finalize(1)

    action_payload = tape.action_dict()
    trajectory_payload = tape.as_dict()
    assert "diagnostics" not in action_payload["entries"][0]
    assert "observation_fingerprint_sha256" not in action_payload["entries"][0]
    assert trajectory_payload["entries"][0]["diagnostics"] == {"phase": "track"}
    assert action_payload["sha256"] != trajectory_payload["sha256"]


def test_p0_planner_fails_closed_without_gpu_visual_audit() -> None:
    backend = SimpleNamespace(
        backend_id="mjwarp_gpu_v1",
        task_id="p0_grasp",
        num_envs=1,
        device="cuda:0",
        render_visual=False,
    )
    with pytest.raises(_MODULE.P0GraspPlannerError, match="visual"):
        _MODULE.CurrentStatePlannerAdapter(backend, SimpleNamespace(act=lambda _: None))


def test_p0_causal_fingerprint_excludes_rendered_media() -> None:
    state = {
        "metadata": "metadata-a",
        "proprio/robot0": "proprio-a",
        "privileged/object": "state-a",
        "rgb/agentview": "rgb-a",
        "depth_m/agentview": "depth-a",
        "segmentation/agentview": "seg-a",
    }
    changed_media = {**state, "rgb/agentview": "rgb-b"}
    assert _MODULE.causal_observation_fingerprint(SimpleNamespace(component_sha256=state)) == _MODULE.causal_observation_fingerprint(SimpleNamespace(component_sha256=changed_media))
