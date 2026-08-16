# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load(
    "_test_gpu_planner_t1_xyz_e0_production",
    "examples/embodiment/run_gpu_planner_t1_xyz_e0.py",
)


@dataclass(frozen=True)
class _Event:
    name: str
    physics_step: int
    time_s: float


class _Observation:
    def __init__(self, index: int, *, raw: bool) -> None:
        self.episode_id = "episode-1"
        self.task_id = "t1_xyz"
        self.physics_step = index * 25
        self.control_step = index
        self.policy_step = index
        self.time_s = index * 0.05
        base = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        rgb = base + np.uint8(index) if raw else np.zeros_like(base)
        self.rgb = {
            "agentview": rgb,
            "robot0_eye_in_hand": np.flip(rgb, axis=1).copy(),
        }
        depth = np.full((8, 8, 1), 0.5 + index * 0.01, dtype=np.float32)
        self.depth_m = {
            "agentview": depth,
            "robot0_eye_in_hand": depth.copy(),
        }
        segmentation = np.full((8, 8, 1), index, dtype=np.int32)
        self.segmentation = {
            "agentview": segmentation,
            "robot0_eye_in_hand": segmentation.copy(),
        }
        self.proprio = {"robot0_proprio_state": np.arange(4, dtype=np.float64)}
        self.privileged = {"eef_pose_xyzw": np.arange(7, dtype=np.float64)}
        self.events_since_last_observation = (
            () if index == 0 else (_Event("approach", index * 25, index * 0.05),)
        )
        self.fingerprint_sha256 = f"{index + 1:064x}"


def test_raw_tape_keeps_gpu_visual_observation_not_state_placeholder(
    tmp_path: Path,
) -> None:
    state = [_Observation(0, raw=False), _Observation(1, raw=False)]
    raw = [_Observation(0, raw=True), _Observation(1, raw=True)]
    command = SimpleNamespace(
        mode=SimpleNamespace(value="E7"),
        values=np.zeros(7, dtype=np.float64),
        policy_step=0,
    )
    result = SimpleNamespace(
        observation=raw[1],
        terminated=True,
        truncated=False,
        success=True,
        termination_reason="success",
        active_stage_progress=1.0,
    )
    output = tmp_path / "raw.tape.npz"

    RUNNER._write_tape_npz(output, state, raw, [command], [result])

    with np.load(output, allow_pickle=False) as tape:
        assert tape["schema_version"].item() == "gpu-planner-t1-xyz-raw-tape-v1"
        assert np.array_equal(
            tape["rgb/agentview"], np.stack([row.rgb["agentview"] for row in raw])
        )
        assert not np.array_equal(
            tape["rgb/agentview"], np.stack([row.rgb["agentview"] for row in state])
        )
        assert tape["rgb/agentview"].dtype == np.uint8
        assert tape["step_result_success"].tolist() == [True]
        assert tape["step_result_termination_reason"].tolist() == ["success"]
        assert tape["raw_observation_fingerprint_sha256"].tolist() == [
            raw[0].fingerprint_sha256,
            raw[1].fingerprint_sha256,
        ]


def test_raw_tape_rejects_step_result_observation_drift(tmp_path: Path) -> None:
    state = [_Observation(0, raw=False), _Observation(1, raw=False)]
    raw = [_Observation(0, raw=True), _Observation(1, raw=True)]
    drifted = _Observation(2, raw=True)
    command = SimpleNamespace(
        mode=SimpleNamespace(value="E7"),
        values=np.zeros(7, dtype=np.float64),
        policy_step=0,
    )
    result = SimpleNamespace(
        observation=drifted,
        terminated=True,
        truncated=False,
        success=True,
        termination_reason="success",
        active_stage_progress=1.0,
    )

    with pytest.raises(ValueError, match="differs from the GPU step results"):
        RUNNER._write_tape_npz(
            tmp_path / "drifted.tape.npz", state, raw, [command], [result]
        )


def test_production_target_is_frozen_to_fifty() -> None:
    supervisor = _load(
        "_test_gpu_planner_t1_xyz_production_supervisor",
        "examples/embodiment/run_gpu_planner_t1_xyz_production_canary.py",
    )
    assert supervisor.TARGET_ACCEPTED == 50
    assert supervisor.TASK_ID == "t1_xyz"
