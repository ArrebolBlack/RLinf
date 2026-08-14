# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only contracts for strict GPUPLAN0 t1_xyz evidence."""

from __future__ import annotations

import copy
import enum
import importlib.util
import json
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _load_runner(name: str) -> Any:
    path = Path(__file__).parents[2] / "examples" / "embodiment" / name
    spec = importlib.util.spec_from_file_location(f"_{path.stem}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E0 = _load_runner("run_gpu_planner_t1_xyz_e0.py")
D32 = _load_runner("run_gpu_planner_t1_xyz_d32.py")
STRICT = E0._STRICT


def _options(parser: Any) -> set[str]:
    return {option for action in parser._actions for option in action.option_strings}


def test_result_runners_expose_only_frozen_manifest_strict_evidence() -> None:
    e0_options = _options(E0._parser())
    d32_options = _options(D32._parser())

    assert "--manifest" in e0_options
    assert "--row-index" in e0_options
    assert "--export-dir" not in e0_options
    assert "--manifest-row" not in e0_options
    assert "--replay-policy" not in e0_options
    assert "--episodes" not in e0_options
    assert "--planner-lookahead-s" not in e0_options
    assert "--manifest" in d32_options
    assert "--replay-policy" not in d32_options
    assert "--episodes" not in d32_options


def test_e0_review_rgb_defaults_to_policy_resolution() -> None:
    assert E0._parser().get_default("image_size") == 224


def test_frozen_execution_uses_state_e7_and_canonical_evaluator() -> None:
    assert STRICT.EXECUTION_CONTRACT["observation_track"] == "state"
    assert STRICT.EXECUTION_CONTRACT["action_mode"] == "E7"
    assert (
        STRICT.EXECUTION_CONTRACT["replay_intermediate_event_drift_blocking"] is False
    )
    assert STRICT.QUALITY_SCHEMA_VERSION == "db0-episode-task-quality-v2"
    assert STRICT.QUALITY_EVALUATOR_ID == "mjwarp_gpu_v1"


def test_task_quality_margin_diagnostic_serializes_lane_zero() -> None:
    class _Backend:
        @staticmethod
        def materialize_task_quality_audit() -> dict[str, np.ndarray]:
            return {
                "physics_step": np.asarray([1250], dtype=np.int64),
                "stage_index": np.asarray([4], dtype=np.int32),
                "bilateral_steps": np.asarray([30], dtype=np.int32),
                "max_bilateral_steps": np.asarray([31], dtype=np.int32),
                "success": np.asarray([0], dtype=np.int32),
                "terminated": np.asarray([0], dtype=np.int32),
                "truncated": np.asarray([0], dtype=np.int32),
                "event_mask": np.asarray([15], dtype=np.int32),
                "event_physics_step": np.arange(11, dtype=np.int64),
                "quality_physics_sample_count": np.asarray([900], dtype=np.int64),
                "quality_has_post_hold_sample": np.asarray([1], dtype=np.int32),
                "quality_maximum_lift_clearance_m": np.asarray(
                    [0.049], dtype=np.float32
                ),
                "quality_maximum_axis_error_rad": np.asarray([0.02], dtype=np.float32),
                "quality_error": np.asarray([0], dtype=np.int32),
                "quality_has_bilateral_hold_margin": np.asarray([1], dtype=np.int32),
                "quality_bilateral_hold_downstream_margin_m": np.asarray(
                    [0.3], dtype=np.float32
                ),
            }

    diagnostic = E0._task_quality_margin_diagnostic(_Backend())

    assert diagnostic["available"] is True
    assert diagnostic["physics_step"] == 1250
    assert diagnostic["quality_maximum_lift_clearance_m"] == pytest.approx(0.049)
    assert diagnostic["event_physics_step"] == list(range(11))
    json.dumps(diagnostic)


def test_d32_does_not_count_replay_audit_mismatch_as_completed_or_success() -> None:
    summary = D32._summary(
        [
            {
                "manifest_index": 0,
                "status": "completed_replay_audit_mismatch",
                "success": True,
                "replay": {"policy": "audit", "passed": False},
            }
        ]
    )

    assert summary["valid_completed"] == 0
    assert summary["valid_successes"] == 0
    assert summary["success_rate"] is None
    assert summary["complete_cohort"] is False


class _ActionMode(enum.Enum):
    E7 = "E7"


@dataclass(frozen=True)
class _Observation:
    episode_id: str
    task_id: str
    physics_step: int
    control_step: int
    policy_step: int
    time_s: float
    rgb: Any
    depth_m: Any
    segmentation: Any
    proprio: Any
    privileged: Any
    events_since_last_observation: tuple[Any, ...]


def _observation(step: int) -> Any:
    cameras = ("agentview", "robot0_eye_in_hand")
    privileged = {"object": np.asarray([step], dtype=np.float64)}
    for camera in cameras:
        privileged[f"{camera}_target_visible_pixels"] = np.asarray(
            [20 + step], dtype=np.int64
        )
        privileged[f"{camera}_target_image_fraction"] = np.asarray(
            [0.25 + step], dtype=np.float64
        )
        privileged[f"{camera}_occluder_visible_pixels"] = np.asarray(
            [3 + step], dtype=np.int64
        )
    return _Observation(
        episode_id="episode-0",
        task_id="t1_xyz",
        physics_step=25 * step,
        control_step=step,
        policy_step=step,
        time_s=0.05 * step,
        rgb={
            camera: np.full((2, 2, 3), 7 + step + index, dtype=np.uint8)
            for index, camera in enumerate(cameras)
        },
        depth_m={
            camera: np.full((2, 2, 1), 0.5 + step + index, dtype=np.float32)
            for index, camera in enumerate(cameras)
        },
        segmentation={
            camera: np.full((2, 2, 1), 2 + step + index, dtype=np.int32)
            for index, camera in enumerate(cameras)
        },
        proprio={"robot0": np.asarray([step], dtype=np.float64)},
        privileged=privileged,
        events_since_last_observation=(),
    )


def _numeric_drift_observation(step: int) -> Any:
    value = _observation(step)
    rgb = {name: np.array(array, copy=True) for name, array in value.rgb.items()}
    rgb["agentview"][0, 0, 0] += 1
    return replace(
        value,
        rgb=rgb,
        proprio={"robot0": np.asarray([step + 1.0e-3], dtype=np.float64)},
        events_since_last_observation=(
            SimpleNamespace(
                name="fingerpad_contact",
                physics_step=25 * step,
                time_s=0.05 * step,
            ),
        ),
    )


def _request() -> Any:
    return SimpleNamespace(
        action_mode=_ActionMode.E7,
        api_version="gpu-native-api-v0.2",
        episode_id="episode-0",
        factors={},
        object_mode="dynamic",
        observation_track="state",
        reset_mode="exact",
        seed=1000,
        split="test_id",
        task_id="t1_xyz",
    )


def test_state_planner_and_review_materialization_are_separate() -> None:
    raw = _observation(0)
    state, review = E0._state_and_review_observation(raw)

    for camera in ("agentview", "robot0_eye_in_hand"):
        assert np.array_equal(review[camera], raw.rgb[camera])
        assert np.count_nonzero(state.rgb[camera]) == 0
        assert np.all(state.depth_m[camera] == 1)
        assert np.count_nonzero(state.segmentation[camera]) == 0
        for suffix in E0._VISUAL_PRIVILEGED_SUFFIXES:
            assert np.count_nonzero(state.privileged[f"{camera}_{suffix}"]) == 0
    assert np.array_equal(state.privileged["object"], raw.privileged["object"])


def _terminal_ledger() -> Any:
    quality_payload = {
        "episode_id": "episode-0",
        "task_id": "t1_xyz",
        "schema_version": STRICT.QUALITY_SCHEMA_VERSION,
        "evaluator_backend_id": STRICT.QUALITY_EVALUATOR_ID,
        "terminal": True,
        "components": {},
    }
    row = SimpleNamespace(
        lane=0,
        episode_id="episode-0",
        task_id="t1_xyz",
        outcome=SimpleNamespace(value="success"),
        terminated=True,
        truncated=False,
        success=True,
        termination_reason="success",
        physics_step=25,
        control_step=1,
        policy_step=1,
        completion=1.0,
        events=(),
        task_quality=SimpleNamespace(to_dict=lambda: dict(quality_payload)),
    )
    return SimpleNamespace(rows=(row,))


def test_fresh_replay_rejects_backend_source_identity_mismatch(
    monkeypatch: Any,
) -> None:
    api = types.ModuleType("se3_wam.benchmark.api")

    class _ActionCommand:
        def __init__(self, *, mode: Any, values: Any, policy_step: int) -> None:
            self.mode = mode
            self.values = values
            self.policy_step = policy_step

    api.ActionCommand = _ActionCommand
    monkeypatch.setitem(sys.modules, "se3_wam", types.ModuleType("se3_wam"))
    monkeypatch.setitem(
        sys.modules,
        "se3_wam.benchmark",
        types.ModuleType("se3_wam.benchmark"),
    )
    monkeypatch.setitem(sys.modules, "se3_wam.benchmark.api", api)

    gpu_backend = types.ModuleType("rlinf.envs.dynamic_benchmark.gpu_backend")
    gpu_backend.assert_terminal_ledger_exact_once = lambda backend, rows: (
        "second_consumption_rejected",
    )
    rlinf = types.ModuleType("rlinf")
    rlinf.__path__ = []
    envs = types.ModuleType("rlinf.envs")
    envs.__path__ = []
    dynamic = types.ModuleType("rlinf.envs.dynamic_benchmark")
    dynamic.__path__ = []
    monkeypatch.setitem(sys.modules, "rlinf", rlinf)
    monkeypatch.setitem(sys.modules, "rlinf.envs", envs)
    monkeypatch.setitem(sys.modules, "rlinf.envs.dynamic_benchmark", dynamic)
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.dynamic_benchmark.gpu_backend",
        gpu_backend,
    )

    request = _request()

    class _FreshReplay:
        provenance = SimpleNamespace(
            backend_id="mjwarp_gpu_v1",
            git_commit="b" * 40,
            git_tree="c" * 40,
        )
        last_terminal_ledger = _terminal_ledger()
        frozen_requests = (request,)

        def reset(self, requests: Any) -> tuple[Any, ...]:
            assert len(tuple(requests)) == 1
            return (_observation(0),)

        def policy_steps(self) -> np.ndarray:
            return np.asarray([0], dtype=np.int64)

        def step(self, commands: Any) -> tuple[Any, ...]:
            assert len(tuple(commands)) == 1
            return (
                SimpleNamespace(
                    observation=_observation(1),
                    terminated=True,
                    truncated=False,
                    success=True,
                    termination_reason="success",
                ),
            )

        def close(self) -> None:
            pass

    primary = SimpleNamespace(
        provenance=SimpleNamespace(
            backend_id="mjwarp_gpu_v1",
            git_commit="a" * 40,
            git_tree="c" * 40,
        ),
        new_replay_backend=lambda: _FreshReplay(),
    )
    command = SimpleNamespace(
        mode=_ActionMode.E7,
        policy_step=0,
        values=np.zeros(7, dtype=np.float64),
    )
    state_0, review_0 = E0._state_and_review_observation(_observation(0))
    state_1, review_1 = E0._state_and_review_observation(_observation(1))
    replay = E0._replay(
        backend=primary,
        request=request,
        observations=(state_0, state_1),
        reviews=(review_0, review_1),
        commands=(command,),
        outcomes=((True, False, True, "success"),),
        terminal_ledger=_terminal_ledger(),
    )

    assert replay["passed"] is False
    assert replay["backend_identity_exact"] is False
    assert replay["fresh_backend_distinct"] is True
    assert replay["observation_tape_exact"] is True
    assert replay["review_tape_exact"] is True
    assert replay["outcomes_exact"] is True
    assert replay["terminal_ledger_exact"] is True
    assert replay["terminal_ledger_exact_once"] is True

    matching_primary = SimpleNamespace(
        provenance=_FreshReplay.provenance,
        new_replay_backend=lambda: _FreshReplay(),
    )
    passed = E0._replay(
        backend=matching_primary,
        request=request,
        observations=(state_0, state_1),
        reviews=(review_0, review_1),
        commands=(command,),
        outcomes=((True, False, True, "success"),),
        terminal_ledger=_terminal_ledger(),
    )
    assert passed["passed"] is True


def test_fresh_replay_records_first_observation_divergence(
    monkeypatch: Any,
) -> None:
    api = types.ModuleType("se3_wam.benchmark.api")

    class _ActionCommand:
        def __init__(self, *, mode: Any, values: Any, policy_step: int) -> None:
            self.mode = mode
            self.values = values
            self.policy_step = policy_step

    api.ActionCommand = _ActionCommand
    monkeypatch.setitem(sys.modules, "se3_wam", types.ModuleType("se3_wam"))
    monkeypatch.setitem(
        sys.modules,
        "se3_wam.benchmark",
        types.ModuleType("se3_wam.benchmark"),
    )
    monkeypatch.setitem(sys.modules, "se3_wam.benchmark.api", api)

    gpu_backend = types.ModuleType("rlinf.envs.dynamic_benchmark.gpu_backend")
    gpu_backend.assert_terminal_ledger_exact_once = lambda backend, rows: (
        "second_consumption_rejected",
    )
    rlinf = types.ModuleType("rlinf")
    rlinf.__path__ = []
    envs = types.ModuleType("rlinf.envs")
    envs.__path__ = []
    dynamic = types.ModuleType("rlinf.envs.dynamic_benchmark")
    dynamic.__path__ = []
    monkeypatch.setitem(sys.modules, "rlinf", rlinf)
    monkeypatch.setitem(sys.modules, "rlinf.envs", envs)
    monkeypatch.setitem(sys.modules, "rlinf.envs.dynamic_benchmark", dynamic)
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.dynamic_benchmark.gpu_backend",
        gpu_backend,
    )

    provenance = SimpleNamespace(
        backend_id="mjwarp_gpu_v1",
        git_commit="a" * 40,
        git_tree="b" * 40,
    )
    request = _request()

    class _DivergentReplay:
        last_terminal_ledger = _terminal_ledger()
        frozen_requests = (request,)

        def __init__(self) -> None:
            self.provenance = provenance

        def reset(self, requests: Any) -> tuple[Any, ...]:
            assert len(tuple(requests)) == 1
            return (_observation(0),)

        def policy_steps(self) -> np.ndarray:
            return np.asarray([0], dtype=np.int64)

        def step(self, commands: Any) -> tuple[Any, ...]:
            assert len(tuple(commands)) == 1
            return (
                SimpleNamespace(
                    observation=_observation(2),
                    terminated=True,
                    truncated=False,
                    success=True,
                    termination_reason="success",
                ),
            )

        def close(self) -> None:
            pass

    primary = SimpleNamespace(
        provenance=provenance,
        new_replay_backend=lambda: _DivergentReplay(),
    )
    command = SimpleNamespace(
        mode=_ActionMode.E7,
        policy_step=0,
        values=np.linspace(-0.6, 0.6, 7, dtype=np.float64),
    )
    state_0, review_0 = E0._state_and_review_observation(_observation(0))
    state_1, review_1 = E0._state_and_review_observation(_observation(1))

    replay = E0._replay(
        backend=primary,
        request=request,
        observations=(state_0, state_1),
        reviews=(review_0, review_1),
        commands=(command,),
        outcomes=((True, False, True, "success"),),
        terminal_ledger=_terminal_ledger(),
    )

    assert replay["passed"] is False
    assert replay["observation_tape_exact"] is False
    assert replay["first_divergence"]["channel"] == "observation_semantics"
    assert replay["first_divergence"]["control_step"] == 1
    assert replay["first_divergence"]["mismatch"]["path"].startswith("observations[1]")
    assert replay["first_divergence"]["transition_action"] == {
        "mode": "E7",
        "policy_step": 0,
        "values": np.linspace(-0.6, 0.6, 7, dtype=np.float64).tolist(),
    }


def test_fresh_replay_reports_numeric_drift_without_blocking(monkeypatch: Any) -> None:
    api = types.ModuleType("se3_wam.benchmark.api")

    class _ActionCommand:
        def __init__(self, *, mode: Any, values: Any, policy_step: int) -> None:
            self.mode = mode
            self.values = values
            self.policy_step = policy_step

    api.ActionCommand = _ActionCommand
    monkeypatch.setitem(sys.modules, "se3_wam", types.ModuleType("se3_wam"))
    monkeypatch.setitem(
        sys.modules, "se3_wam.benchmark", types.ModuleType("se3_wam.benchmark")
    )
    monkeypatch.setitem(sys.modules, "se3_wam.benchmark.api", api)

    gpu_backend = types.ModuleType("rlinf.envs.dynamic_benchmark.gpu_backend")
    gpu_backend.assert_terminal_ledger_exact_once = lambda backend, rows: (
        "second_consumption_rejected",
    )
    rlinf = types.ModuleType("rlinf")
    rlinf.__path__ = []
    envs = types.ModuleType("rlinf.envs")
    envs.__path__ = []
    dynamic = types.ModuleType("rlinf.envs.dynamic_benchmark")
    dynamic.__path__ = []
    monkeypatch.setitem(sys.modules, "rlinf", rlinf)
    monkeypatch.setitem(sys.modules, "rlinf.envs", envs)
    monkeypatch.setitem(sys.modules, "rlinf.envs.dynamic_benchmark", dynamic)
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.dynamic_benchmark.gpu_backend",
        gpu_backend,
    )

    request = _request()
    provenance = SimpleNamespace(
        backend_id="mjwarp_gpu_v1",
        git_commit="a" * 40,
        git_tree="b" * 40,
    )

    class _NumericDriftReplay:
        frozen_requests = (request,)
        last_terminal_ledger = _terminal_ledger()

        def __init__(self) -> None:
            self.provenance = provenance

        def reset(self, requests: Any) -> tuple[Any, ...]:
            assert tuple(requests) == (request,)
            return (_observation(0),)

        def policy_steps(self) -> np.ndarray:
            return np.asarray([0], dtype=np.int64)

        def step(self, commands: Any) -> tuple[Any, ...]:
            assert len(tuple(commands)) == 1
            return (
                SimpleNamespace(
                    observation=_numeric_drift_observation(1),
                    terminated=True,
                    truncated=False,
                    success=True,
                    termination_reason="success",
                ),
            )

        def close(self) -> None:
            pass

    primary = SimpleNamespace(
        provenance=provenance,
        new_replay_backend=lambda: _NumericDriftReplay(),
    )
    command = SimpleNamespace(
        mode=_ActionMode.E7,
        policy_step=0,
        values=np.linspace(-0.6, 0.6, 7, dtype=np.float64),
    )
    state_0, review_0 = E0._state_and_review_observation(_observation(0))
    state_1, review_1 = E0._state_and_review_observation(_observation(1))

    replay = E0._replay(
        backend=primary,
        request=request,
        observations=(state_0, state_1),
        reviews=(review_0, review_1),
        commands=(command,),
        outcomes=((True, False, True, "success"),),
        terminal_ledger=_terminal_ledger(),
    )

    assert replay["passed"] is True
    assert replay["first_divergence"] is None
    assert replay["observation_semantic_structure_exact"] is True
    assert replay["review_semantic_structure_exact"] is True
    assert replay["observation_tape_exact"] is False
    assert replay["observation_event_sequence_exact"] is False
    assert replay["review_tape_exact"] is False
    assert replay["observation_event_drift"]["blocking"] is False
    assert replay["observation_numeric_drift"]["blocking"] is False
    assert replay["review_numeric_drift"]["blocking"] is False


def test_fresh_replay_accepts_one_bounded_terminal_grid_hold(monkeypatch: Any) -> None:
    api = types.ModuleType("se3_wam.benchmark.api")

    class _ActionCommand:
        def __init__(self, *, mode: Any, values: Any, policy_step: int) -> None:
            self.mode = mode
            self.values = values
            self.policy_step = policy_step

    api.ActionCommand = _ActionCommand
    monkeypatch.setitem(sys.modules, "se3_wam", types.ModuleType("se3_wam"))
    monkeypatch.setitem(
        sys.modules, "se3_wam.benchmark", types.ModuleType("se3_wam.benchmark")
    )
    monkeypatch.setitem(sys.modules, "se3_wam.benchmark.api", api)

    gpu_backend = types.ModuleType("rlinf.envs.dynamic_benchmark.gpu_backend")
    gpu_backend.assert_terminal_ledger_exact_once = lambda backend, rows: (
        "second_consumption_rejected",
    )
    rlinf = types.ModuleType("rlinf")
    rlinf.__path__ = []
    envs = types.ModuleType("rlinf.envs")
    envs.__path__ = []
    dynamic = types.ModuleType("rlinf.envs.dynamic_benchmark")
    dynamic.__path__ = []
    monkeypatch.setitem(sys.modules, "rlinf", rlinf)
    monkeypatch.setitem(sys.modules, "rlinf.envs", envs)
    monkeypatch.setitem(sys.modules, "rlinf.envs.dynamic_benchmark", dynamic)
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.dynamic_benchmark.gpu_backend",
        gpu_backend,
    )

    request = _request()
    provenance = SimpleNamespace(
        backend_id="mjwarp_gpu_v1",
        device_name="NVIDIA A100-SXM4-80GB",
        device_ordinal=0,
        device_platform="cuda",
        git_commit="1" * 40,
        git_tree="2" * 40,
        implementation_version="test",
        physical_device_identity_source="warp_cuda_driver",
        physical_device_pci_bus_id="00000000:01:00.0",
        physical_device_uuid="GPU-test",
        precision="float32",
        runtime_versions={},
    )

    def _quality_audit(*, terminal: bool, stage_index: int) -> dict[str, np.ndarray]:
        physics_step = 26 if terminal else 25
        return {
            "physics_step": np.asarray([physics_step], dtype=np.int64),
            "stage_index": np.asarray([stage_index], dtype=np.int32),
            "bilateral_steps": np.asarray([10], dtype=np.int32),
            "max_bilateral_steps": np.asarray([10], dtype=np.int32),
            "success": np.asarray([int(terminal)], dtype=np.int32),
            "terminated": np.asarray([int(terminal)], dtype=np.int32),
            "truncated": np.asarray([0], dtype=np.int32),
            "event_mask": np.asarray([0], dtype=np.int32),
            "event_physics_step": np.full(11, -1, dtype=np.int64),
            "quality_physics_sample_count": np.asarray([physics_step], dtype=np.int64),
            "quality_has_post_hold_sample": np.asarray([1], dtype=np.int32),
            "quality_maximum_lift_clearance_m": np.asarray([0.09], dtype=np.float32),
            "quality_maximum_axis_error_rad": np.asarray([0.02], dtype=np.float32),
            "quality_error": np.asarray([0], dtype=np.int32),
            "quality_has_bilateral_hold_margin": np.asarray([1], dtype=np.int32),
            "quality_bilateral_hold_downstream_margin_m": np.asarray(
                [0.4], dtype=np.float32
            ),
        }

    class _OneStepLateReplay:
        frozen_requests = (request,)

        def __init__(self, *, stage_index: int = 4) -> None:
            self.provenance = provenance
            self.step_count = 0
            self.stage_index = stage_index
            self.last_terminal_ledger = None

        def reset(self, requests: Any) -> tuple[Any, ...]:
            assert tuple(requests) == (request,)
            return (_observation(0),)

        def policy_steps(self) -> np.ndarray:
            return np.asarray([self.step_count], dtype=np.int64)

        def step(self, commands: Any) -> tuple[Any, ...]:
            (command,) = tuple(commands)
            assert np.array_equal(command.values, np.zeros(7, dtype=np.float64))
            self.step_count += 1
            if self.step_count == 1:
                return (
                    SimpleNamespace(
                        observation=_observation(1),
                        terminated=False,
                        truncated=False,
                        success=False,
                        termination_reason=None,
                    ),
                )
            if self.stage_index == 4:
                self.last_terminal_ledger = _terminal_ledger()
                return (
                    SimpleNamespace(
                        observation=_observation(2),
                        terminated=True,
                        truncated=False,
                        success=True,
                        termination_reason="success",
                    ),
                )
            raise AssertionError("terminal grace must not run outside stage 4")

        def materialize_task_quality_audit(self) -> dict[str, np.ndarray]:
            terminal = self.last_terminal_ledger is not None
            return _quality_audit(
                terminal=terminal,
                stage_index=5 if terminal else self.stage_index,
            )

        def close(self) -> None:
            pass

    command = SimpleNamespace(
        mode=_ActionMode.E7,
        policy_step=0,
        values=np.zeros(7, dtype=np.float64),
    )
    state_0, review_0 = E0._state_and_review_observation(_observation(0))
    state_1, review_1 = E0._state_and_review_observation(_observation(1))
    replay_backend = _OneStepLateReplay()
    primary = SimpleNamespace(
        provenance=provenance,
        new_replay_backend=lambda: replay_backend,
    )

    replay = E0._replay(
        backend=primary,
        request=request,
        observations=(state_0, state_1),
        reviews=(review_0, review_1),
        commands=(command,),
        outcomes=((True, False, True, "success"),),
        terminal_ledger=_terminal_ledger(),
    )

    assert replay["passed"] is True
    assert replay["action_tape_exact"] is True
    assert replay["outcomes_exact"] is False
    assert replay["semantic_outcomes_exact"] is True
    assert replay["terminal_ledger_semantic_exact"] is True
    assert replay["terminal_ledger_exact_once"] is True
    assert replay["first_divergence"] is None
    assert replay["terminal_grace"] == {
        "schema_version": "gpu-planner-terminal-grid-grace-v1",
        "mode": "zero_order_hold_last_primary_action_v1",
        "max_control_steps": 1,
        "attempted": True,
        "accepted": True,
        "reason": "semantic_terminal_reached",
        "held_action_values_exact": True,
        "control_steps": 1,
        "physics_steps": 1,
        "stage_index_before": 4,
        "stage_index_after": 5,
        "outcome": [True, False, True, "success"],
        "observation_sha256": E0._observation_digest(
            E0._state_and_review_observation(_observation(2))[0]
        ),
        "review_sha256": E0._review_digest(
            E0._state_and_review_observation(_observation(2))[1]
        ),
    }

    stage_two_backend = _OneStepLateReplay(stage_index=2)
    stage_two = E0._replay(
        backend=SimpleNamespace(
            provenance=provenance,
            new_replay_backend=lambda: stage_two_backend,
        ),
        request=request,
        observations=(state_0, state_1),
        reviews=(review_0, review_1),
        commands=(command,),
        outcomes=((True, False, True, "success"),),
        terminal_ledger=_terminal_ledger(),
    )
    assert stage_two["passed"] is False
    assert stage_two_backend.step_count == 1
    assert stage_two["terminal_grace"]["attempted"] is False


def test_fresh_replay_early_terminal_writes_semantic_blocker(monkeypatch: Any) -> None:
    api = types.ModuleType("se3_wam.benchmark.api")

    class _ActionCommand:
        def __init__(self, *, mode: Any, values: Any, policy_step: int) -> None:
            self.mode = mode
            self.values = values
            self.policy_step = policy_step

    api.ActionCommand = _ActionCommand
    monkeypatch.setitem(sys.modules, "se3_wam", types.ModuleType("se3_wam"))
    monkeypatch.setitem(
        sys.modules,
        "se3_wam.benchmark",
        types.ModuleType("se3_wam.benchmark"),
    )
    monkeypatch.setitem(sys.modules, "se3_wam.benchmark.api", api)

    gpu_backend = types.ModuleType("rlinf.envs.dynamic_benchmark.gpu_backend")
    gpu_backend.assert_terminal_ledger_exact_once = lambda backend, rows: (
        "second_consumption_rejected",
    )
    rlinf = types.ModuleType("rlinf")
    rlinf.__path__ = []
    envs = types.ModuleType("rlinf.envs")
    envs.__path__ = []
    dynamic = types.ModuleType("rlinf.envs.dynamic_benchmark")
    dynamic.__path__ = []
    monkeypatch.setitem(sys.modules, "rlinf", rlinf)
    monkeypatch.setitem(sys.modules, "rlinf.envs", envs)
    monkeypatch.setitem(sys.modules, "rlinf.envs.dynamic_benchmark", dynamic)
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.dynamic_benchmark.gpu_backend",
        gpu_backend,
    )

    request = _request()
    provenance = SimpleNamespace(
        backend_id="mjwarp_gpu_v1",
        device_name="NVIDIA A100-SXM4-80GB",
        device_ordinal=0,
        device_platform="cuda",
        git_commit="1" * 40,
        git_tree="2" * 40,
        implementation_version="test",
        physical_device_identity_source="warp_cuda_driver",
        physical_device_pci_bus_id="00000000:01:00.0",
        physical_device_uuid="GPU-test",
        precision="float32",
        runtime_versions={},
    )

    class _EarlyTerminalReplay:
        frozen_requests = (request,)
        last_terminal_ledger = _terminal_ledger()

        def __init__(self) -> None:
            self.provenance = provenance
            self.step_count = 0
            self.closed = False

        def reset(self, requests: Any) -> tuple[Any, ...]:
            assert tuple(requests) == (request,)
            return (_observation(0),)

        def policy_steps(self) -> np.ndarray:
            return np.asarray([self.step_count], dtype=np.int64)

        def step(self, commands: Any) -> tuple[Any, ...]:
            assert len(tuple(commands)) == 1
            if self.step_count == 0:
                self.step_count = 1
                return (
                    SimpleNamespace(
                        observation=_observation(1),
                        terminated=True,
                        truncated=False,
                        success=True,
                        termination_reason="success",
                    ),
                )
            return (None,)

        def close(self) -> None:
            self.closed = True

    replay_backend = _EarlyTerminalReplay()
    primary = SimpleNamespace(
        provenance=provenance,
        new_replay_backend=lambda: replay_backend,
    )
    commands = tuple(
        SimpleNamespace(
            mode=_ActionMode.E7,
            policy_step=index,
            values=np.zeros(7, dtype=np.float64),
        )
        for index in range(2)
    )
    observations_and_reviews = [
        E0._state_and_review_observation(_observation(index)) for index in range(3)
    ]

    replay = E0._replay(
        backend=primary,
        request=request,
        observations=tuple(value[0] for value in observations_and_reviews),
        reviews=tuple(value[1] for value in observations_and_reviews),
        commands=commands,
        outcomes=(
            (False, False, False, None),
            (True, False, True, "success"),
        ),
        terminal_ledger=_terminal_ledger(),
    )

    assert replay_backend.closed is True
    assert replay["passed"] is False
    assert replay["action_tape_exact"] is True
    assert replay["outcomes_exact"] is False
    assert replay["replay_stop"] == {
        "reason": "backend_returned_none_after_terminal",
        "command_index": 1,
        "policy_step": 1,
        "submitted_action_count": 2,
        "expected_action_count": 2,
    }
    assert replay["first_divergence"]["channel"] == (
        "replay_terminated_before_action_tape_end"
    )
    assert replay["first_divergence"]["control_step"] == 1


def test_semantic_replay_failure_evidence_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "result.semantic-replay-failure.json"
    payload = {
        "status": "blocked_semantic_fresh_replay",
        "evidence_passed": False,
        "qualification_completed": 0,
    }

    E0._write_json_exclusive(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        E0._write_json_exclusive(path, payload)


def _repositories() -> dict[str, dict[str, str]]:
    return {
        "research": {"commit": "1" * 40, "tree": "2" * 40},
        "se3_wam": {
            "commit": "3" * 40,
            "tree": "4" * 40,
            "mujoco_warp_gitlink": "5" * 40,
        },
        "mujoco_warp": {"commit": "5" * 40, "tree": "6" * 40},
        "rlinf": {"commit": "7" * 40, "tree": "8" * 40},
        "dynamic_benchmark": {
            "commit": "9" * 40,
            "tree": "a" * 40,
            "rlinf_gitlink": "7" * 40,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_manifest(tmp_path: Path, phase: str, *, exports: bool = False) -> Any:
    count = 1 if phase == "e0" else 32
    repositories = _repositories()
    source_sha = STRICT.payload_sha256(repositories)
    task_config_sha = "b" * 64
    rows = []
    for index in range(count):
        request = {
            "action_mode": "E7",
            "api_version": "gpu-native-api-v0.2",
            "episode_id": f"episode-{index:02d}",
            "factors": {"object_x_m": float(index) / 1000.0},
            "object_mode": "dynamic",
            "observation_track": "state",
            "reset_mode": "exact",
            "seed": 1000 + index,
            "split": "test_id",
            "task_id": "t1_xyz",
        }
        export_dir = tmp_path / "exports" / f"row-{index:02d}"
        row = {
            "manifest_index": index,
            "candidate_index": index,
            "source_group_id": f"source-{index:02d}",
            "pair_id": f"pair-{index:02d}",
            "pair_member_id": "t1_xyz",
            "export_dir": str(export_dir),
            "export_report_sha256": "c" * 64,
            "request_json_sha256": "d" * 64,
            "sha256sums_sha256": "e" * 64,
            "task_config_sha256": task_config_sha,
            "source_identity_sha256": source_sha,
            "request": request,
        }
        if exports:
            request_json = {
                "action_mode": request["action_mode"],
                "candidate_index": index,
                "episode_id": request["episode_id"],
                "factors": request["factors"],
                "object_mode": request["object_mode"],
                "observation_track": request["observation_track"],
                "pair_id": row["pair_id"],
                "pair_member_id": row["pair_member_id"],
                "reset_mode": request["reset_mode"],
                "seed": request["seed"],
                "source_group_id": row["source_group_id"],
                "split": request["split"],
                "task_id": request["task_id"],
            }
            report = {
                "task_id": "t1_xyz",
                "episode_id": request["episode_id"],
                "row_index": index,
                "task_config_sha256": task_config_sha,
            }
            _write_json(export_dir / "request.json", request_json)
            _write_json(export_dir / "export_report.json", report)
            request_sha = STRICT.sha256_file(export_dir / "request.json")
            report_sha = STRICT.sha256_file(export_dir / "export_report.json")
            (export_dir / "SHA256SUMS").write_text(
                f"{request_sha}  request.json\n{report_sha}  export_report.json\n",
                encoding="utf-8",
            )
            row["request_json_sha256"] = request_sha
            row["export_report_sha256"] = report_sha
            row["sha256sums_sha256"] = STRICT.sha256_file(export_dir / "SHA256SUMS")
        rows.append(row)
    payload = {
        "schema_version": STRICT.MANIFEST_SCHEMA_VERSION,
        "task_id": "t1_xyz",
        "backend_id": "mjwarp_gpu_v1",
        "phase": phase,
        "episode_count": count,
        "cohort_size": 1 if phase == "e0" else 8,
        "cohort_count": 1 if phase == "e0" else 4,
        "execution": dict(STRICT.EXECUTION_CONTRACT),
        "repositories": repositories,
        "source_identity_sha256": source_sha,
        "task_config_sha256": task_config_sha,
        "episode_ids": [row["request"]["episode_id"] for row in rows],
        "candidate_indices": list(range(count)),
        "rows": rows,
    }
    payload["manifest_sha256"] = STRICT.payload_sha256(payload)
    path = tmp_path / f"{phase}-manifest.json"
    _write_json(path, payload)
    return STRICT.load_frozen_manifest(
        path, expected_phase=phase, verify_exports=exports
    )


def _strict_result(
    manifest: Any,
    index: int,
    *,
    success: bool = True,
    termination_reason: str | None = None,
) -> dict[str, Any]:
    row = manifest.row(index)
    episode_id = row["request"]["episode_id"]
    reason = termination_reason or ("success" if success else "timeout")
    if success:
        outcome, terminated, truncated = "success", True, False
    elif reason == "timeout":
        outcome, terminated, truncated = "timeout", False, True
    else:
        outcome, terminated, truncated = "failure", True, False
    quality = (
        {
            "episode_id": episode_id,
            "task_id": "t1_xyz",
            "schema_version": STRICT.QUALITY_SCHEMA_VERSION,
            "evaluator_backend_id": STRICT.QUALITY_EVALUATOR_ID,
            "terminal": True,
        }
        if success
        else None
    )
    action_tape = [{"mode": "E7", "policy_step": 0, "values": [0.0] * 7}]
    trajectory_tape = [
        STRICT.payload_sha256({"row": index, "step": step}) for step in range(2)
    ]
    evidence_root = manifest.path.parent / "evidence" / f"row-{index:02d}"
    evidence_root.mkdir(parents=True, exist_ok=True)
    tape_file = (evidence_root / "result.tape.npz").resolve()
    visual_file = (evidence_root / "result.scene-wrist.gif").resolve()
    tape_file.write_bytes(f"npz-row-{index}".encode())
    visual_file.write_bytes(f"gif-row-{index}".encode())
    source_repositories = {
        name: {"path": f"C:/sealed/{name}", **dict(identity)}
        for name, identity in manifest.payload["repositories"].items()
    }
    se3_identity = manifest.payload["repositories"]["se3_wam"]
    provenance = {
        "backend_id": "mjwarp_gpu_v1",
        "device_platform": "cuda",
        "git_commit": se3_identity["commit"],
        "git_tree": se3_identity["tree"],
        "physical_device_uuid": "GPU-test",
        "runtime_versions": {},
    }
    terminal_ledger = [
        {
            "episode_id": episode_id,
            "task_id": "t1_xyz",
            "outcome": outcome,
            "terminated": terminated,
            "truncated": truncated,
            "success": success,
            "termination_reason": reason,
            "physics_step": 24,
            "control_step": 1,
            "policy_step": 1,
            "task_quality": quality,
        }
    ]
    return {
        "schema_version": STRICT.RESULT_SCHEMA_VERSION,
        "status": "completed_review_evidence",
        "evidence_passed": True,
        "task_id": "t1_xyz",
        "backend_id": "mjwarp_gpu_v1",
        "phase": manifest.phase,
        "manifest_index": index,
        "online_planner": True,
        "frozen_action_replay": False,
        "cpu_physics_or_env_fallback": False,
        "planner_observation_source": STRICT.EXECUTION_CONTRACT[
            "planner_observation_source"
        ],
        "planner_observation_track": "state",
        "review_materialization": STRICT.EXECUTION_CONTRACT["review_materialization"],
        "success": success,
        "termination_reason": reason,
        "control_steps": 1,
        "physics_steps": 25,
        "manifest": {
            "candidate_index": row["candidate_index"],
            "episode_id": episode_id,
            "manifest_index": index,
            "manifest_sha256": manifest.manifest_sha256,
            "source_identity_sha256": manifest.source_identity_sha256,
        },
        "reset_request": dict(row["request"]),
        "source_gate": {
            "passed": True,
            "repositories_exact": True,
            "source_identity_sha256": manifest.source_identity_sha256,
            "repositories": source_repositories,
        },
        "provenance": provenance,
        "quality": {
            "evaluator_backend_id": STRICT.QUALITY_EVALUATOR_ID,
            "schema_version": STRICT.QUALITY_SCHEMA_VERSION,
        },
        "terminal_ledger_gate": {
            "passed": True,
            "exact_once_second_consumption_rejected": True,
        },
        "replay": {
            "mode": "semantic_fresh_backend_v1",
            "passed": True,
            "fresh_backend_distinct": True,
            "backend_identity_exact": True,
            "reset_identity_exact": True,
            "action_tape_exact": True,
            "observation_semantic_structure_exact": True,
            "observation_event_sequence_exact": True,
            "observation_event_drift": {"blocking": False},
            "observation_tape_exact": True,
            "observation_numeric_drift": {"blocking": False},
            "review_semantic_structure_exact": True,
            "review_numeric_drift": {"blocking": False},
            "semantic_outcomes_exact": True,
            "outcomes_exact": True,
            "review_tape_exact": True,
            "source_identity_exact": True,
            "terminal_ledger_semantic_exact": True,
            "terminal_ledger_exact": True,
            "terminal_numeric_drift": {"blocking": False},
            "terminal_ledger_exact_once": True,
            "first_divergence": None,
            "primary_provenance": provenance,
            "replay_provenance": provenance,
            "replay_observation_sha256": STRICT.payload_sha256(trajectory_tape),
            "replay_review_sha256": STRICT.payload_sha256(["review-0", "review-1"]),
            "replay_ledger_sha256": STRICT.payload_sha256(terminal_ledger),
            "terminal_grace": {
                "schema_version": "gpu-planner-terminal-grid-grace-v1",
                "mode": "zero_order_hold_last_primary_action_v1",
                "max_control_steps": 1,
                "attempted": False,
                "accepted": False,
                "reason": "not_required_or_not_admissible",
                "held_action_values_exact": False,
                "control_steps": 0,
                "physics_steps": 0,
                "stage_index_before": None,
                "stage_index_after": None,
                "outcome": None,
                "observation_sha256": None,
                "review_sha256": None,
            },
        },
        "terminal_ledger": terminal_ledger,
        "action_tape": action_tape,
        "trajectory_tape": trajectory_tape,
        "evidence_export": {
            "passed": True,
            "action_tape_sha256": STRICT.payload_sha256(action_tape),
            "trajectory_tape_sha256": STRICT.payload_sha256(trajectory_tape),
            "tape_file": str(tape_file),
            "tape_file_sha256": STRICT.sha256_file(tape_file),
            "visual_file": str(visual_file),
            "visual_sha256": STRICT.sha256_file(visual_file),
        },
    }


def test_manifest_binds_export_hashes_and_rejects_row_reuse(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path / "e0", "e0", exports=True)
    request_path = manifest.export_dir(manifest.row(0)) / "request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest drifted"):
        STRICT.load_frozen_manifest(manifest.path, expected_phase="e0")

    d32 = _build_manifest(tmp_path / "d32", "d32")
    payload = json.loads(d32.path.read_text(encoding="utf-8"))
    payload["rows"][1]["export_dir"] = payload["rows"][0]["export_dir"]
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = STRICT.payload_sha256(payload)
    _write_json(d32.path, payload)
    with pytest.raises(ValueError, match="distinct export"):
        STRICT.load_frozen_manifest(
            d32.path,
            expected_phase="d32",
            verify_exports=False,
        )


def test_manifest_rejects_hybrid_observation_contract(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, "e0")
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["rows"][0]["request"]["observation_track"] = "hybrid"
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = STRICT.payload_sha256(payload)
    _write_json(manifest.path, payload)

    with pytest.raises(ValueError, match="observation_track"):
        STRICT.load_frozen_manifest(
            manifest.path,
            expected_phase="e0",
            verify_exports=False,
        )


def test_row_validator_rejects_nonblocking_identity_and_evidence_gates(
    tmp_path: Path,
) -> None:
    manifest = _build_manifest(tmp_path, "e0")
    row = manifest.row(0)
    valid = _strict_result(manifest, 0)
    STRICT.validate_result_for_row(valid, manifest=manifest, row=row)

    bounded_terminal_grid = copy.deepcopy(valid)
    bounded_terminal_grid["replay"]["outcomes_exact"] = False
    bounded_terminal_grid["replay"]["terminal_grace"] = {
        "schema_version": "gpu-planner-terminal-grid-grace-v1",
        "mode": "zero_order_hold_last_primary_action_v1",
        "max_control_steps": 1,
        "attempted": True,
        "accepted": True,
        "reason": "semantic_terminal_reached",
        "held_action_values_exact": True,
        "control_steps": 1,
        "physics_steps": 1,
        "stage_index_before": 4,
        "stage_index_after": 5,
        "outcome": [True, False, True, "success"],
        "observation_sha256": "c" * 64,
        "review_sha256": "d" * 64,
    }
    STRICT.validate_result_for_row(
        bounded_terminal_grid,
        manifest=manifest,
        row=row,
    )

    invalid_rows = []
    replay_audit = copy.deepcopy(valid)
    replay_audit["replay"]["mode"] = "audit"
    invalid_rows.append(replay_audit)
    for flag in (
        "fresh_backend_distinct",
        "backend_identity_exact",
        "source_identity_exact",
        "reset_identity_exact",
        "action_tape_exact",
        "observation_semantic_structure_exact",
        "review_semantic_structure_exact",
        "semantic_outcomes_exact",
        "terminal_ledger_semantic_exact",
        "terminal_ledger_exact_once",
    ):
        replay_mismatch = copy.deepcopy(valid)
        replay_mismatch["replay"][flag] = False
        invalid_rows.append(replay_mismatch)
    blocking_numeric_drift = copy.deepcopy(valid)
    blocking_numeric_drift["replay"]["observation_numeric_drift"]["blocking"] = True
    invalid_rows.append(blocking_numeric_drift)
    blocking_event_drift = copy.deepcopy(valid)
    blocking_event_drift["replay"]["observation_event_drift"]["blocking"] = True
    invalid_rows.append(blocking_event_drift)
    unbounded_terminal_grid = copy.deepcopy(bounded_terminal_grid)
    unbounded_terminal_grid["replay"]["terminal_grace"]["physics_steps"] = 26
    invalid_rows.append(unbounded_terminal_grid)
    forged_unused_terminal_grid = copy.deepcopy(valid)
    forged_unused_terminal_grid["replay"]["terminal_grace"]["control_steps"] = 1
    invalid_rows.append(forged_unused_terminal_grid)
    wrong_quality = copy.deepcopy(valid)
    wrong_quality["quality"]["schema_version"] = "db0-episode-task-quality-v1"
    invalid_rows.append(wrong_quality)
    wrong_backend = copy.deepcopy(valid)
    wrong_backend["quality"]["evaluator_backend_id"] = (
        "gpu-planner-t1-xyz-strict-evidence-v2"
    )
    wrong_backend["terminal_ledger"][0]["task_quality"]["evaluator_backend_id"] = (
        "gpu-planner-t1-xyz-strict-evidence-v2"
    )
    invalid_rows.append(wrong_backend)
    wrong_terminal_backend = copy.deepcopy(valid)
    wrong_terminal_backend["terminal_ledger"][0]["task_quality"][
        "evaluator_backend_id"
    ] = "gpu-planner-t1-xyz-strict-evidence-v2"
    invalid_rows.append(wrong_terminal_backend)
    reusable_terminal = copy.deepcopy(valid)
    reusable_terminal["terminal_ledger_gate"][
        "exact_once_second_consumption_rejected"
    ] = False
    invalid_rows.append(reusable_terminal)
    unverified_source = copy.deepcopy(valid)
    unverified_source["source_gate"]["repositories_exact"] = False
    invalid_rows.append(unverified_source)
    source_identity_drift = copy.deepcopy(valid)
    source_identity_drift["source_gate"]["repositories"]["se3_wam"]["commit"] = "f" * 40
    invalid_rows.append(source_identity_drift)
    provenance_drift = copy.deepcopy(valid)
    provenance_drift["provenance"]["git_tree"] = "f" * 40
    provenance_drift["replay"]["primary_provenance"]["git_tree"] = "f" * 40
    provenance_drift["replay"]["replay_provenance"]["git_tree"] = "f" * 40
    invalid_rows.append(provenance_drift)
    replay_provenance_drift = copy.deepcopy(valid)
    replay_provenance_drift["replay"]["replay_provenance"] = {
        **replay_provenance_drift["provenance"],
        "physical_device_uuid": "GPU-other",
    }
    invalid_rows.append(replay_provenance_drift)
    incomplete_export = copy.deepcopy(valid)
    incomplete_export["evidence_export"]["passed"] = False
    invalid_rows.append(incomplete_export)

    for invalid in invalid_rows:
        with pytest.raises((RuntimeError, ValueError)):
            STRICT.validate_result_for_row(invalid, manifest=manifest, row=row)

    numeric_drift_is_diagnostic = copy.deepcopy(valid)
    numeric_drift_is_diagnostic["replay"]["observation_event_sequence_exact"] = False
    numeric_drift_is_diagnostic["replay"]["observation_tape_exact"] = False
    numeric_drift_is_diagnostic["replay"]["review_tape_exact"] = False
    numeric_drift_is_diagnostic["replay"]["terminal_ledger_exact"] = False
    STRICT.validate_result_for_row(
        numeric_drift_is_diagnostic,
        manifest=manifest,
        row=row,
    )


def test_row_validator_binds_ledger_and_evidence_files(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, "e0")
    row = manifest.row(0)
    valid = _strict_result(manifest, 0)
    STRICT.validate_result_for_row(valid, manifest=manifest, row=row)

    ledger_mismatch = copy.deepcopy(valid)
    ledger_mismatch["terminal_ledger"][0]["termination_reason"] = "drop"
    with pytest.raises(RuntimeError, match="terminal ledger"):
        STRICT.validate_result_for_row(ledger_mismatch, manifest=manifest, row=row)

    tape_file = Path(valid["evidence_export"]["tape_file"])
    tape_file.write_bytes(b"tampered-tape")
    with pytest.raises(RuntimeError, match="digest drifted"):
        STRICT.validate_result_for_row(valid, manifest=manifest, row=row)

    missing = _strict_result(manifest, 0)
    Path(missing["evidence_export"]["visual_file"]).unlink()
    with pytest.raises(RuntimeError, match="file is missing"):
        STRICT.validate_result_for_row(missing, manifest=manifest, row=row)


def test_d32_requires_all_32_strict_rows_before_counting(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, "d32")
    results = [_strict_result(manifest, index) for index in range(32)]
    summary = STRICT.summarize_d32_results(results, manifest=manifest)
    assert summary["complete_cohort"] is True
    assert summary["valid_completed"] == 32
    assert summary["valid_successes"] == 32
    assert summary["success_rate"] == 1.0

    broken = copy.deepcopy(results)
    broken[0]["status"] = "completed_replay_audit_mismatch"
    broken[0]["replay"]["passed"] = False
    summary = STRICT.summarize_d32_results(broken, manifest=manifest)
    assert summary["complete_cohort"] is False
    assert summary["valid_completed"] == 31
    assert summary["valid_successes"] == 0
    assert summary["success_rate"] is None


def test_d32_drop_rate_uses_only_frozen_drop_reason(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, "d32")
    results = [_strict_result(manifest, index) for index in range(32)]
    results[0] = _strict_result(
        manifest,
        0,
        success=False,
        termination_reason="drop",
    )
    results[1] = _strict_result(
        manifest,
        1,
        success=False,
        termination_reason="timeout",
    )
    results[2] = _strict_result(
        manifest,
        2,
        success=False,
        termination_reason="workspace_exit",
    )

    summary = STRICT.summarize_d32_results(results, manifest=manifest)
    assert summary["complete_cohort"] is True
    assert summary["valid_successes"] == 29
    assert summary["valid_failures"] == 3
    assert summary["drop_count"] == 1
    assert summary["drop_rate"] == pytest.approx(1 / 32)
    assert summary["failure_reason_counts"]["drop"] == 1
    assert summary["failure_reason_counts"]["timeout"] == 1
    assert summary["failure_reason_counts"]["workspace_exit"] == 1
