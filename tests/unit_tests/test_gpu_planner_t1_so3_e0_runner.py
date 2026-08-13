# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import enum
import inspect
import json
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from examples.embodiment import run_gpu_planner_t1_so3_e0 as _MODULE
from rlinf.envs.dynamic_benchmark import t1_so3_planner as _PLANNER_MODULE


@dataclass(frozen=True)
class _Request:
    episode_id: str = "t1-so3-export-0000"
    task_id: str = "t1_so3"
    split: Any = "test_id"
    seed: int = 20261040
    action_mode: Any = "E7"
    observation_track: Any = "state"
    object_mode: str = "cuboid"
    reset_mode: str = "grasp"
    factors: Any = None
    api_version: str = "db-api-v0.1"

    def __post_init__(self) -> None:
        if self.factors is None:
            object.__setattr__(
                self,
                "factors",
                {"yaw_rate_rad_s": 0.25, "object_scale": 1.0},
            )


def test_frozen_manifest_binds_exact_export_request() -> None:
    request = _Request()
    payload = _MODULE._frozen_manifest_payload(request)

    assert payload["schema_version"] == _MODULE.FROZEN_MANIFEST_SCHEMA_VERSION
    assert payload["task_id"] == "t1_so3"
    assert payload["batch_size"] == 1
    assert payload["request"]["split"] == "test_id"
    assert payload["request"]["observation_track"] == "state"
    assert _MODULE._validate_frozen_manifest_payload(payload, request) == payload

    tampered = {**payload, "request": {**payload["request"], "seed": 7}}
    with pytest.raises(ValueError, match="exact export request"):
        _MODULE._validate_frozen_manifest_payload(tampered, request)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("task_id", "t1_xyz"),
        ("split", "train"),
        ("action_mode", "E6"),
        ("observation_track", "hybrid"),
    ),
)
def test_frozen_manifest_rejects_non_t1_so3_state_e7_test_id(
    field: str,
    value: Any,
) -> None:
    request = _Request(**{field: value})

    with pytest.raises(ValueError, match="test_id STATE/E7"):
        _MODULE._frozen_manifest_payload(request)


def test_runner_has_no_row0_or_hidden_planner_tuning_and_replay_is_blocking() -> None:
    parser = _MODULE._parser()
    options = {
        option
        for action in parser._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert "--frozen-manifest" in options
    assert "--manifest-seed" not in options
    assert "--manifest-size" not in options
    assert _MODULE.TASK_QUALITY_SCHEMA_VERSION == "db0-episode-task-quality-v2"
    assert _MODULE.EVALUATION_PHASE == "development"
    assert _MODULE.EXPECTED_HORIZON_STEPS == 160

    source = inspect.getsource(_MODULE._run)
    for forbidden in (
        "make_dataset_candidate_manifest",
        "lift_action_z_max",
        "lift_height_m",
        "contact_anchor_on_first_touch",
        "close_retry_steps",
        "run_horizon",
    ):
        assert forbidden not in source
    assert "run_natural_termination" in source
    assert "except BaseException" not in source
    replay = source.index("replay = replay_action_trajectory")
    pending_write = source.index("_write_json(pending_path, bundle)")
    schema_gate = source.index("_validate_review_bundle(schema, validation_payload)")
    assert replay < schema_gate < pending_write
    finalize_source = inspect.getsource(_MODULE._finalize_bundle)
    assert finalize_source.index("_validate_review_bundle(schema, payload)") < (
        finalize_source.index("pending_path.replace(bundle_path)")
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
    privileged: MappingProxyType[str, np.ndarray]
    events_since_last_observation: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _DeviceReceipt:
    expected_uuid: str
    driver_version: str = "590.48.02"


class _Planner:
    phase = "track"
    orientation_mode = "track_object"
    lookahead_s = 0.1
    grasp_axis_offset_rad = 0.0

    def act(self, observation: _Observation) -> _ActionCommand:
        return _ActionCommand(
            mode=_ActionMode.E7,
            values=np.full(7, 0.1, dtype=np.float32),
            policy_step=observation.policy_step,
        )

    def planner_audit_snapshot(self) -> dict[str, Any]:
        return {"source": "current_observation"}


class _FakeBackend:
    instances: list[_FakeBackend] = []
    backend_id = "mjwarp_gpu_v1"
    task_id = "t1_so3"
    num_envs = 1
    device = SimpleNamespace(type="cuda")
    observation_track = "state"
    render_observations = True
    cohort_horizon_steps = 160

    def __init__(self, **kwargs: Any) -> None:
        assert kwargs["task_quality_schema_version"] == ("db0-episode-task-quality-v2")
        assert kwargs["manifest_size"] == 1
        assert len(kwargs["manifest_requests"]) == 1
        assert "runtime_manifest" not in kwargs
        self.request = replace(
            kwargs["manifest_requests"][0],
            episode_id=(f"{kwargs['manifest_requests'][0].episode_id}-cycle00000000"),
        )
        self.manifest_sha256 = kwargs["manifest_sha256"]
        self.expected_uuid = kwargs["expected_gpu_uuid"]
        self.step = 0
        self.terminal_calls = 0
        self.teacher_audit_materializations = 0
        self.transport_checks = 0
        self.stable_identity = MappingProxyType(
            {
                "backend_id": self.backend_id,
                "task_id": self.task_id,
                "manifest_sha256": self.manifest_sha256,
            }
        )
        self.instances.append(self)

    def next_requests(self) -> tuple[_Request, ...]:
        return (self.request,)

    def reset(self) -> Any:
        self.step = 0
        self.terminal_calls = 0
        return SimpleNamespace(
            episode_ids=(self.request.episode_id,),
            manifest_sha256=self.manifest_sha256,
            manifest_ordinals=(0,),
            seeds=(self.request.seed,),
            generation=1,
        )

    def _observation(self) -> _Observation:
        def digest(value: Any) -> str:
            return _MODULE._sha256(value)

        return _Observation(
            episode_id=self.request.episode_id,
            task_id=self.task_id,
            physics_step=25 * self.step,
            control_step=self.step,
            policy_step=self.step,
            component_sha256=MappingProxyType(
                {
                    "metadata": digest(f"metadata-{self.step}"),
                    "privileged/state": digest(f"state-{self.step}"),
                    "rgb/agentview": digest(f"rgb-{self.step}"),
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

    def materialize_teacher_observations(
        self,
        lanes: tuple[int, ...],
    ) -> tuple[_Observation, ...]:
        assert lanes == (0,)
        self.teacher_audit_materializations += 1
        return (self._observation(),)

    def materialize_health_audit(self) -> dict[str, np.ndarray]:
        return {
            "overflow": np.asarray([0], dtype=np.int64),
            "controller_valid": np.asarray([1], dtype=np.int64),
            "driver_valid": np.asarray([1], dtype=np.int64),
            "physics_step": np.asarray([25 * self.step], dtype=np.int64),
            "terminal": np.asarray([int(self.step == 2)], dtype=np.int64),
        }

    def materialize_review_rgb(
        self,
        lanes: tuple[int, ...],
    ) -> tuple[dict[str, np.ndarray], ...]:
        assert lanes == (0,)
        return (
            {
                "agentview": np.full(
                    (4, 4, 3),
                    self.step + 1,
                    dtype=np.uint8,
                ),
                "robot0_eye_in_hand": np.full(
                    (4, 4, 3),
                    self.step + 2,
                    dtype=np.uint8,
                ),
            },
        )

    def step_device(self, action: Any) -> Any:
        assert np.asarray(action).shape == (1, 7)
        self.transport_checks += 1
        self.step += 1
        return SimpleNamespace(done=np.asarray([self.step == 2], dtype=np.bool_))

    def materialize_terminal_ledger_once(
        self,
        lanes: tuple[int, ...],
        episode_ids: tuple[str, ...],
    ) -> tuple[Any, ...]:
        assert lanes == (0,) and episode_ids == (self.request.episode_id,)
        self.terminal_calls += 1
        assert self.terminal_calls == 1
        return (
            SimpleNamespace(
                lane=0,
                episode_id=self.request.episode_id,
                task_id=self.task_id,
                terminated=False,
                truncated=True,
                success=False,
                termination_reason="timeout",
                completion=0.0,
                task_quality=None,
                events=(),
                physics_step=25 * self.step,
                control_step=self.step,
                policy_step=self.step,
            ),
        )

    def attest_end(self) -> _DeviceReceipt:
        return _DeviceReceipt(expected_uuid=self.expected_uuid)

    def close(self) -> None:
        return None


def _minimal_review_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": _MODULE.SCHEMA_ID,
        "type": "object",
        "required": [
            "schema_version",
            "bundle_status",
            "claim_scope",
            "development_limitations",
            "source_bundle_identity",
            "evaluation",
            "episodes",
        ],
        "properties": {
            "schema_version": {"const": _MODULE.SCHEMA_VERSION},
            "bundle_status": {"const": "complete"},
            "claim_scope": {"pattern": "B=1 engineering E0"},
            "development_limitations": {"type": "array", "minItems": 1},
            "source_bundle_identity": {
                "type": "object",
                "required": [
                    "bundle_id",
                    "bundle_json",
                    "checksums",
                    "import_instructions",
                    "payload_member_count",
                ],
            },
            "evaluation": {
                "type": "object",
                "properties": {"phase": {"const": "development"}},
                "required": ["phase"],
            },
            "episodes": {
                "type": "array",
                "items": {"required": ["terminal"]},
            },
        },
    }


def test_cpu_fake_run_emits_schema_valid_strict_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "sources"
    research_root = source_root / "se3-wam-research"
    se3_root = source_root / "SE3-WAM"
    dynamic_root = source_root / "se3-wam-dynamic-benchmark"
    rlinf_root = dynamic_root / "third_party" / "RLinf"
    schema_path = (
        research_root
        / "docs"
        / "experiments"
        / "GPUPLAN0"
        / "design"
        / "review_bundle_schema.json"
    )
    schema_path.parent.mkdir(parents=True)
    workspace_schema = (
        Path(__file__).resolve().parents[3]
        / "se3-wam-research"
        / "docs"
        / "experiments"
        / "GPUPLAN0"
        / "design"
        / "review_bundle_schema.json"
    )
    schema = (
        json.loads(workspace_schema.read_text(encoding="utf-8"))
        if workspace_schema.is_file()
        else _minimal_review_schema()
    )
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    files = (
        rlinf_root / "rlinf" / "envs" / "dynamic_benchmark" / "t1_so3_planner.py",
        rlinf_root / "rlinf" / "envs" / "dynamic_benchmark" / "gpu_tensor_backend.py",
        rlinf_root / "examples" / "embodiment" / "run_gpu_planner_t1_so3_e0.py",
        se3_root / "src" / "se3_wam" / "benchmark" / "configs" / "t1_so3_v0_16.yaml",
    )
    for index, path in enumerate(files):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture-{index}\n", encoding="utf-8")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    request = _Request()
    manifest_path = tmp_path / "frozen_manifest.json"
    manifest_path.write_text(
        json.dumps(_MODULE._frozen_manifest_payload(request)),
        encoding="utf-8",
    )

    source = {
        "research": {"commit": "1" * 40, "tree": "2" * 40},
        "se3_wam": {
            "commit": "3" * 40,
            "tree": "4" * 40,
            "mjwarp_gitlink": "5" * 40,
            "mjwarp_tree": "6" * 40,
        },
        "rlinf": {"commit": "7" * 40, "tree": "8" * 40},
        "dynamic_benchmark": {
            "commit": "9" * 40,
            "tree": "a" * 40,
            "rlinf_gitlink": "7" * 40,
        },
    }
    source_paths = {
        "research": research_root,
        "se3_wam": se3_root,
        "dynamic_benchmark": dynamic_root,
        "rlinf": rlinf_root,
        "mjwarp": se3_root / "third_party" / "mujoco_warp",
    }

    def verified_source(_args: argparse.Namespace) -> tuple[Any, Any]:
        return json.loads(json.dumps(source)), source_paths

    monkeypatch.setattr(_MODULE, "_verify_source", verified_source)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)

    torch = types.ModuleType("torch")
    torch.float32 = np.float32
    torch.as_tensor = lambda value, dtype, device: np.asarray(value, dtype=dtype)
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    torch.version = SimpleNamespace(cuda="13.0")
    monkeypatch.setitem(sys.modules, "torch", torch)

    se3 = types.ModuleType("se3_wam")
    benchmark = types.ModuleType("se3_wam.benchmark")
    api = types.ModuleType("se3_wam.benchmark.api")
    api.ActionCommand = _ActionCommand
    contracts = types.ModuleType("se3_wam.benchmark.contracts")
    contracts.ActionMode = _ActionMode
    gpu_native = types.ModuleType("se3_wam.benchmark.gpu_native")
    engine = types.ModuleType("se3_wam.benchmark.gpu_native.p0_grasp_engine")
    engine.load_p0_grasp_artifacts = lambda _: SimpleNamespace(reset_request=request)
    teacher = types.ModuleType("se3_wam.benchmark.teacher_factory")
    teacher.make_privileged_teacher = lambda *args, **kwargs: (
        _Planner(),
        {"implementation": "fixture-defaults"},
    )
    for name, module in (
        ("se3_wam", se3),
        ("se3_wam.benchmark", benchmark),
        ("se3_wam.benchmark.api", api),
        ("se3_wam.benchmark.contracts", contracts),
        ("se3_wam.benchmark.gpu_native", gpu_native),
        ("se3_wam.benchmark.gpu_native.p0_grasp_engine", engine),
        ("se3_wam.benchmark.teacher_factory", teacher),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    backend_module = types.ModuleType("rlinf.envs.dynamic_benchmark.gpu_tensor_backend")
    backend_module.GpuNativeTensorBackendEnv = _FakeBackend
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.dynamic_benchmark.gpu_tensor_backend",
        backend_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.dynamic_benchmark.t1_so3_planner",
        _PLANNER_MODULE,
    )
    _FakeBackend.instances = []
    gpu_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    run_root = tmp_path / "run"
    args = argparse.Namespace(
        source_root=source_root,
        run_root=run_root,
        bundle_id="t1-so3-fixture",
        job_id="GPUPLAN0/t1-so3-e0-fixture",
        resource_unit="fixture/lane0",
        container_name="fixture-container",
        export_dir=export_dir,
        frozen_manifest=manifest_path,
        expected_gpu_uuid=gpu_uuid,
        se3_commit="3" * 40,
        se3_tree="4" * 40,
        rlinf_commit="7" * 40,
        rlinf_tree="8" * 40,
        dynamic_commit="9" * 40,
        dynamic_tree="a" * 40,
        dynamic_rlinf_gitlink="7" * 40,
        mjwarp_gitlink="5" * 40,
        mjwarp_tree="6" * 40,
        research_commit="1" * 40,
        research_tree="2" * 40,
        review_schema=schema_path,
        image_size=4,
        prior_failure=[],
        finalize_bundle=None,
    )

    assert _MODULE._run(args) == 0

    bundle_dir = run_root / "review-bundle" / "t1_so3" / "t1-so3-fixture"
    pending = json.loads(
        (bundle_dir / "bundle.pending.json").read_text(encoding="utf-8")
    )
    assert pending["runtime"]["runtime_ledger_lease_released"] is False
    assert not (bundle_dir / "bundle.json").exists()
    assert _MODULE._finalize_bundle(bundle_dir) == 0
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    _MODULE._validate_review_bundle(schema, bundle)
    _MODULE._verify_checksums(bundle_dir)
    assert bundle["evaluation"]["phase"] == "development"
    assert bundle["development_limitations"]
    assert bundle["source_bundle_identity"]["bundle_json"]["path"] == (
        "source_bundle.json"
    )
    assert bundle["episodes"][0]["replay"]["passed"] is True
    assert bundle["runtime"]["runtime_ledger_lease_released"] is True
    assert bundle["owner"]["status"] == "pending_owner_review"
    assert len(_FakeBackend.instances) == 2
    assert all(backend.terminal_calls == 1 for backend in _FakeBackend.instances)
