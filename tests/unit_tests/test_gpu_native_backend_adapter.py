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

"""Contract tests for the GPU-native Dynamic Benchmark backend adapter.

These tests run without SE3-WAM or a CUDA device: they pin the adapter's
request mapping, validation, and the fail-closed environment gating.  The
real CUDA training loop is covered by the PPO GPU smoke on the machine.
"""

from __future__ import annotations

import enum
import pickle
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv
from rlinf.envs.dynamic_benchmark.gpu_backend import (
    GpuNativeBackendEnv,
    GpuNativeBackendUnavailableError,
)


class _ObservationTrack(enum.Enum):
    STATE = "state"
    VISUAL = "visual"
    HYBRID = "hybrid"


class _GpuNativeConsumer(enum.Enum):
    RL = "rl"
    GENERATION = "generation"
    ROLLOUT = "rollout"
    EVAL = "eval"


class _AuditRequest:
    def __init__(self, lanes: tuple[int, ...], include_step_result: bool) -> None:
        self.lanes = lanes
        self.include_step_result = include_step_result


@dataclass(eq=False)
class _FakeRequest:
    task_id: str = "p0_grasp"
    episode_id: str = "ep-0"
    seed: int = 7
    split: Any = None
    action_mode: Any = None
    observation_track: Any = None
    object_mode: str = "cube"
    reset_mode: str = "default"
    factors: Any = None
    api_version: str = "benchmark-api-v0.1"

    def __post_init__(self) -> None:
        if self.split is None:
            self.split = _FakeSplit("test_id")
        if self.action_mode is None:
            self.action_mode = _FakeEnumValue("e7")
        if self.observation_track is None:
            self.observation_track = _FakeEnumValue("state")
        if self.factors is None:
            self.factors = {"speed_class": "normal", "object_position_x_m": 0.0}


class _FakeSplit:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeEnumValue:
    def __init__(self, value: str) -> None:
        self.value = value


@dataclass(frozen=True)
class _FakeObservation:
    lane: int
    step: int

    @property
    def episode_id(self) -> str:
        return f"ep-{self.lane}"

    @property
    def fingerprint_sha256(self) -> str:
        return f"obs-{self.lane}-{self.step}"


class _FakePlannerTeacher:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, observation: Any) -> Any:
        self.act_calls += 1
        return SimpleNamespace(
            values=np.zeros(7, dtype=np.float64),
            policy_step=int(observation.step),
        )


class _FakeEnv:
    backend_id = "mjwarp_gpu_v1"

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.bookkeeping = SimpleNamespace(
            clock=SimpleNamespace(policy_steps=np.arange(batch_size, dtype=np.int64))
        )
        self.provenance = SimpleNamespace(
            backend_id="mjwarp_gpu_v1",
            device_platform="cuda",
            device_name="fake",
            device_ordinal=0,
            runtime_versions={"mujoco": "3.11.0"},
        )
        self.reset_calls = 0
        self.step_calls = 0
        self.planner_teachers: list[_FakePlannerTeacher] = []

    def reset(self, requests: Any) -> None:
        self.reset_calls += 1
        self.step_calls = 0
        if len(tuple(requests)) != self.batch_size:
            raise AssertionError("reset requests must cover the whole batch")

    def materialize_current_observations(self, lanes: tuple[int, ...]) -> tuple[Any, ...]:
        return tuple(_FakeObservation(lane=lane, step=self.step_calls) for lane in lanes)

    def step(self, commands: Any) -> None:
        self.step_calls += 1
        if len(tuple(commands)) != self.batch_size:
            raise AssertionError("step commands must cover the whole batch")

    def materialize_audit(self, request: Any) -> Any:
        if not request.include_step_result:
            return SimpleNamespace(
                lanes=tuple(
                    SimpleNamespace(
                        lane=lane,
                        observation=_FakeObservation(lane=lane, step=self.step_calls),
                        step_result=None,
                    )
                    for lane in request.lanes
                )
            )
        lanes = tuple(
            SimpleNamespace(
                lane=lane,
                observation=_FakeObservation(lane=lane, step=self.step_calls),
                step_result=SimpleNamespace(
                    observation=_FakeObservation(lane=lane, step=self.step_calls),
                    terminated=False,
                    truncated=False,
                    success=False,
                    termination_reason=None,
                ),
            )
            for lane in request.lanes
        )
        return SimpleNamespace(lanes=lanes)

    def close(self) -> None:
        pass


@dataclass
class _FakeSe3Wam:
    factory: Any
    artifacts: Any
    planner_requests: list[Any]


@pytest.fixture
def fake_se3_wam(monkeypatch: pytest.MonkeyPatch) -> _FakeSe3Wam:
    """Install a minimal fake ``se3_wam`` package tree in ``sys.modules``."""

    package_names = (
        "se3_wam",
        "se3_wam.benchmark",
        "se3_wam.benchmark.contracts",
        "se3_wam.benchmark.gpu_native",
        "se3_wam.benchmark.gpu_native.factory",
        "se3_wam.benchmark.gpu_native.p0_grasp_engine",
        "se3_wam.benchmark.gpu_native.tasks",
        "se3_wam.benchmark.gpu_native.audit",
        "se3_wam.benchmark.teacher_factory",
    )
    modules: dict[str, types.ModuleType] = {}
    for name in package_names:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    modules["se3_wam.benchmark.contracts"].ObservationTrack = _ObservationTrack
    modules["se3_wam.benchmark.gpu_native.tasks"].GpuNativeConsumer = _GpuNativeConsumer
    modules["se3_wam.benchmark.gpu_native.audit"].AuditRequest = _AuditRequest
    captured: dict[str, Any] = {}
    factory = _FakeEnv(batch_size=3)
    modules["se3_wam.benchmark.gpu_native.factory"].make_gpu_native_env = (
        lambda task_id, **kwargs: (
            captured.__setitem__("factory_kwargs", (task_id, kwargs)) or factory
        )
    )
    modules["se3_wam.benchmark.gpu_native.p0_grasp_engine"].load_p0_grasp_artifacts = (
        lambda export_dir: SimpleNamespace(reset_request=_FakeRequest())
    )
    planner_requests: list[Any] = []

    def make_privileged_teacher(task_id: str, *, request: Any, image_size: int) -> Any:
        assert task_id == request.task_id
        assert image_size == 64
        planner_requests.append(request)
        teacher = _FakePlannerTeacher()
        factory.planner_teachers.append(teacher)
        return teacher, {"request_episode_id": request.episode_id}

    modules["se3_wam.benchmark.teacher_factory"].make_privileged_teacher = (
        make_privileged_teacher
    )
    return _FakeSe3Wam(
        factory=factory,
        artifacts=captured,
        planner_requests=planner_requests,
    )


def test_backend_module_importable_without_se3_wam() -> None:
    import rlinf.envs.dynamic_benchmark.gpu_backend as module  # noqa: PLC0415

    assert module.GpuNativeBackendUnavailableError is GpuNativeBackendUnavailableError


def test_adapter_validates_constructor_arguments() -> None:
    with pytest.raises(ValueError, match="num_envs"):
        GpuNativeBackendEnv(task_id="p0_grasp", num_envs=0, export_dir="/tmp/x")
    with pytest.raises(ValueError, match="export_dir"):
        GpuNativeBackendEnv(task_id="p0_grasp", num_envs=2, export_dir="  ")
    with pytest.raises(ValueError, match="device_ordinal"):
        GpuNativeBackendEnv(
            task_id="p0_grasp", num_envs=2, export_dir="/tmp/x", device_ordinal=-1
        )


def test_adapter_raises_without_se3_wam(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    original = builtins.__import__

    def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "se3_wam" or name.startswith("se3_wam."):
            raise ImportError("no se3_wam")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(GpuNativeBackendUnavailableError, match="SE3-WAM"):
        GpuNativeBackendEnv(task_id="p0_grasp", num_envs=2, export_dir="/tmp/x")


def test_adapter_request_mapping(fake_se3_wam: _FakeSe3Wam) -> None:
    backend = GpuNativeBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
    )
    task_id, kwargs = fake_se3_wam.artifacts["factory_kwargs"]
    assert task_id == "p0_grasp"
    assert kwargs["consumer"].value == "rl"
    assert kwargs["observation_track"].value == "state"
    assert kwargs["batch_size"] == 3
    assert kwargs["export_dir"] == "/tmp/export"
    assert backend.consumer.value == "rl"
    first = backend.next_request()
    second = backend.next_request()
    assert first.episode_id != second.episode_id
    assert first.task_id == "p0_grasp" and first.seed == 7
    assert first.observation_track.value == "state"
    assert first.factors == _FakeRequest().factors
    assert backend.policy_steps().tolist() == [0, 1, 2]
    results = backend.step([None] * 3)
    assert len(results) == 3
    assert fake_se3_wam.factory.step_calls == 1
    assert backend.backend_id == "mjwarp_gpu_v1"


def test_planner_is_current_state_closed_loop_and_tape_replay_is_diagnostic(
    fake_se3_wam: _FakeSe3Wam,
) -> None:
    backend = GpuNativeBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
    )
    requests = tuple(backend.next_request() for _ in range(3))
    observations = backend.reset_planner(requests)
    assert len(observations) == 3
    assert fake_se3_wam.planner_requests == list(requests)
    assert all(teacher.reset_calls == 1 for teacher in fake_se3_wam.factory.planner_teachers)

    step = backend.planner_step()
    assert step.step_index == 0
    assert all(teacher.act_calls == 1 for teacher in fake_se3_wam.factory.planner_teachers)
    tape = backend.planner_tape()
    assert tape.steps == (step,)

    report = backend.replay_planner_tape_diagnostic(tape)
    assert report["mode"] == "diagnostic_frozen_action_replay"
    assert report["closed_loop_planner"] is False
    assert report["passed"] is True


def test_adapter_reset_requires_full_batch(fake_se3_wam: _FakeSe3Wam) -> None:
    backend = GpuNativeBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
    )
    with pytest.raises(ValueError, match="one request per lane"):
        backend.reset([backend.next_request()])


def test_adapter_not_picklable(fake_se3_wam: _FakeSe3Wam) -> None:
    backend = GpuNativeBackendEnv(
        task_id="p0_grasp",
        num_envs=2,
        export_dir="/tmp/export",
    )
    with pytest.raises(TypeError, match="not picklable"):
        pickle.dumps(backend)


def test_env_gating_requires_export_dir() -> None:
    with pytest.raises(ValueError, match="gpu_native_export_dir"):
        DynamicBenchmarkEnv(
            cfg={
                "task_id": "p0_grasp",
                "split": "train",
                "manifest_seed": 1,
                "manifest_size": 8,
                "image_size": 64,
                "camera_observations": False,
                "gpu_native": True,
            },
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )


def test_env_gating_forbids_process_workers() -> None:
    with pytest.raises(ValueError, match="forbids process workers"):
        DynamicBenchmarkEnv(
            cfg={
                "task_id": "p0_grasp",
                "split": "train",
                "manifest_seed": 1,
                "manifest_size": 8,
                "image_size": 64,
                "camera_observations": False,
                "worker_processes": 2,
                "gpu_native": True,
                "gpu_native_export_dir": "/tmp/export",
            },
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )


def test_gpu_native_auto_reset_rejects_partial_terminal_cohort() -> None:
    env = object.__new__(DynamicBenchmarkEnv)
    env._gpu_backend = object()
    env.num_envs = 2
    with pytest.raises(RuntimeError, match="full terminal cohort"):
        env._handle_auto_reset(np.asarray([True, False]), {}, {})


def test_env_gpu_identity_attributes(fake_se3_wam: _FakeSe3Wam) -> None:
    env = object.__new__(DynamicBenchmarkEnv)
    env.cfg = {
        "task_id": "p0_grasp",
        "gpu_native_export_dir": "/tmp/export",
        "gpu_native_device_ordinal": 1,
    }
    env._gpu_backend = GpuNativeBackendEnv(
        task_id="p0_grasp",
        num_envs=2,
        export_dir="/tmp/export",
        device_ordinal=1,
    )
    env.num_envs = 2
    env.split_name = "train"
    env.base_manifest_seed = 1
    env.manifest_size = 8
    env.image_size = 64
    env.camera_observations = False
    env.worker_threads = 1
    env.worker_processes = 0
    env.process_start_method = "spawn"
    env.task_id = "p0_grasp"
    env.gpu_native = True
    env._state_schema = SimpleNamespace(to_dict=lambda: {"state_dim": 4})
    env.reward_trackers = [SimpleNamespace(is_empty=False) for _ in range(2)]
    env.reward_registries = [
        SimpleNamespace(
            is_empty=False,
            to_dict=lambda: {},
            identity_sha256=lambda: "0" * 64,
        )
        for _ in range(2)
    ]
    env.feature_registry = SimpleNamespace(
        is_empty=True,
        to_dict=lambda: {},
        identity_sha256=lambda: "0" * 64,
    )
    env.task_quality_schema_version = None
    env.task_quality_evaluator_backend_id = None
    identity = env._checkpoint_identity()
    assert identity["gpu_native"] is True
    assert identity["gpu_native_export_dir"] == "/tmp/export"
    assert identity["gpu_native_device_ordinal"] == 1


def test_next_request_bypasses_manifest(fake_se3_wam: _FakeSe3Wam) -> None:
    env = object.__new__(DynamicBenchmarkEnv)
    env._gpu_backend = GpuNativeBackendEnv(
        task_id="p0_grasp", num_envs=2, export_dir="/tmp/export"
    )
    env._manifest_rows = ()
    request = env._next_request()
    assert request.seed == 7
    assert request.episode_id.startswith("p0_grasp-gpu-")
    assert request.observation_track.value == "state"
