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

import enum
import importlib.util
import sys
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_MODULE_PATH = (
    Path(__file__).parents[2]
    / "rlinf"
    / "envs"
    / "dynamic_benchmark"
    / "gpu_tensor_backend.py"
)
_SPEC = importlib.util.spec_from_file_location("_gpu_tensor_backend_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
GpuNativeTensorBackendEnv = _MODULE.GpuNativeTensorBackendEnv
GpuNativeTensorBackendUnavailableError = _MODULE.GpuNativeTensorBackendUnavailableError
_require_single_uuid_visibility = _MODULE._require_single_uuid_visibility
_zero_copy_torch_view = _MODULE._zero_copy_torch_view

_GPU_UUID = "GPU-803b6f88-a884-134a-d92d-cdc532e22e14"


@dataclass(frozen=True)
class _Device:
    name: str

    def __str__(self) -> str:
        return self.name


class _Tensor:
    def __init__(self, shape: tuple[int, ...], device: _Device, pointer: int, dtype: Any) -> None:
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self._pointer = pointer

    def data_ptr(self) -> int:
        return self._pointer

    def is_contiguous(self) -> bool:
        return True


class _WarpArray:
    def __init__(self, tensor: _Tensor) -> None:
        self.shape = tensor.shape
        self.ptr = tensor.data_ptr()
        self.tensor = tensor


class _Capabilities:
    def __init__(self) -> None:
        self.required: tuple[str, ...] | None = None

    def require(self, *names: str) -> None:
        self.required = names


class _FakeEnv:
    backend_id = "mjwarp_gpu_v1"

    def __init__(self, batch_size: int, device: _Device, dtype: Any) -> None:
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.capabilities = _Capabilities()
        self.contract = SimpleNamespace(api_version="gpu-native-api-v0.2")
        self.provenance = SimpleNamespace(physical_device_uuid=_GPU_UUID)
        self.reset_calls: list[tuple[Any, ...]] = []
        self.step_calls: list[Any] = []
        self.closed = False
        self._pointer = 1000
        self._step_arrays: dict[str, _WarpArray] | None = None

    def _array(self, width: int | None = None) -> _WarpArray:
        self._pointer += 100
        shape = (self.batch_size,) if width is None else (self.batch_size, width)
        return _WarpArray(_Tensor(shape, self.device, self._pointer, self.dtype))

    def reset(self, requests: Any) -> Any:
        self.reset_calls.append(tuple(requests))
        return SimpleNamespace(
            observation=self._array(5),
            state=SimpleNamespace(generation=1),
        )

    def step_device(self, action: Any) -> Any:
        self.step_calls.append(action)
        if self._step_arrays is None:
            self._step_arrays = {
                "observation": self._array(5),
                "reward": self._array(),
                "terminated": self._array(),
                "truncated": self._array(),
                "success": self._array(),
                "event_mask": self._array(),
                "terminal_reason": self._array(),
                "physics_step": self._array(),
            }
        return SimpleNamespace(
            **self._step_arrays,
            state=SimpleNamespace(generation=2),
        )

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _Request:
    task_id: str = "p0_grasp"
    episode_id: str = "source"
    observation_track: Any = None


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _FakeEnv, _Device]:
    device = _Device("cuda:0")
    float32 = object()
    torch = types.ModuleType("torch")
    torch.float32 = float32
    torch.device = lambda value: _Device(value)
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        current_stream=lambda selected: ("torch-stream", selected),
    )
    warp = types.ModuleType("warp")
    warp.to_torch_calls = 0

    def to_torch(value: Any) -> Any:
        warp.to_torch_calls += 1
        return value.tensor

    warp.to_torch = to_torch
    warp.stream_from_torch = lambda stream: ("warp-stream", stream)
    warp.ScopedStream = lambda stream: nullcontext(stream)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "warp", warp)

    package_names = (
        "se3_wam",
        "se3_wam.benchmark",
        "se3_wam.benchmark.config",
        "se3_wam.benchmark.contracts",
        "se3_wam.benchmark.gpu_native",
        "se3_wam.benchmark.gpu_native.factory",
        "se3_wam.benchmark.gpu_native.p0_grasp_engine",
        "se3_wam.benchmark.gpu_native.tasks",
    )
    modules: dict[str, types.ModuleType] = {}
    for name in package_names:
        module = types.ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    class ObservationTrack(enum.Enum):
        STATE = "state"

    class GpuNativeConsumer(enum.Enum):
        RL = "rl"

    env = _FakeEnv(3, device, float32)
    captured: dict[str, Any] = {}

    def make_gpu_native_env(task_id: str, **kwargs: Any) -> _FakeEnv:
        captured["factory"] = (task_id, kwargs)
        return env

    modules["se3_wam.benchmark.contracts"].ObservationTrack = ObservationTrack
    modules["se3_wam.benchmark.config"].load_task_config = lambda task_id: {
        "clock": {"horizon_steps": 250}
    }
    modules["se3_wam.benchmark.gpu_native.tasks"].GpuNativeConsumer = GpuNativeConsumer
    modules["se3_wam.benchmark.gpu_native.factory"].make_gpu_native_env = make_gpu_native_env
    modules["se3_wam.benchmark.gpu_native.p0_grasp_engine"].load_p0_grasp_artifacts = (
        lambda export_dir: SimpleNamespace(reset_request=_Request())
    )
    return captured, env, device


def test_visibility_requires_one_physical_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _require_single_uuid_visibility(_GPU_UUID, 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="exactly"):
        _require_single_uuid_visibility(_GPU_UUID, 0)


def test_zero_copy_view_rejects_pointer_change() -> None:
    device = _Device("cuda:0")
    tensor = _Tensor((4, 8), device, 1234, object())
    array = _WarpArray(tensor)
    warp = SimpleNamespace(to_torch=lambda value: value.tensor)
    assert (
        _zero_copy_torch_view(
            warp_module=warp,
            value=array,
            name="observation",
            batch_size=4,
            expected_device=device,
        )
        is tensor
    )
    copied = _Tensor((4, 8), device, 5678, tensor.dtype)
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="copied storage"):
        _zero_copy_torch_view(
            warp_module=SimpleNamespace(to_torch=lambda value: copied),
            value=array,
            name="observation",
            batch_size=4,
            expected_device=device,
        )


def test_tensor_backend_preserves_actions_and_outputs_as_device_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    captured, fake_env, device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
    )
    assert fake_env.capabilities.required == ("full_batch_reset", "device_tensor_step")
    assert backend.cohort_horizon_steps == 250
    assert backend.api_version == "gpu-native-api-v0.2"
    assert captured["factory"][1]["engine_kwargs"] == {
        "expected_device_uuid": _GPU_UUID
    }

    reset = backend.reset()
    assert reset.observation.device == device
    assert len(fake_env.reset_calls[0]) == 3
    action = _Tensor((3, 7), device, 9999, sys.modules["torch"].float32)
    step = backend.step(action)
    assert fake_env.step_calls == [action]
    assert step.observation.device == device
    assert step.physics_step.device == device
    assert step.generation == 2
    first_conversion_count = sys.modules["warp"].to_torch_calls
    backend.step(action)
    assert sys.modules["warp"].to_torch_calls == first_conversion_count
    backend.close()
    assert fake_env.closed
