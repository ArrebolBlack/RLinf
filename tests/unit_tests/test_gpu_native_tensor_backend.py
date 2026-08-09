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
from dataclasses import dataclass, field
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
_PCI_BUS_ID = "00000000:17:00.0"


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
        self.provenance = SimpleNamespace(
            physical_device_uuid=_GPU_UUID,
            physical_device_pci_bus_id=_PCI_BUS_ID,
            physical_device_identity_source="warp_cuda_driver",
        )
        self.reset_calls: list[tuple[Any, ...]] = []
        self.step_calls: list[Any] = []
        self.closed = False
        self.transport_action_offset = 0
        self.transport_stream_ptr = 555
        self._pointer = 1000
        self._step_arrays: dict[str, _WarpArray] | None = None
        self.reset_error = False

    def _array(self, dtype: Any, width: int | None = None) -> _WarpArray:
        self._pointer += 100
        shape = (self.batch_size,) if width is None else (self.batch_size, width)
        return _WarpArray(_Tensor(shape, self.device, self._pointer, dtype))

    def reset(self, requests: Any) -> Any:
        if self.reset_error:
            raise RuntimeError("injected reset failure")
        self.reset_calls.append(tuple(requests))
        return SimpleNamespace(
            observation=self._array(self.dtype, 5),
            state=SimpleNamespace(generation=1),
        )

    def step_device(self, action: Any) -> Any:
        self.step_calls.append(action)
        if self._step_arrays is None:
            self._step_arrays = {
                "observation": self._array(self.dtype, 5),
                "reward": self._array(self.dtype),
                "terminated": self._array("int32"),
                "truncated": self._array("int32"),
                "success": self._array("int32"),
                "event_mask": self._array("int32"),
                "terminal_reason": self._array("int32"),
                "physics_step": self._array("int64"),
            }
        return SimpleNamespace(
            **self._step_arrays,
            state=SimpleNamespace(generation=2),
            transport=SimpleNamespace(
                action_input_ptr=action.data_ptr(),
                action_engine_ptr=action.data_ptr() + self.transport_action_offset,
                stream_ptr=self.transport_stream_ptr,
                output_ptrs={
                    name: value.ptr for name, value in self._step_arrays.items()
                },
            ),
        )

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _Request:
    task_id: str = "p0_grasp"
    episode_id: str = "source"
    split: Any = "train"
    seed: int = 1
    action_mode: Any = "E7"
    observation_track: Any = None
    object_mode: str = "cuboid"
    reset_mode: str = "grasp"
    factors: dict[str, float] = field(default_factory=dict)
    api_version: str = "db-api-v0.1"


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _FakeEnv, _Device]:
    device = _Device("cuda:0")
    float32 = object()
    torch = types.ModuleType("torch")
    torch.float32 = float32
    torch.int32 = "int32"
    torch.int64 = "int64"
    torch.device = lambda value: _Device(value)
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        current_stream=lambda selected: SimpleNamespace(
            cuda_stream=555,
            device=selected,
        ),
        get_device_properties=lambda selected: SimpleNamespace(
            uuid=_GPU_UUID,
            pci_domain_id=0,
            pci_bus_id=0x17,
            pci_device_id=0,
        ),
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
        "se3_wam.benchmark.api",
        "se3_wam.benchmark.config",
        "se3_wam.benchmark.contracts",
        "se3_wam.benchmark.gpu_native",
        "se3_wam.benchmark.gpu_native.factory",
        "se3_wam.benchmark.gpu_native.p0_grasp_engine",
        "se3_wam.benchmark.gpu_native.tasks",
        "se3_wam.benchmark.p0_grasp_manifest",
    )
    modules: dict[str, types.ModuleType] = {}
    for name in package_names:
        module = types.ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    class ObservationTrack(enum.Enum):
        STATE = "state"

    class Split(enum.Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST_ID = "test_id"
        TEST_OOD = "test_ood"

    class GpuNativeConsumer(enum.Enum):
        RL = "rl"

    env = _FakeEnv(3, device, float32)
    captured: dict[str, Any] = {}

    def make_gpu_native_env(task_id: str, **kwargs: Any) -> _FakeEnv:
        captured["factory"] = (task_id, kwargs)
        return env

    modules["se3_wam.benchmark.contracts"].ObservationTrack = ObservationTrack
    modules["se3_wam.benchmark.api"].Split = Split
    modules["se3_wam.benchmark.config"].load_task_config = lambda task_id: {
        "clock": {"horizon_steps": 250}
    }
    modules["se3_wam.benchmark.gpu_native.tasks"].GpuNativeConsumer = GpuNativeConsumer
    modules["se3_wam.benchmark.gpu_native.factory"].make_gpu_native_env = make_gpu_native_env
    modules["se3_wam.benchmark.gpu_native.p0_grasp_engine"].load_p0_grasp_artifacts = (
        lambda export_dir: SimpleNamespace(reset_request=_Request())
    )

    def make_manifest(*, split: Any, attempts: int, manifest_seed: int) -> tuple[Any, ...]:
        rows = []
        for index in range(attempts):
            factors = {
                "initial_x_m": -0.08 + 0.001 * index,
                "initial_y_m": -0.26 + 0.0001 * index,
                "initial_yaw_rad": -1.0 + 0.01 * index,
                "forward_speed_m_s": 0.03 + 0.0001 * index,
                "lateral_amplitude_m": 0.015 + 0.00001 * index,
                "lateral_frequency_hz": 0.3 + 0.0001 * index,
                "lateral_phase_rad": -3.0 + 0.01 * index,
                "yaw_rate_rad_s": -1.0 + 0.01 * index,
            }
            rows.append(
                SimpleNamespace(
                    request=_Request(
                        episode_id=f"manifest-{index:05d}",
                        split=split,
                        seed=manifest_seed + index,
                        action_mode="E7",
                        observation_track=ObservationTrack.STATE,
                        factors=factors,
                    )
                )
            )
        return tuple(rows)

    modules[
        "se3_wam.benchmark.p0_grasp_manifest"
    ].make_p0_grasp_candidate_manifest = make_manifest
    monkeypatch.setattr(
        _MODULE,
        "_nvidia_smi_identity",
        lambda expected_uuid: (_GPU_UUID, _PCI_BUS_ID, "590.48.01"),
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
    assert fake_env.capabilities.required == (
        "full_batch_reset",
        "device_tensor_step",
        "device_terminal_mask",
    )
    assert backend.cohort_horizon_steps == 250
    assert backend.api_version == "gpu-native-api-v0.2"
    assert captured["factory"][1]["engine_kwargs"] == {
        "expected_device_uuid": _GPU_UUID
    }

    reset = backend.reset()
    assert reset.observation.device == device
    assert reset.episode_ids == (
        "manifest-00000-cycle00000000",
        "manifest-00001-cycle00000000",
        "manifest-00002-cycle00000000",
    )
    assert reset.seeds == (20261050, 20261051, 20261052)
    assert reset.manifest_ordinals == (0, 1, 2)
    assert reset.manifest_sha256 == backend.manifest_sha256
    assert len({request.seed for request in fake_env.reset_calls[0]}) == 3
    assert len({tuple(request.factors.values()) for request in fake_env.reset_calls[0]}) == 3
    assert len(fake_env.reset_calls[0]) == 3
    action = _Tensor((3, 7), device, 9999, sys.modules["torch"].float32)
    step = backend.step(action)
    assert fake_env.step_calls == [action]
    assert step.observation.device == device
    assert step.physics_step.device == device
    assert step.generation == 2
    assert backend.device_identity.torch_uuid == _GPU_UUID
    assert backend.device_identity.warp_pci_bus_id == _PCI_BUS_ID
    assert backend.transport_checks == 1
    assert backend.last_transport_receipt["action_input_ptr"] == 9999
    first_conversion_count = sys.modules["warp"].to_torch_calls
    backend.step(action)
    assert sys.modules["warp"].to_torch_calls == first_conversion_count
    backend.close()
    assert fake_env.closed


def test_r0_cursor_is_transactional_boundary_only_and_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        manifest_seed=77,
        manifest_size=6,
    )
    preview = tuple(request.episode_id for request in backend.next_requests())
    fake_env.reset_error = True
    with pytest.raises(RuntimeError, match="injected"):
        backend.reset()
    assert tuple(request.episode_id for request in backend.next_requests()) == preview
    fake_env.reset_error = False
    first = backend.reset()
    assert first.manifest_ordinals == (0, 1, 2)
    action = _Tensor((3, 7), device, 9999, sys.modules["torch"].float32)
    backend._cohort_horizon_steps = 2  # noqa: SLF001
    backend.step(action)
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="reset boundary"):
        backend.manifest_state_dict()
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="fixed horizon"):
        backend.reset()
    backend.step(action)
    state = dict(backend.manifest_state_dict())
    expected_next = tuple(request.episode_id for request in backend.next_requests())

    _captured2, _fake_env2, _device2 = _install_fakes(monkeypatch)
    resumed = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        manifest_seed=77,
        manifest_size=6,
    )
    resumed.load_manifest_state_dict(state)
    assert tuple(request.episode_id for request in resumed.next_requests()) == expected_next
    resumed_reset = resumed.reset()
    assert resumed_reset.manifest_ordinals == (3, 4, 5)
    tampered = dict(state)
    tampered["manifest_sha256"] = "0" * 64
    third_captured, _third_env, _third_device = _install_fakes(monkeypatch)
    assert third_captured == {}
    rejected = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        manifest_seed=77,
        manifest_size=6,
    )
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="identity mismatch"):
        rejected.load_manifest_state_dict(tampered)


def test_tensor_backend_rejects_cross_runtime_device_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    fake_env.provenance.physical_device_pci_bus_id = "00000000:65:00.0"

    with pytest.raises(
        GpuNativeTensorBackendUnavailableError,
        match="PCI identities disagree",
    ):
        GpuNativeTensorBackendEnv(
            task_id="p0_grasp",
            num_envs=3,
            export_dir="/tmp/export",
            expected_gpu_uuid=_GPU_UUID,
        )
    assert fake_env.closed is True


@pytest.mark.parametrize("fault", ["action_pointer", "stream", "reward_dtype"])
def test_tensor_backend_rejects_transport_and_layout_drift(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
    )
    backend.reset()
    action = _Tensor((3, 7), device, 9999, sys.modules["torch"].float32)
    if fault == "action_pointer":
        fake_env.transport_action_offset = 4
    elif fault == "stream":
        fake_env.transport_stream_ptr = 777
    else:
        fake_env.step_device(action)
        assert fake_env._step_arrays is not None
        fake_env._step_arrays["reward"].tensor.dtype = "int32"

    match = "receipt" if fault != "reward_dtype" else "wrong dtype"
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match=match):
        backend.step(action)
