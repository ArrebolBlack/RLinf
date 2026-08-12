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

import ast
import enum
import importlib.util
import inspect
import sys
import textwrap
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
_SPEC = importlib.util.spec_from_file_location(
    "_gpu_tensor_backend_under_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
GpuNativeTensorBackendEnv = _MODULE.GpuNativeTensorBackendEnv
GpuNativeTensorBackendUnavailableError = _MODULE.GpuNativeTensorBackendUnavailableError
GpuNativeTensorTerminalRow = _MODULE.GpuNativeTensorTerminalRow
_require_single_uuid_visibility = _MODULE._require_single_uuid_visibility
_zero_copy_torch_view = _MODULE._zero_copy_torch_view

_GPU_UUID = "GPU-803b6f88-a884-134a-d92d-cdc532e22e14"
_OTHER_GPU_UUID = "GPU-7d65018e-96c7-8c5e-bb64-3a74ca558ab3"
_PCI_BUS_ID = "00000000:17:00.0"
_SE3_SOURCE_COMMIT = "a" * 40
_SE3_SOURCE_TREE = "b" * 40
_EXPORT_DIGESTS = {
    "request_sha256": "1" * 64,
    "bundle_sha256": "2" * 64,
    "model_sha256": "3" * 64,
    "config_sha256": "4" * 64,
    "manifest_sha256": "5" * 64,
}


@dataclass(frozen=True)
class _Device:
    name: str

    @property
    def type(self) -> str:
        return self.name.split(":", 1)[0]

    @property
    def index(self) -> int:
        return int(self.name.split(":", 1)[1])

    def __str__(self) -> str:
        return self.name


class _Tensor:
    def __init__(
        self, shape: tuple[int, ...], device: _Device, pointer: int, dtype: Any
    ) -> None:
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self._pointer = pointer
        self.host_materialization_calls: list[str] = []

    def data_ptr(self) -> int:
        return self._pointer

    def is_contiguous(self) -> bool:
        return True

    def cpu(self) -> Any:
        self.host_materialization_calls.append("cpu")
        raise AssertionError("hot path called cpu()")

    def numpy(self) -> Any:
        self.host_materialization_calls.append("numpy")
        raise AssertionError("hot path called numpy()")

    def item(self) -> Any:
        self.host_materialization_calls.append("item")
        raise AssertionError("hot path called item()")

    def __ne__(self, _other: Any) -> _Tensor:
        return _Tensor(self.shape, self.device, self._pointer + 10_000, "bool")

    def __or__(self, other: _Tensor) -> _Tensor:
        assert other.shape == self.shape
        assert other.device == self.device
        return _Tensor(self.shape, self.device, self._pointer + 20_000, "bool")


class _WarpArray:
    def __init__(self, tensor: _Tensor) -> None:
        self.shape = tensor.shape
        self.ptr = tensor.data_ptr()
        self.tensor = tensor


class _Capabilities:
    __dataclass_fields__ = dict.fromkeys(_MODULE._CAPABILITY_NAMES)

    def __init__(self, values: dict[str, bool]) -> None:
        assert set(values) == set(_MODULE._CAPABILITY_NAMES)
        for name, value in values.items():
            setattr(self, name, value)
        self.required: tuple[str, ...] | None = None

    @property
    def available(self) -> frozenset[str]:
        return frozenset(
            name for name in _MODULE._CAPABILITY_NAMES if getattr(self, name)
        )

    def require(self, *names: str) -> None:
        self.required = names
        missing = [name for name in names if not getattr(self, name)]
        if missing:
            raise RuntimeError(f"missing capabilities: {missing}")


@dataclass(frozen=True)
class _EpisodeQualitySummary:
    episode_id: str
    task_id: str = "p0_grasp"
    terminal: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "terminal": self.terminal,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> _EpisodeQualitySummary:
        return cls(**value)


@dataclass(frozen=True)
class _EventRecord:
    name: str
    physics_step: int
    time_s: float
    details: dict[str, Any] = field(default_factory=dict)


class _TerminalOutcome(enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class _TerminalLedgerRequest:
    lanes: tuple[int, ...]
    episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class _TerminalLedgerRow:
    lane: int
    episode_id: str
    task_id: str
    outcome: _TerminalOutcome
    terminated: bool
    truncated: bool
    success: bool
    termination_reason: str
    physics_step: int
    control_step: int
    policy_step: int
    completion: float
    events: tuple[_EventRecord, ...]
    task_quality: _EpisodeQualitySummary | None


@dataclass(frozen=True)
class _TerminalLedgerBatch:
    backend_id: str
    provenance: Any
    rows: tuple[_TerminalLedgerRow, ...]


@dataclass(frozen=True)
class _ObservationBundle:
    episode_id: str
    task_id: str
    physics_step: int
    control_step: int
    policy_step: int


@dataclass(frozen=True)
class _AuditRequest:
    lanes: tuple[int, ...]
    include_step_result: bool = False


@dataclass(frozen=True)
class _AuditLane:
    lane: int
    episode_id: str
    observation: _ObservationBundle
    step_result: Any | None = None


@dataclass(frozen=True)
class _AuditBatch:
    backend_id: str
    provenance: Any
    lanes: tuple[_AuditLane, ...]


class _FakeEnv:
    backend_id = "mjwarp_gpu_v1"

    def __init__(
        self,
        batch_size: int,
        device: _Device,
        dtype: Any,
        call_order: list[str],
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.call_order = call_order
        self.capabilities = _Capabilities(dict(_MODULE._P0_STATE_ENV_CAPABILITIES))
        self.contract = SimpleNamespace(
            backend_id="mjwarp_gpu_v1",
            api_version="gpu-native-api-v0.2",
            batch_size=batch_size,
            task_id="p0_grasp",
            consumer="rl",
            observation_track="state",
            action_mode="E7",
            physics_hz=500,
            control_hz=20,
            sensor_hz=20,
            capabilities=_Capabilities(dict(_MODULE._P0_STATE_CONTRACT_CAPABILITIES)),
        )
        self.provenance = SimpleNamespace(
            backend_id="mjwarp_gpu_v1",
            device_platform="cuda",
            precision="float32",
            device_ordinal=0,
            git_commit=_SE3_SOURCE_COMMIT,
            git_tree=_SE3_SOURCE_TREE,
            implementation_version="direct-mjwarp-p0-grasp-foundation-v0.1",
            runtime_versions={
                "mujoco": "3.3.4",
                "mujoco-warp": "0.1.0",
                "warp-lang": "1.10.1",
            },
            physical_device_uuid=_GPU_UUID,
            physical_device_pci_bus_id=_PCI_BUS_ID,
            physical_device_identity_source="warp_cuda_driver",
            **_EXPORT_DIGESTS,
        )
        self.reset_calls: list[tuple[Any, ...]] = []
        self.reset_masks: list[Any] = []
        self.reset_write_count = 0
        self.step_calls: list[Any] = []
        self.materialize_audit_calls: list[Any] = []
        self.materialize_terminal_ledger_calls: list[Any] = []
        self.quality_calls: list[dict[str, str]] = []
        self.closed = False
        self.transport_action_offset = 0
        self.transport_stream_ptr = 555
        self._pointer = 1000
        self._step_arrays: dict[str, _WarpArray] | None = None
        self.reset_error = False
        self.control_step = 0
        self.audit_lane_order_override: tuple[int, ...] | None = None
        self.terminal_outcomes: dict[int, dict[str, Any]] = {}
        self._device_terminal_consumed_lanes: set[int] = set()

    def _array(self, dtype: Any, width: int | None = None) -> _WarpArray:
        self._pointer += 100
        shape = (self.batch_size,) if width is None else (self.batch_size, width)
        return _WarpArray(_Tensor(shape, self.device, self._pointer, dtype))

    def reset(self, requests: Any, reset_mask: Any | None = None) -> Any:
        if self.reset_error:
            raise RuntimeError("injected reset failure")
        self.call_order.append("reset")
        self.reset_write_count += 1
        self.reset_calls.append(tuple(requests))
        self.reset_masks.append(reset_mask)
        self._device_terminal_consumed_lanes.clear()
        self.control_step = 0
        return SimpleNamespace(
            observation=self._array(self.dtype, 5),
            state=SimpleNamespace(
                backend_id="mjwarp_gpu_v1",
                batch_size=self.batch_size,
                device_platform="cuda",
                generation=1,
            ),
        )

    def step_device(self, action: Any) -> Any:
        self.step_calls.append(action)
        self.control_step += 1
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
            state=SimpleNamespace(
                backend_id="mjwarp_gpu_v1",
                batch_size=self.batch_size,
                device_platform="cuda",
                generation=2,
            ),
            transport=SimpleNamespace(
                action_input_ptr=action.data_ptr(),
                action_engine_ptr=action.data_ptr() + self.transport_action_offset,
                stream_ptr=self.transport_stream_ptr,
                output_ptrs={
                    name: value.ptr for name, value in self._step_arrays.items()
                },
            ),
        )

    def materialize_audit(self, request: Any) -> _AuditBatch:
        self.materialize_audit_calls.append(request)
        active_requests = self.reset_calls[-1]
        lanes = (
            request.lanes
            if self.audit_lane_order_override is None
            else self.audit_lane_order_override
        )
        return _AuditBatch(
            backend_id="mjwarp_gpu_v1",
            provenance=self.provenance,
            lanes=tuple(
                _AuditLane(
                    lane=lane,
                    episode_id=active_requests[lane].episode_id,
                    observation=_ObservationBundle(
                        episode_id=active_requests[lane].episode_id,
                        task_id="p0_grasp",
                        physics_step=25 * self.control_step,
                        control_step=self.control_step,
                        policy_step=self.control_step,
                    ),
                )
                for lane in lanes
            ),
        )

    def enable_task_quality(self, **kwargs: str) -> None:
        self.call_order.append("enable_task_quality")
        self.quality_calls.append(dict(kwargs))

    def materialize_terminal_ledger(self, request: Any) -> Any:
        self.materialize_terminal_ledger_calls.append(request)
        active_requests = self.reset_calls[-1]
        for lane, requested_episode_id in zip(
            request.lanes, request.episode_ids, strict=True
        ):
            if active_requests[lane].episode_id != requested_episode_id:
                raise RuntimeError("device terminal ledger episode identity mismatch")
            if lane in self._device_terminal_consumed_lanes:
                raise RuntimeError(
                    f"device terminal ledger lane {lane} was already consumed"
                )
        rows = []
        for lane in request.lanes:
            episode_id = active_requests[lane].episode_id
            outcome = {
                "terminated": True,
                "truncated": False,
                "success": False,
                "termination_reason": "invalid_state",
                "completion": 0.5,
                **self.terminal_outcomes.get(lane, {}),
            }
            quality = (
                _EpisodeQualitySummary(episode_id)
                if outcome.pop("quality", False)
                else None
            )
            typed_outcome = (
                _TerminalOutcome.SUCCESS
                if outcome["success"]
                else _TerminalOutcome.TIMEOUT
                if outcome["truncated"]
                else _TerminalOutcome.FAILURE
            )
            rows.append(
                _TerminalLedgerRow(
                    lane=lane,
                    episode_id=episode_id,
                    task_id="p0_grasp",
                    outcome=typed_outcome,
                    events=(
                        _EventRecord("task_start", 0, 0.0),
                        _EventRecord(outcome["termination_reason"], 25, 0.05),
                    ),
                    task_quality=quality,
                    **outcome,
                    physics_step=25,
                    control_step=1,
                    policy_step=1,
                )
            )
        batch = _TerminalLedgerBatch(
            backend_id="mjwarp_gpu_v1",
            provenance=self.provenance,
            rows=tuple(rows),
        )
        self._device_terminal_consumed_lanes.update(request.lanes)
        return batch

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


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch_size: int = 3,
) -> tuple[Any, _FakeEnv, _Device]:
    device = _Device("cuda:0")
    float32 = object()
    captured: dict[str, Any] = {
        "call_order": [],
        "factory_hook": None,
        "nvml_uuid": _GPU_UUID,
        "nvml_pci": _PCI_BUS_ID,
        "driver_version": "590.48.01",
        "export_digests": dict(_EXPORT_DIGESTS),
    }
    torch_properties = SimpleNamespace(
        # PyTorch 2.12 ``str(_CUuuid)`` omits NVML's ``GPU-`` prefix.
        uuid=_GPU_UUID.removeprefix("GPU-"),
        pci_domain_id=0,
        pci_bus_id=0x17,
        pci_device_id=0,
    )
    captured["torch_properties"] = torch_properties
    torch = types.ModuleType("torch")
    torch.float32 = float32
    torch.int32 = "int32"
    torch.int64 = "int64"
    torch.bool = "bool"
    torch.device = lambda value: _Device(value)
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        current_device=lambda: 0,
        current_stream=lambda selected: SimpleNamespace(
            cuda_stream=555,
            device=selected,
        ),
        get_device_properties=lambda selected: torch_properties,
    )
    warp = types.ModuleType("warp")
    warp.to_torch_calls = 0
    warp_device = SimpleNamespace(
        is_cuda=True,
        ordinal=0,
        alias="cuda:0",
        uuid=_GPU_UUID,
        pci_bus_id=_PCI_BUS_ID,
    )
    captured["warp_device"] = warp_device

    def to_torch(value: Any) -> Any:
        warp.to_torch_calls += 1
        return value.tensor

    warp.to_torch = to_torch
    warp.init = lambda: None
    warp.get_device = lambda alias: warp_device
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
        "se3_wam.benchmark.gpu_native.audit",
        "se3_wam.benchmark.gpu_native.factory",
        "se3_wam.benchmark.gpu_native.p0_grasp_engine",
        "se3_wam.benchmark.gpu_native.tasks",
        "se3_wam.benchmark.p0_grasp_manifest",
        "se3_wam.benchmark.task_quality",
    )
    modules: dict[str, types.ModuleType] = {}
    for name in package_names:
        module = types.ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    class ObservationTrack(enum.Enum):
        STATE = "state"
        HYBRID = "hybrid"

    class Split(enum.Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST_ID = "test_id"
        TEST_OOD = "test_ood"

    class GpuNativeConsumer(enum.Enum):
        RL = "rl"

    env = _FakeEnv(batch_size, device, float32, captured["call_order"])

    def make_gpu_native_env(task_id: str, **kwargs: Any) -> _FakeEnv:
        captured["call_order"].append("factory")
        captured["factory"] = (task_id, kwargs)
        hook = captured["factory_hook"]
        if hook is not None:
            hook()
        return env

    modules["se3_wam.benchmark.contracts"].ObservationTrack = ObservationTrack
    modules["se3_wam.benchmark.contracts"].EventRecord = _EventRecord
    modules["se3_wam.benchmark.api"].Split = Split
    modules["se3_wam.benchmark.config"].load_task_config = lambda task_id: {
        "clock": {"horizon_steps": 250}
    }
    modules["se3_wam.benchmark.gpu_native.tasks"].GpuNativeConsumer = GpuNativeConsumer
    modules["se3_wam.benchmark.api"].ObservationBundle = _ObservationBundle
    modules["se3_wam.benchmark.gpu_native.audit"].AuditBatch = _AuditBatch
    modules["se3_wam.benchmark.gpu_native.audit"].AuditLane = _AuditLane
    modules["se3_wam.benchmark.gpu_native.audit"].AuditRequest = _AuditRequest
    modules[
        "se3_wam.benchmark.gpu_native.audit"
    ].TerminalLedgerBatch = _TerminalLedgerBatch
    modules[
        "se3_wam.benchmark.gpu_native.audit"
    ].TerminalLedgerRequest = _TerminalLedgerRequest
    modules["se3_wam.benchmark.gpu_native.audit"].TerminalLedgerRow = _TerminalLedgerRow
    modules["se3_wam.benchmark.gpu_native.audit"].TerminalOutcome = _TerminalOutcome
    modules[
        "se3_wam.benchmark.task_quality"
    ].EpisodeQualitySummary = _EpisodeQualitySummary
    modules[
        "se3_wam.benchmark.gpu_native.factory"
    ].make_gpu_native_env = make_gpu_native_env

    def load_artifacts(export_dir: str) -> Any:
        digests = captured["export_digests"]
        return SimpleNamespace(
            reset_request=_Request(observation_track=ObservationTrack.STATE),
            request_identity_sha256=digests["request_sha256"],
            bundle_sha256=digests["bundle_sha256"],
            model_sha256=digests["model_sha256"],
            config_sha256=digests["config_sha256"],
            manifest_sha256=digests["manifest_sha256"],
        )

    modules[
        "se3_wam.benchmark.gpu_native.p0_grasp_engine"
    ].load_p0_grasp_artifacts = load_artifacts

    def make_manifest(
        *, split: Any, attempts: int, manifest_seed: int
    ) -> tuple[Any, ...]:
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
                        observation_track=ObservationTrack.HYBRID,
                        factors=factors,
                    )
                )
            )
        captured["generated_manifest_requests"] = tuple(row.request for row in rows)
        return tuple(rows)

    modules[
        "se3_wam.benchmark.p0_grasp_manifest"
    ].make_p0_grasp_candidate_manifest = make_manifest
    monkeypatch.setattr(
        _MODULE,
        "_nvidia_smi_identity",
        lambda expected_uuid: (
            captured["nvml_uuid"],
            captured["nvml_pci"],
            captured["driver_version"],
        ),
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
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    assert {
        name: getattr(fake_env.capabilities, name) for name in _MODULE._CAPABILITY_NAMES
    } == dict(_MODULE._P0_STATE_ENV_CAPABILITIES)
    assert backend.cohort_horizon_steps == 250
    assert backend.api_version == "gpu-native-api-v0.2"
    assert captured["factory"][1]["engine_kwargs"] == {
        "expected_device_uuid": _GPU_UUID,
        "expected_model_sha256": _EXPORT_DIGESTS["model_sha256"],
        "expected_source_commit": _SE3_SOURCE_COMMIT,
        "expected_source_tree": _SE3_SOURCE_TREE,
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
    assert tuple(request.episode_id for request in reset.requests) == reset.episode_ids
    assert len({request.seed for request in fake_env.reset_calls[0]}) == 3
    assert (
        len({tuple(request.factors.values()) for request in fake_env.reset_calls[0]})
        == 3
    )
    assert len(fake_env.reset_calls[0]) == 3
    action = _Tensor((3, 7), device, 9999, sys.modules["torch"].float32)
    step = backend.step_device(action)
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


def test_t4_admission_requires_owner_frozen_manifest_before_runtime_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    with pytest.raises(
        GpuNativeTensorBackendUnavailableError,
        match="Owner-frozen caller manifest",
    ):
        GpuNativeTensorBackendEnv(
            task_id="t4_sphere",
            num_envs=1,
            export_dir="/tmp/export",
            expected_gpu_uuid=_GPU_UUID,
            expected_se3_source_commit=_SE3_SOURCE_COMMIT,
            expected_se3_source_tree=_SE3_SOURCE_TREE,
        )


def test_generated_manifest_projects_only_observation_track_to_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    captured, _fake_env, _device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
        manifest_seed=77,
        manifest_size=6,
    )

    source_requests = captured["generated_manifest_requests"]
    projected_requests = backend.sequence_requests
    track = sys.modules["se3_wam.benchmark.contracts"].ObservationTrack
    assert all(request.observation_track is track.HYBRID for request in source_requests)
    assert all(
        request.observation_track is track.STATE for request in projected_requests
    )
    for source, projected in zip(source_requests, projected_requests, strict=True):
        assert projected.task_id == source.task_id
        assert projected.episode_id == source.episode_id
        assert projected.split == source.split
        assert projected.seed == source.seed
        assert projected.action_mode == source.action_mode
        assert projected.object_mode == source.object_mode
        assert projected.reset_mode == source.reset_mode
        assert projected.factors == source.factors
        assert projected.api_version == source.api_version


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
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
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
    backend.step_device(action)
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
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
        manifest_seed=77,
        manifest_size=6,
    )
    resumed.load_manifest_state_dict(state)
    assert (
        tuple(request.episode_id for request in resumed.next_requests())
        == expected_next
    )
    resumed_reset = resumed.reset()
    assert resumed_reset.manifest_ordinals == (3, 4, 5)
    tampered = dict(state)
    tampered["manifest_sha256"] = "0" * 64
    third_captured, _third_env, _third_device = _install_fakes(monkeypatch)
    assert third_captured["call_order"] == []
    rejected = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
        manifest_seed=77,
        manifest_size=6,
    )
    with pytest.raises(
        GpuNativeTensorBackendUnavailableError, match="identity mismatch"
    ):
        rejected.load_manifest_state_dict(tampered)


def test_tensor_backend_rejects_cross_runtime_device_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    fake_env.provenance.physical_device_pci_bus_id = "00000000:65:00.0"

    with pytest.raises(
        GpuNativeTensorBackendUnavailableError,
        match="provenance disagrees",
    ):
        GpuNativeTensorBackendEnv(
            task_id="p0_grasp",
            num_envs=3,
            export_dir="/tmp/export",
            expected_gpu_uuid=_GPU_UUID,
            expected_se3_source_commit=_SE3_SOURCE_COMMIT,
            expected_se3_source_tree=_SE3_SOURCE_TREE,
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
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
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


@pytest.mark.parametrize(
    "fault",
    (
        "torch_uuid",
        "torch_pci",
        "warp_uuid",
        "warp_pci",
        "nvml_uuid",
        "nvml_pci",
        "warp_alias",
        "warp_ordinal",
    ),
)
def test_constructor_rejects_uuid_pci_alias_and_ordinal_drift(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    captured, _fake_env, _device = _install_fakes(monkeypatch)
    if fault == "torch_uuid":
        captured["torch_properties"].uuid = _OTHER_GPU_UUID.removeprefix("GPU-")
    elif fault == "torch_pci":
        captured["torch_properties"].pci_bus_id = 0x65
    elif fault == "warp_uuid":
        captured["warp_device"].uuid = _OTHER_GPU_UUID
    elif fault == "warp_pci":
        captured["warp_device"].pci_bus_id = "00000000:65:00.0"
    elif fault == "nvml_uuid":
        captured["nvml_uuid"] = _OTHER_GPU_UUID
    elif fault == "nvml_pci":
        captured["nvml_pci"] = "00000000:65:00.0"
    elif fault == "warp_alias":
        captured["warp_device"].alias = "cuda:1"
    else:
        captured["warp_device"].ordinal = 1

    with pytest.raises(GpuNativeTensorBackendUnavailableError):
        GpuNativeTensorBackendEnv(
            task_id="p0_grasp",
            num_envs=3,
            export_dir="/tmp/export",
            expected_gpu_uuid=_GPU_UUID,
            expected_se3_source_commit=_SE3_SOURCE_COMMIT,
            expected_se3_source_tree=_SE3_SOURCE_TREE,
        )


def test_constructor_normalizes_only_exact_unprefixed_torch_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    captured, _fake_env, _device = _install_fakes(monkeypatch)
    assert captured["torch_properties"].uuid == _GPU_UUID.removeprefix("GPU-")
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    assert backend.device_identity.torch_uuid == _GPU_UUID

    captured["torch_properties"].uuid = f"MIG-{_GPU_UUID}"
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="PyTorch"):
        backend.attest_end()


def test_attest_end_reobserves_after_success_and_close_checks_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    captured, fake_env, _device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    assert backend.attest_end() == backend.device_identity_start

    captured["torch_properties"].uuid = _OTHER_GPU_UUID.removeprefix("GPU-")
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="PyTorch|identit"):
        backend.attest_end()
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="PyTorch|identit"):
        backend.close()
    assert fake_env.closed is True


def test_constructor_rejects_identity_change_between_start_and_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    captured, fake_env, _device = _install_fakes(monkeypatch)
    captured["factory_hook"] = lambda: captured.__setitem__(
        "driver_version", "590.48.02"
    )

    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="changed during"):
        GpuNativeTensorBackendEnv(
            task_id="p0_grasp",
            num_envs=3,
            export_dir="/tmp/export",
            expected_gpu_uuid=_GPU_UUID,
            expected_se3_source_commit=_SE3_SOURCE_COMMIT,
            expected_se3_source_tree=_SE3_SOURCE_TREE,
        )
    assert fake_env.closed is True


@pytest.mark.parametrize("scope", ("contract", "environment"))
@pytest.mark.parametrize("capability_name", _MODULE._CAPABILITY_NAMES)
def test_exact_v02_capability_difference_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    capability_name: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    capabilities = (
        fake_env.contract.capabilities if scope == "contract" else fake_env.capabilities
    )
    setattr(capabilities, capability_name, not getattr(capabilities, capability_name))

    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="capability"):
        GpuNativeTensorBackendEnv(
            task_id="p0_grasp",
            num_envs=3,
            export_dir="/tmp/export",
            expected_gpu_uuid=_GPU_UUID,
            expected_se3_source_commit=_SE3_SOURCE_COMMIT,
            expected_se3_source_tree=_SE3_SOURCE_TREE,
        )
    assert fake_env.closed is True


def test_old_v01_abi_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    fake_env.contract.api_version = "gpu-native-api-v0.1"
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="v0.2"):
        GpuNativeTensorBackendEnv(
            task_id="p0_grasp",
            num_envs=3,
            export_dir="/tmp/export",
            expected_gpu_uuid=_GPU_UUID,
            expected_se3_source_commit=_SE3_SOURCE_COMMIT,
            expected_se3_source_tree=_SE3_SOURCE_TREE,
        )


def test_task_quality_is_enabled_before_first_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    captured, fake_env, _device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
        task_quality_schema_version="db0-episode-task-quality-v1",
        task_quality_evaluator_backend_id="quality-backend-v1",
    )
    assert captured["call_order"] == ["factory", "enable_task_quality"]
    assert fake_env.quality_calls == [
        {
            "schema_version": "db0-episode-task-quality-v1",
            "evaluator_backend_id": "quality-backend-v1",
        }
    ]
    backend.reset()
    assert captured["call_order"] == ["factory", "enable_task_quality", "reset"]


def test_public_task_quality_enable_is_pre_reset_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    backend.enable_task_quality(
        schema_version="db0-episode-task-quality-v1",
        evaluator_backend_id="quality-backend-v1",
    )
    assert fake_env.quality_calls
    backend.reset()
    with pytest.raises(RuntimeError, match="before the first reset"):
        backend.enable_task_quality(
            schema_version="db0-episode-task-quality-v1",
            evaluator_backend_id="quality-backend-v1",
        )


def test_stable_and_active_export_identity_are_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, _fake_env, _device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    assert not set(_EXPORT_DIGESTS).intersection(backend.stable_identity)
    assert set(backend.active_export_identity) == {
        "export_dir",
        *_EXPORT_DIGESTS,
        "frozen_request",
    }
    assert {
        name: backend.active_export_identity[name] for name in _EXPORT_DIGESTS
    } == _EXPORT_DIGESTS
    assert backend.active_cohort_identity is None
    reset = backend.reset()
    assert backend.active_cohort_identity["episode_ids"] == reset.episode_ids
    assert (
        backend.active_cohort_identity["manifest_ordinals"] == reset.manifest_ordinals
    )
    assert backend.attest_end() == backend.device_identity_start


def test_caller_pinned_manifest_is_exact_and_portable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, _fake_env, _device = _install_fakes(monkeypatch)
    generated = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export-a",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
        manifest_size=6,
    )
    requests = generated.sequence_requests
    digest = generated.manifest_sha256

    _captured2, _fake_env2, _device2 = _install_fakes(monkeypatch)
    pinned = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/different/runtime/path",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
        manifest_requests=requests,
        manifest_sha256=digest,
    )
    assert pinned.sequence_requests == requests
    assert pinned.stable_identity["reset_manifest"]["origin"] == "caller"
    assert pinned.active_export_identity["export_dir"] == "/different/runtime/path"

    _captured3, _fake_env3, _device3 = _install_fakes(monkeypatch)
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="SHA-256"):
        GpuNativeTensorBackendEnv(
            task_id="p0_grasp",
            num_envs=3,
            export_dir="/tmp/export",
            expected_gpu_uuid=_GPU_UUID,
            expected_se3_source_commit=_SE3_SOURCE_COMMIT,
            expected_se3_source_tree=_SE3_SOURCE_TREE,
            manifest_requests=requests,
            manifest_sha256="0" * 64,
        )


def test_caller_manifest_nested_factors_are_frozen_and_defensively_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    split = sys.modules["se3_wam.benchmark.api"].Split.TRAIN
    track = sys.modules["se3_wam.benchmark.contracts"].ObservationTrack.STATE
    requests = tuple(
        _Request(
            episode_id=f"nested-{lane}",
            split=split,
            observation_track=track,
            factors={"nested": {"values": [lane, lane + 1]}},
        )
        for lane in range(3)
    )
    digest = _MODULE._manifest_sha256(requests)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
        manifest_requests=requests,
        manifest_sha256=digest,
    )

    requests[0].factors["nested"]["values"][0] = 999
    exposed = backend.sequence_requests
    assert exposed[0].factors["nested"]["values"] == [0, 1]
    exposed[0].factors["nested"]["values"][0] = 777
    assert backend.sequence_requests[0].factors["nested"]["values"] == [0, 1]
    assert backend.manifest_sha256 == digest
    backend.reset()
    assert fake_env.reset_calls[0][0].factors["nested"]["values"] == [0, 1]


def test_reset_rehashes_frozen_manifest_before_any_engine_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    object.__setattr__(
        backend._manifest_requests[0],  # noqa: SLF001
        "factors",
        {"tampered": True},
    )

    with pytest.raises(
        GpuNativeTensorBackendUnavailableError, match="manifest changed"
    ):
        backend.reset()
    assert fake_env.reset_write_count == 0


def test_b1_and_full_cohort_reset_succeed_but_partial_reset_has_zero_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="partial reset"):
        backend.reset(reset_mask=(True, False, True))
    assert fake_env.reset_write_count == 0
    backend.reset(reset_mask=(True, True, True))
    assert fake_env.reset_write_count == 1

    _captured_b1, fake_env_b1, _device_b1 = _install_fakes(monkeypatch, batch_size=1)
    backend_b1 = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=1,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    reset_b1 = backend_b1.reset(reset_mask=(True,))
    assert len(reset_b1.episode_ids) == 1
    assert fake_env_b1.reset_write_count == 1


def test_teacher_observation_control_plane_is_explicit_ordered_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    with pytest.raises(
        GpuNativeTensorBackendUnavailableError, match="active reset cohort"
    ):
        backend.materialize_teacher_observations()
    reset = backend.reset()
    observations = backend.materialize_teacher_observations((2, 0))
    assert tuple(observation.episode_id for observation in observations) == (
        reset.episode_ids[2],
        reset.episode_ids[0],
    )
    assert all(observation.policy_step == 0 for observation in observations)
    assert backend.teacher_audit_materializations == 1
    assert fake_env.materialize_audit_calls == [
        _AuditRequest(lanes=(2, 0), include_step_result=False)
    ]

    action = _Tensor((3, 7), device, 9999, sys.modules["torch"].float32)
    backend.step_device(action)
    next_observations = backend.materialize_teacher_observations()
    assert all(observation.policy_step == 1 for observation in next_observations)
    assert backend.teacher_audit_materializations == 2
    with pytest.raises(TypeError, match="immutable tuple"):
        backend.materialize_teacher_observations([0])  # type: ignore[arg-type]


def test_teacher_observation_control_plane_rejects_lane_order_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    backend.reset()
    fake_env.audit_lane_order_override = (1, 0)
    with pytest.raises(GpuNativeTensorBackendUnavailableError, match="lane order"):
        backend.materialize_teacher_observations((0, 1))
    assert backend.teacher_audit_materializations == 0


def test_steady_state_ast_and_spies_forbid_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    reset = backend.reset()
    action = _Tensor((3, 7), device, 9999, sys.modules["torch"].float32)
    step = backend.step_device(action)
    forbidden = {"cpu", "numpy", "item", "materialize_audit"}
    for function in (
        GpuNativeTensorBackendEnv.step_device,
        GpuNativeTensorBackendEnv._view,
        _zero_copy_torch_view,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        assert not {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }.intersection(forbidden)
    assert action.host_materialization_calls == []
    assert reset.observation.host_materialization_calls == []
    assert fake_env.materialize_audit_calls == []
    assert backend.teacher_audit_materializations == 0
    output_ptrs = backend.last_transport_receipt["output_ptrs"]
    for name in output_ptrs:
        tensor = getattr(step, name)
        assert tensor.data_ptr() == output_ptrs[name]
        assert tensor.device == device
        assert tensor.is_contiguous()
        assert tensor.host_materialization_calls == []
    assert step.episode_ids == reset.episode_ids
    assert step.done.shape == (3,)
    assert step.done.device == device
    assert step.done.dtype == "bool"


@pytest.mark.parametrize(
    "fault",
    ("output_pointer", "observation_shape", "event_device", "physics_dtype"),
)
def test_each_device_output_contract_drift_fails_closed(
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
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    backend.reset()
    action = _Tensor((3, 7), device, 9999, sys.modules["torch"].float32)
    preview = fake_env.step_device(action)
    assert fake_env._step_arrays is not None
    if fault == "output_pointer":
        fake_env._step_arrays["event_mask"].ptr += 4
    elif fault == "observation_shape":
        fake_env._step_arrays["observation"].tensor.shape = (3, 6)
        fake_env._step_arrays["observation"].shape = (3, 6)
    elif fault == "event_device":
        fake_env._step_arrays["event_mask"].tensor.device = _Device("cuda:1")
    else:
        fake_env._step_arrays["physics_step"].tensor.dtype = "int32"
    assert preview is not None
    with pytest.raises(GpuNativeTensorBackendUnavailableError):
        backend.step_device(action)


def test_terminal_ledger_is_typed_once_and_quality_is_success_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=3,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
        task_quality_schema_version="db0-episode-task-quality-v1",
        task_quality_evaluator_backend_id="quality-backend-v1",
    )
    reset = backend.reset()
    fake_env.terminal_outcomes = {
        0: {
            "terminated": True,
            "truncated": False,
            "success": True,
            "termination_reason": "success",
            "completion": 1.0,
            "quality": True,
        },
        1: {
            "terminated": True,
            "truncated": False,
            "success": False,
            "termination_reason": "invalid_state",
        },
        2: {
            "terminated": False,
            "truncated": True,
            "success": False,
            "termination_reason": "timeout",
        },
    }
    rows = backend.materialize_terminal_ledger_once(
        (0, 1, 2),
        reset.episode_ids,
    )
    assert all(isinstance(row, GpuNativeTensorTerminalRow) for row in rows)
    assert isinstance(rows[0].task_quality, _EpisodeQualitySummary)
    assert rows[0].completion == 1.0
    assert rows[0].safety is None
    assert rows[1].task_quality is None
    assert rows[2].task_quality is None
    assert all(type(event) is _EventRecord for row in rows for event in row.events)
    assert "consumption_state" not in _TerminalLedgerRow.__dataclass_fields__
    adapter_source = inspect.getsource(
        GpuNativeTensorBackendEnv.materialize_terminal_ledger_once
    )
    assert "_materialized_terminal_episode_ids" not in adapter_source
    assert "TerminalLedgerConsumptionState" not in adapter_source
    assert ".consumption_state" not in adapter_source
    assert len(fake_env.materialize_terminal_ledger_calls) == 1
    with pytest.raises(RuntimeError, match="device terminal ledger.*already consumed"):
        backend.materialize_terminal_ledger_once((0,), (reset.episode_ids[0],))
    assert len(fake_env.materialize_terminal_ledger_calls) == 2


def test_terminal_adapter_accepts_live_se3_device_authoritative_abi() -> None:
    from se3_wam.benchmark.contracts import EventRecord
    from se3_wam.benchmark.gpu_native.audit import (
        TerminalLedgerBatch,
        TerminalLedgerRequest,
        TerminalLedgerRow,
        TerminalOutcome,
    )
    from se3_wam.benchmark.gpu_native.contracts import GpuNativeProvenance

    assert "consumption_state" not in TerminalLedgerRow.__dataclass_fields__
    provenance = GpuNativeProvenance(
        implementation_version="terminal-joint-abi-test",
        device_platform="cuda",
        device_name="Fake A100",
        runtime_versions={"warp-lang": "test"},
    )
    batch = TerminalLedgerBatch(
        backend_id="mjwarp_gpu_v1",
        provenance=provenance,
        rows=(
            TerminalLedgerRow(
                lane=0,
                episode_id="joint-terminal-episode",
                task_id="p0_grasp",
                outcome=TerminalOutcome.FAILURE,
                terminated=True,
                truncated=False,
                success=False,
                termination_reason="invalid_state",
                physics_step=7,
                control_step=1,
                policy_step=1,
                completion=0.5,
                events=(
                    EventRecord("task_start", 0, 0.0),
                    EventRecord("invalid_state", 7, 0.014),
                ),
                task_quality=None,
            ),
        ),
    )

    class _LiveTerminalEnv:
        def __init__(self) -> None:
            self.provenance = provenance
            self.calls = 0

        def materialize_terminal_ledger(
            self, request: TerminalLedgerRequest
        ) -> TerminalLedgerBatch:
            self.calls += 1
            assert request == TerminalLedgerRequest(
                lanes=(0,), episode_ids=("joint-terminal-episode",)
            )
            if self.calls > 1:
                raise RuntimeError("device terminal ledger lane 0 was already consumed")
            return batch

    env = _LiveTerminalEnv()
    backend = object.__new__(GpuNativeTensorBackendEnv)
    backend._closed = False  # noqa: SLF001
    backend._active_episode_ids = ("joint-terminal-episode",)  # noqa: SLF001
    backend._num_envs = 1  # noqa: SLF001
    backend._task_id = "p0_grasp"  # noqa: SLF001
    backend._env = env  # noqa: SLF001

    rows = backend.materialize_terminal_ledger_once((0,), ("joint-terminal-episode",))
    assert len(rows) == 1
    assert rows[0].physics_step == 7
    assert env.calls == 1
    with pytest.raises(RuntimeError, match="device terminal ledger.*already consumed"):
        backend.materialize_terminal_ledger_once((0,), ("joint-terminal-episode",))
    assert env.calls == 2


def test_success_terminal_without_typed_quality_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU_UUID)
    _captured, fake_env, _device = _install_fakes(monkeypatch, batch_size=1)
    backend = GpuNativeTensorBackendEnv(
        task_id="p0_grasp",
        num_envs=1,
        export_dir="/tmp/export",
        expected_gpu_uuid=_GPU_UUID,
        expected_se3_source_commit=_SE3_SOURCE_COMMIT,
        expected_se3_source_tree=_SE3_SOURCE_TREE,
    )
    reset = backend.reset()
    fake_env.terminal_outcomes[0] = {
        "terminated": True,
        "truncated": False,
        "success": True,
        "termination_reason": "success",
        "completion": 1.0,
    }
    with pytest.raises(
        GpuNativeTensorBackendUnavailableError, match="typed task quality"
    ):
        backend.materialize_terminal_ledger_once((0,), reset.episode_ids)
