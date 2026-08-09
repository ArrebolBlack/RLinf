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

"""Zero-copy GPU-native Dynamic Benchmark seam for RL samplers.

The legacy :mod:`gpu_backend` adapter deliberately materializes canonical host
objects for correctness smoke tests.  This module is the separate production
data plane: policy actions remain PyTorch CUDA tensors, Warp consumes those
tensors directly, and all outputs are returned as pointer-identical PyTorch
views of engine-owned Warp arrays.  Host audit and episode summaries belong to
a low-frequency control plane and are intentionally absent here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping


class GpuNativeTensorBackendUnavailableError(RuntimeError):
    """Raised instead of copying, materializing, or falling back to CPU."""


_RESET_CURSOR_SCHEMA_VERSION = "rlinf-gpu-native-r0-cursor-v0.1"


def _manifest_request_payload(request: Any) -> dict[str, Any]:
    def enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    return {
        "episode_id": request.episode_id,
        "task_id": request.task_id,
        "split": enum_value(request.split),
        "seed": request.seed,
        "action_mode": enum_value(request.action_mode),
        "observation_track": enum_value(request.observation_track),
        "object_mode": request.object_mode,
        "reset_mode": request.reset_mode,
        "factors": dict(request.factors),
        "api_version": request.api_version,
    }


def _manifest_sha256(requests: tuple[Any, ...]) -> str:
    payload = [_manifest_request_payload(request) for request in requests]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_GPU_UUID_PATTERN = re.compile(
    r"^GPU-([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-"
    r"([0-9a-fA-F]{4})-([0-9a-fA-F]{12})$"
)
_PCI_PATTERN = re.compile(
    r"^(?P<domain>[0-9a-fA-F]{4,8}):(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})(?:\.(?P<function>[0-7]))?$"
)


def _canonical_pci_bus_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise GpuNativeTensorBackendUnavailableError(f"{name} is unavailable")
    match = _PCI_PATTERN.fullmatch(value.strip())
    if match is None or value.strip() != value:
        raise GpuNativeTensorBackendUnavailableError(
            f"{name} is not a canonical PCI bus id"
        )
    return (
        f"{match.group('domain').zfill(8).lower()}:"
        f"{match.group('bus').lower()}:{match.group('device').lower()}."
        f"{match.group('function') or '0'}"
    )


def _canonical_gpu_uuid(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("expected_gpu_uuid must be a canonical GPU UUID")
    match = _GPU_UUID_PATTERN.fullmatch(value.strip())
    if match is None or value.strip() != value:
        raise ValueError("expected_gpu_uuid must be a canonical GPU UUID")
    return "GPU-" + "-".join(part.lower() for part in match.groups())


def _torch_device_identity(torch: Any, device: Any) -> tuple[str, str]:
    properties = torch.cuda.get_device_properties(device)
    try:
        raw_uuid = str(properties.uuid)
        # PyTorch 2.12 exposes ``_CUuuid`` as the RFC-4122 payload without
        # NVIDIA's ``GPU-`` prefix, while CUDA_VISIBLE_DEVICES and NVML use the
        # prefixed spelling.  Add only that fixed namespace prefix, then pass
        # through the same strict canonical parser used for every other source.
        if not raw_uuid.startswith("GPU-"):
            raw_uuid = f"GPU-{raw_uuid}"
        uuid = _canonical_gpu_uuid(raw_uuid)
        pci_bus_id = _canonical_pci_bus_id(
            f"{int(properties.pci_domain_id):08x}:"
            f"{int(properties.pci_bus_id):02x}:"
            f"{int(properties.pci_device_id):02x}.0",
            name="PyTorch PCI bus id",
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise GpuNativeTensorBackendUnavailableError(
            "PyTorch CUDA properties lack trusted UUID/PCI identity"
        ) from exc
    return uuid, pci_bus_id


def _nvidia_smi_identity(expected_uuid: str) -> tuple[str, str, str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuNativeTensorBackendUnavailableError(
            "nvidia-smi identity query failed"
        ) from exc
    matches: list[tuple[str, str, str]] = []
    for line in completed.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3:
            raise GpuNativeTensorBackendUnavailableError(
                "nvidia-smi identity output has an unexpected schema"
            )
        try:
            uuid = _canonical_gpu_uuid(fields[0])
        except ValueError as exc:
            raise GpuNativeTensorBackendUnavailableError(
                "nvidia-smi returned a non-canonical GPU UUID"
            ) from exc
        if uuid == expected_uuid:
            matches.append(
                (
                    uuid,
                    _canonical_pci_bus_id(fields[1], name="nvidia-smi PCI bus id"),
                    fields[2],
                )
            )
    if len(matches) != 1 or not matches[0][2]:
        raise GpuNativeTensorBackendUnavailableError(
            "nvidia-smi did not resolve exactly one expected physical GPU"
        )
    return matches[0]


@dataclass(frozen=True)
class GpuDeviceIdentityReceipt:
    """Cross-runtime identity for the one admitted physical CUDA device."""

    expected_uuid: str
    cuda_visible_devices: str
    torch_uuid: str
    torch_pci_bus_id: str
    warp_uuid: str
    warp_pci_bus_id: str
    warp_identity_source: str
    nvml_uuid: str
    nvml_pci_bus_id: str
    driver_version: str

    def __post_init__(self) -> None:
        uuids = {
            self.expected_uuid,
            self.cuda_visible_devices,
            self.torch_uuid,
            self.warp_uuid,
            self.nvml_uuid,
        }
        if len(uuids) != 1:
            raise GpuNativeTensorBackendUnavailableError(
                "CUDA_VISIBLE_DEVICES/Torch/Warp/NVML UUID identities disagree"
            )
        pci_ids = {
            self.torch_pci_bus_id,
            self.warp_pci_bus_id,
            self.nvml_pci_bus_id,
        }
        if len(pci_ids) != 1:
            raise GpuNativeTensorBackendUnavailableError(
                "Torch/Warp/NVML PCI identities disagree"
            )
        if self.warp_identity_source != "warp_cuda_driver":
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM provenance lacks CUDA-driver Warp identity"
            )
        if not self.driver_version.strip():
            raise GpuNativeTensorBackendUnavailableError(
                "NVIDIA driver version is unavailable"
            )


def _require_single_uuid_visibility(expected_gpu_uuid: str, device_ordinal: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    entries = () if visible is None else tuple(item.strip() for item in visible.split(","))
    if entries != (expected_gpu_uuid,):
        raise GpuNativeTensorBackendUnavailableError(
            "tensor backend requires CUDA_VISIBLE_DEVICES to contain exactly the "
            "expected physical GPU UUID"
        )
    if device_ordinal != 0:
        raise GpuNativeTensorBackendUnavailableError(
            "a singleton UUID CUDA_VISIBLE_DEVICES mapping exposes only logical cuda:0"
        )


def _zero_copy_torch_view(
    *,
    warp_module: Any,
    value: Any,
    name: str,
    batch_size: int,
    expected_device: Any,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: Any | None = None,
    expected_pointer: int | None = None,
) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 1 or int(shape[0]) != batch_size:
        raise GpuNativeTensorBackendUnavailableError(
            f"{name} must have leading GPU batch dimension {batch_size}"
        )
    pointer = getattr(value, "ptr", None)
    if not isinstance(pointer, int) or pointer <= 0:
        raise GpuNativeTensorBackendUnavailableError(
            f"{name} does not expose a verifiable device pointer"
        )
    if expected_pointer is not None and pointer != expected_pointer:
        raise GpuNativeTensorBackendUnavailableError(
            f"{name} engine pointer differs from its transport receipt"
        )
    try:
        tensor = warp_module.to_torch(value)
    except Exception as exc:
        raise GpuNativeTensorBackendUnavailableError(
            f"cannot expose {name} as a PyTorch view"
        ) from exc
    if int(tensor.data_ptr()) != pointer:
        raise GpuNativeTensorBackendUnavailableError(
            f"{name} Warp/PyTorch conversion copied storage"
        )
    if tensor.device != expected_device:
        raise GpuNativeTensorBackendUnavailableError(
            f"{name} view moved away from {expected_device}"
        )
    if not tensor.is_contiguous():
        raise GpuNativeTensorBackendUnavailableError(
            f"{name} view is not contiguous; implicit materialization is forbidden"
        )
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise GpuNativeTensorBackendUnavailableError(
            f"{name} has layout {tuple(tensor.shape)}, expected {expected_shape}"
        )
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise GpuNativeTensorBackendUnavailableError(f"{name} has the wrong dtype")
    return tensor


@dataclass(frozen=True)
class GpuNativeTensorReset:
    observation: Any
    generation: int
    episode_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    manifest_ordinals: tuple[int, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class GpuNativeTensorStep:
    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    success: Any
    event_mask: Any
    terminal_reason: Any
    physics_step: Any
    generation: int

    @property
    def done(self) -> Any:
        """Return a device-local logical mask; no host synchronization occurs."""

        return (self.terminated != 0) | (self.truncated != 0)


class GpuNativeTensorBackendEnv:
    """Own one full-reset cohort and expose its device-only RL data plane."""

    def __init__(
        self,
        *,
        task_id: str,
        num_envs: int,
        export_dir: str,
        expected_gpu_uuid: str,
        device_ordinal: int = 0,
        image_size: int = 64,
        split: str = "train",
        manifest_seed: int = 20261050,
        manifest_size: int = 4096,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("GPU tensor backend requires a positive num_envs")
        if not isinstance(export_dir, str) or not export_dir.strip():
            raise ValueError("GPU tensor backend requires a non-empty export_dir")
        if (
            isinstance(device_ordinal, bool)
            or not isinstance(device_ordinal, int)
            or device_ordinal < 0
        ):
            raise ValueError("device_ordinal must be a non-negative integer")
        expected_gpu_uuid = _canonical_gpu_uuid(expected_gpu_uuid)
        _require_single_uuid_visibility(expected_gpu_uuid, device_ordinal)
        if task_id != "p0_grasp":
            raise ValueError("GPU tensor R0 admission is currently limited to p0_grasp")
        if isinstance(manifest_seed, bool) or not isinstance(manifest_seed, int) or manifest_seed < 0:
            raise ValueError("manifest_seed must be a non-negative integer")
        if (
            isinstance(manifest_size, bool)
            or not isinstance(manifest_size, int)
            or manifest_size < num_envs
        ):
            raise ValueError("manifest_size must be an integer at least as large as num_envs")

        try:
            torch = importlib.import_module("torch")
            warp = importlib.import_module("warp")
            from se3_wam.benchmark.api import Split
            from se3_wam.benchmark.config import load_task_config
            from se3_wam.benchmark.contracts import ObservationTrack
            from se3_wam.benchmark.gpu_native.factory import make_gpu_native_env
            from se3_wam.benchmark.gpu_native.p0_grasp_engine import (
                load_p0_grasp_artifacts,
            )
            from se3_wam.benchmark.gpu_native.tasks import GpuNativeConsumer
            from se3_wam.benchmark.p0_grasp_manifest import (
                make_p0_grasp_candidate_manifest,
            )
        except ImportError as exc:
            raise GpuNativeTensorBackendUnavailableError(
                "GPU tensor backend requires PyTorch, Warp, and the SE3-WAM GPU package"
            ) from exc
        if not torch.cuda.is_available():
            raise GpuNativeTensorBackendUnavailableError(
                "PyTorch CUDA is unavailable; CPU fallback is forbidden"
            )
        device = torch.device(f"cuda:{device_ordinal}")
        try:
            stream_from_torch = warp.stream_from_torch
            scoped_stream = warp.ScopedStream
        except AttributeError as exc:
            raise GpuNativeTensorBackendUnavailableError(
                "Warp runtime lacks PyTorch stream interoperability"
            ) from exc

        self._task_id = task_id
        self._num_envs = num_envs
        self._export_dir = export_dir
        self._expected_gpu_uuid = expected_gpu_uuid
        self._device_ordinal = device_ordinal
        self._device = device
        self._torch = torch
        self._warp = warp
        self._stream_from_torch = stream_from_torch
        self._scoped_stream = scoped_stream
        self._consumer = GpuNativeConsumer.RL
        try:
            self._split = Split(split)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported P0-Grasp manifest split: {split!r}") from exc
        self._manifest_seed = manifest_seed
        self._manifest_size = manifest_size
        manifest_rows = make_p0_grasp_candidate_manifest(
            split=self._split,
            attempts=manifest_size,
            manifest_seed=manifest_seed,
        )
        self._manifest_requests = tuple(row.request for row in manifest_rows)
        if len(self._manifest_requests) != manifest_size:
            raise GpuNativeTensorBackendUnavailableError(
                "P0-Grasp manifest generator returned the wrong number of requests"
            )
        self._manifest_sha256 = _manifest_sha256(self._manifest_requests)
        self._next_cohort_index = 0
        self._steps_since_reset: int | None = None
        task_config = load_task_config(task_id)
        try:
            self._cohort_horizon_steps = int(task_config["clock"]["horizon_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GpuNativeTensorBackendUnavailableError(
                "task config does not provide a positive cohort horizon"
            ) from exc
        if self._cohort_horizon_steps < 1:
            raise GpuNativeTensorBackendUnavailableError(
                "task config does not provide a positive cohort horizon"
            )
        self._env = make_gpu_native_env(
            task_id,
            consumer=self._consumer,
            batch_size=num_envs,
            observation_track=ObservationTrack.STATE,
            export_dir=export_dir,
            device_ordinal=device_ordinal,
            image_size=image_size,
            engine_kwargs={"expected_device_uuid": expected_gpu_uuid},
        )
        try:
            self._env.capabilities.require(
                "full_batch_reset",
                "device_tensor_step",
                "device_terminal_mask",
            )
        except Exception:
            self._env.close()
            raise
        provenance = self._env.provenance
        torch_uuid, torch_pci = _torch_device_identity(torch, device)
        nvml_uuid, nvml_pci, driver_version = _nvidia_smi_identity(expected_gpu_uuid)
        warp_uuid = getattr(provenance, "physical_device_uuid", None)
        warp_pci = getattr(provenance, "physical_device_pci_bus_id", None)
        warp_source = getattr(provenance, "physical_device_identity_source", None)
        if not all(isinstance(value, str) for value in (warp_uuid, warp_pci, warp_source)):
            self._env.close()
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM provenance lacks observed Warp UUID/PCI identity"
            )
        try:
            self._device_identity = GpuDeviceIdentityReceipt(
                expected_uuid=expected_gpu_uuid,
                cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                torch_uuid=torch_uuid,
                torch_pci_bus_id=torch_pci,
                warp_uuid=_canonical_gpu_uuid(warp_uuid),
                warp_pci_bus_id=_canonical_pci_bus_id(
                    warp_pci,
                    name="Warp PCI bus id",
                ),
                warp_identity_source=warp_source,
                nvml_uuid=nvml_uuid,
                nvml_pci_bus_id=nvml_pci,
                driver_version=driver_version,
            )
        except Exception:
            self._env.close()
            raise
        artifacts = load_p0_grasp_artifacts(export_dir)
        self._frozen_request = artifacts.reset_request
        if self._frozen_request.task_id != task_id:
            self._env.close()
            raise GpuNativeTensorBackendUnavailableError(
                "frozen export task_id does not match the requested task"
            )
        self._observation_track = ObservationTrack.STATE
        self._view_cache: dict[str, tuple[int, Any]] = {}
        self._stream_cache: dict[int, Any] = {}
        self._observation_shape: tuple[int, ...] | None = None
        self._last_transport_receipt: Mapping[str, Any] | None = None
        self._transport_checks = 0
        self._closed = False

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> Any:
        return self._device

    @property
    def cohort_horizon_steps(self) -> int:
        return self._cohort_horizon_steps

    @property
    def provenance(self) -> Any:
        return self._env.provenance

    @property
    def api_version(self) -> str:
        return self._env.contract.api_version

    @property
    def device_identity(self) -> GpuDeviceIdentityReceipt:
        return self._device_identity

    @property
    def last_transport_receipt(self) -> Mapping[str, Any] | None:
        return self._last_transport_receipt

    @property
    def transport_checks(self) -> int:
        return self._transport_checks

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def split(self) -> str:
        return self._split.value

    @property
    def manifest_seed(self) -> int:
        return self._manifest_seed

    @property
    def manifest_size(self) -> int:
        return self._manifest_size

    def _stream_scope(self) -> Any:
        torch_stream = self._torch.cuda.current_stream(self._device)
        raw_stream_pointer = getattr(torch_stream, "cuda_stream", None)
        if not isinstance(raw_stream_pointer, int) or raw_stream_pointer <= 0:
            return self._scoped_stream(self._stream_from_torch(torch_stream))
        stream_pointer = int(raw_stream_pointer)
        warp_stream = self._stream_cache.get(stream_pointer)
        if warp_stream is None:
            warp_stream = self._stream_from_torch(torch_stream)
            self._stream_cache[stream_pointer] = warp_stream
        return self._scoped_stream(warp_stream)

    def _view(
        self,
        value: Any,
        name: str,
        *,
        expected_shape: tuple[int, ...] | None = None,
        expected_dtype: Any | None = None,
        expected_pointer: int | None = None,
    ) -> Any:
        pointer = getattr(value, "ptr", None)
        if expected_pointer is not None and pointer != expected_pointer:
            raise GpuNativeTensorBackendUnavailableError(
                f"{name} engine pointer differs from its transport receipt"
            )
        cached = self._view_cache.get(name)
        if isinstance(pointer, int) and cached is not None and cached[0] == pointer:
            tensor = cached[1]
            if (
                int(tensor.data_ptr()) == pointer
                and tensor.device == self._device
                and tensor.is_contiguous()
                and int(tensor.shape[0]) == self._num_envs
                and (expected_shape is None or tuple(tensor.shape) == expected_shape)
                and (expected_dtype is None or tensor.dtype == expected_dtype)
            ):
                return tensor
            self._view_cache.pop(name, None)
        tensor = _zero_copy_torch_view(
            warp_module=self._warp,
            value=value,
            name=name,
            batch_size=self._num_envs,
            expected_device=self._device,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            expected_pointer=expected_pointer,
        )
        self._view_cache[name] = (int(tensor.data_ptr()), tensor)
        return tensor

    def next_requests(self) -> tuple[Any, ...]:
        """Preview the next deterministic cohort without consuming its cursor."""

        requests = []
        first_ordinal = self._next_cohort_index * self._num_envs
        for lane in range(self._num_envs):
            ordinal = first_ordinal + lane
            manifest_index = ordinal % self._manifest_size
            cycle = ordinal // self._manifest_size
            base = self._manifest_requests[manifest_index]
            requests.append(
                replace(
                    base,
                    episode_id=f"{base.episode_id}-cycle{cycle:08d}",
                    observation_track=self._observation_track,
                )
            )
        return tuple(requests)

    def reset(self, requests: Any | None = None) -> GpuNativeTensorReset:
        if self._closed:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor backend is closed")
        if requests is not None:
            raise ValueError("GPU tensor R0 reset requests are owned by the frozen manifest cursor")
        if self._steps_since_reset not in {None, self._cohort_horizon_steps}:
            raise GpuNativeTensorBackendUnavailableError(
                "full-cohort reset is legal only before the first episode or at fixed horizon"
            )
        requests = self.next_requests()
        if len(requests) != self._num_envs or any(request is None for request in requests):
            raise ValueError("GPU tensor reset requires one request for every cohort lane")
        first_ordinal = self._next_cohort_index * self._num_envs
        ordinals = tuple(first_ordinal + lane for lane in range(self._num_envs))
        with self._stream_scope():
            result = self._env.reset(requests)
        observation = self._view(
            result.observation,
            "observation",
            expected_dtype=self._torch.float32,
        )
        if len(observation.shape) != 2:
            raise GpuNativeTensorBackendUnavailableError(
                "state observation must have exact (B, D) layout"
            )
        self._observation_shape = tuple(observation.shape)
        self._next_cohort_index += 1
        self._steps_since_reset = 0
        return GpuNativeTensorReset(
            observation=observation,
            generation=int(result.state.generation),
            episode_ids=tuple(request.episode_id for request in requests),
            seeds=tuple(request.seed for request in requests),
            manifest_ordinals=ordinals,
            manifest_sha256=self._manifest_sha256,
        )

    def step(self, action: Any) -> GpuNativeTensorStep:
        if self._closed:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor backend is closed")
        if self._steps_since_reset is None:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor step requires a reset cohort")
        if self._steps_since_reset >= self._cohort_horizon_steps:
            raise GpuNativeTensorBackendUnavailableError(
                "GPU tensor cohort reached fixed horizon and must reset before another step"
            )
        if action.shape != (self._num_envs, 7):
            raise ValueError(f"GPU tensor action must have shape ({self._num_envs}, 7)")
        if action.dtype != self._torch.float32:
            raise ValueError("GPU tensor action must use torch.float32")
        if action.device != self._device or not action.is_contiguous():
            raise ValueError("GPU tensor action must be contiguous on the backend CUDA device")
        torch_stream = self._torch.cuda.current_stream(self._device)
        torch_stream_pointer = getattr(torch_stream, "cuda_stream", None)
        if (
            isinstance(torch_stream_pointer, bool)
            or not isinstance(torch_stream_pointer, int)
            or torch_stream_pointer < 0
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "PyTorch current stream lacks a CUDA stream handle"
            )
        action_pointer = int(action.data_ptr())
        with self._stream_scope():
            result = self._env.step_device(action)
        transport = getattr(result, "transport", None)
        output_ptrs = getattr(transport, "output_ptrs", None)
        if (
            getattr(transport, "action_input_ptr", None) != action_pointer
            or getattr(transport, "action_engine_ptr", None) != action_pointer
            or getattr(transport, "stream_ptr", None) != torch_stream_pointer
            or not isinstance(output_ptrs, Mapping)
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM action pointer or CUDA stream transport receipt is invalid"
            )
        schemas = {
            "observation": (self._observation_shape, self._torch.float32),
            "reward": ((self._num_envs,), self._torch.float32),
            "terminated": ((self._num_envs,), self._torch.int32),
            "truncated": ((self._num_envs,), self._torch.int32),
            "success": ((self._num_envs,), self._torch.int32),
            "event_mask": ((self._num_envs,), self._torch.int32),
            "terminal_reason": ((self._num_envs,), self._torch.int32),
            "physics_step": ((self._num_envs,), self._torch.int64),
        }
        if self._observation_shape is None or set(output_ptrs) != set(schemas):
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM output pointer receipt has the wrong field set"
            )
        values = {}
        for name, (shape, dtype) in schemas.items():
            assert shape is not None
            values[name] = self._view(
                getattr(result, name),
                name,
                expected_shape=shape,
                expected_dtype=dtype,
                expected_pointer=output_ptrs[name],
            )
        self._last_transport_receipt = MappingProxyType(
            {
                "action_input_ptr": action_pointer,
                "action_engine_ptr": transport.action_engine_ptr,
                "torch_stream_ptr": torch_stream_pointer,
                "warp_stream_ptr": transport.stream_ptr,
                "output_ptrs": MappingProxyType(dict(output_ptrs)),
            }
        )
        self._transport_checks += 1
        self._steps_since_reset += 1
        return GpuNativeTensorStep(
            **values,
            generation=int(result.state.generation),
        )

    def manifest_state_dict(self) -> Mapping[str, Any]:
        """Return a reset-boundary cursor checkpoint with frozen manifest identity."""

        if self._steps_since_reset not in {None, self._cohort_horizon_steps}:
            raise GpuNativeTensorBackendUnavailableError(
                "manifest cursor checkpoint is legal only at a reset boundary"
            )
        return MappingProxyType(
            {
                "schema_version": _RESET_CURSOR_SCHEMA_VERSION,
                "task_id": self._task_id,
                "num_envs": self._num_envs,
                "split": self._split.value,
                "manifest_seed": self._manifest_seed,
                "manifest_size": self._manifest_size,
                "manifest_sha256": self._manifest_sha256,
                "next_cohort_index": self._next_cohort_index,
            }
        )

    def load_manifest_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a cursor only into a fresh backend before its first reset."""

        if self._steps_since_reset is not None or self._next_cohort_index != 0:
            raise GpuNativeTensorBackendUnavailableError(
                "manifest cursor restore requires a fresh backend before first reset"
            )
        expected = {
            "schema_version": _RESET_CURSOR_SCHEMA_VERSION,
            "task_id": self._task_id,
            "num_envs": self._num_envs,
            "split": self._split.value,
            "manifest_seed": self._manifest_seed,
            "manifest_size": self._manifest_size,
            "manifest_sha256": self._manifest_sha256,
        }
        if not isinstance(state, Mapping) or set(state) != {*expected, "next_cohort_index"}:
            raise GpuNativeTensorBackendUnavailableError(
                "manifest cursor checkpoint schema does not match"
            )
        mismatches = {
            name: (state.get(name), value)
            for name, value in expected.items()
            if state.get(name) != value
        }
        cursor = state.get("next_cohort_index")
        if (
            mismatches
            or isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < 0
        ):
            raise GpuNativeTensorBackendUnavailableError(
                f"manifest cursor checkpoint identity mismatch: {mismatches}"
            )
        self._next_cohort_index = cursor

    def close(self) -> None:
        if self._closed:
            return
        self._env.close()
        self._closed = True

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError("GpuNativeTensorBackendEnv is device-resident and not picklable")


__all__ = [
    "GpuDeviceIdentityReceipt",
    "GpuNativeTensorBackendEnv",
    "GpuNativeTensorBackendUnavailableError",
    "GpuNativeTensorReset",
    "GpuNativeTensorStep",
]
