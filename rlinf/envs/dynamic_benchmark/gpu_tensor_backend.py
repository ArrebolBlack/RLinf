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

import importlib
import os
from dataclasses import dataclass, replace
from typing import Any, Mapping


class GpuNativeTensorBackendUnavailableError(RuntimeError):
    """Raised instead of copying, materializing, or falling back to CPU."""


def _canonical_gpu_uuid(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("GPU-")
        or len(value) != 40
        or value.strip() != value
    ):
        raise ValueError("expected_gpu_uuid must be a canonical GPU UUID")
    return value


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
    return tensor


@dataclass(frozen=True)
class GpuNativeTensorReset:
    observation: Any
    generation: int


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

        try:
            torch = importlib.import_module("torch")
            warp = importlib.import_module("warp")
            from se3_wam.benchmark.config import load_task_config
            from se3_wam.benchmark.contracts import ObservationTrack
            from se3_wam.benchmark.gpu_native.factory import make_gpu_native_env
            from se3_wam.benchmark.gpu_native.p0_grasp_engine import (
                load_p0_grasp_artifacts,
            )
            from se3_wam.benchmark.gpu_native.tasks import GpuNativeConsumer
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
            self._env.capabilities.require("full_batch_reset", "device_tensor_step")
        except Exception:
            self._env.close()
            raise
        provenance = self._env.provenance
        if provenance.physical_device_uuid != expected_gpu_uuid:
            self._env.close()
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM provenance did not preserve the physical GPU UUID"
            )
        artifacts = load_p0_grasp_artifacts(export_dir)
        self._frozen_request = artifacts.reset_request
        if self._frozen_request.task_id != task_id:
            self._env.close()
            raise GpuNativeTensorBackendUnavailableError(
                "frozen export task_id does not match the requested task"
            )
        self._observation_track = ObservationTrack.STATE
        self._episode_counter = 0
        self._view_cache: dict[str, tuple[int, Any]] = {}
        self._stream_cache: dict[int, Any] = {}
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

    def _view(self, value: Any, name: str) -> Any:
        pointer = getattr(value, "ptr", None)
        cached = self._view_cache.get(name)
        if isinstance(pointer, int) and cached is not None and cached[0] == pointer:
            tensor = cached[1]
            if (
                int(tensor.data_ptr()) == pointer
                and tensor.device == self._device
                and tensor.is_contiguous()
                and int(tensor.shape[0]) == self._num_envs
            ):
                return tensor
            self._view_cache.pop(name, None)
        tensor = _zero_copy_torch_view(
            warp_module=self._warp,
            value=value,
            name=name,
            batch_size=self._num_envs,
            expected_device=self._device,
        )
        self._view_cache[name] = (int(tensor.data_ptr()), tensor)
        return tensor

    def next_requests(self) -> tuple[Any, ...]:
        requests = []
        for _lane in range(self._num_envs):
            episode_id = f"{self._task_id}-gpu-{self._episode_counter:012d}"
            self._episode_counter += 1
            requests.append(
                replace(
                    self._frozen_request,
                    episode_id=episode_id,
                    observation_track=self._observation_track,
                )
            )
        return tuple(requests)

    def reset(self, requests: Any | None = None) -> GpuNativeTensorReset:
        if self._closed:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor backend is closed")
        requests = self.next_requests() if requests is None else tuple(requests)
        if len(requests) != self._num_envs or any(request is None for request in requests):
            raise ValueError("GPU tensor reset requires one request for every cohort lane")
        with self._stream_scope():
            result = self._env.reset(requests)
        return GpuNativeTensorReset(
            observation=self._view(result.observation, "observation"),
            generation=int(result.state.generation),
        )

    def step(self, action: Any) -> GpuNativeTensorStep:
        if self._closed:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor backend is closed")
        if action.shape != (self._num_envs, 7):
            raise ValueError(f"GPU tensor action must have shape ({self._num_envs}, 7)")
        if action.dtype != self._torch.float32:
            raise ValueError("GPU tensor action must use torch.float32")
        if action.device != self._device or not action.is_contiguous():
            raise ValueError("GPU tensor action must be contiguous on the backend CUDA device")
        with self._stream_scope():
            result = self._env.step_device(action)
        values = {
            name: self._view(getattr(result, name), name)
            for name in (
                "observation",
                "reward",
                "terminated",
                "truncated",
                "success",
                "event_mask",
                "terminal_reason",
                "physics_step",
            )
        }
        return GpuNativeTensorStep(
            **values,
            generation=int(result.state.generation),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._env.close()
        self._closed = True

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError("GpuNativeTensorBackendEnv is device-resident and not picklable")


__all__ = [
    "GpuNativeTensorBackendEnv",
    "GpuNativeTensorBackendUnavailableError",
    "GpuNativeTensorReset",
    "GpuNativeTensorStep",
]
