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
an explicit control plane and are never reached by :meth:`step_device`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping


class GpuNativeTensorBackendUnavailableError(RuntimeError):
    """Raised instead of copying, materializing, or falling back to CPU."""


_RESET_CURSOR_SCHEMA_VERSION = "rlinf-gpu-native-r0-cursor-v0.1"
_GPU_NATIVE_API_VERSION = "gpu-native-api-v0.2"
_GPU_NATIVE_BACKEND_ID = "mjwarp_gpu_v1"
_CAPABILITY_NAMES = (
    "physics",
    "robot_control",
    "state_observation",
    "rgb",
    "depth",
    "segmentation",
    "full_batch_reset",
    "masked_reset",
    "device_tensor_step",
    "device_terminal_mask",
    "snapshot",
    "audit_materialization",
)
_P0_STATE_CONTRACT_CAPABILITIES = MappingProxyType(
    {
        "physics": True,
        "robot_control": True,
        "state_observation": True,
        "rgb": False,
        "depth": False,
        "segmentation": False,
        "full_batch_reset": True,
        "masked_reset": False,
        "device_tensor_step": False,
        "device_terminal_mask": False,
        "snapshot": True,
        "audit_materialization": True,
    }
)
_P0_STATE_ENV_CAPABILITIES = MappingProxyType(
    {
        **_P0_STATE_CONTRACT_CAPABILITIES,
        "device_tensor_step": True,
        "device_terminal_mask": True,
    }
)
_EXPORT_DIGEST_ATTRIBUTES = MappingProxyType(
    {
        "request_sha256": "request_identity_sha256",
        "bundle_sha256": "bundle_sha256",
        "model_sha256": "model_sha256",
        "config_sha256": "config_sha256",
        "manifest_sha256": "manifest_sha256",
    }
)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _canonical_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GpuNativeTensorBackendUnavailableError(
            f"{name} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _full_git_object(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase full Git object id")
    return value


def _freeze_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(name): _freeze_identity(item) for name, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_identity(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_identity(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    """Return a defensive plain JSON tree from an immutable or caller-owned tree."""

    if isinstance(value, Mapping):
        return {str(name): _thaw_json_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json_value(item) for item in value]
    return value


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
        "factors": _thaw_json_value(request.factors),
        "api_version": request.api_version,
    }


def _manifest_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_sha256(requests: tuple[Any, ...]) -> str:
    return _manifest_payload_sha256(
        [_manifest_request_payload(request) for request in requests]
    )


def _freeze_manifest_request(request: Any, payload: Mapping[str, Any]) -> Any:
    """Rebuild one request from its canonical factor snapshot."""

    try:
        frozen = replace(
            request,
            factors=_freeze_identity(_thaw_json_value(payload["factors"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GpuNativeTensorBackendUnavailableError(
            "manifest entries must be dataclass ResetRequest values"
        ) from exc
    if _manifest_request_payload(frozen) != dict(payload):
        raise GpuNativeTensorBackendUnavailableError(
            "manifest request changed while its canonical snapshot was frozen"
        )
    return frozen


def _defensive_manifest_request(request: Any) -> Any:
    """Return a detached request whose nested factors are caller-owned."""

    return replace(request, factors=_thaw_json_value(request.factors))


def _exact_capabilities(value: Any, *, context: str) -> Mapping[str, bool]:
    declared = getattr(value, "__dataclass_fields__", None)
    if not isinstance(declared, Mapping) or tuple(declared) != _CAPABILITY_NAMES:
        raise GpuNativeTensorBackendUnavailableError(
            f"{context} capability vocabulary differs from the exact v0.2 ABI"
        )
    missing = [name for name in _CAPABILITY_NAMES if not hasattr(value, name)]
    if missing:
        raise GpuNativeTensorBackendUnavailableError(
            f"{context} is missing v0.2 capabilities: {missing}"
        )
    rendered = {name: getattr(value, name) for name in _CAPABILITY_NAMES}
    if any(type(item) is not bool for item in rendered.values()):
        raise GpuNativeTensorBackendUnavailableError(
            f"{context} capabilities must be exact booleans"
        )
    available = getattr(value, "available", None)
    if available is None or frozenset(available) != frozenset(
        name for name, enabled in rendered.items() if enabled
    ):
        raise GpuNativeTensorBackendUnavailableError(
            f"{context} capability availability disagrees with its fields"
        )
    return MappingProxyType(rendered)


def _validate_public_contract(
    env: Any,
    *,
    task_id: str,
    num_envs: int,
) -> tuple[Mapping[str, bool], Mapping[str, bool]]:
    contract = getattr(env, "contract", None)
    if contract is None:
        raise GpuNativeTensorBackendUnavailableError(
            "SE3-WAM GPU environment does not expose its public contract"
        )
    expected = {
        "backend_id": _GPU_NATIVE_BACKEND_ID,
        "api_version": _GPU_NATIVE_API_VERSION,
        "batch_size": num_envs,
        "task_id": task_id,
        "consumer": "rl",
        "observation_track": "state",
        "action_mode": "E7",
        "physics_hz": 500,
        "control_hz": 20,
        "sensor_hz": 20,
    }
    mismatches = {
        name: (_enum_value(getattr(contract, name, None)), required)
        for name, required in expected.items()
        if _enum_value(getattr(contract, name, None)) != required
    }
    if mismatches:
        raise GpuNativeTensorBackendUnavailableError(
            f"SE3-WAM GPU-native v0.2 public contract mismatch: {mismatches}"
        )
    if getattr(env, "backend_id", None) != _GPU_NATIVE_BACKEND_ID:
        raise GpuNativeTensorBackendUnavailableError(
            "SE3-WAM environment backend_id differs from the v0.2 contract"
        )
    contract_capabilities = _exact_capabilities(
        getattr(contract, "capabilities", None),
        context="SE3-WAM contract",
    )
    env_capabilities = _exact_capabilities(
        getattr(env, "capabilities", None),
        context="SE3-WAM environment",
    )
    if contract_capabilities != _P0_STATE_CONTRACT_CAPABILITIES:
        raise GpuNativeTensorBackendUnavailableError(
            "SE3-WAM P0 STATE contract capability values differ from clean v0.2"
        )
    if env_capabilities != _P0_STATE_ENV_CAPABILITIES:
        raise GpuNativeTensorBackendUnavailableError(
            "SE3-WAM P0 STATE engine capability values differ from clean v0.2"
        )
    return contract_capabilities, env_capabilities


def _artifact_identity(artifacts: Any) -> Mapping[str, str]:
    values = {}
    for public_name, attribute in _EXPORT_DIGEST_ATTRIBUTES.items():
        values[public_name] = _canonical_sha256(
            getattr(artifacts, attribute, None),
            name=f"active export {public_name}",
        )
    return MappingProxyType(values)


_GPU_UUID_PATTERN = re.compile(
    r"^GPU-([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-"
    r"([0-9a-fA-F]{4})-([0-9a-fA-F]{12})$"
)
_UNPREFIXED_GPU_UUID_PATTERN = re.compile(
    r"^([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-"
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


def _canonical_torch_gpu_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("PyTorch GPU UUID is unavailable")
    stripped = value.strip()
    if stripped != value:
        raise ValueError("PyTorch GPU UUID is not canonical")
    if _UNPREFIXED_GPU_UUID_PATTERN.fullmatch(stripped) is not None:
        stripped = f"GPU-{stripped}"
    return _canonical_gpu_uuid(stripped)


def _torch_device_identity(torch: Any, device: Any) -> tuple[str, str]:
    properties = torch.cuda.get_device_properties(device)
    try:
        # PyTorch 2.12 exposes ``_CUuuid`` as the RFC-4122 payload without
        # NVIDIA's ``GPU-`` namespace prefix.  Only that exact unprefixed form
        # is accepted; arbitrary strings are never promoted into UUIDs.
        uuid = _canonical_torch_gpu_uuid(str(properties.uuid))
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


def _warp_device_identity(
    warp: Any,
    *,
    device_ordinal: int,
) -> tuple[str, str, str, int]:
    try:
        init = getattr(warp, "init", None)
        if callable(init):
            init()
        device = warp.get_device(f"cuda:{device_ordinal}")
        is_cuda = getattr(device, "is_cuda", None)
        if callable(is_cuda):
            is_cuda = is_cuda()
        if is_cuda is not True:
            raise ValueError("Warp device is not CUDA")
        ordinal = getattr(device, "ordinal", None)
        if callable(ordinal):
            ordinal = ordinal()
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal != device_ordinal
        ):
            raise ValueError("Warp device ordinal is not the selected visible ordinal")
        alias = getattr(device, "alias", None)
        if callable(alias):
            alias = alias()
        if not isinstance(alias, str) or alias != f"cuda:{device_ordinal}":
            raise ValueError("Warp device alias is not the selected visible ordinal")
        uuid = getattr(device, "uuid", None)
        if callable(uuid):
            uuid = uuid()
        pci_bus_id = getattr(device, "pci_bus_id", None)
        if callable(pci_bus_id):
            pci_bus_id = pci_bus_id()
        return (
            _canonical_gpu_uuid(uuid),
            _canonical_pci_bus_id(pci_bus_id, name="Warp PCI bus id"),
            alias,
            ordinal,
        )
    except GpuNativeTensorBackendUnavailableError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise GpuNativeTensorBackendUnavailableError(
            "Warp CUDA device lacks trusted UUID/PCI/ordinal/alias identity"
        ) from exc


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
    logical_device_ordinal: int
    torch_uuid: str
    torch_pci_bus_id: str
    torch_device_ordinal: int
    warp_uuid: str
    warp_pci_bus_id: str
    warp_device_ordinal: int
    warp_device_alias: str
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
        ordinals = (
            self.logical_device_ordinal,
            self.torch_device_ordinal,
            self.warp_device_ordinal,
        )
        if any(type(value) is not int for value in ordinals) or ordinals != (0, 0, 0):
            raise GpuNativeTensorBackendUnavailableError(
                "CUDA-visible/Torch/Warp logical ordinals must be exactly zero"
            )
        if self.warp_device_alias != "cuda:0":
            raise GpuNativeTensorBackendUnavailableError(
                "Warp device alias must be exactly cuda:0"
            )
        if self.warp_identity_source != "warp_cuda_driver":
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM provenance lacks CUDA-driver Warp identity"
            )
        if (
            not isinstance(self.driver_version, str)
            or not self.driver_version
            or self.driver_version.strip() != self.driver_version
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "NVIDIA driver version is unavailable"
            )


def _observe_device_identity(
    *,
    torch: Any,
    warp: Any,
    device: Any,
    expected_gpu_uuid: str,
    device_ordinal: int,
    warp_identity_source: str = "warp_cuda_driver",
) -> GpuDeviceIdentityReceipt:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_gpu_uuid:
        raise GpuNativeTensorBackendUnavailableError(
            "CUDA_VISIBLE_DEVICES changed after singleton UUID admission"
        )
    try:
        count = torch.cuda.device_count()
        current = torch.cuda.current_device()
    except (AttributeError, TypeError, ValueError) as exc:
        raise GpuNativeTensorBackendUnavailableError(
            "PyTorch CUDA visible ordinal APIs are unavailable"
        ) from exc
    if type(count) is not int or count != 1:
        raise GpuNativeTensorBackendUnavailableError(
            "singleton UUID visibility must expose exactly one PyTorch CUDA device"
        )
    if type(current) is not int or current != device_ordinal:
        raise GpuNativeTensorBackendUnavailableError(
            "PyTorch current CUDA ordinal differs from the admitted logical ordinal"
        )
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_type != "cuda" or device_index != device_ordinal:
        raise GpuNativeTensorBackendUnavailableError(
            "PyTorch device alias differs from the admitted logical ordinal"
        )
    torch_uuid, torch_pci = _torch_device_identity(torch, device)
    warp_uuid, warp_pci, warp_alias, warp_ordinal = _warp_device_identity(
        warp,
        device_ordinal=device_ordinal,
    )
    nvml_uuid, nvml_pci, driver_version = _nvidia_smi_identity(expected_gpu_uuid)
    return GpuDeviceIdentityReceipt(
        expected_uuid=expected_gpu_uuid,
        cuda_visible_devices=visible,
        logical_device_ordinal=device_ordinal,
        torch_uuid=torch_uuid,
        torch_pci_bus_id=torch_pci,
        torch_device_ordinal=current,
        warp_uuid=warp_uuid,
        warp_pci_bus_id=warp_pci,
        warp_device_ordinal=warp_ordinal,
        warp_device_alias=warp_alias,
        warp_identity_source=warp_identity_source,
        nvml_uuid=nvml_uuid,
        nvml_pci_bus_id=nvml_pci,
        driver_version=driver_version,
    )


def _require_single_uuid_visibility(
    expected_gpu_uuid: str, device_ordinal: int
) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    entries = (
        () if visible is None else tuple(item.strip() for item in visible.split(","))
    )
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
    if isinstance(pointer, bool) or not isinstance(pointer, int) or pointer <= 0:
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
    tensor_pointer = tensor.data_ptr()
    if (
        isinstance(tensor_pointer, bool)
        or not isinstance(tensor_pointer, int)
        or tensor_pointer != pointer
    ):
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
    episode_ids: tuple[str, ...]

    @property
    def done(self) -> Any:
        """Return a device-local logical mask; no host synchronization occurs."""

        return (self.terminated != 0) | (self.truncated != 0)


@dataclass(frozen=True)
class GpuNativeTensorTerminalRow:
    """One explicitly materialized terminal lane from the control plane."""

    lane: int
    episode_id: str
    task_id: str
    terminated: bool
    truncated: bool
    success: bool
    termination_reason: str
    completion: float
    task_quality: Any | None
    observation: Any
    events: tuple[Any, ...]
    physics_step: int
    control_step: int
    policy_step: int

    @property
    def safety(self) -> None:
        """SE3-WAM v0.2 has no canonical scalar safety terminal field."""

        return None


class GpuNativeTensorBackendEnv:
    """Own one full-reset cohort and expose its device-only RL data plane."""

    def __init__(
        self,
        *,
        task_id: str,
        num_envs: int,
        export_dir: str,
        expected_gpu_uuid: str,
        expected_se3_source_commit: str,
        expected_se3_source_tree: str,
        device_ordinal: int = 0,
        image_size: int = 64,
        render_visual: bool = False,
        split: str = "train",
        manifest_seed: int = 20261050,
        manifest_size: int | None = None,
        manifest_requests: tuple[Any, ...] | None = None,
        manifest_sha256: str | None = None,
        task_quality_schema_version: str | None = None,
        task_quality_evaluator_backend_id: str | None = None,
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
        expected_se3_source_commit = _full_git_object(
            expected_se3_source_commit,
            name="expected_se3_source_commit",
        )
        expected_se3_source_tree = _full_git_object(
            expected_se3_source_tree,
            name="expected_se3_source_tree",
        )
        _require_single_uuid_visibility(expected_gpu_uuid, device_ordinal)
        if task_id != "p0_grasp":
            raise ValueError("GPU tensor R0 admission is currently limited to p0_grasp")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size < 1
        ):
            raise ValueError("image_size must be a positive integer")
        if not isinstance(render_visual, bool):
            raise ValueError("render_visual must be a bool")
        if (
            isinstance(manifest_seed, bool)
            or not isinstance(manifest_seed, int)
            or manifest_seed < 0
        ):
            raise ValueError("manifest_seed must be a non-negative integer")
        if (task_quality_schema_version is None) != (
            task_quality_evaluator_backend_id is None
        ):
            raise ValueError(
                "task-quality schema and evaluator backend identity must be supplied together"
            )
        for name, value in (
            ("task_quality_schema_version", task_quality_schema_version),
            ("task_quality_evaluator_backend_id", task_quality_evaluator_backend_id),
        ):
            if value is not None and (
                not isinstance(value, str) or not value or value.strip() != value
            ):
                raise ValueError(f"{name} must be a non-empty trimmed string or None")

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
        self._expected_se3_source_commit = expected_se3_source_commit
        self._expected_se3_source_tree = expected_se3_source_tree
        self._device_ordinal = device_ordinal
        self._device = device
        self._render_visual = render_visual
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
        if manifest_requests is None:
            effective_manifest_size = 4096 if manifest_size is None else manifest_size
            if (
                isinstance(effective_manifest_size, bool)
                or not isinstance(effective_manifest_size, int)
                or effective_manifest_size < num_envs
            ):
                raise ValueError(
                    "manifest_size must be an integer at least as large as num_envs"
                )
            manifest_rows = make_p0_grasp_candidate_manifest(
                split=self._split,
                attempts=effective_manifest_size,
                manifest_seed=manifest_seed,
            )
            requests = tuple(
                replace(
                    row.request,
                    observation_track=ObservationTrack.STATE,
                )
                for row in manifest_rows
            )
            if len(requests) != effective_manifest_size:
                raise GpuNativeTensorBackendUnavailableError(
                    "P0-Grasp manifest generator returned the wrong number of requests"
                )
            manifest_origin = "generated"
        else:
            if not isinstance(manifest_requests, tuple):
                raise TypeError("manifest_requests must be an immutable tuple")
            requests = manifest_requests
            effective_manifest_size = len(requests)
            if effective_manifest_size < num_envs:
                raise ValueError(
                    "caller manifest must contain at least one full cohort"
                )
            if manifest_size is not None and manifest_size != effective_manifest_size:
                raise ValueError("manifest_size conflicts with caller manifest length")
            if manifest_sha256 is None:
                raise ValueError(
                    "caller manifest requires an exact manifest_sha256 pin"
                )
            manifest_origin = "caller"
        for ordinal, request in enumerate(requests):
            if (
                getattr(request, "task_id", None) != task_id
                or _enum_value(getattr(request, "split", None)) != self._split.value
                or _enum_value(getattr(request, "action_mode", None)) != "E7"
                or _enum_value(getattr(request, "observation_track", None)) != "state"
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    f"manifest request {ordinal} differs from the P0 STATE/E7 contract"
                )
        manifest_payloads = tuple(
            _manifest_request_payload(request) for request in requests
        )
        observed_manifest_sha256 = _manifest_payload_sha256(manifest_payloads)
        if manifest_sha256 is not None:
            expected_manifest_sha256 = _canonical_sha256(
                manifest_sha256,
                name="manifest_sha256",
            )
            if observed_manifest_sha256 != expected_manifest_sha256:
                raise GpuNativeTensorBackendUnavailableError(
                    "caller-pinned reset manifest SHA-256 mismatch"
                )
        frozen_manifest_requests = tuple(
            _freeze_manifest_request(request, payload)
            for request, payload in zip(requests, manifest_payloads, strict=True)
        )
        if _manifest_sha256(frozen_manifest_requests) != observed_manifest_sha256:
            raise GpuNativeTensorBackendUnavailableError(
                "frozen reset manifest differs from its canonical snapshot"
            )
        self._manifest_size = effective_manifest_size
        self._manifest_requests = frozen_manifest_requests
        self._manifest_sha256 = observed_manifest_sha256
        self._manifest_origin = manifest_origin
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
        device_identity_start = _observe_device_identity(
            torch=torch,
            warp=warp,
            device=device,
            expected_gpu_uuid=expected_gpu_uuid,
            device_ordinal=device_ordinal,
        )
        artifacts_start = load_p0_grasp_artifacts(export_dir)
        export_identity_start = _artifact_identity(artifacts_start)
        frozen_request = artifacts_start.reset_request
        if (
            getattr(frozen_request, "task_id", None) != task_id
            or _enum_value(getattr(frozen_request, "action_mode", None)) != "E7"
            or _enum_value(getattr(frozen_request, "observation_track", None))
            != "state"
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "frozen export request differs from the exact P0 STATE/E7 contract"
            )
        engine_kwargs = {
            "expected_device_uuid": expected_gpu_uuid,
            "expected_model_sha256": export_identity_start["model_sha256"],
            "expected_source_commit": expected_se3_source_commit,
            "expected_source_tree": expected_se3_source_tree,
        }
        if render_visual:
            engine_kwargs["enable_visual"] = True
        env = make_gpu_native_env(
            task_id,
            consumer=self._consumer,
            batch_size=num_envs,
            observation_track=ObservationTrack.STATE,
            export_dir=export_dir,
            device_ordinal=device_ordinal,
            image_size=image_size,
            engine_kwargs=engine_kwargs,
        )
        try:
            contract_capabilities, env_capabilities = _validate_public_contract(
                env,
                task_id=task_id,
                num_envs=num_envs,
            )
            provenance = env.provenance
            if (
                getattr(provenance, "backend_id", None) != _GPU_NATIVE_BACKEND_ID
                or getattr(provenance, "device_platform", None) != "cuda"
                or getattr(provenance, "precision", None) != "float32"
                or getattr(provenance, "device_ordinal", None) != device_ordinal
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "SE3-WAM provenance backend/device contract differs from clean v0.2"
                )
            if (
                getattr(provenance, "git_commit", None) != expected_se3_source_commit
                or getattr(provenance, "git_tree", None) != expected_se3_source_tree
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "loaded SE3-WAM source commit/tree differs from the pinned clean API"
                )
            implementation_version = getattr(provenance, "implementation_version", None)
            runtime_versions = getattr(provenance, "runtime_versions", None)
            if (
                not isinstance(implementation_version, str)
                or not implementation_version
                or implementation_version.strip() != implementation_version
                or not isinstance(runtime_versions, Mapping)
                or not runtime_versions
                or any(
                    not isinstance(name, str)
                    or not name
                    or name.strip() != name
                    or not isinstance(version, str)
                    or not version
                    or version.strip() != version
                    for name, version in runtime_versions.items()
                )
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "SE3-WAM provenance lacks exact implementation/runtime identity"
                )
            try:
                warp_uuid = _canonical_gpu_uuid(
                    getattr(provenance, "physical_device_uuid", None)
                )
            except ValueError as exc:
                raise GpuNativeTensorBackendUnavailableError(
                    "SE3-WAM provenance lacks a canonical physical GPU UUID"
                ) from exc
            warp_pci = _canonical_pci_bus_id(
                getattr(provenance, "physical_device_pci_bus_id", None),
                name="SE3-WAM provenance PCI bus id",
            )
            warp_source = getattr(provenance, "physical_device_identity_source", None)
            if (
                warp_uuid != device_identity_start.warp_uuid
                or warp_pci != device_identity_start.warp_pci_bus_id
                or warp_source != "warp_cuda_driver"
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "SE3-WAM provenance disagrees with observed Warp UUID/PCI identity"
                )
            artifacts_end = load_p0_grasp_artifacts(export_dir)
            export_identity_end = _artifact_identity(artifacts_end)
            if (
                export_identity_end != export_identity_start
                or _manifest_request_payload(artifacts_end.reset_request)
                != _manifest_request_payload(frozen_request)
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "active export identity changed during backend construction"
                )
            for name, expected in export_identity_start.items():
                if getattr(provenance, name, None) != expected:
                    raise GpuNativeTensorBackendUnavailableError(
                        f"SE3-WAM provenance differs from active export {name}"
                    )
            if task_quality_schema_version is not None:
                env.enable_task_quality(
                    evaluator_backend_id=task_quality_evaluator_backend_id,
                    schema_version=task_quality_schema_version,
                )
            device_identity_end = _observe_device_identity(
                torch=torch,
                warp=warp,
                device=device,
                expected_gpu_uuid=expected_gpu_uuid,
                device_ordinal=device_ordinal,
                warp_identity_source=warp_source,
            )
            if device_identity_end != device_identity_start:
                raise GpuNativeTensorBackendUnavailableError(
                    "CUDA physical/logical identity changed during backend construction"
                )
        except BaseException:
            try:
                env.close()
            except Exception:
                pass
            raise

        self._env = env
        self._load_artifacts = load_p0_grasp_artifacts
        self._frozen_request = frozen_request
        self._active_export_digests = export_identity_start
        self._task_quality_schema_version = task_quality_schema_version
        self._task_quality_evaluator_backend_id = task_quality_evaluator_backend_id
        self._device_identity_start = device_identity_start
        self._device_identity_end = device_identity_end
        self._final_device_identity: GpuDeviceIdentityReceipt | None = None
        self._observation_track = ObservationTrack.STATE
        self._view_cache: dict[str, tuple[int, Any]] = {}
        self._stream_cache: dict[int, Any] = {}
        self._observation_shape: tuple[int, ...] | None = None
        self._active_episode_ids: tuple[str, ...] | None = None
        self._active_manifest_ordinals: tuple[int, ...] | None = None
        self._active_generation: int | None = None
        self._last_transport_receipt: Mapping[str, Any] | None = None
        self._transport_checks = 0
        self._teacher_audit_materializations = 0
        self._contract_capabilities = contract_capabilities
        self._env_capabilities = env_capabilities
        task_quality_identity = (
            None
            if task_quality_schema_version is None
            else {
                "schema_version": task_quality_schema_version,
                "evaluator_backend_id": task_quality_evaluator_backend_id,
            }
        )
        self._stable_identity = _freeze_identity(
            {
                "task_id": task_id,
                "consumer": "rl",
                "batch_size": num_envs,
                "backend_id": _GPU_NATIVE_BACKEND_ID,
                "api_version": _GPU_NATIVE_API_VERSION,
                "source_commit": expected_se3_source_commit,
                "source_tree": expected_se3_source_tree,
                "implementation_version": implementation_version,
                "runtime_versions": dict(runtime_versions),
                "contract_capabilities": dict(contract_capabilities),
                "environment_capabilities": dict(env_capabilities),
                "reset_manifest": {
                    "origin": manifest_origin,
                    "split": self._split.value,
                    "seed": manifest_seed,
                    "size": effective_manifest_size,
                    "sha256": observed_manifest_sha256,
                },
                "task_quality": task_quality_identity,
                "render_visual": render_visual,
                "device_identity": dict(vars(device_identity_start)),
            }
        )
        self._active_export_identity = _freeze_identity(
            {
                "export_dir": export_dir,
                **dict(export_identity_start),
                "frozen_request": _manifest_request_payload(frozen_request),
            }
        )
        self._closed = False

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def backend_id(self) -> str:
        return _GPU_NATIVE_BACKEND_ID

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> Any:
        return self._device

    @property
    def render_visual(self) -> bool:
        return self._render_visual

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
        return self._device_identity_end

    @property
    def device_identity_start(self) -> GpuDeviceIdentityReceipt:
        return self._device_identity_start

    @property
    def device_identity_end(self) -> GpuDeviceIdentityReceipt:
        return self._device_identity_end

    @property
    def stable_identity(self) -> Mapping[str, Any]:
        """Return run-stable identity without active-export digests."""

        return self._stable_identity

    @property
    def stable_run_identity(self) -> Mapping[str, Any]:
        """Descriptive alias for :attr:`stable_identity`."""

        return self._stable_identity

    @property
    def active_export_identity(self) -> Mapping[str, Any]:
        """Return only the export identity that may change between runs."""

        return self._active_export_identity

    @property
    def active_cohort_identity(self) -> Mapping[str, Any] | None:
        if self._active_episode_ids is None:
            return None
        return MappingProxyType(
            {
                "episode_ids": self._active_episode_ids,
                "manifest_ordinals": self._active_manifest_ordinals,
                "generation": self._active_generation,
            }
        )

    @property
    def last_transport_receipt(self) -> Mapping[str, Any] | None:
        return self._last_transport_receipt

    @property
    def transport_checks(self) -> int:
        return self._transport_checks

    @property
    def teacher_audit_materializations(self) -> int:
        """Return successful current-state teacher audit calls for this backend."""

        return self._teacher_audit_materializations

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

    @property
    def sequence_requests(self) -> tuple[Any, ...]:
        """Return defensive copies of the canonical frozen reset sequence."""

        return tuple(
            _defensive_manifest_request(request) for request in self._manifest_requests
        )

    @property
    def task_quality_schema_version(self) -> str | None:
        return self._task_quality_schema_version

    @property
    def task_quality_evaluator_backend_id(self) -> str | None:
        return self._task_quality_evaluator_backend_id

    def _stream_scope(self, torch_stream: Any | None = None) -> Any:
        if torch_stream is None:
            torch_stream = self._torch.cuda.current_stream(self._device)
        raw_stream_pointer = getattr(torch_stream, "cuda_stream", None)
        if (
            isinstance(raw_stream_pointer, bool)
            or not isinstance(raw_stream_pointer, int)
            or raw_stream_pointer < 0
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "PyTorch current stream lacks a CUDA stream handle"
            )
        if raw_stream_pointer == 0:
            return self._scoped_stream(self._stream_from_torch(torch_stream))
        stream_pointer = raw_stream_pointer
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
        if (
            isinstance(pointer, int)
            and not isinstance(pointer, bool)
            and cached is not None
            and cached[0] == pointer
        ):
            tensor = cached[1]
            tensor_pointer = tensor.data_ptr()
            if (
                isinstance(tensor_pointer, int)
                and not isinstance(tensor_pointer, bool)
                and tensor_pointer == pointer
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
        self._view_cache[name] = (tensor.data_ptr(), tensor)
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
                    factors=_thaw_json_value(base.factors),
                )
            )
        return tuple(requests)

    def reset(
        self,
        requests: Any | None = None,
        reset_mask: Any | None = None,
    ) -> GpuNativeTensorReset:
        if self._closed:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor backend is closed")
        if requests is not None:
            raise ValueError(
                "GPU tensor R0 reset requests are owned by the frozen manifest cursor"
            )
        if _manifest_sha256(self._manifest_requests) != self._manifest_sha256:
            raise GpuNativeTensorBackendUnavailableError(
                "frozen reset manifest changed before reset"
            )
        normalized_reset_mask = None
        if reset_mask is not None:
            try:
                numpy = importlib.import_module("numpy")
                normalized_reset_mask = numpy.asarray(reset_mask)
            except Exception as exc:
                raise ValueError("reset_mask must be a host bool vector") from exc
            if (
                normalized_reset_mask.dtype != numpy.bool_
                or normalized_reset_mask.shape != (self._num_envs,)
                or not bool(numpy.all(normalized_reset_mask))
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "GPU tensor backend rejects partial reset before any engine write"
                )
        if self._steps_since_reset not in {None, self._cohort_horizon_steps}:
            raise GpuNativeTensorBackendUnavailableError(
                "full-cohort reset is legal only before the first episode or at fixed horizon"
            )
        request_values = self.next_requests()
        if len(request_values) != self._num_envs or any(
            request is None for request in request_values
        ):
            raise ValueError(
                "GPU tensor reset requires one request for every cohort lane"
            )
        first_ordinal = self._next_cohort_index * self._num_envs
        ordinals = tuple(first_ordinal + lane for lane in range(self._num_envs))
        with self._stream_scope():
            if normalized_reset_mask is None:
                result = self._env.reset(request_values)
            else:
                result = self._env.reset(request_values, normalized_reset_mask)
        observation = self._view(
            result.observation,
            "observation",
            expected_dtype=self._torch.float32,
        )
        if len(observation.shape) != 2:
            raise GpuNativeTensorBackendUnavailableError(
                "state observation must have exact (B, D) layout"
            )
        state = getattr(result, "state", None)
        if (
            getattr(state, "backend_id", None) != _GPU_NATIVE_BACKEND_ID
            or getattr(state, "batch_size", None) != self._num_envs
            or getattr(state, "device_platform", None) != "cuda"
            or isinstance(getattr(state, "generation", None), bool)
            or not isinstance(getattr(state, "generation", None), int)
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM reset returned incompatible device state identity"
            )
        self._observation_shape = tuple(observation.shape)
        self._next_cohort_index += 1
        self._steps_since_reset = 0
        episode_ids = tuple(request.episode_id for request in request_values)
        self._active_episode_ids = episode_ids
        self._active_manifest_ordinals = ordinals
        self._active_generation = state.generation
        return GpuNativeTensorReset(
            observation=observation,
            generation=state.generation,
            episode_ids=episode_ids,
            seeds=tuple(request.seed for request in request_values),
            manifest_ordinals=ordinals,
            manifest_sha256=self._manifest_sha256,
        )

    def step_device(self, action: Any) -> GpuNativeTensorStep:
        """Advance the exact SE3 v0.2 device-tensor seam without host payload reads."""

        if self._closed:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor backend is closed")
        if self._steps_since_reset is None or self._active_episode_ids is None:
            raise GpuNativeTensorBackendUnavailableError(
                "GPU tensor step requires a reset cohort"
            )
        if self._steps_since_reset >= self._cohort_horizon_steps:
            raise GpuNativeTensorBackendUnavailableError(
                "GPU tensor cohort reached fixed horizon and must reset before another step"
            )
        if action.shape != (self._num_envs, 7):
            raise ValueError(f"GPU tensor action must have shape ({self._num_envs}, 7)")
        if action.dtype != self._torch.float32:
            raise ValueError("GPU tensor action must use torch.float32")
        if action.device != self._device or not action.is_contiguous():
            raise ValueError(
                "GPU tensor action must be contiguous on the backend CUDA device"
            )
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
        action_pointer = action.data_ptr()
        if (
            isinstance(action_pointer, bool)
            or not isinstance(action_pointer, int)
            or action_pointer <= 0
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "GPU tensor action lacks a positive device pointer"
            )
        with self._stream_scope(torch_stream):
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
        if (
            self._observation_shape is None
            or set(output_ptrs) != set(schemas)
            or any(
                isinstance(pointer, bool)
                or not isinstance(pointer, int)
                or pointer <= 0
                for pointer in output_ptrs.values()
            )
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM output pointer receipt has the wrong field set"
            )
        state = getattr(result, "state", None)
        if (
            getattr(state, "backend_id", None) != _GPU_NATIVE_BACKEND_ID
            or getattr(state, "batch_size", None) != self._num_envs
            or getattr(state, "device_platform", None) != "cuda"
            or isinstance(getattr(state, "generation", None), bool)
            or not isinstance(getattr(state, "generation", None), int)
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM tensor step returned incompatible device state identity"
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
            generation=state.generation,
            episode_ids=self._active_episode_ids,
        )

    def step(self, action: Any) -> GpuNativeTensorStep:
        """Compatibility spelling for callers already using the tensor backend."""

        return self.step_device(action)

    def enable_task_quality(
        self,
        *,
        schema_version: str,
        evaluator_backend_id: str,
    ) -> None:
        """Enable typed terminal quality strictly before the first reset."""

        if self._closed:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor backend is closed")
        if self._steps_since_reset is not None or self._next_cohort_index != 0:
            raise RuntimeError("task quality must be enabled before the first reset")
        if self._task_quality_schema_version is not None:
            raise RuntimeError("GPU tensor task quality is already enabled")
        for name, value in (
            ("schema_version", schema_version),
            ("evaluator_backend_id", evaluator_backend_id),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty trimmed string")
        self._env.enable_task_quality(
            evaluator_backend_id=evaluator_backend_id,
            schema_version=schema_version,
        )
        self._task_quality_schema_version = schema_version
        self._task_quality_evaluator_backend_id = evaluator_backend_id
        updated = dict(self._stable_identity)
        updated["task_quality"] = {
            "schema_version": schema_version,
            "evaluator_backend_id": evaluator_backend_id,
        }
        self._stable_identity = _freeze_identity(updated)

    def materialize_teacher_observations(
        self,
        lanes: tuple[int, ...] | None = None,
    ) -> tuple[Any, ...]:
        """Materialize current observations for an explicit teacher control plane.

        This method is intentionally separate from the device-only stepping path.
        Callers must pass only lanes that remain active; the returned observations
        preserve the requested lane order and are bound to the active episode and
        control clock.
        """

        if self._closed:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor backend is closed")
        if self._active_episode_ids is None or self._steps_since_reset is None:
            raise GpuNativeTensorBackendUnavailableError(
                "teacher observation materialization requires an active reset cohort"
            )
        selected = tuple(range(self._num_envs)) if lanes is None else lanes
        if not isinstance(selected, tuple):
            raise TypeError("teacher audit lanes must be an immutable tuple")
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("teacher audit lanes must be non-empty and unique")
        if any(
            isinstance(lane, bool)
            or not isinstance(lane, int)
            or not 0 <= lane < self._num_envs
            for lane in selected
        ):
            raise ValueError("teacher audit lane is outside the active cohort")
        try:
            from se3_wam.benchmark.api import ObservationBundle
            from se3_wam.benchmark.gpu_native.audit import (
                AuditBatch,
                AuditLane,
                AuditRequest,
            )
        except ImportError as exc:
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM current-state audit API is unavailable"
            ) from exc
        audit = self._env.materialize_audit(
            AuditRequest(lanes=selected, include_step_result=False)
        )
        if (
            type(audit) is not AuditBatch
            or audit.backend_id != _GPU_NATIVE_BACKEND_ID
            or audit.provenance != self._env.provenance
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "teacher audit backend/provenance identity mismatch"
            )
        materialized = audit.lanes
        if (
            not isinstance(materialized, tuple)
            or tuple(getattr(row, "lane", None) for row in materialized) != selected
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "teacher audit changed the requested lane order"
            )
        observations = []
        for lane, row in zip(selected, materialized, strict=True):
            if type(row) is not AuditLane or row.step_result is not None:
                raise GpuNativeTensorBackendUnavailableError(
                    "teacher audit row differs from the observation-only public ABI"
                )
            observation = row.observation
            if (
                row.episode_id != self._active_episode_ids[lane]
                or type(observation) is not ObservationBundle
                or observation.episode_id != self._active_episode_ids[lane]
                or observation.task_id != self._task_id
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "teacher audit changed lane, episode, or task identity"
                )
            if (
                observation.control_step != self._steps_since_reset
                or observation.policy_step != self._steps_since_reset
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "teacher audit clock differs from the active control boundary"
                )
            observations.append(observation)
        self._teacher_audit_materializations += 1
        return tuple(observations)

    def materialize_terminal_ledger_once(
        self,
        indices: tuple[int, ...],
        episode_ids: tuple[str, ...],
    ) -> tuple[GpuNativeTensorTerminalRow, ...]:
        """Materialize selected terminal lanes once at an explicit control boundary."""

        if self._closed:
            raise GpuNativeTensorBackendUnavailableError("GPU tensor backend is closed")
        if not isinstance(indices, tuple) or not isinstance(episode_ids, tuple):
            raise TypeError("terminal indices and episode_ids must be immutable tuples")
        if not indices or len(indices) != len(episode_ids):
            raise ValueError(
                "terminal indices and episode_ids must be non-empty and aligned"
            )
        if len(set(indices)) != len(indices) or len(set(episode_ids)) != len(
            episode_ids
        ):
            raise ValueError("terminal indices and episode_ids must be unique")
        if self._active_episode_ids is None:
            raise GpuNativeTensorBackendUnavailableError(
                "terminal materialization requires an active reset cohort"
            )
        for lane, episode_id in zip(indices, episode_ids, strict=True):
            if (
                isinstance(lane, bool)
                or not isinstance(lane, int)
                or not 0 <= lane < self._num_envs
            ):
                raise ValueError("terminal lane is outside the active cohort")
            if (
                not isinstance(episode_id, str)
                or not episode_id
                or episode_id.strip() != episode_id
                or self._active_episode_ids[lane] != episode_id
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal episode identity differs from the active cohort"
                )
        try:
            from se3_wam.benchmark.contracts import EventRecord
            from se3_wam.benchmark.gpu_native.audit import (
                TerminalLedgerBatch,
                TerminalLedgerRequest,
                TerminalLedgerRow,
                TerminalOutcome,
            )
            from se3_wam.benchmark.task_quality import EpisodeQualitySummary
        except ImportError as exc:
            raise GpuNativeTensorBackendUnavailableError(
                "SE3-WAM terminal-ledger API is unavailable"
            ) from exc
        ledger = self._env.materialize_terminal_ledger(
            TerminalLedgerRequest(lanes=indices, episode_ids=episode_ids)
        )
        if (
            type(ledger) is not TerminalLedgerBatch
            or ledger.backend_id != _GPU_NATIVE_BACKEND_ID
            or ledger.provenance != self._env.provenance
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "terminal ledger backend/provenance identity mismatch"
            )
        ledger_rows = ledger.rows
        if (
            not isinstance(ledger_rows, tuple)
            or tuple(getattr(row, "lane", None) for row in ledger_rows) != indices
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "terminal ledger changed the requested lane order"
            )
        rows = []
        for lane, episode_id, terminal in zip(
            indices,
            episode_ids,
            ledger_rows,
            strict=True,
        ):
            if type(terminal) is not TerminalLedgerRow:
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger rows must use the exact public ABI"
                )
            if terminal.episode_id != episode_id or terminal.task_id != self._task_id:
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger changed episode/task identity"
                )
            terminated = terminal.terminated
            truncated = terminal.truncated
            success = terminal.success
            reason = terminal.termination_reason
            completion = terminal.completion
            quality = terminal.task_quality
            if any(
                type(value) is not bool for value in (terminated, truncated, success)
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger masks are not exact booleans"
                )
            if not (terminated or truncated) or (terminated and truncated):
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger row is not exactly one terminal outcome"
                )
            if success and not terminated:
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger success does not imply termination"
                )
            if not isinstance(reason, str) or not reason or reason.strip() != reason:
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger row lacks a canonical termination reason"
                )
            if (
                type(completion) is not float
                or not math.isfinite(completion)
                or not 0.0 <= completion <= 1.0
                or (success and completion != 1.0)
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger row lacks canonical completion"
                )
            expected_outcome = (
                TerminalOutcome.SUCCESS
                if success
                else TerminalOutcome.TIMEOUT
                if truncated
                else TerminalOutcome.FAILURE
            )
            if terminal.outcome is not expected_outcome:
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger typed outcome disagrees with its flags"
                )
            if success and terminated:
                if (
                    type(quality) is not EpisodeQualitySummary
                    or quality.episode_id != episode_id
                    or quality.task_id != self._task_id
                    or quality.terminal is not True
                ):
                    raise GpuNativeTensorBackendUnavailableError(
                        "successful terminal ledger lacks typed task quality"
                    )
            elif quality is not None:
                raise GpuNativeTensorBackendUnavailableError(
                    "failure/timeout terminal ledger must not carry task quality"
                )
            events = terminal.events
            if not isinstance(events, tuple) or any(
                type(event) is not EventRecord for event in events
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger events do not use the exact EventRecord ABI"
                )
            clocks = (
                terminal.physics_step,
                terminal.control_step,
                terminal.policy_step,
            )
            if any(type(value) is not int or value < 0 for value in clocks):
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger clock is not a non-negative integer"
                )
            if (
                clocks[1] < 1
                or clocks[2] != clocks[1]
                or not 25 * (clocks[1] - 1) < clocks[0] <= 25 * clocks[1]
            ):
                raise GpuNativeTensorBackendUnavailableError(
                    "terminal ledger clocks do not identify a control interval"
                )
            rows.append(
                GpuNativeTensorTerminalRow(
                    lane=lane,
                    episode_id=episode_id,
                    task_id=self._task_id,
                    terminated=terminated,
                    truncated=truncated,
                    success=success,
                    termination_reason=reason,
                    completion=completion,
                    task_quality=quality,
                    observation=None,
                    events=events,
                    physics_step=clocks[0],
                    control_step=clocks[1],
                    policy_step=clocks[2],
                )
            )
        return tuple(rows)

    def attest_end(self) -> GpuDeviceIdentityReceipt:
        """Re-observe device and active export identity at the final control boundary."""

        if self._closed:
            raise GpuNativeTensorBackendUnavailableError(
                "end attestation must run before the backend is closed"
            )
        identity = _observe_device_identity(
            torch=self._torch,
            warp=self._warp,
            device=self._device,
            expected_gpu_uuid=self._expected_gpu_uuid,
            device_ordinal=self._device_ordinal,
        )
        if identity != self._device_identity_start:
            raise GpuNativeTensorBackendUnavailableError(
                "CUDA physical/logical identity changed between run start and end"
            )
        artifacts = self._load_artifacts(self._export_dir)
        if (
            _artifact_identity(artifacts) != self._active_export_digests
            or _manifest_request_payload(artifacts.reset_request)
            != self._active_export_identity["frozen_request"]
        ):
            raise GpuNativeTensorBackendUnavailableError(
                "active export identity changed between run start and end"
            )
        self._device_identity_end = identity
        self._final_device_identity = identity
        return identity

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
        if not isinstance(state, Mapping) or set(state) != {
            *expected,
            "next_cohort_index",
        }:
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
        attestation_error: BaseException | None = None
        try:
            self.attest_end()
        except BaseException as exc:
            attestation_error = exc
        try:
            self._env.close()
        finally:
            self._closed = True
        if attestation_error is not None:
            raise attestation_error

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError(
            "GpuNativeTensorBackendEnv is device-resident and not picklable"
        )


__all__ = [
    "GpuDeviceIdentityReceipt",
    "GpuNativeTensorBackendEnv",
    "GpuNativeTensorBackendUnavailableError",
    "GpuNativeTensorReset",
    "GpuNativeTensorStep",
    "GpuNativeTensorTerminalRow",
]
