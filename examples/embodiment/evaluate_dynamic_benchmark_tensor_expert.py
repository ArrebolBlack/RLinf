#!/usr/bin/env python3
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

"""Evaluate a frozen expert through the device-only Dynamic Benchmark path.

The command is deliberately split into a small parent and a fresh worker process.
The parent owns no CUDA state.  The worker verifies caller-pinned source, runtime,
policy, reset-sequence, export, and physical-device identities before stepping.
During a cohort, observations, actions, rewards, terminal signals, and accounting
remain on the selected device.  Host materialization is restricted to one typed
terminal-ledger read per completed cohort.

This executable produces validation evidence.  It never labels an artifact as a
production result; a separate scientific gate must make that decision.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

SEQUENCE_SCHEMA = "rlinf-dynamic-benchmark-tensor-validation-sequence-v0.1"
SOURCE_MANIFEST_SCHEMA = "rlinf-dynamic-benchmark-source-manifest-v0.1"
WORKER_SPEC_SCHEMA = "rlinf-dynamic-benchmark-tensor-worker-spec-v0.1"
EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-tensor-evaluation-v0.1"
EPISODE_LEDGER_SCHEMA = "rlinf-dynamic-benchmark-tensor-episode-ledger-v0.1"
POLICY_SCHEMAS = frozenset(
    {
        "rlinf-dynamic-benchmark-expert-policy-v0.1",
        "rlinf-gpuenv0-tensor-offpolicy-smoke-v0.1",
        "rlinf-gpuenv0-tensor-offpolicy-smoke-v0.2",
        "rlinf-gpuenv0-tensor-offpolicy-smoke-v0.3",
        "rlinf-gpuenv0-tensor-ppo-smoke-v0.1",
        "rlinf-gpuenv0-tensor-ppo-smoke-v0.2",
    }
)
_SHA256_LENGTH = 64
_GIT_OBJECT_LENGTH = 40
_REQUEST_FIELDS = (
    "episode_id",
    "task_id",
    "split",
    "seed",
    "action_mode",
    "observation_track",
    "object_mode",
    "reset_mode",
    "factors",
    "api_version",
)
_PORTABLE_EXPORT_FIELDS = (
    "request_sha256",
    "bundle_sha256",
    "model_sha256",
    "config_sha256",
    "manifest_sha256",
    "frozen_request",
)


class TensorEvaluationError(RuntimeError):
    """Raised when evidence cannot be produced without weakening an identity."""


class DevicePolicy(Protocol):
    """Minimal policy seam consumed by the device-only rollout loop."""

    observation_dim: int
    algorithm: str
    checkpoint_schema: str

    def act(self, observation: Any) -> Any:
        """Return a contiguous float32 action batch on the observation device."""


class DeviceLedger(Protocol):
    """Preallocated episode-accounting seam used by host fakes and CUDA runs."""

    def record(self, action: Any, step: Any) -> None:
        """Record one device step without host materialization."""

    def materialize_once(self) -> tuple[dict[str, Any], ...]:
        """Return one terminal host row per lane exactly once."""


@dataclass(frozen=True)
class ProcessIdentity:
    """Boot-scoped process identity used to prove fresh worker execution."""

    pid: int
    parent_pid: int
    boot_id: str
    start_ticks: int
    identity_source: str

    def __post_init__(self) -> None:
        for name in ("pid", "parent_pid", "start_ticks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.pid < 1 or self.start_ticks < 1:
            raise ValueError("pid and start_ticks must be positive")
        for name in ("boot_id", "identity_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty trimmed string")


@dataclass(frozen=True)
class PinnedSequence:
    """Exact caller-owned reset sequence and portable active-export identity."""

    task_id: str
    split: str
    manifest_seed: int
    manifest_sha256: str
    api_version: str
    task_quality_schema_version: str
    task_quality_evaluator_backend_id: str
    active_export_identity: Mapping[str, Any]
    requests: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PinnedSequence:
        """Validate and freeze an exact JSON sequence payload."""

        expected = {
            "schema_version",
            "task_id",
            "split",
            "manifest_seed",
            "manifest_sha256",
            "api_version",
            "task_quality",
            "active_export_identity",
            "requests",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError(
                "validation sequence keys do not match the released schema"
            )
        if payload["schema_version"] != SEQUENCE_SCHEMA:
            raise ValueError("unsupported validation sequence schema")
        for name in ("task_id", "split", "api_version"):
            _nonempty_string(name, payload[name])
        manifest_seed = payload["manifest_seed"]
        if (
            isinstance(manifest_seed, bool)
            or not isinstance(manifest_seed, int)
            or manifest_seed < 0
        ):
            raise ValueError("manifest_seed must be a non-negative integer")
        manifest_sha256 = require_sha256("manifest_sha256", payload["manifest_sha256"])
        quality = payload["task_quality"]
        if not isinstance(quality, Mapping) or set(quality) != {
            "schema_version",
            "evaluator_backend_id",
        }:
            raise ValueError("task_quality identity has an invalid schema")
        _nonempty_string("task quality schema_version", quality["schema_version"])
        _nonempty_string(
            "task quality evaluator_backend_id", quality["evaluator_backend_id"]
        )
        export_identity = portable_export_identity(payload["active_export_identity"])
        raw_requests = payload["requests"]
        if not isinstance(raw_requests, list) or not raw_requests:
            raise ValueError("validation sequence requests must be a non-empty list")
        requests: list[Mapping[str, Any]] = []
        episode_ids: set[str] = set()
        for ordinal, raw in enumerate(raw_requests):
            request = validate_request_payload(
                raw,
                expected_task_id=payload["task_id"],
                expected_split=payload["split"],
            )
            episode_id = request["episode_id"]
            if episode_id in episode_ids:
                raise ValueError(f"duplicate sequence episode id {episode_id!r}")
            episode_ids.add(episode_id)
            requests.append(request)
            if ordinal >= len(raw_requests):  # pragma: no cover - defensive invariant
                raise AssertionError("request ordinal escaped its sequence")
        observed_manifest_sha256 = manifest_requests_sha256(requests)
        if observed_manifest_sha256 != manifest_sha256:
            raise ValueError("validation sequence manifest SHA-256 mismatch")
        return cls(
            task_id=payload["task_id"],
            split=payload["split"],
            manifest_seed=manifest_seed,
            manifest_sha256=manifest_sha256,
            api_version=payload["api_version"],
            task_quality_schema_version=quality["schema_version"],
            task_quality_evaluator_backend_id=quality["evaluator_backend_id"],
            active_export_identity=export_identity,
            requests=tuple(requests),
        )


@dataclass(frozen=True)
class CohortResult:
    """One completed device cohort after its single terminal materialization."""

    episodes: tuple[dict[str, Any], ...]
    allocated_steps: int
    valid_steps: int
    rollout_seconds: float


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def require_sha256(name: str, value: Any) -> str:
    """Return one canonical lowercase SHA-256 or raise."""

    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def require_git_object(name: str, value: Any) -> str:
    """Return one full lowercase Git object id or raise."""

    if (
        not isinstance(value, str)
        or len(value) != _GIT_OBJECT_LENGTH
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git object id")
    return value


def jsonable(value: Any) -> Any:
    """Convert typed identity objects to strict canonical-JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("identity mappings must use string keys")
            converted[key] = jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return jsonable(to_dict())
    raise TypeError(f"unsupported receipt value {type(value)!r}")


def assert_strict_finite(value: Any, *, path: str = "$") -> None:
    """Reject non-JSON values, non-string keys, NaN, and infinities recursively."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string mapping key at {path}")
            assert_strict_finite(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_strict_finite(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"non-JSON value at {path}: {type(value)!r}")


def canonical_json(value: Any) -> str:
    """Serialize strict finite JSON deterministically."""

    converted = jsonable(value)
    assert_strict_finite(converted)
    return json.dumps(
        converted,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def payload_sha256(value: Any) -> str:
    """Hash one canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a regular file without loading it into memory."""

    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_pin(path: Path, expected_sha256: str, *, name: str) -> str:
    """Verify an immutable caller-provided file identity."""

    expected = require_sha256(f"expected {name} SHA-256", expected_sha256)
    observed = file_sha256(path.resolve(strict=True))
    if observed != expected:
        raise TensorEvaluationError(f"{name} SHA-256 mismatch")
    return observed


def load_pinned_json(path: Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    """Load a pinned finite JSON mapping and verify bytes before parsing."""

    verify_file_pin(path, expected_sha256, name=name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    assert_strict_finite(payload)
    return payload


def validate_request_payload(
    payload: Any,
    *,
    expected_task_id: str,
    expected_split: str,
) -> Mapping[str, Any]:
    """Validate one canonical ResetRequest-shaped JSON row."""

    if not isinstance(payload, Mapping) or set(payload) != set(_REQUEST_FIELDS):
        raise ValueError(
            "reset request fields do not match the canonical manifest schema"
        )
    for name in (
        "episode_id",
        "task_id",
        "split",
        "action_mode",
        "observation_track",
        "object_mode",
        "reset_mode",
        "api_version",
    ):
        _nonempty_string(f"request {name}", payload[name])
    if payload["task_id"] != expected_task_id or payload["split"] != expected_split:
        raise ValueError("reset request task/split differs from sequence identity")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("request seed must be a non-negative integer")
    if not isinstance(payload["factors"], Mapping):
        raise ValueError("request factors must be a mapping")
    frozen = {name: jsonable(payload[name]) for name in _REQUEST_FIELDS}
    assert_strict_finite(frozen)
    return frozen


def manifest_requests_sha256(requests: Sequence[Mapping[str, Any]]) -> str:
    """Hash a manifest using the backend's canonical request payload contract."""

    return payload_sha256(list(requests))


def runtime_request_payload(
    sequence: PinnedSequence, ordinal: int
) -> Mapping[str, Any]:
    """Return the cursor-owned runtime request identity for one global ordinal."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("runtime request ordinal must be a non-negative integer")
    manifest_size = len(sequence.requests)
    manifest_index = ordinal % manifest_size
    cycle = ordinal // manifest_size
    request = dict(sequence.requests[manifest_index])
    request["episode_id"] = f"{request['episode_id']}-cycle{cycle:08d}"
    return request


def portable_export_identity(identity: Any) -> Mapping[str, Any]:
    """Strip only the relocatable export path and retain every active digest."""

    if not isinstance(identity, Mapping):
        raise ValueError("active export identity must be a mapping")
    missing = set(_PORTABLE_EXPORT_FIELDS) - set(identity)
    extras = set(identity) - {*_PORTABLE_EXPORT_FIELDS, "export_dir"}
    if missing or extras:
        raise ValueError(
            f"active export identity fields drifted: missing={sorted(missing)}, "
            f"extra={sorted(extras)}"
        )
    portable: dict[str, Any] = {}
    for name in _PORTABLE_EXPORT_FIELDS:
        value = jsonable(identity[name])
        if name.endswith("_sha256"):
            value = require_sha256(f"active export {name}", value)
        portable[name] = value
    assert_strict_finite(portable)
    return portable


def _linux_process_identity(pid: int, proc_root: Path) -> ProcessIdentity:
    stat_path = proc_root / str(pid) / "stat"
    boot_path = proc_root / "sys" / "kernel" / "random" / "boot_id"
    stat = stat_path.read_text(encoding="ascii").strip()
    close_paren = stat.rfind(")")
    if close_paren < 1:
        raise TensorEvaluationError(f"cannot parse {stat_path}")
    parsed_pid = int(stat[: stat.find(" ")])
    remainder = stat[close_paren + 2 :].split()
    if parsed_pid != pid or len(remainder) <= 19:
        raise TensorEvaluationError(f"incomplete process identity in {stat_path}")
    parent_pid = int(remainder[1])
    start_ticks = int(remainder[19])
    boot_id = boot_path.read_text(encoding="ascii").strip().lower()
    return ProcessIdentity(
        pid=pid,
        parent_pid=parent_pid,
        boot_id=_nonempty_string("Linux boot id", boot_id),
        start_ticks=start_ticks,
        identity_source="linux_procfs",
    )


def _windows_process_times(pid: int) -> tuple[int, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_query_limited_information = 0x1000
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"OpenProcess({pid}) failed")
    try:
        creation = ctypes.c_ulonglong()
        exit_time = ctypes.c_ulonglong()
        kernel = ctypes.c_ulonglong()
        user = ctypes.c_ulonglong()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise OSError(ctypes.get_last_error(), f"GetProcessTimes({pid}) failed")
    finally:
        kernel32.CloseHandle(handle)
    return int(creation.value), int(exit_time.value)


def _windows_boot_time() -> int:
    class SystemTimeOfDayInformation(ctypes.Structure):
        _fields_ = [
            ("boot_time", ctypes.c_longlong),
            ("current_time", ctypes.c_longlong),
            ("time_zone_bias", ctypes.c_longlong),
            ("time_zone_id", ctypes.c_ulong),
            ("reserved", ctypes.c_ulong),
            ("boot_time_bias", ctypes.c_ulonglong),
            ("sleep_time_bias", ctypes.c_ulonglong),
        ]

    information = SystemTimeOfDayInformation()
    status = ctypes.WinDLL("ntdll").NtQuerySystemInformation(
        3,
        ctypes.byref(information),
        ctypes.sizeof(information),
        None,
    )
    if status != 0 or information.boot_time <= 0:
        raise OSError(f"NtQuerySystemInformation failed with NTSTATUS {status:#x}")
    return int(information.boot_time)


def _windows_parent_pid(pid: int) -> int:
    """Read one process's parent PID from the Windows process snapshot."""

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32FirstW.restype = ctypes.c_int
    kernel32.Process32NextW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32NextW.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot in {None, invalid_handle}:
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        available = bool(kernel32.Process32FirstW(snapshot, ctypes.byref(entry)))
        while available:
            if int(entry.th32ProcessID) == pid:
                return int(entry.th32ParentProcessID)
            available = bool(kernel32.Process32NextW(snapshot, ctypes.byref(entry)))
    finally:
        kernel32.CloseHandle(snapshot)
    raise ProcessLookupError(pid)


def _windows_process_identity(pid: int) -> ProcessIdentity:
    creation, _exit_time = _windows_process_times(pid)
    boot_time = _windows_boot_time()
    machine = platform.node().encode("utf-8")
    boot_id = hashlib.sha256(
        machine + b"\0" + str(boot_time).encode("ascii")
    ).hexdigest()
    parent_pid = _windows_parent_pid(pid)
    return ProcessIdentity(
        pid=pid,
        parent_pid=parent_pid,
        boot_id=boot_id,
        start_ticks=creation,
        identity_source="windows_kernel_times",
    )


def collect_process_identity(
    pid: int | None = None,
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessIdentity:
    """Collect a boot-scoped process identity on Linux or Windows."""

    selected = os.getpid() if pid is None else pid
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
        raise ValueError("pid must be a positive integer")
    if (proc_root / "self").exists() or (proc_root / str(selected) / "stat").exists():
        return _linux_process_identity(selected, proc_root)
    if os.name == "nt":
        return _windows_process_identity(selected)
    raise TensorEvaluationError("boot/start-tick process identity is unavailable")


def process_identity_matches(
    expected: Mapping[str, Any], observed: ProcessIdentity
) -> bool:
    """Compare a serialized process identity without coercing field types."""

    return dict(expected) == asdict(observed)


def git_output(root: Path, *arguments: str) -> str:
    """Return one non-interactive Git query result."""

    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def validate_source_manifest(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Validate exact source roots, commits, trees, and import module names."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "sources",
    }:
        raise ValueError("source manifest keys do not match the released schema")
    if payload["schema_version"] != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported source manifest schema")
    sources = payload["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("source manifest requires at least one source")
    result = []
    names: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {
            "name",
            "root",
            "commit",
            "tree",
            "module",
        }:
            raise ValueError("source pin keys do not match the released schema")
        name = _nonempty_string("source name", source["name"])
        if name in names:
            raise ValueError(f"duplicate source name {name!r}")
        names.add(name)
        _nonempty_string("source root", source["root"])
        _nonempty_string("source module", source["module"])
        require_git_object("source commit", source["commit"])
        require_git_object("source tree", source["tree"])
        result.append(dict(source))
    if not {"rlinf", "se3_wam"}.issubset(names):
        raise ValueError("source manifest must pin rlinf and se3_wam")
    return tuple(result)


def capture_source_snapshot(
    source_pins: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Resolve imports and fail closed on commit, tree, or worktree drift."""

    snapshot: dict[str, Any] = {}
    for pin in source_pins:
        root = Path(pin["root"]).resolve(strict=True)
        module = importlib.import_module(pin["module"])
        imported_file_value = getattr(module, "__file__", None)
        if not isinstance(imported_file_value, str):
            raise TensorEvaluationError(f"source module {pin['module']!r} has no file")
        imported_file = Path(imported_file_value).resolve(strict=True)
        if not imported_file.is_relative_to(root):
            raise TensorEvaluationError(
                f"loaded module {imported_file} is outside pinned source {root}"
            )
        observed_commit = git_output(root, "rev-parse", "HEAD")
        observed_tree = git_output(root, "show", "-s", "--format=%T", "HEAD")
        dirty = git_output(root, "status", "--porcelain=v1")
        if observed_commit != pin["commit"] or observed_tree != pin["tree"] or dirty:
            raise TensorEvaluationError(
                f"source identity drift for {pin['name']}: "
                f"commit={observed_commit}, tree={observed_tree}, dirty={bool(dirty)}"
            )
        snapshot[pin["name"]] = {
            "root": str(root),
            "module": pin["module"],
            "loaded_file": str(imported_file),
            "commit": observed_commit,
            "tree": observed_tree,
            "tracked_worktree_clean": True,
        }
    return snapshot


def validate_backend_runtime_identity(
    stable_identity: Mapping[str, Any],
    *,
    runtime_payload: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    sequence: PinnedSequence,
) -> None:
    """Bind the backend's stable API/source/runtime identity to caller pins."""

    reset_manifest = stable_identity.get("reset_manifest")
    task_quality = stable_identity.get("task_quality")
    expected_quality = {
        "schema_version": sequence.task_quality_schema_version,
        "evaluator_backend_id": sequence.task_quality_evaluator_backend_id,
    }
    if (
        stable_identity.get("task_id") != sequence.task_id
        or stable_identity.get("api_version") != sequence.api_version
        or not isinstance(reset_manifest, Mapping)
        or reset_manifest.get("origin") != "caller"
        or reset_manifest.get("split") != sequence.split
        or reset_manifest.get("seed") != sequence.manifest_seed
        or reset_manifest.get("size") != len(sequence.requests)
        or reset_manifest.get("sha256") != sequence.manifest_sha256
        or task_quality != expected_quality
    ):
        raise TensorEvaluationError("backend stable reset/API/quality identity drifted")
    se3_source = source_snapshot.get("se3_wam")
    if not isinstance(se3_source, Mapping) or (
        stable_identity.get("source_commit") != se3_source.get("commit")
        or stable_identity.get("source_tree") != se3_source.get("tree")
    ):
        raise TensorEvaluationError(
            "backend source identity differs from loaded se3_wam"
        )
    expected_versions = runtime_payload.get("versions")
    observed_versions = stable_identity.get("runtime_versions")
    if not isinstance(expected_versions, Mapping) or not isinstance(
        observed_versions, Mapping
    ):
        raise TensorEvaluationError("runtime manifest/backend lacks a versions mapping")
    aliases = {
        "mujoco": ("mujoco",),
        "mujoco-warp": ("mujoco-warp", "mujoco-mjx"),
        "warp-lang": ("warp-lang",),
    }
    mismatches = {}
    for observed_name, expected_names in aliases.items():
        expected = next(
            (
                str(expected_versions[name])
                for name in expected_names
                if name in expected_versions
            ),
            None,
        )
        observed = observed_versions.get(observed_name)
        if expected is None or observed != expected:
            mismatches[observed_name] = {"expected": expected, "observed": observed}
    if mismatches:
        raise TensorEvaluationError(
            f"backend/runtime manifest version mismatch: {mismatches}"
        )


def validate_finite_tensor_mapping(
    torch_module: Any,
    state: Mapping[str, Any],
    *,
    name: str,
    selected_keys: set[str] | None = None,
) -> None:
    """Validate checkpoint tensors on the host before any CUDA hot path begins."""

    keys = set(state) if selected_keys is None else selected_keys
    for key in keys:
        value = state.get(key)
        if not isinstance(key, str) or not isinstance(value, torch_module.Tensor):
            raise ValueError(f"{name} must contain named tensors")
        if not bool(torch_module.isfinite(value).all()):
            raise ValueError(f"{name} tensor {key!r} contains NaN or Inf")


def build_reset_requests(sequence: PinnedSequence) -> tuple[Any, ...]:
    """Construct typed public SE3 ResetRequest values from the pinned JSON rows."""

    from se3_wam.benchmark.api import ResetRequest, Split
    from se3_wam.benchmark.contracts import ActionMode, ObservationTrack

    return tuple(
        ResetRequest(
            episode_id=row["episode_id"],
            task_id=row["task_id"],
            split=Split(row["split"]),
            seed=row["seed"],
            action_mode=ActionMode(row["action_mode"]),
            observation_track=ObservationTrack(row["observation_track"]),
            object_mode=row["object_mode"],
            reset_mode=row["reset_mode"],
            factors=row["factors"],
            api_version=row["api_version"],
        )
        for row in sequence.requests
    )


class TorchEpisodeLedger:
    """Preallocated device ledger for returns, valid steps, and action cost."""

    def __init__(self, torch_module: Any, *, num_envs: int, device: Any) -> None:
        self._torch = torch_module
        self._num_envs = num_envs
        self._device = device
        self._returns = torch_module.zeros(
            num_envs, dtype=torch_module.float32, device=device
        )
        self._action_cost = torch_module.zeros(
            num_envs, dtype=torch_module.float32, device=device
        )
        self._valid_steps = torch_module.zeros(
            num_envs, dtype=torch_module.int64, device=device
        )
        self._done = torch_module.zeros(
            num_envs, dtype=torch_module.bool, device=device
        )
        self._materialized = False

    def record(self, action: Any, step: Any) -> None:
        """Accumulate exactly the active prefix of every allocated lane."""

        if self._materialized:
            raise TensorEvaluationError("cannot record after ledger materialization")
        active = ~self._done
        active_float = active.to(dtype=self._torch.float32)
        self._returns.add_(step.reward * active_float)
        self._action_cost.add_(action.square().sum(dim=-1) * active_float)
        self._valid_steps.add_(active.to(dtype=self._torch.int64))
        self._done.logical_or_(active & step.done)

    def materialize_once(self) -> tuple[dict[str, Any], ...]:
        """Copy the compact numeric terminal ledger to host exactly once."""

        if self._materialized:
            raise TensorEvaluationError(
                "device episode ledger was already materialized"
            )
        self._materialized = True
        packed = self._torch.stack(
            (
                self._returns,
                self._action_cost,
                self._valid_steps.to(dtype=self._torch.float32),
                self._done.to(dtype=self._torch.float32),
            ),
            dim=-1,
        )
        host_rows = packed.detach().to(device="cpu").tolist()
        rows = []
        for lane, values in enumerate(host_rows):
            episode_return, action_cost, valid_steps, done = values
            row = {
                "lane": lane,
                "return": float(episode_return),
                "episode_cost": float(action_cost),
                "valid_steps": int(valid_steps),
                "done": bool(done),
            }
            assert_strict_finite(row)
            rows.append(row)
        return tuple(rows)


def _make_expert_policy(
    torch_module: Any,
    payload: Mapping[str, Any],
    device: Any,
) -> DevicePolicy:
    nn = torch_module.nn
    config = payload.get("config")
    state_schema = payload.get("state_schema")
    state_dict = payload.get("model")
    normalizer = payload.get("normalizer")
    if not all(
        isinstance(value, Mapping)
        for value in (config, state_schema, state_dict, normalizer)
    ):
        raise ValueError("expert policy config/state/model/normalizer is incomplete")
    algorithm = str(config.get("algorithm", ""))
    if algorithm not in {"bc", "sac", "rlpd", "ppo"}:
        raise ValueError(
            "tensor evaluation supports bc/sac/rlpd/ppo; residual planners are host-only"
        )
    observation_dim = state_schema.get("state_dim")
    mask_dim = state_schema.get("mask_dim")
    if (
        isinstance(observation_dim, bool)
        or not isinstance(observation_dim, int)
        or observation_dim < 1
        or isinstance(mask_dim, bool)
        or not isinstance(mask_dim, int)
        or not 0 <= mask_dim <= observation_dim
    ):
        raise ValueError("expert policy state schema is invalid")

    class ExpertActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(observation_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
            )
            self.actor_mean = nn.Linear(256, 7)
            if algorithm == "ppo":
                self.actor_logstd = nn.Parameter(torch_module.empty(1, 7))
            else:
                self.actor_logstd = nn.Linear(256, 7)

        def forward(self, observation: Any) -> Any:
            return torch_module.tanh(self.actor_mean(self.backbone(observation)))

    module = ExpertActor()
    actor_keys = set(module.state_dict())
    if not isinstance(state_dict, Mapping) or any(
        not isinstance(key, str) for key in state_dict
    ):
        raise ValueError("expert policy state must use string keys")
    missing = actor_keys - set(state_dict)
    if missing:
        raise ValueError(f"expert actor is missing state keys: {sorted(missing)}")
    auxiliary = set(state_dict) - actor_keys
    if algorithm == "ppo":
        if not auxiliary or any(not key.startswith("value_head.") for key in auxiliary):
            raise ValueError(
                "expert PPO checkpoint must contain only actor and value-head state"
            )
    else:
        buffer_keys = {"action_scale", "action_bias"}
        q_keys = auxiliary - buffer_keys
        q_heads = config.get("q_heads", 2)
        if (
            isinstance(q_heads, bool)
            or not isinstance(q_heads, int)
            or q_heads < 2
            or auxiliary & buffer_keys != buffer_keys
            or not q_keys
            or any(not key.startswith("q_head.qs.") for key in q_keys)
        ):
            raise ValueError("expert SAC-family checkpoint auxiliary state is invalid")
        q_indices = set()
        for key in q_keys:
            parts = key.split(".")
            if len(parts) < 4 or not parts[2].isdigit():
                raise ValueError("expert Q-head state key has an invalid structure")
            q_indices.add(int(parts[2]))
        if q_indices != set(range(q_heads)):
            raise ValueError("expert Q-head count differs from policy config")
        for key, expected in (("action_scale", 1.0), ("action_bias", 0.0)):
            value = state_dict[key]
            if (
                not isinstance(value, torch_module.Tensor)
                or value.numel() != 1
                or not bool(value.reshape(()) == expected)
            ):
                raise ValueError(f"expert {key} buffer has unsupported action bounds")
    validate_finite_tensor_mapping(
        torch_module,
        state_dict,
        name="expert checkpoint state",
    )
    module.load_state_dict({key: state_dict[key] for key in actor_keys}, strict=True)
    if set(normalizer) != {"dimension", "mask_dim", "epsilon", "count", "mean", "m2"}:
        raise ValueError("expert normalizer state fields are not exact")
    count = normalizer.get("count")
    epsilon = normalizer.get("epsilon")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
        or normalizer.get("dimension") != observation_dim
        or normalizer.get("mask_dim") != mask_dim
    ):
        raise ValueError("expert normalizer metadata is invalid")
    mean = torch_module.as_tensor(normalizer.get("mean"), dtype=torch_module.float64)
    m2 = torch_module.as_tensor(normalizer.get("m2"), dtype=torch_module.float64)
    if tuple(mean.shape) != (observation_dim,) or tuple(m2.shape) != (observation_dim,):
        raise ValueError("expert normalizer tensor shapes are invalid")
    if (
        not bool(torch_module.isfinite(mean).all())
        or not bool(torch_module.isfinite(m2).all())
        or bool((m2 < 0).any())
    ):
        raise ValueError("expert normalizer statistics are non-finite or invalid")
    statistics = None
    if count >= 2:
        variance = m2 / max(1, count - 1)
        normalized_mean = mean.clone()
        scale = torch_module.sqrt(torch_module.clamp(variance, min=float(epsilon) ** 2))
        if mask_dim:
            normalized_mean[-mask_dim:] = 0.0
            scale[-mask_dim:] = 1.0
        statistics = (
            normalized_mean.to(device=device, dtype=torch_module.float32),
            scale.to(device=device, dtype=torch_module.float32),
        )
    module.to(device)
    module.eval()

    class Adapter:
        checkpoint_schema = "rlinf-dynamic-benchmark-expert-policy-v0.1"

        def __init__(self) -> None:
            self.observation_dim = observation_dim
            self.algorithm = algorithm

        def act(self, observation: Any) -> Any:
            value = observation.to(dtype=torch_module.float32)
            if statistics is not None:
                center, spread = statistics
                value = torch_module.clamp((value - center) / spread, -10.0, 10.0)
            with torch_module.inference_mode():
                return module(value).contiguous()

    return Adapter()


def _make_ppo_smoke_policy(
    torch_module: Any,
    payload: Mapping[str, Any],
    device: Any,
) -> DevicePolicy:
    nn = torch_module.nn
    config = payload.get("config")
    state = payload.get("model")
    observation_dim = payload.get("observation_dim")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("tensor PPO checkpoint config/model is incomplete")
    hidden_size = config.get("hidden_size")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (observation_dim, hidden_size)
    ):
        raise ValueError("tensor PPO dimensions are invalid")

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden_size, 7)
            self.critic = nn.Linear(hidden_size, 1)
            self.log_std = nn.Parameter(torch_module.empty(7))

        def forward(self, observation: Any) -> Any:
            return torch_module.tanh(self.actor(self.backbone(observation)))

    module = ActorCritic()
    model_keys = set(module.state_dict())
    validate_finite_tensor_mapping(
        torch_module,
        state,
        name="tensor PPO model state",
        selected_keys=model_keys,
    )
    module.load_state_dict(state, strict=True)
    module.to(device)
    module.eval()

    class Adapter:
        checkpoint_schema = str(payload["schema_version"])
        algorithm = "ppo"

        def __init__(self) -> None:
            self.observation_dim = observation_dim

        def act(self, observation: Any) -> Any:
            with torch_module.inference_mode():
                return module(observation).contiguous()

    return Adapter()


def _make_offpolicy_smoke_policy(
    torch_module: Any,
    payload: Mapping[str, Any],
    device: Any,
) -> DevicePolicy:
    nn = torch_module.nn
    config = payload.get("config")
    state = payload.get("actor")
    observation_dim = payload.get("observation_dim")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("tensor off-policy checkpoint config/actor is incomplete")
    hidden_size = config.get("hidden_size")
    algorithm = config.get("algorithm")
    if algorithm not in {"sac", "rlpd"}:
        raise ValueError("tensor off-policy checkpoint algorithm is invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (observation_dim, hidden_size)
    ):
        raise ValueError("tensor off-policy dimensions are invalid")

    class Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.mean = nn.Linear(hidden_size, 7)
            self.log_std = nn.Linear(hidden_size, 7)

        def forward(self, observation: Any) -> Any:
            return torch_module.tanh(self.mean(self.backbone(observation)))

    module = Actor()
    validate_finite_tensor_mapping(
        torch_module,
        state,
        name="tensor off-policy actor state",
        selected_keys=set(module.state_dict()),
    )
    module.load_state_dict(state, strict=True)
    module.to(device)
    module.eval()

    class Adapter:
        checkpoint_schema = str(payload["schema_version"])

        def __init__(self) -> None:
            self.observation_dim = observation_dim
            self.algorithm = algorithm

        def act(self, observation: Any) -> Any:
            with torch_module.inference_mode():
                return module(observation).contiguous()

    return Adapter()


def load_device_policy(
    torch_module: Any,
    path: Path,
    *,
    expected_sha256: str,
    device: Any,
) -> tuple[DevicePolicy, Mapping[str, Any]]:
    """Load one pinned supported checkpoint into a deterministic device policy."""

    observed_sha256 = verify_file_pin(path, expected_sha256, name="policy")
    payload = torch_module.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("policy checkpoint must contain a mapping")
    schema = payload.get("schema_version")
    if schema not in POLICY_SCHEMAS:
        raise ValueError(f"unsupported tensor evaluation policy schema {schema!r}")
    if schema == "rlinf-dynamic-benchmark-expert-policy-v0.1":
        policy = _make_expert_policy(torch_module, payload, device)
    elif schema in {
        "rlinf-gpuenv0-tensor-ppo-smoke-v0.1",
        "rlinf-gpuenv0-tensor-ppo-smoke-v0.2",
    }:
        policy = _make_ppo_smoke_policy(torch_module, payload, device)
    else:
        policy = _make_offpolicy_smoke_policy(torch_module, payload, device)
    config = payload.get("config")
    policy_identity = {
        "path": str(path.resolve(strict=True)),
        "sha256": observed_sha256,
        "schema_version": schema,
        "algorithm": policy.algorithm,
        "observation_dim": policy.observation_dim,
        "training_identity": {
            key: jsonable(config[key])
            for key in (
                "task",
                "seed",
                "split",
                "manifest_seed",
                "rlinf_commit",
                "se3_commit",
                "benchmark_commit",
                "runtime_manifest_sha256",
                "expected_gpu_uuid",
            )
            if isinstance(config, Mapping) and key in config
        },
    }
    return policy, policy_identity


def validate_reset_against_requests(
    reset: Any,
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    expected_ordinals: Sequence[int],
    manifest_sha256: str,
) -> None:
    """Reject reset cursor, episode, seed, or manifest divergence."""

    expected_episode_ids = tuple(row["episode_id"] for row in expected_rows)
    expected_seeds = tuple(row["seed"] for row in expected_rows)
    if (
        tuple(reset.episode_ids) != expected_episode_ids
        or tuple(reset.seeds) != expected_seeds
        or tuple(reset.manifest_ordinals) != tuple(expected_ordinals)
        or reset.manifest_sha256 != manifest_sha256
    ):
        raise TensorEvaluationError(
            "backend reset diverged from caller-pinned sequence"
        )


def normalize_terminal_rows(
    terminal_rows: Sequence[Any],
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    numeric_rows: Sequence[Mapping[str, Any]],
    quality_type: type,
    quality_from_dict: Callable[[Mapping[str, Any]], Any],
) -> tuple[dict[str, Any], ...]:
    """Validate typed terminal rows and join them to the compact numeric ledger."""

    if len(terminal_rows) != len(expected_rows) or len(numeric_rows) != len(
        expected_rows
    ):
        raise TensorEvaluationError("terminal ledger row count differs from the cohort")
    normalized = []
    for lane, (terminal, request, numeric) in enumerate(
        zip(terminal_rows, expected_rows, numeric_rows, strict=True)
    ):
        episode_id = request["episode_id"]
        if terminal.lane != lane or terminal.episode_id != episode_id:
            raise TensorEvaluationError("terminal ledger lane/episode identity drifted")
        if terminal.task_id != request["task_id"]:
            raise TensorEvaluationError("terminal ledger task identity drifted")
        for name in ("terminated", "truncated", "success"):
            if not isinstance(getattr(terminal, name), bool):
                raise TensorEvaluationError(f"terminal {name} must be typed bool")
        if terminal.terminated == terminal.truncated:
            raise TensorEvaluationError(
                "terminal row must be exactly terminated or truncated"
            )
        if terminal.success and (not terminal.terminated or terminal.truncated):
            raise TensorEvaluationError(
                "success must be a canonical terminated outcome"
            )
        completion = terminal.completion
        if (
            isinstance(completion, bool)
            or not isinstance(completion, (int, float))
            or not math.isfinite(float(completion))
            or not 0.0 <= float(completion) <= 1.0
        ):
            raise TensorEvaluationError("terminal completion must be finite in [0, 1]")
        quality_payload = None
        if terminal.success and terminal.terminated:
            if not isinstance(terminal.task_quality, quality_type):
                raise TensorEvaluationError(
                    "successful terminal row lacks typed canonical task_quality"
                )
            quality_payload = jsonable(terminal.task_quality.to_dict())
            decoded = quality_from_dict(quality_payload)
            if (
                not isinstance(decoded, quality_type)
                or decoded != terminal.task_quality
            ):
                raise TensorEvaluationError(
                    "task_quality failed canonical typed round-trip"
                )
            if (
                not bool(getattr(decoded, "terminal", False))
                or getattr(decoded, "episode_id", None) != episode_id
                or getattr(decoded, "task_id", None) != request["task_id"]
            ):
                raise TensorEvaluationError("task_quality terminal identity drifted")
        elif terminal.task_quality is not None:
            raise TensorEvaluationError(
                "failure/timeout task_quality must be strictly None"
            )
        if not numeric.get("done") or numeric.get("lane") != lane:
            raise TensorEvaluationError(
                "device numeric ledger did not observe terminal completion"
            )
        reason = getattr(
            terminal.termination_reason, "value", terminal.termination_reason
        )
        _nonempty_string("termination_reason", reason)
        row = {
            "schema_version": EPISODE_LEDGER_SCHEMA,
            "ordinal": -1,
            "lane": lane,
            "episode_id": episode_id,
            "task_id": request["task_id"],
            "split": request["split"],
            "seed": request["seed"],
            "return": float(numeric["return"]),
            "success": terminal.success,
            "terminated": terminal.terminated,
            "truncated": terminal.truncated,
            "safety": None,
            "safety_available": False,
            "safety_failure": None,
            "completion": float(completion),
            "task_quality": quality_payload,
            "episode_cost": float(numeric["episode_cost"]),
            "episode_cost_kind": "normalized_action_l2_sum",
            "control_steps": int(numeric["valid_steps"]),
            "terminal_reason": reason,
            "terminal_physics_step": int(terminal.physics_step),
            "terminal_control_step": int(terminal.control_step),
            "terminal_policy_step": int(terminal.policy_step),
        }
        assert_strict_finite(row)
        normalized.append(row)
    return tuple(normalized)


def run_device_cohort(
    *,
    env: Any,
    policy: DevicePolicy,
    expected_rows: Sequence[Mapping[str, Any]],
    expected_ordinals: Sequence[int],
    ledger_factory: Callable[[int, Any], DeviceLedger],
    synchronize: Callable[[Any], None],
    quality_type: type,
    quality_from_dict: Callable[[Mapping[str, Any]], Any],
    clock: Callable[[], float] = time.perf_counter,
) -> CohortResult:
    """Run one fixed-horizon device cohort and materialize only at its boundary."""

    reset = env.reset()
    validate_reset_against_requests(
        reset,
        expected_rows=expected_rows,
        expected_ordinals=expected_ordinals,
        manifest_sha256=env.manifest_sha256,
    )
    observation_shape = tuple(reset.observation.shape)
    if observation_shape != (env.num_envs, policy.observation_dim):
        raise TensorEvaluationError(
            f"policy/backend observation shape mismatch: {observation_shape}"
        )
    ledger = ledger_factory(env.num_envs, env.device)
    observation = reset.observation
    synchronize(env.device)
    started = clock()
    for _control_step in range(env.cohort_horizon_steps):
        action = policy.act(observation)
        step = env.step_device(action)
        if tuple(step.episode_ids) != tuple(reset.episode_ids):
            raise TensorEvaluationError("step episode identity drifted within a cohort")
        ledger.record(action, step)
        observation = step.observation
    synchronize(env.device)
    rollout_seconds = clock() - started
    if not math.isfinite(rollout_seconds) or rollout_seconds <= 0.0:
        raise TensorEvaluationError(
            "cohort rollout duration must be finite and positive"
        )
    terminal_rows = env.materialize_terminal_ledger_once(
        tuple(range(env.num_envs)), tuple(reset.episode_ids)
    )
    numeric_rows = ledger.materialize_once()
    episodes = list(
        normalize_terminal_rows(
            terminal_rows,
            expected_rows=expected_rows,
            numeric_rows=numeric_rows,
            quality_type=quality_type,
            quality_from_dict=quality_from_dict,
        )
    )
    for ordinal, row in zip(expected_ordinals, episodes, strict=True):
        row["ordinal"] = ordinal
    valid_steps = sum(row["control_steps"] for row in episodes)
    return CohortResult(
        episodes=tuple(episodes),
        allocated_steps=env.cohort_horizon_steps * env.num_envs,
        valid_steps=valid_steps,
        rollout_seconds=rollout_seconds,
    )


def summarize_episodes(
    episodes: Sequence[Mapping[str, Any]],
    *,
    allocated_steps: int,
    valid_steps: int,
    rollout_seconds: float,
) -> Mapping[str, Any]:
    """Summarize validation metrics without making a scientific claim."""

    if not episodes:
        raise ValueError("cannot summarize an empty episode sequence")
    if allocated_steps < 1 or valid_steps < 1 or valid_steps > allocated_steps:
        raise ValueError("invalid allocated/valid step totals")
    if not math.isfinite(rollout_seconds) or rollout_seconds <= 0.0:
        raise ValueError("rollout_seconds must be finite and positive")
    count = len(episodes)
    result = {
        "episodes": count,
        "mean_return": sum(float(row["return"]) for row in episodes) / count,
        "success_rate": sum(bool(row["success"]) for row in episodes) / count,
        "safety": {"available": False, "failure_rate": None},
        "safety_available": False,
        "safety_failure_rate": None,
        "mean_completion": sum(float(row["completion"]) for row in episodes) / count,
        "mean_episode_cost": sum(float(row["episode_cost"]) for row in episodes)
        / count,
        "task_quality_success_count": sum(
            row["task_quality"] is not None for row in episodes
        ),
        "throughput": {
            "rollout_seconds": rollout_seconds,
            "allocated_env_steps": allocated_steps,
            "valid_env_steps": valid_steps,
            "allocated_env_steps_per_s": allocated_steps / rollout_seconds,
            "valid_env_steps_per_s": valid_steps / rollout_seconds,
            "episodes_per_s": count / rollout_seconds,
        },
    }
    assert_strict_finite(result)
    return result


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one strict finite JSON mapping."""

    rendered = json.dumps(
        jsonable(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    assert_strict_finite(payload)
    _atomic_bytes(path, (rendered + "\n").encode("utf-8"))


def _rename_directory_with_claim(source: Path, destination: Path) -> None:
    """Publish by rename under a cooperative, atomic no-clobber claim."""

    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite {destination}")
    claim = destination.with_name(f".{destination.name}.publish.lock")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(claim, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"publication is already in progress for {destination}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((source.name + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(destination):
            raise FileExistsError(f"refusing to overwrite {destination}")
        try:
            os.rename(source, destination)
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY} or os.path.lexists(
                destination
            ):
                raise FileExistsError(f"refusing to overwrite {destination}") from exc
            raise
    finally:
        claim.unlink(missing_ok=True)


def rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing an existing destination.

    Linux filesystems without ``RENAME_NOREPLACE`` use an exclusive publication
    claim before an ordinary same-filesystem rename.  The claim serializes every
    cooperating publisher and fails closed if a prior publisher was interrupted.
    """

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file_write_through = 0x00000008
        kernel32.MoveFileExW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        ]
        kernel32.MoveFileExW.restype = ctypes.c_int
        if not kernel32.MoveFileExW(
            str(source), str(destination), move_file_write_through
        ):
            error = ctypes.get_last_error()
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite {destination}")
            raise OSError(
                error, f"atomic directory publication failed for {destination}"
            )
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            _rename_directory_with_claim(source, destination)
            return
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(f"refusing to overwrite {destination}")
            unsupported = {
                errno.EINVAL,
                errno.ENOSYS,
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                getattr(errno, "ENOTSUP", errno.EINVAL),
            }
            if error in unsupported:
                _rename_directory_with_claim(source, destination)
                return
            raise OSError(
                error, f"atomic directory publication failed for {destination}"
            )
        return
    raise TensorEvaluationError(
        "atomic no-replace directory publication is unavailable on this platform"
    )


def publish_result_bundle(
    output: Path,
    *,
    result_name: str,
    result: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
) -> Mapping[str, str]:
    """Publish a no-clobber result directory by one same-filesystem rename."""

    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        ledger_bytes = b"".join(
            (canonical_json(row) + "\n").encode("utf-8") for row in episodes
        )
        ledger_path = stage / "episodes.jsonl"
        _atomic_bytes(ledger_path, ledger_bytes)
        ledger_sha256 = file_sha256(ledger_path)
        finalized = dict(result)
        finalized["episode_ledger"] = {
            "path": "episodes.jsonl",
            "sha256": ledger_sha256,
            "rows": len(episodes),
        }
        finalized["payload_sha256"] = payload_sha256(finalized)
        assert_strict_finite(finalized)
        result_path = stage / result_name
        atomic_json(result_path, finalized)
        digests = {
            result_name: file_sha256(result_path),
            "episodes.jsonl": ledger_sha256,
        }
        sums = "".join(
            f"{digest}  {name}\n" for name, digest in sorted(digests.items())
        )
        _atomic_bytes(stage / "SHA256SUMS", sums.encode("ascii"))
        rename_directory_no_replace(stage, output)
        return digests
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def verify_result_bundle(
    path: Path, *, result_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify a published bundle and return its result plus episode rows."""

    directory = path.resolve(strict=True)
    expected_files = {result_name, "episodes.jsonl", "SHA256SUMS"}
    actual_files = {candidate.name for candidate in directory.iterdir()}
    if actual_files != expected_files or any(
        not candidate.is_file() or candidate.is_symlink()
        for candidate in directory.iterdir()
    ):
        raise ValueError("result bundle artifact set is not exact regular files")
    sums_path = directory / "SHA256SUMS"
    lines = sums_path.read_text(encoding="ascii").splitlines()
    expected_names = {result_name, "episodes.jsonl"}
    observed: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError("SHA256SUMS line has an invalid shape")
        digest, name = parts
        require_sha256("artifact digest", digest)
        candidate = PurePosixPath(name)
        if candidate.is_absolute() or ".." in candidate.parts or name in observed:
            raise ValueError("SHA256SUMS contains an unsafe or duplicate path")
        observed[name] = digest
    if set(observed) != expected_names:
        raise ValueError("SHA256SUMS artifact set is not exact")
    for name, digest in observed.items():
        if file_sha256(directory / name) != digest:
            raise TensorEvaluationError(f"artifact SHA-256 mismatch for {name}")
    result = json.loads((directory / result_name).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result artifact must contain a mapping")
    assert_strict_finite(result)
    payload_digest = result.get("payload_sha256")
    require_sha256("payload_sha256", payload_digest)
    if (
        payload_sha256(
            {key: value for key, value in result.items() if key != "payload_sha256"}
        )
        != payload_digest
    ):
        raise TensorEvaluationError("result payload SHA-256 mismatch")
    episodes = []
    for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("episode ledger row must be a mapping")
        assert_strict_finite(row)
        episodes.append(row)
    ledger = result.get("episode_ledger")
    if not isinstance(ledger, Mapping) or ledger != {
        "path": "episodes.jsonl",
        "sha256": observed["episodes.jsonl"],
        "rows": len(episodes),
    }:
        raise TensorEvaluationError("episode ledger identity mismatch")
    return result, episodes


def _read_runtime_identity(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    payload = load_pinned_json(path, expected_sha256, name="runtime manifest")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": expected_sha256,
        "payload": payload,
    }


def _build_backend(
    spec: Mapping[str, Any],
    sequence: PinnedSequence,
    requests: tuple[Any, ...],
) -> Any:
    from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import (
        GpuNativeTensorBackendEnv,
    )

    source_manifest = load_pinned_json(
        Path(spec["source_manifest_path"]),
        spec["source_manifest_sha256"],
        name="source manifest",
    )
    source_pins = validate_source_manifest(source_manifest)
    se3_source = next(pin for pin in source_pins if pin["name"] == "se3_wam")

    return GpuNativeTensorBackendEnv(
        task_id=sequence.task_id,
        num_envs=spec["num_envs"],
        export_dir=spec["export_dir"],
        expected_gpu_uuid=spec["expected_gpu_uuid"],
        expected_se3_source_commit=se3_source["commit"],
        expected_se3_source_tree=se3_source["tree"],
        device_ordinal=spec["device_ordinal"],
        image_size=spec["image_size"],
        split=sequence.split,
        manifest_seed=sequence.manifest_seed,
        manifest_size=len(requests),
        manifest_requests=requests,
        manifest_sha256=sequence.manifest_sha256,
        task_quality_schema_version=sequence.task_quality_schema_version,
        task_quality_evaluator_backend_id=(sequence.task_quality_evaluator_backend_id),
    )


def execute_worker(
    spec: Mapping[str, Any],
    *,
    backend_factory: Callable[[Mapping[str, Any], PinnedSequence, tuple[Any, ...]], Any]
    | None = None,
) -> Mapping[str, Any]:
    """Execute one already-fresh worker and return its unpublished evidence."""

    if spec.get("schema_version") != WORKER_SPEC_SCHEMA:
        raise ValueError("worker spec schema mismatch")
    parent_start = spec.get("parent_process_start")
    if not isinstance(parent_start, Mapping):
        raise ValueError("worker spec lacks parent process identity")
    current = collect_process_identity()
    if current.pid == parent_start.get("pid") or current.parent_pid != parent_start.get(
        "pid"
    ):
        raise TensorEvaluationError("worker was not launched as a fresh direct child")
    observed_parent = collect_process_identity(current.parent_pid)
    if not process_identity_matches(parent_start, observed_parent):
        raise TensorEvaluationError(
            "parent PID was reused or its boot/start identity drifted"
        )
    child_start = current

    sequence_path = Path(spec["sequence_path"])
    sequence_payload = load_pinned_json(
        sequence_path,
        spec["sequence_sha256"],
        name="validation sequence",
    )
    sequence = PinnedSequence.from_payload(sequence_payload)
    start = spec["start_ordinal"]
    count = spec["episode_count"]
    num_envs = spec["num_envs"]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (count, num_envs)
        )
        or isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
    ):
        raise ValueError("worker ordinal/count/num_envs values are invalid")
    stop = start + count
    if stop > len(sequence.requests) or start % num_envs or count % num_envs:
        raise ValueError("worker ownership must be in-bounds full contiguous cohorts")
    source_manifest_path = Path(spec["source_manifest_path"])
    source_manifest = load_pinned_json(
        source_manifest_path,
        spec["source_manifest_sha256"],
        name="source manifest",
    )
    source_pins = validate_source_manifest(source_manifest)
    source_start = capture_source_snapshot(source_pins)
    runtime_path = Path(spec["runtime_manifest_path"])
    runtime_start = _read_runtime_identity(
        runtime_path, spec["runtime_manifest_sha256"]
    )
    policy_path = Path(spec["policy_path"])
    verify_file_pin(policy_path, spec["policy_sha256"], name="policy")

    torch_module = importlib.import_module("torch")
    if not torch_module.cuda.is_available():
        raise TensorEvaluationError("CUDA is unavailable; CPU fallback is forbidden")
    requests = build_reset_requests(sequence)
    make_backend = _build_backend if backend_factory is None else backend_factory
    env = make_backend(spec, sequence, requests)
    episodes: list[dict[str, Any]] = []
    allocated_steps = 0
    valid_steps = 0
    rollout_seconds = 0.0
    try:
        if env.api_version != sequence.api_version:
            raise TensorEvaluationError(
                "backend API version differs from pinned sequence"
            )
        stable_start = jsonable(env.stable_identity)
        validate_backend_runtime_identity(
            stable_start,
            runtime_payload=runtime_start["payload"],
            source_snapshot=source_start,
            sequence=sequence,
        )
        if env.manifest_sha256 != sequence.manifest_sha256:
            raise TensorEvaluationError("backend manifest differs from pinned sequence")
        export_start = jsonable(env.active_export_identity)
        if portable_export_identity(export_start) != dict(
            sequence.active_export_identity
        ):
            raise TensorEvaluationError(
                "active export identity differs from pinned sequence"
            )
        device_start = jsonable(env.device_identity_start)
        policy, policy_identity = load_device_policy(
            torch_module,
            policy_path,
            expected_sha256=spec["policy_sha256"],
            device=env.device,
        )
        policy_task = policy_identity["training_identity"].get("task")
        if policy_task is not None and policy_task != sequence.task_id:
            raise TensorEvaluationError("policy task differs from validation sequence")
        cursor = dict(env.manifest_state_dict())
        cursor["next_cohort_index"] = start // num_envs
        env.load_manifest_state_dict(cursor)
        from se3_wam.benchmark.task_quality import EpisodeQualitySummary

        for cohort_start in range(start, stop, num_envs):
            ordinals = tuple(range(cohort_start, cohort_start + num_envs))
            expected_rows = tuple(
                runtime_request_payload(sequence, ordinal) for ordinal in ordinals
            )
            cohort = run_device_cohort(
                env=env,
                policy=policy,
                expected_rows=expected_rows,
                expected_ordinals=ordinals,
                ledger_factory=lambda size, device: TorchEpisodeLedger(
                    torch_module, num_envs=size, device=device
                ),
                synchronize=lambda device: torch_module.cuda.synchronize(device),
                quality_type=EpisodeQualitySummary,
                quality_from_dict=EpisodeQualitySummary.from_dict,
            )
            episodes.extend(cohort.episodes)
            allocated_steps += cohort.allocated_steps
            valid_steps += cohort.valid_steps
            rollout_seconds += cohort.rollout_seconds

        cohort_count = count // num_envs
        expected_transport_checks = cohort_count * env.cohort_horizon_steps
        transport_receipt = env.last_transport_receipt
        if env.transport_checks != expected_transport_checks or not isinstance(
            transport_receipt, Mapping
        ):
            raise TensorEvaluationError(
                "step_device transport receipt count/identity is incomplete"
            )
        active_cohort_end = jsonable(env.active_cohort_identity)
        last_ordinals = list(range(stop - num_envs, stop))
        last_episode_ids = [
            runtime_request_payload(sequence, ordinal)["episode_id"]
            for ordinal in last_ordinals
        ]
        if not isinstance(active_cohort_end, Mapping) or (
            active_cohort_end.get("episode_ids") != last_episode_ids
            or active_cohort_end.get("manifest_ordinals") != last_ordinals
        ):
            raise TensorEvaluationError("active cohort end identity drifted")
        attested_end = env.attest_end()
        device_end = jsonable(env.device_identity_end)
        if device_end != jsonable(attested_end) or device_end != device_start:
            raise TensorEvaluationError(
                "physical device identity drifted during rollout"
            )
        stable_end = jsonable(env.stable_identity)
        export_end = jsonable(env.active_export_identity)
        if stable_end != stable_start or export_end != export_start:
            raise TensorEvaluationError(
                "backend stable/export identity drifted during rollout"
            )
        if portable_export_identity(export_end) != dict(
            sequence.active_export_identity
        ):
            raise TensorEvaluationError("active export digest drifted during rollout")
    finally:
        env.close()

    if [row["ordinal"] for row in episodes] != list(range(start, stop)):
        raise TensorEvaluationError("worker episode ownership/order diverged")
    if len({row["episode_id"] for row in episodes}) != len(episodes):
        raise TensorEvaluationError("worker produced duplicate episode ids")
    if file_sha256(policy_path) != spec["policy_sha256"]:
        raise TensorEvaluationError("policy changed during worker execution")
    if file_sha256(sequence_path) != spec["sequence_sha256"]:
        raise TensorEvaluationError("validation sequence changed during execution")
    source_end = capture_source_snapshot(source_pins)
    if source_end != source_start:
        raise TensorEvaluationError("source identity drifted during worker execution")
    runtime_end = _read_runtime_identity(runtime_path, spec["runtime_manifest_sha256"])
    if runtime_end != runtime_start:
        raise TensorEvaluationError("runtime manifest drifted during worker execution")
    child_end = collect_process_identity()
    if child_end != child_start:
        raise TensorEvaluationError(
            "child boot/start identity drifted during execution"
        )
    summary = summarize_episodes(
        episodes,
        allocated_steps=allocated_steps,
        valid_steps=valid_steps,
        rollout_seconds=rollout_seconds,
    )
    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete",
        "mode": spec["mode"],
        "claim_scope": {
            "science_gate_state": "not_run",
            "production_qualified": False,
            "statement": "Validation evidence only; no production-quality claim.",
        },
        "ownership": {
            "start_ordinal": start,
            "episode_count": count,
            "stop_ordinal_exclusive": stop,
        },
        "policy_identity": policy_identity,
        "sequence_identity": {
            "path": str(sequence_path.resolve(strict=True)),
            "sha256": spec["sequence_sha256"],
            "manifest_sha256": sequence.manifest_sha256,
            "task_id": sequence.task_id,
            "split": sequence.split,
            "manifest_seed": sequence.manifest_seed,
            "api_version": sequence.api_version,
            "request_count": len(sequence.requests),
        },
        "process_identity": {
            "parent_start": dict(parent_start),
            "child_start": asdict(child_start),
            "child_end": asdict(child_end),
        },
        "source_identity": {"start": source_start, "end": source_end},
        "runtime_identity": {"start": runtime_start, "end": runtime_end},
        "backend_identity": {
            "stable_start": stable_start,
            "stable_end": stable_end,
            "active_export_start": export_start,
            "active_export_end": export_end,
            "portable_active_export": dict(sequence.active_export_identity),
            "device_start": device_start,
            "device_end": device_end,
        },
        "data_plane": {
            "step_api": "step_device",
            "device_ledger": "preallocated_per_cohort",
            "hot_path_host_materializations": 0,
            "terminal_control_plane_materializations": cohort_count,
            "transport_checks": env.transport_checks,
            "expected_transport_checks": expected_transport_checks,
            "last_transport_receipt": jsonable(transport_receipt),
            "active_cohort_end": active_cohort_end,
        },
        "metrics": summary,
        "episodes": episodes,
    }


def validate_worker_spec(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the parent-to-child contract before either side uses it."""

    required = {
        "schema_version",
        "mode",
        "policy_path",
        "policy_sha256",
        "sequence_path",
        "sequence_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "runtime_manifest_path",
        "runtime_manifest_sha256",
        "export_dir",
        "expected_gpu_uuid",
        "device_ordinal",
        "image_size",
        "num_envs",
        "start_ordinal",
        "episode_count",
        "parent_process_start",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("worker spec keys do not match the released schema")
    if payload["schema_version"] != WORKER_SPEC_SCHEMA:
        raise ValueError("unsupported worker spec schema")
    if payload["mode"] not in {"validation", "rollout_shard"}:
        raise ValueError("worker mode is invalid")
    for name in (
        "policy_path",
        "sequence_path",
        "source_manifest_path",
        "runtime_manifest_path",
        "export_dir",
        "expected_gpu_uuid",
    ):
        _nonempty_string(name, payload[name])
    for name in (
        "policy_sha256",
        "sequence_sha256",
        "source_manifest_sha256",
        "runtime_manifest_sha256",
    ):
        require_sha256(name, payload[name])
    for name in ("image_size", "num_envs", "episode_count"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("device_ordinal", "start_ordinal"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    parent = payload["parent_process_start"]
    if not isinstance(parent, Mapping):
        raise ValueError("parent_process_start must be a mapping")
    ProcessIdentity(**parent)
    assert_strict_finite(payload)
    return dict(payload)


def launch_fresh_worker(
    spec: Mapping[str, Any],
    *,
    script_path: Path,
    timeout_s: float | None = None,
) -> Mapping[str, Any]:
    """Launch and validate a fresh worker without initializing CUDA in the parent."""

    validated = validate_worker_spec(spec)
    parent_start = ProcessIdentity(**validated["parent_process_start"])
    if parent_start != collect_process_identity():
        raise TensorEvaluationError("parent identity changed before worker launch")
    with tempfile.TemporaryDirectory(prefix="rlinf-tensor-eval-worker-") as temporary:
        root = Path(temporary)
        spec_path = root / "worker_spec.json"
        result_path = root / "worker_result.json"
        failure_path = root / "worker_failure.json"
        atomic_json(spec_path, validated)
        spec_sha256 = file_sha256(spec_path)
        command = [
            sys.executable,
            str(script_path.resolve(strict=True)),
            "--worker-spec",
            str(spec_path),
            "--expected-worker-spec-sha256",
            spec_sha256,
            "--worker-result",
            str(result_path),
            "--worker-failure",
            str(failure_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise TensorEvaluationError("fresh evaluation worker timed out") from exc
        if completed.returncode != 0:
            failure = None
            if failure_path.is_file():
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
            raise TensorEvaluationError(
                f"fresh evaluation worker failed closed: {failure or completed.stderr.strip()}"
            )
        if not result_path.is_file() or failure_path.exists():
            raise TensorEvaluationError(
                "fresh worker result/failure channel is inconsistent"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("fresh worker result must be a mapping")
        assert_strict_finite(result)
        process = result.get("process_identity")
        if (
            not isinstance(process, Mapping)
            or set(process) != {"parent_start", "child_start", "child_end"}
            or process.get("parent_start") != asdict(parent_start)
        ):
            raise TensorEvaluationError("fresh worker parent identity receipt drifted")
        child_start = process.get("child_start")
        child_end = process.get("child_end")
        if not isinstance(child_start, Mapping) or not isinstance(child_end, Mapping):
            raise TensorEvaluationError(
                "fresh worker child identity receipt is missing"
            )
        child_start_identity = ProcessIdentity(**child_start)
        child_end_identity = ProcessIdentity(**child_end)
        direct_parent_mismatch = (
            child_start_identity.identity_source != "windows_kernel_times"
            and child_start_identity.parent_pid != parent_start.pid
        )
        if (
            child_start_identity != child_end_identity
            or child_start_identity.pid == parent_start.pid
            or direct_parent_mismatch
            or child_start_identity.boot_id != parent_start.boot_id
        ):
            raise TensorEvaluationError(
                "fresh worker PID/boot/start receipt is invalid"
            )
        result["worker_stdio"] = {
            "stdout_bytes": len(completed.stdout.encode("utf-8")),
            "stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")
            ).hexdigest(),
            "stderr_bytes": len(completed.stderr.encode("utf-8")),
            "stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
        }
        return result


def launch_identity_probe(script_path: Path) -> Mapping[str, Any]:
    """Launch a dependency-free child used by host subprocess tests."""

    parent = collect_process_identity()
    completed = subprocess.run(
        [sys.executable, str(script_path.resolve(strict=True)), "--identity-probe"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.stderr:
        raise TensorEvaluationError("identity probe emitted stderr")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, Mapping):
        raise ValueError("identity probe payload must be a mapping")
    child = ProcessIdentity(**payload)
    direct_parent_mismatch = os.name != "nt" and child.parent_pid != parent.pid
    if (
        child.pid == parent.pid
        or direct_parent_mismatch
        or child.boot_id != parent.boot_id
    ):
        raise TensorEvaluationError("identity probe was not a fresh same-boot child")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--expected-policy-sha256")
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--expected-sequence-sha256")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--expected-runtime-manifest-sha256")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-envs", type=int, required=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-timeout-s", type=float)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-worker-spec-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-failure", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--identity-probe", action="store_true", help=argparse.SUPPRESS)
    return parser


def _required_parent_argument(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is None:
        raise ValueError(f"--{name.replace('_', '-')} is required")
    return value


def _worker_main(args: argparse.Namespace) -> int:
    for name in (
        "worker_spec",
        "expected_worker_spec_sha256",
        "worker_result",
        "worker_failure",
    ):
        _required_parent_argument(args, name)
    if args.worker_result.exists() or args.worker_failure.exists():
        raise FileExistsError("worker result/failure channel already exists")
    try:
        spec = load_pinned_json(
            args.worker_spec,
            args.expected_worker_spec_sha256,
            name="worker spec",
        )
        result = execute_worker(validate_worker_spec(spec))
        atomic_json(args.worker_result, result)
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": "rlinf-dynamic-benchmark-tensor-worker-failure-v0.1",
            "status": "failed_closed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "process_identity": asdict(collect_process_identity()),
        }
        atomic_json(args.worker_failure, failure)
        return 1


def _parent_main(args: argparse.Namespace) -> int:
    for name in (
        "policy",
        "expected_policy_sha256",
        "sequence",
        "expected_sequence_sha256",
        "source_manifest",
        "expected_source_manifest_sha256",
        "runtime_manifest",
        "expected_runtime_manifest_sha256",
        "export_dir",
        "expected_gpu_uuid",
        "num_envs",
        "output",
    ):
        _required_parent_argument(args, name)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sequence_payload = load_pinned_json(
        args.sequence,
        args.expected_sequence_sha256,
        name="validation sequence",
    )
    sequence = PinnedSequence.from_payload(sequence_payload)
    if len(sequence.requests) % args.num_envs:
        raise ValueError("validation sequence must contain full num_envs cohorts")
    verify_file_pin(args.policy, args.expected_policy_sha256, name="policy")
    load_pinned_json(
        args.source_manifest,
        args.expected_source_manifest_sha256,
        name="source manifest",
    )
    load_pinned_json(
        args.runtime_manifest,
        args.expected_runtime_manifest_sha256,
        name="runtime manifest",
    )
    parent_start = collect_process_identity()
    spec = {
        "schema_version": WORKER_SPEC_SCHEMA,
        "mode": "validation",
        "policy_path": str(args.policy.resolve(strict=True)),
        "policy_sha256": args.expected_policy_sha256,
        "sequence_path": str(args.sequence.resolve(strict=True)),
        "sequence_sha256": args.expected_sequence_sha256,
        "source_manifest_path": str(args.source_manifest.resolve(strict=True)),
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "runtime_manifest_path": str(args.runtime_manifest.resolve(strict=True)),
        "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
        "export_dir": str(args.export_dir.resolve(strict=True)),
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "device_ordinal": args.device_ordinal,
        "image_size": args.image_size,
        "num_envs": args.num_envs,
        "start_ordinal": 0,
        "episode_count": len(sequence.requests),
        "parent_process_start": asdict(parent_start),
    }
    result = dict(
        launch_fresh_worker(
            spec,
            script_path=Path(__file__),
            timeout_s=args.worker_timeout_s,
        )
    )
    if (
        result.get("schema_version") != EVALUATION_SCHEMA
        or result.get("status") != "complete"
        or result.get("mode") != "validation"
    ):
        raise TensorEvaluationError(
            "fresh validation worker returned an invalid result"
        )
    parent_end = collect_process_identity()
    if parent_end != parent_start:
        raise TensorEvaluationError(
            "parent boot/start identity drifted during evaluation"
        )
    result["process_identity"]["parent_end"] = asdict(parent_end)
    episodes = result.pop("episodes")
    publish_result_bundle(
        args.output,
        result_name="evaluation.json",
        result=result,
        episodes=episodes,
    )
    return 0


def main() -> int:
    """Run the parent CLI, the hidden worker, or the host identity probe."""

    args = _parser().parse_args()
    if args.identity_probe:
        print(canonical_json(asdict(collect_process_identity())))
        return 0
    if args.worker_spec is not None:
        return _worker_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
