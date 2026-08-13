# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Auditable current-state Planner control for the GPU-native P0 route.

The Planner may execute on the CPU. Physics, observations used for each
decision, executed actions, terminal evaluation, and review rendering remain
owned by the ``mjwarp_gpu_v1`` CUDA backend. Replay is a separate audit path:
it consumes a finalized tape and never calls a Planner.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

P0_GRASP_PLANNER_TAPE_SCHEMA_VERSION = "se3-wam-p0-grasp-planner-tape-v2"
GPU_NATIVE_BACKEND_ID = "mjwarp_gpu_v1"
E7_ACTION_WIDTH = 7
PHYSICS_STEPS_PER_CONTROL = 25
_VISUAL_COMPONENT_PREFIXES = ("rgb/", "depth_m/", "segmentation/")


class P0GraspPlannerError(RuntimeError):
    """Raised when the causal Planner/tape contract cannot be proven."""


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _jsonable(value.tolist()),
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    enum_value = _enum_value(value)
    if enum_value is not value:
        return _jsonable(enum_value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise P0GraspPlannerError("planner tape value is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _require_digest(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise P0GraspPlannerError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise P0GraspPlannerError(f"{name} must be a non-negative integer")
    return value


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise P0GraspPlannerError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise P0GraspPlannerError(f"{name} must be finite")
    return result


def observation_components(observation: Any) -> Mapping[str, str]:
    """Return validated component digests for one current host audit packet."""

    values = getattr(observation, "component_sha256", None)
    if not isinstance(values, Mapping) or not values:
        raise P0GraspPlannerError(
            "current observation does not expose component_sha256"
        )
    normalized = {}
    for name, digest in values.items():
        if not isinstance(name, str) or not name:
            raise P0GraspPlannerError("observation component name is invalid")
        normalized[name] = _require_digest(f"observation component {name}", digest)
    return MappingProxyType(dict(sorted(normalized.items())))


def causal_observation_fingerprint(observation: Any) -> str:
    """Fingerprint STATE inputs while keeping rendered media separately auditable."""

    components = observation_components(observation)
    causal = {
        name: digest
        for name, digest in components.items()
        if not name.startswith(_VISUAL_COMPONENT_PREFIXES)
    }
    if not causal:
        raise P0GraspPlannerError("current observation has no causal STATE components")
    return _digest(causal)


def planner_step_diagnostics(observation: Any, planner: Any) -> Mapping[str, Any]:
    """Record bounded Planner metadata without replacing the observation packet."""

    audit = getattr(planner, "planner_audit_snapshot", None)
    payload = audit() if callable(audit) else {}
    if not isinstance(payload, Mapping):
        raise P0GraspPlannerError("Planner audit snapshot must be a mapping")
    events = tuple(
        str(getattr(event, "name", event))
        for event in getattr(observation, "events_since_last_observation", ())
    )
    return MappingProxyType(
        {
            "phase": _enum_value(getattr(planner, "phase", None)),
            "planner_audit": dict(payload),
            "events": events,
        }
    )


def _health_snapshot(backend: Any) -> Mapping[str, int]:
    audit = backend.materialize_health_audit()
    required = (
        "overflow",
        "controller_valid",
        "driver_valid",
        "physics_step",
        "terminal",
    )
    if not isinstance(audit, Mapping) or any(name not in audit for name in required):
        raise P0GraspPlannerError("GPU health audit lacks required fields")
    result = {}
    for name in required:
        values = np.asarray(audit[name])
        if values.shape != (1,) or not np.all(np.isfinite(values)):
            raise P0GraspPlannerError(
                f"GPU health audit {name} is not one finite B=1 value"
            )
        scalar = float(values[0])
        if not scalar.is_integer() or scalar < 0:
            raise P0GraspPlannerError(f"GPU health audit {name} is not integral")
        result[name] = int(scalar)
    guards = (result["controller_valid"], result["driver_valid"])
    if (
        result["overflow"] != 0
        or result["terminal"] not in (0, 1)
        or any(value not in (0, 1) for value in guards)
        or (result["terminal"] == 0 and guards != (1, 1))
    ):
        raise P0GraspPlannerError("GPU finite/overflow/controller/driver gate failed")
    return MappingProxyType(result)


def _review_frame(backend: Any) -> Mapping[str, np.ndarray]:
    rows = backend.materialize_review_rgb((0,))
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise P0GraspPlannerError("GPU review materialization did not return B=1")
    row = rows[0]
    if tuple(row) != ("agentview", "robot0_eye_in_hand"):
        raise P0GraspPlannerError("GPU review materialization changed the camera set")
    normalized = {}
    for camera in ("agentview", "robot0_eye_in_hand"):
        frame = np.asarray(row[camera])
        if (
            frame.ndim != 3
            or frame.shape[-1] != 3
            or frame.dtype != np.uint8
            or not np.all(np.isfinite(frame))
        ):
            raise P0GraspPlannerError(
                f"GPU review frame {camera} has an invalid RGB layout"
            )
        normalized[camera] = np.array(frame, copy=True)
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class PlannerTapeIdentity:
    task_id: str
    backend_id: str
    num_envs: int
    manifest_sha256: str
    episode_ids: tuple[str, ...]
    manifest_ordinals: tuple[int, ...]
    seeds: tuple[int, ...]
    horizon_steps: int
    max_horizon_steps: int
    backend_identity_sha256: str
    observation_track: str = "state"
    action_mode: str = "E7"
    render_observations: bool = True
    physics_steps_per_control: int = PHYSICS_STEPS_PER_CONTROL

    def __post_init__(self) -> None:
        if self.task_id != "p0_grasp" or self.backend_id != GPU_NATIVE_BACKEND_ID:
            raise P0GraspPlannerError("Planner tape requires p0_grasp/mjwarp_gpu_v1")
        if self.num_envs != 1:
            raise P0GraspPlannerError("P0 E0 Planner tape requires B=1")
        _require_digest("manifest_sha256", self.manifest_sha256)
        _require_digest("backend_identity_sha256", self.backend_identity_sha256)
        if len(self.episode_ids) != 1 or not self.episode_ids[0]:
            raise P0GraspPlannerError("Planner tape requires one episode identity")
        if len(self.manifest_ordinals) != 1 or len(self.seeds) != 1:
            raise P0GraspPlannerError("reset identity does not match B=1")
        for value in (*self.manifest_ordinals, *self.seeds):
            _nonnegative_int("reset identity integer", value)
        _nonnegative_int("horizon_steps", self.horizon_steps)
        _nonnegative_int("max_horizon_steps", self.max_horizon_steps)
        if not 1 <= self.horizon_steps <= self.max_horizon_steps:
            raise P0GraspPlannerError(
                "Planner tape horizon is outside the backend bound"
            )
        if (
            self.observation_track != "state"
            or self.action_mode != "E7"
            or self.render_observations is not True
            or self.physics_steps_per_control != PHYSICS_STEPS_PER_CONTROL
        ):
            raise P0GraspPlannerError(
                "Planner tape requires STATE/E7 with independent GPU rendering and 25:1 clock"
            )

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class PlannerTapeEntry:
    lane: int
    episode_id: str
    task_id: str
    physics_step: int
    control_step: int
    policy_step: int
    observation_fingerprint_sha256: str
    observation_component_sha256: Mapping[str, str]
    action: tuple[float, ...]
    diagnostics: Mapping[str, Any]
    health_after: Mapping[str, int]
    done_after: bool

    def __post_init__(self) -> None:
        for name in ("lane", "physics_step", "control_step", "policy_step"):
            _nonnegative_int(name, getattr(self, name))
        if self.lane != 0 or self.task_id != "p0_grasp" or not self.episode_id:
            raise P0GraspPlannerError("P0 E0 tape entry identity is invalid")
        if self.control_step != self.policy_step:
            raise P0GraspPlannerError("observation control/policy clocks disagree")
        if self.physics_step != PHYSICS_STEPS_PER_CONTROL * self.control_step:
            raise P0GraspPlannerError(
                "pre-action observation is off the 500/20 Hz clock"
            )
        _require_digest(
            "observation_fingerprint_sha256",
            self.observation_fingerprint_sha256,
        )
        if not isinstance(self.observation_component_sha256, Mapping):
            raise P0GraspPlannerError("observation components must be a mapping")
        for name, digest in self.observation_component_sha256.items():
            _require_digest(f"observation component {name}", digest)
        values = tuple(_finite_float("action value", value) for value in self.action)
        if len(values) != E7_ACTION_WIDTH or any(
            value < -1.0 or value > 1.0 for value in values
        ):
            raise P0GraspPlannerError("tape action must be an E7 vector in [-1, 1]")
        if not isinstance(self.diagnostics, Mapping) or not isinstance(
            self.health_after, Mapping
        ):
            raise P0GraspPlannerError("tape diagnostics/health must be mappings")
        if type(self.done_after) is not bool:
            raise P0GraspPlannerError("done_after must be an exact boolean")
        object.__setattr__(self, "action", values)
        object.__setattr__(
            self,
            "observation_component_sha256",
            MappingProxyType(dict(self.observation_component_sha256)),
        )
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )
        object.__setattr__(
            self, "health_after", MappingProxyType(dict(self.health_after))
        )

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class ReviewFrameRecord:
    ordinal: int
    physics_step: int
    control_step: int
    policy_step: int
    scene_sha256: str
    wrist_sha256: str

    def __post_init__(self) -> None:
        for name in ("ordinal", "physics_step", "control_step", "policy_step"):
            _nonnegative_int(name, getattr(self, name))
        if self.control_step != self.policy_step or self.ordinal != self.control_step:
            raise P0GraspPlannerError("review frame clocks are not sequential")
        if self.physics_step != PHYSICS_STEPS_PER_CONTROL * self.control_step:
            raise P0GraspPlannerError("review frame is off the exact 500/20 Hz clock")
        _require_digest("scene_sha256", self.scene_sha256)
        _require_digest("wrist_sha256", self.wrist_sha256)

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


class ActionTrajectoryTape:
    """Append-only online tape finalized only at natural termination."""

    def __init__(
        self,
        identity: PlannerTapeIdentity,
        *,
        reset_health: Mapping[str, int],
    ) -> None:
        if not isinstance(identity, PlannerTapeIdentity):
            raise TypeError("identity must be a PlannerTapeIdentity")
        if not isinstance(reset_health, Mapping):
            raise TypeError("reset_health must be a mapping")
        self._identity = identity
        self._reset_health = MappingProxyType(dict(reset_health))
        self._entries: list[PlannerTapeEntry] = []
        self._review_frames: list[ReviewFrameRecord] = []
        self._finalized = False

    @property
    def identity(self) -> PlannerTapeIdentity:
        return self._identity

    @property
    def entries(self) -> tuple[PlannerTapeEntry, ...]:
        return tuple(self._entries)

    @property
    def review_frames(self) -> tuple[ReviewFrameRecord, ...]:
        return tuple(self._review_frames)

    @property
    def complete(self) -> bool:
        return (
            self._finalized
            and len(self._entries) == self._identity.horizon_steps
            and len(self._review_frames) == self._identity.horizon_steps + 1
            and self._entries[-1].done_after
            and not any(entry.done_after for entry in self._entries[:-1])
        )

    @property
    def sha256(self) -> str:
        return _digest(self.as_dict(include_sha256=False))

    def append_review_frame(
        self,
        *,
        physics_step: int,
        frame: Mapping[str, np.ndarray],
    ) -> None:
        if self._finalized:
            raise P0GraspPlannerError("planner tape is already finalized")
        ordinal = len(self._review_frames)
        if ordinal > self._identity.max_horizon_steps:
            raise P0GraspPlannerError("review frame tape exceeded the fixed horizon")
        self._review_frames.append(
            ReviewFrameRecord(
                ordinal=ordinal,
                physics_step=physics_step,
                control_step=ordinal,
                policy_step=ordinal,
                scene_sha256=_array_digest(frame["agentview"]),
                wrist_sha256=_array_digest(frame["robot0_eye_in_hand"]),
            )
        )

    def append(self, entry: PlannerTapeEntry) -> None:
        if self._finalized:
            raise P0GraspPlannerError("planner tape is already finalized")
        expected_step = len(self._entries)
        if expected_step >= self._identity.max_horizon_steps:
            raise P0GraspPlannerError("planner tape reached the fixed backend horizon")
        if (
            entry.policy_step != expected_step
            or entry.episode_id != self._identity.episode_ids[0]
            or any(value.done_after for value in self._entries)
        ):
            raise P0GraspPlannerError(
                "P0 E0 tape order/identity/terminal state differs"
            )
        self._entries.append(entry)

    def finalize(self, horizon_steps: int) -> None:
        _nonnegative_int("natural horizon_steps", horizon_steps)
        if not 1 <= horizon_steps <= self._identity.max_horizon_steps:
            raise P0GraspPlannerError("natural horizon is outside the backend bound")
        if len(self._entries) != horizon_steps or not self._entries[-1].done_after:
            raise P0GraspPlannerError(
                "natural horizon differs from the action/done tape"
            )
        if len(self._review_frames) != horizon_steps + 1:
            raise P0GraspPlannerError(
                "review frames do not cover reset through terminal"
            )
        self._identity = replace(self._identity, horizon_steps=horizon_steps)
        self._finalized = True

    def entry_for_step(self, policy_step: int) -> PlannerTapeEntry:
        _nonnegative_int("policy_step", policy_step)
        if policy_step >= len(self._entries):
            raise P0GraspPlannerError(f"tape has no row for policy_step {policy_step}")
        entry = self._entries[policy_step]
        if entry.policy_step != policy_step:
            raise P0GraspPlannerError("tape row order changed")
        return entry

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": P0_GRASP_PLANNER_TAPE_SCHEMA_VERSION,
            "identity": self._identity.as_dict(),
            "reset_health": _jsonable(self._reset_health),
            "entries": [entry.as_dict() for entry in self._entries],
            "review_frames": [frame.as_dict() for frame in self._review_frames],
            "complete": self.complete,
        }
        if include_sha256:
            payload["sha256"] = self.sha256
        return payload

    def action_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": P0_GRASP_PLANNER_TAPE_SCHEMA_VERSION,
            "identity": self._identity.as_dict(),
            "entries": [
                {
                    "lane": entry.lane,
                    "episode_id": entry.episode_id,
                    "task_id": entry.task_id,
                    "physics_step": entry.physics_step,
                    "control_step": entry.control_step,
                    "policy_step": entry.policy_step,
                    "action": list(entry.action),
                }
                for entry in self._entries
            ],
            "complete": self.complete,
        }
        payload["sha256"] = _digest(payload)
        return payload


def _validate_backend(backend: Any) -> None:
    backend_id = getattr(backend, "backend_id", None)
    if backend_id is None:
        backend_id = getattr(getattr(backend, "provenance", None), "backend_id", None)
    if backend_id != GPU_NATIVE_BACKEND_ID:
        raise P0GraspPlannerError("P0 Planner requires backend=mjwarp_gpu_v1")
    device = getattr(backend, "device", None)
    if getattr(device, "type", None) != "cuda" and not str(device).startswith("cuda"):
        raise P0GraspPlannerError("P0 Planner requires a CUDA data plane")
    if getattr(backend, "task_id", None) != "p0_grasp":
        raise P0GraspPlannerError("P0 Planner control is restricted to p0_grasp")
    if getattr(backend, "num_envs", None) != 1:
        raise P0GraspPlannerError("P0 Planner E0 requires num_envs=1")
    if getattr(backend, "observation_track", None) != "state":
        raise P0GraspPlannerError("P0 Planner E0 requires the frozen STATE track")
    if getattr(backend, "render_observations", None) is not True:
        raise P0GraspPlannerError("P0 Planner E0 requires independent GPU rendering")


def _tape_identity_from_reset(backend: Any, reset: Any) -> PlannerTapeIdentity:
    _validate_backend(backend)
    stable_identity = getattr(backend, "stable_identity", None)
    if not isinstance(stable_identity, Mapping):
        raise P0GraspPlannerError("GPU backend does not expose a stable run identity")
    return PlannerTapeIdentity(
        task_id="p0_grasp",
        backend_id=GPU_NATIVE_BACKEND_ID,
        num_envs=1,
        manifest_sha256=reset.manifest_sha256,
        episode_ids=tuple(reset.episode_ids),
        manifest_ordinals=tuple(reset.manifest_ordinals),
        seeds=tuple(reset.seeds),
        horizon_steps=int(backend.cohort_horizon_steps),
        max_horizon_steps=int(backend.cohort_horizon_steps),
        backend_identity_sha256=_digest(stable_identity),
    )


def _command_from_planner(planner: Any, observation: Any) -> np.ndarray:
    command = planner.act(observation)
    from se3_wam.benchmark.api import ActionCommand
    from se3_wam.benchmark.contracts import ActionMode

    if type(command) is not ActionCommand or command.mode is not ActionMode.E7:
        raise P0GraspPlannerError(
            "online P0 Planner must emit the exact E7 ActionCommand"
        )
    if command.policy_step != observation.policy_step:
        raise P0GraspPlannerError(
            "Planner action policy_step differs from current observation"
        )
    values = np.asarray(command.values, dtype=np.float32)
    if values.shape != (E7_ACTION_WIDTH,) or not np.all(np.isfinite(values)):
        raise P0GraspPlannerError("online P0 Planner action must be a finite E7 vector")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise P0GraspPlannerError("online P0 Planner action must lie in [-1, 1]")
    return np.array(values, dtype=np.float32, copy=True)


def _read_done(value: Any) -> bool:
    detached = value.detach() if hasattr(value, "detach") else value
    host = detached.cpu() if hasattr(detached, "cpu") else detached
    values = host.numpy() if hasattr(host, "numpy") else host
    normalized = np.asarray(values)
    if normalized.shape != (1,) or int(normalized[0]) not in (0, 1):
        raise P0GraspPlannerError("GPU done mask is not a one-lane boolean")
    return bool(normalized[0])


class CurrentStatePlannerAdapter:
    """Run one CPU Planner decision per current GPU STATE observation."""

    def __init__(self, backend: Any, planner: Any) -> None:
        _validate_backend(backend)
        if not callable(getattr(planner, "act", None)):
            raise TypeError("P0 Planner must expose act(observation)")
        self._backend = backend
        self._planner = planner
        self._tape: ActionTrajectoryTape | None = None
        self._scene_frames: list[np.ndarray] = []
        self._wrist_frames: list[np.ndarray] = []

    @property
    def tape(self) -> ActionTrajectoryTape:
        if self._tape is None:
            raise P0GraspPlannerError("Planner adapter has no active reset cohort")
        return self._tape

    @property
    def scene_frames(self) -> tuple[np.ndarray, ...]:
        return tuple(self._scene_frames)

    @property
    def wrist_frames(self) -> tuple[np.ndarray, ...]:
        return tuple(self._wrist_frames)

    def _append_frame(self, frame: Mapping[str, np.ndarray], physics_step: int) -> None:
        self.tape.append_review_frame(physics_step=physics_step, frame=frame)
        self._scene_frames.append(np.array(frame["agentview"], copy=True))
        self._wrist_frames.append(np.array(frame["robot0_eye_in_hand"], copy=True))

    def reset(self) -> Any:
        reset = self._backend.reset()
        reset_planner = getattr(self._planner, "reset", None)
        if callable(reset_planner):
            reset_planner()
        reset_health = _health_snapshot(self._backend)
        if reset_health["physics_step"] != 0:
            raise P0GraspPlannerError("GPU reset did not start at physics step zero")
        self._tape = ActionTrajectoryTape(
            _tape_identity_from_reset(self._backend, reset),
            reset_health=reset_health,
        )
        self._scene_frames = []
        self._wrist_frames = []
        self._append_frame(_review_frame(self._backend), physics_step=0)
        return reset

    def step(self) -> Any:
        tape = self.tape
        if len(tape.entries) >= tape.identity.max_horizon_steps:
            raise P0GraspPlannerError("Planner adapter reached the backend horizon")
        observations = self._backend.materialize_teacher_observations((0,))
        if len(observations) != 1:
            raise P0GraspPlannerError("current-state audit did not return one lane")
        observation = observations[0]
        expected_step = len(tape.entries)
        if (
            observation.task_id != "p0_grasp"
            or observation.episode_id != tape.identity.episode_ids[0]
            or observation.physics_step != PHYSICS_STEPS_PER_CONTROL * expected_step
            or observation.control_step != expected_step
            or observation.policy_step != expected_step
        ):
            raise P0GraspPlannerError(
                "current observation identity/clock differs from the tape"
            )
        components = observation_components(observation)
        values = _command_from_planner(self._planner, observation)
        try:
            torch = importlib.import_module("torch")
            device_action = torch.as_tensor(
                values.reshape(1, E7_ACTION_WIDTH),
                dtype=torch.float32,
                device=self._backend.device,
            )
        except (ImportError, TypeError, ValueError, RuntimeError) as exc:
            raise P0GraspPlannerError(
                "CPU Planner action could not reach the CUDA E7 seam"
            ) from exc
        result = self._backend.step_device(device_action)
        done = _read_done(result.done)
        health = _health_snapshot(self._backend)
        upper = PHYSICS_STEPS_PER_CONTROL * (expected_step + 1)
        if health["physics_step"] != upper:
            raise P0GraspPlannerError(
                "GPU step did not execute exactly 25 physics steps"
            )
        if health["terminal"] != int(done):
            raise P0GraspPlannerError(
                "GPU done mask and terminal health audit disagree"
            )
        self._append_frame(
            _review_frame(self._backend),
            physics_step=health["physics_step"],
        )
        tape.append(
            PlannerTapeEntry(
                lane=0,
                episode_id=observation.episode_id,
                task_id=observation.task_id,
                physics_step=observation.physics_step,
                control_step=observation.control_step,
                policy_step=observation.policy_step,
                observation_fingerprint_sha256=causal_observation_fingerprint(
                    observation
                ),
                observation_component_sha256=components,
                action=tuple(float(value) for value in values),
                diagnostics=planner_step_diagnostics(observation, self._planner),
                health_after=health,
                done_after=done,
            )
        )
        return result

    def run_natural_termination(self) -> tuple[Any, tuple[Any, ...]]:
        last_result = None
        for _ in range(self.tape.identity.max_horizon_steps):
            last_result = self.step()
            if self.tape.entries[-1].done_after:
                break
        else:
            raise P0GraspPlannerError(
                "P0 Planner did not reach natural termination at the fixed horizon"
            )
        natural_steps = len(self.tape.entries)
        self.tape.finalize(natural_steps)
        rows = self._backend.materialize_terminal_ledger_once(
            (0,),
            self.tape.identity.episode_ids,
        )
        if (
            len(rows) != 1
            or rows[0].lane != 0
            or rows[0].episode_id != self.tape.identity.episode_ids[0]
            or rows[0].task_id != "p0_grasp"
            or rows[0].control_step != natural_steps
            or rows[0].policy_step != natural_steps
            or not PHYSICS_STEPS_PER_CONTROL * (natural_steps - 1)
            < rows[0].physics_step
            <= PHYSICS_STEPS_PER_CONTROL * natural_steps
        ):
            raise P0GraspPlannerError("terminal ledger identity differs from the tape")
        return last_result, rows


@dataclass(frozen=True)
class ReplayStepReceipt:
    policy_step: int
    observation_fingerprint_sha256: str
    action_sha256: str
    physics_step_after: int
    done_after: bool


@dataclass(frozen=True)
class ReplayReceipt:
    steps: tuple[ReplayStepReceipt, ...]
    terminal_rows: tuple[Any, ...]
    backend_identity_sha256: str
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def replay_action_trajectory(
    backend: Any,
    tape: ActionTrajectoryTape,
) -> ReplayReceipt:
    """Replay one complete tape on a fresh backend without calling a Planner."""

    _validate_backend(backend)
    if not isinstance(tape, ActionTrajectoryTape) or not tape.complete:
        raise P0GraspPlannerError(
            "replay requires one complete natural-termination tape"
        )
    preview = tuple(request.episode_id for request in backend.next_requests())
    if preview != tape.identity.episode_ids:
        raise P0GraspPlannerError(
            "replay backend is not at the recorded manifest cursor"
        )
    reset = backend.reset()
    actual = _tape_identity_from_reset(backend, reset)
    expected = replace(actual, horizon_steps=tape.identity.horizon_steps)
    if expected != tape.identity:
        raise P0GraspPlannerError(
            "replay reset identity differs from the recorded tape"
        )
    reset_health = _health_snapshot(backend)
    if dict(reset_health) != dict(tape._reset_health):
        raise P0GraspPlannerError("replay reset health differs from the online tape")
    # Match the online adapter's reset boundary exactly.  The review copy is a
    # CUDA synchronization point even though rendered media are excluded from
    # the causal STATE fingerprint.
    _review_frame(backend)

    torch = importlib.import_module("torch")
    receipts = []
    for policy_step in range(tape.identity.horizon_steps):
        observations = backend.materialize_teacher_observations((0,))
        if len(observations) != 1:
            raise P0GraspPlannerError(
                "replay observation audit did not return one lane"
            )
        observation = observations[0]
        entry = tape.entry_for_step(policy_step)
        observed_fingerprint = causal_observation_fingerprint(observation)
        if (
            observation.episode_id != entry.episode_id
            or observation.policy_step != policy_step
            or observed_fingerprint != entry.observation_fingerprint_sha256
        ):
            raise P0GraspPlannerError(
                f"replay observation fingerprint differs at policy_step {policy_step}"
            )
        action_values = np.asarray(entry.action, dtype=np.float32).reshape(
            1, E7_ACTION_WIDTH
        )
        action = torch.as_tensor(
            action_values,
            dtype=torch.float32,
            device=backend.device,
        )
        result = backend.step_device(action)
        done = _read_done(result.done)
        health = _health_snapshot(backend)
        if done != entry.done_after or dict(health) != dict(entry.health_after):
            raise P0GraspPlannerError(
                f"replay step result differs at policy_step {policy_step}"
            )
        # Preserve the online step -> health -> review ordering before the next
        # current-state observation is materialized.
        _review_frame(backend)
        receipts.append(
            ReplayStepReceipt(
                policy_step=policy_step,
                observation_fingerprint_sha256=observed_fingerprint,
                action_sha256=_array_digest(action_values),
                physics_step_after=health["physics_step"],
                done_after=done,
            )
        )
    if not receipts[-1].done_after:
        raise P0GraspPlannerError("replay did not reproduce natural termination")
    rows = backend.materialize_terminal_ledger_once((0,), tape.identity.episode_ids)
    if (
        len(rows) != 1
        or rows[0].lane != 0
        or rows[0].episode_id != tape.identity.episode_ids[0]
        or rows[0].task_id != "p0_grasp"
        or rows[0].control_step != tape.identity.horizon_steps
        or rows[0].policy_step != tape.identity.horizon_steps
        or not PHYSICS_STEPS_PER_CONTROL * (tape.identity.horizon_steps - 1)
        < rows[0].physics_step
        <= PHYSICS_STEPS_PER_CONTROL * tape.identity.horizon_steps
    ):
        raise P0GraspPlannerError("replay terminal ledger identity differs")
    backend_identity_sha256 = _digest(backend.stable_identity)
    if backend_identity_sha256 != tape.identity.backend_identity_sha256:
        raise P0GraspPlannerError("replay backend identity changed after execution")
    return ReplayReceipt(
        steps=tuple(receipts),
        terminal_rows=tuple(rows),
        backend_identity_sha256=backend_identity_sha256,
        passed=True,
    )


__all__ = [
    "ActionTrajectoryTape",
    "CurrentStatePlannerAdapter",
    "E7_ACTION_WIDTH",
    "GPU_NATIVE_BACKEND_ID",
    "P0GraspPlannerError",
    "P0_GRASP_PLANNER_TAPE_SCHEMA_VERSION",
    "PlannerTapeEntry",
    "PlannerTapeIdentity",
    "ReplayReceipt",
    "ReplayStepReceipt",
    "ReviewFrameRecord",
    "causal_observation_fingerprint",
    "observation_components",
    "planner_step_diagnostics",
    "replay_action_trajectory",
]
