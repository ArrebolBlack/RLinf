# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Auditable current-state Planner control for the GPU-native P0 route.

The Planner is allowed to execute on CPU.  Physics, observations used for the
decision, actions executed by the environment, and terminal state remain on the
``mjwarp_gpu_v1`` CUDA backend.  Replay is an explicitly separate audit path:
it never calls a Planner and is never used to produce online actions.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np


P0_GRASP_PLANNER_TAPE_SCHEMA_VERSION = "se3-wam-p0-grasp-planner-tape-v1"
GPU_NATIVE_BACKEND_ID = "mjwarp_gpu_v1"
E7_ACTION_WIDTH = 7


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    enum_value = _enum_value(value)
    if enum_value is not value:
        return _jsonable(enum_value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
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


def observation_fingerprint(observation: Any) -> str:
    """Return the canonical fingerprint of one current host audit packet."""

    fingerprint = getattr(observation, "fingerprint_sha256", None)
    if isinstance(fingerprint, str):
        return _require_digest("observation fingerprint", fingerprint)
    components = getattr(observation, "component_sha256", None)
    if isinstance(components, Mapping):
        return _digest(components)
    if is_dataclass(observation):
        return _digest({field.name: getattr(observation, field.name) for field in fields(observation)})
    raise P0GraspPlannerError(
        "current observation does not expose fingerprint_sha256 or component_sha256"
    )


def planner_step_diagnostics(observation: Any, planner: Any) -> Mapping[str, Any]:
    """Record bounded causal metadata without replacing the observation packet."""

    audit = getattr(planner, "planner_audit_snapshot", None)
    audit_payload = audit() if callable(audit) else {}
    if not isinstance(audit_payload, Mapping):
        raise P0GraspPlannerError("Planner audit snapshot must be a mapping")
    events = tuple(
        str(getattr(event, "name", event))
        for event in getattr(observation, "events_since_last_observation", ())
    )
    components = getattr(observation, "component_sha256", None)
    return MappingProxyType(
        {
            "phase": _enum_value(getattr(planner, "phase", None)),
            "planner_audit": dict(audit_payload),
            "observation_components_sha256": (
                None if not isinstance(components, Mapping) else _digest(components)
            ),
            "events": events,
        }
    )


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
    render_visual: bool
    observation_track: str = "state"
    action_mode: str = "E7"

    def __post_init__(self) -> None:
        if self.task_id != "p0_grasp":
            raise P0GraspPlannerError("Planner tape identity is restricted to p0_grasp")
        if self.backend_id != GPU_NATIVE_BACKEND_ID:
            raise P0GraspPlannerError("Planner tape requires mjwarp_gpu_v1")
        if self.num_envs != 1:
            raise P0GraspPlannerError("P0 E0 Planner tape requires B=1")
        _require_digest("manifest_sha256", self.manifest_sha256)
        _require_digest("backend_identity_sha256", self.backend_identity_sha256)
        if len(self.episode_ids) != self.num_envs:
            raise P0GraspPlannerError("episode_ids do not match num_envs")
        if len(self.manifest_ordinals) != self.num_envs or len(self.seeds) != self.num_envs:
            raise P0GraspPlannerError("reset identity does not match num_envs")
        if any(not isinstance(value, str) or not value for value in self.episode_ids):
            raise P0GraspPlannerError("episode_ids must be non-empty strings")
        for value in (*self.manifest_ordinals, *self.seeds):
            _nonnegative_int("reset identity integer", value)
        _nonnegative_int("horizon_steps", self.horizon_steps)
        _nonnegative_int("max_horizon_steps", self.max_horizon_steps)
        if not 1 <= self.horizon_steps <= self.max_horizon_steps:
            raise P0GraspPlannerError("Planner tape horizon is outside the backend bound")
        if self.render_visual is not True:
            raise P0GraspPlannerError("P0 E0 Planner tape requires GPU visual rendering")
        if self.observation_track != "state" or self.action_mode != "E7":
            raise P0GraspPlannerError("Planner tape requires the frozen STATE/E7 contract")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "backend_id": self.backend_id,
            "num_envs": self.num_envs,
            "manifest_sha256": self.manifest_sha256,
            "episode_ids": list(self.episode_ids),
            "manifest_ordinals": list(self.manifest_ordinals),
            "seeds": list(self.seeds),
            "horizon_steps": self.horizon_steps,
            "max_horizon_steps": self.max_horizon_steps,
            "backend_identity_sha256": self.backend_identity_sha256,
            "render_visual": self.render_visual,
            "observation_track": self.observation_track,
            "action_mode": self.action_mode,
        }


@dataclass(frozen=True)
class PlannerTapeEntry:
    lane: int
    episode_id: str
    task_id: str
    policy_step: int
    observation_fingerprint_sha256: str
    action: tuple[float, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonnegative_int("lane", self.lane)
        _nonnegative_int("policy_step", self.policy_step)
        if self.lane != 0 or self.task_id != "p0_grasp":
            raise P0GraspPlannerError("P0 E0 tape entries require lane 0 and task p0_grasp")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise P0GraspPlannerError("tape entry episode_id must be non-empty")
        _require_digest("observation_fingerprint_sha256", self.observation_fingerprint_sha256)
        values = tuple(_finite_float("action value", value) for value in self.action)
        if len(values) != E7_ACTION_WIDTH or any(value < -1.0 or value > 1.0 for value in values):
            raise P0GraspPlannerError("tape entry action must be an E7 vector in [-1, 1]")
        if not isinstance(self.diagnostics, Mapping):
            raise P0GraspPlannerError("tape entry diagnostics must be a mapping")
        object.__setattr__(self, "action", values)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "policy_step": self.policy_step,
            "observation_fingerprint_sha256": self.observation_fingerprint_sha256,
            "action": list(self.action),
            "diagnostics": _jsonable(self.diagnostics),
        }


class ActionTrajectoryTape:
    """Append-only online tape that can be finalized at natural termination."""

    def __init__(self, identity: PlannerTapeIdentity) -> None:
        if not isinstance(identity, PlannerTapeIdentity):
            raise TypeError("identity must be a PlannerTapeIdentity")
        self._identity = identity
        self._entries: list[PlannerTapeEntry] = []
        self._finalized = False

    @property
    def identity(self) -> PlannerTapeIdentity:
        return self._identity

    @property
    def entries(self) -> tuple[PlannerTapeEntry, ...]:
        return tuple(self._entries)

    @property
    def complete(self) -> bool:
        return self._finalized and len(self._entries) == self._identity.horizon_steps

    @property
    def sha256(self) -> str:
        return _digest(self.as_dict(include_sha256=False))

    def append(self, entry: PlannerTapeEntry) -> None:
        if self._finalized:
            raise P0GraspPlannerError("planner tape is already finalized")
        expected_step = len(self._entries)
        if expected_step >= self._identity.max_horizon_steps:
            raise P0GraspPlannerError("planner tape reached the fixed backend horizon")
        if entry.policy_step != expected_step or entry.episode_id != self._identity.episode_ids[0]:
            raise P0GraspPlannerError("P0 E0 tape entry order or identity differs")
        self._entries.append(entry)

    def finalize(self, horizon_steps: int) -> None:
        _nonnegative_int("natural horizon_steps", horizon_steps)
        if not 1 <= horizon_steps <= self._identity.max_horizon_steps:
            raise P0GraspPlannerError("natural horizon is outside the backend bound")
        if len(self._entries) != horizon_steps:
            raise P0GraspPlannerError("natural horizon differs from the recorded action count")
        self._identity = replace(self._identity, horizon_steps=horizon_steps)
        self._finalized = True

    def entries_for_step(self, policy_step: int) -> tuple[PlannerTapeEntry, ...]:
        _nonnegative_int("policy_step", policy_step)
        rows = tuple(entry for entry in self._entries if entry.policy_step == policy_step)
        if len(rows) != 1:
            raise P0GraspPlannerError(f"P0 E0 tape has no unique row for policy_step {policy_step}")
        return rows

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": P0_GRASP_PLANNER_TAPE_SCHEMA_VERSION,
            "identity": self._identity.as_dict(),
            "entries": [entry.as_dict() for entry in self._entries],
            "complete": self.complete,
        }
        if include_sha256:
            payload["sha256"] = self.sha256
        return payload

    def action_dict(self) -> dict[str, Any]:
        payload = self.as_dict(include_sha256=False)
        payload["entries"] = [
            {
                "lane": entry.lane,
                "episode_id": entry.episode_id,
                "task_id": entry.task_id,
                "policy_step": entry.policy_step,
                "action": list(entry.action),
            }
            for entry in self._entries
        ]
        payload["sha256"] = _digest(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionTrajectoryTape":
        if not isinstance(payload, Mapping):
            raise P0GraspPlannerError("planner tape payload must be a mapping")
        if payload.get("schema_version") != P0_GRASP_PLANNER_TAPE_SCHEMA_VERSION:
            raise P0GraspPlannerError("planner tape schema version differs")
        identity_payload = payload.get("identity")
        if not isinstance(identity_payload, Mapping):
            raise P0GraspPlannerError("planner tape identity is missing")
        identity = PlannerTapeIdentity(
            task_id=identity_payload.get("task_id"),
            backend_id=identity_payload.get("backend_id"),
            num_envs=identity_payload.get("num_envs"),
            manifest_sha256=identity_payload.get("manifest_sha256"),
            episode_ids=tuple(identity_payload.get("episode_ids", ())),
            manifest_ordinals=tuple(identity_payload.get("manifest_ordinals", ())),
            seeds=tuple(identity_payload.get("seeds", ())),
            horizon_steps=identity_payload.get("horizon_steps"),
            max_horizon_steps=identity_payload.get("max_horizon_steps", identity_payload.get("horizon_steps")),
            backend_identity_sha256=identity_payload.get("backend_identity_sha256"),
            render_visual=identity_payload.get("render_visual"),
            observation_track=identity_payload.get("observation_track", "state"),
            action_mode=identity_payload.get("action_mode", "E7"),
        )
        tape = cls(identity)
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise P0GraspPlannerError("planner tape entries must be a list")
        for raw in entries:
            if not isinstance(raw, Mapping):
                raise P0GraspPlannerError("planner tape entry must be a mapping")
            tape.append(
                PlannerTapeEntry(
                    lane=raw.get("lane"),
                    episode_id=raw.get("episode_id"),
                    task_id=raw.get("task_id"),
                    policy_step=raw.get("policy_step"),
                    observation_fingerprint_sha256=raw.get(
                        "observation_fingerprint_sha256", _digest(raw.get("action"))
                    ),
                    action=tuple(raw.get("action", ())),
                    diagnostics=raw.get("diagnostics", {}),
                )
            )
        if payload.get("complete") is True:
            tape._finalized = True
        supplied_sha = payload.get("sha256")
        if supplied_sha is not None and _require_digest("planner tape sha256", supplied_sha) != tape.sha256:
            raise P0GraspPlannerError("planner tape SHA-256 does not match its canonical payload")
        return tape


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
    if getattr(backend, "render_visual", False) is not True:
        raise P0GraspPlannerError("P0 Planner E0 requires the GPU visual scene/wrist render seam")
    if getattr(backend, "num_envs", None) != 1:
        raise P0GraspPlannerError("P0 Planner E0 requires num_envs=1")


def _tape_identity_from_reset(backend: Any, reset: Any) -> PlannerTapeIdentity:
    _validate_backend(backend)
    stable_identity = getattr(backend, "stable_identity", None)
    if not isinstance(stable_identity, Mapping):
        stable_identity = {
            "backend_id": GPU_NATIVE_BACKEND_ID,
            "task_id": "p0_grasp",
            "render_visual": True,
        }
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
        render_visual=True,
    )


def _command_from_planner(planner: Any, observation: Any) -> np.ndarray:
    command = planner.act(observation) if callable(getattr(planner, "act", None)) else planner(observation)
    from se3_wam.benchmark.api import ActionCommand
    from se3_wam.benchmark.contracts import ActionMode

    if type(command) is not ActionCommand or command.mode is not ActionMode.E7:
        raise P0GraspPlannerError("online P0 Planner must emit the exact E7 ActionCommand")
    if command.policy_step != observation.policy_step:
        raise P0GraspPlannerError("Planner action policy_step differs from current observation")
    values = np.asarray(command.values, dtype=np.float32)
    if values.shape != (E7_ACTION_WIDTH,) or not np.all(np.isfinite(values)):
        raise P0GraspPlannerError("online P0 Planner action must be a finite E7 vector")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise P0GraspPlannerError("online P0 Planner action must lie in [-1, 1]")
    return np.array(values, dtype=np.float32, copy=True)


def _read_done(value: Any) -> bool:
    value = value.detach() if hasattr(value, "detach") else value
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    values = np.asarray(value)
    if values.shape != (1,) or int(values[0]) not in (0, 1):
        raise P0GraspPlannerError("GPU done mask is not a one-lane boolean")
    return bool(values[0])


class CurrentStatePlannerAdapter:
    """Run one CPU Planner decision per current GPU observation."""

    def __init__(self, backend: Any, planner: Any) -> None:
        _validate_backend(backend)
        if not callable(getattr(planner, "act", None)):
            raise TypeError("P0 Planner must expose act(observation)")
        self._backend = backend
        self._planner = planner
        self._tape: ActionTrajectoryTape | None = None
        self._observation_history: list[Any] = []
        self._last_observation: Any | None = None

    @property
    def tape(self) -> ActionTrajectoryTape:
        if self._tape is None:
            raise P0GraspPlannerError("Planner adapter has no active reset cohort")
        return self._tape

    @property
    def observation_history(self) -> tuple[Any, ...]:
        return tuple(self._observation_history)

    @property
    def last_observation(self) -> Any:
        if self._last_observation is None:
            raise P0GraspPlannerError("Planner adapter has no current observation")
        return self._last_observation

    def reset(self) -> Any:
        reset = self._backend.reset()
        reset_planner = getattr(self._planner, "reset", None)
        if callable(reset_planner):
            reset_planner()
        self._tape = ActionTrajectoryTape(_tape_identity_from_reset(self._backend, reset))
        self._observation_history = []
        self._last_observation = None
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
            or observation.control_step != expected_step
            or observation.policy_step != expected_step
        ):
            raise P0GraspPlannerError("current observation identity/clock differs from the tape")
        self._last_observation = observation
        self._observation_history.append(observation)
        values = _command_from_planner(self._planner, observation)
        try:
            torch = importlib.import_module("torch")
            device_action = torch.as_tensor(
                values.reshape(1, E7_ACTION_WIDTH),
                dtype=torch.float32,
                device=self._backend.device,
            )
        except (ImportError, TypeError, ValueError, RuntimeError) as exc:
            raise P0GraspPlannerError("CPU Planner action could not reach the CUDA E7 seam") from exc
        result = self._backend.step_device(device_action)
        tape.append(
            PlannerTapeEntry(
                lane=0,
                episode_id=observation.episode_id,
                task_id=observation.task_id,
                policy_step=observation.policy_step,
                observation_fingerprint_sha256=observation_fingerprint(observation),
                action=tuple(float(value) for value in values),
                diagnostics=planner_step_diagnostics(observation, self._planner),
            )
        )
        return result

    def run_natural_termination(self) -> tuple[Any, tuple[Any, ...]]:
        last_result = None
        for _ in range(self.tape.identity.max_horizon_steps):
            last_result = self.step()
            if _read_done(last_result.done):
                break
        else:
            raise P0GraspPlannerError("P0 Planner did not reach natural termination at the fixed horizon")
        natural_steps = len(self.tape.entries)
        self.tape.finalize(natural_steps)
        rows = self._backend.materialize_terminal_ledger_once(
            (0,),
            self.tape.identity.episode_ids,
        )
        if len(rows) != 1 or rows[0].lane != 0 or rows[0].episode_id != self.tape.identity.episode_ids[0]:
            raise P0GraspPlannerError("terminal ledger identity differs from the tape")
        return last_result, rows


@dataclass(frozen=True)
class ReplayReceipt:
    results: tuple[Any, ...]
    terminal_rows: tuple[Any, ...]
    passed: bool


def replay_action_trajectory(backend: Any, tape: ActionTrajectoryTape) -> ReplayReceipt:
    """Replay a complete tape for audit; no Planner call is permitted here."""

    _validate_backend(backend)
    if not isinstance(tape, ActionTrajectoryTape) or not tape.complete:
        raise P0GraspPlannerError("replay requires one complete natural-termination tape")
    preview = getattr(backend, "next_requests", None)
    if callable(preview):
        request_ids = tuple(request.episode_id for request in preview())
        if request_ids != tape.identity.episode_ids:
            raise P0GraspPlannerError("replay backend is not at the recorded manifest cursor")
    reset = backend.reset()
    actual = _tape_identity_from_reset(backend, reset)
    expected = replace(actual, horizon_steps=tape.identity.horizon_steps)
    if expected != tape.identity:
        raise P0GraspPlannerError("replay reset identity differs from the recorded tape")
    torch = importlib.import_module("torch")
    results = []
    for policy_step in range(tape.identity.horizon_steps):
        observations = backend.materialize_teacher_observations((0,))
        if len(observations) != 1:
            raise P0GraspPlannerError("replay observation audit did not return one lane")
        observation = observations[0]
        entry = tape.entries_for_step(policy_step)[0]
        if (
            observation.episode_id != entry.episode_id
            or observation.policy_step != policy_step
            or observation_fingerprint(observation) != entry.observation_fingerprint_sha256
        ):
            raise P0GraspPlannerError(f"replay observation fingerprint differs at policy_step {policy_step}")
        action = torch.as_tensor(
            np.asarray(entry.action, dtype=np.float32).reshape(1, E7_ACTION_WIDTH),
            dtype=torch.float32,
            device=backend.device,
        )
        result = backend.step_device(action)
        results.append(result)
    if not _read_done(results[-1].done):
        raise P0GraspPlannerError("replay did not reproduce natural termination")
    rows = backend.materialize_terminal_ledger_once((0,), tape.identity.episode_ids)
    if len(rows) != 1 or rows[0].episode_id != tape.identity.episode_ids[0]:
        raise P0GraspPlannerError("replay terminal ledger identity differs")
    return ReplayReceipt(results=tuple(results), terminal_rows=tuple(rows), passed=True)


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
    "observation_fingerprint",
    "planner_step_diagnostics",
    "replay_action_trajectory",
]
