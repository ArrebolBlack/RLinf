# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Current-state Planner execution on the real ``mjwarp_gpu_v1`` seam.

This module deliberately keeps Planner computation on the host while requiring
the state transition, rendering, observation materialization, and terminal
quality receipt to come from the CUDA backend.  A frozen action tape is only an
audit artifact produced after a live rollout; it is never used as the rollout
controller.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

BACKEND_ID = "mjwarp_gpu_v1"
DEFAULT_QUALITY_SCHEMA_VERSION = "db0-episode-task-quality-v1"


class GpuPlannerUnavailableError(RuntimeError):
    """Raised when a Planner request cannot be proven GPU-native."""


class GpuPlannerReplayError(RuntimeError):
    """Raised when an independent CUDA replay diverges from a Planner tape."""


def _fingerprint(value: Any) -> str:
    fingerprint = getattr(value, "fingerprint_sha256", None)
    if callable(fingerprint):
        fingerprint = fingerprint()
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise GpuPlannerReplayError(
            "observation does not expose a 64-character fingerprint"
        )
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise GpuPlannerReplayError(
            "observation fingerprint is not hexadecimal"
        ) from exc
    return fingerprint


def _action_digest(actions: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    for action in actions:
        mode = getattr(action, "mode", None)
        mode_value = getattr(mode, "value", mode)
        values = np.asarray(getattr(action, "values", None), dtype=np.float64)
        policy_step = getattr(action, "policy_step", None)
        if not isinstance(mode_value, str) or not isinstance(
            policy_step, (int, np.integer)
        ):
            raise ValueError("Planner action lacks mode/policy_step identity")
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(
                "Planner action values must be finite one-dimensional data"
            )
        encoded_mode = mode_value.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded_mode)))
        digest.update(encoded_mode)
        digest.update(struct.pack("<qI", int(policy_step), values.size))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _result_signature(result: Any) -> dict[str, Any]:
    observation = getattr(result, "observation", None)
    return {
        "observation_fingerprint_sha256": _fingerprint(observation),
        "terminated": bool(getattr(result, "terminated", False)),
        "truncated": bool(getattr(result, "truncated", False)),
        "success": bool(getattr(result, "success", False)),
        "termination_reason": getattr(result, "termination_reason", None),
        "active_stage_progress": float(getattr(result, "active_stage_progress", 0.0)),
    }


def _source_identity(backend: Any) -> Mapping[str, Any]:
    provenance = getattr(backend, "provenance", None)
    if provenance is None:
        raise GpuPlannerUnavailableError(
            "GPU backend does not expose runtime provenance"
        )
    values = {
        name: getattr(provenance, name, None)
        for name in (
            "backend_id",
            "implementation_version",
            "device_platform",
            "device_name",
            "device_ordinal",
            "precision",
            "git_commit",
            "git_tree",
            "model_sha256",
            "config_sha256",
            "request_sha256",
            "bundle_sha256",
            "manifest_sha256",
            "physical_device_uuid",
            "physical_device_pci_bus_id",
            "physical_device_identity_source",
        )
    }
    runtime_versions = getattr(provenance, "runtime_versions", {})
    if not isinstance(runtime_versions, Mapping):
        raise GpuPlannerUnavailableError(
            "GPU provenance runtime_versions is not a mapping"
        )
    values["runtime_versions"] = dict(runtime_versions)
    return MappingProxyType(values)


def _quality_payload(row: Any) -> dict[str, Any] | None:
    quality = getattr(row, "task_quality", None)
    if quality is None:
        return None
    to_dict = getattr(quality, "to_dict", None)
    if not callable(to_dict):
        raise GpuPlannerReplayError("terminal row quality is not serializable")
    payload = to_dict()
    if not isinstance(payload, dict):
        raise GpuPlannerReplayError("terminal row quality did not return a mapping")
    return payload


@dataclass(frozen=True)
class PlannerTape:
    """Complete in-memory live rollout receipt for one GPU episode."""

    request: Any
    observations: tuple[Any, ...]
    actions: tuple[Any, ...]
    results: tuple[Any, ...]
    terminal_row: Any
    source_identity: Mapping[str, Any]
    action_tape_sha256: str

    def __post_init__(self) -> None:
        if len(self.observations) != len(self.actions) + 1:
            raise ValueError(
                "Planner tape must contain one more observation than action"
            )
        if len(self.results) != len(self.actions):
            raise ValueError("Planner tape result/action lengths differ")
        if not isinstance(self.source_identity, Mapping):
            raise ValueError("Planner tape source identity must be a mapping")
        if self.action_tape_sha256 != _action_digest(self.actions):
            raise ValueError("Planner tape action digest does not match its actions")
        object.__setattr__(
            self, "source_identity", MappingProxyType(dict(self.source_identity))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return compact JSON-compatible identity and replay metadata."""

        actions = []
        for action in self.actions:
            mode = getattr(
                getattr(action, "mode", None), "value", getattr(action, "mode", None)
            )
            actions.append(
                {
                    "mode": mode,
                    "policy_step": int(getattr(action, "policy_step")),
                    "values": np.asarray(
                        getattr(action, "values"), dtype=np.float64
                    ).tolist(),
                }
            )
        terminal = {
            "lane": getattr(self.terminal_row, "lane", None),
            "episode_id": getattr(self.terminal_row, "episode_id", None),
            "task_id": getattr(self.terminal_row, "task_id", None),
            "outcome": getattr(
                getattr(self.terminal_row, "outcome", None), "value", None
            ),
            "terminated": bool(getattr(self.terminal_row, "terminated", False)),
            "truncated": bool(getattr(self.terminal_row, "truncated", False)),
            "success": bool(getattr(self.terminal_row, "success", False)),
            "termination_reason": getattr(
                self.terminal_row, "termination_reason", None
            ),
            "physics_step": getattr(self.terminal_row, "physics_step", None),
            "control_step": getattr(self.terminal_row, "control_step", None),
            "policy_step": getattr(self.terminal_row, "policy_step", None),
            "completion": getattr(self.terminal_row, "completion", None),
            "task_quality": _quality_payload(self.terminal_row),
        }
        return {
            "episode_id": getattr(self.request, "episode_id", None),
            "task_id": getattr(self.request, "task_id", None),
            "action_tape_sha256": self.action_tape_sha256,
            "observation_fingerprints": [
                _fingerprint(value) for value in self.observations
            ],
            "actions": actions,
            "results": [_result_signature(value) for value in self.results],
            "terminal": terminal,
            "source_identity": dict(self.source_identity),
        }

    def to_json(self) -> str:
        """Return canonical JSON for an audit artifact."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class GpuCurrentStatePlanner:
    """Run a host Planner against current CUDA observations only.

    ``planner_factory`` may construct a CPU teacher/Planner because it only
    computes the next E7 command.  It never receives a raw CPU environment and
    never advances physics outside ``GpuNativeBackendEnv``.
    """

    def __init__(
        self,
        *,
        backend: Any,
        task_id: str,
        planner_factory: Callable[[str, Any], Any] | None = None,
        max_control_steps: int = 420,
        evaluator_backend_id: str | None = None,
        quality_schema_version: str = DEFAULT_QUALITY_SCHEMA_VERSION,
    ) -> None:
        self._require_gpu_backend(backend, task_id)
        if (
            isinstance(getattr(backend, "num_envs", None), bool)
            or not isinstance(getattr(backend, "num_envs", None), int)
            or getattr(backend, "num_envs", 0) < 1
        ):
            raise ValueError("current-state Planner requires at least one GPU lane")
        if (
            isinstance(max_control_steps, bool)
            or not isinstance(max_control_steps, int)
            or max_control_steps < 1
        ):
            raise ValueError("max_control_steps must be a positive integer")
        self.backend = backend
        self.task_id = task_id
        self.max_control_steps = max_control_steps
        self._planner_factory = planner_factory or self._default_planner_factory
        self.source_identity = _source_identity(backend)
        if evaluator_backend_id is not None and not getattr(
            backend, "task_quality_enabled", False
        ):
            enable_quality = getattr(backend, "enable_task_quality", None)
            if enable_quality is None:
                raise GpuPlannerUnavailableError(
                    "t3 Planner quality was requested but the backend has no quality seam"
                )
            enable_quality(
                evaluator_backend_id=evaluator_backend_id,
                schema_version=quality_schema_version,
            )

    @staticmethod
    def _require_gpu_backend(backend: Any, task_id: str) -> None:
        if getattr(backend, "backend_id", None) != BACKEND_ID:
            raise GpuPlannerUnavailableError(
                f"Planner requires backend_id={BACKEND_ID!r}; CPU/raw env adapters are forbidden"
            )
        if getattr(backend, "task_id", None) != task_id:
            raise GpuPlannerUnavailableError(
                "Planner task_id does not match the GPU backend"
            )
        provenance = getattr(backend, "provenance", None)
        platform = str(getattr(provenance, "device_platform", "")).strip().lower()
        if platform not in {"cuda", "gpu"}:
            raise GpuPlannerUnavailableError(
                "Planner requires CUDA physics/render/observation provenance; CPU fallback is forbidden"
            )
        track = getattr(backend, "observation_track", None)
        track_value = getattr(track, "value", track)
        if track_value == "future_oracle":
            raise GpuPlannerUnavailableError(
                "Planner cannot consume future-oracle observations"
            )

    @staticmethod
    def _default_planner_factory(task_id: str, request: Any) -> Any:
        from se3_wam.benchmark.teacher_factory import make_privileged_teacher

        return make_privileged_teacher(task_id, request=request)

    @staticmethod
    def _coerce_action(raw_action: Any, request: Any, observation: Any) -> Any:
        expected_mode = getattr(request, "action_mode", None)
        expected_policy_step = getattr(observation, "policy_step", None)
        if expected_policy_step is None:
            raise ValueError("current observation has no policy_step")
        raw_mode = getattr(raw_action, "mode", None)
        raw_values = getattr(raw_action, "values", None)
        raw_policy_step = getattr(raw_action, "policy_step", expected_policy_step)
        if raw_values is None:
            raise ValueError("Planner output has no values")
        values = np.asarray(raw_values, dtype=np.float64)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("t3 E7 Planner output must be finite shape (7,)")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("Planner output must already be normalized to [-1, 1]")
        if raw_mode is not None and raw_mode is not expected_mode:
            raise ValueError(
                "Planner output action mode differs from the frozen E7 request"
            )
        if int(raw_policy_step) != int(expected_policy_step):
            raise ValueError(
                "Planner action policy_step is not the current observation clock"
            )
        if raw_mode is not None and hasattr(raw_action, "policy_step"):
            return raw_action
        try:
            from se3_wam.benchmark.api import ActionCommand
        except ImportError as exc:
            raise GpuPlannerUnavailableError(
                "SE(3)-WAM ActionCommand is required to build a live GPU action"
            ) from exc
        return ActionCommand(
            mode=expected_mode,
            values=values,
            policy_step=int(expected_policy_step),
        )

    def rollout(self, request: Any | None = None) -> PlannerTape:
        """Execute one live current-observation rollout and return its tape."""

        if self.backend.num_envs != 1:
            raise ValueError("rollout requires a one-lane GPU backend; use rollout_batch")
        if request is None:
            request = self.backend.next_request()
        if getattr(request, "task_id", None) != self.task_id:
            raise ValueError("Planner request task_id differs from the configured task")
        planner_result = self._planner_factory(self.task_id, request)
        planner = (
            planner_result[0] if isinstance(planner_result, tuple) else planner_result
        )
        reset = getattr(planner, "reset", None)
        if callable(reset):
            reset()
        reset_observations = tuple(self.backend.reset((request,)))
        if len(reset_observations) != 1 or reset_observations[0] is None:
            raise RuntimeError(
                "GPU backend reset did not return one current observation"
            )
        observations: list[Any] = [reset_observations[0]]
        actions: list[Any] = []
        results: list[Any] = []
        for _ in range(self.max_control_steps):
            observation = observations[-1]
            act = getattr(planner, "act", None)
            if not callable(act):
                raise TypeError("Planner must expose act(current_observation)")
            action = self._coerce_action(act(observation), request, observation)
            step_results = tuple(self.backend.step((action,)))
            if len(step_results) != 1 or step_results[0] is None:
                raise RuntimeError("GPU backend step did not return one current result")
            result = step_results[0]
            next_observation = getattr(result, "observation", None)
            if next_observation is None:
                raise RuntimeError("GPU step result has no next current observation")
            actions.append(action)
            results.append(result)
            observations.append(next_observation)
            if bool(getattr(result, "terminated", False)) or bool(
                getattr(result, "truncated", False)
            ):
                break
        else:
            raise RuntimeError(
                "Planner rollout exceeded the frozen GPU environment horizon"
            )

        terminal_rows = tuple(getattr(self.backend, "last_terminal_rows", ()))
        if len(terminal_rows) != 1:
            raise RuntimeError(
                "GPU Planner rollout must produce exactly one exact-once terminal ledger row"
            )
        return PlannerTape(
            request=request,
            observations=tuple(observations),
            actions=tuple(actions),
            results=tuple(results),
            terminal_row=terminal_rows[0],
            source_identity=self.source_identity,
            action_tape_sha256=_action_digest(tuple(actions)),
        )

    def rollout_batch(self, requests: Any) -> tuple[PlannerTape, ...]:
        """Execute independent live current-observation rollouts in one GPU batch.

        Each lane owns its own host Planner instance and action/observation tape;
        the CUDA backend remains the only state transition and terminal/evaluator
        authority.  Inactive lanes submit ``None`` after natural termination so
        the batch can finish without padding a terminal episode with actions.
        """

        request_tuple = tuple(requests)
        if len(request_tuple) != self.backend.num_envs:
            raise ValueError("batch rollout requires one request per GPU lane")
        if not request_tuple:
            raise ValueError("batch rollout requires at least one request")
        for request in request_tuple:
            if getattr(request, "task_id", None) != self.task_id:
                raise ValueError("Planner request task_id differs from the configured task")

        planners: list[Any] = []
        for request in request_tuple:
            planner_result = self._planner_factory(self.task_id, request)
            planner = (
                planner_result[0]
                if isinstance(planner_result, tuple)
                else planner_result
            )
            reset = getattr(planner, "reset", None)
            if callable(reset):
                reset()
            planners.append(planner)

        reset_observations = tuple(self.backend.reset(request_tuple))
        if len(reset_observations) != len(request_tuple) or any(
            observation is None for observation in reset_observations
        ):
            raise RuntimeError("GPU backend batch reset did not return all current observations")

        observations = [[observation] for observation in reset_observations]
        actions: list[list[Any]] = [[] for _ in request_tuple]
        results: list[list[Any]] = [[] for _ in request_tuple]
        terminal_rows: list[Any] = []
        active = np.ones(len(request_tuple), dtype=np.bool_)

        for _ in range(self.max_control_steps):
            commands: list[Any | None] = [None] * len(request_tuple)
            for lane in np.flatnonzero(active):
                lane_index = int(lane)
                act = getattr(planners[lane_index], "act", None)
                if not callable(act):
                    raise TypeError("Planner must expose act(current_observation)")
                commands[lane_index] = self._coerce_action(
                    act(observations[lane_index][-1]),
                    request_tuple[lane_index],
                    observations[lane_index][-1],
                )

            step_results = tuple(self.backend.step(tuple(commands)))
            if len(step_results) != len(request_tuple):
                raise RuntimeError("GPU backend batch step returned the wrong lane count")
            terminal_rows.extend(tuple(getattr(self.backend, "last_terminal_rows", ())))

            done = np.zeros(len(request_tuple), dtype=np.bool_)
            for lane in np.flatnonzero(active):
                lane_index = int(lane)
                result = step_results[lane_index]
                if result is None:
                    raise RuntimeError("GPU backend batch step lost an active lane result")
                next_observation = getattr(result, "observation", None)
                if next_observation is None:
                    raise RuntimeError("GPU batch step result has no next current observation")
                actions[lane_index].append(commands[lane_index])
                results[lane_index].append(result)
                observations[lane_index].append(next_observation)
                done[lane_index] = bool(getattr(result, "terminated", False)) or bool(
                    getattr(result, "truncated", False)
                )
            active &= ~done
            if not np.any(active):
                break
        else:
            raise RuntimeError("Planner batch rollout exceeded the frozen GPU environment horizon")

        rows_by_lane: dict[int, Any] = {}
        for row in terminal_rows:
            lane = int(getattr(row, "lane"))
            if lane in rows_by_lane:
                raise RuntimeError(f"GPU Planner batch produced duplicate terminal row for lane {lane}")
            rows_by_lane[lane] = row
        if set(rows_by_lane) != set(range(len(request_tuple))):
            raise RuntimeError(
                "GPU Planner batch must produce exactly one terminal ledger row per lane"
            )

        return tuple(
            PlannerTape(
                request=request_tuple[lane],
                observations=tuple(observations[lane]),
                actions=tuple(actions[lane]),
                results=tuple(results[lane]),
                terminal_row=rows_by_lane[lane],
                source_identity=self.source_identity,
                action_tape_sha256=_action_digest(tuple(actions[lane])),
            )
            for lane in range(len(request_tuple))
        )

    def replay(
        self, tape: PlannerTape, *, backend: Any | None = None
    ) -> Mapping[str, Any]:
        """Replay a completed tape on a fresh/current CUDA backend instance."""

        replay_backend = self.backend if backend is None else backend
        self._require_gpu_backend(replay_backend, self.task_id)
        if getattr(replay_backend, "num_envs", None) != 1:
            raise ValueError("GPU Planner replay requires one GPU lane")
        if _source_identity(replay_backend) != tape.source_identity:
            raise GpuPlannerReplayError(
                "GPU replay provenance differs from the Planner tape"
            )
        if _action_digest(tape.actions) != tape.action_tape_sha256:
            raise GpuPlannerReplayError(
                "Planner tape action digest changed before replay"
            )
        observations = tuple(replay_backend.reset((tape.request,)))
        if len(observations) != 1 or _fingerprint(observations[0]) != _fingerprint(
            tape.observations[0]
        ):
            raise GpuPlannerReplayError(
                "GPU replay reset observation fingerprint diverged"
            )
        replay_results: list[Any] = []
        for index, action in enumerate(tape.actions):
            step_results = tuple(replay_backend.step((action,)))
            if len(step_results) != 1 or step_results[0] is None:
                raise GpuPlannerReplayError("GPU replay lost a step result")
            result = step_results[0]
            expected = _result_signature(tape.results[index])
            actual = _result_signature(result)
            if actual != expected:
                raise GpuPlannerReplayError(
                    f"GPU replay result diverged at control step {index + 1}: "
                    f"expected={expected}, actual={actual}"
                )
            replay_results.append(result)
        replay_rows = tuple(getattr(replay_backend, "last_terminal_rows", ()))
        if len(replay_rows) != 1:
            raise GpuPlannerReplayError(
                "GPU replay did not produce exactly one terminal row"
            )
        expected_quality = _quality_payload(tape.terminal_row)
        actual_quality = _quality_payload(replay_rows[0])
        if actual_quality != expected_quality:
            raise GpuPlannerReplayError("GPU replay task-quality summary diverged")
        return {
            "passed": True,
            "action_count": len(replay_results),
            "action_tape_sha256": tape.action_tape_sha256,
            "observation_fingerprints": [
                _fingerprint(observation) for observation in tape.observations
            ],
            "terminal_quality_summary_sha256": (
                actual_quality.get("summary_sha256")
                if actual_quality is not None
                else None
            ),
        }


__all__ = [
    "DEFAULT_QUALITY_SCHEMA_VERSION",
    "GpuCurrentStatePlanner",
    "GpuPlannerReplayError",
    "GpuPlannerUnavailableError",
    "PlannerTape",
]
