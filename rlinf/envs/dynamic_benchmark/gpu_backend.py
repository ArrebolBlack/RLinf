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

"""Thin GPU-native Dynamic Benchmark backend adapter for RLinf.

This adapter maps RLinf's batched environment requests onto the canonical
SE(3)-WAM GPU-native backend seam (``se3_wam.benchmark.gpu_native.factory``:
``make_gpu_native_env``, backend ``mjwarp_gpu_v1``).  It owns one homogeneous
device batch, consumes E7 actions, and materializes selected lanes into the
backend-neutral ``ObservationBundle``/``StepResult`` contracts through the
explicit audit seam.  Reward, termination, success, and timeout stay inside
the backend; RLinf keeps its own reward shaping on top of the materialized
events and stage progress.

For ``planner_mode='online_privileged_teacher_v1'`` the adapter delegates the
control-plane loop to the SE(3)-WAM surface Planner seam.  The Planner runs on
CPU by design, but every action is computed from the latest GPU audit packet
and immediately submitted to that same ``mjwarp_gpu_v1`` environment.  The
runtime feature manifest and evaluator identity are mandatory in this mode.

The GPU backend requires every lane to reuse the frozen export identity.  The
legacy homogeneous path may allocate a fresh ``episode_id`` while keeping every
other reset field exact.  Per-row export runners enable the stricter mode, which
also freezes ``episode_id`` and validates the complete request before reset or
step.  Neither path falls back to the CPU ``make_mujoco_env`` implementation.

``se3_wam`` imports are deliberately lazy so this module (and the tests that
exercise its request mapping) can be imported on machines without the SE3-WAM
source or the pinned MJWarp runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np


class GpuNativeBackendUnavailableError(RuntimeError):
    """Raised when the SE3-WAM GPU-native backend seam cannot be built."""


def reset_request_identity(request: Any) -> dict[str, Any]:
    """Return every field in the canonical ``ResetRequest`` identity."""

    factors = getattr(request, "factors", None)
    if not isinstance(factors, Mapping):
        raise ValueError("GPU reset request factors must be a mapping")

    def enum_value(name: str) -> Any:
        value = getattr(request, name, None)
        return getattr(value, "value", value)

    identity = {
        "api_version": getattr(request, "api_version", None),
        "episode_id": getattr(request, "episode_id", None),
        "task_id": getattr(request, "task_id", None),
        "split": enum_value("split"),
        "seed": getattr(request, "seed", None),
        "action_mode": enum_value("action_mode"),
        "observation_track": enum_value("observation_track"),
        "object_mode": getattr(request, "object_mode", None),
        "reset_mode": getattr(request, "reset_mode", None),
        "factors": dict(factors),
    }
    if any(value is None for name, value in identity.items() if name != "factors"):
        raise ValueError("GPU reset request is missing canonical identity fields")
    return identity


def require_exact_reset_request(
    actual: Any,
    expected: Any,
    *,
    allow_episode_id_change: bool = False,
    context: str = "GPU reset",
) -> None:
    """Fail before physics if an export-bound reset field differs."""

    actual_identity = reset_request_identity(actual)
    expected_identity = reset_request_identity(expected)
    compared_fields = tuple(
        name
        for name in expected_identity
        if not (allow_episode_id_change and name == "episode_id")
    )
    mismatches = {
        name: {
            "expected": expected_identity[name],
            "actual": actual_identity[name],
        }
        for name in compared_fields
        if actual_identity[name] != expected_identity[name]
    }
    if mismatches:
        raise ValueError(f"{context} request identity mismatch: {mismatches}")


class GpuNativeBackendEnv:
    """Own one MjWarpGpuEnv RL batch and translate RLinf contracts.

    The public surface intentionally mirrors the parts of
    :class:`~rlinf.envs.dynamic_benchmark.dynamic_benchmark_env.DynamicBenchmarkEnv`
    that the training loop consumes: request source, reset, step, close, and a
    provenance probe used as no-CPU-fallback evidence.
    """

    def __init__(
        self,
        *,
        task_id: str,
        num_envs: int,
        export_dir: str,
        device_ordinal: int = 0,
        image_size: int = 64,
        camera_observations: bool = False,
        planner_mode: str | None = None,
        runtime_manifest_path: str | None = None,
        runtime_manifest_sha256: str | None = None,
        evaluator_backend_id: str | None = None,
        expected_gpu_uuid: str | None = None,
        expected_se3_source_commit: str | None = None,
        expected_se3_source_tree: str | None = None,
        task_quality_schema_version: str | None = None,
        task_quality_evaluator_backend_id: str | None = None,
        observation_track: str = "state",
        require_exact_export_identity: bool = False,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("GPU-native backend requires a positive num_envs")
        if not isinstance(export_dir, str) or not export_dir.strip():
            raise ValueError("GPU-native backend requires a non-empty export_dir")
        if (
            isinstance(device_ordinal, bool)
            or not isinstance(device_ordinal, int)
            or device_ordinal < 0
        ):
            raise ValueError("device_ordinal must be a non-negative integer")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size < 16
        ):
            raise ValueError("image_size must be a positive integer of at least 16")
        if not isinstance(camera_observations, bool):
            raise ValueError("camera_observations must be bool")
        if not isinstance(require_exact_export_identity, bool):
            raise ValueError("require_exact_export_identity must be bool")
        if planner_mode not in {None, "online_privileged_teacher_v1"}:
            raise ValueError("planner_mode is not a registered GPU Planner mode")
        planner_enabled = planner_mode is not None
        if planner_enabled and task_id not in {"t1_belt", "t1_occ", "t5_replan"}:
            raise ValueError(
                "online privileged Planner mode is registered only for surface T1 or t5_replan"
            )
        if planner_enabled:
            if not isinstance(runtime_manifest_path, str) or not runtime_manifest_path.strip():
                raise ValueError("online Planner mode requires runtime_manifest_path")
            if not isinstance(runtime_manifest_sha256, str) or not runtime_manifest_sha256.strip():
                raise ValueError("online Planner mode requires runtime_manifest_sha256")
            if task_id == "t5_replan" and (
                not isinstance(task_quality_schema_version, str)
                or not task_quality_schema_version.strip()
            ):
                raise ValueError(
                    "online T5 Planner mode requires task_quality_schema_version"
                )
            if not isinstance(evaluator_backend_id, str) or not evaluator_backend_id.strip():
                raise ValueError("online Planner mode requires evaluator_backend_id")
        try:
            from se3_wam.benchmark.contracts import ObservationTrack
            from se3_wam.benchmark.gpu_native.factory import make_gpu_native_env
            from se3_wam.benchmark.gpu_native.p0_grasp_engine import (
                load_p0_grasp_artifacts,
            )
            from se3_wam.benchmark.gpu_native.tasks import GpuNativeConsumer
        except ImportError as exc:
            raise GpuNativeBackendUnavailableError(
                "the GPU-native Dynamic Benchmark backend requires the SE3-WAM "
                "benchmark source with the GPU-native package on PYTHONPATH"
            ) from exc
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
            configured_observation_track = ObservationTrack(observation_track)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported GPU-native observation_track: {observation_track!r}"
            ) from exc

        self._task_id = task_id
        self._num_envs = num_envs
        self._export_dir = export_dir
        self._device_ordinal = device_ordinal
        self._image_size = image_size
        self._expected_gpu_uuid = expected_gpu_uuid
        self._expected_se3_source_commit = expected_se3_source_commit
        self._expected_se3_source_tree = expected_se3_source_tree
        self._task_quality_schema_version = task_quality_schema_version
        self._task_quality_evaluator_backend_id = task_quality_evaluator_backend_id
        self._require_exact_export_identity = require_exact_export_identity
        self._consumer = GpuNativeConsumer.RL
        runtime_evidence = ()
        if planner_enabled:
            try:
                from se3_wam.benchmark.gpu_native.tasks import (
                    load_runtime_feature_evidence,
                )

                runtime_evidence = (
                    load_runtime_feature_evidence(
                        runtime_manifest_path,
                        expected_manifest_sha256=runtime_manifest_sha256,
                    ),
                )
            except Exception as exc:
                raise GpuNativeBackendUnavailableError(
                    "online Planner mode requires a passed, hash-pinned surface-velocity manifest"
                ) from exc
        self._observation_track = (
            ObservationTrack.HYBRID
            if planner_enabled or camera_observations
            else configured_observation_track
        )
        engine_kwargs: dict[str, Any] = {}
        for name, value in (
            ("expected_device_uuid", expected_gpu_uuid),
            ("expected_source_commit", expected_se3_source_commit),
            ("expected_source_tree", expected_se3_source_tree),
        ):
            if value is not None:
                engine_kwargs[name] = value
        factory_kwargs: dict[str, Any] = {}
        if engine_kwargs:
            factory_kwargs["engine_kwargs"] = engine_kwargs
        self._env = make_gpu_native_env(
            task_id,
            consumer=self._consumer,
            batch_size=num_envs,
            observation_track=self._observation_track,
            export_dir=export_dir,
            device_ordinal=device_ordinal,
            runtime_evidence=runtime_evidence,
            image_size=image_size,
            **factory_kwargs,
        )
        self._artifacts = load_p0_grasp_artifacts(export_dir)
        self._frozen_request = self._artifacts.reset_request
        if self._frozen_request.task_id != task_id:
            raise GpuNativeBackendUnavailableError(
                "frozen export task_id does not match the requested task"
            )
        frozen_track = getattr(self._frozen_request.observation_track, "value", None)
        if frozen_track != self._observation_track.value:
            raise GpuNativeBackendUnavailableError(
                "frozen export observation track does not match the GPU adapter track"
            )
        self._planner_mode = planner_mode
        self._planner = None
        self._task_quality_enabled = False
        if task_quality_schema_version is not None:
            self.enable_task_quality(
                evaluator_backend_id=task_quality_evaluator_backend_id,
                schema_version=task_quality_schema_version,
            )
        elif planner_enabled:
            self._env.enable_task_quality(evaluator_backend_id=evaluator_backend_id)
            self._task_quality_enabled = True
        if planner_enabled:
            try:
                if task_id == "t5_replan":
                    from se3_wam.benchmark.gpu_native.keyed_puck_planner import (
                        KeyedPuckReplanPlanner,
                    )

                    self._planner = KeyedPuckReplanPlanner(
                        gpu_env=self._env,
                        task_id=task_id,
                        image_size=image_size,
                    )
                else:
                    from se3_wam.benchmark.gpu_native.surface_planner import (
                        SurfaceCapturePlanner,
                    )

                    self._planner = SurfaceCapturePlanner(
                        gpu_env=self._env,
                        task_id=task_id,
                        image_size=image_size,
                        runtime_manifest_sha256=runtime_manifest_sha256,
                        runtime_device_uuid=runtime_evidence[0].device_uuid,
                    )
            except Exception as exc:
                raise GpuNativeBackendUnavailableError(
                    "online Planner mode could not establish the GPU quality/Planner seam"
                ) from exc
        self._episode_counter = 0
        self._last_terminal_ledger: Any | None = None
        self._action_tapes: list[list[Any]] = [[] for _ in range(num_envs)]
        self._observation_tapes: list[list[Any]] = [[] for _ in range(num_envs)]
        self._active_mask = np.ones(num_envs, dtype=np.bool_)
        self._episode_ids: tuple[str | None, ...] = (None,) * num_envs
        self._last_observations: tuple[Any | None, ...] = (None,) * num_envs
        self._last_terminal_rows: tuple[Any, ...] = ()
        self._reset_identity_validated = False

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def export_dir(self) -> str:
        return self._export_dir

    @property
    def consumer(self) -> Any:
        return self._consumer

    @property
    def observation_track(self) -> Any:
        return self._observation_track

    @property
    def planner_enabled(self) -> bool:
        return self._planner is not None

    @property
    def backend_id(self) -> str:
        return self._env.backend_id

    @property
    def provenance(self) -> Any:
        return self._env.provenance

    @property
    def frozen_request(self) -> Any:
        return self._frozen_request

    @property
    def task_quality_enabled(self) -> bool:
        return self._task_quality_enabled

    @property
    def require_exact_export_identity(self) -> bool:
        """Whether reset also freezes the artifact's exact episode identity."""

        return self._require_exact_export_identity

    @property
    def frozen_requests(self) -> tuple[Any, ...]:
        """Return the export-bound request for each homogeneous lane."""

        return tuple(self._frozen_request for _ in range(self._num_envs))

    @property
    def current_observations(self) -> tuple[Any | None, ...]:
        """Return the latest materialized observation for each lane."""

        return self._last_observations

    @property
    def last_terminal_rows(self) -> tuple[Any, ...]:
        """Return terminal rows materialized by the most recent step."""

        return self._last_terminal_rows

    @property
    def last_terminal_ledger(self) -> Any | None:
        """Return the one-shot terminal receipt produced by the latest step."""

        return self._last_terminal_ledger

    def enable_task_quality(
        self,
        *,
        evaluator_backend_id: str,
        schema_version: str,
    ) -> None:
        """Enable the backend's released success-quality schema before reset."""

        if self._task_quality_enabled:
            raise RuntimeError("GPU task quality is already enabled")
        method = getattr(self._env, "enable_task_quality", None)
        if method is None:
            raise GpuNativeBackendUnavailableError(
                "the GPU-native engine does not expose task-quality admission"
            )
        method(
            evaluator_backend_id=evaluator_backend_id,
            schema_version=schema_version,
        )
        self._task_quality_enabled = True

    def action_tape(self, lane: int = 0) -> tuple[Any, ...]:
        """Return the complete live action tape for one lane."""

        self._validate_lane(lane)
        return tuple(
            value.copy() if hasattr(value, "copy") else value
            for value in self._action_tapes[lane]
        )

    def observation_tape(self, lane: int = 0) -> tuple[Any, ...]:
        """Return the host audit observations corresponding to one live tape."""

        self._validate_lane(lane)
        return tuple(self._observation_tapes[lane])

    def _validate_lane(self, lane: int) -> None:
        if (
            isinstance(lane, bool)
            or not isinstance(lane, int)
            or not 0 <= lane < self._num_envs
        ):
            raise IndexError(f"lane must be an integer in [0, {self._num_envs})")

    def new_replay_backend(self) -> GpuNativeBackendEnv:
        """Build an independent CUDA backend for action-tape replay."""

        return type(self)(
            task_id=self._task_id,
            num_envs=self._num_envs,
            export_dir=self._export_dir,
            device_ordinal=self._device_ordinal,
            image_size=self._image_size,
            expected_gpu_uuid=self._expected_gpu_uuid,
            expected_se3_source_commit=self._expected_se3_source_commit,
            expected_se3_source_tree=self._expected_se3_source_tree,
            task_quality_schema_version=self._task_quality_schema_version,
            task_quality_evaluator_backend_id=self._task_quality_evaluator_backend_id,
            observation_track=self._observation_track.value,
            require_exact_export_identity=self._require_exact_export_identity,
        )

    def next_request(self) -> Any:
        """Return the frozen export request with a fresh per-lane episode id.

        The configured observation track is carried through to the GPU engine;
        the wrapper validates requests against the frozen artifact identity.
        """
        request = self._frozen_request
        if self._require_exact_export_identity:
            return request
        episode_id = f"{self._task_id}-gpu-{self._episode_counter:08d}"
        self._episode_counter += 1
        return replace(
            request,
            episode_id=episode_id,
            observation_track=self._observation_track,
        )

    def validate_frozen_request(self, request: Any, *, exact_episode_id: bool) -> None:
        """Validate one request against the loaded export without side effects."""

        if not isinstance(exact_episode_id, bool):
            raise ValueError("exact_episode_id must be bool")
        require_exact_reset_request(
            request,
            self._frozen_request,
            allow_episode_id_change=not exact_episode_id,
            context=f"GPU export {self._export_dir}",
        )

    def policy_steps(self) -> Any:
        """Return the host clock policy steps expected for the next commands."""
        return np.asarray(
            self._env.bookkeeping.clock.policy_steps,
            dtype=np.int64,
        ).copy()

    def _arm_t5_event_tapes(self, requests: tuple[Any, ...]) -> None:
        """Arm benchmark-owned T5 intervention tapes before the first step."""

        if self._task_id != "t5_replan":
            return
        try:
            from se3_wam.benchmark.config import load_task_config
            from se3_wam.benchmark.keyed_puck import T5EventTape

            from .t5_runtime import t5_branch_for_episode
        except ImportError as exc:
            raise GpuNativeBackendUnavailableError(
                "T5 GPU event-tape support requires the keyed-puck runtime"
            ) from exc
        arm_event_tapes = getattr(self._env, "arm_event_tapes", None)
        if not callable(arm_event_tapes):
            raise GpuNativeBackendUnavailableError(
                "T5 GPU event-tape support requires the canonical GPU event-tape seam"
            )
        config = load_task_config(self._task_id)
        try:
            trigger_gate_y_m = float(config["event"]["default_trigger_gate_y_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GpuNativeBackendUnavailableError(
                "T5 task config lacks the frozen event trigger gate"
            ) from exc
        arm_event_tapes(
            tuple(
                T5EventTape(
                    event_id=(
                        f"rlinf-{request.split.value}-{request.episode_id}-"
                        f"{t5_branch_for_episode(request.episode_id)}"
                    ),
                    branch=t5_branch_for_episode(request.episode_id),
                    trigger_gate_y_m=trigger_gate_y_m,
                )
                for request in requests
            )
        )

    def reset(self, requests: Any) -> tuple[Any, ...]:
        """Reset the whole batch and materialize every active lane."""
        requests_tuple = tuple(requests)
        if len(requests_tuple) != self._num_envs:
            raise ValueError("GPU-native reset requires one request per lane")
        self._reset_identity_validated = False
        for lane, request in enumerate(requests_tuple):
            require_exact_reset_request(
                request,
                self._frozen_request,
                allow_episode_id_change=not self._require_exact_export_identity,
                context=f"GPU lane {lane} export {self._export_dir}",
            )
        if self._planner is not None:
            observations = self._planner.reset(requests_tuple)
            self._reset_identity_validated = True
            return observations
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        self._env.reset(requests_tuple)
        self._arm_t5_event_tapes(requests_tuple)
        audit = self._env.materialize_audit(
            AuditRequest(lanes=tuple(range(self._num_envs)), include_step_result=False)
        )
        observations = tuple(lane.observation for lane in audit.lanes)
        if len(observations) != self._num_envs:
            raise RuntimeError("GPU-native reset lost lane observations")
        self._last_terminal_ledger = None
        self._last_terminal_rows = ()
        self._action_tapes = [[] for _ in range(self._num_envs)]
        self._observation_tapes = [[observation] for observation in observations]
        self._active_mask = np.ones(self._num_envs, dtype=np.bool_)
        self._episode_ids = tuple(
            getattr(request, "episode_id", None) for request in requests_tuple
        )
        self._last_observations = observations
        self._reset_identity_validated = True
        return observations

    def step(
        self,
        commands: Any,
        *,
        active_mask: Any | None = None,
    ) -> tuple[Any | None, ...]:
        """Step the whole batch and materialize every active lane."""
        commands_tuple = tuple(commands)
        if self._require_exact_export_identity and not self._reset_identity_validated:
            raise RuntimeError(
                "exact-export GPU backend requires a validated reset before step"
            )
        if self._planner is not None:
            raise GpuNativeBackendUnavailableError(
                "online Planner mode rejects external action batches; use step_planner()"
            )
        if len(commands_tuple) != self._num_envs:
            raise ValueError("GPU-native step requires one command per lane")
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        explicit_mask = active_mask is not None
        if explicit_mask:
            requested_mask = np.asarray(active_mask)
            if (
                requested_mask.shape != (self._num_envs,)
                or requested_mask.dtype != np.bool_
            ):
                raise ValueError(
                    f"active_mask must be bool shape ({self._num_envs},)"
                )
            if not np.any(requested_mask):
                raise ValueError("active_mask must select at least one lane")
            for lane, (command, active) in enumerate(
                zip(commands_tuple, requested_mask, strict=True)
            ):
                if not active and command is not None:
                    raise ValueError(f"inactive GPU lane {lane} must carry None")
            selected_mask = self._active_mask & requested_mask
        # The all-None case is retained for the legacy adapter contract tests;
        # real MjWarp batches must provide E7 commands for every selected lane.
        elif all(value is None for value in commands_tuple) and np.all(
            self._active_mask
        ):
            selected_mask = np.array(self._active_mask, copy=True)
        else:
            selected_mask = self._active_mask & np.asarray(
                [value is not None for value in commands_tuple], dtype=np.bool_
            )
        selected_lanes = tuple(int(value) for value in np.flatnonzero(selected_mask))
        if not selected_lanes:
            self._last_terminal_ledger = None
            self._last_terminal_rows = ()
            return (None,) * self._num_envs
        if hasattr(self._env, "mark_done"):
            self._env.step(commands_tuple, active_mask=selected_mask)
        else:
            self._env.step(commands_tuple)
        audit = self._env.materialize_audit(
            AuditRequest(lanes=selected_lanes, include_step_result=True)
        )
        result_by_lane: dict[int, Any] = {}
        for lane in audit.lanes:
            if lane.step_result is None:
                raise RuntimeError("GPU-native step lost a lane step result")
            result_by_lane[lane.lane] = lane.step_result
            self._last_observations = self._replace_observation(
                self._last_observations,
                lane.lane,
                lane.step_result.observation,
            )
        self._last_terminal_ledger = None
        terminal_lanes = tuple(
            lane
            for lane, result in result_by_lane.items()
            if bool(getattr(result, "terminated", False))
            or bool(getattr(result, "truncated", False))
        )
        if self._task_quality_enabled and terminal_lanes and not explicit_mask:
            from se3_wam.benchmark.gpu_native.audit import TerminalLedgerRequest

            episode_ids = tuple(
                self._env.bookkeeping.episode_ids[lane] for lane in terminal_lanes
            )
            if any(not isinstance(episode_id, str) for episode_id in episode_ids):
                raise RuntimeError("GPU-native terminal lane lost its episode identity")
            ledger = self._env.materialize_terminal_ledger(
                TerminalLedgerRequest(
                    lanes=terminal_lanes,
                    episode_ids=episode_ids,
                )
            )
            if tuple(row.lane for row in ledger.rows) != terminal_lanes:
                raise RuntimeError("GPU-native terminal ledger changed lane order")
            self._last_terminal_ledger = ledger
            self._last_terminal_rows = tuple(ledger.rows)
        else:
            self._last_terminal_rows = ()
        done_mask = np.zeros(self._num_envs, dtype=np.bool_)
        results: list[Any | None] = [None] * self._num_envs
        for lane in selected_lanes:
            command = commands_tuple[lane]
            result = result_by_lane[lane]
            results[lane] = result
            values = getattr(command, "values", None)
            self._action_tapes[lane].append(
                values.copy() if hasattr(values, "copy") else values
            )
            self._observation_tapes[lane].append(result.observation)
            done_mask[lane] = bool(getattr(result, "terminated", False)) or bool(
                getattr(result, "truncated", False)
            )
        if np.any(done_mask) and not explicit_mask:
            if hasattr(self._env, "mark_done"):
                self._env.mark_done(done_mask)
            self._active_mask &= ~done_mask
        return tuple(results)

    def materialize_terminal_ledger(
        self,
        lanes: Sequence[int],
        episode_ids: Sequence[str],
    ) -> Any:
        """Materialize caller-pinned terminal rows before host mark_done."""

        try:
            from se3_wam.benchmark.gpu_native.audit import TerminalLedgerRequest
        except ImportError as exc:
            raise GpuNativeBackendUnavailableError(
                "terminal-ledger materialization requires the SE3-WAM GPU audit seam"
            ) from exc
        request = TerminalLedgerRequest(
            lanes=tuple(int(lane) for lane in lanes),
            episode_ids=tuple(str(episode_id) for episode_id in episode_ids),
        )
        materialize = getattr(self._env, "materialize_terminal_ledger", None)
        if not callable(materialize):
            raise GpuNativeBackendUnavailableError(
                "GPU engine does not expose the terminal-ledger seam"
            )
        return materialize(request)

    def mark_done(self, done_mask: Any) -> None:
        """Close host bookkeeping after terminal audit/ledger consumption."""

        mask = np.asarray(done_mask)
        if mask.shape != (self._num_envs,) or mask.dtype != np.bool_:
            raise ValueError(f"done_mask must be bool shape ({self._num_envs},)")
        mark_done = getattr(self._env, "mark_done", None)
        if not callable(mark_done):
            raise GpuNativeBackendUnavailableError(
                "GPU environment does not expose explicit done bookkeeping"
            )
        mark_done(mask)
        self._active_mask &= ~mask

    @staticmethod
    def _replace_observation(
        observations: tuple[Any | None, ...], lane: int, observation: Any
    ) -> tuple[Any | None, ...]:
        updated = list(observations)
        updated[lane] = observation
        return tuple(updated)

    def step_planner(self) -> tuple[Any, ...]:
        """Advance one online CPU-Plan/GPU-physics control interval."""
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.step()

    def planner_tape(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        """Return an append-only audit tape; it is never accepted as an action input."""
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.tape()

    def planner_media(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        """Return copied GPU-rendered scene/wrist frames for review packaging."""
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.media()

    def planner_tape_sha256(self) -> str:
        """Return the hash of the complete structured Planner audit tape."""
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.tape_sha256()

    def planner_media_sha256(self) -> tuple[str, ...]:
        """Return hashes of the captured GPU-rendered scene/wrist frames."""
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.media_sha256()

    def planner_replay_audit(
        self,
        *,
        request: Any,
        tape: Any,
        diagnostic_only: bool = False,
        require_terminal: bool = True,
    ) -> Any:
        """Replay a completed T5 tape only through the explicit audit boundary."""
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        try:
            from se3_wam.benchmark.gpu_native.keyed_puck_replay import (
                replay_frozen_t5_action_tape,
            )
        except ImportError as exc:
            raise GpuNativeBackendUnavailableError(
                "T5 Planner replay audit requires the keyed-puck replay seam"
            ) from exc
        return replay_frozen_t5_action_tape(
            gpu_env=self._env,
            request=request,
            tape=tape,
            diagnostic_only=diagnostic_only,
            require_terminal=require_terminal,
        )

    def materialize_current_observations(self) -> tuple[Any, ...]:
        """Return live GPU observations for a CPU Planner control boundary."""

        materializer = getattr(self._env, "materialize_planner_observations", None)
        if not callable(materializer):
            raise GpuNativeBackendUnavailableError(
                "SE3-WAM GPU backend lacks the current-state Planner adapter"
            )
        observations = tuple(materializer())
        if len(observations) != self._num_envs:
            raise GpuNativeBackendUnavailableError(
                "GPU Planner adapter returned an incomplete cohort"
            )
        return observations

    def materialize_replay_probe_audit(self) -> Mapping[str, Any]:
        """Return the explicit diagnostic-only T3 engine state audit."""

        materializer = getattr(self._env, "materialize_replay_probe_audit", None)
        if not callable(materializer):
            raise GpuNativeBackendUnavailableError(
                "SE3-WAM GPU backend lacks the T3 replay-probe audit seam"
            )
        payload = materializer()
        if not isinstance(payload, Mapping) or payload.get("diagnostic_only") is not True:
            raise GpuNativeBackendUnavailableError(
                "T3 replay-probe audit returned an invalid diagnostic receipt"
            )
        return payload

    def verify_replay_probe_snapshot_roundtrip(self) -> Mapping[str, Any]:
        """Require the explicit B=1 snapshot save/load diagnostic to be exact."""

        verifier = getattr(self._env, "verify_replay_probe_snapshot_roundtrip", None)
        if not callable(verifier):
            raise GpuNativeBackendUnavailableError(
                "SE3-WAM GPU backend lacks the T3 snapshot roundtrip seam"
            )
        receipt = verifier()
        if not isinstance(receipt, Mapping) or receipt.get("exact") is not True:
            raise GpuNativeBackendUnavailableError(
                "T3 snapshot roundtrip did not produce an exact receipt"
            )
        return receipt

    def replay_planner_tape_diagnostic(
        self,
        tape: Any | None = None,
    ) -> tuple[tuple[Any, ...], ...]:
        """Run explicitly labeled frozen-action replay on the GPU for diagnostics."""
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.replay_tape_diagnostic(tape)

    @property
    def last_planner_actions(self) -> Any:
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.last_actions

    @property
    def last_planner_seconds(self) -> float:
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.last_planner_seconds

    @property
    def last_environment_seconds(self) -> float:
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.last_environment_seconds

    def close(self) -> None:
        self._env.close()

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError("GpuNativeBackendEnv is device-resident and not picklable")


class GpuNativeMultiExportBackendEnv:
    """Run lane-local frozen exports on one CUDA device.

    The canonical MJWarp engine currently admits only a full-batch reset and
    one artifact identity per engine. D32 still needs distinct reset factors per
    episode, so this wrapper owns one batch-size-one GPU engine per lane and
    presents the same lane-oriented adapter surface to the live Planner. This
    is deliberately a composition of GPU-native engines, never a CPU physics
    or CPU environment fallback.
    """

    def __init__(
        self,
        *,
        task_id: str,
        export_dirs: Sequence[str],
        device_ordinal: int = 0,
        image_size: int = 64,
        observation_track: Any = "hybrid",
        task_quality_evaluator_backend_id: str | None = None,
        task_quality_schema_version: str | None = None,
    ) -> None:
        dirs = tuple(export_dirs)
        if not dirs:
            raise ValueError("GPU-native multi-export backend requires at least one export")
        if any(not isinstance(path, str) or not path.strip() for path in dirs):
            raise ValueError("GPU-native multi-export paths must be non-empty strings")
        if len(set(dirs)) != len(dirs):
            raise ValueError("GPU-native multi-export paths must be unique")
        self._task_id = task_id
        self._export_dirs = dirs
        self._backends: list[GpuNativeBackendEnv] = []
        try:
            self._backends = [
                GpuNativeBackendEnv(
                    task_id=task_id,
                    num_envs=1,
                    export_dir=path,
                    device_ordinal=device_ordinal,
                    image_size=image_size,
                    observation_track=observation_track,
                    task_quality_evaluator_backend_id=task_quality_evaluator_backend_id,
                    task_quality_schema_version=task_quality_schema_version,
                    require_exact_export_identity=True,
                )
                for path in dirs
            ]
        except BaseException:
            for backend in reversed(self._backends):
                backend.close()
            raise
        self._frozen_requests = tuple(
            backend.frozen_request for backend in self._backends
        )
        provenance = self._backends[0].provenance
        identity = (
            getattr(provenance, "physical_device_uuid", None),
            getattr(provenance, "physical_device_pci_bus_id", None),
            getattr(provenance, "device_ordinal", None),
        )
        for backend in self._backends[1:]:
            other = backend.provenance
            if (
                getattr(other, "physical_device_uuid", None),
                getattr(other, "physical_device_pci_bus_id", None),
                getattr(other, "device_ordinal", None),
            ) != identity:
                self.close()
                raise GpuNativeBackendUnavailableError(
                    "lane-local GPU exports resolved to different physical devices"
                )

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def num_envs(self) -> int:
        return len(self._backends)

    @property
    def export_dirs(self) -> tuple[str, ...]:
        return self._export_dirs

    @property
    def backend_id(self) -> str:
        return self._backends[0].backend_id

    @property
    def observation_track(self) -> Any:
        return self._backends[0].observation_track

    @property
    def provenance(self) -> Any:
        return self._backends[0].provenance

    @property
    def frozen_requests(self) -> tuple[Any, ...]:
        return self._frozen_requests

    def next_request(self) -> Any:
        """Return an exact export-bound request for compatibility callers."""

        index = getattr(self, "_next_request_index", 0)
        request = self._frozen_requests[index % self.num_envs]
        self._next_request_index = index + 1
        return request

    def reset(self, requests: Sequence[Any]) -> tuple[Any, ...]:
        request_tuple = tuple(requests)
        if len(request_tuple) != self.num_envs:
            raise ValueError("GPU-native multi-export reset requires one request per lane")
        # Validate the complete cohort before resetting even lane zero.  A late
        # mismatch must not leave a partially advanced multi-export batch.
        for lane, (backend, request) in enumerate(
            zip(self._backends, request_tuple, strict=True)
        ):
            require_exact_reset_request(
                request,
                backend.frozen_request,
                context=f"GPU multi-export lane {lane}",
            )
        observations = []
        for backend, request in zip(self._backends, request_tuple, strict=True):
            lane_observations = backend.reset((request,))
            if len(lane_observations) != 1:
                raise RuntimeError("lane-local GPU reset changed its cardinality")
            observations.append(lane_observations[0])
        return tuple(observations)

    def step(
        self,
        commands: Sequence[Any | None],
        *,
        active_mask: Any | None = None,
    ) -> tuple[Any | None, ...]:
        command_tuple = tuple(commands)
        if len(command_tuple) != self.num_envs:
            raise ValueError("GPU-native multi-export step requires one command per lane")
        if active_mask is None:
            mask = np.ones(self.num_envs, dtype=np.bool_)
        else:
            mask = np.asarray(active_mask)
            if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
                raise ValueError(
                    f"active_mask must be bool shape ({self.num_envs},)"
                )
        results: list[Any | None] = [None] * self.num_envs
        for lane, (backend, command, active) in enumerate(
            zip(self._backends, command_tuple, mask, strict=True)
        ):
            if not active:
                if command is not None:
                    raise ValueError(f"inactive GPU lane {lane} must carry None")
                continue
            lane_results = backend.step(
                (command,),
                active_mask=np.asarray([True], dtype=np.bool_),
            )
            if len(lane_results) != 1:
                raise RuntimeError("lane-local GPU step changed its cardinality")
            results[lane] = lane_results[0]
        return tuple(results)

    def materialize_terminal_ledger(
        self,
        lanes: Sequence[int],
        episode_ids: Sequence[str],
    ) -> Any:
        from dataclasses import replace as dataclass_replace

        from se3_wam.benchmark.gpu_native.audit import TerminalLedgerBatch

        lane_tuple = tuple(int(lane) for lane in lanes)
        episode_tuple = tuple(str(episode_id) for episode_id in episode_ids)
        if len(lane_tuple) != len(episode_tuple) or not lane_tuple:
            raise ValueError("terminal ledger lanes and episode IDs must be aligned")
        rows = []
        for lane, episode_id in zip(lane_tuple, episode_tuple, strict=True):
            if not 0 <= lane < self.num_envs:
                raise ValueError(f"terminal ledger lane {lane} is outside the multi-export batch")
            batch = self._backends[lane].materialize_terminal_ledger(
                (0,), (episode_id,)
            )
            if len(batch.rows) != 1:
                raise RuntimeError("lane-local terminal ledger changed its cardinality")
            rows.append(dataclass_replace(batch.rows[0], lane=lane))
        return TerminalLedgerBatch(
            backend_id=self.backend_id,
            provenance=self.provenance,
            rows=tuple(rows),
        )

    def mark_done(self, done_mask: Any) -> None:
        mask = np.asarray(done_mask)
        if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
            raise ValueError(f"done_mask must be bool shape ({self.num_envs},)")
        for lane, backend in enumerate(self._backends):
            if mask[lane]:
                backend.mark_done(np.asarray([True], dtype=np.bool_))

    def close(self) -> None:
        for backend in reversed(self._backends):
            backend.close()

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError("GpuNativeMultiExportBackendEnv is device-resident and not picklable")


@dataclass(frozen=True)
class PlannerTapeReplay:
    """Explicit open-loop diagnostic replay result, never a live Planner mode."""

    step_results: tuple[tuple[Any | None, ...], ...]
    terminal_rows: tuple[Any, ...]
    mode: str = "open_loop_diagnostic"


class GpuNativePlannerAdapter:
    """Run the causal privileged T4 Planner against current GPU observations.

    The teacher is constructed separately for every exact ResetRequest and
    receives the current host-materialized observation after every CUDA control
    interval. Recorded action tapes are an audit artifact; the live step path
    always calls teacher.act and never consumes a frozen tape.
    """

    def __init__(
        self,
        *,
        task_id: str,
        num_envs: int,
        export_dir: str | None = None,
        export_dirs: Sequence[str] | None = None,
        device_ordinal: int = 0,
        image_size: int = 64,
        observation_track: Any = "hybrid",
        evaluator_backend_id: str,
        schema_version: str | None = None,
    ) -> None:
        self._task_id = task_id
        if (export_dir is None) == (export_dirs is None):
            raise ValueError("provide exactly one of export_dir or export_dirs")
        quality_schema = (
            schema_version
            if schema_version is not None
            else "db0-episode-task-quality-v1"
        )
        if export_dirs is not None:
            export_dirs = tuple(export_dirs)
            if len(export_dirs) != num_envs:
                raise ValueError("export_dirs length must equal num_envs")
            self._backend = GpuNativeMultiExportBackendEnv(
                task_id=task_id,
                export_dirs=export_dirs,
                device_ordinal=device_ordinal,
                image_size=image_size,
                observation_track=observation_track,
                task_quality_evaluator_backend_id=evaluator_backend_id,
                task_quality_schema_version=quality_schema,
            )
        else:
            assert export_dir is not None
            self._backend = GpuNativeBackendEnv(
                task_id=task_id,
                num_envs=num_envs,
                export_dir=export_dir,
                device_ordinal=device_ordinal,
                image_size=image_size,
                observation_track=observation_track,
                task_quality_evaluator_backend_id=evaluator_backend_id,
                task_quality_schema_version=quality_schema,
            )
        self._teachers: tuple[Any, ...] = ()
        self._teacher_metadata: tuple[Mapping[str, Any], ...] = ()
        self._requests: tuple[Any, ...] = ()
        self._observations: tuple[Any, ...] | None = None
        self._active_mask = np.zeros(num_envs, dtype=np.bool_)
        self._action_tapes: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
        self._terminal_rows: list[Any] = []

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def num_envs(self) -> int:
        return self._backend.num_envs

    @property
    def backend_id(self) -> str:
        return self._backend.backend_id

    @property
    def observation_track(self) -> Any:
        return self._backend.observation_track

    @property
    def active_mask(self) -> np.ndarray:
        return self._active_mask.copy()

    @property
    def provenance(self) -> Any:
        return self._backend.provenance

    @property
    def replay(self) -> bool:
        """Live Planner mode is never tape replay."""

        return False

    @property
    def observations(self) -> tuple[Any, ...] | None:
        return self._observations

    @property
    def teacher_metadata(self) -> tuple[Mapping[str, Any], ...]:
        return self._teacher_metadata

    @property
    def action_tapes(self) -> tuple[tuple[Mapping[str, Any], ...], ...]:
        return tuple(tuple(dict(entry) for entry in tape) for tape in self._action_tapes)

    @property
    def terminal_rows(self) -> tuple[Any, ...]:
        return tuple(self._terminal_rows)

    @property
    def action_tape_sha256(self) -> str:
        payload = {
            "task_id": self._task_id,
            "episode_ids": [
                getattr(request, "episode_id", None) for request in self._requests
            ],
            "lanes": self._action_tapes,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def frozen_requests(self) -> tuple[Any, ...]:
        """Return the exact request identity bound by each export artifact."""

        return tuple(self._backend.frozen_requests)

    def next_request(self) -> Any:
        return self._backend.next_request()

    def reset(self, requests: Sequence[Any]) -> tuple[Any, ...]:
        request_tuple = tuple(requests)
        if len(request_tuple) != self.num_envs:
            raise ValueError("GPU Planner reset requires one request per lane")
        try:
            from se3_wam.benchmark.teacher_factory import make_privileged_teacher
        except ImportError as exc:
            raise GpuNativeBackendUnavailableError(
                "GPU Planner requires the pinned SE3-WAM teacher factory"
            ) from exc

        teachers: list[Any] = []
        metadata: list[Mapping[str, Any]] = []
        for request in request_tuple:
            teacher, teacher_metadata = make_privileged_teacher(
                self._task_id,
                request=request,
            )
            if not callable(getattr(teacher, "act", None)) or not callable(
                getattr(teacher, "reset", None)
            ):
                raise GpuNativeBackendUnavailableError(
                    "per-reset Planner factory returned a non-causal teacher"
                )
            teachers.append(teacher)
            metadata.append(dict(teacher_metadata))

        observations = self._backend.reset(request_tuple)
        for teacher in teachers:
            teacher.reset()
        for request, observation in zip(request_tuple, observations, strict=True):
            if observation.episode_id != request.episode_id:
                raise RuntimeError("GPU Planner reset changed the episode identity")
            if observation.task_id != self._task_id:
                raise RuntimeError("GPU Planner reset changed the task identity")
        self._teachers = tuple(teachers)
        self._teacher_metadata = tuple(metadata)
        self._requests = request_tuple
        self._observations = tuple(observations)
        self._active_mask = np.ones(self.num_envs, dtype=np.bool_)
        self._action_tapes = [[] for _ in range(self.num_envs)]
        self._terminal_rows = []
        return self._observations

    @staticmethod
    def _action_record(action: Any) -> dict[str, Any]:
        return {
            "mode": action.mode.value,
            "policy_step": int(action.policy_step),
            "values": [
                float(value)
                for value in np.asarray(action.values, dtype=np.float64)
            ],
        }

    def step(self) -> tuple[Any | None, ...]:
        if self._observations is None:
            raise RuntimeError("GPU Planner must be reset before stepping")
        if not np.any(self._active_mask):
            raise RuntimeError("all GPU Planner lanes are terminal; reset is required")
        try:
            from se3_wam.benchmark.api import ActionCommand
            from se3_wam.benchmark.contracts import ActionMode
        except ImportError as exc:
            raise GpuNativeBackendUnavailableError(
                "GPU Planner requires the pinned SE(3)-WAM public action ABI"
            ) from exc

        commands: list[Any | None] = [None] * self.num_envs
        for lane, (teacher, observation, active) in enumerate(
            zip(self._teachers, self._observations, self._active_mask, strict=True)
        ):
            if not active:
                continue
            action = teacher.act(observation)
            if type(action) is not ActionCommand:
                raise RuntimeError("Planner teacher must return the exact ActionCommand ABI")
            if action.mode is not ActionMode.E7:
                raise RuntimeError("GPU Planner requires E7 actions")
            if action.policy_step != observation.policy_step:
                raise RuntimeError(
                    f"Planner lane {lane} action policy_step does not match current observation"
                )
            commands[lane] = action
            self._action_tapes[lane].append(self._action_record(action))

        results = self._backend.step(commands, active_mask=self._active_mask)
        if len(results) != self.num_envs:
            raise RuntimeError("GPU Planner backend changed the lane cardinality")
        updated = list(self._observations)
        done_mask = np.zeros(self.num_envs, dtype=np.bool_)
        for lane, (result, active) in enumerate(zip(results, self._active_mask, strict=True)):
            if not active:
                continue
            if result is None:
                raise RuntimeError(f"GPU Planner lost active lane {lane} StepResult")
            observation = result.observation
            if observation.episode_id != self._requests[lane].episode_id:
                raise RuntimeError(f"GPU Planner lane {lane} episode identity drifted")
            updated[lane] = observation
            if result.terminated or result.truncated:
                ledger = self._backend.materialize_terminal_ledger(
                    (lane,),
                    (observation.episode_id,),
                )
                if len(ledger.rows) != 1 or ledger.rows[0].lane != lane:
                    raise RuntimeError("GPU Planner terminal ledger changed lane identity")
                if ledger.rows[0].episode_id != observation.episode_id:
                    raise RuntimeError("GPU Planner terminal ledger changed episode identity")
                self._terminal_rows.extend(ledger.rows)
                done_mask[lane] = True
        if np.any(done_mask):
            self._backend.mark_done(done_mask)
            self._active_mask &= ~done_mask
        self._observations = tuple(updated)
        return results

    def close(self) -> None:
        self._backend.close()

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError("GpuNativePlannerAdapter is device-resident and not picklable")


def replay_recorded_tape(
    backend: GpuNativeBackendEnv,
    requests: Sequence[Any],
    tapes: Sequence[Sequence[Mapping[str, Any]]],
) -> PlannerTapeReplay:
    """Replay a completed tape as an explicitly open-loop diagnostic.

    This helper never constructs or calls a teacher and is intentionally
    separate from GpuNativePlannerAdapter.step; callers must label its output
    as replay evidence rather than closed-loop Planner evidence.
    """

    request_tuple = tuple(requests)
    tape_tuple = tuple(tuple(dict(entry) for entry in tape) for tape in tapes)
    if len(request_tuple) != backend.num_envs or len(tape_tuple) != backend.num_envs:
        raise ValueError("replay requests and tapes must cover the whole GPU batch")
    try:
        from se3_wam.benchmark.api import ActionCommand
        from se3_wam.benchmark.contracts import ActionMode
    except ImportError as exc:
        raise GpuNativeBackendUnavailableError(
            "GPU tape replay requires the pinned SE(3)-WAM public action ABI"
        ) from exc

    observations = list(backend.reset(request_tuple))
    active_mask = np.ones(backend.num_envs, dtype=np.bool_)
    consumed = [0] * backend.num_envs
    replay_steps: list[tuple[Any | None, ...]] = []
    terminal_rows: list[Any] = []
    max_tape_length = max((len(tape) for tape in tape_tuple), default=0)
    for time_index in range(max_tape_length):
        commands: list[Any | None] = [None] * backend.num_envs
        for lane, active in enumerate(active_mask):
            if not active:
                continue
            if time_index >= len(tape_tuple[lane]):
                raise RuntimeError(
                    f"open-loop tape lane {lane} ended before its GPU terminal state"
                )
            record = tape_tuple[lane][time_index]
            if record.get("mode") != ActionMode.E7.value:
                raise ValueError("open-loop replay tape contains a non-E7 action")
            command = ActionCommand(
                mode=ActionMode.E7,
                values=np.asarray(record.get("values"), dtype=np.float64),
                policy_step=int(record.get("policy_step")),
            )
            if command.policy_step != observations[lane].policy_step:
                raise RuntimeError(
                    f"open-loop replay lane {lane} policy_step does not match the current GPU clock"
                )
            commands[lane] = command
            consumed[lane] += 1
        results = backend.step(commands, active_mask=active_mask)
        replay_steps.append(tuple(results))
        done_mask = np.zeros(backend.num_envs, dtype=np.bool_)
        for lane, (result, active) in enumerate(zip(results, active_mask, strict=True)):
            if not active:
                continue
            if result is None:
                raise RuntimeError(f"open-loop replay lost active lane {lane}")
            observations[lane] = result.observation
            if result.terminated or result.truncated:
                ledger = backend.materialize_terminal_ledger(
                    (lane,),
                    (observations[lane].episode_id,),
                )
                terminal_rows.extend(ledger.rows)
                done_mask[lane] = True
        if np.any(done_mask):
            backend.mark_done(done_mask)
            active_mask &= ~done_mask
        if not np.any(active_mask):
            break
    if np.any(active_mask):
        raise RuntimeError("open-loop replay tape did not reach terminal state for every lane")
    if any(count != len(tape) for count, tape in zip(consumed, tape_tuple, strict=True)):
        raise RuntimeError("open-loop replay tape contains actions after terminal state")
    return PlannerTapeReplay(
        step_results=tuple(replay_steps),
        terminal_rows=tuple(terminal_rows),
    )
