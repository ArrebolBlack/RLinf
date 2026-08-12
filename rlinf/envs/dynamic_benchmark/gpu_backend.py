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

The GPU backend accepts either one homogeneous frozen export identity or a
lane-local tuple of frozen export artifacts. The latter is the D32 path: each
lane keeps its own task/seed/split/factors identity while every physics,
rendering, and observation operation remains on MJWarp CUDA. Neither path
falls back to the CPU ``make_mujoco_env`` environment.

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
        observation_track: Any | None = None,
        task_quality_evaluator_backend_id: str | None = None,
        task_quality_schema_version: str | None = None,
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
        if (task_quality_evaluator_backend_id is None) != (
            task_quality_schema_version is None
        ):
            raise ValueError(
                "task-quality evaluator identity and schema version must be supplied together"
            )
        if task_quality_evaluator_backend_id is not None and (
            not task_quality_evaluator_backend_id
            or task_quality_evaluator_backend_id.strip() != task_quality_evaluator_backend_id
        ):
            raise ValueError("task-quality evaluator identity must be non-empty and trimmed")
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
        try:
            selected_observation_track = (
                ObservationTrack.STATE
                if observation_track is None
                else ObservationTrack(observation_track)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "observation_track must be one of visual, state, or hybrid"
            ) from exc
        if selected_observation_track.value not in {"visual", "state", "hybrid"}:
            raise ValueError("future_oracle is forbidden for GPU Planner observations")
        self._task_id = task_id
        self._num_envs = num_envs
        self._export_dir = export_dir
        self._consumer = GpuNativeConsumer.RL
        self._observation_track = selected_observation_track
        self._env = make_gpu_native_env(
            task_id,
            consumer=self._consumer,
            batch_size=num_envs,
            observation_track=selected_observation_track,
            export_dir=export_dir,
            device_ordinal=device_ordinal,
            image_size=image_size,
        )
        self._artifacts = load_p0_grasp_artifacts(export_dir)
        self._frozen_request = self._artifacts.reset_request
        if self._frozen_request.task_id != task_id:
            raise GpuNativeBackendUnavailableError(
                "frozen export task_id does not match the requested task"
            )
        if task_quality_evaluator_backend_id is not None:
            enable_task_quality = getattr(self._env, "enable_task_quality", None)
            if not callable(enable_task_quality):
                raise GpuNativeBackendUnavailableError(
                    "GPU engine does not expose the task-quality admission seam"
                )
            enable_task_quality(
                evaluator_backend_id=task_quality_evaluator_backend_id,
                schema_version=task_quality_schema_version,
            )
        self._episode_counter = 0

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
    def backend_id(self) -> str:
        return self._env.backend_id

    @property
    def provenance(self) -> Any:
        return self._env.provenance

    @property
    def frozen_request(self) -> Any:
        return self._frozen_request

    @property
    def frozen_requests(self) -> tuple[Any, ...]:
        """Return the export-bound request for each homogeneous lane."""

        return tuple(self._frozen_request for _ in range(self._num_envs))

    def next_request(self) -> Any:
        """Return the frozen export request with a fresh per-lane episode id.

        The observation track is normalized to the adapter's selected contract:
        the wrapper validates requests against its contract before the engine
        re-normalizes them to the frozen artifact identity internally.
        """
        request = self._frozen_request
        episode_id = f"{self._task_id}-gpu-{self._episode_counter:08d}"
        self._episode_counter += 1
        return replace(
            request,
            episode_id=episode_id,
            observation_track=self._observation_track,
        )

    def policy_steps(self) -> Any:
        """Return the host clock policy steps expected for the next commands."""
        import numpy as np

        return np.asarray(
            self._env.bookkeeping.clock.policy_steps,
            dtype=np.int64,
        ).copy()

    def reset(self, requests: Any) -> tuple[Any, ...]:
        """Reset the whole batch and materialize every active lane."""
        request_tuple = tuple(requests)
        if len(request_tuple) != self._num_envs:
            raise ValueError("GPU-native reset requires one request per lane")
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        self._env.reset(request_tuple)
        audit = self._env.materialize_audit(
            AuditRequest(lanes=tuple(range(self._num_envs)), include_step_result=False)
        )
        observations = tuple(lane.observation for lane in audit.lanes)
        if len(observations) != self._num_envs:
            raise RuntimeError("GPU-native reset lost lane observations")
        return observations

    def step(self, commands: Any, *, active_mask: Any | None = None) -> tuple[Any, ...]:
        """Step selected lanes and materialize only their current observations.

        active_mask is optional for compatibility with the original
        homogeneous-batch adapter. A supplied mask is the explicit seam used
        by the live Planner adapter after one lane reaches a terminal state;
        inactive lanes must carry None commands and are never replayed.
        """
        command_tuple = tuple(commands)
        if len(command_tuple) != self._num_envs:
            raise ValueError("GPU-native step requires one command per lane")
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        explicit_mask = active_mask is not None
        if active_mask is None:
            mask = np.ones(self._num_envs, dtype=np.bool_)
        else:
            mask = np.asarray(active_mask)
            if mask.shape != (self._num_envs,) or mask.dtype != np.bool_:
                raise ValueError(
                    f"active_mask must be bool shape ({self._num_envs},)"
                )
            if not np.any(mask):
                raise ValueError("active_mask must select at least one lane")
            for lane, (command, active) in enumerate(zip(command_tuple, mask, strict=True)):
                if not active and command is not None:
                    raise ValueError(f"inactive GPU lane {lane} must carry None")

        if explicit_mask:
            self._env.step(command_tuple, active_mask=mask)
        else:
            self._env.step(command_tuple)
        audit = self._env.materialize_audit(
            AuditRequest(
                lanes=tuple(int(lane) for lane in np.flatnonzero(mask)),
                include_step_result=True,
            )
        )
        results: list[Any | None] = [None] * self._num_envs
        for lane in audit.lanes:
            if lane.step_result is None:
                raise RuntimeError(f"GPU-native step lost lane {lane.lane} step result")
            results[lane.lane] = lane.step_result
        if any(results[int(lane)] is None for lane in np.flatnonzero(mask)):
            raise RuntimeError("GPU-native step lost one or more active lane results")
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

        mark_done = getattr(self._env, "mark_done", None)
        if not callable(mark_done):
            raise GpuNativeBackendUnavailableError(
                "GPU environment does not expose explicit done bookkeeping"
            )
        mark_done(done_mask)

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
