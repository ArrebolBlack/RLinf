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

The GPU backend currently requires every lane to reuse the one frozen export
identity (task/seed/split/factors/...); only ``episode_id`` is free.  This
adapter therefore sources reset requests from the frozen export artifact and
never falls back to the CPU ``make_mujoco_env`` path.

``se3_wam`` imports are deliberately lazy so this module (and the tests that
exercise its request mapping) can be imported on machines without the SE3-WAM
source or the pinned MJWarp runtime.
"""

from __future__ import annotations

from dataclasses import replace
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
        task_quality_schema_version: str | None = None,
        task_quality_evaluator_backend_id: str | None = None,
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
        self._task_id = task_id
        self._num_envs = num_envs
        self._export_dir = export_dir
        self._consumer = GpuNativeConsumer.RL
        track = (
            ObservationTrack.STATE if observation_track is None else observation_track
        )
        if not isinstance(track, ObservationTrack):
            raise ValueError("observation_track must be a SE(3)-WAM ObservationTrack")
        self._env = make_gpu_native_env(
            task_id,
            consumer=self._consumer,
            batch_size=num_envs,
            observation_track=track,
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
        frozen_track = getattr(self._frozen_request.observation_track, "value", None)
        if frozen_track != track.value:
            raise GpuNativeBackendUnavailableError(
                "frozen export observation track does not match the GPU adapter track"
            )
        self._observation_track = track
        self._episode_counter = 0
        self._active_mask = np.ones(num_envs, dtype=np.bool_)
        self._episode_ids: tuple[str | None, ...] = (None,) * num_envs
        self._last_observations: tuple[Any | None, ...] = (None,) * num_envs
        self._last_terminal_rows: tuple[Any, ...] = ()
        self._task_quality_enabled = False
        if (task_quality_schema_version is None) != (
            task_quality_evaluator_backend_id is None
        ):
            raise ValueError(
                "GPU task quality requires both schema version and evaluator backend ID"
            )
        if task_quality_schema_version is not None:
            assert task_quality_evaluator_backend_id is not None
            self.enable_task_quality(
                evaluator_backend_id=task_quality_evaluator_backend_id,
                schema_version=task_quality_schema_version,
            )

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
    def backend_id(self) -> str:
        return self._env.backend_id

    @property
    def provenance(self) -> Any:
        return self._env.provenance

    @property
    def frozen_request(self) -> Any:
        return self._frozen_request

    @property
    def observation_track(self) -> Any:
        return self._observation_track

    @property
    def current_observations(self) -> tuple[Any | None, ...]:
        """Return the latest materialized observation for each lane."""

        return self._last_observations

    @property
    def last_terminal_rows(self) -> tuple[Any, ...]:
        """Return terminal rows materialized by the most recent step."""

        return self._last_terminal_rows

    @property
    def task_quality_enabled(self) -> bool:
        return self._task_quality_enabled

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

    def next_request(self) -> Any:
        """Return the frozen export request with a fresh per-lane episode id.

        The observation track is normalized to the adapter's STATE contract:
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
        self._active_mask = np.ones(self._num_envs, dtype=np.bool_)
        self._episode_ids = tuple(
            getattr(request, "episode_id", None) for request in request_tuple
        )
        self._last_observations = observations
        self._last_terminal_rows = ()
        return observations

    def step(self, commands: Any) -> tuple[Any, ...]:
        """Step selected active lanes from current E7 commands."""
        command_tuple = tuple(commands)
        if len(command_tuple) != self._num_envs:
            raise ValueError("GPU-native step requires one command per lane")
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        # The all-None case is retained for the legacy adapter contract tests;
        # real MjWarp batches must provide E7 commands for every selected lane.
        if all(value is None for value in command_tuple) and np.all(self._active_mask):
            selected_mask = np.array(self._active_mask, copy=True)
        else:
            selected_mask = self._active_mask & np.asarray(
                [value is not None for value in command_tuple], dtype=np.bool_
            )
        selected_lanes = tuple(int(value) for value in np.flatnonzero(selected_mask))
        if not selected_lanes:
            self._last_terminal_rows = ()
            return (None,) * self._num_envs
        if hasattr(self._env, "mark_done"):
            self._env.step(command_tuple, active_mask=selected_mask)
        else:
            self._env.step(command_tuple)
        audit = self._env.materialize_audit(
            AuditRequest(lanes=selected_lanes, include_step_result=True)
        )
        result_by_lane = {}
        for lane in audit.lanes:
            if lane.step_result is None:
                raise RuntimeError("GPU-native step lost a lane step result")
            result_by_lane[lane.lane] = lane.step_result
            self._last_observations = self._replace_observation(
                self._last_observations, lane.lane, lane.step_result.observation
            )

        terminal_rows: list[Any] = []
        terminal_method = getattr(self._env, "materialize_terminal_ledger", None)
        if terminal_method is not None:
            from se3_wam.benchmark.gpu_native.audit import TerminalLedgerRequest

            for lane in selected_lanes:
                result = result_by_lane[lane]
                if not bool(getattr(result, "terminated", False)) and not bool(
                    getattr(result, "truncated", False)
                ):
                    continue
                episode_id = self._episode_ids[lane]
                if episode_id is None:
                    raise RuntimeError("terminal lane has no reset episode identity")
                terminal_batch = terminal_method(
                    TerminalLedgerRequest(lanes=(lane,), episode_ids=(episode_id,))
                )
                terminal_rows.extend(terminal_batch.rows)
        self._last_terminal_rows = tuple(terminal_rows)

        done_mask = np.zeros(self._num_envs, dtype=np.bool_)
        for lane in selected_lanes:
            result = result_by_lane[lane]
            done_mask[lane] = bool(getattr(result, "terminated", False)) or bool(
                getattr(result, "truncated", False)
            )
        if np.any(done_mask):
            if hasattr(self._env, "mark_done"):
                self._env.mark_done(done_mask)
            self._active_mask &= ~done_mask

        results: list[Any | None] = [None] * self._num_envs
        for lane, result in result_by_lane.items():
            results[lane] = result
        return tuple(results)

    @staticmethod
    def _replace_observation(
        observations: tuple[Any | None, ...], lane: int, observation: Any
    ) -> tuple[Any | None, ...]:
        updated = list(observations)
        updated[lane] = observation
        return tuple(updated)

    def close(self) -> None:
        self._env.close()

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError("GpuNativeBackendEnv is device-resident and not picklable")
