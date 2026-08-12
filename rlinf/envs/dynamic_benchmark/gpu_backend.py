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
        if planner_mode not in {None, "online_privileged_teacher_v1"}:
            raise ValueError("planner_mode is not a registered GPU Planner mode")
        planner_enabled = planner_mode is not None
        if planner_enabled and task_id not in {"t1_belt", "t1_occ"}:
            raise ValueError("online privileged Planner mode is registered only for surface T1 tasks")
        if planner_enabled:
            if not isinstance(runtime_manifest_path, str) or not runtime_manifest_path.strip():
                raise ValueError("online Planner mode requires runtime_manifest_path")
            if not isinstance(runtime_manifest_sha256, str) or not runtime_manifest_sha256.strip():
                raise ValueError("online Planner mode requires runtime_manifest_sha256")
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
        self._planner_mode = planner_mode
        self._planner = None
        if task_quality_schema_version is not None:
            self._env.enable_task_quality(
                evaluator_backend_id=task_quality_evaluator_backend_id,
                schema_version=task_quality_schema_version,
            )
        elif planner_enabled:
            self._env.enable_task_quality(evaluator_backend_id=evaluator_backend_id)
        self._task_quality_enabled = bool(
            task_quality_schema_version is not None or planner_enabled
        )
        if planner_enabled:
            try:
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
    def last_terminal_ledger(self) -> Any | None:
        """Return the one-shot terminal receipt produced by the latest step."""

        return self._last_terminal_ledger

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
        )

    def next_request(self) -> Any:
        """Return the frozen export request with a fresh per-lane episode id.

        The configured observation track is carried through to the GPU engine;
        the wrapper validates requests against the frozen artifact identity.
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
        requests_tuple = tuple(requests)
        if len(requests_tuple) != self._num_envs:
            raise ValueError("GPU-native reset requires one request per lane")
        if self._planner is not None:
            return self._planner.reset(requests_tuple)
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        self._env.reset(requests_tuple)
        audit = self._env.materialize_audit(
            AuditRequest(lanes=tuple(range(self._num_envs)), include_step_result=False)
        )
        observations = tuple(lane.observation for lane in audit.lanes)
        if len(observations) != self._num_envs:
            raise RuntimeError("GPU-native reset lost lane observations")
        self._last_terminal_ledger = None
        self._action_tapes = [[] for _ in range(self._num_envs)]
        self._observation_tapes = [[observation] for observation in observations]
        return observations

    def step(self, commands: Any) -> tuple[Any, ...]:
        """Step the whole batch and materialize every active lane."""
        commands_tuple = tuple(commands)
        if self._planner is not None:
            raise GpuNativeBackendUnavailableError(
                "online Planner mode rejects external action batches; use step_planner()"
            )
        if len(commands_tuple) != self._num_envs:
            raise ValueError("GPU-native step requires one command per lane")
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        self._env.step(commands_tuple)
        audit = self._env.materialize_audit(
            AuditRequest(lanes=tuple(range(self._num_envs)), include_step_result=True)
        )
        results = tuple(lane.step_result for lane in audit.lanes)
        if len(results) != self._num_envs or any(value is None for value in results):
            raise RuntimeError("GPU-native step lost lane step results")
        self._last_terminal_ledger = None
        terminal_lanes = tuple(
            lane
            for lane, result in enumerate(results)
            if bool(getattr(result, "terminated", False))
            or bool(getattr(result, "truncated", False))
        )
        if self._task_quality_enabled and terminal_lanes:
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
        for lane, (command, result) in enumerate(zip(commands_tuple, results, strict=True)):
            values = getattr(command, "values", None)
            self._action_tapes[lane].append(
                values.copy() if hasattr(values, "copy") else values
            )
            self._observation_tapes[lane].append(result.observation)
        return results

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
