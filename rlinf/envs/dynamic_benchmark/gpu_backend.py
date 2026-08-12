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

For ``planner_mode='online_privileged_teacher_v1'`` the adapter delegates the
control-plane loop to the SE(3)-WAM keyed-puck Planner seam.  The Planner runs
on CPU by design, but every action is computed from the latest GPU observation
packet and immediately submitted to that same ``mjwarp_gpu_v1`` environment.
The runtime feature manifest and evaluator identity are mandatory in this mode.

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
        task_quality_schema_version: str | None = None,
        evaluator_backend_id: str | None = None,
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
        if planner_enabled and task_id != "t5_replan":
            raise ValueError("online privileged Planner mode is registered only for t5_replan")
        if planner_enabled:
            if not isinstance(runtime_manifest_path, str) or not runtime_manifest_path.strip():
                raise ValueError("online Planner mode requires runtime_manifest_path")
            if not isinstance(runtime_manifest_sha256, str) or not runtime_manifest_sha256.strip():
                raise ValueError("online Planner mode requires runtime_manifest_sha256")
            if not isinstance(task_quality_schema_version, str) or not task_quality_schema_version.strip():
                raise ValueError("online Planner mode requires task_quality_schema_version")
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
        self._task_id = task_id
        self._num_envs = num_envs
        self._export_dir = export_dir
        self._consumer = GpuNativeConsumer.RL
        runtime_evidence = ()
        if planner_enabled:
            try:
                from se3_wam.benchmark.gpu_native.tasks import load_runtime_feature_evidence

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
        observation_track = (
            ObservationTrack.HYBRID
            if planner_enabled or camera_observations
            else ObservationTrack.STATE
        )
        self._env = make_gpu_native_env(
            task_id,
            consumer=self._consumer,
            batch_size=num_envs,
            observation_track=observation_track,
            export_dir=export_dir,
            device_ordinal=device_ordinal,
            image_size=image_size,
            runtime_evidence=runtime_evidence,
        )
        self._artifacts = load_p0_grasp_artifacts(export_dir)
        self._frozen_request = self._artifacts.reset_request
        if self._frozen_request.task_id != task_id:
            raise GpuNativeBackendUnavailableError(
                "frozen export task_id does not match the requested task"
            )
        self._observation_track = observation_track
        self._planner_mode = planner_mode
        self._planner = None
        if planner_enabled:
            try:
                from se3_wam.benchmark.gpu_native.keyed_puck_planner import (
                    KeyedPuckReplanPlanner,
                )

                self._env.enable_task_quality(
                    evaluator_backend_id=evaluator_backend_id,
                    schema_version=task_quality_schema_version,
                )
                self._planner = KeyedPuckReplanPlanner(
                    gpu_env=self._env,
                    task_id=task_id,
                    image_size=image_size,
                )
            except Exception as exc:
                raise GpuNativeBackendUnavailableError(
                    "online T5 Planner mode could not establish the GPU quality/Planner seam"
                ) from exc
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

        def make_tape(request: Any) -> Any:
            branch = t5_branch_for_episode(request.episode_id)
            return T5EventTape(
                event_id=f"rlinf-{request.split.value}-{request.episode_id}-{branch}",
                branch=branch,
                trigger_gate_y_m=trigger_gate_y_m,
            )

        tapes = tuple(make_tape(request) for request in requests)
        arm_event_tapes(tapes)

    def policy_steps(self) -> Any:
        """Return the host clock policy steps expected for the next commands."""
        import numpy as np

        return np.asarray(
            self._env.bookkeeping.clock.policy_steps,
            dtype=np.int64,
        ).copy()

    def reset(self, requests: Any) -> tuple[Any, ...]:
        """Reset the whole batch and materialize every active lane."""
        request_values = tuple(requests)
        if len(request_values) != self._num_envs:
            raise ValueError("GPU-native reset requires one request per lane")
        if self._planner is not None:
            return self._planner.reset(request_values)
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        self._env.reset(request_values)
        self._arm_t5_event_tapes(request_values)
        audit = self._env.materialize_audit(
            AuditRequest(lanes=tuple(range(self._num_envs)), include_step_result=False)
        )
        observations = tuple(lane.observation for lane in audit.lanes)
        if len(observations) != self._num_envs:
            raise RuntimeError("GPU-native reset lost lane observations")
        return observations

    def step(self, commands: Any) -> tuple[Any, ...]:
        """Step the whole batch and materialize every active lane."""
        if self._planner is not None:
            raise GpuNativeBackendUnavailableError(
                "online Planner mode rejects external action batches; use step_planner()"
            )
        command_values = tuple(commands)
        if len(command_values) != self._num_envs:
            raise ValueError("GPU-native step requires one command per lane")
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        self._env.step(command_values)
        audit = self._env.materialize_audit(
            AuditRequest(lanes=tuple(range(self._num_envs)), include_step_result=True)
        )
        results = tuple(lane.step_result for lane in audit.lanes)
        if len(results) != self._num_envs or any(value is None for value in results):
            raise RuntimeError("GPU-native step lost lane step results")
        return results

    def step_planner(self) -> tuple[Any, ...]:
        """Advance one online CPU-Planner/GPU-physics control interval."""
        if self._planner is None:
            raise GpuNativeBackendUnavailableError("online Planner mode is not enabled")
        return self._planner.step()

    def planner_tape(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        """Return an append-only audit tape, never an action input."""
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

    def close(self) -> None:
        self._env.close()

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError("GpuNativeBackendEnv is device-resident and not picklable")
