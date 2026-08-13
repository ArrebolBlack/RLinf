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

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class GpuNativePlannerStep:
    """One online CPU-Planner control interval over the CUDA batch.

    ``observations`` are materialized before the teacher acts and ``results``
    are materialized after the same GPU environment advances.  Keeping both in
    the record makes the action/trajectory tape auditable without making the
    tape part of the online decision path.
    """

    step_index: int
    observations: tuple[Any, ...]
    actions: tuple[Any, ...]
    results: tuple[Any, ...]

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise ValueError("planner step_index must be an integer")
        if self.step_index < 0:
            raise ValueError("planner step_index must be non-negative")
        if not self.observations or not (
            len(self.observations) == len(self.actions) == len(self.results)
        ):
            raise ValueError("planner step observations, actions, and results must align")


@dataclass(frozen=True)
class GpuNativePlannerTape:
    """Immutable audit tape produced by online Planner control.

    Replaying this tape is explicitly diagnostic frozen-action replay.  It is
    never used by :meth:`GpuNativeBackendEnv.planner_step`, which calls the
    teacher on the freshly materialized current observation every time.
    """

    requests: tuple[Any, ...]
    initial_observations: tuple[Any, ...]
    steps: tuple[GpuNativePlannerStep, ...]
    teacher_metadata: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.requests or len(self.requests) != len(self.initial_observations):
            raise ValueError("planner tape requests and initial observations must align")
        if len(self.teacher_metadata) != len(self.requests):
            raise ValueError("planner tape teacher metadata must align with requests")
        expected_index = 0
        for step in self.steps:
            if step.step_index != expected_index:
                raise ValueError("planner tape step indices must be contiguous")
            expected_index += 1


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
        observation_track: str | Any | None = None,
        render_observations: bool = False,
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
        if not isinstance(render_observations, bool):
            raise TypeError("render_observations must be a boolean")
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
        if observation_track is None:
            selected_observation_track = ObservationTrack.STATE
        elif isinstance(observation_track, str):
            try:
                selected_observation_track = ObservationTrack(observation_track)
            except ValueError as exc:
                raise ValueError(
                    "observation_track must be state, visual, or hybrid"
                ) from exc
        elif isinstance(observation_track, ObservationTrack):
            selected_observation_track = observation_track
        else:
            raise TypeError("observation_track must be an ObservationTrack or string")
        if (task_quality_schema_version is None) != (
            task_quality_evaluator_backend_id is None
        ):
            raise ValueError(
                "task quality schema and evaluator backend id must be supplied together"
            )
        self._task_id = task_id
        self._num_envs = num_envs
        self._export_dir = export_dir
        self._image_size = image_size
        self._consumer = GpuNativeConsumer.RL
        self._observation_track = selected_observation_track
        self._render_observations = render_observations
        self._env = make_gpu_native_env(
            task_id,
            consumer=self._consumer,
            batch_size=num_envs,
            observation_track=selected_observation_track,
            export_dir=export_dir,
            device_ordinal=device_ordinal,
            image_size=image_size,
            render_observations=render_observations,
        )
        if task_quality_schema_version is not None:
            enable_task_quality = getattr(self._env, "enable_task_quality", None)
            if not callable(enable_task_quality):
                raise GpuNativeBackendUnavailableError(
                    "GPU-native backend does not expose the task-quality evaluator seam"
                )
            enable_task_quality(
                evaluator_backend_id=task_quality_evaluator_backend_id,
                schema_version=task_quality_schema_version,
            )
        self._artifacts = load_p0_grasp_artifacts(export_dir)
        self._frozen_request = self._artifacts.reset_request
        if self._frozen_request.task_id != task_id:
            raise GpuNativeBackendUnavailableError(
                "frozen export task_id does not match the requested task"
            )
        self._episode_counter = 0
        self._last_observations: tuple[Any, ...] | None = None
        self._last_results: tuple[Any, ...] | None = None
        self._active_requests: tuple[Any, ...] | None = None
        self._teachers: tuple[Any, ...] | None = None
        self._teacher_metadata: tuple[Mapping[str, Any], ...] = ()
        self._planner_initial_observations: tuple[Any, ...] | None = None
        self._planner_tape_steps: list[GpuNativePlannerStep] = []
        self._terminal_ledger_consumed: set[str] = set()

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
    def render_observations(self) -> bool:
        return self._render_observations

    @property
    def last_observations(self) -> tuple[Any, ...] | None:
        """Return the latest materialized observation tuple, if reset."""

        return self._last_observations

    @property
    def teacher_metadata(self) -> tuple[Mapping[str, Any], ...]:
        """Return per-lane teacher preparation metadata for the active cohort."""

        return self._teacher_metadata

    def next_request(self) -> Any:
        """Return the frozen export request with a fresh per-lane episode id.

        The observation track is normalized to the adapter's selected contract;
        the wrapper validates requests before the engine re-normalizes them to
        the frozen artifact identity internally.
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

    def _validate_requests(self, requests: Any) -> tuple[Any, ...]:
        selected = tuple(requests)
        if len(selected) != self._num_envs:
            raise ValueError("GPU-native reset requires one request per lane")
        if any(request is None for request in selected):
            raise ValueError("GPU-native reset requires a ResetRequest for every lane")
        for request in selected:
            if getattr(request, "task_id", None) != self._task_id:
                raise ValueError("GPU-native reset request task_id does not match the backend")
        return selected

    def _materialize_current_observations(self) -> tuple[Any, ...]:
        materializer = getattr(self._env, "materialize_current_observations", None)
        if callable(materializer):
            observations = tuple(materializer(tuple(range(self._num_envs))))
        else:
            from se3_wam.benchmark.gpu_native.audit import AuditRequest

            audit = self._env.materialize_audit(
                AuditRequest(lanes=tuple(range(self._num_envs)), include_step_result=False)
            )
            observations = tuple(lane.observation for lane in audit.lanes)
            if any(lane.step_result is not None for lane in audit.lanes):
                raise GpuNativeBackendUnavailableError(
                    "GPU current-state audit returned a StepResult"
                )
        if len(observations) != self._num_envs:
            raise GpuNativeBackendUnavailableError(
                "GPU current-state materialization lost lane observations"
            )
        return observations

    def _step_commands(self, commands: Any) -> tuple[Any, ...]:
        selected = tuple(commands)
        if len(selected) != self._num_envs:
            raise ValueError("GPU-native step requires one command per lane")
        from se3_wam.benchmark.gpu_native.audit import AuditRequest

        self._env.step(selected)
        audit = self._env.materialize_audit(
            AuditRequest(lanes=tuple(range(self._num_envs)), include_step_result=True)
        )
        results = tuple(lane.step_result for lane in audit.lanes)
        if len(results) != self._num_envs or any(value is None for value in results):
            raise GpuNativeBackendUnavailableError(
                "GPU-native step lost lane step results"
            )
        self._last_observations = tuple(result.observation for result in results)
        self._last_results = results
        return results

    def reset(self, requests: Any) -> tuple[Any, ...]:
        """Reset the whole batch and materialize every active lane."""
        selected = self._validate_requests(requests)
        self._env.reset(selected)
        observations = self._materialize_current_observations()
        self._active_requests = selected
        self._last_observations = observations
        self._last_results = None
        self._teachers = None
        self._teacher_metadata = ()
        self._planner_initial_observations = None
        self._planner_tape_steps = []
        self._terminal_ledger_consumed = set()
        return observations

    def step(self, commands: Any) -> tuple[Any, ...]:
        """Step the whole batch and materialize every active lane."""
        return self._step_commands(commands)

    def reset_planner(self, requests: Any) -> tuple[Any, ...]:
        """Reset and construct one causal teacher per exact episode request."""

        observations = self.reset(requests)
        selected = self._active_requests
        assert selected is not None
        try:
            from se3_wam.benchmark.teacher_factory import make_privileged_teacher
        except ImportError as exc:
            raise GpuNativeBackendUnavailableError(
                "GPU Planner mode requires the canonical SE3-WAM teacher factory"
            ) from exc
        teachers: list[Any] = []
        metadata: list[Mapping[str, Any]] = []
        for request in selected:
            teacher, preparation = make_privileged_teacher(
                request.task_id,
                request=request,
                image_size=self._image_size,
            )
            if not callable(getattr(teacher, "act", None)):
                raise GpuNativeBackendUnavailableError(
                    "canonical GPU Planner teacher does not expose act(observation)"
                )
            reset_teacher = getattr(teacher, "reset", None)
            if not callable(reset_teacher):
                raise GpuNativeBackendUnavailableError(
                    "canonical GPU Planner teacher does not expose reset()"
                )
            reset_teacher()
            teachers.append(teacher)
            metadata.append(MappingProxyType(dict(preparation)))
        self._teachers = tuple(teachers)
        self._teacher_metadata = tuple(metadata)
        self._planner_initial_observations = observations
        self._planner_tape_steps = []
        return observations

    def planner_step(self) -> GpuNativePlannerStep:
        """Run one closed-loop CPU Planner step against the live CUDA state.

        The teacher consumes ``last_observations`` produced by the preceding
        reset/step boundary.  The returned step is an audit record only; no
        stored action is ever consulted to choose the next Planner action.
        """

        if self._teachers is None or self._last_observations is None:
            raise RuntimeError("planner_step requires reset_planner before stepping")
        observations = self._last_observations
        actions = tuple(
            teacher.act(observation)
            for teacher, observation in zip(self._teachers, observations, strict=True)
        )
        results = self._step_commands(actions)
        step = GpuNativePlannerStep(
            step_index=len(self._planner_tape_steps),
            observations=observations,
            actions=actions,
            results=results,
        )
        self._planner_tape_steps.append(step)
        return step

    def planner_tape(self) -> GpuNativePlannerTape:
        """Return the immutable online Planner action/trajectory audit tape."""

        if self._teachers is None or self._planner_initial_observations is None:
            raise RuntimeError("planner_tape requires reset_planner before stepping")
        if self._active_requests is None:
            raise RuntimeError("planner_tape has no active reset requests")
        return GpuNativePlannerTape(
            requests=self._active_requests,
            initial_observations=self._planner_initial_observations,
            steps=tuple(self._planner_tape_steps),
            teacher_metadata=self._teacher_metadata,
        )

    @staticmethod
    def _observation_fingerprint(observation: Any) -> str | None:
        value = getattr(observation, "fingerprint_sha256", None)
        return value if isinstance(value, str) else None

    @classmethod
    def _observation_matches(cls, actual: Any, expected: Any) -> bool:
        actual_fingerprint = cls._observation_fingerprint(actual)
        expected_fingerprint = cls._observation_fingerprint(expected)
        return (
            actual_fingerprint is not None
            and expected_fingerprint is not None
            and actual_fingerprint == expected_fingerprint
        )

    @classmethod
    def _result_signature(cls, result: Any) -> tuple[Any, ...]:
        return (
            cls._observation_fingerprint(getattr(result, "observation", None)),
            getattr(result, "terminated", None),
            getattr(result, "truncated", None),
            getattr(result, "success", None),
            getattr(result, "termination_reason", None),
        )

    def replay_planner_tape_diagnostic(
        self,
        tape: GpuNativePlannerTape,
    ) -> Mapping[str, Any]:
        """Replay a tape for GPU determinism diagnostics only.

        This method deliberately uses recorded actions and never calls a
        teacher.  The report labels that fact explicitly, so its result cannot
        be mistaken for a closed-loop Planner result.
        """

        if not isinstance(tape, GpuNativePlannerTape):
            raise TypeError("tape must be GpuNativePlannerTape")
        if len(tape.requests) != self._num_envs:
            raise ValueError("planner tape batch size does not match the backend")
        initial = self.reset(tape.requests)
        initial_matches = tuple(
            self._observation_matches(actual, expected)
            for actual, expected in zip(initial, tape.initial_observations, strict=True)
        )
        step_reports: list[Mapping[str, Any]] = []
        for expected in tape.steps:
            current = self._last_observations
            assert current is not None
            pre_matches = tuple(
                self._observation_matches(actual, expected_observation)
                for actual, expected_observation in zip(
                    current, expected.observations, strict=True
                )
            )
            actual_results = self._step_commands(expected.actions)
            result_matches = tuple(
                self._result_signature(actual) == self._result_signature(recorded)
                for actual, recorded in zip(actual_results, expected.results, strict=True)
            )
            step_reports.append(
                MappingProxyType(
                    {
                        "step_index": expected.step_index,
                        "pre_observation_matches": pre_matches,
                        "result_matches": result_matches,
                    }
                )
            )
        passed = all(initial_matches) and all(
            all(report["pre_observation_matches"])
            and all(report["result_matches"])
            for report in step_reports
        )
        return MappingProxyType(
            {
                "mode": "diagnostic_frozen_action_replay",
                "closed_loop_planner": False,
                "initial_observation_matches": initial_matches,
                "steps": tuple(step_reports),
                "passed": bool(passed),
            }
        )

    def materialize_terminal_ledger_once(
        self,
        lanes: tuple[int, ...] | None = None,
    ) -> Any:
        """Consume terminal rows once, preserving GPU quality/terminal identity."""

        if self._active_requests is None or self._last_results is None:
            raise RuntimeError("terminal ledger requires an active stepped cohort")
        selected = tuple(range(self._num_envs)) if lanes is None else tuple(lanes)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("terminal ledger lanes must be non-empty and unique")
        if any(
            isinstance(lane, bool) or not isinstance(lane, int) or not 0 <= lane < self._num_envs
            for lane in selected
        ):
            raise ValueError("terminal ledger lane is outside the GPU batch")
        if any(
            not (
                self._last_results[lane].terminated
                or self._last_results[lane].truncated
            )
            for lane in selected
        ):
            raise RuntimeError("terminal ledger requires a terminal result for every lane")
        episode_ids = tuple(self._active_requests[lane].episode_id for lane in selected)
        if self._terminal_ledger_consumed.intersection(episode_ids):
            raise RuntimeError("terminal ledger row was requested more than once")
        from se3_wam.benchmark.gpu_native.audit import TerminalLedgerRequest

        ledger = self._env.materialize_terminal_ledger(
            TerminalLedgerRequest(lanes=selected, episode_ids=episode_ids)
        )
        self._terminal_ledger_consumed.update(episode_ids)
        return ledger

    def close(self) -> None:
        self._env.close()

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError("GpuNativeBackendEnv is device-resident and not picklable")


__all__ = [
    "GpuNativeBackendEnv",
    "GpuNativeBackendUnavailableError",
    "GpuNativePlannerStep",
    "GpuNativePlannerTape",
]
