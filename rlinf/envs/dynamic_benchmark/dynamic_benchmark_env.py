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

"""RLinf vector wrapper for the canonical SE(3)-WAM Dynamic Benchmark."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union

import gym
import numpy as np
import torch

from .reward import DynamicBenchmarkReward
from .state_schema import DynamicBenchmarkStateSchema
from .t5_runtime import arm_hidden_t5_event

__all__ = ["DynamicBenchmarkEnv"]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _torch_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _torch_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_torch_clone(item) for item in value]
    return copy.deepcopy(value)


def _pack_process_observation(observation: Any) -> dict[str, Any]:
    """Convert immutable mapping proxies into a pickle-safe observation payload."""

    return {
        "episode_id": observation.episode_id,
        "task_id": observation.task_id,
        "physics_step": observation.physics_step,
        "control_step": observation.control_step,
        "policy_step": observation.policy_step,
        "time_s": observation.time_s,
        "rgb": copy.deepcopy(dict(observation.rgb)),
        "depth_m": copy.deepcopy(dict(observation.depth_m)),
        "segmentation": copy.deepcopy(dict(observation.segmentation)),
        "proprio": copy.deepcopy(dict(observation.proprio)),
        "privileged": copy.deepcopy(dict(observation.privileged)),
        "events_since_last_observation": [
            {
                "name": event.name,
                "physics_step": event.physics_step,
                "time_s": event.time_s,
                "details": copy.deepcopy(dict(event.details)),
            }
            for event in observation.events_since_last_observation
        ],
        "api_version": observation.api_version,
    }


def _observation_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Fingerprint the pickle-safe observation snapshot stored in a checkpoint."""

    return hashlib.sha256(pickle.dumps(dict(payload), protocol=5)).hexdigest()


def _state_tensor_sha256(states: Any) -> str:
    tensor = torch.as_tensor(states).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _unpack_observation_payload(
    payload: Mapping[str, Any],
    *,
    observation_type: Any,
    event_type: Any,
) -> Any:
    events = tuple(
        event_type(
            name=str(event["name"]),
            physics_step=int(event["physics_step"]),
            time_s=float(event["time_s"]),
            details=copy.deepcopy(dict(event["details"])),
        )
        for event in payload["events_since_last_observation"]
    )
    return observation_type(
        episode_id=str(payload["episode_id"]),
        task_id=str(payload["task_id"]),
        physics_step=int(payload["physics_step"]),
        control_step=int(payload["control_step"]),
        policy_step=int(payload["policy_step"]),
        time_s=float(payload["time_s"]),
        rgb=copy.deepcopy(dict(payload["rgb"])),
        depth_m=copy.deepcopy(dict(payload["depth_m"])),
        segmentation=copy.deepcopy(dict(payload["segmentation"])),
        proprio=copy.deepcopy(dict(payload["proprio"])),
        privileged=copy.deepcopy(dict(payload["privileged"])),
        events_since_last_observation=events,
        api_version=str(payload["api_version"]),
    )


def _assert_restore_observation_identity(observed: Any, expected: Any) -> None:
    for name in (
        "episode_id",
        "task_id",
        "physics_step",
        "control_step",
        "policy_step",
    ):
        if getattr(observed, name) != getattr(expected, name):
            raise RuntimeError(
                f"Dynamic Benchmark restored observation {name} does not match checkpoint"
            )


class _DynamicBenchmarkProcessHandler:
    """Own a fixed shard of canonical MuJoCo environments in one subprocess."""

    def __init__(self, payload: Mapping[str, Any], indices: tuple[int, ...]) -> None:
        from se3_wam.benchmark.api import ActionCommand, ActionMode, ObservationBundle
        from se3_wam.benchmark.config import load_task_config
        from se3_wam.benchmark.contracts import EventRecord
        from se3_wam.benchmark.keyed_puck import T5EventTape
        from se3_wam.benchmark.suite import make_mujoco_env

        self._ActionCommand = ActionCommand
        self._ActionMode = ActionMode
        self._ObservationBundle = ObservationBundle
        self._EventRecord = EventRecord
        self._load_task_config = load_task_config
        self._T5EventTape = T5EventTape
        self._task_id = str(payload["task_id"])
        self._split_name = str(payload["split_name"])
        self._process_residual_planner = bool(
            payload.get("process_residual_planner", False)
        )
        self._make_privileged_teacher = None
        if self._process_residual_planner:
            from se3_wam.benchmark.teacher_factory import make_privileged_teacher

            self._make_privileged_teacher = make_privileged_teacher
        self._indices = indices
        self._envs = {
            index: make_mujoco_env(
                self._task_id,
                image_size=int(payload["image_size"]),
                camera_observations=bool(payload["camera_observations"]),
            )
            for index in indices
        }
        self._observations: dict[int, Any] = {}
        self._teachers: dict[int, Any] = {}

    def ready_metadata(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "indices": self._indices,
            "horizons": {
                index: int(self._envs[index].horizon_steps) for index in self._indices
            },
        }

    def _arm_hidden_t5_event(self, env: Any, request: Any) -> None:
        arm_hidden_t5_event(
            task_id=self._task_id,
            split_name=str(request.split.value),
            env=env,
            request=request,
            load_task_config=self._load_task_config,
            event_tape_type=self._T5EventTape,
        )

    def _reset_residual_planner(self, index: int, request: Any) -> None:
        if not self._process_residual_planner:
            return
        if self._make_privileged_teacher is None:
            raise RuntimeError("process residual planner factory is unavailable")
        teacher, _ = self._make_privileged_teacher(self._task_id, request=request)
        if hasattr(teacher, "reset"):
            teacher.reset()
        self._teachers[index] = teacher

    @staticmethod
    def _compose_residual_action(
        planner_values: Any,
        residual_values: Any,
        residual_scale: float,
    ) -> np.ndarray:
        planner = torch.as_tensor(planner_values, dtype=torch.float32, device="cpu")
        residual = torch.as_tensor(residual_values, dtype=torch.float32, device="cpu")
        if planner.shape != (7,) or residual.shape != (7,):
            raise ValueError("process planner and residual actions must have shape (7,)")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("process residual_scale must be in (0, 1]")
        return (
            torch.clamp(planner + float(residual_scale) * residual, -1.0, 1.0)
            .numpy()
            .copy()
        )

    @staticmethod
    def _pack_step_result(
        env: Any,
        step_result: Any,
        *,
        action_values: Any | None = None,
        planner_action_s: float = 0.0,
        environment_step_s: float = 0.0,
    ) -> dict[str, Any]:
        payload = {
            "observation": _pack_process_observation(step_result.observation),
            "terminated": step_result.terminated,
            "truncated": step_result.truncated,
            "success": step_result.success,
            "termination_reason": step_result.termination_reason,
            "active_stage_progress": step_result.active_stage_progress,
            "event_names": tuple(event.name for event in env._ledger.events),
            "planner_action_s": float(planner_action_s),
            "environment_step_s": float(environment_step_s),
        }
        if action_values is not None:
            payload["action_values"] = np.asarray(action_values, dtype=np.float32)
        return payload

    def handle(
        self,
        command: str,
        items: list[tuple[int, Any]],
    ) -> list[tuple[int, Any]]:
        results = []
        for index, payload in items:
            env = self._envs[index]
            if command == "reset":
                request = payload
                observation = env.reset(request)
                self._arm_hidden_t5_event(env, request)
                self._observations[index] = observation
                self._reset_residual_planner(index, request)
                result = _pack_process_observation(observation)
            elif command == "step":
                action = self._ActionCommand(
                    mode=self._ActionMode(str(payload["action_mode"])),
                    values=payload["values"],
                    policy_step=int(payload["policy_step"]),
                )
                environment_step_start = time.perf_counter()
                step_result = env.step(action)
                environment_step_s = time.perf_counter() - environment_step_start
                self._observations[index] = step_result.observation
                result = self._pack_step_result(
                    env,
                    step_result,
                    environment_step_s=environment_step_s,
                )
            elif command == "step_residual_planner":
                if not self._process_residual_planner:
                    raise RuntimeError("process residual planner is not configured")
                observation = self._observations.get(index)
                teacher = self._teachers.get(index)
                if observation is None or teacher is None:
                    raise RuntimeError("process residual planner is not initialized")
                planner_start = time.perf_counter()
                planner_values = teacher.act(observation).values
                action_values = self._compose_residual_action(
                    planner_values,
                    payload["values"],
                    float(payload["residual_scale"]),
                )
                planner_action_s = time.perf_counter() - planner_start
                action = self._ActionCommand(
                    mode=self._ActionMode(str(payload["action_mode"])),
                    values=action_values,
                    policy_step=int(payload["policy_step"]),
                )
                environment_step_start = time.perf_counter()
                step_result = env.step(action)
                environment_step_s = time.perf_counter() - environment_step_start
                self._observations[index] = step_result.observation
                result = self._pack_step_result(
                    env,
                    step_result,
                    action_values=action_values,
                    planner_action_s=planner_action_s,
                    environment_step_s=environment_step_s,
                )
            elif command == "save":
                observation = self._observations.get(index)
                if observation is None:
                    raise RuntimeError("process checkpoint requires an initialized observation")
                result = {
                    "env_state": env.save_state(),
                    "observation": _pack_process_observation(observation),
                }
            elif command == "restore":
                request, env_state, observation_payload = payload
                env.reset(request)
                self._arm_hidden_t5_event(env, request)
                observed = env.load_state(env_state)
                observation = _unpack_observation_payload(
                    observation_payload,
                    observation_type=self._ObservationBundle,
                    event_type=self._EventRecord,
                )
                _assert_restore_observation_identity(observed, observation)
                self._observations[index] = observation
                self._reset_residual_planner(index, request)
                result = _pack_process_observation(observation)
            else:
                raise ValueError(
                    f"unsupported Dynamic Benchmark process command {command!r}"
                )
            results.append((index, result))
        return results

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()
        self._observations.clear()
        self._teachers.clear()


def _make_dynamic_benchmark_process_handler(
    payload: Mapping[str, Any],
    indices: tuple[int, ...],
) -> _DynamicBenchmarkProcessHandler:
    return _DynamicBenchmarkProcessHandler(payload, indices)


class DynamicBenchmarkEnv(gym.Env):
    """Vectorized, privileged-state Dynamic Benchmark environment.

    The underlying benchmark remains the source of reset identity, dynamics,
    success, termination, and exact-replay semantics.  This adapter only adds a
    frozen state vector and a current-state reward for RLinf.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ) -> None:
        super().__init__()
        if num_envs < 1:
            raise ValueError("DynamicBenchmarkEnv requires at least one environment")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.record_metrics = bool(record_metrics)
        self.task_id = str(_cfg_get(cfg, "task_id", ""))
        if not self.task_id:
            raise ValueError("DynamicBenchmarkEnv requires cfg.task_id")
        self.split_name = str(_cfg_get(cfg, "split", "train"))
        self.base_manifest_seed = int(_cfg_get(cfg, "manifest_seed", 20261050))
        self.base_manifest_seed += self.seed_offset * 1_000_003
        self.manifest_size = int(_cfg_get(cfg, "manifest_size", 4096))
        if self.manifest_size < self.num_envs:
            raise ValueError("manifest_size must be at least num_envs")
        if self.manifest_size % 2:
            self.manifest_size += 1
        self.image_size = int(_cfg_get(cfg, "image_size", 64))
        if self.image_size < 64:
            raise ValueError("Dynamic Benchmark image_size must be at least 64")
        self.camera_observations = bool(_cfg_get(cfg, "camera_observations", False))
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.worker_threads = int(_cfg_get(cfg, "worker_threads", 1))
        if self.worker_threads < 1:
            raise ValueError("Dynamic Benchmark worker_threads must be positive")
        self.worker_processes = int(_cfg_get(cfg, "worker_processes", 0))
        if self.worker_processes < 0:
            raise ValueError("Dynamic Benchmark worker_processes must be non-negative")
        self.process_start_method = str(_cfg_get(cfg, "process_start_method", "spawn"))
        self.process_timeout_s = float(_cfg_get(cfg, "process_timeout_s", 120.0))
        if self.process_timeout_s <= 0.0:
            raise ValueError("Dynamic Benchmark process_timeout_s must be positive")
        if self.worker_processes and self.worker_threads != 1:
            raise ValueError(
                "Dynamic Benchmark process workers require worker_threads=1 per process"
            )
        self.process_residual_planner = bool(
            _cfg_get(cfg, "process_residual_planner", False)
        )
        if self.process_residual_planner and not self.worker_processes:
            raise ValueError("process residual planner requires process workers")
        self.use_rel_reward = False
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        self.num_group = max(1, self.num_envs // max(1, self.group_size))
        configured_prompt = _cfg_get(cfg, "task_prompt", None)
        self.task_prompt = (
            f"Solve Dynamic Benchmark task {self.task_id}."
            if configured_prompt is None
            else str(configured_prompt)
        )

        self._load_benchmark_contracts()
        if self.task_id not in self._task_ids:
            raise ValueError(
                f"task {self.task_id!r} is not RL-training eligible; available={self._task_ids}"
            )
        self._manifest_generation = 0
        self._manifest_cursor = 0
        self._manifest_rows: tuple[Any, ...] = ()
        self._refresh_manifest()
        self._process_vector = None
        if self.worker_processes:
            from .process_vector import OrderedProcessVector

            self.envs = []
            self._executor = None
            self._process_vector = OrderedProcessVector(
                num_envs=self.num_envs,
                num_workers=self.worker_processes,
                handler_factory=_make_dynamic_benchmark_process_handler,
                handler_payload={
                    "task_id": self.task_id,
                    "split_name": self.split_name,
                    "image_size": self.image_size,
                    "camera_observations": self.camera_observations,
                    "process_residual_planner": self.process_residual_planner,
                },
                start_method=self.process_start_method,
                timeout_s=self.process_timeout_s,
            )
            horizons = {
                int(index): int(horizon)
                for metadata in self._process_vector.ready_metadata
                for index, horizon in metadata["horizons"].items()
            }
            if set(horizons) != set(range(self.num_envs)):
                raise RuntimeError(
                    "Dynamic Benchmark process workers lost environment indices"
                )
            self.horizon_steps = horizons[0]
            if any(horizon != self.horizon_steps for horizon in horizons.values()):
                raise RuntimeError(
                    "Dynamic Benchmark vector members disagree on horizon"
                )
        else:
            self.envs = [
                self._make_mujoco_env(
                    self.task_id,
                    image_size=self.image_size,
                    camera_observations=self.camera_observations,
                )
                for _ in range(self.num_envs)
            ]
            self._executor = (
                ThreadPoolExecutor(
                    max_workers=min(self.worker_threads, self.num_envs),
                    thread_name_prefix="dynamic-benchmark",
                )
                if self.worker_threads > 1 and self.num_envs > 1
                else None
            )
            self.horizon_steps = int(self.envs[0].horizon_steps)
            if any(int(env.horizon_steps) != self.horizon_steps for env in self.envs):
                raise RuntimeError(
                    "Dynamic Benchmark vector members disagree on horizon"
                )
        self._raw_observations: list[Any | None] = [None] * self.num_envs
        self._requests: list[Any | None] = [None] * self.num_envs
        self._state_schema: DynamicBenchmarkStateSchema | None = None
        self._last_obs: dict[str, torch.Tensor] | None = None
        self._needs_reset = np.ones(self.num_envs, dtype=bool)
        self._is_start = True
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reward_trackers = [self._make_reward() for _ in range(self.num_envs)]
        self.info_logging_keys = ["success", "trajectory_completion"]
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.reset()
        assert self._state_schema is not None
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.num_envs, self._state_schema.state_dim),
                    dtype=np.float32,
                )
            }
        )

    def _load_benchmark_contracts(self) -> None:
        try:
            from se3_wam.benchmark.api import (
                ActionCommand,
                ObservationBundle,
                Split,
                StepResult,
            )
            from se3_wam.benchmark.config import load_task_config
            from se3_wam.benchmark.contracts import EventRecord
            from se3_wam.benchmark.dataset_manifest import (
                make_dataset_candidate_manifest,
            )
            from se3_wam.benchmark.keyed_puck import T5EventTape
            from se3_wam.benchmark.p0_grasp_manifest import (
                make_p0_grasp_candidate_manifest,
            )
            from se3_wam.benchmark.registry import (
                ACTIVE_TASK_IDS,
                RL_EXPERT_TASK_IDS,
                get_task_spec,
            )
            from se3_wam.benchmark.suite import make_mujoco_env
        except ImportError as exc:
            raise ImportError(
                "DynamicBenchmarkEnv requires the SE3-WAM benchmark source on PYTHONPATH"
            ) from exc
        self._ActionCommand = ActionCommand
        self._ObservationBundle = ObservationBundle
        self._Split = Split
        self._StepResult = StepResult
        self._EventRecord = EventRecord
        self._make_dataset_candidate_manifest = make_dataset_candidate_manifest
        self._make_p0_grasp_candidate_manifest = make_p0_grasp_candidate_manifest
        self._load_task_config = load_task_config
        self._T5EventTape = T5EventTape
        self._active_task_ids = tuple(ACTIVE_TASK_IDS)
        self._task_ids = tuple(RL_EXPERT_TASK_IDS)
        self._get_task_spec = get_task_spec
        self._make_mujoco_env = make_mujoco_env
        try:
            self._split = Split(self.split_name)
        except ValueError as exc:
            raise ValueError(
                f"unsupported Dynamic Benchmark split {self.split_name!r}"
            ) from exc

    def _refresh_manifest(self) -> None:
        manifest_seed = self.base_manifest_seed + self._manifest_generation * 10_000_019
        if self.task_id == "p0_grasp":
            rows = self._make_p0_grasp_candidate_manifest(
                split=self._split,
                attempts=self.manifest_size,
                manifest_seed=manifest_seed,
            )
        else:
            all_rows = self._make_dataset_candidate_manifest(
                split=self._split,
                attempts_per_task=self.manifest_size,
                manifest_seed=manifest_seed,
                tasks=self._active_task_ids,
            )
            rows = tuple(row for row in all_rows if row.request.task_id == self.task_id)
        if len(rows) != self.manifest_size:
            raise RuntimeError(
                f"manifest produced {len(rows)} rows for {self.task_id}, expected {self.manifest_size}"
            )
        self._manifest_rows = tuple(rows)
        self._manifest_cursor = 0

    def set_manifest_context(self, *, split_name: str, base_manifest_seed: int) -> None:
        """Switch manifest identity without rebuilding the underlying simulators.

        The caller must either reset every vector member before stepping or load a
        checkpoint whose identity matches the new context. Keeping this operation
        separate from reset makes validation borrowing exactly reversible.
        """

        try:
            split = self._Split(str(split_name))
        except ValueError as exc:
            raise ValueError(
                f"unsupported Dynamic Benchmark split {split_name!r}"
            ) from exc
        self.split_name = str(split_name)
        self._split = split
        self.base_manifest_seed = int(base_manifest_seed)
        self._manifest_generation = 0
        self._refresh_manifest()

    def manifest_cache_state(self) -> dict[str, Any]:
        """Return an in-memory cache for the current frozen manifest generation."""

        return {
            "schema_version": "rlinf-dynamic-benchmark-manifest-cache-v0.1",
            "task_id": self.task_id,
            "split_name": self.split_name,
            "base_manifest_seed": self.base_manifest_seed,
            "manifest_size": self.manifest_size,
            "manifest_generation": self._manifest_generation,
            "manifest_rows": self._manifest_rows,
        }

    def load_manifest_cache_state(self, state: Mapping[str, Any]) -> None:
        """Install a validated in-memory manifest cache without regenerating it."""

        if state.get("schema_version") != "rlinf-dynamic-benchmark-manifest-cache-v0.1":
            raise ValueError("unsupported Dynamic Benchmark manifest cache schema")
        if str(state.get("task_id")) != self.task_id:
            raise ValueError("Dynamic Benchmark manifest cache task does not match env")
        if int(state.get("manifest_size", -1)) != self.manifest_size:
            raise ValueError("Dynamic Benchmark manifest cache size does not match env")
        split_name = str(state["split_name"])
        try:
            split = self._Split(split_name)
        except ValueError as exc:
            raise ValueError(
                f"unsupported Dynamic Benchmark split {split_name!r}"
            ) from exc
        generation = int(state["manifest_generation"])
        if generation < 0:
            raise ValueError("Dynamic Benchmark manifest cache generation must be non-negative")
        rows = tuple(state["manifest_rows"])
        if len(rows) != self.manifest_size:
            raise ValueError("Dynamic Benchmark manifest cache row count does not match env")
        if any(
            row.request.task_id != self.task_id
            or str(row.request.split.value) != split_name
            for row in rows
        ):
            raise ValueError("Dynamic Benchmark manifest cache row identity does not match env")
        self.split_name = split_name
        self._split = split
        self.base_manifest_seed = int(state["base_manifest_seed"])
        self._manifest_generation = generation
        self._manifest_rows = rows
        self._manifest_cursor = 0

    def _next_request(self) -> Any:
        if self._manifest_cursor == len(self._manifest_rows):
            self._manifest_generation += 1
            self._refresh_manifest()
        request = self._manifest_rows[self._manifest_cursor].request
        self._manifest_cursor += 1
        return request

    def _make_reward(self) -> DynamicBenchmarkReward:
        task = self._get_task_spec(self.task_id)
        return DynamicBenchmarkReward(
            success_stages=task.success_stages,
            progress_scale=float(_cfg_get(self.cfg, "reward_progress_scale", 5.0)),
            success_bonus=float(_cfg_get(self.cfg, "reward_success_bonus", 10.0)),
            failure_penalty=float(_cfg_get(self.cfg, "reward_failure_penalty", -3.0)),
            safety_penalty=float(_cfg_get(self.cfg, "reward_safety_penalty", -10.0)),
            timeout_penalty=float(_cfg_get(self.cfg, "reward_timeout_penalty", -1.0)),
            step_penalty=float(_cfg_get(self.cfg, "reward_step_penalty", -0.01)),
            action_l2_scale=float(_cfg_get(self.cfg, "reward_action_l2_scale", -0.001)),
        )

    @property
    def total_num_group_envs(self) -> int:
        return np.iinfo(np.uint32).max // 2

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def instruction(self) -> list[str]:
        return [self.task_prompt] * self.num_envs

    @property
    def state_schema(self) -> dict[str, Any]:
        if self._state_schema is None:
            raise RuntimeError("state schema is unavailable before the first reset")
        return self._state_schema.to_dict()

    @property
    def reward_schema(self) -> Mapping[str, Any]:
        return self.reward_trackers[0].to_dict()

    @property
    def process_worker_pids(self) -> tuple[int, ...]:
        """Return stable child PIDs, or an empty tuple for the in-process backend."""

        if self._process_vector is None:
            return ()
        return self._process_vector.worker_pids

    def _encode(self, observation: Any, request: Any) -> np.ndarray:
        if self._state_schema is None:
            self._state_schema = DynamicBenchmarkStateSchema.from_observation(
                task_id=self.task_id,
                task_ids=self._task_ids,
                observation=observation,
                factors=request.factors,
            )
        return self._state_schema.encode(
            observation=observation,
            factors=request.factors,
            horizon_steps=self.horizon_steps,
        )

    def _unpack_process_observation(self, payload: Mapping[str, Any]) -> Any:
        return _unpack_observation_payload(
            payload,
            observation_type=self._ObservationBundle,
            event_type=self._EventRecord,
        )

    def _reset_metrics(self, indices: np.ndarray) -> None:
        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        self._elapsed_steps[tensor_indices] = 0
        self.prev_step_reward[tensor_indices] = 0.0
        self.returns[tensor_indices] = 0.0
        self.success_once[tensor_indices] = False
        for index in indices:
            self.reward_trackers[int(index)].reset()

    def _arm_hidden_t5_event(self, env: Any, request: Any) -> str | None:
        return arm_hidden_t5_event(
            task_id=self.task_id,
            split_name=str(request.split.value),
            env=env,
            request=request,
            load_task_config=self._load_task_config,
            event_tape_type=self._T5EventTape,
        )

    def _reset_one(self, item: tuple[int, Any]) -> tuple[int, Any]:
        index, request = item
        observation = self.envs[index].reset(request)
        self._arm_hidden_t5_event(self.envs[index], request)
        return index, observation

    def _step_one(
        self, item: tuple[int, np.ndarray]
    ) -> tuple[int, Any, Any, tuple[str, ...]]:
        index, values = item
        env = self.envs[index]
        observation = self._raw_observations[index]
        request = self._requests[index]
        if observation is None or request is None:
            raise RuntimeError("Dynamic Benchmark vector member is not initialized")
        action = self._ActionCommand(
            mode=request.action_mode,
            values=values,
            policy_step=observation.policy_step,
        )
        result = env.step(action)
        event_names = tuple(event.name for event in env._ledger.events)
        return index, action, result, event_names

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        del seed
        options = {} if options is None else dict(options)
        env_idx_value = options.pop("env_idx", None)
        if options:
            raise ValueError(
                f"unsupported Dynamic Benchmark reset options: {sorted(options)}"
            )
        indices = (
            np.arange(self.num_envs, dtype=np.int64)
            if env_idx_value is None
            else np.asarray(env_idx_value, dtype=np.int64).reshape(-1)
        )
        if np.any(indices < 0) or np.any(indices >= self.num_envs):
            raise IndexError("reset env_idx is out of range")
        self._reset_metrics(indices)
        states = (
            np.zeros((self.num_envs, 0), dtype=np.float32)
            if self._last_obs is None
            else self._last_obs["states"].cpu().numpy().copy()
        )
        reset_items = [(int(index), self._next_request()) for index in indices]
        requests_by_index = dict(reset_items)
        if self._process_vector is not None:
            reset_results = [
                (index, self._unpack_process_observation(payload))
                for index, payload in self._process_vector.run("reset", reset_items)
            ]
        elif self._executor is None or len(reset_items) < 2:
            reset_results = [self._reset_one(item) for item in reset_items]
        else:
            reset_results = list(self._executor.map(self._reset_one, reset_items))
        for index, observation in reset_results:
            request = requests_by_index[index]
            state = self._encode(observation, request)
            if states.shape[1] == 0:
                states = np.zeros((self.num_envs, state.size), dtype=np.float32)
            states[index] = state
            self._raw_observations[index] = observation
            self._requests[index] = request
            self._needs_reset[index] = False
        obs = {"states": torch.as_tensor(states, dtype=torch.float32)}
        self._last_obs = obs
        self._is_start = True
        return obs, {}

    def _normalize_actions(
        self, actions: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        array = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        )
        if array.ndim == 1:
            array = np.repeat(array[None, :], self.num_envs, axis=0)
        if array.shape != (self.num_envs, 7):
            raise ValueError(
                f"Dynamic Benchmark actions must have shape {(self.num_envs, 7)}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Dynamic Benchmark actions contain NaN or Inf")
        return np.clip(array.astype(np.float64, copy=False), -1.0, 1.0)

    def step(
        self,
        actions: Union[np.ndarray, torch.Tensor],
        auto_reset: bool = True,
        *,
        process_residual_planner_scale: float | None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        action_array = self._normalize_actions(actions)
        if process_residual_planner_scale is not None:
            if self._process_vector is None or not self.process_residual_planner:
                raise ValueError(
                    "process_residual_planner_scale requires configured process workers"
                )
            if not 0.0 < process_residual_planner_scale <= 1.0:
                raise ValueError("process residual planner scale must be in (0, 1]")
        applied_action_array = action_array.copy()
        if self._last_obs is None:
            self.reset()
        assert self._last_obs is not None
        states = self._last_obs["states"].cpu().numpy().copy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        successes = np.zeros(self.num_envs, dtype=bool)
        completions = np.zeros(self.num_envs, dtype=np.float32)
        termination_reasons: list[str | None] = [None] * self.num_envs
        component_rows: list[dict[str, float]] = [{} for _ in range(self.num_envs)]
        event_name_rows: list[list[str]] = [[] for _ in range(self.num_envs)]
        active_stage_progresses = np.zeros(self.num_envs, dtype=np.float64)
        process_planner_action_s = np.zeros(self.num_envs, dtype=np.float64)
        process_environment_step_s = np.zeros(self.num_envs, dtype=np.float64)
        stepped = np.zeros(self.num_envs, dtype=bool)
        active_items: list[tuple[int, np.ndarray]] = []
        for index in range(self.num_envs):
            if self._needs_reset[index]:
                terminations[index] = True
                component_rows[index] = {"total": 0.0}
                continue
            active_items.append((index, action_array[index]))
        if self._process_vector is not None:
            process_items = []
            for index, values in active_items:
                observation = self._raw_observations[index]
                request = self._requests[index]
                if observation is None or request is None:
                    raise RuntimeError(
                        "Dynamic Benchmark vector member is not initialized"
                    )
                process_items.append(
                    (
                        index,
                        {
                            "values": values,
                            "action_mode": request.action_mode.value,
                            "policy_step": observation.policy_step,
                        },
                    )
                )
            process_command = (
                "step_residual_planner"
                if process_residual_planner_scale is not None
                else "step"
            )
            if process_residual_planner_scale is not None:
                for _, process_payload in process_items:
                    process_payload["residual_scale"] = float(
                        process_residual_planner_scale
                    )
            process_results = self._process_vector.run(process_command, process_items)
            step_results = []
            for index, payload in process_results:
                observation = self._raw_observations[index]
                request = self._requests[index]
                assert observation is not None and request is not None
                applied_values = payload.get("action_values", action_array[index])
                applied_action_array[index] = np.asarray(applied_values, dtype=np.float64)
                process_planner_action_s[index] = float(
                    payload.get("planner_action_s", 0.0)
                )
                process_environment_step_s[index] = float(
                    payload.get("environment_step_s", 0.0)
                )
                action = self._ActionCommand(
                    mode=request.action_mode,
                    values=applied_values,
                    policy_step=observation.policy_step,
                )
                result = self._StepResult(
                    observation=self._unpack_process_observation(
                        payload["observation"]
                    ),
                    terminated=bool(payload["terminated"]),
                    truncated=bool(payload["truncated"]),
                    success=bool(payload["success"]),
                    termination_reason=payload["termination_reason"],
                    active_stage_progress=float(payload["active_stage_progress"]),
                )
                step_results.append(
                    (index, action, result, payload["event_names"])
                )
        elif self._executor is None or len(active_items) < 2:
            step_results = [self._step_one(item) for item in active_items]
        else:
            step_results = list(self._executor.map(self._step_one, active_items))
        for index, action, result, event_names in step_results:
            request = self._requests[index]
            assert request is not None
            event_name_rows[index] = list(event_names)
            active_stage_progresses[index] = float(result.active_stage_progress)
            reward, components = self.reward_trackers[index].step(
                action=action.values,
                event_names=event_names,
                active_stage_progress=result.active_stage_progress,
                success=bool(result.success),
                terminated=bool(result.terminated),
                truncated=bool(result.truncated),
                termination_reason=result.termination_reason,
            )
            states[index] = self._encode(result.observation, request)
            self._raw_observations[index] = result.observation
            rewards[index] = reward
            terminations[index] = bool(result.terminated)
            truncations[index] = bool(result.truncated)
            successes[index] = bool(result.success)
            completions[index] = self.reward_trackers[index].potential(
                event_names=event_names,
                active_stage_progress=result.active_stage_progress,
            )
            termination_reasons[index] = result.termination_reason
            component_rows[index] = components
            stepped[index] = True
            if result.terminated or result.truncated:
                self._needs_reset[index] = True

        self._elapsed_steps[torch.as_tensor(stepped)] += 1
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        termination_tensor = torch.as_tensor(terminations, dtype=torch.bool)
        truncation_tensor = torch.as_tensor(truncations, dtype=torch.bool)
        success_tensor = torch.as_tensor(successes, dtype=torch.bool)
        completion_tensor = torch.as_tensor(completions, dtype=torch.float32)
        self.prev_step_reward = reward_tensor.clone()
        if self.record_metrics:
            self.returns += reward_tensor
            self.success_once |= success_tensor
        infos: dict[str, Any] = {
            "success": success_tensor,
            "trajectory_completion": completion_tensor,
            "termination_reason": termination_reasons,
            "reward_components": {
                name: torch.as_tensor(
                    [row.get(name, 0.0) for row in component_rows], dtype=torch.float32
                )
                for name in sorted({name for row in component_rows for name in row})
            },
            "reward_inputs": {
                "stepped": torch.as_tensor(stepped, dtype=torch.bool),
                "action": torch.as_tensor(applied_action_array, dtype=torch.float64),
                "event_names": event_name_rows,
                "active_stage_progress": torch.as_tensor(
                    active_stage_progresses, dtype=torch.float64
                ),
                "success": success_tensor.clone(),
                "terminated": termination_tensor.clone(),
                "truncated": truncation_tensor.clone(),
                "termination_reason": list(termination_reasons),
            },
            "process_timings": {
                "planner_action_s": torch.as_tensor(
                    process_planner_action_s, dtype=torch.float64
                ),
                "environment_step_s": torch.as_tensor(
                    process_environment_step_s, dtype=torch.float64
                ),
            },
            "episode": {
                "return": self.returns.clone(),
                "episode_len": self._elapsed_steps.float().clone(),
                "success_once": self.success_once.clone(),
            },
        }
        if self.ignore_terminations:
            infos["episode"]["terminated_at_end"] = termination_tensor.clone()
            termination_tensor = torch.zeros_like(termination_tensor)
        obs = {"states": torch.as_tensor(states, dtype=torch.float32)}
        dones = torch.logical_or(termination_tensor, truncation_tensor)
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        self._last_obs = obs
        self._is_start = False
        return obs, reward_tensor, termination_tensor, truncation_tensor, infos

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        obs: dict[str, torch.Tensor],
        infos: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        final_obs = _torch_clone(obs)
        final_info = _torch_clone(infos)
        indices = torch.arange(self.num_envs)[dones].cpu().numpy()
        reset_obs, reset_infos = self.reset(options={"env_idx": indices})
        reset_infos["final_observation"] = final_obs
        reset_infos["final_info"] = final_info
        reset_infos["_final_info"] = dones
        reset_infos["_final_observation"] = dones
        reset_infos["_elapsed_steps"] = dones
        return reset_obs, reset_infos

    def chunk_step(
        self, chunk_actions: Union[np.ndarray, torch.Tensor]
    ) -> tuple[
        list[dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        action_chunks = (
            chunk_actions
            if isinstance(chunk_actions, torch.Tensor)
            else torch.as_tensor(chunk_actions)
        )
        if action_chunks.ndim != 3 or tuple(action_chunks.shape[::2]) != (
            self.num_envs,
            7,
        ):
            raise ValueError(
                "chunk_actions must have shape "
                f"[num_envs, chunk_steps, 7], got {tuple(action_chunks.shape)}"
            )
        obs_list = []
        infos_list = []
        rewards = []
        terminations = []
        truncations = []
        for chunk_index in range(action_chunks.shape[1]):
            obs, reward, terminated, truncated, infos = self.step(
                action_chunks[:, chunk_index], auto_reset=False
            )
            obs_list.append(obs)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos_list.append(infos)
        reward_tensor = torch.stack(rewards, dim=1)
        raw_terminations = torch.stack(terminations, dim=1)
        raw_truncations = torch.stack(truncations, dim=1)
        past_terminations = raw_terminations.any(dim=1)
        past_truncations = raw_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )
        chunk_terminations = torch.zeros_like(raw_terminations)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            reward_tensor,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def sample_action_space(self) -> torch.Tensor:
        return torch.as_tensor(self.action_space.sample(), dtype=torch.float32)

    def checkpoint_state(self) -> dict[str, Any]:
        """Capture vector env, manifest, reward, and metric state for exact resume."""

        if (
            self._last_obs is None
            or any(item is None for item in self._requests)
            or any(item is None for item in self._raw_observations)
        ):
            raise RuntimeError("Dynamic Benchmark checkpoint requires initialized envs")
        if self._process_vector is None:
            snapshots = []
            for env, observation in zip(
                self.envs, self._raw_observations, strict=True
            ):
                assert observation is not None
                snapshots.append(
                    {
                        "env_state": env.save_state(),
                        "observation": _pack_process_observation(observation),
                    }
                )
        else:
            snapshots = [
                value
                for _, value in self._process_vector.run(
                    "save", ((index, None) for index in range(self.num_envs))
                )
            ]
        env_states = [snapshot["env_state"] for snapshot in snapshots]
        raw_observations = [snapshot["observation"] for snapshot in snapshots]
        raw_observation_sha256 = [
            _observation_payload_sha256(observation)
            for observation in raw_observations
        ]
        last_obs = _torch_clone(self._last_obs)
        last_obs_sha256 = _state_tensor_sha256(last_obs["states"])
        request_rows = []
        for request in self._requests:
            assert request is not None
            request_rows.append(
                {
                    "episode_id": request.episode_id,
                    "task_id": request.task_id,
                    "split": request.split.value,
                    "seed": request.seed,
                    "action_mode": request.action_mode.value,
                    "observation_track": request.observation_track.value,
                    "object_mode": request.object_mode,
                    "reset_mode": request.reset_mode,
                    "factors": dict(request.factors),
                    "api_version": request.api_version,
                }
            )
        identity = self._checkpoint_identity()
        identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": "rlinf-dynamic-benchmark-checkpoint-v0.2",
            "identity": identity,
            "identity_sha256": identity_sha256,
            "manifest_generation": self._manifest_generation,
            "manifest_cursor": self._manifest_cursor,
            "requests": request_rows,
            "env_states": env_states,
            "raw_observations": raw_observations,
            "raw_observation_sha256": raw_observation_sha256,
            "last_obs": last_obs,
            "last_obs_sha256": last_obs_sha256,
            "reward_states": [tracker.state_dict() for tracker in self.reward_trackers],
            "needs_reset": self._needs_reset.copy(),
            "elapsed_steps": self._elapsed_steps.clone(),
            "prev_step_reward": self.prev_step_reward.clone(),
            "returns": self.returns.clone(),
            "success_once": self.success_once.clone(),
            "is_start": self._is_start,
        }

    def _checkpoint_identity(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split_name,
            "base_manifest_seed": self.base_manifest_seed,
            "manifest_size": self.manifest_size,
            "image_size": self.image_size,
            "camera_observations": self.camera_observations,
            "num_envs": self.num_envs,
            "worker_threads": self.worker_threads,
            "worker_processes": self.worker_processes,
            "process_start_method": self.process_start_method,
            "state_schema": self.state_schema,
        }

    def load_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        refresh_manifest: bool = True,
    ) -> None:
        """Restore a checkpoint produced by :meth:`checkpoint_state`."""

        if state.get("schema_version") != "rlinf-dynamic-benchmark-checkpoint-v0.2":
            raise ValueError("unsupported Dynamic Benchmark checkpoint schema")
        identity = dict(state["identity"])
        expected = self._checkpoint_identity()
        if identity != expected:
            raise ValueError("Dynamic Benchmark checkpoint identity does not match env")
        observed_identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if observed_identity_sha256 != state["identity_sha256"]:
            raise ValueError("Dynamic Benchmark checkpoint identity hash mismatch")
        manifest_generation = int(state["manifest_generation"])
        if not refresh_manifest and (
            self._manifest_generation != manifest_generation
            or len(self._manifest_rows) != self.manifest_size
        ):
            raise ValueError(
                "Dynamic Benchmark cached manifest generation does not match checkpoint"
            )
        request_rows = list(state["requests"])
        env_states = list(state["env_states"])
        raw_observation_payloads = list(state["raw_observations"])
        raw_observation_sha256 = list(state["raw_observation_sha256"])
        reward_states = list(state["reward_states"])
        if not (
            len(request_rows)
            == len(env_states)
            == len(raw_observation_payloads)
            == len(raw_observation_sha256)
            == len(reward_states)
            == self.num_envs
        ):
            raise ValueError("Dynamic Benchmark checkpoint vector length mismatch")
        for payload, expected_sha256 in zip(
            raw_observation_payloads, raw_observation_sha256, strict=True
        ):
            if _observation_payload_sha256(payload) != expected_sha256:
                raise ValueError("Dynamic Benchmark checkpoint observation hash mismatch")
        from se3_wam.benchmark.api import (
            ActionMode,
            ObservationTrack,
            ResetRequest,
            Split,
        )

        raw_observations = []
        requests = []
        restore_items = []
        authoritative_observations = [
            self._unpack_process_observation(payload)
            for payload in raw_observation_payloads
        ]
        for index, (row, env_state, authoritative_observation) in enumerate(
            zip(
                request_rows,
                env_states,
                authoritative_observations,
                strict=True,
            )
        ):
            request = ResetRequest(
                episode_id=str(row["episode_id"]),
                task_id=str(row["task_id"]),
                split=Split(str(row["split"])),
                seed=int(row["seed"]),
                action_mode=ActionMode(str(row["action_mode"])),
                observation_track=ObservationTrack(str(row["observation_track"])),
                object_mode=str(row["object_mode"]),
                reset_mode=str(row["reset_mode"]),
                factors=dict(row["factors"]),
                api_version=str(row["api_version"]),
            )
            requests.append(request)
            if (
                authoritative_observation.episode_id != request.episode_id
                or authoritative_observation.task_id != request.task_id
            ):
                raise ValueError(
                    "Dynamic Benchmark checkpoint observation identity does not match request"
                )
            if self._process_vector is None:
                env = self.envs[index]
                env.reset(request)
                observed = env.load_state(env_state)
                _assert_restore_observation_identity(
                    observed, authoritative_observation
                )
                raw_observations.append(authoritative_observation)
            else:
                restore_items.append(
                    (
                        index,
                        (request, env_state, raw_observation_payloads[index]),
                    )
                )
        if self._process_vector is not None:
            raw_observations = [
                self._unpack_process_observation(payload)
                for _, payload in self._process_vector.run("restore", restore_items)
            ]
        self._manifest_generation = manifest_generation
        if refresh_manifest:
            self._refresh_manifest()
        manifest_cursor = int(state["manifest_cursor"])
        if not 0 <= manifest_cursor <= len(self._manifest_rows):
            raise ValueError("Dynamic Benchmark checkpoint manifest cursor is invalid")
        self._manifest_cursor = manifest_cursor
        self._requests = requests
        self._raw_observations = raw_observations
        encoded = np.stack(
            [
                self._encode(observation, request)
                for observation, request in zip(raw_observations, requests, strict=True)
            ]
        )
        encoded_states = torch.as_tensor(encoded, dtype=torch.float32)
        saved_last_obs = state["last_obs"]
        if not isinstance(saved_last_obs, Mapping) or set(saved_last_obs) != {"states"}:
            raise ValueError("Dynamic Benchmark checkpoint last observation is invalid")
        saved_states = torch.as_tensor(saved_last_obs["states"]).clone()
        if saved_states.dtype != torch.float32 or saved_states.shape != encoded_states.shape:
            raise ValueError("Dynamic Benchmark checkpoint last observation shape mismatch")
        if _state_tensor_sha256(saved_states) != state["last_obs_sha256"]:
            raise ValueError("Dynamic Benchmark checkpoint last observation hash mismatch")
        if not torch.equal(encoded_states, saved_states):
            raise RuntimeError(
                "Dynamic Benchmark checkpoint raw/vector observations are inconsistent"
            )
        self._last_obs = {"states": saved_states}
        for tracker, tracker_state in zip(
            self.reward_trackers, reward_states, strict=True
        ):
            tracker.load_state_dict(tracker_state)
        self._needs_reset = np.asarray(state["needs_reset"], dtype=bool).copy()
        if self._needs_reset.shape != (self.num_envs,):
            raise ValueError("Dynamic Benchmark checkpoint needs_reset shape mismatch")
        for name, dtype in (
            ("elapsed_steps", torch.int32),
            ("prev_step_reward", torch.float32),
            ("returns", torch.float32),
            ("success_once", torch.bool),
        ):
            value = torch.as_tensor(state[name], dtype=dtype).clone()
            if value.shape != (self.num_envs,):
                raise ValueError(f"Dynamic Benchmark checkpoint {name} shape mismatch")
            target_name = "_elapsed_steps" if name == "elapsed_steps" else name
            setattr(self, target_name, value)
        self._is_start = bool(state["is_start"])

    def close(self) -> None:
        if self._process_vector is not None:
            self._process_vector.close()
            self._process_vector = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for env in self.envs:
            env.close()
