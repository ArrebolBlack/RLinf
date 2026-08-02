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
from collections.abc import Mapping
from typing import Any, Optional, Union

import gym
import numpy as np
import torch

from .reward import DynamicBenchmarkReward
from .state_schema import DynamicBenchmarkStateSchema

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
        self.envs = [
            self._make_mujoco_env(
                self.task_id,
                image_size=self.image_size,
                camera_observations=self.camera_observations,
            )
            for _ in range(self.num_envs)
        ]
        self.horizon_steps = int(self.envs[0].horizon_steps)
        if any(int(env.horizon_steps) != self.horizon_steps for env in self.envs):
            raise RuntimeError("Dynamic Benchmark vector members disagree on horizon")
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
            from se3_wam.benchmark.api import ActionCommand, Split
            from se3_wam.benchmark.dataset_manifest import (
                make_dataset_candidate_manifest,
            )
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
        self._Split = Split
        self._make_dataset_candidate_manifest = make_dataset_candidate_manifest
        self._make_p0_grasp_candidate_manifest = make_p0_grasp_candidate_manifest
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

    def _reset_metrics(self, indices: np.ndarray) -> None:
        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        self._elapsed_steps[tensor_indices] = 0
        self.prev_step_reward[tensor_indices] = 0.0
        self.returns[tensor_indices] = 0.0
        self.success_once[tensor_indices] = False
        for index in indices:
            self.reward_trackers[int(index)].reset()

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
        for index in indices:
            request = self._next_request()
            observation = self.envs[int(index)].reset(request)
            state = self._encode(observation, request)
            if states.shape[1] == 0:
                states = np.zeros((self.num_envs, state.size), dtype=np.float32)
            states[int(index)] = state
            self._raw_observations[int(index)] = observation
            self._requests[int(index)] = request
            self._needs_reset[int(index)] = False
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
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        action_array = self._normalize_actions(actions)
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
        component_rows: list[dict[str, float]] = []
        event_name_rows: list[list[str]] = []
        active_stage_progresses = np.zeros(self.num_envs, dtype=np.float64)
        stepped = np.zeros(self.num_envs, dtype=bool)
        for index, env in enumerate(self.envs):
            if self._needs_reset[index]:
                terminations[index] = True
                component_rows.append({"total": 0.0})
                event_name_rows.append([])
                continue
            observation = self._raw_observations[index]
            request = self._requests[index]
            if observation is None or request is None:
                raise RuntimeError("Dynamic Benchmark vector member is not initialized")
            action = self._ActionCommand(
                mode=request.action_mode,
                values=action_array[index],
                policy_step=observation.policy_step,
            )
            result = env.step(action)
            event_names = tuple(event.name for event in env._ledger.events)
            event_name_rows.append(list(event_names))
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
            component_rows.append(components)
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
                "action": torch.as_tensor(action_array, dtype=torch.float64),
                "event_names": event_name_rows,
                "active_stage_progress": torch.as_tensor(
                    active_stage_progresses, dtype=torch.float64
                ),
                "success": success_tensor.clone(),
                "terminated": termination_tensor.clone(),
                "truncated": truncation_tensor.clone(),
                "termination_reason": list(termination_reasons),
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

        if self._last_obs is None or any(item is None for item in self._requests):
            raise RuntimeError("Dynamic Benchmark checkpoint requires initialized envs")
        env_states = [env.save_state() for env in self.envs]
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
        identity = {
            "task_id": self.task_id,
            "split": self.split_name,
            "base_manifest_seed": self.base_manifest_seed,
            "manifest_size": self.manifest_size,
            "image_size": self.image_size,
            "num_envs": self.num_envs,
            "state_schema": self.state_schema,
        }
        identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": "rlinf-dynamic-benchmark-checkpoint-v0.1",
            "identity": identity,
            "identity_sha256": identity_sha256,
            "manifest_generation": self._manifest_generation,
            "manifest_cursor": self._manifest_cursor,
            "requests": request_rows,
            "env_states": env_states,
            "reward_states": [tracker.state_dict() for tracker in self.reward_trackers],
            "needs_reset": self._needs_reset.copy(),
            "elapsed_steps": self._elapsed_steps.clone(),
            "prev_step_reward": self.prev_step_reward.clone(),
            "returns": self.returns.clone(),
            "success_once": self.success_once.clone(),
            "is_start": self._is_start,
        }

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpoint produced by :meth:`checkpoint_state`."""

        if state.get("schema_version") != "rlinf-dynamic-benchmark-checkpoint-v0.1":
            raise ValueError("unsupported Dynamic Benchmark checkpoint schema")
        identity = dict(state["identity"])
        expected = {
            "task_id": self.task_id,
            "split": self.split_name,
            "base_manifest_seed": self.base_manifest_seed,
            "manifest_size": self.manifest_size,
            "image_size": self.image_size,
            "num_envs": self.num_envs,
            "state_schema": self.state_schema,
        }
        if identity != expected:
            raise ValueError("Dynamic Benchmark checkpoint identity does not match env")
        observed_identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if observed_identity_sha256 != state["identity_sha256"]:
            raise ValueError("Dynamic Benchmark checkpoint identity hash mismatch")
        request_rows = list(state["requests"])
        env_states = list(state["env_states"])
        reward_states = list(state["reward_states"])
        if not (
            len(request_rows)
            == len(env_states)
            == len(reward_states)
            == self.num_envs
        ):
            raise ValueError("Dynamic Benchmark checkpoint vector length mismatch")
        from se3_wam.benchmark.api import (
            ActionMode,
            ObservationTrack,
            ResetRequest,
            Split,
        )

        raw_observations = []
        requests = []
        for env, row, env_state in zip(
            self.envs, request_rows, env_states, strict=True
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
            env.reset(request)
            raw_observations.append(env.load_state(env_state))
            requests.append(request)
        self._manifest_generation = int(state["manifest_generation"])
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
                for observation, request in zip(
                    raw_observations, requests, strict=True
                )
            ]
        )
        self._last_obs = {"states": torch.as_tensor(encoded, dtype=torch.float32)}
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
        for env in self.envs:
            env.close()
