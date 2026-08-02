#!/usr/bin/env python3
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

"""Train a resumable privileged-state SAC/RLPD expert on Dynamic Benchmark.

The residual-RLPD arm keeps the privileged planner in the executed action path
and learns a bounded correction.  Its critic and replay operate in residual
action space, so the zero action is exactly the frozen planner policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

if TYPE_CHECKING:
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy



@dataclass(frozen=True)
class TrainConfig:
    task: str
    algorithm: str
    rlinf_commit: str
    benchmark_commit: str
    seed: int
    num_envs: int
    eval_num_envs: int
    total_env_steps: int
    random_env_steps: int
    demo_episodes: int
    demo_max_attempts: int
    allow_failed_demos: bool
    demo_ratio: float
    bc_steps: int
    batch_size: int
    replay_capacity: int
    updates_per_vector_step: int
    q_heads: int
    q_target_subset: int
    gamma: float
    tau: float
    actor_lr: float
    actor_bc_weight: float
    residual_scale: float
    critic_lr: float
    alpha_lr: float
    initial_alpha: float
    target_entropy: float
    eval_interval: int
    eval_episodes: int
    checkpoint_interval: int
    log_interval: int
    train_manifest_seed: int
    validation_manifest_seed: int
    manifest_size: int
    image_size: int


class RunningNormalizer:
    """Float64 batch-Welford statistics with an unnormalized trailing mask."""

    def __init__(self, dimension: int, mask_dim: int, epsilon: float = 1e-5) -> None:
        if not 0 <= mask_dim <= dimension:
            raise ValueError("mask_dim must lie within the state dimension")
        self.dimension = int(dimension)
        self.mask_dim = int(mask_dim)
        self.epsilon = float(epsilon)
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.m2 = torch.zeros(dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        batch = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(
            -1, self.dimension
        )
        if batch.shape[0] == 0:
            return
        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean.add_(delta * (batch_count / total))
        self.m2.add_(batch_m2 + delta.square() * self.count * batch_count / total)
        self.count = total

    def normalize(self, values: torch.Tensor, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        if self.count < 2:
            return tensor
        variance = self.m2 / max(1, self.count - 1)
        mean = self.mean.clone()
        scale = torch.sqrt(torch.clamp(variance, min=self.epsilon**2))
        if self.mask_dim:
            mean[-self.mask_dim :] = 0.0
            scale[-self.mask_dim :] = 1.0
        normalized = (tensor - mean.to(device=device, dtype=torch.float32)) / scale.to(
            device=device, dtype=torch.float32
        )
        return torch.clamp(normalized, -10.0, 10.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "mask_dim": self.mask_dim,
            "epsilon": self.epsilon,
            "count": self.count,
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["dimension"]) != self.dimension or int(state["mask_dim"]) != self.mask_dim:
            raise ValueError("normalizer shape does not match checkpoint")
        self.epsilon = float(state["epsilon"])
        self.count = int(state["count"])
        self.mean.copy_(torch.as_tensor(state["mean"], dtype=torch.float64))
        self.m2.copy_(torch.as_tensor(state["m2"], dtype=torch.float64))


class TransitionReplay:
    """Bounded CPU replay with replacement sampling and exact cursor restore."""

    FIELDS = (
        "states",
        "actions",
        "rewards",
        "next_states",
        "terminated",
        "truncated",
    )

    def __init__(self, capacity: int, state_dim: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.cursor = 0
        self.size = 0
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.states = torch.empty((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.empty((capacity, 7), dtype=torch.float32)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.next_states = torch.empty((capacity, state_dim), dtype=torch.float32)
        self.terminated = torch.empty((capacity, 1), dtype=torch.bool)
        self.truncated = torch.empty((capacity, 1), dtype=torch.bool)

    def add(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        rows = int(torch.as_tensor(states).shape[0])
        if rows < 1:
            return
        payload = {
            "states": torch.as_tensor(states, dtype=torch.float32, device="cpu"),
            "actions": torch.as_tensor(actions, dtype=torch.float32, device="cpu"),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device="cpu").reshape(-1, 1),
            "next_states": torch.as_tensor(next_states, dtype=torch.float32, device="cpu"),
            "terminated": torch.as_tensor(terminated, dtype=torch.bool, device="cpu").reshape(-1, 1),
            "truncated": torch.as_tensor(truncated, dtype=torch.bool, device="cpu").reshape(-1, 1),
        }
        if any(value.shape[0] != rows for value in payload.values()):
            raise ValueError("replay fields disagree on batch length")
        for offset in range(rows):
            index = (self.cursor + offset) % self.capacity
            for name, value in payload.items():
                getattr(self, name)[index].copy_(value[offset])
        self.cursor = (self.cursor + rows) % self.capacity
        self.size = min(self.capacity, self.size + rows)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size < 1:
            raise RuntimeError("cannot sample an empty replay")
        indices = torch.randint(
            self.size,
            (int(batch_size),),
            generator=self.generator,
            device="cpu",
        )
        return {name: getattr(self, name).index_select(0, indices) for name in self.FIELDS}

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "cursor": self.cursor,
            "size": self.size,
            "generator_state": self.generator.get_state(),
            "data": {name: getattr(self, name)[: self.size].clone() for name in self.FIELDS},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["state_dim"]) != self.state_dim:
            raise ValueError("replay state dimension does not match checkpoint")
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError("checkpoint replay does not fit configured capacity")
        self.cursor = int(state["cursor"]) % self.capacity
        self.size = size
        for name in self.FIELDS:
            value = torch.as_tensor(state["data"][name], dtype=getattr(self, name).dtype)
            if value.shape != getattr(self, name)[:size].shape:
                raise ValueError(f"replay field {name} has the wrong shape")
            getattr(self, name)[:size].copy_(value)
        self.generator.set_state(state["generator_state"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML defaults; source commits and output stay explicit CLI arguments.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--algorithm",
        choices=("bc", "sac", "rlpd", "residual_rlpd"),
        default="rlpd",
    )
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--demo-replay-in",
        type=Path,
        help="Validated demo replay cache from a matching source/task/seed identity.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--eval-num-envs", type=int, default=4)
    parser.add_argument("--total-env-steps", type=int, default=200_000)
    parser.add_argument("--random-env-steps", type=int, default=2_000)
    parser.add_argument("--demo-episodes", type=int, default=32)
    parser.add_argument("--demo-max-attempts", type=int, default=320)
    parser.add_argument("--allow-failed-demos", action="store_true")
    parser.add_argument("--demo-ratio", type=float, default=0.5)
    parser.add_argument("--bc-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=250_000)
    parser.add_argument("--updates-per-vector-step", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=10)
    parser.add_argument("--q-target-subset", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--actor-bc-weight", type=float, default=0.0)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.25,
        help="Executed action is clamp(planner + residual_scale * policy_residual).",
    )
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--initial-alpha", type=float, default=0.01)
    parser.add_argument("--target-entropy", type=float, default=-7.0)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--train-manifest-seed", type=int, default=20261050)
    parser.add_argument("--validation-manifest-seed", type=int, default=20261150)
    parser.add_argument("--manifest-size", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=64)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Load recipe defaults without hiding run identity or output provenance."""
    arguments = sys.argv[1:] if argv is None else argv
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(arguments)
    parser = _parser()
    if pre_args.config is not None:
        import yaml

        payload = yaml.safe_load(pre_args.config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Dynamic Benchmark config must be a YAML mapping")
        explicit_only = {
            "benchmark_commit",
            "config",
            "demo_replay_in",
            "output",
            "resume",
            "rlinf_commit",
        }
        actions = {action.dest: action for action in parser._actions}
        unknown = sorted(set(payload) - set(actions))
        forbidden = sorted(set(payload) & explicit_only)
        if unknown:
            raise ValueError(f"unknown Dynamic Benchmark config keys: {unknown}")
        if forbidden:
            raise ValueError(f"run-specific keys must stay on the CLI: {forbidden}")
        parser.set_defaults(**payload)
        for name in payload:
            actions[name].required = False
    return parser.parse_args(arguments)


def _config(args: argparse.Namespace) -> TrainConfig:
    demo_ratio = 0.0 if args.algorithm == "sac" else float(args.demo_ratio)
    config = TrainConfig(
        task=args.task,
        algorithm=args.algorithm,
        rlinf_commit=args.rlinf_commit,
        benchmark_commit=args.benchmark_commit,
        seed=args.seed,
        num_envs=args.num_envs,
        eval_num_envs=args.eval_num_envs,
        total_env_steps=args.total_env_steps,
        random_env_steps=args.random_env_steps,
        demo_episodes=args.demo_episodes,
        demo_max_attempts=args.demo_max_attempts,
        allow_failed_demos=args.allow_failed_demos,
        demo_ratio=demo_ratio,
        bc_steps=args.bc_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        updates_per_vector_step=args.updates_per_vector_step,
        q_heads=args.q_heads,
        q_target_subset=args.q_target_subset,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        actor_bc_weight=args.actor_bc_weight,
        residual_scale=args.residual_scale,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        initial_alpha=args.initial_alpha,
        target_entropy=args.target_entropy,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        train_manifest_seed=args.train_manifest_seed,
        validation_manifest_seed=args.validation_manifest_seed,
        manifest_size=args.manifest_size,
        image_size=args.image_size,
    )
    if config.num_envs < 1 or config.eval_num_envs < 1:
        raise ValueError("environment counts must be positive")
    for name, commit in (
        ("rlinf_commit", config.rlinf_commit),
        ("benchmark_commit", config.benchmark_commit),
    ):
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError(f"{name} must be a full lowercase Git commit")
    if config.q_heads < 2 or not 1 <= config.q_target_subset <= config.q_heads:
        raise ValueError("Q ensemble/subset configuration is invalid")
    if not 0.0 <= config.demo_ratio <= 1.0:
        raise ValueError("demo_ratio must be in [0, 1]")
    if config.actor_bc_weight < 0.0:
        raise ValueError("actor_bc_weight must be non-negative")
    if not 0.0 < config.residual_scale <= 1.0:
        raise ValueError("residual_scale must be in (0, 1]")
    if config.algorithm in {"bc", "rlpd", "residual_rlpd"} and config.demo_episodes < 1:
        raise ValueError("BC/RLPD requires at least one demonstration episode")
    if config.batch_size < 2 or config.replay_capacity < config.batch_size:
        raise ValueError("replay capacity must be at least the batch size")
    return config


def _env_cfg(config: TrainConfig, *, split: str, seed: int, num_envs: int) -> dict[str, Any]:
    return {
        "task_id": config.task,
        "split": split,
        "manifest_seed": seed,
        "manifest_size": max(config.manifest_size, 2 * num_envs),
        "image_size": config.image_size,
        "camera_observations": False,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _demo_replay_identity(
    config: TrainConfig,
    state_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "rlinf-dynamic-benchmark-demo-identity-v0.1",
        "task": config.task,
        "rlinf_commit": config.rlinf_commit,
        "benchmark_commit": config.benchmark_commit,
        "seed": config.seed,
        "num_envs": config.num_envs,
        "demo_episodes": config.demo_episodes,
        "demo_max_attempts": config.demo_max_attempts,
        "allow_failed_demos": config.allow_failed_demos,
        "train_manifest_seed": config.train_manifest_seed,
        "manifest_size": config.manifest_size,
        "image_size": config.image_size,
        "state_schema": state_schema,
    }


def _save_demo_replay_cache(
    path: Path,
    identity: dict[str, Any],
    demo_summary: dict[str, Any],
    replay: TransitionReplay,
    normalizer: RunningNormalizer,
) -> str:
    payload = {
        "schema_version": "rlinf-dynamic-benchmark-demo-replay-v0.1",
        "identity": identity,
        "demo_summary": demo_summary,
        "replay": replay.state_dict(),
        "normalizer": normalizer.state_dict(),
        "rng_after_collection": _rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return _file_sha256(path)


def _load_demo_replay_cache(
    path: Path,
    expected_identity: dict[str, Any],
    replay: TransitionReplay,
    normalizer: RunningNormalizer,
) -> tuple[dict[str, Any], str]:
    cache_sha256 = _file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "rlinf-dynamic-benchmark-demo-replay-v0.1":
        raise ValueError("demo replay cache schema does not match")
    if payload.get("identity") != expected_identity:
        raise ValueError("demo replay cache identity does not match current run")
    replay.load_state_dict(payload["replay"])
    normalizer.load_state_dict(payload["normalizer"])
    _restore_rng(payload["rng_after_collection"])
    demo_summary = payload["demo_summary"]
    if int(demo_summary["accepted_episodes"]) != int(
        expected_identity["demo_episodes"]
    ):
        raise ValueError("demo replay cache episode count does not match")
    if replay.size != int(demo_summary["transitions"]):
        raise ValueError("demo replay cache transition count does not match")
    return demo_summary, cache_sha256


def _policy_action(
    model: MLPPolicy,
    normalizer: RunningNormalizer,
    states: torch.Tensor,
    device: torch.device,
    *,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = normalizer.normalize(states, device)
    mean, log_std = model._sample_actions(normalized)
    distribution = Normal(mean, torch.exp(log_std))
    raw = distribution.rsample() if stochastic else mean
    action = torch.tanh(raw)
    log_prob = distribution.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
    return action, log_prob.sum(dim=-1, keepdim=True)


def _make_planner_teachers(task: str, env: Any) -> list[Any]:
    """Construct one reset planner per live vector environment request."""
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    teachers = []
    for request in env._requests:
        teacher, _ = make_privileged_teacher(task, request=request)
        if hasattr(teacher, "reset"):
            teacher.reset()
        teachers.append(teacher)
    return teachers


def _reset_planner_teachers(task: str, env: Any, teachers: list[Any], indices: list[int]) -> None:
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    for index in indices:
        teacher, _ = make_privileged_teacher(task, request=env._requests[index])
        if hasattr(teacher, "reset"):
            teacher.reset()
        teachers[index] = teacher


def _planner_actions(env: Any, teachers: list[Any]) -> torch.Tensor:
    if len(teachers) != len(env._raw_observations):
        raise ValueError("planner count does not match vector environment")
    values = []
    for index, teacher in enumerate(teachers):
        observation = env._raw_observations[index]
        if observation is None:
            raise RuntimeError("planner-residual environment lost its raw observation")
        values.append(np.asarray(teacher.act(observation).values, dtype=np.float32))
    actions = torch.as_tensor(np.stack(values), dtype=torch.float32)
    if actions.shape != (len(teachers), 7):
        raise ValueError(f"planner action has unexpected shape {tuple(actions.shape)}")
    return actions


def _compose_residual_actions(
    planner_actions: torch.Tensor,
    residual_actions: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    planner = torch.as_tensor(planner_actions, dtype=torch.float32, device="cpu")
    residual = torch.as_tensor(residual_actions, dtype=torch.float32, device="cpu")
    if planner.shape != residual.shape or planner.shape[-1] != 7:
        raise ValueError("planner and residual actions must have matching [..., 7] shapes")
    if not 0.0 < residual_scale <= 1.0:
        raise ValueError("residual_scale must be in (0, 1]")
    return torch.clamp(planner + float(residual_scale) * residual, -1.0, 1.0)


def _mixed_batch(
    online: TransitionReplay,
    demos: TransitionReplay,
    batch_size: int,
    demo_ratio: float,
) -> dict[str, torch.Tensor]:
    if online.size == 0 and demos.size == 0:
        raise RuntimeError("both replay buffers are empty")
    demo_rows = int(round(batch_size * demo_ratio)) if demos.size else 0
    if online.size == 0:
        demo_rows = batch_size
    online_rows = batch_size - demo_rows
    if demos.size == 0:
        online_rows = batch_size
    chunks = []
    if online_rows:
        chunks.append(online.sample(online_rows))
    if demo_rows:
        chunks.append(demos.sample(demo_rows))
    merged = {
        name: torch.cat([chunk[name] for chunk in chunks], dim=0)
        for name in TransitionReplay.FIELDS
    }
    permutation = torch.randperm(batch_size)
    return {name: value.index_select(0, permutation) for name, value in merged.items()}


def _collect_demos(
    config: TrainConfig,
    replay: TransitionReplay,
    normalizer: RunningNormalizer,
) -> dict[str, Any]:
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    count = min(config.num_envs, config.demo_episodes)
    env = DynamicBenchmarkEnv(
        cfg=_env_cfg(
            config,
            split="train",
            seed=config.train_manifest_seed + 700_001,
            num_envs=count,
        ),
        num_envs=count,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    accepted = 0
    attempts = 0
    successes = 0
    trajectories: list[list[tuple[torch.Tensor, ...]]] = [[] for _ in range(count)]
    try:
        obs = env._last_obs
        if obs is None:
            raise RuntimeError("demo environment did not initialize its state")
        teachers = []
        for request in env._requests:
            teacher, _ = make_privileged_teacher(config.task, request=request)
            if hasattr(teacher, "reset"):
                teacher.reset()
            teachers.append(teacher)
        while accepted < config.demo_episodes and attempts < config.demo_max_attempts:
            action_values = []
            for index, teacher in enumerate(teachers):
                observation = env._raw_observations[index]
                if observation is None:
                    raise RuntimeError("demo environment lost its raw observation")
                action_values.append(np.asarray(teacher.act(observation).values, dtype=np.float32))
            actions = torch.as_tensor(np.stack(action_values), dtype=torch.float32)
            next_obs, rewards, terminated, truncated, infos = env.step(
                actions,
                auto_reset=False,
            )
            dones = torch.logical_or(terminated, truncated)
            for index in range(count):
                trajectories[index].append(
                    (
                        obs["states"][index].clone(),
                        actions[index].clone(),
                        rewards[index].clone(),
                        next_obs["states"][index].clone(),
                        terminated[index].clone(),
                        truncated[index].clone(),
                    )
                )
            done_indices = torch.arange(count)[dones].tolist()
            for index in done_indices:
                attempts += 1
                success = bool(infos["success"][index])
                successes += int(success)
                if success or config.allow_failed_demos:
                    trajectory = trajectories[index]
                    fields = list(zip(*trajectory, strict=True))
                    replay.add(*(torch.stack(field) for field in fields))
                    normalizer.update(torch.stack(fields[0]))
                    normalizer.update(torch.stack(fields[3]))
                    accepted += 1
                trajectories[index] = []
            if accepted >= config.demo_episodes or attempts >= config.demo_max_attempts:
                break
            if done_indices:
                reset_obs, _ = env.reset(options={"env_idx": done_indices})
                next_obs = reset_obs
                for index in done_indices:
                    teacher, _ = make_privileged_teacher(
                        config.task,
                        request=env._requests[index],
                    )
                    if hasattr(teacher, "reset"):
                        teacher.reset()
                    teachers[index] = teacher
            obs = next_obs
    finally:
        env.close()
    if accepted < config.demo_episodes:
        raise RuntimeError(
            "planner did not supply enough accepted demonstrations: "
            f"accepted={accepted}, attempts={attempts}, target={config.demo_episodes}"
        )
    return {
        "accepted_episodes": accepted,
        "attempted_episodes": attempts,
        "successful_attempts": successes,
        "success_rate": successes / max(1, attempts),
        "transitions": replay.size,
    }


def _evaluate(
    config: TrainConfig,
    model: MLPPolicy,
    normalizer: RunningNormalizer,
    device: torch.device,
) -> dict[str, Any]:
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    preserved_rng = _rng_state()
    count = min(config.eval_num_envs, config.eval_episodes)
    env = DynamicBenchmarkEnv(
        cfg=_env_cfg(
            config,
            split="validation",
            seed=config.validation_manifest_seed,
            num_envs=count,
        ),
        num_envs=count,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    records = []
    episode_returns = torch.zeros(count)
    episode_effort = torch.zeros(count)
    episode_steps = torch.zeros(count, dtype=torch.int64)
    safety_failures = set(env.reward_schema["safety_failures"])
    teachers = _make_planner_teachers(config.task, env) if config.algorithm == "residual_rlpd" else None
    model.eval()
    try:
        obs = env._last_obs
        if obs is None:
            raise RuntimeError("evaluation environment did not initialize its state")
        while len(records) < config.eval_episodes:
            with torch.inference_mode():
                actions, _ = _policy_action(
                    model,
                    normalizer,
                    obs["states"],
                    device,
                    stochastic=False,
                )
            policy_actions = actions.cpu()
            env_actions = policy_actions
            if teachers is not None:
                env_actions = _compose_residual_actions(
                    _planner_actions(env, teachers),
                    policy_actions,
                    config.residual_scale,
                )
            next_obs, rewards, terminated, truncated, infos = env.step(env_actions, auto_reset=False)
            episode_returns += rewards
            episode_effort += env_actions.square().sum(dim=-1)
            episode_steps += 1
            dones = torch.logical_or(terminated, truncated)
            done_indices = torch.arange(count)[dones].tolist()
            for index in done_indices:
                reason = infos["termination_reason"][index]
                records.append(
                    {
                        "success": bool(infos["success"][index]),
                        "safety_failure": reason in safety_failures,
                        "termination_reason": reason,
                        "trajectory_completion": float(
                            infos["trajectory_completion"][index]
                        ),
                        "return": float(episode_returns[index]),
                        "duration_steps": int(episode_steps[index]),
                        "action_l2_sum": float(episode_effort[index]),
                    }
                )
                episode_returns[index] = 0.0
                episode_effort[index] = 0.0
                episode_steps[index] = 0
            if len(records) >= config.eval_episodes:
                break
            if done_indices:
                next_obs, _ = env.reset(options={"env_idx": done_indices})
                if teachers is not None:
                    _reset_planner_teachers(config.task, env, teachers, done_indices)
            obs = next_obs
    finally:
        env.close()
        model.train()
        _restore_rng(preserved_rng)
    records = records[: config.eval_episodes]
    return {
        "episodes": len(records),
        "success_rate": float(np.mean([record["success"] for record in records])),
        "safety_failure_rate": float(
            np.mean([record["safety_failure"] for record in records])
        ),
        "mean_completion": float(
            np.mean([record["trajectory_completion"] for record in records])
        ),
        "mean_return": float(np.mean([record["return"] for record in records])),
        "mean_duration_steps": float(
            np.mean([record["duration_steps"] for record in records])
        ),
        "mean_action_l2_sum": float(
            np.mean([record["action_l2_sum"] for record in records])
        ),
        "records": records,
    }


def _score(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["success_rate"]),
        -float(metrics["safety_failure_rate"]),
        float(metrics["mean_completion"]),
        float(metrics["mean_return"]),
        -float(metrics["mean_duration_steps"]),
        -float(metrics["mean_action_l2_sum"]),
    )


def _parameter_groups(model: MLPPolicy) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    critic = list(model.q_head.parameters())
    critic_ids = {id(parameter) for parameter in critic}
    actor = [parameter for parameter in model.parameters() if id(parameter) not in critic_ids]
    if not actor or not critic or {id(parameter) for parameter in actor} & critic_ids:
        raise RuntimeError("actor/critic parameter partition is invalid")
    return actor, critic


def _bc_warm_start(
    config: TrainConfig,
    model: MLPPolicy,
    actor_optimizer: torch.optim.Optimizer,
    normalizer: RunningNormalizer,
    demos: TransitionReplay,
    device: torch.device,
    metrics_path: Path,
) -> None:
    if config.bc_steps <= 0 or demos.size == 0:
        return
    model.train()
    for step in range(1, config.bc_steps + 1):
        batch = demos.sample(config.batch_size)
        states = normalizer.normalize(batch["states"], device)
        target = batch["actions"].to(device)
        if config.algorithm == "residual_rlpd":
            target = torch.zeros_like(target)
        mean, _ = model._sample_actions(states)
        loss = F.mse_loss(torch.tanh(mean), target)
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in actor_optimizer.param_groups for parameter in group["params"]],
            10.0,
        )
        actor_optimizer.step()
        if step == 1 or step % max(1, config.bc_steps // 10) == 0:
            _append_jsonl(
                metrics_path,
                {"event": "bc", "bc_step": step, "bc_loss": float(loss.detach())},
            )


def _sac_update(
    config: TrainConfig,
    model: MLPPolicy,
    target_q: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: torch.optim.Optimizer,
    normalizer: RunningNormalizer,
    online: TransitionReplay,
    demos: TransitionReplay,
    device: torch.device,
) -> dict[str, float]:
    batch = _mixed_batch(online, demos, config.batch_size, config.demo_ratio)
    states = normalizer.normalize(batch["states"], device)
    next_states = normalizer.normalize(batch["next_states"], device)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device)
    dones = torch.logical_or(batch["terminated"], batch["truncated"]).to(
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        next_actions, next_log_prob = _policy_action(
            model,
            normalizer,
            batch["next_states"],
            device,
            stochastic=True,
        )
        target_values = target_q(next_states, next_actions)
        subset = torch.randperm(config.q_heads, device=device)[: config.q_target_subset]
        target_min = target_values.index_select(-1, subset).min(dim=-1, keepdim=True).values
        alpha = log_alpha.exp()
        bootstrap = target_min - alpha * next_log_prob
        target = rewards + config.gamma * (1.0 - dones) * bootstrap
    predicted = model.q_head(states, actions)
    critic_loss = F.mse_loss(predicted, target.expand_as(predicted))
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.q_head.parameters(), 10.0)
    critic_optimizer.step()

    for parameter in model.q_head.parameters():
        parameter.requires_grad_(False)
    policy_actions, log_prob = _policy_action(
        model,
        normalizer,
        batch["states"],
        device,
        stochastic=True,
    )
    policy_q = model.q_head(states, policy_actions).mean(dim=-1, keepdim=True)
    alpha = log_alpha.exp()
    actor_sac_loss = (alpha.detach() * log_prob - policy_q).mean()
    actor_bc_loss = torch.zeros((), dtype=torch.float32, device=device)
    if config.actor_bc_weight > 0.0 and demos.size > 0:
        demo_batch = demos.sample(config.batch_size)
        demo_states = normalizer.normalize(demo_batch["states"], device)
        demo_actions = demo_batch["actions"].to(device)
        if config.algorithm == "residual_rlpd":
            demo_actions = torch.zeros_like(demo_actions)
        demo_mean, _ = model._sample_actions(demo_states)
        actor_bc_loss = F.mse_loss(torch.tanh(demo_mean), demo_actions)
    actor_loss = actor_sac_loss + config.actor_bc_weight * actor_bc_loss
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for group in actor_optimizer.param_groups for parameter in group["params"]],
        10.0,
    )
    actor_optimizer.step()
    for parameter in model.q_head.parameters():
        parameter.requires_grad_(True)

    alpha_loss = -(log_alpha * (log_prob + config.target_entropy).detach()).mean()
    alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_optimizer.step()
    with torch.no_grad():
        for online_parameter, target_parameter in zip(
            model.q_head.parameters(), target_q.parameters(), strict=True
        ):
            target_parameter.lerp_(online_parameter, config.tau)
    return {
        "critic_loss": float(critic_loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "actor_sac_loss": float(actor_sac_loss.detach()),
        "actor_bc_loss": float(actor_bc_loss.detach()),
        "alpha_loss": float(alpha_loss.detach()),
        "alpha": float(log_alpha.exp().detach()),
        "q_data": float(predicted.detach().mean()),
        "q_target": float(target.detach().mean()),
    }


def _save_policy(
    path: Path,
    config: TrainConfig,
    model: MLPPolicy,
    normalizer: RunningNormalizer,
    state_schema: dict[str, Any],
    metrics: dict[str, Any],
    env_steps: int,
) -> None:
    payload = {
        "schema_version": "rlinf-dynamic-benchmark-expert-policy-v0.1",
        "config": asdict(config),
        "model": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "state_schema": state_schema,
        "validation": metrics,
        "env_steps": env_steps,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    args = _parse_args()
    config = _config(args)
    if args.demo_replay_in is not None and config.algorithm == "sac":
        raise ValueError("SAC does not consume a demonstration replay cache")
    if args.demo_replay_in is not None and args.resume is not None:
        raise ValueError("resume already contains demonstrations; do not pass demo replay cache")
    if not torch.cuda.is_available():
        raise RuntimeError("Dynamic Benchmark expert training requires CUDA")
    if args.resume is None:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"refusing non-empty output directory {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    heartbeat_path = args.output / "heartbeat.json"
    config_path = args.output / "config.json"
    _atomic_json(config_path, asdict(config))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda:0")
    env = DynamicBenchmarkEnv(
        cfg=_env_cfg(
            config,
            split="train",
            seed=config.train_manifest_seed,
            num_envs=config.num_envs,
        ),
        num_envs=config.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    obs = env._last_obs
    if obs is None:
        raise RuntimeError("training environment did not initialize its state")
    state_schema = env.state_schema
    state_dim = int(state_schema["state_dim"])
    normalizer = RunningNormalizer(state_dim, int(state_schema["mask_dim"]))
    demos = TransitionReplay(config.replay_capacity, state_dim, config.seed + 11)
    online = TransitionReplay(config.replay_capacity, state_dim, config.seed + 17)
    model = MLPPolicy(
        obs_dim=state_dim,
        action_dim=7,
        num_action_chunks=1,
        add_value_head=False,
        add_q_head=True,
        q_head_type="default",
        critic_obs_dim=state_dim,
        num_q_heads=config.q_heads,
    ).to(device)
    target_q = type(model.q_head)(
        hidden_size=state_dim,
        action_feature_dim=7,
        hidden_dims=[256, 256, 256],
        num_q_heads=config.q_heads,
        output_dim=1,
    ).to(device)
    target_q.load_state_dict(model.q_head.state_dict())
    target_q.requires_grad_(False)
    actor_parameters, critic_parameters = _parameter_groups(model)
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=config.critic_lr)
    log_alpha = torch.tensor(
        math.log(config.initial_alpha),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
    global_env_steps = 0
    update_steps = 0
    best_score: tuple[float, ...] | None = None
    best_metrics: dict[str, Any] | None = None
    next_eval = config.eval_interval
    next_checkpoint = config.checkpoint_interval
    next_log = config.log_interval
    recent_episodes: list[dict[str, Any]] = []
    demo_source: dict[str, Any] | None = None
    last_validation: dict[str, Any] | None = None
    last_validation_env_steps = -1
    training_teachers: list[Any] | None = None
    start_time = time.monotonic()

    def checkpoint(path: Path) -> None:
        state = {
            "schema_version": "rlinf-dynamic-benchmark-trainer-checkpoint-v0.1",
            "config": asdict(config),
            "state_schema": state_schema,
            "model": model.state_dict(),
            "target_q": target_q.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "alpha_optimizer": alpha_optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "demos": demos.state_dict(),
            "demo_source": demo_source,
            "online": online.state_dict(),
            "env": env.checkpoint_state(),
            "global_env_steps": global_env_steps,
            "update_steps": update_steps,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "next_eval": next_eval,
            "next_checkpoint": next_checkpoint,
            "next_log": next_log,
            "last_validation": last_validation,
            "last_validation_env_steps": last_validation_env_steps,
            "planner_teachers": training_teachers,
            "rng": _rng_state(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    if args.resume is not None:
        restored = torch.load(args.resume, map_location="cpu", weights_only=False)
        if restored["config"] != asdict(config) or restored["state_schema"] != state_schema:
            raise ValueError("resume config/state schema does not match current run")
        model.load_state_dict(restored["model"])
        target_q.load_state_dict(restored["target_q"])
        actor_optimizer.load_state_dict(restored["actor_optimizer"])
        critic_optimizer.load_state_dict(restored["critic_optimizer"])
        log_alpha.data.copy_(restored["log_alpha"].to(device))
        alpha_optimizer.load_state_dict(restored["alpha_optimizer"])
        normalizer.load_state_dict(restored["normalizer"])
        demos.load_state_dict(restored["demos"])
        demo_source = restored.get("demo_source")
        online.load_state_dict(restored["online"])
        env.load_checkpoint_state(restored["env"])
        obs = env._last_obs
        if obs is None:
            raise RuntimeError("resumed environment did not expose its current state")
        global_env_steps = int(restored["global_env_steps"])
        update_steps = int(restored["update_steps"])
        best_score = restored["best_score"]
        best_metrics = restored["best_metrics"]
        next_eval = int(restored["next_eval"])
        next_checkpoint = int(restored["next_checkpoint"])
        next_log = int(restored["next_log"])
        last_validation = restored.get("last_validation")
        last_validation_env_steps = int(restored.get("last_validation_env_steps", -1))
        training_teachers = restored.get("planner_teachers")
        if config.algorithm == "residual_rlpd":
            if training_teachers is None or len(training_teachers) != config.num_envs:
                raise ValueError("residual checkpoint is missing planner teacher state")
        elif training_teachers is not None:
            raise ValueError("non-residual checkpoint unexpectedly contains planner state")
        _restore_rng(restored["rng"])
        _append_jsonl(metrics_path, {"event": "resume", "env_steps": global_env_steps})
    else:
        demo_summary = None
        if config.algorithm in {"bc", "rlpd", "residual_rlpd"}:
            demo_identity = _demo_replay_identity(config, state_schema)
            if args.demo_replay_in is None:
                demo_summary = _collect_demos(config, demos, normalizer)
                demo_cache_path = args.output / "demo_replay.pt"
                demo_cache_sha256 = _save_demo_replay_cache(
                    demo_cache_path,
                    demo_identity,
                    demo_summary,
                    demos,
                    normalizer,
                )
                demo_source = {
                    "mode": "collected",
                    "sha256": demo_cache_sha256,
                }
            else:
                demo_summary, demo_cache_sha256 = _load_demo_replay_cache(
                    args.demo_replay_in,
                    demo_identity,
                    demos,
                    normalizer,
                )
                demo_source = {
                    "mode": "cache",
                    "sha256": demo_cache_sha256,
                }
            _append_jsonl(
                metrics_path,
                {"event": "demos", "demo_source": demo_source, **demo_summary},
            )
            if config.algorithm == "residual_rlpd":
                # The persisted cache stays algorithm-neutral (planner actions), while
                # the residual MDP treats the same planner transitions as action zero.
                demos.actions[: demos.size].zero_()
        normalizer.update(obs["states"])
        _bc_warm_start(
            config,
            model,
            actor_optimizer,
            normalizer,
            demos,
            device,
            metrics_path,
        )
        initial_validation = _evaluate(config, model, normalizer, device)
        last_validation = initial_validation
        last_validation_env_steps = 0
        best_score = _score(initial_validation)
        best_metrics = initial_validation
        _append_jsonl(
            metrics_path,
            {"event": "validation", "env_steps": 0, **initial_validation},
        )
        _save_policy(
            args.output / "best_policy.pt",
            config,
            model,
            normalizer,
            state_schema,
            initial_validation,
            0,
        )
        if config.algorithm == "bc":
            summary = {
                "schema_version": "rlinf-dynamic-benchmark-expert-summary-v0.1",
                "status": "complete",
                "config": asdict(config),
                "demo_summary": demo_summary,
                "demo_source": demo_source,
                "best_validation": best_metrics,
                "best_score": best_score,
                "env_steps": 0,
                "update_steps": 0,
            }
            _atomic_json(args.output / "summary.json", summary)
            env.close()
            return

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    episode_returns = torch.zeros(config.num_envs)
    episode_steps = torch.zeros(config.num_envs, dtype=torch.int64)
    if config.algorithm == "residual_rlpd" and training_teachers is None:
        training_teachers = _make_planner_teachers(config.task, env)
    try:
        while global_env_steps < config.total_env_steps and not stop_requested:
            if global_env_steps < config.random_env_steps:
                policy_actions = torch.empty((config.num_envs, 7)).uniform_(-1.0, 1.0)
            else:
                with torch.inference_mode():
                    policy_actions, _ = _policy_action(
                        model,
                        normalizer,
                        obs["states"],
                        device,
                        stochastic=True,
                    )
                    policy_actions = policy_actions.cpu()
            env_actions = policy_actions
            if training_teachers is not None:
                env_actions = _compose_residual_actions(
                    _planner_actions(env, training_teachers),
                    policy_actions,
                    config.residual_scale,
                )
            next_obs, rewards, terminated, truncated, infos = env.step(env_actions, auto_reset=False)
            online.add(
                obs["states"],
                policy_actions,
                rewards,
                next_obs["states"],
                terminated,
                truncated,
            )
            normalizer.update(next_obs["states"])
            episode_returns += rewards
            episode_steps += 1
            dones = torch.logical_or(terminated, truncated)
            done_indices = torch.arange(config.num_envs)[dones].tolist()
            for index in done_indices:
                record = {
                    "success": bool(infos["success"][index]),
                    "termination_reason": infos["termination_reason"][index],
                    "trajectory_completion": float(
                        infos["trajectory_completion"][index]
                    ),
                    "return": float(episode_returns[index]),
                    "duration_steps": int(episode_steps[index]),
                }
                recent_episodes.append(record)
                episode_returns[index] = 0.0
                episode_steps[index] = 0
            if done_indices:
                next_obs, _ = env.reset(options={"env_idx": done_indices})
                if training_teachers is not None:
                    _reset_planner_teachers(
                        config.task,
                        env,
                        training_teachers,
                        done_indices,
                    )
                normalizer.update(next_obs["states"].index_select(0, torch.tensor(done_indices)))
            obs = next_obs
            global_env_steps += config.num_envs

            if online.size >= max(config.batch_size, config.random_env_steps):
                update_metrics = None
                for _ in range(config.updates_per_vector_step):
                    update_metrics = _sac_update(
                        config,
                        model,
                        target_q,
                        actor_optimizer,
                        critic_optimizer,
                        log_alpha,
                        alpha_optimizer,
                        normalizer,
                        online,
                        demos,
                        device,
                    )
                    update_steps += 1
            else:
                update_metrics = None

            if global_env_steps >= next_log:
                elapsed = max(time.monotonic() - start_time, 1e-6)
                payload: dict[str, Any] = {
                    "event": "train",
                    "env_steps": global_env_steps,
                    "update_steps": update_steps,
                    "online_replay_size": online.size,
                    "demo_replay_size": demos.size,
                    "env_steps_per_second": global_env_steps / elapsed,
                }
                if recent_episodes:
                    payload.update(
                        recent_episode_count=len(recent_episodes),
                        recent_success_rate=float(
                            np.mean([item["success"] for item in recent_episodes])
                        ),
                        recent_mean_completion=float(
                            np.mean(
                                [item["trajectory_completion"] for item in recent_episodes]
                            )
                        ),
                        recent_mean_return=float(
                            np.mean([item["return"] for item in recent_episodes])
                        ),
                    )
                    recent_episodes.clear()
                if update_metrics is not None:
                    payload.update(update_metrics)
                _append_jsonl(metrics_path, payload)
                _atomic_json(heartbeat_path, payload)
                next_log += config.log_interval

            if global_env_steps >= next_eval:
                validation = _evaluate(config, model, normalizer, device)
                last_validation = validation
                last_validation_env_steps = global_env_steps
                score = _score(validation)
                _append_jsonl(
                    metrics_path,
                    {"event": "validation", "env_steps": global_env_steps, **validation},
                )
                if best_score is None or score > tuple(best_score):
                    best_score = score
                    best_metrics = validation
                    _save_policy(
                        args.output / "best_policy.pt",
                        config,
                        model,
                        normalizer,
                        state_schema,
                        validation,
                        global_env_steps,
                    )
                next_eval += config.eval_interval

            if global_env_steps >= next_checkpoint:
                next_checkpoint += config.checkpoint_interval
                checkpoint(args.output / "checkpoint_latest.pt")

        checkpoint(args.output / "checkpoint_latest.pt")
        if last_validation is not None and last_validation_env_steps == global_env_steps:
            final_validation = last_validation
            _append_jsonl(
                metrics_path,
                {
                    "event": "validation_reused",
                    "env_steps": global_env_steps,
                    "reason": "scheduled_validation_matches_final_state",
                },
            )
        else:
            final_validation = _evaluate(config, model, normalizer, device)
            last_validation = final_validation
            last_validation_env_steps = global_env_steps
        final_score = _score(final_validation)
        if best_score is None or final_score > tuple(best_score):
            best_score = final_score
            best_metrics = final_validation
            _save_policy(
                args.output / "best_policy.pt",
                config,
                model,
                normalizer,
                state_schema,
                final_validation,
                global_env_steps,
            )
        _save_policy(
            args.output / "final_policy.pt",
            config,
            model,
            normalizer,
            state_schema,
            final_validation,
            global_env_steps,
        )
        summary = {
            "schema_version": "rlinf-dynamic-benchmark-expert-summary-v0.1",
            "status": "stopped" if stop_requested else "complete",
            "config": asdict(config),
            "demo_source": demo_source,
            "best_validation": best_metrics,
            "best_score": best_score,
            "final_validation": final_validation,
            "env_steps": global_env_steps,
            "update_steps": update_steps,
            "wall_time_s": time.monotonic() - start_time,
        }
        rendered = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        summary["payload_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
        _atomic_json(args.output / "summary.json", summary)
    finally:
        env.close()


if __name__ == "__main__":
    main()
