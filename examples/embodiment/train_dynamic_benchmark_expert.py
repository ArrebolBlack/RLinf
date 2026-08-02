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

"""Train a resumable privileged-state SAC/RLPD expert on Dynamic Benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


@dataclass(frozen=True)
class TrainConfig:
    task: str
    algorithm: str
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
    parser.add_argument("--task", required=True)
    parser.add_argument("--algorithm", choices=("bc", "sac", "rlpd"), default="rlpd")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
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


def _config(args: argparse.Namespace) -> TrainConfig:
    demo_ratio = 0.0 if args.algorithm == "sac" else float(args.demo_ratio)
    config = TrainConfig(
        task=args.task,
        algorithm=args.algorithm,
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
    if config.q_heads < 2 or not 1 <= config.q_target_subset <= config.q_heads:
        raise ValueError("Q ensemble/subset configuration is invalid")
    if not 0.0 <= config.demo_ratio <= 1.0:
        raise ValueError("demo_ratio must be in [0, 1]")
    if config.algorithm in {"bc", "rlpd"} and config.demo_episodes < 1:
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
        obs, _ = env.reset()
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
    model.eval()
    try:
        obs, _ = env.reset()
        while len(records) < config.eval_episodes:
            with torch.inference_mode():
                actions, _ = _policy_action(
                    model,
                    normalizer,
                    obs["states"],
                    device,
                    stochastic=False,
                )
            next_obs, rewards, terminated, truncated, infos = env.step(
                actions.cpu(), auto_reset=False
            )
            episode_returns += rewards
            episode_effort += actions.cpu().square().sum(dim=-1)
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
    actor_loss = (alpha.detach() * log_prob - policy_q).mean()
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
    args = _parser().parse_args()
    config = _config(args)
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
    obs, _ = env.reset()
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
            "online": online.state_dict(),
            "env": env.checkpoint_state(),
            "global_env_steps": global_env_steps,
            "update_steps": update_steps,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "next_eval": next_eval,
            "next_checkpoint": next_checkpoint,
            "next_log": next_log,
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
        _restore_rng(restored["rng"])
        _append_jsonl(metrics_path, {"event": "resume", "env_steps": global_env_steps})
    else:
        demo_summary = None
        if config.algorithm in {"bc", "rlpd"}:
            demo_summary = _collect_demos(config, demos, normalizer)
            _append_jsonl(metrics_path, {"event": "demos", **demo_summary})
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
    try:
        while global_env_steps < config.total_env_steps and not stop_requested:
            if global_env_steps < config.random_env_steps:
                actions = torch.empty((config.num_envs, 7)).uniform_(-1.0, 1.0)
            else:
                with torch.inference_mode():
                    actions, _ = _policy_action(
                        model,
                        normalizer,
                        obs["states"],
                        device,
                        stochastic=True,
                    )
                    actions = actions.cpu()
            next_obs, rewards, terminated, truncated, infos = env.step(
                actions,
                auto_reset=False,
            )
            online.add(
                obs["states"],
                actions,
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
        final_validation = _evaluate(config, model, normalizer, device)
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
