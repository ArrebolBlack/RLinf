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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
    demo_rlinf_commit: str
    benchmark_commit: str
    seed: int
    demo_seed: int
    num_envs: int
    eval_num_envs: int
    demo_num_envs: int
    env_worker_threads: int
    eval_worker_threads: int
    env_worker_processes: int
    eval_worker_processes: int
    process_start_method: str
    sampler_learner_overlap: bool
    total_env_steps: int
    random_env_steps: int
    demo_episodes: int
    demo_max_attempts: int
    allow_failed_demos: bool
    demo_ratio: float
    bc_steps: int
    batch_size: int
    replay_capacity: int
    replay_storage: str
    non_blocking_copy: bool
    updates_per_vector_step: int
    timing_sample_interval: int
    q_heads: int
    q_target_subset: int
    gamma: float
    tau: float
    actor_lr: float
    actor_bc_weight: float
    residual_scale: float
    reward_safety_penalty: float
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

    def statistics(
        self,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.count < 2:
            return None
        variance = self.m2 / max(1, self.count - 1)
        mean = self.mean.clone()
        scale = torch.sqrt(torch.clamp(variance, min=self.epsilon**2))
        if self.mask_dim:
            mean[-self.mask_dim :] = 0.0
            scale[-self.mask_dim :] = 1.0
        return (
            mean.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
            scale.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        )

    @staticmethod
    def normalize_with_statistics(
        values: torch.Tensor,
        statistics: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        tensor = values.to(dtype=torch.float32)
        if statistics is None:
            return tensor
        mean, scale = statistics
        normalized = (tensor - mean) / scale
        return torch.clamp(normalized, -10.0, 10.0)

    def normalize(
        self,
        values: torch.Tensor,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32).to(
            device, non_blocking=non_blocking
        )
        return self.normalize_with_statistics(
            tensor,
            self.statistics(device, non_blocking=non_blocking),
        )

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
        if (
            int(state["dimension"]) != self.dimension
            or int(state["mask_dim"]) != self.mask_dim
        ):
            raise ValueError("normalizer shape does not match checkpoint")
        self.epsilon = float(state["epsilon"])
        self.count = int(state["count"])
        self.mean.copy_(torch.as_tensor(state["mean"], dtype=torch.float64))
        self.m2.copy_(torch.as_tensor(state["m2"], dtype=torch.float64))


class TransitionReplay:
    """Bounded replay with selectable storage and exact CPU sampling RNG."""

    FIELDS = (
        "states",
        "actions",
        "rewards",
        "next_states",
        "terminated",
        "truncated",
    )

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        seed: int,
        *,
        storage: str = "pageable_cpu",
        device: torch.device | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        if storage not in {"pageable_cpu", "pinned_cpu", "gpu"}:
            raise ValueError(f"unsupported replay storage {storage!r}")
        if storage == "gpu" and (device is None or device.type != "cuda"):
            raise ValueError("gpu replay requires a CUDA device")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.storage = storage
        self.storage_device = torch.device("cpu") if storage != "gpu" else device
        assert self.storage_device is not None
        allocation: dict[str, Any] = {"device": self.storage_device}
        if storage == "pinned_cpu":
            allocation["pin_memory"] = True
        self.cursor = 0
        self.size = 0
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.states = torch.empty(
            (capacity, state_dim), dtype=torch.float32, **allocation
        )
        self.actions = torch.empty((capacity, 7), dtype=torch.float32, **allocation)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, **allocation)
        self.next_states = torch.empty(
            (capacity, state_dim), dtype=torch.float32, **allocation
        )
        self.terminated = torch.empty((capacity, 1), dtype=torch.bool, **allocation)
        self.truncated = torch.empty((capacity, 1), dtype=torch.bool, **allocation)

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
            "states": torch.as_tensor(states, dtype=torch.float32).to(
                self.storage_device
            ),
            "actions": torch.as_tensor(actions, dtype=torch.float32).to(
                self.storage_device
            ),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32)
            .reshape(-1, 1)
            .to(self.storage_device),
            "next_states": torch.as_tensor(next_states, dtype=torch.float32).to(
                self.storage_device
            ),
            "terminated": torch.as_tensor(terminated, dtype=torch.bool)
            .reshape(-1, 1)
            .to(self.storage_device),
            "truncated": torch.as_tensor(truncated, dtype=torch.bool)
            .reshape(-1, 1)
            .to(self.storage_device),
        }
        if any(value.shape[0] != rows for value in payload.values()):
            raise ValueError("replay fields disagree on batch length")
        retained_rows = min(rows, self.capacity)
        skipped_rows = rows - retained_rows
        write_cursor = (self.cursor + skipped_rows) % self.capacity
        first_rows = min(retained_rows, self.capacity - write_cursor)
        for name, value in payload.items():
            retained = value[skipped_rows:]
            getattr(self, name)[write_cursor : write_cursor + first_rows].copy_(
                retained[:first_rows]
            )
            if first_rows < retained_rows:
                getattr(self, name)[: retained_rows - first_rows].copy_(
                    retained[first_rows:]
                )
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
        if self.storage == "gpu":
            indices = indices.to(self.storage_device)
        if self.storage != "pinned_cpu":
            return {
                name: getattr(self, name).index_select(0, indices)
                for name in self.FIELDS
            }
        sampled = {}
        for name in self.FIELDS:
            source = getattr(self, name)
            output = torch.empty(
                (int(batch_size), *source.shape[1:]),
                dtype=source.dtype,
                device="cpu",
                pin_memory=True,
            )
            torch.index_select(source, 0, indices, out=output)
            sampled[name] = output
        return sampled

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "cursor": self.cursor,
            "size": self.size,
            "generator_state": self.generator.get_state(),
            "data": {
                name: getattr(self, name)[: self.size].detach().cpu().clone()
                for name in self.FIELDS
            },
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
            value = torch.as_tensor(
                state["data"][name], dtype=getattr(self, name).dtype
            ).to(self.storage_device)
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
    parser.add_argument(
        "--demo-rlinf-commit",
        help="RLinf commit embedded in --demo-replay-in; defaults to --rlinf-commit.",
    )
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--demo-replay-in",
        type=Path,
        help="Validated demo replay cache from a matching producer identity.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--demo-seed",
        type=int,
        help=(
            "Seed embedded in --demo-replay-in; defaults to --seed. Set this "
            "explicitly when multiple learner seeds share one frozen demo cache."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--eval-num-envs", type=int, default=4)
    parser.add_argument(
        "--demo-num-envs",
        type=int,
        default=2,
        help="Frozen demo-collection vector width; independent of training num_envs.",
    )
    parser.add_argument("--env-worker-threads", type=int, default=1)
    parser.add_argument("--eval-worker-threads", type=int, default=1)
    parser.add_argument(
        "--env-worker-processes",
        type=int,
        default=0,
        help="Persistent subprocess shards for training environment reset/step (0 is serial).",
    )
    parser.add_argument(
        "--eval-worker-processes",
        type=int,
        default=0,
        help="Persistent subprocess shards for validation environment reset/step (0 is serial).",
    )
    parser.add_argument(
        "--process-start-method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
        help="Environment subprocess start method; spawn is CUDA-safe and portable.",
    )
    parser.add_argument(
        "--sampler-learner-overlap",
        action="store_true",
        help=(
            "Dispatch one process-backed environment step on a dedicated IPC thread "
            "while SAC updates consume the preceding replay snapshot."
        ),
    )
    parser.add_argument("--total-env-steps", type=int, default=200_000)
    parser.add_argument("--random-env-steps", type=int, default=2_000)
    parser.add_argument("--demo-episodes", type=int, default=32)
    parser.add_argument("--demo-max-attempts", type=int, default=320)
    parser.add_argument("--allow-failed-demos", action="store_true")
    parser.add_argument("--demo-ratio", type=float, default=0.5)
    parser.add_argument("--bc-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=250_000)
    parser.add_argument(
        "--replay-storage",
        choices=("pageable_cpu", "pinned_cpu", "gpu"),
        default="pageable_cpu",
    )
    parser.add_argument("--non-blocking-copy", action="store_true")
    parser.add_argument("--updates-per-vector-step", type=int, default=4)
    parser.add_argument(
        "--timing-sample-interval",
        type=int,
        default=20,
        help="Profile one learner update per N updates with CUDA events.",
    )
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
    parser.add_argument(
        "--reward-safety-penalty",
        type=float,
        default=-10.0,
        help="Terminal reward applied to safety failures during training.",
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
            "demo_seed",
            "demo_rlinf_commit",
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
        demo_rlinf_commit=args.demo_rlinf_commit or args.rlinf_commit,
        benchmark_commit=args.benchmark_commit,
        seed=args.seed,
        demo_seed=args.seed if args.demo_seed is None else args.demo_seed,
        num_envs=args.num_envs,
        eval_num_envs=args.eval_num_envs,
        demo_num_envs=args.demo_num_envs,
        env_worker_threads=args.env_worker_threads,
        eval_worker_threads=args.eval_worker_threads,
        env_worker_processes=args.env_worker_processes,
        eval_worker_processes=args.eval_worker_processes,
        process_start_method=args.process_start_method,
        sampler_learner_overlap=args.sampler_learner_overlap,
        total_env_steps=args.total_env_steps,
        random_env_steps=args.random_env_steps,
        demo_episodes=args.demo_episodes,
        demo_max_attempts=args.demo_max_attempts,
        allow_failed_demos=args.allow_failed_demos,
        demo_ratio=demo_ratio,
        bc_steps=args.bc_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        replay_storage=args.replay_storage,
        non_blocking_copy=args.non_blocking_copy,
        updates_per_vector_step=args.updates_per_vector_step,
        timing_sample_interval=args.timing_sample_interval,
        q_heads=args.q_heads,
        q_target_subset=args.q_target_subset,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        actor_bc_weight=args.actor_bc_weight,
        residual_scale=args.residual_scale,
        reward_safety_penalty=args.reward_safety_penalty,
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
    if min(config.num_envs, config.eval_num_envs, config.demo_num_envs) < 1:
        raise ValueError("environment counts must be positive")
    if config.env_worker_threads < 1 or config.eval_worker_threads < 1:
        raise ValueError("environment worker thread counts must be positive")
    if config.env_worker_processes < 0 or config.eval_worker_processes < 0:
        raise ValueError("environment worker process counts must be non-negative")
    if config.env_worker_processes and config.env_worker_threads != 1:
        raise ValueError("training process workers require env_worker_threads=1")
    if config.eval_worker_processes and config.eval_worker_threads != 1:
        raise ValueError("evaluation process workers require eval_worker_threads=1")
    if config.sampler_learner_overlap and not config.env_worker_processes:
        raise ValueError("sampler/learner overlap requires training process workers")
    for name, commit in (
        ("rlinf_commit", config.rlinf_commit),
        ("demo_rlinf_commit", config.demo_rlinf_commit),
        ("benchmark_commit", config.benchmark_commit),
    ):
        if len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError(f"{name} must be a full lowercase Git commit")
    if config.q_heads < 2 or not 1 <= config.q_target_subset <= config.q_heads:
        raise ValueError("Q ensemble/subset configuration is invalid")
    if not 0.0 <= config.demo_ratio <= 1.0:
        raise ValueError("demo_ratio must be in [0, 1]")
    if config.actor_bc_weight < 0.0:
        raise ValueError("actor_bc_weight must be non-negative")
    if not 0.0 < config.residual_scale <= 1.0:
        raise ValueError("residual_scale must be in (0, 1]")
    if (
        not math.isfinite(config.reward_safety_penalty)
        or config.reward_safety_penalty > 0.0
    ):
        raise ValueError("reward_safety_penalty must be finite and non-positive")
    if config.algorithm in {"bc", "rlpd", "residual_rlpd"} and config.demo_episodes < 1:
        raise ValueError("BC/RLPD requires at least one demonstration episode")
    if config.batch_size < 2 or config.replay_capacity < config.batch_size:
        raise ValueError("replay capacity must be at least the batch size")
    if config.timing_sample_interval < 1:
        raise ValueError("timing_sample_interval must be positive")
    if config.non_blocking_copy and config.replay_storage == "pageable_cpu":
        raise ValueError("non-blocking copies require pinned_cpu or gpu replay")
    return config


def _env_cfg(
    config: TrainConfig,
    *,
    split: str,
    seed: int,
    num_envs: int,
    worker_threads: int,
    worker_processes: int = 0,
    process_start_method: str = "spawn",
) -> dict[str, Any]:
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
        "worker_threads": worker_threads,
        "worker_processes": worker_processes,
        "process_start_method": process_start_method,
        "reward_safety_penalty": config.reward_safety_penalty,
    }


def _timed_call(call: Callable[[], Any]) -> tuple[Any, float, float]:
    """Run one sampler call and retain monotonic boundaries for overlap accounting."""

    started = time.perf_counter()
    value = call()
    finished = time.perf_counter()
    return value, started, finished


def _overlap_sample_and_update(
    executor: ThreadPoolExecutor,
    sample: Callable[[], Any],
    update: Callable[[], Any],
) -> tuple[Any, Any, dict[str, float]]:
    """Overlap process-backed sampling with learner work and expose exact wall timings."""

    future = executor.submit(_timed_call, sample)
    update_started = time.perf_counter()
    update_result = update()
    update_finished = time.perf_counter()
    wait_started = time.perf_counter()
    sample_result, sample_started, sample_finished = future.result()
    wait_finished = time.perf_counter()
    overlap_seconds = max(
        0.0,
        min(sample_finished, update_finished) - max(sample_started, update_started),
    )
    return (
        sample_result,
        update_result,
        {
            "environment_step_s": sample_finished - sample_started,
            "sampler_learner_overlap_s": overlap_seconds,
            "sampler_wait_after_update_s": wait_finished - wait_started,
            "overlapped_update_wall_s": update_finished - update_started,
        },
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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


class PhaseTimings:
    """Accumulate phase wall times without making logging cadence part of training."""

    def __init__(self) -> None:
        self.total_seconds: dict[str, float] = {}
        self.total_counts: dict[str, int] = {}
        self.interval_seconds: dict[str, float] = {}
        self.interval_counts: dict[str, int] = {}

    def add(self, name: str, seconds: float, *, count: int = 1) -> None:
        if seconds < 0.0 or count < 1:
            raise ValueError("phase timing observations must be non-negative")
        self.total_seconds[name] = self.total_seconds.get(name, 0.0) + float(seconds)
        self.total_counts[name] = self.total_counts.get(name, 0) + int(count)
        self.interval_seconds[name] = self.interval_seconds.get(name, 0.0) + float(
            seconds
        )
        self.interval_counts[name] = self.interval_counts.get(name, 0) + int(count)

    @staticmethod
    def _render(seconds: dict[str, float], counts: dict[str, int]) -> dict[str, Any]:
        return {
            name: {
                "total_s": seconds[name],
                "samples": counts[name],
                "mean_ms": 1000.0 * seconds[name] / max(1, counts[name]),
            }
            for name in sorted(seconds)
        }

    def snapshot(self, *, reset_interval: bool) -> dict[str, Any]:
        payload = {
            "cumulative": self._render(self.total_seconds, self.total_counts),
            "interval": self._render(self.interval_seconds, self.interval_counts),
        }
        if reset_interval:
            self.interval_seconds.clear()
            self.interval_counts.clear()
        return payload

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_seconds": dict(self.total_seconds),
            "total_counts": dict(self.total_counts),
            "interval_seconds": dict(self.interval_seconds),
            "interval_counts": dict(self.interval_counts),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total_seconds = {
            str(name): float(value) for name, value in state["total_seconds"].items()
        }
        self.total_counts = {
            str(name): int(value) for name, value in state["total_counts"].items()
        }
        self.interval_seconds = {
            str(name): float(value) for name, value in state["interval_seconds"].items()
        }
        self.interval_counts = {
            str(name): int(value) for name, value in state["interval_counts"].items()
        }


def _demo_replay_identity(
    config: TrainConfig,
    state_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "rlinf-dynamic-benchmark-demo-identity-v0.1",
        "task": config.task,
        "rlinf_commit": config.demo_rlinf_commit,
        "benchmark_commit": config.benchmark_commit,
        "seed": config.demo_seed,
        # This is the frozen demo producer width, not the learner vector width.
        # Keeping it separate lets utilization candidates consume the exact same
        # cache payload while scanning num_envs.
        "num_envs": config.demo_num_envs,
        "demo_episodes": config.demo_episodes,
        "demo_max_attempts": config.demo_max_attempts,
        "allow_failed_demos": config.allow_failed_demos,
        "reward_safety_penalty": config.reward_safety_penalty,
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
    cached_identity = payload.get("identity")
    if (
        isinstance(cached_identity, dict)
        and "reward_safety_penalty" not in cached_identity
    ):
        # Caches produced before reward-safety provenance was added used the
        # canonical -10 penalty but otherwise share the same v0.1 identity.
        # Upgrade only that exact legacy omission; non-canonical penalties and
        # every other identity mismatch remain fail-closed.
        cached_identity = dict(cached_identity, reward_safety_penalty=-10.0)
    if cached_identity != expected_identity:
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


def _load_demo_replay_cache_for_training(
    path: Path,
    expected_identity: dict[str, Any],
    replay: TransitionReplay,
    normalizer: RunningNormalizer,
    *,
    training_seed: int,
    demo_seed: int,
) -> tuple[dict[str, Any], str]:
    """Load frozen demo data without collapsing distinct learner RNG streams.

    The same-seed path preserves historical collect-vs-cache exact parity. A
    learner with a different seed keeps the RNG and replay-sampler states already
    established from that learner seed while consuming identical cached data.
    """

    learner_rng_state = _rng_state()
    learner_replay_generator_state = replay.generator.get_state().clone()
    summary, cache_sha256 = _load_demo_replay_cache(
        path,
        expected_identity,
        replay,
        normalizer,
    )
    if training_seed != demo_seed:
        _restore_rng(learner_rng_state)
        replay.generator.set_state(learner_replay_generator_state)
    return summary, cache_sha256


def _policy_action(
    model: MLPPolicy,
    normalizer: RunningNormalizer,
    states: torch.Tensor,
    device: torch.device,
    *,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = normalizer.normalize(states, device)
    return _policy_action_normalized(model, normalized, stochastic=stochastic)


def _policy_action_normalized(
    model: MLPPolicy,
    normalized_states: torch.Tensor,
    *,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_std = model._sample_actions(normalized_states)
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


def _reset_planner_teachers(
    task: str, env: Any, teachers: list[Any], indices: list[int]
) -> None:
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
        raise ValueError(
            "planner and residual actions must have matching [..., 7] shapes"
        )
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
    if online.storage != demos.storage:
        raise ValueError("online and demonstration replay storage must match")
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
    merged = {}
    for name in TransitionReplay.FIELDS:
        values = [chunk[name] for chunk in chunks]
        if len(values) == 1:
            merged[name] = values[0]
        elif online.storage == "pinned_cpu":
            output = torch.empty(
                (batch_size, *values[0].shape[1:]),
                dtype=values[0].dtype,
                device="cpu",
                pin_memory=True,
            )
            cursor = 0
            for value in values:
                output[cursor : cursor + value.shape[0]].copy_(value)
                cursor += value.shape[0]
            merged[name] = output
        else:
            merged[name] = torch.cat(values, dim=0)
    permutation = torch.randperm(batch_size)
    if online.storage == "gpu":
        permutation = permutation.to(online.storage_device)
    shuffled = {}
    for name, value in merged.items():
        if online.storage != "pinned_cpu":
            shuffled[name] = value.index_select(0, permutation)
            continue
        output = torch.empty(
            value.shape,
            dtype=value.dtype,
            device="cpu",
            pin_memory=True,
        )
        torch.index_select(value, 0, permutation, out=output)
        shuffled[name] = output
    return shuffled


def _collect_demos(
    config: TrainConfig,
    replay: TransitionReplay,
    normalizer: RunningNormalizer,
) -> dict[str, Any]:
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    count = min(config.demo_num_envs, config.demo_episodes)
    env = DynamicBenchmarkEnv(
        cfg=_env_cfg(
            config,
            split="train",
            seed=config.train_manifest_seed + 700_001,
            num_envs=count,
            worker_threads=1,
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
                action_values.append(
                    np.asarray(teacher.act(observation).values, dtype=np.float32)
                )
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
    evaluation_start = time.monotonic()
    environment_step_s = 0.0
    environment_reset_s = 0.0
    vector_steps = 0
    stepped_env_rows = 0
    count = min(config.eval_num_envs, config.eval_episodes)
    env = DynamicBenchmarkEnv(
        cfg=_env_cfg(
            config,
            split="validation",
            seed=config.validation_manifest_seed,
            num_envs=count,
            worker_threads=config.eval_worker_threads,
            worker_processes=config.eval_worker_processes,
            process_start_method=config.process_start_method,
        ),
        num_envs=count,
        seed_offset=0,
        total_num_processes=max(1, min(config.eval_worker_processes, count)),
        worker_info=None,
    )
    process_worker_pids = env.process_worker_pids
    records = []
    episode_returns = torch.zeros(count)
    episode_effort = torch.zeros(count)
    episode_steps = torch.zeros(count, dtype=torch.int64)
    active_episode_ids: list[int | None] = list(range(count))
    next_episode_id = count
    safety_failures = set(env.reward_schema["safety_failures"])
    teachers = (
        _make_planner_teachers(config.task, env)
        if config.algorithm == "residual_rlpd"
        else None
    )
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
            step_start = time.perf_counter()
            next_obs, rewards, terminated, truncated, infos = env.step(
                env_actions, auto_reset=False
            )
            environment_step_s += time.perf_counter() - step_start
            vector_steps += 1
            stepped_env_rows += sum(
                episode_id is not None for episode_id in active_episode_ids
            )
            episode_returns += rewards
            episode_effort += env_actions.square().sum(dim=-1)
            episode_steps += 1
            dones = torch.logical_or(terminated, truncated)
            done_indices = torch.arange(count)[dones].tolist()
            completed_indices = []
            for index in done_indices:
                episode_id = active_episode_ids[index]
                if episode_id is None:
                    continue
                reason = infos["termination_reason"][index]
                records.append(
                    {
                        "manifest_episode_index": episode_id,
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
                active_episode_ids[index] = None
                completed_indices.append(index)
            if len(records) >= config.eval_episodes:
                break
            reset_indices = []
            for index in completed_indices:
                if next_episode_id >= config.eval_episodes:
                    break
                active_episode_ids[index] = next_episode_id
                next_episode_id += 1
                reset_indices.append(index)
            if reset_indices:
                reset_start = time.perf_counter()
                next_obs, _ = env.reset(options={"env_idx": reset_indices})
                environment_reset_s += time.perf_counter() - reset_start
                if teachers is not None:
                    _reset_planner_teachers(config.task, env, teachers, reset_indices)
            obs = next_obs
    finally:
        env.close()
        model.train()
        _restore_rng(preserved_rng)
    records.sort(key=lambda record: record["manifest_episode_index"])
    wall_time_s = time.monotonic() - evaluation_start
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
        "wall_time_s": wall_time_s,
        "env_steps_per_second": stepped_env_rows / max(wall_time_s, 1e-6),
        "environment_step_mean_ms": 1000.0 * environment_step_s / max(1, vector_steps),
        "environment_reset_total_s": environment_reset_s,
        "worker_processes": len(process_worker_pids),
        "worker_pids": process_worker_pids,
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


def _parameter_groups(
    model: MLPPolicy,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    critic = list(model.q_head.parameters())
    critic_ids = {id(parameter) for parameter in critic}
    actor = [
        parameter for parameter in model.parameters() if id(parameter) not in critic_ids
    ]
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
        states = normalizer.normalize(
            batch["states"],
            device,
            non_blocking=config.non_blocking_copy,
        )
        target = batch["actions"].to(
            device,
            non_blocking=config.non_blocking_copy,
        )
        if config.algorithm == "residual_rlpd":
            target = torch.zeros_like(target)
        mean, _ = model._sample_actions(states)
        loss = F.mse_loss(torch.tanh(mean), target)
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in actor_optimizer.param_groups
                for parameter in group["params"]
            ],
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
    *,
    profile_timing: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    gpu_replay = online.storage == "gpu" or demos.storage == "gpu"
    sample_cuda_start = (
        torch.cuda.Event(enable_timing=True) if profile_timing and gpu_replay else None
    )
    sample_cuda_end = (
        torch.cuda.Event(enable_timing=True) if profile_timing and gpu_replay else None
    )
    if sample_cuda_start is not None:
        sample_cuda_start.record()
    sample_start = time.perf_counter()
    batch = _mixed_batch(online, demos, config.batch_size, config.demo_ratio)
    demo_batch = (
        demos.sample(config.batch_size)
        if config.actor_bc_weight > 0.0 and demos.size > 0
        else None
    )
    replay_sample_s = time.perf_counter() - sample_start
    if sample_cuda_end is not None:
        sample_cuda_end.record()
    transfer_start = torch.cuda.Event(enable_timing=True) if profile_timing else None
    transfer_end = torch.cuda.Event(enable_timing=True) if profile_timing else None
    critic_end = torch.cuda.Event(enable_timing=True) if profile_timing else None
    actor_end = torch.cuda.Event(enable_timing=True) if profile_timing else None
    update_end = torch.cuda.Event(enable_timing=True) if profile_timing else None
    if transfer_start is not None:
        transfer_start.record()
    device_batch = {
        name: value.to(device, non_blocking=config.non_blocking_copy)
        for name, value in batch.items()
    }
    device_demo_batch = (
        {
            name: value.to(device, non_blocking=config.non_blocking_copy)
            for name, value in demo_batch.items()
        }
        if demo_batch is not None
        else None
    )
    statistics = normalizer.statistics(
        device,
        non_blocking=config.non_blocking_copy,
    )
    if transfer_end is not None:
        transfer_end.record()
    states = normalizer.normalize_with_statistics(device_batch["states"], statistics)
    next_states = normalizer.normalize_with_statistics(
        device_batch["next_states"], statistics
    )
    actions = device_batch["actions"]
    rewards = device_batch["rewards"]
    dones = torch.logical_or(device_batch["terminated"], device_batch["truncated"]).to(
        dtype=torch.float32
    )
    with torch.no_grad():
        next_actions, next_log_prob = _policy_action_normalized(
            model,
            next_states,
            stochastic=True,
        )
        target_values = target_q(next_states, next_actions)
        subset = torch.randperm(config.q_heads, device=device)[: config.q_target_subset]
        target_min = (
            target_values.index_select(-1, subset).min(dim=-1, keepdim=True).values
        )
        alpha = log_alpha.exp()
        bootstrap = target_min - alpha * next_log_prob
        target = rewards + config.gamma * (1.0 - dones) * bootstrap
    predicted = model.q_head(states, actions)
    critic_loss = F.mse_loss(predicted, target.expand_as(predicted))
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.q_head.parameters(), 10.0)
    critic_optimizer.step()
    if critic_end is not None:
        critic_end.record()

    for parameter in model.q_head.parameters():
        parameter.requires_grad_(False)
    policy_actions, log_prob = _policy_action_normalized(
        model,
        states,
        stochastic=True,
    )
    policy_q = model.q_head(states, policy_actions).mean(dim=-1, keepdim=True)
    alpha = log_alpha.exp()
    actor_sac_loss = (alpha.detach() * log_prob - policy_q).mean()
    actor_bc_loss = torch.zeros((), dtype=torch.float32, device=device)
    if device_demo_batch is not None:
        demo_states = normalizer.normalize_with_statistics(
            device_demo_batch["states"], statistics
        )
        demo_actions = device_demo_batch["actions"]
        if config.algorithm == "residual_rlpd":
            demo_actions = torch.zeros_like(demo_actions)
        demo_mean, _ = model._sample_actions(demo_states)
        actor_bc_loss = F.mse_loss(torch.tanh(demo_mean), demo_actions)
    actor_loss = actor_sac_loss + config.actor_bc_weight * actor_bc_loss
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [
            parameter
            for group in actor_optimizer.param_groups
            for parameter in group["params"]
        ],
        10.0,
    )
    actor_optimizer.step()
    for parameter in model.q_head.parameters():
        parameter.requires_grad_(True)
    if actor_end is not None:
        actor_end.record()

    alpha_loss = -(log_alpha * (log_prob + config.target_entropy).detach()).mean()
    alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_optimizer.step()
    with torch.no_grad():
        for online_parameter, target_parameter in zip(
            model.q_head.parameters(), target_q.parameters(), strict=True
        ):
            target_parameter.lerp_(online_parameter, config.tau)
    timing = {} if gpu_replay else {"replay_sample_s": replay_sample_s}
    if update_end is not None:
        assert (
            transfer_start is not None
            and transfer_end is not None
            and critic_end is not None
            and actor_end is not None
        )
        update_end.record()
        update_end.synchronize()
        timing.update(
            h2d_s=transfer_start.elapsed_time(transfer_end) / 1000.0,
            critic_update_s=transfer_end.elapsed_time(critic_end) / 1000.0,
            actor_update_s=critic_end.elapsed_time(actor_end) / 1000.0,
            alpha_target_update_s=actor_end.elapsed_time(update_end) / 1000.0,
            forward_backward_update_s=transfer_end.elapsed_time(update_end) / 1000.0,
        )
        if sample_cuda_start is not None and sample_cuda_end is not None:
            timing["replay_sample_s"] = (
                sample_cuda_start.elapsed_time(sample_cuda_end) / 1000.0
            )
    metrics = {
        "critic_loss": float(critic_loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "actor_sac_loss": float(actor_sac_loss.detach()),
        "actor_bc_loss": float(actor_bc_loss.detach()),
        "alpha_loss": float(alpha_loss.detach()),
        "alpha": float(log_alpha.exp().detach()),
        "q_data": float(predicted.detach().mean()),
        "q_target": float(target.detach().mean()),
    }
    return metrics, timing


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
        raise ValueError(
            "resume already contains demonstrations; do not pass demo replay cache"
        )
    if (
        args.resume is None
        and args.demo_replay_in is None
        and config.demo_rlinf_commit != config.rlinf_commit
    ):
        raise ValueError(
            "demo_rlinf_commit may differ from rlinf_commit only with --demo-replay-in"
        )
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
    config_sha256 = _file_sha256(config_path)
    sampler_contract = {
        "mode": "overlap" if config.sampler_learner_overlap else "synchronous",
        "max_inflight_vector_steps": 1 if config.sampler_learner_overlap else 0,
        "policy_lag_vector_steps": 0,
        "replay_snapshot_lag_vector_steps": (
            1 if config.sampler_learner_overlap else 0
        ),
        "dropped_vector_steps": 0,
    }
    _append_jsonl(
        metrics_path,
        {
            "event": "run_start",
            "config_sha256": config_sha256,
            "pid": os.getpid(),
            "replay_storage": config.replay_storage,
            "non_blocking_copy": config.non_blocking_copy,
            "env_worker_threads": config.env_worker_threads,
            "eval_worker_threads": config.eval_worker_threads,
            "env_worker_processes": config.env_worker_processes,
            "eval_worker_processes": config.eval_worker_processes,
            "process_start_method": config.process_start_method,
            "sampler_learner_overlap": config.sampler_learner_overlap,
            "sampler_contract": sampler_contract,
        },
    )

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
            worker_threads=config.env_worker_threads,
            worker_processes=config.env_worker_processes,
            process_start_method=config.process_start_method,
        ),
        num_envs=config.num_envs,
        seed_offset=0,
        total_num_processes=max(1, min(config.env_worker_processes, config.num_envs)),
        worker_info=None,
    )
    _append_jsonl(
        metrics_path,
        {
            "event": "environment_workers",
            "env_worker_processes": config.env_worker_processes,
            "worker_pids": env.process_worker_pids,
        },
    )
    obs = env._last_obs
    if obs is None:
        raise RuntimeError("training environment did not initialize its state")
    state_schema = env.state_schema
    state_dim = int(state_schema["state_dim"])
    normalizer = RunningNormalizer(state_dim, int(state_schema["mask_dim"]))
    demos = TransitionReplay(
        config.replay_capacity,
        state_dim,
        config.seed + 11,
        storage=config.replay_storage,
        device=device,
    )
    online = TransitionReplay(
        config.replay_capacity,
        state_dim,
        config.seed + 17,
        storage=config.replay_storage,
        device=device,
    )
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
    phase_timings = PhaseTimings()
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
            "phase_timings": phase_timings.state_dict(),
            "rng": _rng_state(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    if args.resume is not None:
        restored = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            restored["config"] != asdict(config)
            or restored["state_schema"] != state_schema
        ):
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
        if restored.get("phase_timings") is not None:
            phase_timings.load_state_dict(restored["phase_timings"])
        if config.algorithm == "residual_rlpd":
            if training_teachers is None or len(training_teachers) != config.num_envs:
                raise ValueError("residual checkpoint is missing planner teacher state")
        elif training_teachers is not None:
            raise ValueError(
                "non-residual checkpoint unexpectedly contains planner state"
            )
        _restore_rng(restored["rng"])
        _append_jsonl(metrics_path, {"event": "resume", "env_steps": global_env_steps})
    else:
        demo_summary = None
        if config.algorithm in {"bc", "rlpd", "residual_rlpd"}:
            demo_identity = _demo_replay_identity(config, state_schema)
            if args.demo_replay_in is None:
                if config.demo_seed != config.seed:
                    raise ValueError(
                        "demo_seed may differ from seed only with --demo-replay-in"
                    )
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
                demo_summary, demo_cache_sha256 = _load_demo_replay_cache_for_training(
                    args.demo_replay_in,
                    demo_identity,
                    demos,
                    normalizer,
                    training_seed=config.seed,
                    demo_seed=config.demo_seed,
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
                "config_sha256": config_sha256,
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
    training_start_time = time.monotonic()
    training_start_env_steps = global_env_steps
    training_start_update_steps = update_steps
    last_log_time = training_start_time
    last_log_env_steps = global_env_steps
    last_log_update_steps = update_steps
    sampler_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="dynamic-benchmark-sampler")
        if config.sampler_learner_overlap
        else None
    )

    def learner_updates() -> dict[str, Any] | None:
        nonlocal update_steps
        update_metrics = None
        if online.size < max(config.batch_size, config.random_env_steps):
            return update_metrics
        for _ in range(config.updates_per_vector_step):
            update_metrics, update_timing = _sac_update(
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
                profile_timing=(
                    update_steps % config.timing_sample_interval == 0
                ),
            )
            for name, seconds in update_timing.items():
                phase_timings.add(name, seconds)
            update_steps += 1
        return update_metrics

    try:
        while global_env_steps < config.total_env_steps and not stop_requested:
            vector_step_start = time.perf_counter()
            action_start = time.perf_counter()
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
            phase_timings.add(
                "environment_action",
                time.perf_counter() - action_start,
            )
            updates_overlapped = False
            update_metrics = None
            if (
                sampler_executor is not None
                and online.size >= max(config.batch_size, config.random_env_steps)
            ):
                sample_result, update_metrics, overlap_timing = (
                    _overlap_sample_and_update(
                        sampler_executor,
                        lambda: env.step(env_actions, auto_reset=False),
                        learner_updates,
                    )
                )
                next_obs, rewards, terminated, truncated, infos = sample_result
                phase_timings.add(
                    "environment_step", overlap_timing.pop("environment_step_s")
                )
                for name, seconds in overlap_timing.items():
                    phase_timings.add(name, seconds)
                updates_overlapped = True
            else:
                environment_step_start = time.perf_counter()
                next_obs, rewards, terminated, truncated, infos = env.step(
                    env_actions, auto_reset=False
                )
                phase_timings.add(
                    "environment_step",
                    time.perf_counter() - environment_step_start,
                )
            replay_add_start = time.perf_counter()
            online.add(
                obs["states"],
                policy_actions,
                rewards,
                next_obs["states"],
                terminated,
                truncated,
            )
            phase_timings.add(
                "replay_add",
                time.perf_counter() - replay_add_start,
            )
            normalizer_start = time.perf_counter()
            normalizer.update(next_obs["states"])
            phase_timings.add(
                "normalizer_update",
                time.perf_counter() - normalizer_start,
            )
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
                reset_start = time.perf_counter()
                next_obs, _ = env.reset(options={"env_idx": done_indices})
                if training_teachers is not None:
                    _reset_planner_teachers(
                        config.task,
                        env,
                        training_teachers,
                        done_indices,
                    )
                normalizer.update(
                    next_obs["states"].index_select(0, torch.tensor(done_indices))
                )
                phase_timings.add(
                    "environment_reset",
                    time.perf_counter() - reset_start,
                    count=len(done_indices),
                )
            obs = next_obs
            global_env_steps += config.num_envs

            if not updates_overlapped:
                update_metrics = learner_updates()

            phase_timings.add(
                "vector_step_wall",
                time.perf_counter() - vector_step_start,
            )

            if global_env_steps >= next_log:
                now = time.monotonic()
                elapsed = max(now - start_time, 1e-6)
                training_elapsed = max(now - training_start_time, 1e-6)
                interval_elapsed = max(now - last_log_time, 1e-6)
                payload: dict[str, Any] = {
                    "event": "train",
                    "env_steps": global_env_steps,
                    "update_steps": update_steps,
                    "online_replay_size": online.size,
                    "demo_replay_size": demos.size,
                    "env_steps_per_second": global_env_steps / elapsed,
                    "training_env_steps_per_second": (
                        global_env_steps - training_start_env_steps
                    )
                    / training_elapsed,
                    "update_steps_per_second": (
                        update_steps - training_start_update_steps
                    )
                    / training_elapsed,
                    "interval_env_steps_per_second": (
                        global_env_steps - last_log_env_steps
                    )
                    / interval_elapsed,
                    "interval_update_steps_per_second": (
                        update_steps - last_log_update_steps
                    )
                    / interval_elapsed,
                    "wall_time_s": elapsed,
                    "training_wall_time_s": training_elapsed,
                    "phase_timings": phase_timings.snapshot(reset_interval=True),
                }
                if recent_episodes:
                    payload.update(
                        recent_episode_count=len(recent_episodes),
                        recent_success_rate=float(
                            np.mean([item["success"] for item in recent_episodes])
                        ),
                        recent_mean_completion=float(
                            np.mean(
                                [
                                    item["trajectory_completion"]
                                    for item in recent_episodes
                                ]
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
                last_log_time = now
                last_log_env_steps = global_env_steps
                last_log_update_steps = update_steps
                next_log += config.log_interval

            if global_env_steps >= next_eval:
                validation = _evaluate(config, model, normalizer, device)
                last_validation = validation
                last_validation_env_steps = global_env_steps
                score = _score(validation)
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "validation",
                        "env_steps": global_env_steps,
                        **validation,
                    },
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

        training_end_time = time.monotonic()
        checkpoint(args.output / "checkpoint_latest.pt")
        if (
            last_validation is not None
            and last_validation_env_steps == global_env_steps
        ):
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
            "training_wall_time_s": training_end_time - training_start_time,
            "training_env_steps_per_second": (
                global_env_steps - training_start_env_steps
            )
            / max(training_end_time - training_start_time, 1e-6),
            "update_steps_per_second": (update_steps - training_start_update_steps)
            / max(training_end_time - training_start_time, 1e-6),
            "phase_timings": phase_timings.snapshot(reset_interval=False),
            "sampler_contract": sampler_contract,
            "config_sha256": config_sha256,
        }
        rendered = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        summary["payload_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
        _atomic_json(args.output / "summary.json", summary)
    finally:
        if sampler_executor is not None:
            sampler_executor.shutdown(wait=True, cancel_futures=True)
        env.close()


if __name__ == "__main__":
    main()
