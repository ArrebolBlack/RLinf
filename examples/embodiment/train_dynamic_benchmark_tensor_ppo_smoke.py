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

"""Run a bounded, zero-host-materialization PPO smoke on GPUENV0.

Unlike the older correctness smoke, this executable uses the production tensor
data plane.  Policy actions are consumed in place by Warp, environment outputs
remain CUDA views, transition storage is preallocated on the same GPU, and each
cohort resets only at the task's canonical fixed horizon.  Early terminal lanes
remain physically allocated but post-terminal transitions are masked on device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from rlinf.data.device_transition_buffer import DeviceFieldSpec, DeviceTransitionBuffer
from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import GpuNativeTensorBackendEnv


@dataclass(frozen=True)
class TensorPPOConfig:
    task: str
    export_dir: str
    expected_gpu_uuid: str
    seed: int
    num_envs: int
    cohorts: int
    warmup_steps: int
    hidden_size: int
    ppo_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    learning_rate: float
    image_size: int
    device_ordinal: int


class TensorActorCritic(nn.Module):
    """Small MLP matching the policy-size regime that leaves GPU headroom."""

    def __init__(self, observation_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, 7)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((7,), -0.5))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def distribution_and_value(self, observation: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        hidden = self.backbone(observation)
        return Normal(self.actor(hidden), self.log_std.exp()), self.critic(hidden).squeeze(-1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="p0_grasp")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--cohorts", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device-ordinal", type=int, default=0)
    return parser


def _config(args: argparse.Namespace) -> TensorPPOConfig:
    config = TensorPPOConfig(
        task=args.task,
        export_dir=str(args.export_dir),
        expected_gpu_uuid=args.expected_gpu_uuid,
        seed=args.seed,
        num_envs=args.num_envs,
        cohorts=args.cohorts,
        warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        device_ordinal=args.device_ordinal,
    )
    for name in ("num_envs", "cohorts", "hidden_size", "ppo_epochs", "minibatch_size"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if not 0.0 < config.clip_coef < 1.0:
        raise ValueError("clip_coef must be in (0, 1)")
    return config


def _sample(
    model: TensorActorCritic,
    observation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distribution, value = model.distribution_and_value(observation)
    raw_action = distribution.sample()
    action = torch.tanh(raw_action)
    log_prob = (
        distribution.log_prob(raw_action) - torch.log(1.0 - action.square() + 1e-6)
    ).sum(dim=-1)
    return action, raw_action, log_prob, value


def _masked_gae(
    *,
    reward: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    valid: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE entirely on device and exclude post-terminal cohort padding."""

    done = terminated | truncated
    advantage = torch.zeros_like(reward)
    accumulator = torch.zeros_like(reward[0])
    for step in range(reward.shape[0] - 1, -1, -1):
        row_valid = valid[step].to(reward.dtype)
        delta = (
            reward[step]
            + gamma * (~terminated[step]).to(reward.dtype) * next_value[step]
            - value[step]
        )
        accumulator = (
            delta
            + gamma
            * gae_lambda
            * (~done[step]).to(reward.dtype)
            * accumulator
        ) * row_valid
        advantage[step] = accumulator
    return advantage, advantage + value


def _ppo_update(
    *,
    config: TensorPPOConfig,
    model: TensorActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Any,
) -> tuple[dict[str, float], int]:
    horizon, num_envs = rollout.valid.shape
    flat_valid = rollout.valid.reshape(horizon * num_envs)
    extras = rollout.extras
    advantage, returns = _masked_gae(
        reward=rollout.reward,
        value=extras["value"],
        next_value=extras["next_value"],
        terminated=rollout.terminated,
        truncated=rollout.truncated,
        valid=rollout.valid,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    observation = rollout.observation.reshape(horizon * num_envs, -1)[flat_valid]
    action = rollout.action.reshape(horizon * num_envs, 7)[flat_valid]
    raw_action = extras["raw_action"].reshape(horizon * num_envs, 7)[flat_valid]
    old_log_prob = extras["log_prob"].reshape(horizon * num_envs)[flat_valid]
    selected_advantage = advantage.reshape(horizon * num_envs)[flat_valid]
    returns = returns.reshape(horizon * num_envs)[flat_valid]
    selected_advantage = (selected_advantage - selected_advantage.mean()) / (
        selected_advantage.std(unbiased=False) + 1e-8
    )
    sample_count = int(observation.shape[0])
    metric_rows: list[torch.Tensor] = []
    updates = 0
    for _epoch in range(config.ppo_epochs):
        permutation = torch.randperm(sample_count, device=observation.device)
        for start in range(0, sample_count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            if indices.numel() < 2:
                continue
            selected_observation = observation.index_select(0, indices)
            selected_action = action.index_select(0, indices)
            selected_raw_action = raw_action.index_select(0, indices)
            distribution, predicted_value = model.distribution_and_value(selected_observation)
            new_log_prob = (
                distribution.log_prob(selected_raw_action)
                - torch.log(1.0 - selected_action.square() + 1e-6)
            ).sum(dim=-1)
            log_ratio = new_log_prob - old_log_prob.index_select(0, indices)
            ratio = log_ratio.exp()
            batch_advantage = selected_advantage.index_select(0, indices)
            policy_loss = torch.maximum(
                -batch_advantage * ratio,
                -batch_advantage
                * ratio.clamp(1.0 - config.clip_coef, 1.0 + config.clip_coef),
            ).mean()
            value_loss = 0.5 * (
                predicted_value - returns.index_select(0, indices)
            ).square().mean()
            entropy = distribution.entropy().sum(dim=-1).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            metric_rows.append(
                torch.stack(
                    (
                        policy_loss.detach(),
                        value_loss.detach(),
                        entropy.detach(),
                        grad_norm.detach(),
                        ((ratio - 1.0) - log_ratio).mean().detach(),
                    )
                )
            )
            updates += 1
    if not metric_rows:
        raise RuntimeError("PPO smoke produced no optimizer updates")
    metrics = torch.stack(metric_rows).mean(dim=0)
    torch.cuda.synchronize(observation.device)
    names = ("policy_loss", "value_loss", "entropy", "grad_norm", "approx_kl")
    payload = {name: float(metrics[index]) for index, name in enumerate(names)}
    if not all(np.isfinite(value) for value in payload.values()):
        raise RuntimeError(f"non-finite PPO metrics: {payload}")
    return payload, updates


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    config = _config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("tensor PPO smoke requires CUDA; CPU fallback is forbidden")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "config.json", asdict(config))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    env = GpuNativeTensorBackendEnv(
        task_id=config.task,
        num_envs=config.num_envs,
        export_dir=config.export_dir,
        expected_gpu_uuid=config.expected_gpu_uuid,
        device_ordinal=config.device_ordinal,
        image_size=config.image_size,
    )
    try:
        initial = env.reset()
        observation = initial.observation
        observation_dim = int(observation.shape[1])
        model = TensorActorCritic(observation_dim, config.hidden_size).to(env.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        buffer = DeviceTransitionBuffer(
            capacity=env.cohort_horizon_steps,
            num_envs=config.num_envs,
            observation_shape=(observation_dim,),
            action_shape=(7,),
            device=env.device,
            observation_dtype=observation.dtype,
            action_dtype=torch.float32,
            reward_dtype=torch.float32,
            extra_fields={
                "raw_action": DeviceFieldSpec((7,), torch.float32),
                "log_prob": DeviceFieldSpec((), torch.float32),
                "value": DeviceFieldSpec((), torch.float32),
                "next_value": DeviceFieldSpec((), torch.float32),
            },
        )

        model.eval()
        for _step in range(config.warmup_steps):
            with torch.inference_mode():
                action, _raw, _log_prob, _value = _sample(model, observation)
                warm = env.step(action.contiguous())
            observation = warm.observation
        torch.cuda.synchronize(env.device)

        cohort_rows = []
        update_steps = 0
        total_allocated_steps = 0
        total_valid_steps = 0
        total_successes = 0
        wall_started = time.perf_counter()
        for cohort in range(config.cohorts):
            reset = env.reset()
            observation = reset.observation
            buffer.reset_cohort()
            model.eval()
            torch.cuda.synchronize(env.device)
            rollout_started = time.perf_counter()
            for _step in range(env.cohort_horizon_steps):
                with torch.inference_mode():
                    action, raw_action, log_prob, value = _sample(model, observation)
                    step = env.step(action.contiguous())
                    _next_distribution, next_value = model.distribution_and_value(step.observation)
                buffer.append(
                    observation=observation,
                    action=action,
                    reward=step.reward,
                    next_observation=step.observation,
                    terminated=step.terminated,
                    truncated=step.truncated,
                    success=step.success,
                    extras={
                        "raw_action": raw_action,
                        "log_prob": log_prob,
                        "value": value,
                        "next_value": next_value,
                    },
                )
                observation = step.observation
            torch.cuda.synchronize(env.device)
            rollout_seconds = time.perf_counter() - rollout_started
            rollout = buffer.view()

            model.train()
            update_started = time.perf_counter()
            metrics, updates = _ppo_update(
                config=config,
                model=model,
                optimizer=optimizer,
                rollout=rollout,
            )
            update_seconds = time.perf_counter() - update_started
            update_steps += updates
            allocated_steps = env.cohort_horizon_steps * config.num_envs
            valid_steps = int(rollout.valid.sum())
            successes = int((rollout.success & rollout.done & rollout.valid).sum())
            total_allocated_steps += allocated_steps
            total_valid_steps += valid_steps
            total_successes += successes
            cohort_rows.append(
                {
                    "cohort": cohort,
                    "allocated_steps": allocated_steps,
                    "valid_steps": valid_steps,
                    "terminal_successes": successes,
                    "rollout_seconds": rollout_seconds,
                    "update_seconds": update_seconds,
                    "allocated_env_steps_per_s": allocated_steps / rollout_seconds,
                    "valid_env_steps_per_s": valid_steps / rollout_seconds,
                    "optimizer_updates": updates,
                    **metrics,
                }
            )
        torch.cuda.synchronize(env.device)
        wall_seconds = time.perf_counter() - wall_started

        checkpoint_path = args.output / "checkpoint_latest.pt"
        torch.save(
            {
                "schema_version": "rlinf-gpuenv0-tensor-ppo-smoke-v0.1",
                "config": asdict(config),
                "cohort_horizon_steps": env.cohort_horizon_steps,
                "observation_dim": observation_dim,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "completed_cohorts": config.cohorts,
                "update_steps": update_steps,
            },
            checkpoint_path,
        )
        provenance = env.provenance
        report = {
            "schema_version": "rlinf-gpuenv0-tensor-ppo-smoke-report-v0.1",
            "status": "passed",
            "config": asdict(config),
            "backend": {
                "backend_id": provenance.backend_id,
                "api_version": env.api_version,
                "implementation_version": provenance.implementation_version,
                "device_name": provenance.device_name,
                "device_ordinal": provenance.device_ordinal,
                "physical_device_uuid": provenance.physical_device_uuid,
                "runtime_versions": dict(provenance.runtime_versions),
                "model_sha256": provenance.model_sha256,
                "config_sha256": provenance.config_sha256,
            },
            "data_plane": {
                "action_transport": "torch_cuda_tensor_direct_to_warp",
                "output_transport": "pointer_identical_warp_to_torch_views",
                "reset_policy": "canonical_fixed_horizon_full_cohort",
                "post_terminal_policy": "device_valid_mask",
                "hot_path_host_materializations": 0,
                "cohort_horizon_steps": env.cohort_horizon_steps,
            },
            "train": {
                "cohorts": cohort_rows,
                "total_allocated_steps": total_allocated_steps,
                "total_valid_steps": total_valid_steps,
                "terminal_successes": total_successes,
                "optimizer_updates": update_steps,
                "wall_seconds": wall_seconds,
                "allocated_env_steps_per_s": total_allocated_steps / wall_seconds,
                "valid_env_steps_per_s": total_valid_steps / wall_seconds,
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _file_sha256(checkpoint_path),
            },
        }
        _atomic_json(args.output / "report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
