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

"""Bounded PPO training smoke on the GPU-native Dynamic Benchmark backend.

This is the D5 wiring acceptance smoke: a real RLinf training loop (policy
rollout + PPO update on a small single-GPU batch) driven entirely through the
SE(3)-WAM GPU-native backend seam (``make_gpu_native_env`` /
``mjwarp_gpu_v1``).  It asserts:

- the training environment owns a GPU-native backend and zero CPU MuJoCo envs;
- every train step completes without NaN/Inf losses or gradients;
- checkpoints save and reload (in-process reload plus a cross-process
  ``--resume`` run);
- the report carries backend provenance as the no-CPU-fallback evidence.

The conventional CPU evaluation harness is intentionally out of scope here;
only the training loop must be GPU-native for this wiring gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Normal


@dataclass(frozen=True)
class SmokeConfig:
    task: str
    export_dir: str
    seed: int
    num_envs: int
    rollout_steps: int
    ppo_epochs: int
    minibatch_size: int
    total_env_steps: int
    checkpoint_interval: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    learning_rate: float
    manifest_size: int
    image_size: int
    device_ordinal: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="p0_grasp")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--total-env-steps", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--manifest-size", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device-ordinal", type=int, default=0)
    return parser


def _config(args: argparse.Namespace) -> SmokeConfig:
    config = SmokeConfig(
        task=args.task,
        export_dir=str(args.export_dir),
        seed=args.seed,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        total_env_steps=args.total_env_steps,
        checkpoint_interval=args.checkpoint_interval,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        manifest_size=args.manifest_size,
        image_size=args.image_size,
        device_ordinal=args.device_ordinal,
    )
    if config.num_envs < 1 or config.rollout_steps < 1 or config.ppo_epochs < 1:
        raise ValueError("smoke sizes must be positive")
    if (
        config.total_env_steps < config.num_envs
        or config.total_env_steps % config.num_envs
    ):
        raise ValueError("total_env_steps must be positive and divisible by num_envs")
    if not 0.0 <= config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    return config


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
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.count < 2:
            return None
        variance = self.m2 / (self.count - 1)
        scale = torch.sqrt(variance + self.epsilon).to(device)
        return self.mean.to(device), scale

    def normalize(
        self,
        values: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32).to(device)
        statistics = self.statistics(device)
        if statistics is None:
            return tensor
        mean, scale = statistics
        return torch.clamp((tensor - mean) / scale, -10.0, 10.0)

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


def _env_cfg(config: SmokeConfig) -> dict[str, Any]:
    return {
        "task_id": config.task,
        "split": "train",
        "manifest_seed": 20261050 + config.seed * 1_000_003,
        "manifest_size": max(config.manifest_size, 2 * config.num_envs),
        "image_size": config.image_size,
        "camera_observations": False,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
        "worker_threads": 1,
        "worker_processes": 0,
        "process_start_method": "spawn",
        "process_residual_planner": False,
        "reward_safety_penalty": -10.0,
        "features": {},
        "reward_components": {},
        "reward_lift_shaping_weight": 0.0,
        "reward_orientation_shaping_weight": 0.0,
        "state_derived_features": [],
        "gpu_native": True,
        "gpu_native_export_dir": config.export_dir,
        "gpu_native_device_ordinal": config.device_ordinal,
    }


def _policy_step(
    model: Any,
    normalized_states: torch.Tensor,
    *,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, log_std = model._sample_actions(normalized_states)
    distribution = Normal(mean, torch.exp(log_std))
    raw_actions = distribution.sample() if stochastic else mean
    actions = torch.tanh(raw_actions)
    log_prob = distribution.log_prob(raw_actions) - torch.log(
        1.0 - actions.square() + 1e-6
    )
    values = model.value_head(normalized_states)
    return actions, raw_actions, log_prob.sum(dim=-1), values.squeeze(-1)


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    deltas = rewards + gamma * (~terminated).to(rewards.dtype) * next_values - values
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(rewards[0])
    dones = torch.logical_or(terminated, truncated)
    for step in range(rewards.shape[0] - 1, -1, -1):
        accumulator = (
            deltas[step]
            + gamma * gae_lambda * (~dones[step]).to(rewards.dtype) * accumulator
        )
        advantages[step] = accumulator
    return advantages, advantages + values


def _ppo_update(
    config: SmokeConfig,
    model: Any,
    optimizer: Any,
    rollout: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    states = rollout["states"]
    actions = rollout["actions"]
    raw_actions = rollout["raw_actions"]
    old_log_prob = rollout["log_prob"]
    returns = rollout["returns"].to(device)
    advantages = rollout["advantages"].to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    totals: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "grad_norm": [],
    }
    updates = 0
    for _ in range(config.ppo_epochs):
        indices = torch.randperm(states.shape[0], device="cpu")
        for start in range(0, states.shape[0], config.minibatch_size):
            selected = indices[start : start + config.minibatch_size]
            if selected.numel() < 2:
                continue
            batch = states.index_select(0, selected).to(device)
            batch_actions = actions.index_select(0, selected).to(device)
            batch_raw = raw_actions.index_select(0, selected).to(device)
            mean, log_std = model._sample_actions(batch)
            distribution = Normal(mean, torch.exp(log_std))
            new_log_prob = (
                distribution.log_prob(batch_raw)
                - torch.log(1.0 - batch_actions.square() + 1e-6)
            ).sum(dim=-1)
            log_ratio = new_log_prob - old_log_prob.index_select(0, selected).to(device)
            ratio = log_ratio.exp()
            unclipped = -advantages.index_select(0, selected) * ratio
            clipped = -advantages.index_select(0, selected) * ratio.clamp(
                1.0 - config.clip_coef, 1.0 + config.clip_coef
            )
            policy_loss = torch.maximum(unclipped, clipped).mean()
            predicted_values = model.value_head(batch).squeeze(-1)
            value_loss = (
                0.5
                * (predicted_values - returns.index_select(0, selected)).square().mean()
            )
            entropy = distribution.entropy().sum(dim=-1).mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            optimizer.step()
            updates += 1
            totals["policy_loss"].append(float(policy_loss.detach()))
            totals["value_loss"].append(float(value_loss.detach()))
            totals["entropy"].append(float(entropy.detach()))
            totals["approx_kl"].append(
                float(((ratio - 1.0) - log_ratio).mean().detach())
            )
            totals["clip_fraction"].append(
                float(
                    ((ratio - 1.0).abs() > config.clip_coef)
                    .to(torch.float32)
                    .mean()
                    .detach()
                )
            )
            totals["grad_norm"].append(float(grad_norm.detach()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    args = _parser().parse_args()
    config = _config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint_latest.pt"

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device(f"cuda:{config.device_ordinal}")

    env = DynamicBenchmarkEnv(
        cfg=_env_cfg(config),
        num_envs=config.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        backend = env._gpu_backend
        if backend is None:
            raise RuntimeError("GPU-native training env did not build its backend")
        provenance = backend.provenance
        provenance_payload = {
            "backend_id": provenance.backend_id,
            "device_platform": provenance.device_platform,
            "device_name": provenance.device_name,
            "device_ordinal": provenance.device_ordinal,
            "runtime_versions": dict(provenance.runtime_versions),
        }
        if provenance.device_platform not in {"gpu", "cuda"}:
            raise RuntimeError("GPU-native backend reported a non-GPU device platform")
        if len(env.envs) != 0:
            raise RuntimeError("GPU-native training env owns CPU MuJoCo environments")

        state_schema = env.state_schema
        state_dim = int(state_schema["state_dim"])
        normalizer = RunningNormalizer(state_dim, int(state_schema["mask_dim"]))
        model = MLPPolicy(
            obs_dim=state_dim,
            action_dim=7,
            num_action_chunks=1,
            add_value_head=True,
            add_q_head=False,
            num_q_heads=2,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        global_env_steps = 0
        update_steps = 0
        next_checkpoint = config.checkpoint_interval
        episode_returns = torch.zeros(config.num_envs)
        episode_steps = torch.zeros(config.num_envs, dtype=torch.int64)
        recent_episodes: list[dict[str, Any]] = []
        checkpoint_sha = None

        def checkpoint() -> None:
            nonlocal checkpoint_sha
            state = {
                "schema_version": "rlinf-dynamic-benchmark-ppo-gpu-smoke-v0.1",
                "config": asdict(config),
                "state_schema": state_schema,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "normalizer": normalizer.state_dict(),
                "env": env.checkpoint_state(),
                "global_env_steps": global_env_steps,
                "update_steps": update_steps,
            }
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint_path)
            checkpoint_sha = _file_sha256(checkpoint_path)

        if args.resume is not None:
            restored = torch.load(args.resume, map_location="cpu", weights_only=False)
            if (
                restored["schema_version"]
                != "rlinf-dynamic-benchmark-ppo-gpu-smoke-v0.1"
            ):
                raise ValueError("resume checkpoint schema mismatch")
            if restored["state_schema"] != state_schema:
                raise ValueError("resume state schema does not match current run")
            model.load_state_dict(restored["model"])
            optimizer.load_state_dict(restored["optimizer"])
            normalizer.load_state_dict(restored["normalizer"])
            env.load_checkpoint_state(restored["env"])
            global_env_steps = int(restored["global_env_steps"])
            update_steps = int(restored["update_steps"])
            next_checkpoint = global_env_steps + config.checkpoint_interval
            episode_returns.copy_(
                restored.get("episode_returns", torch.zeros(config.num_envs))
            )
            episode_steps.copy_(
                restored.get(
                    "episode_steps", torch.zeros(config.num_envs, dtype=torch.int64)
                )
            )
            recent_episodes = list(restored.get("recent_episodes", []))

        obs = env._last_obs
        if obs is None:
            raise RuntimeError("GPU-native training env did not initialize its state")
        normalizer.update(obs["states"])

        loss_rows: list[dict[str, float]] = []
        rollout_started = time.monotonic()
        while global_env_steps < config.total_env_steps:
            remaining_vector_steps = (
                config.total_env_steps - global_env_steps
            ) // config.num_envs
            horizon = min(config.rollout_steps, remaining_vector_steps)
            storage: dict[str, list[torch.Tensor]] = {
                "states": [],
                "raw_actions": [],
                "actions": [],
                "log_prob": [],
                "values": [],
                "rewards": [],
                "next_values": [],
                "terminated": [],
                "truncated": [],
            }
            model.eval()
            for _ in range(horizon):
                normalized_states = normalizer.normalize(obs["states"], device)
                with torch.inference_mode():
                    actions, raw_actions, log_prob, values = _policy_step(
                        model, normalized_states, stochastic=True
                    )
                next_obs, rewards, terminated, truncated, infos = env.step(
                    actions.cpu(), auto_reset=False
                )
                normalized_next_states = normalizer.normalize(
                    next_obs["states"], device
                )
                with torch.inference_mode():
                    next_values = model.value_head(normalized_next_states).squeeze(-1)
                storage["states"].append(normalized_states.cpu())
                storage["raw_actions"].append(raw_actions.cpu())
                storage["actions"].append(actions.cpu())
                storage["log_prob"].append(log_prob.cpu())
                storage["values"].append(values.cpu())
                storage["rewards"].append(rewards.clone())
                storage["next_values"].append(next_values.cpu())
                storage["terminated"].append(terminated.clone())
                storage["truncated"].append(truncated.clone())
                normalizer.update(next_obs["states"])
                episode_returns += rewards
                episode_steps += 1
                dones = torch.logical_or(terminated, truncated)
                done_indices = torch.arange(config.num_envs)[dones].tolist()
                for index in done_indices:
                    recent_episodes.append(
                        {
                            "success": bool(infos["success"][index]),
                            "termination_reason": infos["termination_reason"][index],
                            "trajectory_completion": float(
                                infos["trajectory_completion"][index]
                            ),
                            "return": float(episode_returns[index]),
                            "duration_steps": int(episode_steps[index]),
                        }
                    )
                    episode_returns[index] = 0.0
                    episode_steps[index] = 0
                if done_indices:
                    next_obs, _ = env.reset(options={"env_idx": done_indices})
                    normalizer.update(
                        next_obs["states"].index_select(0, torch.tensor(done_indices))
                    )
                obs = next_obs
                global_env_steps += config.num_envs
            rollout = {name: torch.stack(values) for name, values in storage.items()}
            rollout["advantages"], rollout["returns"] = _compute_gae(
                rollout["rewards"],
                rollout["values"],
                rollout["next_values"],
                rollout["terminated"],
                rollout["truncated"],
                config.gamma,
                config.gae_lambda,
            )
            model.train()
            metrics = _ppo_update(config, model, optimizer, rollout, device)
            update_steps += 1
            finite = {
                name: bool(np.isfinite(metrics[name]))
                for name in ("policy_loss", "value_loss", "entropy", "grad_norm")
            }
            if not all(finite.values()):
                raise RuntimeError(
                    f"non-finite train metrics at update {update_steps}: {finite}"
                )
            loss_rows.append(
                {
                    "update_steps": update_steps,
                    "global_env_steps": global_env_steps,
                    **metrics,
                }
            )
            if global_env_steps >= next_checkpoint:
                checkpoint()
                next_checkpoint += config.checkpoint_interval
        rollout_seconds = time.monotonic() - rollout_started
        checkpoint()

        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reload_checks = {
            "schema_match": reloaded["schema_version"]
            == "rlinf-dynamic-benchmark-ppo-gpu-smoke-v0.1",
            "model_keys_match": set(reloaded["model"]) == set(model.state_dict()),
            "env_state_present": "env" in reloaded
            and isinstance(reloaded["env"], dict),
        }
        reloaded_env = DynamicBenchmarkEnv(
            cfg=_env_cfg(config),
            num_envs=config.num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        try:
            reloaded_env.load_checkpoint_state(reloaded["env"])
            reloaded_env_obs = reloaded_env._last_obs
            reload_checks["env_reload_ok"] = reloaded_env_obs is not None
            if (
                reloaded_env_obs is not None
                and reloaded_env_obs["states"].shape != obs["states"].shape
            ):
                reload_checks["env_reload_ok"] = False
        finally:
            reloaded_env.close()

        report = {
            "schema_version": "rlinf-gpu-backend-ppo-smoke-v0.1",
            "status": "passed" if all(reload_checks.values()) else "failed",
            "task": config.task,
            "export_dir": config.export_dir,
            "resumed_from": str(args.resume) if args.resume is not None else None,
            "train": {
                "global_env_steps": global_env_steps,
                "update_steps": update_steps,
                "control_steps": global_env_steps,
                "physics_steps": global_env_steps * 25,
                "loss_rows": loss_rows,
                "recent_episodes": recent_episodes,
                "recent_success_rate": float(
                    np.mean([item["success"] for item in recent_episodes])
                )
                if recent_episodes
                else None,
                "throughput": {
                    "env_steps_per_s": global_env_steps / rollout_seconds,
                    "control_steps_per_s": global_env_steps / rollout_seconds,
                    "physics_steps_per_s": global_env_steps * 25 / rollout_seconds,
                },
            },
            "backend": provenance_payload,
            "no_cpu_fallback": {
                "gpu_backend_used": True,
                "cpu_mujoco_envs_owned": len(env.envs),
                "device_platform_cuda": provenance.device_platform in {"gpu", "cuda"},
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "reload_checks": reload_checks,
            },
            "limits": {
                "eval_harness": "cpu_harness_out_of_scope",
                "env_snapshot": "device_resident_not_serialized",
            },
        }
        _atomic_json(args.output / "report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not all(reload_checks.values()) or report["status"] != "passed":
            return 1
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
