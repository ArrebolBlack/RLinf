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

"""Train a resumable privileged-state PPO expert on Dynamic Benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from torch.distributions import Normal

from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    _append_jsonl,
    _atomic_json,
    _env_cfg,
    _evaluate,
    _restore_rng,
    _rng_state,
    _save_policy,
    _score,
)

if TYPE_CHECKING:
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


@dataclass(frozen=True)
class PPOConfig:
    task: str
    algorithm: str
    rlinf_commit: str
    benchmark_commit: str
    seed: int
    num_envs: int
    eval_num_envs: int
    total_env_steps: int
    rollout_steps: int
    ppo_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    learning_rate: float
    eval_interval: int
    eval_episodes: int
    checkpoint_interval: int
    log_interval: int
    train_manifest_seed: int
    validation_manifest_seed: int
    manifest_size: int
    image_size: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--eval-num-envs", type=int, default=2)
    parser.add_argument("--total-env-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=250)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
    arguments = sys.argv[1:] if argv is None else argv
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(arguments)
    parser = _parser()
    if pre_args.config is not None:
        import yaml

        payload = yaml.safe_load(pre_args.config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Dynamic Benchmark PPO config must be a YAML mapping")
        explicit_only = {
            "benchmark_commit",
            "config",
            "output",
            "resume",
            "rlinf_commit",
        }
        actions = {action.dest: action for action in parser._actions}
        unknown = sorted(set(payload) - set(actions))
        forbidden = sorted(set(payload) & explicit_only)
        if unknown:
            raise ValueError(f"unknown Dynamic Benchmark PPO config keys: {unknown}")
        if forbidden:
            raise ValueError(f"run-specific keys must stay on the CLI: {forbidden}")
        parser.set_defaults(**payload)
        for name in payload:
            actions[name].required = False
    return parser.parse_args(arguments)


def _config(args: argparse.Namespace) -> PPOConfig:
    config = PPOConfig(
        task=args.task,
        algorithm="ppo",
        rlinf_commit=args.rlinf_commit,
        benchmark_commit=args.benchmark_commit,
        seed=args.seed,
        num_envs=args.num_envs,
        eval_num_envs=args.eval_num_envs,
        total_env_steps=args.total_env_steps,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        train_manifest_seed=args.train_manifest_seed,
        validation_manifest_seed=args.validation_manifest_seed,
        manifest_size=args.manifest_size,
        image_size=args.image_size,
    )
    for name, commit in (
        ("rlinf_commit", config.rlinf_commit),
        ("benchmark_commit", config.benchmark_commit),
    ):
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError(f"{name} must be a full lowercase Git commit")
    if config.num_envs < 1 or config.eval_num_envs < 1:
        raise ValueError("environment counts must be positive")
    if config.total_env_steps < config.num_envs or config.total_env_steps % config.num_envs:
        raise ValueError("total_env_steps must be positive and divisible by num_envs")
    if config.rollout_steps < 1 or config.ppo_epochs < 1 or config.minibatch_size < 2:
        raise ValueError("PPO rollout/update sizes must be positive")
    if not 0.0 <= config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if not 0.0 < config.clip_coef < 1.0:
        raise ValueError("clip_coef must be in (0, 1)")
    if min(
        config.value_coef,
        config.entropy_coef,
        config.max_grad_norm,
        config.learning_rate,
    ) < 0.0:
        raise ValueError("PPO coefficients and learning rate must be non-negative")
    return config


def _policy_step(
    model: MLPPolicy,
    normalized_states: torch.Tensor,
    *,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, log_std = model._sample_actions(normalized_states)
    distribution = Normal(mean, torch.exp(log_std))
    raw_actions = distribution.sample() if stochastic else mean
    actions = torch.tanh(raw_actions)
    log_prob = distribution.log_prob(raw_actions) - torch.log(1.0 - actions.square() + 1e-6)
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
    tensors = (rewards, values, next_values, terminated, truncated)
    if len({tuple(tensor.shape) for tensor in tensors}) != 1 or rewards.ndim != 2:
        raise ValueError("GAE inputs must share [time, env] shape")
    deltas = rewards + gamma * (~terminated).to(rewards.dtype) * next_values - values
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(rewards[0])
    dones = torch.logical_or(terminated, truncated)
    for step in range(rewards.shape[0] - 1, -1, -1):
        accumulator = (
            deltas[step]
            + gamma
            * gae_lambda
            * (~dones[step]).to(rewards.dtype)
            * accumulator
        )
        advantages[step] = accumulator
    return advantages, advantages + values


def _ppo_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    loss = torch.maximum(unclipped, clipped).mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coef).to(torch.float32).mean()
    return loss, approx_kl, clip_fraction


def _ppo_update(
    config: PPOConfig,
    model: MLPPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, float], int]:
    batch_size = int(rollout["states"].shape[0] * rollout["states"].shape[1])
    states = rollout["states"].reshape(batch_size, -1).to(device)
    raw_actions = rollout["raw_actions"].reshape(batch_size, 7).to(device)
    actions = rollout["actions"].reshape(batch_size, 7).to(device)
    old_log_prob = rollout["log_prob"].reshape(batch_size).to(device)
    advantages = rollout["advantages"].reshape(batch_size).to(device)
    returns = rollout["returns"].reshape(batch_size).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    updates = 0
    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            selected_states = states.index_select(0, indices)
            selected_raw = raw_actions.index_select(0, indices)
            selected_actions = actions.index_select(0, indices)
            mean, log_std = model._sample_actions(selected_states)
            distribution = Normal(mean, torch.exp(log_std))
            new_log_prob = distribution.log_prob(selected_raw) - torch.log(
                1.0 - selected_actions.square() + 1e-6
            )
            new_log_prob = new_log_prob.sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1).mean()
            policy_loss, approx_kl, clip_fraction = _ppo_policy_loss(
                new_log_prob,
                old_log_prob.index_select(0, indices),
                advantages.index_select(0, indices),
                config.clip_coef,
            )
            predicted_values = model.value_head(selected_states).squeeze(-1)
            value_loss = 0.5 * (
                predicted_values - returns.index_select(0, indices)
            ).square().mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            updates += 1
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            totals["approx_kl"] += float(approx_kl.detach())
            totals["clip_fraction"] += float(clip_fraction.detach())
    return {name: value / max(1, updates) for name, value in totals.items()}, updates


def main() -> None:
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    args = _parse_args()
    config = _config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Dynamic Benchmark PPO training requires CUDA")
    if args.resume is None and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    heartbeat_path = args.output / "heartbeat.json"
    _atomic_json(args.output / "config.json", asdict(config))

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
    best_score: tuple[float, ...] | None = None
    best_metrics: dict[str, Any] | None = None
    last_validation: dict[str, Any] | None = None
    last_validation_env_steps = -1
    next_eval = config.eval_interval
    next_checkpoint = config.checkpoint_interval
    next_log = config.log_interval
    episode_returns = torch.zeros(config.num_envs)
    episode_steps = torch.zeros(config.num_envs, dtype=torch.int64)
    recent_episodes: list[dict[str, Any]] = []
    start_time = time.monotonic()

    def checkpoint(path: Path) -> None:
        state = {
            "schema_version": "rlinf-dynamic-benchmark-ppo-checkpoint-v0.1",
            "config": asdict(config),
            "state_schema": state_schema,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "env": env.checkpoint_state(),
            "global_env_steps": global_env_steps,
            "update_steps": update_steps,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "last_validation": last_validation,
            "last_validation_env_steps": last_validation_env_steps,
            "next_eval": next_eval,
            "next_checkpoint": next_checkpoint,
            "next_log": next_log,
            "episode_returns": episode_returns,
            "episode_steps": episode_steps,
            "recent_episodes": recent_episodes,
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
        optimizer.load_state_dict(restored["optimizer"])
        normalizer.load_state_dict(restored["normalizer"])
        env.load_checkpoint_state(restored["env"])
        obs = env._last_obs
        if obs is None:
            raise RuntimeError("resumed environment did not expose its current state")
        global_env_steps = int(restored["global_env_steps"])
        update_steps = int(restored["update_steps"])
        best_score = restored["best_score"]
        best_metrics = restored["best_metrics"]
        last_validation = restored.get("last_validation")
        last_validation_env_steps = int(restored.get("last_validation_env_steps", -1))
        next_eval = int(restored["next_eval"])
        next_checkpoint = int(restored["next_checkpoint"])
        next_log = int(restored["next_log"])
        episode_returns.copy_(restored["episode_returns"])
        episode_steps.copy_(restored["episode_steps"])
        recent_episodes[:] = restored["recent_episodes"]
        _restore_rng(restored["rng"])
        _append_jsonl(metrics_path, {"event": "resume", "env_steps": global_env_steps})
    else:
        normalizer.update(obs["states"])
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

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while global_env_steps < config.total_env_steps and not stop_requested:
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
                        model,
                        normalized_states,
                        stochastic=True,
                    )
                next_obs, rewards, terminated, truncated, infos = env.step(
                    actions.cpu(),
                    auto_reset=False,
                )
                normalized_next_states = normalizer.normalize(next_obs["states"], device)
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
            update_metrics, completed_updates = _ppo_update(
                config,
                model,
                optimizer,
                rollout,
                device,
            )
            update_steps += completed_updates

            if global_env_steps >= next_log:
                elapsed = max(time.monotonic() - start_time, 1e-6)
                payload: dict[str, Any] = {
                    "event": "train",
                    "env_steps": global_env_steps,
                    "update_steps": update_steps,
                    "env_steps_per_second": global_env_steps / elapsed,
                    **update_metrics,
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
                _append_jsonl(metrics_path, payload)
                _atomic_json(heartbeat_path, payload)
                while next_log <= global_env_steps:
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
                while next_eval <= global_env_steps:
                    next_eval += config.eval_interval

            if global_env_steps >= next_checkpoint:
                while next_checkpoint <= global_env_steps:
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
