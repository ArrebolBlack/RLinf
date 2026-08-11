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

"""Run bounded device-tensor SAC/RLPD smoke training on GPUENV0.

The environment, transition cohort, online replay, optional demonstration replay,
sampling RNGs, policy, critics, and optimizer work all remain on one CUDA device.
Online-policy host materialization occurs only after synchronized cohort/update
boundaries for evidence and checkpointing.  RLPD can either retain the mechanical
zero-action smoke seed or collect quality-gated privileged-teacher demonstrations
from explicitly materialized current GPU state.  The latter control plane is
reported separately and never weakens the online device-only path.  Default-off
actor behavior-cloning warm-start and demonstration regularization can keep RLPD
away from critic-extrapolated actions without changing the environment contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from rlinf.data.device_replay_buffer import (
    DeviceImitationReplayBuffer,
    DeviceReplayBatch,
    DeviceReplayBuffer,
)
from rlinf.data.device_transition_buffer import DeviceTransitionBuffer
from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import GpuNativeTensorBackendEnv

CHECKPOINT_SCHEMA = "rlinf-gpuenv0-tensor-offpolicy-smoke-v0.6"
REPORT_SCHEMA = "rlinf-gpuenv0-tensor-offpolicy-smoke-report-v0.6"
DEMO_QUALITY_SCHEMA = "rlinf-gpuenv0-demo-quality-v0.3"
DAGGER_CORRECTION_SCHEMA = "rlinf-gpuenv0-dagger-correction-v0.1"
MINIMUM_SUCCESS_ONLY_DEMO_EPISODES = 24
DEMO_PRODUCERS = {
    "zero_action": "zero_action_device_cohort_v1",
    "privileged_teacher": "current_gpu_state_privileged_teacher_v2",
}
TEACHER_OVERRIDE_FLOAT_RANGES = {
    "lookahead_s": (0.0, 1.0, True),
    "contact_lookahead_s": (0.0, 0.5, True),
    "hover_height_m": (0.0, 0.25, False),
    "grasp_height_offset_m": (-0.02, 0.04, True),
    "lift_height_m": (0.05, 0.20, False),
    "track_to_descend_distance_m": (0.005, 0.15, False),
    "close_horizontal_tolerance_m": (0.005, 0.08, False),
    "close_vertical_tolerance_m": (0.005, 0.08, False),
    "lift_action_z_max": (0.05, 1.0, False),
}
TEACHER_OVERRIDE_OPTIONAL_POSITIVE_INTS = {
    "close_retry_steps",
    "lift_contact_loss_retry_steps",
}
TEACHER_OVERRIDE_NONNEGATIVE_INTS = {"post_hold_settle_steps"}
TEACHER_OVERRIDE_BOOLS = {"lift_on_instantaneous_bilateral"}


@dataclass(frozen=True)
class TensorOffPolicyConfig:
    task: str
    algorithm: str
    export_dir: str
    expected_gpu_uuid: str
    seed: int
    split: str
    manifest_seed: int
    manifest_size: int
    num_envs: int
    cohorts: int
    warmup_steps: int
    hidden_size: int
    replay_capacity: int
    batch_size: int
    updates_per_cohort: int
    demo_ratio: float
    actor_bc_pretrain_updates: int
    actor_bc_weight: float
    actor_sac_weight: float
    dagger_cohorts: int
    dagger_bc_updates_per_cohort: int
    dagger_correction_ratio: float
    demo_policy: str | None
    demo_cohorts: int
    minimum_demo_success_rate: float
    demo_success_only_replay: bool
    minimum_qualified_demo_episodes: int
    demo_teacher_overrides: dict[str, Any]
    demo_teacher_overrides_sha256: str | None
    gamma: float
    tau: float
    actor_learning_rate: float
    critic_learning_rate: float
    alpha_learning_rate: float
    initial_alpha: float
    target_entropy: float
    image_size: int
    device_ordinal: int
    se3_source: str
    se3_commit: str
    se3_tree: str
    rlinf_source: str
    rlinf_commit: str
    rlinf_tree: str
    runtime_manifest: str
    runtime_manifest_sha256: str
    expected_cpuset: str


class TensorGaussianActor(nn.Module):
    """Small tanh-Gaussian policy for the under-filled MLP GPU regime."""

    def __init__(self, observation_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, 7)
        self.log_std = nn.Linear(hidden_size, 7)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)

    def distribution(self, observation: torch.Tensor) -> Normal:
        hidden = self.backbone(observation)
        return Normal(self.mean(hidden), self.log_std(hidden).clamp(-5.0, 2.0).exp())

    def sample(
        self,
        observation: torch.Tensor,
        *,
        stochastic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        raw_action = distribution.rsample() if stochastic else distribution.mean
        action = torch.tanh(raw_action)
        log_probability = (
            distribution.log_prob(raw_action) - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)
        return action, log_probability


def _q_network(observation_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(observation_dim + 7, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )


class TensorTwinQ(nn.Module):
    """Twin independent Q-functions with a compact MLP footprint."""

    def __init__(self, observation_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.q1 = _q_network(observation_dim, hidden_size)
        self.q2 = _q_network(observation_dim, hidden_size)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((observation, action), dim=-1)
        return torch.cat((self.q1(features), self.q2(features)), dim=-1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="p0_grasp")
    parser.add_argument("--algorithm", choices=("sac", "rlpd"), required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest-seed", type=int, default=20261050)
    parser.add_argument("--manifest-size", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--cohorts", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--updates-per-cohort", type=int, default=8)
    parser.add_argument("--demo-ratio", type=float, default=0.5)
    parser.add_argument(
        "--actor-bc-pretrain-updates",
        type=int,
        default=0,
        help=(
            "Device-only actor behavior-cloning updates after a qualified "
            "success-only demonstration bank; disabled by default."
        ),
    )
    parser.add_argument(
        "--actor-bc-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of a separately sampled demonstration BC loss in every "
            "online actor update; disabled by default."
        ),
    )
    parser.add_argument(
        "--actor-sac-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the SAC objective in each online actor update; defaults to "
            "one and may be reduced only by an explicitly recorded RLPD arm."
        ),
    )
    parser.add_argument(
        "--dagger-cohorts",
        type=int,
        default=0,
        help=(
            "Deterministic learner cohorts whose visited GPU states receive "
            "privileged-teacher correction labels; disabled by default."
        ),
    )
    parser.add_argument(
        "--dagger-bc-updates-per-cohort",
        type=int,
        default=0,
        help=(
            "Device-only actor BC updates after each DAgger correction cohort; "
            "disabled by default."
        ),
    )
    parser.add_argument(
        "--dagger-correction-ratio",
        type=float,
        default=0.5,
        help=(
            "Correction-replay fraction of each DAgger/online BC minibatch; the "
            "remainder comes from terminal-success demonstrations."
        ),
    )
    parser.add_argument(
        "--demo-policy",
        choices=tuple(DEMO_PRODUCERS),
        default="zero_action",
    )
    parser.add_argument("--demo-cohorts", type=int, default=1)
    parser.add_argument("--minimum-demo-success-rate", type=float, default=0.0)
    parser.add_argument(
        "--demo-success-only-replay",
        action="store_true",
        help=(
            "Insert only full terminal-success privileged-teacher lanes into the "
            "device demonstration replay; all attempts remain in the ledger."
        ),
    )
    parser.add_argument(
        "--minimum-qualified-demo-episodes",
        type=int,
        default=0,
        help=(
            "Minimum successful episodes required by success-only replay; GPUENV0 "
            f"requires at least {MINIMUM_SUCCESS_ONLY_DEMO_EPISODES}."
        ),
    )
    parser.add_argument(
        "--demo-teacher-overrides",
        type=Path,
        help=(
            "Strict JSON object of audited privileged-teacher planner overrides; "
            "task/evaluator thresholds are not configurable here."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha-learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-alpha", type=float, default=0.01)
    parser.add_argument("--target-entropy", type=float, default=-7.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--se3-source", type=Path, required=True)
    parser.add_argument("--se3-commit", required=True)
    parser.add_argument("--se3-tree", required=True)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--rlinf-tree", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--expected-cpuset", required=True)
    return parser


def _load_demo_teacher_overrides(
    path: Path | None,
) -> tuple[dict[str, Any], str | None]:
    """Load a bounded planner-only override object and its exact file identity."""

    if path is None:
        return {}, None
    resolved = path.expanduser().resolve(strict=True)
    raw = resolved.read_bytes()

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate demo teacher override key: {key}")
            result[key] = value
        return result

    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        raise ValueError(
            "demo teacher overrides must be a JSON object with string keys"
        )
    allowed = (
        set(TEACHER_OVERRIDE_FLOAT_RANGES)
        | TEACHER_OVERRIDE_OPTIONAL_POSITIVE_INTS
        | TEACHER_OVERRIDE_NONNEGATIVE_INTS
        | TEACHER_OVERRIDE_BOOLS
    )
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported demo teacher override keys: {unknown}")
    normalized: dict[str, Any] = {}
    for name in sorted(payload):
        value = payload[name]
        if name in TEACHER_OVERRIDE_FLOAT_RANGES:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"demo teacher override {name} must be numeric")
            numeric = float(value)
            minimum, maximum, minimum_inclusive = TEACHER_OVERRIDE_FLOAT_RANGES[name]
            lower_valid = numeric >= minimum if minimum_inclusive else numeric > minimum
            if not math.isfinite(numeric) or not lower_valid or numeric > maximum:
                bracket = "[" if minimum_inclusive else "("
                raise ValueError(
                    f"demo teacher override {name} must be in {bracket}{minimum}, {maximum}]"
                )
            normalized[name] = numeric
        elif name in TEACHER_OVERRIDE_OPTIONAL_POSITIVE_INTS:
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 160
            ):
                raise ValueError(
                    f"demo teacher override {name} must be null or an integer in [1, 160]"
                )
            normalized[name] = value
        elif name in TEACHER_OVERRIDE_NONNEGATIVE_INTS:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 20
            ):
                raise ValueError(
                    f"demo teacher override {name} must be an integer in [0, 20]"
                )
            normalized[name] = value
        else:
            if not isinstance(value, bool):
                raise ValueError(f"demo teacher override {name} must be boolean")
            normalized[name] = value
    return normalized, hashlib.sha256(raw).hexdigest()


def _config(args: argparse.Namespace) -> TensorOffPolicyConfig:
    demo_ratio = float(args.demo_ratio) if args.algorithm == "rlpd" else 0.0
    demo_policy = str(args.demo_policy) if args.algorithm == "rlpd" else None
    demo_cohorts = int(args.demo_cohorts) if args.algorithm == "rlpd" else 0
    minimum_demo_success_rate = (
        float(args.minimum_demo_success_rate) if args.algorithm == "rlpd" else 0.0
    )
    demo_success_only_replay = bool(args.demo_success_only_replay)
    minimum_qualified_demo_episodes = int(args.minimum_qualified_demo_episodes)
    demo_teacher_overrides, demo_teacher_overrides_sha256 = (
        _load_demo_teacher_overrides(args.demo_teacher_overrides)
    )
    config = TensorOffPolicyConfig(
        task=args.task,
        algorithm=args.algorithm,
        export_dir=str(args.export_dir),
        expected_gpu_uuid=args.expected_gpu_uuid,
        seed=args.seed,
        split=args.split,
        manifest_seed=args.manifest_seed,
        manifest_size=args.manifest_size,
        num_envs=args.num_envs,
        cohorts=args.cohorts,
        warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        updates_per_cohort=args.updates_per_cohort,
        demo_ratio=demo_ratio,
        actor_bc_pretrain_updates=args.actor_bc_pretrain_updates,
        actor_bc_weight=args.actor_bc_weight,
        actor_sac_weight=args.actor_sac_weight,
        dagger_cohorts=args.dagger_cohorts,
        dagger_bc_updates_per_cohort=args.dagger_bc_updates_per_cohort,
        dagger_correction_ratio=args.dagger_correction_ratio,
        demo_policy=demo_policy,
        demo_cohorts=demo_cohorts,
        minimum_demo_success_rate=minimum_demo_success_rate,
        demo_success_only_replay=demo_success_only_replay,
        minimum_qualified_demo_episodes=minimum_qualified_demo_episodes,
        demo_teacher_overrides=demo_teacher_overrides,
        demo_teacher_overrides_sha256=demo_teacher_overrides_sha256,
        gamma=args.gamma,
        tau=args.tau,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        alpha_learning_rate=args.alpha_learning_rate,
        initial_alpha=args.initial_alpha,
        target_entropy=args.target_entropy,
        image_size=args.image_size,
        device_ordinal=args.device_ordinal,
        se3_source=str(args.se3_source),
        se3_commit=args.se3_commit,
        se3_tree=args.se3_tree,
        rlinf_source=str(args.rlinf_source),
        rlinf_commit=args.rlinf_commit,
        rlinf_tree=args.rlinf_tree,
        runtime_manifest=str(args.runtime_manifest),
        runtime_manifest_sha256=args.runtime_manifest_sha256,
        expected_cpuset=args.expected_cpuset,
    )
    for name in (
        "num_envs",
        "cohorts",
        "hidden_size",
        "replay_capacity",
        "batch_size",
        "updates_per_cohort",
        "manifest_size",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.seed < 0 or config.manifest_seed < 0:
        raise ValueError("seeds must be non-negative")
    if config.manifest_size < config.num_envs:
        raise ValueError("manifest_size must be at least num_envs")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.actor_bc_pretrain_updates < 0:
        raise ValueError("actor_bc_pretrain_updates must be non-negative")
    if not math.isfinite(config.actor_bc_weight) or config.actor_bc_weight < 0.0:
        raise ValueError("actor_bc_weight must be finite and non-negative")
    if not math.isfinite(config.actor_sac_weight) or config.actor_sac_weight < 0.0:
        raise ValueError("actor_sac_weight must be finite and non-negative")
    if config.dagger_cohorts < 0:
        raise ValueError("dagger_cohorts must be non-negative")
    if config.dagger_bc_updates_per_cohort < 0:
        raise ValueError("dagger_bc_updates_per_cohort must be non-negative")
    if not math.isfinite(config.dagger_correction_ratio) or not (
        0.0 < config.dagger_correction_ratio < 1.0
    ):
        raise ValueError(
            "dagger_correction_ratio must be finite and strictly between zero and one"
        )
    if config.replay_capacity < config.batch_size:
        raise ValueError("replay_capacity must be at least batch_size")
    if not 0.0 <= config.demo_ratio <= 1.0:
        raise ValueError("demo_ratio must be in [0, 1]")
    if config.algorithm == "rlpd" and not 0.0 < config.demo_ratio < 1.0:
        raise ValueError("RLPD smoke requires demo_ratio strictly between zero and one")
    if config.algorithm == "rlpd":
        if config.demo_policy not in DEMO_PRODUCERS:
            raise ValueError("RLPD requires a supported demo_policy")
        if config.demo_cohorts < 1:
            raise ValueError("RLPD requires at least one demonstration cohort")
        if not 0.0 <= config.minimum_demo_success_rate <= 1.0:
            raise ValueError("minimum_demo_success_rate must be in [0, 1]")
        if config.minimum_qualified_demo_episodes < 0:
            raise ValueError("minimum_qualified_demo_episodes must be non-negative")
        if (
            config.demo_policy == "zero_action"
            and config.minimum_demo_success_rate != 0.0
        ):
            raise ValueError(
                "zero-action mechanical demonstrations cannot carry a quality gate"
            )
        if config.demo_success_only_replay:
            if config.demo_policy != "privileged_teacher":
                raise ValueError(
                    "success-only replay requires privileged_teacher demonstrations"
                )
            if config.minimum_demo_success_rate != 0.0:
                raise ValueError(
                    "success-only replay uses an episode-count gate, not a success-rate gate"
                )
            if (
                config.minimum_qualified_demo_episodes
                < MINIMUM_SUCCESS_ONLY_DEMO_EPISODES
            ):
                raise ValueError(
                    "success-only replay requires at least "
                    f"{MINIMUM_SUCCESS_ONLY_DEMO_EPISODES} qualified demo episodes"
                )
            attempted_episodes = config.demo_cohorts * config.num_envs
            if config.minimum_qualified_demo_episodes > attempted_episodes:
                raise ValueError(
                    "minimum_qualified_demo_episodes exceeds configured demo attempts"
                )
        elif config.minimum_qualified_demo_episodes != 0:
            raise ValueError(
                "minimum_qualified_demo_episodes requires success-only replay"
            )
        elif (
            config.demo_policy == "privileged_teacher"
            and config.minimum_demo_success_rate <= 0.0
        ):
            raise ValueError(
                "privileged-teacher demonstrations require a positive quality gate"
            )
        if (
            config.demo_policy != "privileged_teacher"
            and config.demo_teacher_overrides_sha256 is not None
        ):
            raise ValueError(
                "demo teacher overrides require privileged_teacher demonstrations"
            )
    elif (
        config.demo_teacher_overrides_sha256 is not None
        or config.demo_success_only_replay
        or config.minimum_qualified_demo_episodes != 0
    ):
        raise ValueError(
            "demo teacher overrides and success-only replay require RLPD "
            "privileged_teacher demos"
        )
    if config.actor_bc_pretrain_updates > 0 or config.actor_bc_weight > 0.0:
        if (
            config.algorithm != "rlpd"
            or config.demo_policy != "privileged_teacher"
            or not config.demo_success_only_replay
        ):
            raise ValueError(
                "actor BC requires RLPD privileged_teacher success-only replay"
            )
    dagger_enabled = config.dagger_cohorts > 0
    if dagger_enabled != (config.dagger_bc_updates_per_cohort > 0):
        raise ValueError(
            "dagger_cohorts and dagger_bc_updates_per_cohort must be enabled together"
        )
    if dagger_enabled:
        if (
            config.algorithm != "rlpd"
            or config.demo_policy != "privileged_teacher"
            or not config.demo_success_only_replay
            or config.actor_bc_pretrain_updates < 1
        ):
            raise ValueError(
                "DAgger correction requires RLPD privileged_teacher success-only "
                "replay and actor BC pretraining"
            )
        if config.batch_size < 2:
            raise ValueError("DAgger correction requires batch_size at least two")
    if config.actor_sac_weight != 1.0 and config.algorithm != "rlpd":
        raise ValueError("non-default actor_sac_weight requires RLPD")
    if (
        config.actor_sac_weight == 0.0
        and config.actor_bc_weight == 0.0
        and not dagger_enabled
    ):
        raise ValueError(
            "zero actor_sac_weight requires an actor BC or DAgger update path"
        )
    if not 0.0 <= config.gamma <= 1.0 or not 0.0 < config.tau <= 1.0:
        raise ValueError("gamma/tau are outside their valid ranges")
    if (
        min(
            config.actor_learning_rate,
            config.critic_learning_rate,
            config.alpha_learning_rate,
            config.initial_alpha,
        )
        <= 0.0
    ):
        raise ValueError("learning rates and initial_alpha must be positive")
    if not math.isfinite(config.target_entropy):
        raise ValueError("target_entropy must be finite")
    for name in ("se3_commit", "se3_tree", "rlinf_commit", "rlinf_tree"):
        value = getattr(config, name)
        if len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{name} must be a full lowercase Git object id")
    if len(config.runtime_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in config.runtime_manifest_sha256
    ):
        raise ValueError("runtime_manifest_sha256 must be a lowercase SHA-256")
    return config


def _mixed_batch(
    online: DeviceReplayBuffer,
    demos: DeviceReplayBuffer,
    *,
    batch_size: int,
    demo_ratio: float,
) -> DeviceReplayBatch:
    demo_rows = int(round(batch_size * demo_ratio))
    online_rows = batch_size - demo_rows
    chunks = []
    if online_rows:
        chunks.append(online.sample(online_rows))
    if demo_rows:
        chunks.append(demos.sample(demo_rows))
    if len(chunks) == 1:
        return chunks[0]
    merged = {
        name: torch.cat([getattr(chunk, name) for chunk in chunks], dim=0)
        for name in DeviceReplayBuffer.FIELDS
    }
    permutation = torch.randperm(batch_size, device=online.device)
    return DeviceReplayBatch(
        **{name: value.index_select(0, permutation) for name, value in merged.items()}
    )


def _sample_behavior_cloning_batch(
    demos: DeviceReplayBuffer,
    corrections: DeviceImitationReplayBuffer,
    *,
    batch_size: int,
    correction_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Sample successful demos plus separately labelled learner-visited states."""

    correction_rows = 0
    if corrections.size > 0:
        correction_rows = max(
            1,
            min(batch_size - 1, int(round(batch_size * correction_ratio))),
        )
    demo_rows = batch_size - correction_rows
    if demo_rows < 1 or demos.size < 1:
        raise RuntimeError("behavior cloning requires successful demonstration rows")
    demo_batch = demos.sample(demo_rows)
    if correction_rows == 0:
        return demo_batch.observation, demo_batch.action, demo_rows, 0
    correction_batch = corrections.sample(correction_rows)
    observation = torch.cat(
        (demo_batch.observation, correction_batch.observation), dim=0
    )
    action = torch.cat((demo_batch.action, correction_batch.action), dim=0)
    permutation = torch.randperm(batch_size, device=demos.device)
    return (
        observation.index_select(0, permutation),
        action.index_select(0, permutation),
        demo_rows,
        correction_rows,
    )


def _actor_bc_pretrain(
    *,
    config: TensorOffPolicyConfig,
    actor: TensorGaussianActor,
    actor_optimizer: torch.optim.Optimizer,
    demos: DeviceReplayBuffer,
) -> dict[str, Any]:
    """Optionally fit the actor to qualified demonstrations on its CUDA device."""

    before_sha256 = _structured_sha256(actor.state_dict())
    updates = config.actor_bc_pretrain_updates
    if updates == 0:
        return {
            "enabled": False,
            "configured_updates": 0,
            "completed_updates": 0,
            "demo_rows": demos.size,
            "device": str(demos.device),
            "first_loss": None,
            "last_loss": None,
            "mean_loss": None,
            "actor_sha256_before": before_sha256,
            "actor_sha256_after": before_sha256,
            "parameters_changed": False,
        }
    if demos.size < 1:
        raise RuntimeError("actor BC pretraining requires a non-empty demo replay")

    actor.train()
    first_loss: torch.Tensor | None = None
    last_loss: torch.Tensor | None = None
    loss_sum: torch.Tensor | None = None
    for _update in range(updates):
        batch = demos.sample(config.batch_size)
        predicted_action, _ = actor.sample(batch.observation, stochastic=False)
        loss = F.mse_loss(predicted_action, batch.action)
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        actor_optimizer.step()
        detached = loss.detach()
        if first_loss is None:
            first_loss = detached.clone()
            loss_sum = detached.clone()
        else:
            assert loss_sum is not None
            loss_sum = loss_sum + detached
        last_loss = detached

    torch.cuda.synchronize(demos.device)
    assert first_loss is not None and last_loss is not None and loss_sum is not None
    losses = {
        "first_loss": float(first_loss),
        "last_loss": float(last_loss),
        "mean_loss": float(loss_sum / updates),
    }
    if not all(np.isfinite(value) for value in losses.values()):
        raise RuntimeError(f"non-finite actor BC pretraining metrics: {losses}")
    after_sha256 = _structured_sha256(actor.state_dict())
    if after_sha256 == before_sha256:
        raise RuntimeError("actor BC pretraining did not change actor parameters")
    return {
        "enabled": True,
        "configured_updates": updates,
        "completed_updates": updates,
        "demo_rows": demos.size,
        "device": str(demos.device),
        **losses,
        "actor_sha256_before": before_sha256,
        "actor_sha256_after": after_sha256,
        "parameters_changed": True,
    }


def _dagger_bc_updates(
    *,
    config: TensorOffPolicyConfig,
    actor: TensorGaussianActor,
    actor_optimizer: torch.optim.Optimizer,
    demos: DeviceReplayBuffer,
    corrections: DeviceImitationReplayBuffer,
) -> dict[str, Any]:
    """Fit the actor to successful demos and learner-state correction labels."""

    updates = config.dagger_bc_updates_per_cohort
    if updates < 1 or corrections.size < 1:
        raise RuntimeError("DAgger BC requires configured updates and correction rows")
    before_sha256 = _structured_sha256(actor.state_dict())
    actor.train()
    first_loss: torch.Tensor | None = None
    last_loss: torch.Tensor | None = None
    loss_sum: torch.Tensor | None = None
    sampled_demo_rows = 0
    sampled_correction_rows = 0
    for _update in range(updates):
        observation, target_action, demo_rows, correction_rows = (
            _sample_behavior_cloning_batch(
                demos,
                corrections,
                batch_size=config.batch_size,
                correction_ratio=config.dagger_correction_ratio,
            )
        )
        predicted_action, _ = actor.sample(observation, stochastic=False)
        loss = F.mse_loss(predicted_action, target_action)
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        actor_optimizer.step()
        detached = loss.detach()
        if first_loss is None:
            first_loss = detached.clone()
            loss_sum = detached.clone()
        else:
            assert loss_sum is not None
            loss_sum = loss_sum + detached
        last_loss = detached
        sampled_demo_rows += demo_rows
        sampled_correction_rows += correction_rows

    torch.cuda.synchronize(demos.device)
    assert first_loss is not None and last_loss is not None and loss_sum is not None
    losses = {
        "first_loss": float(first_loss),
        "last_loss": float(last_loss),
        "mean_loss": float(loss_sum / updates),
    }
    if not all(np.isfinite(value) for value in losses.values()):
        raise RuntimeError(f"non-finite DAgger BC metrics: {losses}")
    after_sha256 = _structured_sha256(actor.state_dict())
    if after_sha256 == before_sha256:
        raise RuntimeError("DAgger BC did not change actor parameters")
    return {
        "configured_updates": updates,
        "completed_updates": updates,
        "batch_size": config.batch_size,
        "correction_ratio": config.dagger_correction_ratio,
        "sampled_demo_rows": sampled_demo_rows,
        "sampled_correction_rows": sampled_correction_rows,
        "demo_replay_rows": demos.size,
        "correction_replay_rows": corrections.size,
        "device": str(demos.device),
        **losses,
        "actor_sha256_before": before_sha256,
        "actor_sha256_after": after_sha256,
        "parameters_changed": True,
    }


def _offpolicy_updates(
    *,
    config: TensorOffPolicyConfig,
    actor: TensorGaussianActor,
    critic: TensorTwinQ,
    target_critic: TensorTwinQ,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: torch.optim.Optimizer,
    online: DeviceReplayBuffer,
    demos: DeviceReplayBuffer,
    corrections: DeviceImitationReplayBuffer,
) -> tuple[dict[str, float], int]:
    metric_rows = []
    for _update in range(config.updates_per_cohort):
        batch = _mixed_batch(
            online,
            demos,
            batch_size=config.batch_size,
            demo_ratio=config.demo_ratio,
        )
        actor_bc_batch = (
            _sample_behavior_cloning_batch(
                demos,
                corrections,
                batch_size=config.batch_size,
                correction_ratio=config.dagger_correction_ratio,
            )
            if config.actor_bc_weight > 0.0
            else None
        )
        with torch.no_grad():
            next_action, next_log_probability = actor.sample(
                batch.next_observation,
                stochastic=True,
            )
            target_q = (
                target_critic(batch.next_observation, next_action).min(dim=-1).values
            )
            target = batch.reward + config.gamma * (~batch.done).to(
                batch.reward.dtype
            ) * (target_q - log_alpha.exp() * next_log_probability)
        predicted = critic(batch.observation, batch.action)
        critic_loss = F.mse_loss(predicted, target.unsqueeze(-1).expand_as(predicted))
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        critic_optimizer.step()

        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        policy_action, log_probability = actor.sample(
            batch.observation,
            stochastic=True,
        )
        policy_q = critic(batch.observation, policy_action).min(dim=-1).values
        actor_sac_loss = (log_alpha.exp().detach() * log_probability - policy_q).mean()
        actor_bc_loss = actor_sac_loss.new_zeros(())
        if actor_bc_batch is not None:
            bc_observation, bc_target_action, _demo_rows, _correction_rows = (
                actor_bc_batch
            )
            demo_action, _ = actor.sample(
                bc_observation,
                stochastic=False,
            )
            actor_bc_loss = F.mse_loss(demo_action, bc_target_action)
        actor_loss = (
            config.actor_sac_weight * actor_sac_loss
            + config.actor_bc_weight * actor_bc_loss
        )
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        actor_optimizer.step()
        for parameter in critic.parameters():
            parameter.requires_grad_(True)

        alpha_loss = -(
            log_alpha * (log_probability + config.target_entropy).detach()
        ).mean()
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optimizer.step()
        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                critic.parameters(), target_critic.parameters(), strict=True
            ):
                target_parameter.lerp_(online_parameter, config.tau)
        metric_rows.append(
            torch.stack(
                (
                    critic_loss.detach(),
                    actor_loss.detach(),
                    actor_sac_loss.detach(),
                    actor_bc_loss.detach(),
                    alpha_loss.detach(),
                    log_alpha.exp().detach(),
                    predicted.detach().mean(),
                    target.detach().mean(),
                    critic_grad_norm.detach(),
                    actor_grad_norm.detach(),
                )
            )
        )
    metrics = torch.stack(metric_rows).mean(dim=0)
    torch.cuda.synchronize(online.device)
    names = (
        "critic_loss",
        "actor_loss",
        "actor_sac_loss",
        "actor_bc_loss",
        "alpha_loss",
        "alpha",
        "q_data",
        "q_target",
        "critic_grad_norm",
        "actor_grad_norm",
    )
    payload = {name: float(metrics[index]) for index, name in enumerate(names)}
    if not all(np.isfinite(value) for value in payload.values()):
        raise RuntimeError(f"non-finite {config.algorithm} metrics: {payload}")
    return payload, config.updates_per_cohort


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_value(digest: Any, value: Any) -> None:
    """Hash nested checkpoint state independently of tensor device placement."""

    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(b"I" + str(value).encode("ascii") + b";")
    elif isinstance(value, float):
        digest.update(b"F" + struct.pack("!d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + str(len(encoded)).encode("ascii") + b":" + encoded)
    elif isinstance(value, bytes):
        digest.update(b"Y" + str(len(value)).encode("ascii") + b":" + value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T" + str(tensor.dtype).encode("ascii") + b";")
        _hash_value(digest, tuple(tensor.shape))
        digest.update(tensor.numpy().tobytes(order="C"))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A" + str(array.dtype).encode("ascii") + b";")
        _hash_value(digest, tuple(array.shape))
        digest.update(array.tobytes(order="C"))
    elif isinstance(value, np.generic):
        _hash_value(digest, value.item())
    elif isinstance(value, Mapping):
        digest.update(b"M")
        for key in sorted(
            value, key=lambda item: (type(item).__qualname__, repr(item))
        ):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
        digest.update(b"m")
    elif isinstance(value, (list, tuple)):
        digest.update(b"L" if isinstance(value, list) else b"Q")
        for item in value:
            _hash_value(digest, item)
        digest.update(b"l")
    else:
        raise TypeError(f"unsupported checkpoint digest value {type(value)!r}")


def _structured_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _parameter_sha256(
    actor: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    log_alpha: torch.Tensor,
) -> str:
    return _structured_sha256(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": target_critic.state_dict(),
            "log_alpha": log_alpha.detach(),
        }
    )


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
    if set(state) != expected:
        raise ValueError(f"resume RNG state fields drifted: {sorted(state)}")
    cuda_states = state["torch_cuda_all"]
    if len(cuda_states) != torch.cuda.device_count():
        raise ValueError("resume CUDA RNG device count does not match runtime")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(cuda_states)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_state_sha256(payload: Mapping[str, Any]) -> str:
    return _structured_sha256(
        {key: value for key, value in payload.items() if key != "training_state_sha256"}
    )


def _load_resume_checkpoint(
    path: Path,
    config: TensorOffPolicyConfig,
) -> tuple[Path, str, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    sha256 = _file_sha256(resolved)
    restored = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(restored, dict):
        raise ValueError("resume checkpoint payload is not a mapping")
    if restored.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("resume checkpoint schema mismatch")
    if restored.get("config") != asdict(config):
        raise ValueError("resume checkpoint config identity does not match current run")
    required = {
        "cohort_horizon_steps",
        "observation_dim",
        "actor",
        "critic",
        "target_critic",
        "actor_optimizer",
        "critic_optimizer",
        "log_alpha",
        "alpha_optimizer",
        "online_replay",
        "demo_replay",
        "correction_replay",
        "manifest_cursor",
        "completed_online_cohorts",
        "completed_demo_cohorts",
        "completed_dagger_cohorts",
        "demo_quality",
        "dagger_correction",
        "actor_bc_pretrain",
        "update_steps",
        "rng_state",
        "parameter_sha256",
        "training_state_sha256",
    }
    missing = sorted(required - set(restored))
    if missing:
        raise ValueError(f"resume checkpoint is missing required fields: {missing}")
    observed_digest = _checkpoint_state_sha256(restored)
    if observed_digest != restored["training_state_sha256"]:
        raise ValueError("resume checkpoint training-state SHA-256 mismatch")
    if (
        int(restored["completed_online_cohorts"]) < 1
        or int(restored["update_steps"]) < 1
    ):
        raise ValueError("resume checkpoint does not contain completed training")
    completed_demo = int(restored["completed_demo_cohorts"])
    expected_demo = config.demo_cohorts if config.algorithm == "rlpd" else 0
    if completed_demo != expected_demo:
        raise ValueError(
            "resume demonstration-cohort counter violates algorithm contract"
        )
    completed_dagger = int(restored["completed_dagger_cohorts"])
    if completed_dagger != config.dagger_cohorts:
        raise ValueError("resume DAgger-cohort counter violates config contract")
    demo_quality = restored["demo_quality"]
    if not isinstance(demo_quality, Mapping):
        raise ValueError("resume demonstration quality evidence is not a mapping")
    if config.algorithm == "rlpd" and (
        demo_quality.get("demo_policy") != config.demo_policy
        or int(demo_quality.get("demo_cohorts", -1)) != config.demo_cohorts
        or bool(demo_quality.get("success_only_replay"))
        != config.demo_success_only_replay
        or int(demo_quality.get("minimum_qualified_episodes", -1))
        != config.minimum_qualified_demo_episodes
        or demo_quality.get("gate_passed") is not True
    ):
        raise ValueError("resume demonstration quality evidence differs from config")
    actor_bc_pretrain = restored["actor_bc_pretrain"]
    expected_bc_updates = config.actor_bc_pretrain_updates
    if (
        not isinstance(actor_bc_pretrain, Mapping)
        or int(actor_bc_pretrain.get("configured_updates", -1)) != expected_bc_updates
        or int(actor_bc_pretrain.get("completed_updates", -1)) != expected_bc_updates
        or bool(actor_bc_pretrain.get("enabled")) != (expected_bc_updates > 0)
        or (
            expected_bc_updates > 0
            and actor_bc_pretrain.get("parameters_changed") is not True
        )
    ):
        raise ValueError("resume actor BC pretraining evidence differs from config")
    dagger_correction = restored["dagger_correction"]
    if (
        not isinstance(dagger_correction, Mapping)
        or bool(dagger_correction.get("enabled")) != (config.dagger_cohorts > 0)
        or int(dagger_correction.get("configured_cohorts", -1)) != config.dagger_cohorts
        or int(dagger_correction.get("completed_cohorts", -1)) != config.dagger_cohorts
    ):
        raise ValueError("resume DAgger correction evidence differs from config")
    return resolved, sha256, restored


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_identity(
    *,
    root: str,
    expected_commit: str,
    expected_tree: str,
    imported_file: str,
) -> dict[str, Any]:
    source = Path(root).resolve(strict=True)
    loaded = Path(imported_file).resolve(strict=True)
    if not loaded.is_relative_to(source):
        raise RuntimeError(f"loaded module {loaded} is outside frozen source {source}")
    observed_commit = _git_output(source, "rev-parse", "HEAD")
    observed_tree = _git_output(source, "show", "-s", "--format=%T", "HEAD")
    dirty = _git_output(source, "status", "--porcelain=v1")
    if observed_commit != expected_commit or observed_tree != expected_tree or dirty:
        raise RuntimeError(
            f"source identity mismatch for {source}: commit={observed_commit}, "
            f"tree={observed_tree}, dirty={bool(dirty)}"
        )
    return {
        "path": str(source),
        "loaded_file": str(loaded),
        "commit": observed_commit,
        "tree": observed_tree,
        "tracked_worktree_clean": True,
    }


def _parse_cpuset(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("expected_cpuset contains an empty item")
        if "-" in item:
            start_text, stop_text = item.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if start < 0 or stop < start:
                raise ValueError("expected_cpuset contains an invalid range")
            result.update(range(start, stop + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError("expected_cpuset contains a negative CPU")
            result.add(cpu)
    if not result:
        raise ValueError("expected_cpuset must not be empty")
    return result


def _preflight(config: TensorOffPolicyConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    import se3_wam

    import rlinf

    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("tensor off-policy gate requires Linux CPU affinity")
    expected_cpus = _parse_cpuset(config.expected_cpuset)
    observed_cpus = set(os.sched_getaffinity(0))
    if observed_cpus != expected_cpus:
        raise RuntimeError(
            f"CPU affinity mismatch: expected {sorted(expected_cpus)}, "
            f"observed {sorted(observed_cpus)}"
        )
    sources = {
        "se3_wam": _source_identity(
            root=config.se3_source,
            expected_commit=config.se3_commit,
            expected_tree=config.se3_tree,
            imported_file=se3_wam.__file__,
        ),
        "rlinf": _source_identity(
            root=config.rlinf_source,
            expected_commit=config.rlinf_commit,
            expected_tree=config.rlinf_tree,
            imported_file=rlinf.__file__,
        ),
    }
    runtime_path = Path(config.runtime_manifest).resolve(strict=True)
    observed_manifest_sha256 = _file_sha256(runtime_path)
    if observed_manifest_sha256 != config.runtime_manifest_sha256:
        raise RuntimeError("runtime manifest SHA-256 mismatch")
    runtime = {
        "path": str(runtime_path),
        "sha256": observed_manifest_sha256,
        "payload": json.loads(runtime_path.read_text(encoding="utf-8")),
    }
    inventory = {
        "expected_cpuset": config.expected_cpuset,
        "observed_cpus": sorted(observed_cpus),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_executable": os.path.realpath(sys.executable),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    return {"sources": sources, "runtime": runtime}, inventory


def _validate_runtime_versions(
    provenance: Any, runtime_payload: dict[str, Any]
) -> None:
    expected = runtime_payload.get("versions")
    if not isinstance(expected, dict):
        raise RuntimeError("runtime manifest does not contain a versions mapping")
    observed = dict(provenance.runtime_versions)
    candidates = {
        "mujoco": ("mujoco",),
        "mujoco-warp": ("mujoco-warp", "mujoco-mjx"),
        "warp-lang": ("warp-lang",),
    }
    mismatches = {}
    for observed_name, expected_names in candidates.items():
        expected_value = next(
            (str(expected[name]) for name in expected_names if name in expected), None
        )
        if expected_value is None or observed.get(observed_name) != expected_value:
            mismatches[observed_name] = {
                "observed": observed.get(observed_name),
                "expected": expected_value,
            }
    if mismatches:
        raise RuntimeError(
            f"runtime manifest/provenance version mismatch: {mismatches}"
        )


def _new_transition_buffer(
    env: GpuNativeTensorBackendEnv,
    observation_dim: int,
    observation_dtype: torch.dtype,
) -> DeviceTransitionBuffer:
    return DeviceTransitionBuffer(
        capacity=env.cohort_horizon_steps,
        num_envs=env.num_envs,
        observation_shape=(observation_dim,),
        action_shape=(7,),
        device=env.device,
        observation_dtype=observation_dtype,
        action_dtype=torch.float32,
        reward_dtype=torch.float32,
        terminal_signal_dtype=torch.int32,
        event_mask_dtype=torch.int32,
        terminal_reason_dtype=torch.int32,
        physics_step_dtype=torch.int64,
    )


def _rollout_cohort(
    *,
    env: GpuNativeTensorBackendEnv,
    actor: TensorGaussianActor,
    buffer: DeviceTransitionBuffer,
    reset: Any,
    mode: str,
) -> tuple[Any, float]:
    if mode not in {"zero_demo", "online"}:
        raise ValueError(f"unsupported device-only rollout mode {mode!r}")
    observation = reset.observation
    zero_action = torch.zeros((env.num_envs, 7), dtype=torch.float32, device=env.device)
    buffer.reset_cohort()
    actor.eval()
    torch.cuda.synchronize(env.device)
    started = time.perf_counter()
    for _step in range(env.cohort_horizon_steps):
        with torch.inference_mode():
            if mode == "zero_demo":
                action = zero_action
            else:
                action, _log_probability = actor.sample(
                    observation,
                    stochastic=True,
                )
                action = action.contiguous()
        buffer.begin_step(observation=observation, action=action)
        try:
            with torch.inference_mode():
                step = env.step(action)
            buffer.commit_step(
                reward=step.reward,
                next_observation=step.observation,
                terminated=step.terminated,
                truncated=step.truncated,
                success=step.success,
                event_mask=step.event_mask,
                terminal_reason=step.terminal_reason,
                physics_step=step.physics_step,
            )
        except BaseException:
            if buffer.pending:
                buffer.abort_step()
            raise
        observation = step.observation
    torch.cuda.synchronize(env.device)
    return buffer.view(), time.perf_counter() - started


def _validated_teacher_action(
    command: Any,
    observation: Any,
    *,
    action_mode: Any,
) -> np.ndarray:
    """Return one exact finite E7 teacher action at the observation clock."""

    if getattr(command, "mode", None) is not action_mode:
        raise RuntimeError("privileged teacher changed the E7 action contract")
    if getattr(command, "policy_step", None) != observation.policy_step:
        raise RuntimeError(
            "privileged teacher action clock differs from GPU observation"
        )
    values = np.asarray(getattr(command, "values", None), dtype=np.float64)
    if (
        values.shape != (7,)
        or not np.all(np.isfinite(values))
        or np.any(values < -1.0)
        or np.any(values > 1.0)
    ):
        raise RuntimeError(
            "privileged teacher produced an invalid normalized E7 action"
        )
    return values.astype(np.float32, copy=True)


def _apply_demo_teacher_overrides(teacher: Any, overrides: Mapping[str, Any]) -> None:
    """Apply already-validated planner parameters without touching task semantics."""

    for name, value in overrides.items():
        if not hasattr(teacher, name):
            raise RuntimeError(
                f"privileged teacher does not expose audited planner parameter {name}"
            )
        setattr(teacher, name, value)


def _rollout_privileged_teacher_cohort(
    *,
    env: GpuNativeTensorBackendEnv,
    buffer: DeviceTransitionBuffer,
    reset: Any,
    teacher_overrides: Mapping[str, Any] | None = None,
) -> tuple[Any, float, dict[str, Any]]:
    """Collect one demo cohort from current GPU state through the host control plane."""

    from se3_wam.benchmark.contracts import ActionMode
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    effective_overrides = dict(teacher_overrides or {})
    teachers = []
    teacher_metadata = []
    for _lane in range(env.num_envs):
        teacher, metadata = make_privileged_teacher(env.task_id)
        _apply_demo_teacher_overrides(teacher, effective_overrides)
        teacher.reset()
        teachers.append(teacher)
        teacher_metadata.append(
            {**dict(metadata), "runtime_planner_overrides": effective_overrides}
        )
    if any(metadata != teacher_metadata[0] for metadata in teacher_metadata[1:]):
        raise RuntimeError("privileged teacher metadata differs across cohort lanes")
    observation_audit_calls_before = env.teacher_audit_materializations
    observation_audit_lanes = 0
    terminal_mask_host_materializations = 0
    host_to_device_action_transfers = 0
    active = np.ones(env.num_envs, dtype=np.bool_)
    zero_action = torch.zeros((env.num_envs, 7), dtype=torch.float32, device=env.device)
    device_observation = reset.observation
    buffer.reset_cohort()
    torch.cuda.synchronize(env.device)
    started = time.perf_counter()
    for _step in range(env.cohort_horizon_steps):
        active_lanes = tuple(int(lane) for lane in np.flatnonzero(active))
        if active_lanes:
            observations = env.materialize_teacher_observations(active_lanes)
            if len(observations) != len(active_lanes):
                raise RuntimeError(
                    "teacher audit row count differs from active GPU lanes"
                )
            host_actions = np.zeros((env.num_envs, 7), dtype=np.float32)
            for lane, teacher_observation in zip(
                active_lanes, observations, strict=True
            ):
                if (
                    teacher_observation.episode_id != reset.episode_ids[lane]
                    or teacher_observation.task_id != env.task_id
                    or teacher_observation.control_step != _step
                    or teacher_observation.policy_step != _step
                ):
                    raise RuntimeError(
                        "privileged teacher observation identity/clock drifted"
                    )
                command = teachers[lane].act(teacher_observation)
                host_actions[lane] = _validated_teacher_action(
                    command,
                    teacher_observation,
                    action_mode=ActionMode.E7,
                )
            action = torch.as_tensor(
                host_actions,
                dtype=torch.float32,
                device=env.device,
            ).contiguous()
            observation_audit_lanes += len(active_lanes)
            host_to_device_action_transfers += 1
        else:
            action = zero_action
        buffer.begin_step(observation=device_observation, action=action)
        try:
            with torch.inference_mode():
                step = env.step(action)
            buffer.commit_step(
                reward=step.reward,
                next_observation=step.observation,
                terminated=step.terminated,
                truncated=step.truncated,
                success=step.success,
                event_mask=step.event_mask,
                terminal_reason=step.terminal_reason,
                physics_step=step.physics_step,
            )
        except BaseException:
            if buffer.pending:
                buffer.abort_step()
            raise
        device_observation = step.observation
        if active_lanes:
            done = np.asarray(
                step.done.detach().to(device="cpu").numpy(), dtype=np.bool_
            )
            if done.shape != (env.num_envs,):
                raise RuntimeError("GPU terminal mask has the wrong host audit shape")
            active &= ~done
            terminal_mask_host_materializations += 1
    torch.cuda.synchronize(env.device)
    observation_audit_calls = (
        env.teacher_audit_materializations - observation_audit_calls_before
    )
    evidence = {
        "producer": DEMO_PRODUCERS["privileged_teacher"],
        "teacher_count": len(teachers),
        "teacher_metadata": teacher_metadata[0],
        "observation_audit_calls": observation_audit_calls,
        "observation_audit_lanes": observation_audit_lanes,
        "terminal_mask_host_materializations": terminal_mask_host_materializations,
        "host_to_device_action_transfers": host_to_device_action_transfers,
        "online_hot_path_host_materializations": 0,
    }
    return buffer.view(), time.perf_counter() - started, evidence


def _rollout_dagger_correction_cohort(
    *,
    env: GpuNativeTensorBackendEnv,
    actor: TensorGaussianActor,
    online_buffer: DeviceTransitionBuffer,
    correction_buffer: DeviceTransitionBuffer,
    reset: Any,
    teacher_overrides: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, float, dict[str, Any]]:
    """Execute the GPU actor while labelling its visited states with the teacher."""

    from se3_wam.benchmark.contracts import ActionMode
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    effective_overrides = dict(teacher_overrides or {})
    teachers = []
    teacher_metadata = []
    for _lane in range(env.num_envs):
        teacher, metadata = make_privileged_teacher(env.task_id)
        _apply_demo_teacher_overrides(teacher, effective_overrides)
        teacher.reset()
        teachers.append(teacher)
        teacher_metadata.append(
            {**dict(metadata), "runtime_planner_overrides": effective_overrides}
        )
    if any(metadata != teacher_metadata[0] for metadata in teacher_metadata[1:]):
        raise RuntimeError("DAgger teacher metadata differs across cohort lanes")

    observation_audit_calls_before = env.teacher_audit_materializations
    observation_audit_lanes = 0
    terminal_mask_host_materializations = 0
    host_to_device_label_transfers = 0
    active = np.ones(env.num_envs, dtype=np.bool_)
    zero_label = torch.zeros((env.num_envs, 7), dtype=torch.float32, device=env.device)
    observation = reset.observation
    online_buffer.reset_cohort()
    correction_buffer.reset_cohort()
    actor.eval()
    torch.cuda.synchronize(env.device)
    started = time.perf_counter()
    for _step in range(env.cohort_horizon_steps):
        with torch.inference_mode():
            learner_action, _log_probability = actor.sample(
                observation,
                stochastic=False,
            )
            learner_action = learner_action.contiguous()
        active_lanes = tuple(int(lane) for lane in np.flatnonzero(active))
        if active_lanes:
            teacher_observations = env.materialize_teacher_observations(active_lanes)
            if len(teacher_observations) != len(active_lanes):
                raise RuntimeError(
                    "DAgger teacher audit row count differs from active GPU lanes"
                )
            host_labels = np.zeros((env.num_envs, 7), dtype=np.float32)
            for lane, teacher_observation in zip(
                active_lanes, teacher_observations, strict=True
            ):
                if (
                    teacher_observation.episode_id != reset.episode_ids[lane]
                    or teacher_observation.task_id != env.task_id
                    or teacher_observation.control_step != _step
                    or teacher_observation.policy_step != _step
                ):
                    raise RuntimeError(
                        "DAgger teacher observation identity/clock drifted"
                    )
                command = teachers[lane].act(teacher_observation)
                host_labels[lane] = _validated_teacher_action(
                    command,
                    teacher_observation,
                    action_mode=ActionMode.E7,
                )
            correction_label = torch.as_tensor(
                host_labels,
                dtype=torch.float32,
                device=env.device,
            ).contiguous()
            observation_audit_lanes += len(active_lanes)
            host_to_device_label_transfers += 1
        else:
            correction_label = zero_label

        online_buffer.begin_step(observation=observation, action=learner_action)
        correction_buffer.begin_step(observation=observation, action=correction_label)
        try:
            with torch.inference_mode():
                step = env.step(learner_action)
            commit = {
                "reward": step.reward,
                "next_observation": step.observation,
                "terminated": step.terminated,
                "truncated": step.truncated,
                "success": step.success,
                "event_mask": step.event_mask,
                "terminal_reason": step.terminal_reason,
                "physics_step": step.physics_step,
            }
            online_buffer.commit_step(**commit)
            correction_buffer.commit_step(**commit)
        except BaseException:
            if online_buffer.pending:
                online_buffer.abort_step()
            if correction_buffer.pending:
                correction_buffer.abort_step()
            raise
        observation = step.observation
        if active_lanes:
            done = np.asarray(
                step.done.detach().to(device="cpu").numpy(), dtype=np.bool_
            )
            if done.shape != (env.num_envs,):
                raise RuntimeError("GPU terminal mask has the wrong DAgger audit shape")
            active &= ~done
            terminal_mask_host_materializations += 1

    torch.cuda.synchronize(env.device)
    observation_audit_calls = (
        env.teacher_audit_materializations - observation_audit_calls_before
    )
    evidence = {
        "producer": "learner_visited_gpu_state_privileged_teacher_correction_v1",
        "executed_action_source": "deterministic_gpu_actor",
        "label_action_source": DEMO_PRODUCERS["privileged_teacher"],
        "labels_are_successful_demonstrations": False,
        "teacher_count": len(teachers),
        "teacher_metadata": teacher_metadata[0],
        "observation_audit_calls": observation_audit_calls,
        "observation_audit_lanes": observation_audit_lanes,
        "terminal_mask_host_materializations": terminal_mask_host_materializations,
        "host_to_device_label_transfers": host_to_device_label_transfers,
        "online_hot_path_host_materializations": 0,
    }
    return (
        online_buffer.view(),
        correction_buffer.view(),
        time.perf_counter() - started,
        evidence,
    )


def _successful_demo_lane_mask(rollout: Any) -> Any:
    """Return the device-local lanes whose valid terminal transition succeeded."""

    lane_mask = (rollout.success & rollout.done & rollout.valid).any(dim=0)
    expected_shape = (int(rollout.valid.shape[1]),)
    if tuple(lane_mask.shape) != expected_shape:
        raise RuntimeError("successful demo lane mask has the wrong shape")
    if lane_mask.dtype != torch.bool or lane_mask.device != rollout.valid.device:
        raise RuntimeError("successful demo lane mask left the rollout CUDA device")
    return lane_mask.contiguous()


def _add_rollout_to_replay(
    replay: DeviceReplayBuffer,
    rollout: Any,
    *,
    lane_mask: Any | None = None,
) -> int:
    """Insert a rollout, optionally retaining only selected lanes on the device."""

    horizon, num_envs = rollout.valid.shape
    if lane_mask is None:
        rows = horizon * num_envs
        observation = rollout.observation.reshape(rows, -1)
        action = rollout.action.reshape(rows, 7)
        reward = rollout.reward.reshape(rows)
        next_observation = rollout.next_observation.reshape(rows, -1)
        terminated = rollout.terminated.reshape(rows)
        truncated = rollout.truncated.reshape(rows)
        valid = rollout.valid.reshape(rows)
    else:
        if tuple(lane_mask.shape) != (num_envs,):
            raise ValueError("replay lane_mask has the wrong shape")
        if lane_mask.dtype != torch.bool or lane_mask.device != rollout.valid.device:
            raise ValueError("replay lane_mask must be bool on the rollout device")
        selector = rollout.valid & lane_mask.unsqueeze(0)
        observation = rollout.observation[selector].reshape(
            -1, rollout.observation.shape[-1]
        )
        action = rollout.action[selector].reshape(-1, 7)
        reward = rollout.reward[selector].reshape(-1)
        next_observation = rollout.next_observation[selector].reshape(
            -1, rollout.next_observation.shape[-1]
        )
        terminated = rollout.terminated[selector].reshape(-1)
        truncated = rollout.truncated[selector].reshape(-1)
        valid = rollout.valid[selector].reshape(-1)
        rows = int(observation.shape[0])
        if rows == 0:
            return 0
        observation = observation.contiguous()
        action = action.contiguous()
        reward = reward.contiguous()
        next_observation = next_observation.contiguous()
        terminated = terminated.contiguous()
        truncated = truncated.contiguous()
        valid = valid.contiguous()
    replay.add_batch(
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        terminated=terminated,
        truncated=truncated,
        valid=valid,
    )
    return rows


def _add_correction_rollout_to_replay(
    replay: DeviceImitationReplayBuffer,
    rollout: Any,
) -> int:
    """Insert only valid observation/teacher-action labels on the rollout device."""

    selector = rollout.valid
    observation = rollout.observation[selector].reshape(
        -1, rollout.observation.shape[-1]
    )
    action = rollout.action[selector].reshape(-1, rollout.action.shape[-1])
    rows = int(observation.shape[0])
    if rows == 0:
        return 0
    replay.add_batch(
        observation=observation.contiguous(),
        action=action.contiguous(),
    )
    return rows


def _cohort_evidence(
    *,
    config: TensorOffPolicyConfig,
    env: GpuNativeTensorBackendEnv,
    reset: Any,
    rollout: Any,
    role: str,
    cohort: int,
    rollout_seconds: float,
    update_seconds: float,
    updates: int,
    metrics: dict[str, float],
    seen_episode_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allocated_steps = env.cohort_horizon_steps * config.num_envs
    valid_steps = int(rollout.valid.sum())
    successes = int((rollout.success & rollout.done & rollout.valid).sum())
    done_valid = rollout.done & rollout.valid
    lane_valid_steps = rollout.valid.sum(dim=0).cpu().tolist()
    lane_returns = (
        (rollout.reward * rollout.valid.to(rollout.reward.dtype))
        .sum(dim=0)
        .cpu()
        .tolist()
    )
    lane_terminated = (rollout.terminated & rollout.valid).any(dim=0).cpu().tolist()
    lane_truncated = (rollout.truncated & rollout.valid).any(dim=0).cpu().tolist()
    lane_success = (rollout.success & done_valid).any(dim=0).cpu().tolist()
    lane_terminal_reason = (
        torch.where(
            done_valid,
            rollout.terminal_reason,
            torch.zeros_like(rollout.terminal_reason),
        )
        .amax(dim=0)
        .cpu()
        .tolist()
    )
    lane_terminal_physics_step = (
        torch.where(
            done_valid,
            rollout.physics_step,
            torch.zeros_like(rollout.physics_step),
        )
        .amax(dim=0)
        .cpu()
        .tolist()
    )
    ledger_rows = []
    for lane, episode_id in enumerate(reset.episode_ids):
        if episode_id in seen_episode_ids:
            raise RuntimeError(f"duplicate cohort episode id {episode_id}")
        seen_episode_ids.add(episode_id)
        ledger_rows.append(
            {
                "role": role,
                "cohort": cohort,
                "lane": lane,
                "episode_id": episode_id,
                "seed": reset.seeds[lane],
                "manifest_ordinal": reset.manifest_ordinals[lane],
                "manifest_sha256": reset.manifest_sha256,
                "task_id": config.task,
                "algorithm": config.algorithm,
                "backend_id": env.provenance.backend_id,
                "physical_device_uuid": config.expected_gpu_uuid,
                "valid_steps": int(lane_valid_steps[lane]),
                "return": float(lane_returns[lane]),
                "terminated": bool(lane_terminated[lane]),
                "truncated": bool(lane_truncated[lane]),
                "success": bool(lane_success[lane]),
                "terminal_reason_code": int(lane_terminal_reason[lane]),
                "terminal_physics_step": int(lane_terminal_physics_step[lane]),
            }
        )
    summary = {
        "role": role,
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
    return summary, ledger_rows


def main() -> int:
    args = _parser().parse_args()
    config = _config(args)
    provenance_bundle, inventory = _preflight(config)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "tensor off-policy smoke requires CUDA; CPU fallback is forbidden"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "config.json", asdict(config))
    ledger_path = args.output / "episode_ledger.jsonl"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    resume_path: Path | None = None
    resume_sha256: str | None = None
    restored: dict[str, Any] | None = None
    if args.resume_from is not None:
        resume_path, resume_sha256, restored = _load_resume_checkpoint(
            args.resume_from,
            config,
        )

    env = GpuNativeTensorBackendEnv(
        task_id=config.task,
        num_envs=config.num_envs,
        export_dir=config.export_dir,
        expected_gpu_uuid=config.expected_gpu_uuid,
        expected_se3_source_commit=config.se3_commit,
        expected_se3_source_tree=config.se3_tree,
        device_ordinal=config.device_ordinal,
        image_size=config.image_size,
        split=config.split,
        manifest_seed=config.manifest_seed,
        manifest_size=config.manifest_size,
    )
    try:
        _validate_runtime_versions(
            env.provenance,
            provenance_bundle["runtime"]["payload"],
        )
        properties = torch.cuda.get_device_properties(env.device)
        free_memory, total_memory = torch.cuda.mem_get_info(env.device)
        inventory.update(
            {
                "gpu_name": properties.name,
                "gpu_compute_capability": [properties.major, properties.minor],
                "gpu_total_memory_bytes": int(total_memory),
                "gpu_free_memory_bytes_at_start": int(free_memory),
            }
        )
        completed_online_before = 0
        completed_demo_before = 0
        completed_dagger_before = 0
        update_steps_before = 0
        restored_manifest_cursor: dict[str, Any] | None = None
        if restored is not None:
            if int(restored["cohort_horizon_steps"]) != env.cohort_horizon_steps:
                raise ValueError(
                    "resume cohort horizon does not match live environment"
                )
            restored_manifest_cursor = dict(restored["manifest_cursor"])
            env.load_manifest_state_dict(restored_manifest_cursor)
            completed_online_before = int(restored["completed_online_cohorts"])
            completed_demo_before = int(restored["completed_demo_cohorts"])
            completed_dagger_before = int(restored["completed_dagger_cohorts"])
            update_steps_before = int(restored["update_steps"])
        initial = env.reset()
        observation = initial.observation
        observation_dim = int(observation.shape[1])
        if restored is not None and int(restored["observation_dim"]) != observation_dim:
            raise ValueError(
                "resume observation dimension does not match live environment"
            )
        required_correction_capacity = (
            config.dagger_cohorts * config.num_envs * env.cohort_horizon_steps
        )
        if (
            config.dagger_cohorts > 0
            and config.replay_capacity < required_correction_capacity
        ):
            raise ValueError(
                "replay_capacity cannot retain every configured DAgger correction row"
            )

        actor = TensorGaussianActor(observation_dim, config.hidden_size).to(env.device)
        critic = TensorTwinQ(observation_dim, config.hidden_size).to(env.device)
        target_critic = TensorTwinQ(observation_dim, config.hidden_size).to(env.device)
        target_critic.load_state_dict(critic.state_dict())
        target_critic.requires_grad_(False)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=config.actor_learning_rate
        )
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=config.critic_learning_rate
        )
        log_alpha = nn.Parameter(
            torch.tensor(
                math.log(config.initial_alpha),
                dtype=torch.float32,
                device=env.device,
            )
        )
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.alpha_learning_rate)
        online = DeviceReplayBuffer(
            capacity=config.replay_capacity,
            observation_shape=(observation_dim,),
            action_shape=(7,),
            device=env.device,
            seed=config.seed + 17,
            observation_dtype=observation.dtype,
            action_dtype=torch.float32,
            reward_dtype=torch.float32,
        )
        demos = DeviceReplayBuffer(
            capacity=config.replay_capacity,
            observation_shape=(observation_dim,),
            action_shape=(7,),
            device=env.device,
            seed=config.seed + 11,
            observation_dtype=observation.dtype,
            action_dtype=torch.float32,
            reward_dtype=torch.float32,
        )
        correction_capacity = config.replay_capacity if config.dagger_cohorts > 0 else 1
        corrections = DeviceImitationReplayBuffer(
            capacity=correction_capacity,
            observation_shape=(observation_dim,),
            action_shape=(7,),
            device=env.device,
            seed=config.seed + 23,
            observation_dtype=observation.dtype,
            action_dtype=torch.float32,
        )
        transition = _new_transition_buffer(env, observation_dim, observation.dtype)
        correction_transition = _new_transition_buffer(
            env, observation_dim, observation.dtype
        )

        if restored is not None:
            actor.load_state_dict(restored["actor"], strict=True)
            critic.load_state_dict(restored["critic"], strict=True)
            target_critic.load_state_dict(restored["target_critic"], strict=True)
            actor_optimizer.load_state_dict(restored["actor_optimizer"])
            critic_optimizer.load_state_dict(restored["critic_optimizer"])
            log_alpha.data.copy_(restored["log_alpha"].to(env.device))
            alpha_optimizer.load_state_dict(restored["alpha_optimizer"])
            online.load_state_dict(restored["online_replay"])
            demos.load_state_dict(restored["demo_replay"])
            corrections.load_state_dict(restored["correction_replay"])
            parameter_sha256_start = _parameter_sha256(
                actor, critic, target_critic, log_alpha
            )
            if parameter_sha256_start != restored["parameter_sha256"]:
                raise ValueError("resume parameter SHA-256 does not match checkpoint")
            _restore_rng_state(restored["rng_state"])
        else:
            parameter_sha256_start = _parameter_sha256(
                actor, critic, target_critic, log_alpha
            )

        actor.eval()
        for _step in range(config.warmup_steps):
            with torch.inference_mode():
                actor.sample(observation, stochastic=False)
        torch.cuda.synchronize(env.device)

        cohort_rows = []
        update_steps = update_steps_before
        invocation_update_steps = 0
        total_allocated_steps = 0
        total_valid_steps = 0
        total_successes = 0
        seen_episode_ids: set[str] = set()
        wall_started = time.perf_counter()
        next_reset = initial
        completed_demo_after = completed_demo_before
        demo_quality_path = args.output / "demo_quality.json"
        if restored is not None:
            demo_quality = dict(restored["demo_quality"])
            actor_bc_pretrain = dict(restored["actor_bc_pretrain"])
            _atomic_json(demo_quality_path, demo_quality)
        elif config.algorithm == "rlpd":
            demo_allocated_steps = 0
            demo_valid_steps = 0
            demo_successes = 0
            demo_episodes = 0
            demo_replay_inserted_rows = 0
            demo_replay_valid_steps = 0
            demo_attempt_manifest_ordinals: list[int] = []
            successful_demo_episode_ids: list[str] = []
            successful_demo_manifest_ordinals: list[int] = []
            demo_manifest_sha256: str | None = None
            demo_control_plane = []
            for demo_cohort in range(config.demo_cohorts):
                reset = next_reset
                if config.demo_policy == "privileged_teacher":
                    (
                        demo_rollout,
                        demo_rollout_seconds,
                        control_plane,
                    ) = _rollout_privileged_teacher_cohort(
                        env=env,
                        buffer=transition,
                        reset=reset,
                        teacher_overrides=config.demo_teacher_overrides,
                    )
                    role = (
                        "privileged_teacher_attempt"
                        if config.demo_success_only_replay
                        else "quality_gated_privileged_teacher_demo"
                    )
                else:
                    demo_rollout, demo_rollout_seconds = _rollout_cohort(
                        env=env,
                        actor=actor,
                        buffer=transition,
                        reset=reset,
                        mode="zero_demo",
                    )
                    control_plane = {
                        "producer": DEMO_PRODUCERS["zero_action"],
                        "teacher_count": 0,
                        "teacher_metadata": None,
                        "observation_audit_calls": 0,
                        "observation_audit_lanes": 0,
                        "terminal_mask_host_materializations": 0,
                        "host_to_device_action_transfers": 0,
                        "online_hot_path_host_materializations": 0,
                    }
                    role = "mechanical_demo"
                summary, ledger_rows = _cohort_evidence(
                    config=config,
                    env=env,
                    reset=reset,
                    rollout=demo_rollout,
                    role=role,
                    cohort=demo_cohort,
                    rollout_seconds=demo_rollout_seconds,
                    update_seconds=0.0,
                    updates=0,
                    metrics={},
                    seen_episode_ids=seen_episode_ids,
                )
                observed_manifest_sha256 = str(reset.manifest_sha256)
                if demo_manifest_sha256 is None:
                    demo_manifest_sha256 = observed_manifest_sha256
                elif observed_manifest_sha256 != demo_manifest_sha256:
                    raise RuntimeError("demo attempt manifest identity drifted")
                demo_attempt_manifest_ordinals.extend(
                    int(row["manifest_ordinal"]) for row in ledger_rows
                )
                successful_rows = [row for row in ledger_rows if row["success"]]
                successful_demo_episode_ids.extend(
                    str(row["episode_id"]) for row in successful_rows
                )
                successful_demo_manifest_ordinals.extend(
                    int(row["manifest_ordinal"]) for row in successful_rows
                )
                if config.demo_success_only_replay:
                    expected_replay_rows = sum(
                        int(row["valid_steps"]) for row in successful_rows
                    )
                    replay_rows = _add_rollout_to_replay(
                        demos,
                        demo_rollout,
                        lane_mask=_successful_demo_lane_mask(demo_rollout),
                    )
                    if replay_rows != expected_replay_rows:
                        raise RuntimeError(
                            "device success-only selector differs from terminal ledger"
                        )
                    for row in ledger_rows:
                        row["demo_replay_selected"] = bool(row["success"])
                    selection_mode = "terminal_success_full_lane_device_filter_v1"
                else:
                    replay_rows = _add_rollout_to_replay(demos, demo_rollout)
                    expected_replay_rows = sum(
                        int(row["valid_steps"]) for row in ledger_rows
                    )
                    for row in ledger_rows:
                        row["demo_replay_selected"] = True
                    selection_mode = "all_lanes_with_validity_mask_v1"
                demo_replay_inserted_rows += replay_rows
                demo_replay_valid_steps += expected_replay_rows
                control_plane["cohort"] = demo_cohort
                control_plane["replay_selection"] = {
                    "mode": selection_mode,
                    "selector_device": str(demo_rollout.valid.device),
                    "selector_host_materializations": 0,
                    "attempted_lanes": len(ledger_rows),
                    "selected_lanes": (
                        len(successful_rows)
                        if config.demo_success_only_replay
                        else len(ledger_rows)
                    ),
                    "inserted_rows": replay_rows,
                    "selected_valid_rows": expected_replay_rows,
                }
                demo_control_plane.append(control_plane)
                cohort_rows.append(summary)
                _append_jsonl_rows(ledger_path, ledger_rows)
                demo_allocated_steps += summary["allocated_steps"]
                demo_valid_steps += summary["valid_steps"]
                demo_successes += summary["terminal_successes"]
                demo_episodes += len(reset.episode_ids)
                total_allocated_steps += summary["allocated_steps"]
                total_valid_steps += summary["valid_steps"]
                total_successes += summary["terminal_successes"]
                completed_demo_after += 1
                if demo_cohort + 1 < config.demo_cohorts:
                    next_reset = env.reset()
            demo_success_rate = demo_successes / demo_episodes
            expected_demo_episodes = config.demo_cohorts * config.num_envs
            attempt_coverage_passed = (
                demo_episodes == expected_demo_episodes
                and len(seen_episode_ids) == expected_demo_episodes
                and len(demo_attempt_manifest_ordinals) == expected_demo_episodes
                and len(set(demo_attempt_manifest_ordinals)) == expected_demo_episodes
            )
            replay_retains_all_selected_rows = (
                demos.inserted_rows == demo_replay_inserted_rows
                and demos.size == demo_replay_inserted_rows
            )
            if config.demo_success_only_replay:
                if len(successful_demo_episode_ids) != demo_successes:
                    raise RuntimeError(
                        "successful demo identity count differs from terminal ledger"
                    )
                demo_gate_passed = (
                    demo_successes >= config.minimum_qualified_demo_episodes
                    and attempt_coverage_passed
                    and replay_retains_all_selected_rows
                    and demo_replay_inserted_rows == demo_replay_valid_steps
                )
                gate_mode = "minimum_qualified_episode_count_v1"
                replay_selected_episodes = demo_successes
            else:
                demo_gate_passed = (
                    config.demo_policy == "zero_action"
                    or demo_success_rate >= config.minimum_demo_success_rate
                )
                gate_mode = (
                    "mechanical_seed_v1"
                    if config.demo_policy == "zero_action"
                    else "minimum_terminal_success_rate_v1"
                )
                replay_selected_episodes = demo_episodes
            demo_quality = {
                "schema_version": DEMO_QUALITY_SCHEMA,
                "demo_policy": config.demo_policy,
                "producer": DEMO_PRODUCERS[config.demo_policy],
                "teacher_overrides": config.demo_teacher_overrides,
                "teacher_overrides_sha256": config.demo_teacher_overrides_sha256,
                "demo_cohorts": config.demo_cohorts,
                "episodes": demo_episodes,
                "terminal_successes": demo_successes,
                "success_rate": demo_success_rate,
                "minimum_success_rate": config.minimum_demo_success_rate,
                "success_only_replay": config.demo_success_only_replay,
                "gate_mode": gate_mode,
                "minimum_qualified_episodes": (config.minimum_qualified_demo_episodes),
                "allocated_steps": demo_allocated_steps,
                "valid_steps": demo_valid_steps,
                "successful_episode_ids": successful_demo_episode_ids,
                "attempt_coverage": {
                    "manifest_sha256": demo_manifest_sha256,
                    "expected_episodes": expected_demo_episodes,
                    "observed_episodes": demo_episodes,
                    "unique_episode_ids": len(seen_episode_ids),
                    "manifest_ordinals": demo_attempt_manifest_ordinals,
                    "successful_manifest_ordinals": (successful_demo_manifest_ordinals),
                    "passed": attempt_coverage_passed,
                },
                "replay_selection": {
                    "mode": selection_mode,
                    "device_filter": config.demo_success_only_replay,
                    "selected_episodes": replay_selected_episodes,
                    "excluded_attempts": demo_episodes - replay_selected_episodes,
                    "inserted_rows": demo_replay_inserted_rows,
                    "selected_valid_rows": demo_replay_valid_steps,
                    "capacity": demos.capacity,
                    "retains_all_selected_rows": replay_retains_all_selected_rows,
                    "failed_attempt_transitions_inserted": (
                        0 if config.demo_success_only_replay else None
                    ),
                },
                "quality_qualified": (
                    config.demo_policy == "privileged_teacher" and demo_gate_passed
                ),
                "gate_passed": demo_gate_passed,
                "control_plane": demo_control_plane,
            }
            _atomic_json(demo_quality_path, demo_quality)
            if not demo_gate_passed:
                failure = (
                    "privileged-teacher success-only demonstration bank failed its "
                    "predeclared count, coverage, or replay-retention gate"
                    if config.demo_success_only_replay
                    else "privileged-teacher demonstration success rate failed its "
                    "predeclared quality gate"
                )
                raise RuntimeError(failure)
            next_reset = env.reset()
        else:
            demo_quality = {
                "schema_version": DEMO_QUALITY_SCHEMA,
                "demo_policy": None,
                "producer": None,
                "teacher_overrides": {},
                "teacher_overrides_sha256": None,
                "demo_cohorts": 0,
                "episodes": 0,
                "terminal_successes": 0,
                "success_rate": None,
                "minimum_success_rate": 0.0,
                "success_only_replay": False,
                "gate_mode": "not_applicable",
                "minimum_qualified_episodes": 0,
                "allocated_steps": 0,
                "valid_steps": 0,
                "successful_episode_ids": [],
                "attempt_coverage": {
                    "manifest_sha256": None,
                    "expected_episodes": 0,
                    "observed_episodes": 0,
                    "unique_episode_ids": 0,
                    "manifest_ordinals": [],
                    "successful_manifest_ordinals": [],
                    "passed": True,
                },
                "replay_selection": {
                    "mode": "not_applicable",
                    "device_filter": False,
                    "selected_episodes": 0,
                    "excluded_attempts": 0,
                    "inserted_rows": 0,
                    "selected_valid_rows": 0,
                    "capacity": demos.capacity,
                    "retains_all_selected_rows": True,
                    "failed_attempt_transitions_inserted": None,
                },
                "quality_qualified": False,
                "gate_passed": True,
                "control_plane": [],
            }
            _atomic_json(demo_quality_path, demo_quality)

        if restored is None:
            actor_bc_pretrain = _actor_bc_pretrain(
                config=config,
                actor=actor,
                actor_optimizer=actor_optimizer,
                demos=demos,
            )

        dagger_correction_path = args.output / "dagger_correction.json"
        completed_dagger_after = completed_dagger_before
        if restored is not None:
            dagger_correction = dict(restored["dagger_correction"])
            _atomic_json(dagger_correction_path, dagger_correction)
        elif config.dagger_cohorts > 0:
            dagger_allocated_steps = 0
            dagger_valid_steps = 0
            dagger_successes = 0
            dagger_episodes = 0
            dagger_replay_inserted_rows = 0
            dagger_bc_update_steps = 0
            dagger_control_plane = []
            dagger_bc_rows = []
            for dagger_cohort in range(config.dagger_cohorts):
                reset = next_reset if dagger_cohort == 0 else env.reset()
                (
                    dagger_online_rollout,
                    dagger_label_rollout,
                    rollout_seconds,
                    control_plane,
                ) = _rollout_dagger_correction_cohort(
                    env=env,
                    actor=actor,
                    online_buffer=transition,
                    correction_buffer=correction_transition,
                    reset=reset,
                    teacher_overrides=config.demo_teacher_overrides,
                )
                _add_rollout_to_replay(online, dagger_online_rollout)
                correction_rows = _add_correction_rollout_to_replay(
                    corrections,
                    dagger_label_rollout,
                )
                actor.train()
                update_started = time.perf_counter()
                bc_evidence = _dagger_bc_updates(
                    config=config,
                    actor=actor,
                    actor_optimizer=actor_optimizer,
                    demos=demos,
                    corrections=corrections,
                )
                update_seconds = time.perf_counter() - update_started
                summary, ledger_rows = _cohort_evidence(
                    config=config,
                    env=env,
                    reset=reset,
                    rollout=dagger_online_rollout,
                    role="dagger_learner_correction",
                    cohort=dagger_cohort,
                    rollout_seconds=rollout_seconds,
                    update_seconds=update_seconds,
                    updates=int(bc_evidence["completed_updates"]),
                    metrics={
                        "dagger_bc_first_loss": float(bc_evidence["first_loss"]),
                        "dagger_bc_last_loss": float(bc_evidence["last_loss"]),
                        "dagger_bc_mean_loss": float(bc_evidence["mean_loss"]),
                    },
                    seen_episode_ids=seen_episode_ids,
                )
                expected_correction_rows = sum(
                    int(row["valid_steps"]) for row in ledger_rows
                )
                if correction_rows != expected_correction_rows:
                    raise RuntimeError(
                        "DAgger correction selector differs from learner terminal ledger"
                    )
                for row in ledger_rows:
                    row["demo_replay_selected"] = False
                    row["dagger_correction_selected"] = True
                    row["dagger_correction_rows"] = int(row["valid_steps"])
                    row["executed_action_source"] = "deterministic_gpu_actor"
                    row["correction_label_source"] = DEMO_PRODUCERS[
                        "privileged_teacher"
                    ]
                control_plane["cohort"] = dagger_cohort
                control_plane["replay_selection"] = {
                    "mode": "all_valid_learner_visited_rows_device_filter_v1",
                    "selector_device": str(dagger_label_rollout.valid.device),
                    "selector_host_materializations": 0,
                    "attempted_lanes": len(ledger_rows),
                    "inserted_rows": correction_rows,
                    "selected_valid_rows": expected_correction_rows,
                    "successful_demo_rows": 0,
                }
                dagger_control_plane.append(control_plane)
                dagger_bc_rows.append({"cohort": dagger_cohort, **bc_evidence})
                cohort_rows.append(summary)
                _append_jsonl_rows(ledger_path, ledger_rows)
                dagger_allocated_steps += summary["allocated_steps"]
                dagger_valid_steps += summary["valid_steps"]
                dagger_successes += summary["terminal_successes"]
                dagger_episodes += len(reset.episode_ids)
                dagger_replay_inserted_rows += correction_rows
                dagger_bc_update_steps += int(bc_evidence["completed_updates"])
                total_allocated_steps += summary["allocated_steps"]
                total_valid_steps += summary["valid_steps"]
                total_successes += summary["terminal_successes"]
                completed_dagger_after += 1

            correction_replay_retained = (
                corrections.inserted_rows == dagger_replay_inserted_rows
                and corrections.size == dagger_replay_inserted_rows
                and dagger_replay_inserted_rows == dagger_valid_steps
            )
            if not correction_replay_retained:
                raise RuntimeError(
                    "DAgger correction replay did not retain every labelled valid row"
                )
            dagger_correction = {
                "schema_version": DAGGER_CORRECTION_SCHEMA,
                "enabled": True,
                "configured_cohorts": config.dagger_cohorts,
                "completed_cohorts": completed_dagger_after,
                "episodes": dagger_episodes,
                "terminal_successes": dagger_successes,
                "allocated_steps": dagger_allocated_steps,
                "valid_steps": dagger_valid_steps,
                "executed_action_source": "deterministic_gpu_actor",
                "label_action_source": DEMO_PRODUCERS["privileged_teacher"],
                "labels_are_successful_demonstrations": False,
                "teacher_overrides": config.demo_teacher_overrides,
                "teacher_overrides_sha256": config.demo_teacher_overrides_sha256,
                "replay": {
                    "semantics": "learner_visited_state_teacher_action_labels_v1",
                    "inserted_rows": dagger_replay_inserted_rows,
                    "valid_rows": dagger_valid_steps,
                    "capacity": corrections.capacity,
                    "retains_all_labelled_rows": correction_replay_retained,
                    "inserted_into_successful_demo_replay": False,
                },
                "bc_updates_per_cohort": config.dagger_bc_updates_per_cohort,
                "bc_updates": dagger_bc_update_steps,
                "correction_ratio": config.dagger_correction_ratio,
                "bc_cohorts": dagger_bc_rows,
                "control_plane": dagger_control_plane,
            }
            _atomic_json(dagger_correction_path, dagger_correction)
            next_reset = env.reset()
        else:
            dagger_correction = {
                "schema_version": DAGGER_CORRECTION_SCHEMA,
                "enabled": False,
                "configured_cohorts": 0,
                "completed_cohorts": 0,
                "episodes": 0,
                "terminal_successes": 0,
                "allocated_steps": 0,
                "valid_steps": 0,
                "executed_action_source": None,
                "label_action_source": None,
                "labels_are_successful_demonstrations": False,
                "teacher_overrides": {},
                "teacher_overrides_sha256": None,
                "replay": {
                    "semantics": "disabled",
                    "inserted_rows": 0,
                    "valid_rows": 0,
                    "capacity": corrections.capacity,
                    "retains_all_labelled_rows": True,
                    "inserted_into_successful_demo_replay": False,
                },
                "bc_updates_per_cohort": 0,
                "bc_updates": 0,
                "correction_ratio": config.dagger_correction_ratio,
                "bc_cohorts": [],
                "control_plane": [],
            }
            _atomic_json(dagger_correction_path, dagger_correction)

        for local_cohort in range(config.cohorts):
            global_cohort = completed_online_before + local_cohort
            reset = next_reset if local_cohort == 0 else env.reset()
            online_rollout, rollout_seconds = _rollout_cohort(
                env=env,
                actor=actor,
                buffer=transition,
                reset=reset,
                mode="online",
            )
            _add_rollout_to_replay(online, online_rollout)
            actor.train()
            critic.train()
            update_started = time.perf_counter()
            metrics, updates = _offpolicy_updates(
                config=config,
                actor=actor,
                critic=critic,
                target_critic=target_critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                log_alpha=log_alpha,
                alpha_optimizer=alpha_optimizer,
                online=online,
                demos=demos,
                corrections=corrections,
            )
            update_seconds = time.perf_counter() - update_started
            update_steps += updates
            invocation_update_steps += updates
            summary, ledger_rows = _cohort_evidence(
                config=config,
                env=env,
                reset=reset,
                rollout=online_rollout,
                role="online",
                cohort=global_cohort,
                rollout_seconds=rollout_seconds,
                update_seconds=update_seconds,
                updates=updates,
                metrics=metrics,
                seen_episode_ids=seen_episode_ids,
            )
            cohort_rows.append(summary)
            _append_jsonl_rows(ledger_path, ledger_rows)
            total_allocated_steps += summary["allocated_steps"]
            total_valid_steps += summary["valid_steps"]
            total_successes += summary["terminal_successes"]

        torch.cuda.synchronize(env.device)
        wall_seconds = time.perf_counter() - wall_started
        parameter_sha256_end = _parameter_sha256(
            actor, critic, target_critic, log_alpha
        )
        if invocation_update_steps < 1:
            raise RuntimeError(
                f"{config.algorithm} smoke completed no optimizer updates"
            )
        if parameter_sha256_end == parameter_sha256_start:
            raise RuntimeError("off-policy optimizer updates did not change parameters")

        checkpoint_path = args.output / "checkpoint_latest.pt"
        completed_online_after = completed_online_before + config.cohorts
        checkpoint_payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "config": asdict(config),
            "cohort_horizon_steps": env.cohort_horizon_steps,
            "observation_dim": observation_dim,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": target_critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "log_alpha": log_alpha.detach(),
            "alpha_optimizer": alpha_optimizer.state_dict(),
            "online_replay": online.state_dict(),
            "demo_replay": demos.state_dict(),
            "correction_replay": corrections.state_dict(),
            "manifest_cursor": dict(env.manifest_state_dict()),
            "completed_online_cohorts": completed_online_after,
            "completed_demo_cohorts": completed_demo_after,
            "completed_dagger_cohorts": completed_dagger_after,
            "demo_quality": demo_quality,
            "dagger_correction": dagger_correction,
            "actor_bc_pretrain": actor_bc_pretrain,
            "update_steps": update_steps,
            "rng_state": _capture_rng_state(),
            "parameter_sha256": parameter_sha256_end,
        }
        checkpoint_payload["training_state_sha256"] = _checkpoint_state_sha256(
            checkpoint_payload
        )
        _atomic_torch_save(checkpoint_path, checkpoint_payload)
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_reload_checks = {
            "schema_match": reloaded.get("schema_version") == CHECKPOINT_SCHEMA,
            "completed_online_cohorts_match": int(
                reloaded.get("completed_online_cohorts", -1)
            )
            == completed_online_after,
            "completed_demo_cohorts_match": int(
                reloaded.get("completed_demo_cohorts", -1)
            )
            == completed_demo_after,
            "completed_dagger_cohorts_match": int(
                reloaded.get("completed_dagger_cohorts", -1)
            )
            == completed_dagger_after,
            "demo_quality_match": reloaded.get("demo_quality") == demo_quality,
            "dagger_correction_match": reloaded.get("dagger_correction")
            == dagger_correction,
            "actor_bc_pretrain_match": reloaded.get("actor_bc_pretrain")
            == actor_bc_pretrain,
            "update_steps_match": int(reloaded.get("update_steps", -1)) == update_steps,
            "manifest_cursor_match": reloaded.get("manifest_cursor")
            == checkpoint_payload["manifest_cursor"],
            "parameter_sha256_match": reloaded.get("parameter_sha256")
            == parameter_sha256_end,
            "training_state_sha256_match": reloaded.get("training_state_sha256")
            == _checkpoint_state_sha256(reloaded),
        }
        if not all(checkpoint_reload_checks.values()):
            raise RuntimeError(
                f"checkpoint reload validation failed: {checkpoint_reload_checks}"
            )

        provenance = env.provenance
        transport_receipt = env.last_transport_receipt
        if transport_receipt is None or env.transport_checks < 1:
            raise RuntimeError("tensor off-policy smoke produced no transport receipt")
        source_provenance = {
            "sources": provenance_bundle["sources"],
            "runtime": {
                "path": provenance_bundle["runtime"]["path"],
                "sha256": provenance_bundle["runtime"]["sha256"],
                "versions": provenance_bundle["runtime"]["payload"]["versions"],
            },
        }
        online_valid_rows = int(
            checkpoint_payload["online_replay"]["data"]["valid"].sum()
        )
        demo_valid_rows = int(checkpoint_payload["demo_replay"]["data"]["valid"].sum())
        correction_valid_rows = int(checkpoint_payload["correction_replay"]["size"])
        if config.demo_success_only_replay and (
            demo_valid_rows
            != int(demo_quality["replay_selection"]["selected_valid_rows"])
        ):
            raise RuntimeError(
                "checkpointed demo replay differs from success-only selection evidence"
            )
        if correction_valid_rows != int(dagger_correction["replay"]["valid_rows"]):
            raise RuntimeError(
                "checkpointed correction replay differs from DAgger selection evidence"
            )
        demo_control_plane = list(demo_quality["control_plane"])
        demo_observation_audits = sum(
            int(row["observation_audit_calls"]) for row in demo_control_plane
        )
        demo_terminal_mask_reads = sum(
            int(row["terminal_mask_host_materializations"])
            for row in demo_control_plane
        )
        demo_action_transfers = sum(
            int(row["host_to_device_action_transfers"]) for row in demo_control_plane
        )
        dagger_control_plane = list(dagger_correction["control_plane"])
        dagger_observation_audits = sum(
            int(row["observation_audit_calls"]) for row in dagger_control_plane
        )
        dagger_terminal_mask_reads = sum(
            int(row["terminal_mask_host_materializations"])
            for row in dagger_control_plane
        )
        dagger_label_transfers = sum(
            int(row["host_to_device_label_transfers"]) for row in dagger_control_plane
        )
        if (
            restored is None
            and env.teacher_audit_materializations
            != demo_observation_audits + dagger_observation_audits
        ):
            raise RuntimeError(
                "backend teacher-audit counter differs from demo/correction evidence"
            )
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "passed",
            "config": asdict(config),
            "claim_scope": {
                "algorithm": config.algorithm,
                "device_tensor_training_smoke": True,
                "quality_qualified": False,
                "rlpd_demo_quality_qualified": bool(demo_quality["quality_qualified"]),
                "rlpd_demo_producer": demo_quality["producer"],
                "actor_bc_pretrain_enabled": bool(actor_bc_pretrain["enabled"]),
                "actor_bc_regularization_enabled": config.actor_bc_weight > 0.0,
                "actor_sac_weight": config.actor_sac_weight,
                "dagger_correction_enabled": bool(dagger_correction["enabled"]),
                "dagger_labels_are_successful_demonstrations": False,
                "statement": (
                    (
                        "RLPD demo replay contains only complete terminal-success "
                        "GPU lanes and passed its predeclared qualified-count, "
                        "manifest-coverage, and retention gates; learned-policy "
                        "quality still requires separate held-out evaluation."
                        if demo_quality["success_only_replay"]
                        else "RLPD demo data passed its predeclared current-GPU-state "
                        "privileged-teacher success-rate gate; learned-policy quality "
                        "still requires separate held-out evaluation."
                    )
                    if demo_quality["quality_qualified"]
                    else (
                        "RLPD demo data is a mechanical real-environment seed used "
                        "only to validate mixed replay; no expert-quality claim is made."
                        if config.algorithm == "rlpd"
                        else "SAC smoke validates mechanics and resume, not policy quality."
                    )
                ),
            },
            "source_provenance": source_provenance,
            "inventory": inventory,
            "backend": {
                "backend_id": provenance.backend_id,
                "api_version": env.api_version,
                "implementation_version": provenance.implementation_version,
                "device_name": provenance.device_name,
                "device_ordinal": provenance.device_ordinal,
                "physical_device_uuid": provenance.physical_device_uuid,
                "physical_device_pci_bus_id": provenance.physical_device_pci_bus_id,
                "physical_device_identity_source": provenance.physical_device_identity_source,
                "cross_runtime_device_identity": asdict(env.device_identity),
                "runtime_versions": dict(provenance.runtime_versions),
                "model_sha256": provenance.model_sha256,
                "config_sha256": provenance.config_sha256,
            },
            "data_plane": {
                "action_transport": "torch_cuda_tensor_direct_to_warp",
                "output_transport": "pointer_identical_warp_to_torch_views",
                "action_pointer_identity_verified": True,
                "output_pointer_identity_verified": True,
                "torch_warp_stream_identity_verified": True,
                "transport_checks": env.transport_checks,
                "last_transport_receipt": {
                    "action_input_ptr": transport_receipt["action_input_ptr"],
                    "action_engine_ptr": transport_receipt["action_engine_ptr"],
                    "torch_stream_ptr": transport_receipt["torch_stream_ptr"],
                    "warp_stream_ptr": transport_receipt["warp_stream_ptr"],
                    "output_ptrs": dict(transport_receipt["output_ptrs"]),
                },
                "reset_policy": "canonical_fixed_horizon_full_cohort",
                "post_terminal_policy": (
                    "device_semantic_inactive_mask_plus_replay_validity_mask; "
                    "physical_world_allocation_not_compacted"
                ),
                "replay_sampling": "cuda_multinomial_with_cuda_generator",
                "online_hot_path_host_materializations": 0,
                "demonstration_control_plane_host_materializations": (
                    demo_observation_audits + demo_terminal_mask_reads
                ),
                "demonstration_observation_audit_calls": demo_observation_audits,
                "demonstration_terminal_mask_host_materializations": (
                    demo_terminal_mask_reads
                ),
                "demonstration_host_to_device_action_transfers": demo_action_transfers,
                "demonstration_replay_selection": demo_quality["replay_selection"],
                "demonstration_selector_host_materializations": 0,
                "dagger_control_plane_host_materializations": (
                    dagger_observation_audits + dagger_terminal_mask_reads
                ),
                "dagger_observation_audit_calls": dagger_observation_audits,
                "dagger_terminal_mask_host_materializations": (
                    dagger_terminal_mask_reads
                ),
                "dagger_host_to_device_label_transfers": dagger_label_transfers,
                "dagger_correction_selector_host_materializations": 0,
                "dagger_executed_policy_device_only": bool(
                    dagger_correction["enabled"]
                ),
                "actor_bc_pretrain_device_only": True,
                "actor_bc_regularization_device_only": True,
                "dagger_bc_updates_device_only": bool(dagger_correction["enabled"]),
                "cohort_horizon_steps": env.cohort_horizon_steps,
                "warmup_policy_only_steps": config.warmup_steps,
            },
            "replay": {
                "capacity_each": config.replay_capacity,
                "correction_capacity": corrections.capacity,
                "online_size": online.size,
                "online_valid_rows": online_valid_rows,
                "online_inserted_rows": online.inserted_rows,
                "demo_size": demos.size,
                "demo_valid_rows": demo_valid_rows,
                "demo_inserted_rows": demos.inserted_rows,
                "demo_success_only": bool(demo_quality["success_only_replay"]),
                "demo_selected_episodes": int(
                    demo_quality["replay_selection"]["selected_episodes"]
                ),
                "demo_excluded_attempts": int(
                    demo_quality["replay_selection"]["excluded_attempts"]
                ),
                "demo_ratio": config.demo_ratio,
                "actor_bc_weight": config.actor_bc_weight,
                "actor_sac_weight": config.actor_sac_weight,
                "correction_size": corrections.size,
                "correction_valid_rows": correction_valid_rows,
                "correction_inserted_rows": corrections.inserted_rows,
                "correction_labels_are_successful_demonstrations": False,
                "dagger_correction_ratio": config.dagger_correction_ratio,
                "sampling_rng_checkpointed": True,
            },
            "demo_quality": {
                **demo_quality,
                "path": str(demo_quality_path),
                "sha256": _file_sha256(demo_quality_path),
            },
            "dagger_correction": {
                **dagger_correction,
                "path": str(dagger_correction_path),
                "sha256": _file_sha256(dagger_correction_path),
            },
            "train": {
                "cohorts": cohort_rows,
                "completed_online_cohorts_before": completed_online_before,
                "completed_online_cohorts_after": completed_online_after,
                "completed_demo_cohorts_before": completed_demo_before,
                "completed_demo_cohorts_after": completed_demo_after,
                "completed_dagger_cohorts_before": completed_dagger_before,
                "completed_dagger_cohorts_after": completed_dagger_after,
                "total_allocated_steps": total_allocated_steps,
                "total_valid_steps": total_valid_steps,
                "terminal_successes": total_successes,
                "optimizer_updates_before": update_steps_before,
                "optimizer_updates_this_invocation": invocation_update_steps,
                "optimizer_updates_after": update_steps,
                "actor_bc_pretrain": actor_bc_pretrain,
                "actor_bc_weight": config.actor_bc_weight,
                "actor_sac_weight": config.actor_sac_weight,
                "dagger_bc_updates": int(dagger_correction["bc_updates"]),
                "parameter_sha256_start": parameter_sha256_start,
                "parameter_sha256_end": parameter_sha256_end,
                "parameters_changed": parameter_sha256_start != parameter_sha256_end,
                "wall_seconds": wall_seconds,
                "allocated_env_steps_per_s": total_allocated_steps / wall_seconds,
                "valid_env_steps_per_s": total_valid_steps / wall_seconds,
            },
            "resume": {
                "resumed": restored is not None,
                "source_path": str(resume_path) if resume_path is not None else None,
                "source_sha256": resume_sha256,
                "source_training_state_sha256": (
                    restored["training_state_sha256"] if restored is not None else None
                ),
                "restored_manifest_cursor": restored_manifest_cursor,
                "parameter_sha256_matches_source": (
                    parameter_sha256_start == restored["parameter_sha256"]
                    if restored is not None
                    else None
                ),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _file_sha256(checkpoint_path),
                "schema_version": checkpoint_payload["schema_version"],
                "training_state_sha256": checkpoint_payload["training_state_sha256"],
                "reload_checks": checkpoint_reload_checks,
            },
            "episode_ledger": {
                "path": str(ledger_path),
                "sha256": _file_sha256(ledger_path),
                "rows": len(seen_episode_ids),
                "unique_episode_ids": len(seen_episode_ids),
                "manifest_sha256": env.manifest_sha256,
                "manifest_cursor": dict(env.manifest_state_dict()),
            },
        }
        _atomic_json(args.output / "report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
