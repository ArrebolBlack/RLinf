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
Host materialization occurs only after synchronized cohort/update boundaries for
evidence and checkpointing.  The RLPD demonstration cohort is deliberately a
mechanical zero-action seed: it validates the mixed-replay infrastructure but is
not a quality-qualified expert demonstration.
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

from rlinf.data.device_replay_buffer import DeviceReplayBatch, DeviceReplayBuffer
from rlinf.data.device_transition_buffer import DeviceTransitionBuffer
from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import GpuNativeTensorBackendEnv

CHECKPOINT_SCHEMA = "rlinf-gpuenv0-tensor-offpolicy-smoke-v0.1"
REPORT_SCHEMA = "rlinf-gpuenv0-tensor-offpolicy-smoke-report-v0.1"
DEMO_PRODUCER = "zero_action_device_cohort_v1"


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


def _config(args: argparse.Namespace) -> TensorOffPolicyConfig:
    demo_ratio = float(args.demo_ratio) if args.algorithm == "rlpd" else 0.0
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
    if config.replay_capacity < config.batch_size:
        raise ValueError("replay_capacity must be at least batch_size")
    if not 0.0 <= config.demo_ratio <= 1.0:
        raise ValueError("demo_ratio must be in [0, 1]")
    if config.algorithm == "rlpd" and not 0.0 < config.demo_ratio < 1.0:
        raise ValueError("RLPD smoke requires demo_ratio strictly between zero and one")
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
) -> tuple[dict[str, float], int]:
    metric_rows = []
    for _update in range(config.updates_per_cohort):
        batch = _mixed_batch(
            online,
            demos,
            batch_size=config.batch_size,
            demo_ratio=config.demo_ratio,
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
        actor_loss = (log_alpha.exp().detach() * log_probability - policy_q).mean()
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
        "manifest_cursor",
        "completed_online_cohorts",
        "completed_demo_cohorts",
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
    if completed_demo != (1 if config.algorithm == "rlpd" else 0):
        raise ValueError(
            "resume demonstration-cohort counter violates algorithm contract"
        )
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
    observation = reset.observation
    zero_action = torch.zeros((env.num_envs, 7), dtype=torch.float32, device=env.device)
    buffer.reset_cohort()
    actor.eval()
    torch.cuda.synchronize(env.device)
    started = time.perf_counter()
    for _step in range(env.cohort_horizon_steps):
        with torch.inference_mode():
            if mode == "demo":
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


def _add_rollout_to_replay(replay: DeviceReplayBuffer, rollout: Any) -> None:
    horizon, num_envs = rollout.valid.shape
    rows = horizon * num_envs
    replay.add_batch(
        observation=rollout.observation.reshape(rows, -1),
        action=rollout.action.reshape(rows, 7),
        reward=rollout.reward.reshape(rows),
        next_observation=rollout.next_observation.reshape(rows, -1),
        terminated=rollout.terminated.reshape(rows),
        truncated=rollout.truncated.reshape(rows),
        valid=rollout.valid.reshape(rows),
    )


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
            update_steps_before = int(restored["update_steps"])
        initial = env.reset()
        observation = initial.observation
        observation_dim = int(observation.shape[1])
        if restored is not None and int(restored["observation_dim"]) != observation_dim:
            raise ValueError(
                "resume observation dimension does not match live environment"
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
        transition = _new_transition_buffer(env, observation_dim, observation.dtype)

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
        if config.algorithm == "rlpd" and restored is None:
            demo_rollout, demo_rollout_seconds = _rollout_cohort(
                env=env,
                actor=actor,
                buffer=transition,
                reset=next_reset,
                mode="demo",
            )
            _add_rollout_to_replay(demos, demo_rollout)
            summary, ledger_rows = _cohort_evidence(
                config=config,
                env=env,
                reset=next_reset,
                rollout=demo_rollout,
                role="mechanical_demo",
                cohort=0,
                rollout_seconds=demo_rollout_seconds,
                update_seconds=0.0,
                updates=0,
                metrics={},
                seen_episode_ids=seen_episode_ids,
            )
            cohort_rows.append(summary)
            _append_jsonl_rows(ledger_path, ledger_rows)
            total_allocated_steps += summary["allocated_steps"]
            total_valid_steps += summary["valid_steps"]
            total_successes += summary["terminal_successes"]
            completed_demo_after = 1
            next_reset = env.reset()

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
            "manifest_cursor": dict(env.manifest_state_dict()),
            "completed_online_cohorts": completed_online_after,
            "completed_demo_cohorts": completed_demo_after,
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
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "passed",
            "config": asdict(config),
            "claim_scope": {
                "algorithm": config.algorithm,
                "device_tensor_training_smoke": True,
                "quality_qualified": False,
                "rlpd_demo_quality_qualified": False,
                "rlpd_demo_producer": DEMO_PRODUCER
                if config.algorithm == "rlpd"
                else None,
                "statement": (
                    "RLPD demo data is a mechanical real-environment seed used only "
                    "to validate mixed replay; no expert-quality claim is made."
                    if config.algorithm == "rlpd"
                    else "SAC smoke validates mechanics and resume, not policy quality."
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
                "hot_path_host_materializations": 0,
                "cohort_horizon_steps": env.cohort_horizon_steps,
                "warmup_policy_only_steps": config.warmup_steps,
            },
            "replay": {
                "capacity_each": config.replay_capacity,
                "online_size": online.size,
                "online_valid_rows": online_valid_rows,
                "online_inserted_rows": online.inserted_rows,
                "demo_size": demos.size,
                "demo_valid_rows": demo_valid_rows,
                "demo_inserted_rows": demos.inserted_rows,
                "demo_ratio": config.demo_ratio,
                "sampling_rng_checkpointed": True,
            },
            "train": {
                "cohorts": cohort_rows,
                "completed_online_cohorts_before": completed_online_before,
                "completed_online_cohorts_after": completed_online_after,
                "completed_demo_cohorts_before": completed_demo_before,
                "completed_demo_cohorts_after": completed_demo_after,
                "total_allocated_steps": total_allocated_steps,
                "total_valid_steps": total_valid_steps,
                "terminal_successes": total_successes,
                "optimizer_updates_before": update_steps_before,
                "optimizer_updates_this_invocation": invocation_update_steps,
                "optimizer_updates_after": update_steps,
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
