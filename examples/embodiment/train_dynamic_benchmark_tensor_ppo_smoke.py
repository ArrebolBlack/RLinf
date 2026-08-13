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
Checkpoints are written only at cohort boundaries and contain the model,
optimizer, RNG streams, and manifest cursor required for strict cross-process
continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
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
    split: str
    manifest_seed: int
    manifest_size: int
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
    se3_source: str
    se3_commit: str
    se3_tree: str
    rlinf_source: str
    rlinf_commit: str
    rlinf_tree: str
    runtime_manifest: str
    runtime_manifest_sha256: str
    expected_cpuset: str


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

    def distribution_and_value(
        self, observation: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        hidden = self.backbone(observation)
        return Normal(self.actor(hidden), self.log_std.exp()), self.critic(
            hidden
        ).squeeze(-1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="p0_grasp")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest-seed", type=int, default=20261050)
    parser.add_argument("--manifest-size", type=int, default=4096)
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


def _config(args: argparse.Namespace) -> TensorPPOConfig:
    config = TensorPPOConfig(
        task=args.task,
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
        "ppo_epochs",
        "minibatch_size",
        "manifest_size",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.manifest_seed < 0:
        raise ValueError("manifest_seed must be non-negative")
    if config.manifest_size < config.num_envs:
        raise ValueError("manifest_size must be at least num_envs")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if not 0.0 < config.clip_coef < 1.0:
        raise ValueError("clip_coef must be in (0, 1)")
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
            delta + gamma * gae_lambda * (~done[step]).to(reward.dtype) * accumulator
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
            distribution, predicted_value = model.distribution_and_value(
                selected_observation
            )
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
            value_loss = (
                0.5
                * (predicted_value - returns.index_select(0, indices)).square().mean()
            )
            entropy = distribution.entropy().sum(dim=-1).mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
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


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Return a stable digest of tensor names, schemas, and values."""

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_sha256(model: nn.Module) -> str:
    return _state_dict_sha256(model.state_dict())


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
        raise ValueError(
            "resume CUDA RNG device count mismatch: "
            f"checkpoint={len(cuda_states)}, runtime={torch.cuda.device_count()}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(cuda_states)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_resume_checkpoint(
    path: Path, config: TensorPPOConfig
) -> tuple[Path, str, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    sha256 = _file_sha256(resolved)
    restored = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(restored, dict):
        raise ValueError("resume checkpoint payload is not a mapping")
    if restored.get("schema_version") != "rlinf-gpuenv0-tensor-ppo-smoke-v0.2":
        raise ValueError("resume checkpoint schema mismatch")
    if restored.get("config") != asdict(config):
        raise ValueError("resume checkpoint config identity does not match current run")
    required = {
        "cohort_horizon_steps",
        "observation_dim",
        "model",
        "optimizer",
        "manifest_cursor",
        "completed_cohorts",
        "update_steps",
        "rng_state",
        "parameter_sha256",
    }
    missing = sorted(required - set(restored))
    if missing:
        raise ValueError(f"resume checkpoint is missing required fields: {missing}")
    if int(restored["completed_cohorts"]) < 1 or int(restored["update_steps"]) < 1:
        raise ValueError("resume checkpoint does not contain completed training")
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


def _preflight(config: TensorPPOConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    import se3_wam

    import rlinf

    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("tensor PPO gate requires Linux CPU-affinity introspection")
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
            (str(expected[name]) for name in expected_names if name in expected),
            None,
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


def main() -> int:
    args = _parser().parse_args()
    config = _config(args)
    provenance_bundle, inventory = _preflight(config)
    if not torch.cuda.is_available():
        raise RuntimeError("tensor PPO smoke requires CUDA; CPU fallback is forbidden")
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
        runtime_manifest=config.runtime_manifest,
        runtime_manifest_sha256=config.runtime_manifest_sha256,
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
        completed_cohorts_before = 0
        update_steps_before = 0
        restored_manifest_cursor: dict[str, Any] | None = None
        if restored is not None:
            if int(restored["cohort_horizon_steps"]) != env.cohort_horizon_steps:
                raise ValueError(
                    "resume cohort horizon does not match the live environment"
                )
            restored_manifest_cursor = dict(restored["manifest_cursor"])
            env.load_manifest_state_dict(restored_manifest_cursor)
            completed_cohorts_before = int(restored["completed_cohorts"])
            update_steps_before = int(restored["update_steps"])
        initial = env.reset()
        observation = initial.observation
        observation_dim = int(observation.shape[1])
        if restored is not None and int(restored["observation_dim"]) != observation_dim:
            raise ValueError(
                "resume observation dimension does not match the live environment"
            )
        model = TensorActorCritic(observation_dim, config.hidden_size).to(env.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        if restored is not None:
            model.load_state_dict(restored["model"], strict=True)
            optimizer.load_state_dict(restored["optimizer"])
            parameter_sha256_start = _model_sha256(model)
            if parameter_sha256_start != restored["parameter_sha256"]:
                raise ValueError(
                    "resume model parameter SHA-256 does not match checkpoint"
                )
            _restore_rng_state(restored["rng_state"])
        else:
            parameter_sha256_start = _model_sha256(model)
        buffer = DeviceTransitionBuffer(
            capacity=env.cohort_horizon_steps,
            num_envs=config.num_envs,
            observation_shape=(observation_dim,),
            action_shape=(7,),
            device=env.device,
            observation_dtype=observation.dtype,
            action_dtype=torch.float32,
            reward_dtype=torch.float32,
            terminal_signal_dtype=torch.int32,
            event_mask_dtype=torch.int32,
            terminal_reason_dtype=torch.int32,
            physics_step_dtype=torch.int64,
            extra_fields={
                "raw_action": DeviceFieldSpec((7,), torch.float32),
                "log_prob": DeviceFieldSpec((), torch.float32),
                "value": DeviceFieldSpec((), torch.float32),
                "next_value": DeviceFieldSpec((), torch.float32, phase="commit"),
            },
        )

        model.eval()
        for _step in range(config.warmup_steps):
            with torch.inference_mode():
                distribution, _value = model.distribution_and_value(observation)
                torch.tanh(distribution.mean)
        torch.cuda.synchronize(env.device)

        cohort_rows = []
        update_steps = update_steps_before
        invocation_update_steps = 0
        total_allocated_steps = 0
        total_valid_steps = 0
        total_successes = 0
        seen_episode_ids: set[str] = set()
        wall_started = time.perf_counter()
        for local_cohort in range(config.cohorts):
            cohort = completed_cohorts_before + local_cohort
            reset = initial if local_cohort == 0 else env.reset()
            observation = reset.observation
            buffer.reset_cohort()
            model.eval()
            torch.cuda.synchronize(env.device)
            rollout_started = time.perf_counter()
            for _step in range(env.cohort_horizon_steps):
                with torch.inference_mode():
                    action, raw_action, log_prob, value = _sample(model, observation)
                    action = action.contiguous()
                buffer.begin_step(
                    observation=observation,
                    action=action,
                    extras={
                        "raw_action": raw_action,
                        "log_prob": log_prob,
                        "value": value,
                    },
                )
                try:
                    with torch.inference_mode():
                        step = env.step(action)
                        _next_distribution, next_value = model.distribution_and_value(
                            step.observation
                        )
                    buffer.commit_step(
                        reward=step.reward,
                        next_observation=step.observation,
                        terminated=step.terminated,
                        truncated=step.truncated,
                        success=step.success,
                        event_mask=step.event_mask,
                        terminal_reason=step.terminal_reason,
                        physics_step=step.physics_step,
                        extras={"next_value": next_value},
                    )
                except BaseException:
                    if buffer.pending:
                        buffer.abort_step()
                    raise
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
            invocation_update_steps += updates
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
            lane_terminated = (
                (rollout.terminated & rollout.valid).any(dim=0).cpu().tolist()
            )
            lane_truncated = (
                (rollout.truncated & rollout.valid).any(dim=0).cpu().tolist()
            )
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
                        "cohort": cohort,
                        "lane": lane,
                        "episode_id": episode_id,
                        "seed": reset.seeds[lane],
                        "manifest_ordinal": reset.manifest_ordinals[lane],
                        "manifest_sha256": reset.manifest_sha256,
                        "task_id": config.task,
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
            _append_jsonl_rows(ledger_path, ledger_rows)
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
        parameter_sha256_end = _model_sha256(model)
        if invocation_update_steps < 1:
            raise RuntimeError("PPO smoke completed no optimizer updates")
        if parameter_sha256_end == parameter_sha256_start:
            raise RuntimeError("PPO optimizer updates did not change model parameters")

        checkpoint_path = args.output / "checkpoint_latest.pt"
        completed_cohorts_after = completed_cohorts_before + config.cohorts
        checkpoint_payload = {
            "schema_version": "rlinf-gpuenv0-tensor-ppo-smoke-v0.2",
            "config": asdict(config),
            "cohort_horizon_steps": env.cohort_horizon_steps,
            "observation_dim": observation_dim,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "manifest_cursor": dict(env.manifest_state_dict()),
            "completed_cohorts": completed_cohorts_after,
            "update_steps": update_steps,
            "rng_state": _capture_rng_state(),
            "parameter_sha256": parameter_sha256_end,
        }
        _atomic_torch_save(checkpoint_path, checkpoint_payload)
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_reload_checks = {
            "schema_match": reloaded.get("schema_version")
            == "rlinf-gpuenv0-tensor-ppo-smoke-v0.2",
            "completed_cohorts_match": int(reloaded.get("completed_cohorts", -1))
            == completed_cohorts_after,
            "update_steps_match": int(reloaded.get("update_steps", -1)) == update_steps,
            "manifest_cursor_match": reloaded.get("manifest_cursor")
            == checkpoint_payload["manifest_cursor"],
            "parameter_sha256_match": reloaded.get("parameter_sha256")
            == parameter_sha256_end,
            "model_payload_sha256_match": _state_dict_sha256(reloaded["model"])
            == parameter_sha256_end,
        }
        if not all(checkpoint_reload_checks.values()):
            raise RuntimeError(
                f"checkpoint reload validation failed: {checkpoint_reload_checks}"
            )
        provenance = env.provenance
        transport_receipt = env.last_transport_receipt
        if transport_receipt is None or env.transport_checks < 1:
            raise RuntimeError("tensor PPO produced no validated transport receipt")
        source_provenance = {
            "sources": provenance_bundle["sources"],
            "runtime": {
                "path": provenance_bundle["runtime"]["path"],
                "sha256": provenance_bundle["runtime"]["sha256"],
                "versions": provenance_bundle["runtime"]["payload"]["versions"],
            },
        }
        report = {
            "schema_version": "rlinf-gpuenv0-tensor-ppo-smoke-report-v0.2",
            "status": "passed",
            "config": asdict(config),
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
                "physical_device_identity_source": (
                    provenance.physical_device_identity_source
                ),
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
                "hot_path_host_materializations": 0,
                "cohort_horizon_steps": env.cohort_horizon_steps,
                "warmup_policy_only_steps": config.warmup_steps,
            },
            "train": {
                "cohorts": cohort_rows,
                "completed_cohorts_before": completed_cohorts_before,
                "completed_cohorts_after": completed_cohorts_after,
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
