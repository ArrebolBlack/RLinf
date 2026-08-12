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

"""Single-GPU visual-policy Direct PPO runner for GPUENV0."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import compute_ppo_actor_loss, compute_ppo_critic_loss
from rlinf.data.direct_ppo_rollout_buffer import (
    DirectPPORollout,
    DirectPPORolloutBuffer,
    select_rollout_observations,
)
from rlinf.envs.dynamic_benchmark.direct_ppo_reward import (
    DirectPPOReward,
    DirectPPORewardViolation,
)
from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import (
    GpuNativeTensorBackendEnv,
    GpuNativeVisualPolicyObservation,
)
from rlinf.models.embodiment.direct_ppo_visual_actor_critic import (
    DirectPPOVisualActorCritic,
)


@dataclass(frozen=True)
class DirectPPORunConfig:
    name: str
    seed: int
    num_envs: int
    cohorts: int
    rollout_horizon: int
    minibatch_size: int
    ppo_epochs: int
    encoder_batch_size: int
    manifest_name: str
    image_size: int = 64
    checkpoint_every_cohorts: int = 10

    def __post_init__(self) -> None:
        for name in (
            "num_envs",
            "cohorts",
            "rollout_horizon",
            "minibatch_size",
            "ppo_epochs",
            "encoder_batch_size",
            "image_size",
            "checkpoint_every_cohorts",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.name or not self.manifest_name:
            raise ValueError("Direct PPO run names must be non-empty")


@dataclass(frozen=True)
class DirectPPOSourceIdentity:
    se3_commit: str
    se3_tree: str
    rlinf_commit: str
    rlinf_tree: str
    expected_gpu_uuid: str
    export_dir: str


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _device_assert(condition: torch.Tensor, message: str) -> None:
    """Queue a fail-closed CUDA assertion without a per-minibatch host sync."""

    if not isinstance(condition, torch.Tensor) or condition.numel() != 1:
        raise TypeError("Direct PPO device assertion must be a scalar tensor")
    if condition.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise RuntimeError(message)


class _CudaStageTimer:
    """Accumulate CUDA event durations and materialize them at cohort boundaries."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._pairs: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def start(self) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, name: str, started: torch.cuda.Event) -> None:
        ended = torch.cuda.Event(enable_timing=True)
        ended.record()
        self._pairs.append((name, started, ended))

    def materialize_into(self, totals: dict[str, float]) -> None:
        torch.cuda.synchronize(self.device)
        for name, started, ended in self._pairs:
            totals[name] += float(started.elapsed_time(ended)) / 1000.0
        self._pairs.clear()


def _sample_action(
    model: DirectPPOVisualActorCritic,
    observation: GpuNativeVisualPolicyObservation,
    *,
    encoder_batch_size: int,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distribution, value = model.distribution_and_value(
        observation, encoder_batch_size=encoder_batch_size
    )
    raw_action = distribution.sample() if stochastic else distribution.mean
    action = torch.tanh(raw_action)
    log_prob = (
        distribution.log_prob(raw_action)
        - torch.log(torch.clamp(1.0 - action.square(), min=1.0e-6))
    ).sum(dim=-1)
    return action.contiguous(), raw_action, log_prob, value


def _native_gae(rollout: DirectPPORollout, ppo: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat((rollout.value, rollout.next_value[-1:]), dim=0)
    dones = torch.zeros(
        (rollout.horizon + 1, rollout.num_envs),
        dtype=torch.bool,
        device=rollout.reward.device,
    )
    dones[1:] = rollout.terminated | rollout.truncated
    valid_count = int(rollout.valid.sum())
    advantages, returns = compute_gae_advantages_and_returns(
        rewards=rollout.reward,
        values=values,
        dones=dones,
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        normalize_advantages=valid_count >= 2,
        normalize_returns=False,
        loss_mask=rollout.valid,
    )
    if (
        tuple(advantages.shape) != tuple(rollout.reward.shape)
        or tuple(returns.shape) != tuple(rollout.reward.shape)
    ):
        raise RuntimeError("RLinf native GAE produced wrong-shaped output")
    _device_assert(
        torch.isfinite(advantages[rollout.valid]).all()
        & torch.isfinite(returns[rollout.valid]).all(),
        "RLinf native GAE produced NaN or Inf",
    )
    return advantages, returns


def _gradient_norm(parameters: Any) -> torch.Tensor:
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not terms:
        return torch.tensor(0.0)
    return torch.sqrt(torch.stack(terms).sum())


def _ppo_update(
    *,
    model: DirectPPOVisualActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: DirectPPORollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    ppo: Mapping[str, Any],
    config: DirectPPORunConfig,
) -> tuple[dict[str, float], int, int]:
    total = rollout.horizon * rollout.num_envs
    flat_valid = rollout.valid.reshape(total)
    valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
    sample_count = int(valid_indices.numel())
    if sample_count < 1:
        raise RuntimeError("Direct PPO rollout has no valid samples")
    flat_action = rollout.action.reshape(total, 7)
    flat_raw_action = rollout.raw_action.reshape(total, 7)
    flat_old_log_prob = rollout.old_log_prob.reshape(total)
    flat_old_value = rollout.value.reshape(total)
    flat_advantage = advantages.reshape(total)
    flat_return = returns.reshape(total)
    metric_rows = []
    updates = 0
    learner_samples = 0
    for _epoch in range(config.ppo_epochs):
        permutation = valid_indices[
            torch.randperm(sample_count, device=valid_indices.device)
        ]
        for start in range(0, sample_count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            if indices.numel() < 1:
                continue
            observation = select_rollout_observations(rollout, indices)
            distribution, predicted_value = model.distribution_and_value(
                observation,
                encoder_batch_size=min(config.encoder_batch_size, int(indices.numel())),
            )
            selected_action = flat_action.index_select(0, indices)
            selected_raw_action = flat_raw_action.index_select(0, indices)
            new_log_prob = (
                distribution.log_prob(selected_raw_action)
                - torch.log(torch.clamp(1.0 - selected_action.square(), min=1.0e-6))
            ).sum(dim=-1)
            mask = torch.ones_like(new_log_prob, dtype=torch.bool)
            actor_loss, actor_metrics = compute_ppo_actor_loss(
                logprobs=new_log_prob.float(),
                old_logprobs=flat_old_log_prob.index_select(0, indices).float(),
                advantages=flat_advantage.index_select(0, indices).float(),
                clip_ratio_low=float(ppo["clip_ratio_low"]),
                clip_ratio_high=float(ppo["clip_ratio_high"]),
                loss_mask=mask,
            )
            critic_loss, critic_metrics = compute_ppo_critic_loss(
                values=predicted_value.float(),
                returns=flat_return.index_select(0, indices).float(),
                prev_values=flat_old_value.index_select(0, indices).float(),
                value_clip=float(ppo["value_clip"]),
                huber_delta=float(ppo["huber_delta"]),
                loss_mask=mask,
            )
            entropy = distribution.entropy().sum(dim=-1).mean()
            loss = (
                actor_loss
                + float(ppo["value_coef"]) * critic_loss
                - float(ppo["entropy_coef"]) * entropy
            )
            _device_assert(torch.isfinite(loss), "PPO loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_grad = _gradient_norm(model.parameters()).to(loss.device)
            visual_grad = _gradient_norm(model.image_encoder.parameters()).to(loss.device)
            _device_assert(
                torch.isfinite(total_grad) & torch.isfinite(visual_grad),
                "PPO gradient is NaN or Inf",
            )
            clipped_grad = nn.utils.clip_grad_norm_(
                model.parameters(), float(ppo["max_grad_norm"])
            )
            optimizer.step()
            parameter_finite = torch.stack(
                [torch.isfinite(parameter).all() for parameter in model.parameters()]
            ).all()
            _device_assert(
                parameter_finite, "PPO optimizer produced NaN or Inf parameters"
            )
            metric_rows.append(
                torch.stack(
                    (
                        actor_loss.detach(),
                        critic_loss.detach(),
                        entropy.detach(),
                        total_grad.detach(),
                        visual_grad.detach(),
                        clipped_grad.detach().float(),
                        actor_metrics["actor/approx_kl"].detach().float(),
                        actor_metrics["actor/clip_fraction"].detach().float(),
                        critic_metrics["critic/value_clip_ratio"].detach().float(),
                    )
                )
            )
            updates += 1
            learner_samples += int(indices.numel())
    if not metric_rows:
        raise RuntimeError("Direct PPO completed no optimizer updates")
    metrics = torch.stack(metric_rows).mean(dim=0)
    names = (
        "policy_loss",
        "value_loss",
        "entropy",
        "gradient_norm",
        "visual_gradient_norm",
        "preclip_gradient_norm",
        "approx_kl",
        "clip_fraction",
        "value_clip_fraction",
    )
    payload = {name: float(metrics[index]) for index, name in enumerate(names)}
    if not all(math.isfinite(value) for value in payload.values()):
        raise RuntimeError("PPO update metrics are NaN or Inf")
    return payload, updates, learner_samples


class DirectPPORunner:
    """Own one frozen manifest, GPU environment, reward, and PPO learner."""

    def __init__(
        self,
        *,
        contract: Mapping[str, Any],
        contract_path: Path,
        source: DirectPPOSourceIdentity,
        config: DirectPPORunConfig,
        output: Path,
        resume_from: Path | None = None,
        model_state: Mapping[str, torch.Tensor] | None = None,
        policy_mode: str = "stochastic",
        train: bool = True,
        verify_ledger_double_consume: bool = False,
    ) -> None:
        self.contract = contract
        self.contract_path = contract_path
        self.contract_sha256 = _file_sha256(contract_path)
        self.source = source
        self.config = config
        self.output = output
        self.resume_from = resume_from
        if policy_mode not in {"stochastic", "deterministic", "random"}:
            raise ValueError("policy_mode must be stochastic, deterministic, or random")
        if train and policy_mode != "stochastic":
            raise ValueError("PPO training requires stochastic on-policy actions")
        self.policy_mode = policy_mode
        self.train_enabled = train
        self.verify_ledger_double_consume = verify_ledger_double_consume
        self.ppo = contract["ppo"]
        self.manifest = contract["manifests"][config.manifest_name]
        if self.manifest["observation_track"] != "hybrid":
            raise ValueError("Direct PPO requires the frozen hybrid manifest")
        if config.num_envs > int(self.manifest["size"]):
            raise ValueError("manifest is smaller than num_envs")
        evaluator = contract["evaluator"]
        self.env = GpuNativeTensorBackendEnv(
            task_id=str(contract["task_id"]),
            num_envs=config.num_envs,
            export_dir=source.export_dir,
            expected_gpu_uuid=source.expected_gpu_uuid,
            expected_se3_source_commit=source.se3_commit,
            expected_se3_source_tree=source.se3_tree,
            image_size=config.image_size,
            split=str(self.manifest["split"]),
            manifest_seed=int(self.manifest["seed"]),
            manifest_size=int(self.manifest["size"]),
            manifest_sha256=str(self.manifest["sha256"]),
            task_quality_schema_version=str(evaluator["task_quality_schema_version"]),
            task_quality_evaluator_backend_id=str(
                evaluator["task_quality_evaluator_backend_id"]
            ),
            observation_track="hybrid",
        )
        if self.env.cohort_horizon_steps % config.rollout_horizon:
            self.env.close()
            raise ValueError("rollout_horizon must divide the canonical cohort horizon")
        self.device = self.env.device
        self.model = DirectPPOVisualActorCritic(
            hidden_size=int(self.ppo["hidden_size"]), image_size=config.image_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.ppo["learning_rate"])
        )
        self.reward = DirectPPOReward(
            contract, num_envs=config.num_envs, device=self.device
        )
        self.completed_cohorts = 0
        self.update_steps = 0
        self.learner_samples = 0
        if resume_from is not None:
            self._load_checkpoint(resume_from)
        elif model_state is not None:
            self.model.load_state_dict(model_state, strict=True)

    def _load_checkpoint(self, path: Path) -> None:
        restored = torch.load(path, map_location="cpu", weights_only=False)
        if restored.get("schema_version") != "rlinf-gpuenv0-direct-ppo-checkpoint-v1":
            raise ValueError("Direct PPO checkpoint schema mismatch")
        if restored.get("contract_sha256") != self.contract_sha256:
            raise ValueError("Direct PPO checkpoint reward contract mismatch")
        if restored.get("run_config") != asdict(self.config):
            raise ValueError("Direct PPO checkpoint run config mismatch")
        self.model.load_state_dict(restored["model"], strict=True)
        self.optimizer.load_state_dict(restored["optimizer"])
        self.env.load_manifest_state_dict(restored["manifest_cursor"])
        self.completed_cohorts = int(restored["completed_cohorts"])
        self.update_steps = int(restored["update_steps"])
        self.learner_samples = int(restored["learner_samples"])
        random.setstate(restored["rng"]["python"])
        np.random.set_state(restored["rng"]["numpy"])
        torch.set_rng_state(restored["rng"]["torch_cpu"])
        torch.cuda.set_rng_state_all(restored["rng"]["torch_cuda_all"])
        if _state_dict_sha256(self.model.state_dict()) != restored["parameter_sha256"]:
            raise ValueError("Direct PPO checkpoint parameter digest mismatch")

    def _checkpoint(self) -> tuple[Path, str]:
        path = self.output / "checkpoint_latest.pt"
        parameter_sha256 = _state_dict_sha256(self.model.state_dict())
        payload = {
            "schema_version": "rlinf-gpuenv0-direct-ppo-checkpoint-v1",
            "contract_sha256": self.contract_sha256,
            "run_config": asdict(self.config),
            "source": asdict(self.source),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "manifest_cursor": dict(self.env.manifest_state_dict()),
            "completed_cohorts": self.completed_cohorts,
            "update_steps": self.update_steps,
            "learner_samples": self.learner_samples,
            "parameter_sha256": parameter_sha256,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda_all": torch.cuda.get_rng_state_all(),
            },
        }
        _atomic_torch_save(path, payload)
        return path, _file_sha256(path)

    def _visual_evidence(
        self, observation: GpuNativeVisualPolicyObservation
    ) -> dict[str, Any]:
        self.model.eval()
        with torch.inference_mode():
            actual, _value = self.model.distribution_and_value(
                observation, encoder_batch_size=self.config.encoder_batch_size
            )
            zero_observation = GpuNativeVisualPolicyObservation(
                proprio=observation.proprio,
                rgb={name: torch.zeros_like(value) for name, value in observation.rgb.items()},
                depth_m={
                    name: torch.zeros_like(value)
                    for name, value in observation.depth_m.items()
                },
                segmentation={
                    name: torch.zeros_like(value)
                    for name, value in observation.segmentation.items()
                },
            )
            zeroed, _zero_value = self.model.distribution_and_value(
                zero_observation, encoder_batch_size=self.config.encoder_batch_size
            )
        rgb_std = {
            camera: float(value.std(unbiased=False))
            for camera, value in observation.rgb.items()
        }
        depth_finite_positive = {
            camera: bool(torch.isfinite(value).all() & (value > 0.0).any())
            for camera, value in observation.depth_m.items()
        }
        target_pixels = {
            camera: int(((value == 85) | (value == 86)).sum())
            for camera, value in observation.segmentation.items()
        }
        action_sensitivity = float((actual.mean - zeroed.mean).abs().max())
        checks = {
            "rgb_nonconstant": all(value > 0.0 for value in rgb_std.values()),
            "depth_finite_positive": all(depth_finite_positive.values()),
            "target_visible": sum(target_pixels.values()) > 0,
            "policy_action_depends_on_visual": action_sensitivity > 1.0e-8,
        }
        if not all(checks.values()):
            raise RuntimeError(f"GPU render/policy-consumption evidence failed: {checks}")
        return {
            "checks": checks,
            "rgb_std": rgb_std,
            "depth_finite_positive": depth_finite_positive,
            "target_pixels": target_pixels,
            "action_mean_max_abs_change_when_visual_zeroed": action_sensitivity,
            "policy_information_boundary": observation.information_boundary,
            "policy_dataclass_fields": list(observation.__dataclass_fields__),
        }

    def _save_render_witness(
        self, observation: GpuNativeVisualPolicyObservation
    ) -> dict[str, Any]:
        """Persist the public tensors actually presented to the policy at reset."""

        path = self.output / "render_witness.pt"
        payload: dict[str, Any] = {
            "schema_version": "rlinf-gpuenv0-direct-ppo-render-witness-v1",
            "information_boundary": observation.information_boundary,
            "proprio": observation.proprio[:1].detach().cpu(),
        }
        for camera in ("agentview", "robot0_eye_in_hand"):
            payload[f"{camera}.rgb"] = observation.rgb[camera][:1].detach().cpu()
            payload[f"{camera}.depth_m"] = (
                observation.depth_m[camera][:1].detach().cpu()
            )
            payload[f"{camera}.segmentation"] = (
                observation.segmentation[camera][:1].detach().cpu()
            )
        _atomic_torch_save(path, payload)
        return {
            "path": str(path),
            "sha256": _file_sha256(path),
            "contains_privileged_state": False,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
        }

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        initial_policy_path = self.output / "initial_policy.pt"
        if self.train_enabled and self.completed_cohorts == 0:
            _atomic_torch_save(
                initial_policy_path,
                {
                    "schema_version": "rlinf-gpuenv0-direct-ppo-policy-v1",
                    "contract_sha256": self.contract_sha256,
                    "model": self.model.state_dict(),
                    "parameter_sha256": _state_dict_sha256(self.model.state_dict()),
                },
            )
        buffer = DirectPPORolloutBuffer(
            capacity=self.config.rollout_horizon,
            num_envs=self.config.num_envs,
            image_size=self.config.image_size,
            device=self.device,
        )
        parameter_sha256_start = _state_dict_sha256(self.model.state_dict())
        stage_seconds = {
            "reset": 0.0,
            "policy": 0.0,
            "environment_render_physics": 0.0,
            "reward": 0.0,
            "rollout_storage": 0.0,
            "advantage": 0.0,
            "update": 0.0,
            "audit": 0.0,
            "checkpoint": 0.0,
        }
        cuda_timer = _CudaStageTimer(self.device)
        allocated_steps = 0
        valid_steps = 0
        rendered_frames = 0
        optimizer_updates_invocation = 0
        learner_samples_invocation = 0
        episode_rows = []
        update_rows = []
        cohort_rows = []
        ledger_double_consume_rejected = None
        wall_started = time.perf_counter()
        reset_started = time.perf_counter()
        reset = self.env.reset()
        torch.cuda.synchronize(self.device)
        stage_seconds["reset"] += time.perf_counter() - reset_started
        policy_observation = self.env.visual_policy_observation(reset.observation)
        reward_state = self.env.privileged_reward_state(reset.observation)
        self.reward.reset(reward_state)
        render_witness = self._save_render_witness(policy_observation)
        visual_evidence = self._visual_evidence(policy_observation)

        for local_cohort in range(self.config.cohorts):
            cohort = self.completed_cohorts
            torch.cuda.synchronize(self.device)
            cohort_started = time.perf_counter()
            cohort_started_epoch_s = time.time()
            cohort_allocated_before = allocated_steps
            cohort_valid_before = valid_steps
            cohort_updates_before = optimizer_updates_invocation
            cohort_samples_before = learner_samples_invocation
            if local_cohort:
                reset_started = time.perf_counter()
                reset = self.env.reset()
                torch.cuda.synchronize(self.device)
                stage_seconds["reset"] += time.perf_counter() - reset_started
                policy_observation = self.env.visual_policy_observation(reset.observation)
                reward_state = self.env.privileged_reward_state(reset.observation)
                self.reward.reset(reward_state)
            active = torch.ones(
                self.config.num_envs, dtype=torch.bool, device=self.device
            )
            terminal_success = torch.zeros_like(active)
            terminal_reason = torch.zeros(
                self.config.num_envs, dtype=torch.int32, device=self.device
            )
            terminal_physics_step = torch.zeros(
                self.config.num_envs, dtype=torch.int64, device=self.device
            )
            cohort_update_rows = []
            for _chunk_start in range(
                0, self.env.cohort_horizon_steps, self.config.rollout_horizon
            ):
                buffer.reset()
                for _step in range(self.config.rollout_horizon):
                    valid = active.clone()
                    policy_started = cuda_timer.start()
                    self.model.eval()
                    with torch.inference_mode():
                        if self.policy_mode == "random":
                            action = torch.empty(
                                (self.config.num_envs, 7),
                                dtype=torch.float32,
                                device=self.device,
                            ).uniform_(-1.0, 1.0)
                            action = torch.clamp(action, -0.999999, 0.999999).contiguous()
                            raw_action = torch.atanh(action)
                            old_log_prob = torch.zeros(
                                self.config.num_envs,
                                dtype=torch.float32,
                                device=self.device,
                            )
                            _distribution, value = self.model.distribution_and_value(
                                policy_observation,
                                encoder_batch_size=self.config.encoder_batch_size,
                            )
                        else:
                            action, raw_action, old_log_prob, value = _sample_action(
                                self.model,
                                policy_observation,
                                encoder_batch_size=self.config.encoder_batch_size,
                                stochastic=self.policy_mode == "stochastic",
                            )
                    cuda_timer.stop("policy", policy_started)
                    storage_started = cuda_timer.start()
                    buffer.begin_step(
                        observation=policy_observation,
                        action=action,
                        raw_action=raw_action,
                        old_log_prob=old_log_prob,
                        value=value,
                        valid=valid,
                    )
                    cuda_timer.stop("rollout_storage", storage_started)
                    try:
                        env_started = cuda_timer.start()
                        step = self.env.step(action)
                        cuda_timer.stop("environment_render_physics", env_started)
                        next_policy_observation = self.env.visual_policy_observation(
                            step.observation
                        )
                        next_reward_state = self.env.privileged_reward_state(step.observation)
                        next_policy_started = cuda_timer.start()
                        with torch.inference_mode():
                            _next_distribution, next_value = (
                                self.model.distribution_and_value(
                                    next_policy_observation,
                                    encoder_batch_size=self.config.encoder_batch_size,
                                )
                            )
                        cuda_timer.stop("policy", next_policy_started)
                        reward_started = cuda_timer.start()
                        reward_step = self.reward.step(
                            state=next_reward_state,
                            action=action,
                            terminated=step.terminated,
                            truncated=step.truncated,
                            success=step.success,
                            terminal_reason=step.terminal_reason,
                            valid=valid,
                        )
                        _device_assert(
                            torch.isfinite(step.reward).all(),
                            "engine task reward is NaN or Inf",
                        )
                        cuda_timer.stop("reward", reward_started)
                        storage_started = cuda_timer.start()
                        buffer.commit_step(
                            next_value=next_value,
                            reward=reward_step.reward,
                            terminated=step.terminated,
                            truncated=step.truncated,
                        )
                        cuda_timer.stop("rollout_storage", storage_started)
                    except BaseException:
                        if buffer.pending:
                            buffer.abort_step()
                        raise
                    done = step.done & valid
                    terminal_success = torch.where(done, step.success.bool(), terminal_success)
                    terminal_reason = torch.where(done, step.terminal_reason, terminal_reason)
                    terminal_physics_step = torch.where(
                        done, step.physics_step, terminal_physics_step
                    )
                    active &= ~done
                    policy_observation = next_policy_observation
                    reward_state = next_reward_state
                    allocated_steps += self.config.num_envs
                    rendered_frames += self.config.num_envs * 2
                rollout = buffer.view()
                chunk_valid_steps = int(rollout.valid.sum())
                valid_steps += chunk_valid_steps
                if chunk_valid_steps == 0:
                    continue
                advantage_started = cuda_timer.start()
                advantages, returns = _native_gae(rollout, self.ppo)
                cuda_timer.stop("advantage", advantage_started)
                if self.train_enabled:
                    self.model.train()
                    update_started = cuda_timer.start()
                    metrics, updates, samples = _ppo_update(
                        model=self.model,
                        optimizer=self.optimizer,
                        rollout=rollout,
                        advantages=advantages,
                        returns=returns,
                        ppo=self.ppo,
                        config=self.config,
                    )
                    cuda_timer.stop("update", update_started)
                    self.update_steps += updates
                    self.learner_samples += samples
                    optimizer_updates_invocation += updates
                    learner_samples_invocation += samples
                    cohort_update_rows.append(
                        {
                            "cohort": cohort,
                            "valid_samples": chunk_valid_steps,
                            "optimizer_updates": updates,
                            "learner_samples": samples,
                            **metrics,
                        }
                    )
            if bool(active.any()):
                raise RuntimeError("canonical cohort ended without terminal/truncation")
            audit_started = time.perf_counter()
            health = self.env.materialize_health_audit()
            lanes = tuple(range(self.config.num_envs))
            ledger = self.env.materialize_terminal_ledger_once(lanes, reset.episode_ids)
            if self.verify_ledger_double_consume and ledger_double_consume_rejected is None:
                try:
                    self.env.materialize_terminal_ledger_once(lanes, reset.episode_ids)
                except RuntimeError as exc:
                    ledger_double_consume_rejected = "already consumed" in str(exc)
                else:
                    ledger_double_consume_rejected = False
                if ledger_double_consume_rejected is not True:
                    raise RuntimeError("terminal ledger accepted duplicate consumption")
            decompositions = self.reward.consume_episode_summaries(lanes)
            stage_seconds["audit"] += time.perf_counter() - audit_started
            terminal_success_host = terminal_success.cpu().tolist()
            terminal_reason_host = terminal_reason.cpu().tolist()
            terminal_physics_host = terminal_physics_step.cpu().tolist()
            for row, decomposition in zip(ledger, decompositions, strict=True):
                lane = row.lane
                expected_reason = 0 if row.truncated else terminal_reason_host[lane]
                # The device step returns the end of the 20 Hz control interval;
                # the canonical terminal ledger latches the exact 500 Hz event
                # clock inside that interval.  Reconcile the two clocks without
                # discarding the more precise ledger timestamp.
                interval_end = terminal_physics_host[lane]
                terminal_clock_in_interval = (
                    interval_end - 25 < row.physics_step <= interval_end
                )
                if (
                    row.success != terminal_success_host[lane]
                    or not terminal_clock_in_interval
                    or (not row.truncated and expected_reason == 0)
                ):
                    raise RuntimeError(
                        "terminal ledger differs from device terminal signals: "
                        f"lane={lane}, ledger_success={row.success}, "
                        f"device_success={terminal_success_host[lane]}, "
                        f"ledger_physics_step={row.physics_step}, "
                        f"device_physics_step={terminal_physics_host[lane]}, "
                        f"ledger_terminated={row.terminated}, "
                        f"ledger_truncated={row.truncated}, "
                        f"ledger_reason={row.termination_reason!r}, "
                        f"device_reason={terminal_reason_host[lane]}"
                    )
                quality = None if row.task_quality is None else row.task_quality.to_dict()
                episode_rows.append(
                    {
                        "cohort": cohort,
                        "lane": lane,
                        "episode_id": row.episode_id,
                        "manifest_ordinal": reset.manifest_ordinals[lane],
                        "seed": reset.seeds[lane],
                        "success": row.success,
                        "terminated": row.terminated,
                        "truncated": row.truncated,
                        "termination_reason": row.termination_reason,
                        "completion": row.completion,
                        "physics_step": row.physics_step,
                        "control_step": row.control_step,
                        "policy_step": row.policy_step,
                        "task_quality": quality,
                        "reward": decomposition,
                    }
                )
            if any(np.asarray(health["overflow"]).reshape(-1)):
                raise RuntimeError("health audit reported overflow")
            update_rows.extend(cohort_update_rows)
            cuda_timer.materialize_into(stage_seconds)
            cohort_wall_seconds = time.perf_counter() - cohort_started
            cohort_ended_epoch_s = time.time()
            cohort_rows.append(
                {
                    "cohort": cohort,
                    "started_at_epoch_s": cohort_started_epoch_s,
                    "ended_at_epoch_s": cohort_ended_epoch_s,
                    "allocated_env_steps": allocated_steps - cohort_allocated_before,
                    "valid_env_steps": valid_steps - cohort_valid_before,
                    "rendered_frames": (
                        (allocated_steps - cohort_allocated_before) * 2
                    ),
                    "optimizer_updates": (
                        optimizer_updates_invocation - cohort_updates_before
                    ),
                    "learner_samples": (
                        learner_samples_invocation - cohort_samples_before
                    ),
                    "wall_seconds": cohort_wall_seconds,
                    "end_to_end_valid_env_steps_per_s": (
                        (valid_steps - cohort_valid_before) / cohort_wall_seconds
                    ),
                    "render_enabled_frames_per_s": (
                        (allocated_steps - cohort_allocated_before)
                        * 2
                        / cohort_wall_seconds
                    ),
                }
            )
            self.completed_cohorts += 1
            if (
                self.train_enabled
                and self.completed_cohorts % self.config.checkpoint_every_cohorts == 0
            ):
                checkpoint_started = time.perf_counter()
                self._checkpoint()
                stage_seconds["checkpoint"] += time.perf_counter() - checkpoint_started

        cuda_timer.materialize_into(stage_seconds)
        wall_seconds = time.perf_counter() - wall_started
        checkpoint_started = time.perf_counter()
        checkpoint_path, checkpoint_sha256 = self._checkpoint()
        stage_seconds["checkpoint"] += time.perf_counter() - checkpoint_started
        parameter_sha256_end = _state_dict_sha256(self.model.state_dict())
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reload_model = DirectPPOVisualActorCritic(
            hidden_size=int(self.ppo["hidden_size"]), image_size=self.config.image_size
        )
        reload_model.load_state_dict(reloaded["model"], strict=True)
        reload_parameter_sha256 = _state_dict_sha256(reload_model.state_dict())
        checkpoint_reload = {
            "schema_match": reloaded.get("schema_version")
            == "rlinf-gpuenv0-direct-ppo-checkpoint-v1",
            "contract_match": reloaded.get("contract_sha256") == self.contract_sha256,
            "parameter_digest_match": reload_parameter_sha256 == parameter_sha256_end,
            "manifest_cursor_match": reloaded.get("manifest_cursor")
            == dict(self.env.manifest_state_dict()),
        }
        if not all(checkpoint_reload.values()):
            raise RuntimeError(f"Direct PPO checkpoint reload failed: {checkpoint_reload}")
        if self.train_enabled and (
            optimizer_updates_invocation < 1
            or learner_samples_invocation < 1
            or parameter_sha256_start == parameter_sha256_end
            or not update_rows
            or min(row["gradient_norm"] for row in update_rows) <= 0.0
            or min(row["visual_gradient_norm"] for row in update_rows) <= 0.0
        ):
            raise RuntimeError("Direct PPO optimizer/visual-gradient canary failed")
        end_identity = self.env.attest_end()
        report = {
            "schema_version": "rlinf-gpuenv0-direct-ppo-run-report-v1",
            "status": "passed",
            "contract": {
                "path": str(self.contract_path),
                "sha256": self.contract_sha256,
                "schema_version": self.contract["schema_version"],
            },
            "config": asdict(self.config),
            "policy_mode": self.policy_mode,
            "source": asdict(self.source),
            "backend": {
                "backend_id": self.env.provenance.backend_id,
                "api_version": self.env.api_version,
                "physical_device_uuid": self.env.provenance.physical_device_uuid,
                "device_identity": asdict(end_identity),
                "manifest_sha256": self.env.manifest_sha256,
                "observation_track": self.env.stable_identity["observation_track"],
                "cpu_env_or_physics_fallback": False,
            },
            "visual_policy_evidence": {**visual_evidence, "witness": render_witness},
            "algorithm": {
                "name": "ppo",
                "advantage_callable": (
                    "rlinf.algorithms.advantages.compute_gae_advantages_and_returns"
                ),
                "actor_loss_callable": "rlinf.algorithms.losses.compute_ppo_actor_loss",
                "critic_loss_callable": "rlinf.algorithms.losses.compute_ppo_critic_loss",
                "on_policy_replay_capacity": 0,
                "planner_demo_bc_dagger_consumed": False,
                "config": dict(self.ppo),
            },
            "train": {
                "enabled": self.train_enabled,
                "allocated_env_steps": allocated_steps,
                "valid_env_steps": valid_steps,
                "rendered_frames": rendered_frames,
                "optimizer_updates": optimizer_updates_invocation,
                "learner_samples": learner_samples_invocation,
                "parameter_sha256_start": parameter_sha256_start,
                "parameter_sha256_end": parameter_sha256_end,
                "parameters_changed": parameter_sha256_start != parameter_sha256_end,
                "update_rows": update_rows,
                "cohorts": cohort_rows,
                "episodes": episode_rows,
            },
            "timing": {
                "wall_seconds": wall_seconds,
                "stage_seconds": stage_seconds,
                "render_enabled_frames_per_s": rendered_frames / wall_seconds,
                "render_enabled_valid_env_steps_per_s": valid_steps / wall_seconds,
                "learner_samples_per_s": learner_samples_invocation
                / max(stage_seconds["update"], 1.0e-9),
                "optimizer_updates_per_s": optimizer_updates_invocation
                / max(stage_seconds["update"], 1.0e-9),
                "end_to_end_valid_env_steps_per_s": valid_steps / wall_seconds,
            },
            "terminal_ledger": {
                "rows": len(episode_rows),
                "unique_episode_ids": len(
                    {row["episode_id"] for row in episode_rows}
                ),
                "double_consume_rejected": ledger_double_consume_rejected,
                "clocks_validated": True,
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "reload": checkpoint_reload,
            },
        }
        _atomic_json(self.output / "report.json", report)
        return report

    def close(self) -> None:
        self.env.close()


__all__ = [
    "DirectPPORunConfig",
    "DirectPPORunner",
    "DirectPPOSourceIdentity",
]
