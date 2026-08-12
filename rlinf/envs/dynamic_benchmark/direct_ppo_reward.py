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

"""Frozen, bounded reward for GPUENV0 visual-policy Direct PPO.

The policy never receives this module's privileged reward state.  Potential
differences are symmetric-clipped, terminal/event rewards are exact-once, and
all bookkeeping stays on the environment CUDA device until an explicit episode
summary is consumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import (
    GpuNativePrivilegedRewardState,
)

CONTRACT_SCHEMA_VERSION = "gpuenv0-direct-ppo-throughput-contract-v1"
REWARD_COMPONENT_NAMES = (
    "terminal_success",
    "approach_delta",
    "grasp_delta",
    "lift_delta",
    "bilateral_contact_once",
    "stable_lift_once",
    "time",
    "action_magnitude",
    "action_jitter",
    "terminal_failure",
    "invalid_state",
    "timeout",
    "overflow",
)
_POSITIVE_SHAPING_COMPONENTS = (
    "approach_delta",
    "grasp_delta",
    "lift_delta",
    "bilateral_contact_once",
    "stable_lift_once",
)
_INVALID_STATE_REASON_CODE = 2


class DirectPPORewardViolation(RuntimeError):
    """Raised when a reward-contract or reward-hacking check fails closed."""


def _device_assert(condition: torch.Tensor, message: str) -> None:
    """Fail closed without adding a CUDA-to-host synchronization to every step."""

    if not isinstance(condition, torch.Tensor) or condition.numel() != 1:
        raise TypeError("device reward assertion must be a scalar tensor")
    if condition.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise DirectPPORewardViolation(message)


@dataclass(frozen=True)
class DirectPPORewardStep:
    reward: torch.Tensor
    unclipped_reward: torch.Tensor
    components: Mapping[str, torch.Tensor]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class DirectPPOReward:
    """Vectorized exact-once reward and episode audit on one CUDA device."""

    def __init__(
        self,
        contract: Mapping[str, Any],
        *,
        num_envs: int,
        device: torch.device,
    ) -> None:
        if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ValueError("Direct PPO reward contract schema mismatch")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        reward = _mapping(contract.get("reward"), "reward")
        if reward.get("normalization") != "none":
            raise ValueError("Direct PPO reward normalization must remain disabled")
        total_clip = reward.get("total_clip")
        if (
            not isinstance(total_clip, list)
            or len(total_clip) != 2
            or _finite_float(total_clip[0], "total_clip[0]") >= 0.0
            or _finite_float(total_clip[1], "total_clip[1]") <= 0.0
        ):
            raise ValueError("reward total_clip must straddle zero")
        self.total_clip = (float(total_clip[0]), float(total_clip[1]))
        component_contract = _mapping(reward.get("components"), "reward.components")
        if tuple(component_contract) != REWARD_COMPONENT_NAMES:
            raise ValueError("reward component names/order drifted")
        self.weights = {
            name: _finite_float(component_contract[name]["weight"], f"{name}.weight")
            for name in REWARD_COMPONENT_NAMES
        }
        parameters = _mapping(reward.get("parameters"), "reward.parameters")
        self.approach_scale = _finite_float(
            parameters.get("approach_distance_scale_m"), "approach_distance_scale_m"
        )
        self.approach_delta_clip = _finite_float(
            parameters.get("approach_delta_clip"), "approach_delta_clip"
        )
        self.grasp_delta_clip = _finite_float(
            parameters.get("grasp_delta_clip"), "grasp_delta_clip"
        )
        self.lift_scale = _finite_float(parameters.get("lift_scale_m"), "lift_scale_m")
        self.lift_delta_clip = _finite_float(
            parameters.get("lift_delta_clip"), "lift_delta_clip"
        )
        self.stable_lift_potential = _finite_float(
            parameters.get("stable_lift_potential"), "stable_lift_potential"
        )
        self.stable_lift_max_speed = _finite_float(
            parameters.get("stable_lift_max_linear_speed_m_s"),
            "stable_lift_max_linear_speed_m_s",
        )
        if min(
            self.approach_scale,
            self.approach_delta_clip,
            self.grasp_delta_clip,
            self.lift_scale,
            self.lift_delta_clip,
            self.stable_lift_potential,
            self.stable_lift_max_speed,
        ) <= 0.0:
            raise ValueError("reward scales, clips, and stable-lift thresholds must be positive")
        limits = _mapping(
            _mapping(contract.get("reward_hacking_checks"), "reward_hacking_checks").get(
                "limits"
            ),
            "reward_hacking_checks.limits",
        )
        self.hacking_limits = {
            name: _finite_float(value, f"reward_hacking_checks.limits.{name}")
            for name, value in limits.items()
        }
        expected_limits = {
            "high_return_failure",
            "contact_without_lift_max_potential",
            "contact_without_lift_min_return",
            "jitter_mean_normalized",
            "jitter_positive_shaping",
            "minimum_success_control_steps",
        }
        if set(self.hacking_limits) != expected_limits:
            raise ValueError("reward-hacking limit vocabulary drifted")
        minimum_steps = self.hacking_limits["minimum_success_control_steps"]
        if not minimum_steps.is_integer() or minimum_steps < 1:
            raise ValueError("minimum_success_control_steps must be a positive integer")

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.initial_object_z = torch.zeros(num_envs, device=device)
        self.previous_approach = torch.zeros(num_envs, device=device)
        self.previous_grasp = torch.zeros(num_envs, device=device)
        self.previous_lift = torch.zeros(num_envs, device=device)
        self.previous_action = torch.zeros((num_envs, 7), device=device)
        self.terminal_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.summary_consumed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.bilateral_consumed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stable_lift_consumed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.max_lift_potential = torch.zeros(num_envs, device=device)
        self.control_steps = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.normalized_jitter_sum = torch.zeros(num_envs, device=device)
        self.component_totals = torch.zeros(
            (num_envs, len(REWARD_COMPONENT_NAMES)), device=device
        )
        self.clipped_return = torch.zeros(num_envs, device=device)
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _validate_state(self, state: GpuNativePrivilegedRewardState) -> None:
        if not isinstance(state, GpuNativePrivilegedRewardState):
            raise TypeError("reward state must use the privileged reward-only dataclass")
        if state.information_boundary != "reward_only_not_policy_visible":
            raise DirectPPORewardViolation("reward/policy information boundary drifted")
        tensors = {
            "eef_position_m": (state.eef_position_m, (self.num_envs, 3)),
            "object_position_m": (state.object_position_m, (self.num_envs, 3)),
            "object_linear_velocity_m_s": (
                state.object_linear_velocity_m_s,
                (self.num_envs, 3),
            ),
            "fingerpad_contact_flags": (
                state.fingerpad_contact_flags,
                (self.num_envs, 2),
            ),
            "post_hold_contact_valid": (
                state.post_hold_contact_valid,
                (self.num_envs,),
            ),
        }
        for name, (tensor, shape) in tensors.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device != self.device
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != shape
            ):
                raise DirectPPORewardViolation(
                    f"reward state {name} has the wrong CUDA schema"
                )
            _device_assert(
                torch.isfinite(tensor).all(), f"reward state {name} contains NaN or Inf"
            )

    def _potentials(
        self, state: GpuNativePrivilegedRewardState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distance = torch.linalg.vector_norm(
            state.eef_position_m - state.object_position_m, dim=-1
        )
        approach = torch.clamp(1.0 - distance / self.approach_scale, 0.0, 1.0)
        contacts = torch.clamp(state.fingerpad_contact_flags, 0.0, 1.0)
        grasp = 0.5 * contacts.sum(dim=-1)
        lift = torch.clamp(
            (state.object_position_m[:, 2] - self.initial_object_z) / self.lift_scale,
            0.0,
            1.0,
        )
        return approach, grasp, lift

    def reset(
        self,
        state: GpuNativePrivilegedRewardState,
        mask: torch.Tensor | None = None,
    ) -> None:
        self._validate_state(state)
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.device != self.device
            or mask.dtype != torch.bool
            or tuple(mask.shape) != (self.num_envs,)
            or not bool(mask.any())
        ):
            raise ValueError("reward reset mask must select CUDA lanes")
        if bool((mask & self.terminal_seen & ~self.summary_consumed).any()):
            raise DirectPPORewardViolation(
                "terminal reward state reset before episode decomposition was consumed"
            )
        self.initial_object_z[mask] = state.object_position_m[mask, 2]
        approach, grasp, _lift = self._potentials(state)
        self.previous_approach[mask] = approach[mask]
        self.previous_grasp[mask] = grasp[mask]
        self.previous_lift[mask] = 0.0
        self.previous_action[mask] = 0.0
        self.terminal_seen[mask] = False
        self.summary_consumed[mask] = False
        self.bilateral_consumed[mask] = False
        self.stable_lift_consumed[mask] = False
        self.max_lift_potential[mask] = 0.0
        self.control_steps[mask] = 0
        self.normalized_jitter_sum[mask] = 0.0
        self.component_totals[mask] = 0.0
        self.clipped_return[mask] = 0.0
        self._initialized[mask] = True

    def step(
        self,
        *,
        state: GpuNativePrivilegedRewardState,
        action: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        success: torch.Tensor,
        terminal_reason: torch.Tensor,
        valid: torch.Tensor,
        overflow: torch.Tensor | None = None,
    ) -> DirectPPORewardStep:
        self._validate_state(state)
        if (
            not isinstance(action, torch.Tensor)
            or action.device != self.device
            or action.dtype != torch.float32
            or tuple(action.shape) != (self.num_envs, 7)
        ):
            raise DirectPPORewardViolation("policy action has the wrong CUDA schema")
        _device_assert(torch.isfinite(action).all(), "policy action contains NaN or Inf")
        signals = {
            "terminated": terminated,
            "truncated": truncated,
            "success": success,
            "terminal_reason": terminal_reason,
            "valid": valid,
        }
        for name, tensor in signals.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device != self.device
                or tuple(tensor.shape) != (self.num_envs,)
            ):
                raise DirectPPORewardViolation(f"{name} has the wrong device schema")
        valid_mask = valid.bool()
        terminated_mask = terminated.bool()
        truncated_mask = truncated.bool()
        success_mask = success.bool()
        reason = terminal_reason.to(torch.int64)
        terminal = terminated_mask | truncated_mask
        _device_assert(
            ~(valid_mask & ~self._initialized).any(),
            "reward step reached an uninitialized lane",
        )
        _device_assert(
            ~(terminal & self.terminal_seen).any(),
            "terminal reward was consumed more than once",
        )
        _device_assert(
            ~(
                (terminated_mask & truncated_mask)
                | (success_mask & (~terminated_mask | truncated_mask))
            ).any(),
            "terminal masks are inconsistent",
        )
        _device_assert(
            ~(terminated_mask & ~success_mask & (reason == 0)).any(),
            "terminal failure lacks a reason code",
        )
        _device_assert(
            ~(terminal & ~valid_mask).any(),
            "inactive lane emitted a repeated terminal signal",
        )

        approach, grasp, lift = self._potentials(state)
        bilateral = grasp >= 1.0
        speed = torch.linalg.vector_norm(state.object_linear_velocity_m_s, dim=-1)
        stable = (
            bilateral
            & (lift >= self.stable_lift_potential)
            & (state.post_hold_contact_valid >= 0.5)
            & (speed <= self.stable_lift_max_speed)
        )
        bilateral_once = valid_mask & bilateral & ~self.bilateral_consumed
        stable_once = valid_mask & stable & ~self.stable_lift_consumed
        next_steps = self.control_steps + valid_mask.to(torch.int64)
        minimum_steps = int(self.hacking_limits["minimum_success_control_steps"])
        stable_evidence = self.stable_lift_consumed | stable_once
        _device_assert(
            ~(success_mask & ((next_steps < minimum_steps) | ~stable_evidence)).any(),
            "success arrived before stable-lift evidence",
        )

        zeros = torch.zeros(self.num_envs, device=self.device)
        mask_float = valid_mask.to(torch.float32)
        component_values = {
            name: zeros.clone() for name in REWARD_COMPONENT_NAMES
        }
        component_values["terminal_success"] = (
            self.weights["terminal_success"] * success_mask.to(torch.float32)
        )
        component_values["approach_delta"] = self.weights["approach_delta"] * torch.clamp(
            approach - self.previous_approach,
            -self.approach_delta_clip,
            self.approach_delta_clip,
        ) * mask_float
        component_values["grasp_delta"] = self.weights["grasp_delta"] * torch.clamp(
            grasp - self.previous_grasp,
            -self.grasp_delta_clip,
            self.grasp_delta_clip,
        ) * mask_float
        component_values["lift_delta"] = self.weights["lift_delta"] * torch.clamp(
            lift - self.previous_lift,
            -self.lift_delta_clip,
            self.lift_delta_clip,
        ) * mask_float
        component_values["bilateral_contact_once"] = (
            self.weights["bilateral_contact_once"] * bilateral_once.to(torch.float32)
        )
        component_values["stable_lift_once"] = (
            self.weights["stable_lift_once"] * stable_once.to(torch.float32)
        )
        component_values["time"] = self.weights["time"] * mask_float
        action_magnitude = action.square().mean(dim=-1)
        normalized_jitter = (action - self.previous_action).square().mean(dim=-1) / 4.0
        component_values["action_magnitude"] = (
            self.weights["action_magnitude"] * action_magnitude * mask_float
        )
        component_values["action_jitter"] = (
            self.weights["action_jitter"] * normalized_jitter * mask_float
        )
        invalid = terminated_mask & ~success_mask & (reason == _INVALID_STATE_REASON_CODE)
        failed = terminated_mask & ~success_mask & ~invalid
        component_values["terminal_failure"] = (
            self.weights["terminal_failure"] * failed.to(torch.float32)
        )
        component_values["invalid_state"] = (
            self.weights["invalid_state"] * invalid.to(torch.float32)
        )
        component_values["timeout"] = (
            self.weights["timeout"] * truncated_mask.to(torch.float32)
        )
        if overflow is not None:
            if (
                not isinstance(overflow, torch.Tensor)
                or overflow.device != self.device
                or tuple(overflow.shape) != (self.num_envs,)
            ):
                raise DirectPPORewardViolation("overflow signal has the wrong device schema")
            overflow_mask = overflow.bool()
            component_values["overflow"] = (
                self.weights["overflow"] * overflow_mask.to(torch.float32)
            )
        else:
            overflow_mask = torch.zeros_like(valid_mask)

        stacked = torch.stack(
            [component_values[name] for name in REWARD_COMPONENT_NAMES], dim=-1
        )
        unclipped = stacked.sum(dim=-1)
        clipped = torch.clamp(unclipped, *self.total_clip)
        _device_assert(
            torch.isfinite(stacked).all() & torch.isfinite(clipped).all(),
            "reward contains NaN or Inf",
        )
        self.component_totals += stacked
        self.clipped_return += clipped
        self.max_lift_potential = torch.maximum(self.max_lift_potential, lift * mask_float)
        self.normalized_jitter_sum += normalized_jitter * mask_float
        self.control_steps = next_steps
        self.previous_approach = torch.where(valid_mask, approach, self.previous_approach)
        self.previous_grasp = torch.where(valid_mask, grasp, self.previous_grasp)
        self.previous_lift = torch.where(valid_mask, lift, self.previous_lift)
        self.previous_action = torch.where(valid_mask[:, None], action, self.previous_action)
        self.bilateral_consumed |= bilateral_once
        self.stable_lift_consumed |= stable_once
        self.terminal_seen |= terminal

        positive_indices = [
            REWARD_COMPONENT_NAMES.index(name) for name in _POSITIVE_SHAPING_COMPONENTS
        ]
        positive_shaping = self.component_totals[:, positive_indices].sum(dim=-1)
        steps_float = torch.clamp(self.control_steps.to(torch.float32), min=1.0)
        mean_jitter = self.normalized_jitter_sum / steps_float
        failure_terminal = terminal & ~success_mask
        violations = {
            "high-return failure": failure_terminal
            & (self.clipped_return >= self.hacking_limits["high_return_failure"]),
            "contact without lift": failure_terminal
            & self.bilateral_consumed
            & (
                self.max_lift_potential
                < self.hacking_limits["contact_without_lift_max_potential"]
            )
            & (
                self.clipped_return
                > self.hacking_limits["contact_without_lift_min_return"]
            ),
            "action jitter farming": terminal
            & ~self.stable_lift_consumed
            & (mean_jitter > self.hacking_limits["jitter_mean_normalized"])
            & (positive_shaping > self.hacking_limits["jitter_positive_shaping"]),
        }
        for name, violation in violations.items():
            _device_assert(~violation.any(), f"reward-hacking check failed: {name}")
        _device_assert(~overflow_mask.any(), "overflow is a hard stop")
        return DirectPPORewardStep(
            reward=clipped,
            unclipped_reward=unclipped,
            components=component_values,
        )

    def consume_episode_summaries(self, indices: tuple[int, ...]) -> tuple[dict[str, Any], ...]:
        """Materialize each terminal episode decomposition exactly once."""

        if not indices or len(set(indices)) != len(indices):
            raise ValueError("episode summary indices must be non-empty and unique")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.num_envs:
                raise ValueError("episode summary lane is outside the batch")
        selected = torch.tensor(indices, dtype=torch.int64, device=self.device)
        if not bool(self.terminal_seen.index_select(0, selected).all()):
            raise DirectPPORewardViolation("episode summary requested before terminal")
        if bool(self.summary_consumed.index_select(0, selected).any()):
            raise DirectPPORewardViolation("episode decomposition was consumed more than once")
        component_rows = self.component_totals.index_select(0, selected).detach().cpu()
        steps = self.control_steps.index_select(0, selected).detach().cpu()
        max_lift = self.max_lift_potential.index_select(0, selected).detach().cpu()
        bilateral = self.bilateral_consumed.index_select(0, selected).detach().cpu()
        stable = self.stable_lift_consumed.index_select(0, selected).detach().cpu()
        jitter = self.normalized_jitter_sum.index_select(0, selected).detach().cpu()
        clipped_returns = self.clipped_return.index_select(0, selected).detach().cpu()
        summaries = []
        for row, lane in enumerate(indices):
            components = {
                name: float(component_rows[row, column])
                for column, name in enumerate(REWARD_COMPONENT_NAMES)
            }
            control_steps = int(steps[row])
            summaries.append(
                {
                    "lane": lane,
                    "components": components,
                    "return": float(clipped_returns[row]),
                    "clipped_return": float(clipped_returns[row]),
                    "unclipped_return": float(component_rows[row].sum()),
                    "control_steps": control_steps,
                    "max_lift_potential": float(max_lift[row]),
                    "bilateral_contact_consumed": bool(bilateral[row]),
                    "stable_lift_consumed": bool(stable[row]),
                    "mean_normalized_action_jitter": float(jitter[row])
                    / max(1, control_steps),
                }
            )
        self.summary_consumed[selected] = True
        return tuple(summaries)


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "DirectPPOReward",
    "DirectPPORewardStep",
    "DirectPPORewardViolation",
    "REWARD_COMPONENT_NAMES",
]
