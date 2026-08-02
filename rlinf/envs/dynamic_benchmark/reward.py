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

"""Current-state reward contract for Dynamic Benchmark expert training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

REWARD_SCHEMA_VERSION = "rlinf-dynamic-benchmark-reward-v0.2"
DEFAULT_SAFETY_FAILURES = frozenset(
    {
        "drop",
        "driver_blocked",
        "early_robot_contact",
        "extra_impact",
        "extra_striker_impact",
        "forbidden_wire_contact",
        "invalid_state",
        "object_goal_collision_unsafe",
        "release_impulse",
        "striker_trap",
        "tray_contact",
        "trap_or_block",
        "unsafe_stage_block",
        "unsafe_contact",
        "unstable_tipping",
        "wand_drop",
        "workspace_exit",
        "downstream_exit",
        "wrong_striker_contact",
    }
)


class DynamicBenchmarkReward:
    """Potential-difference reward with terminal and action-cost terms."""

    def __init__(
        self,
        *,
        success_stages: Sequence[str],
        progress_scale: float = 5.0,
        success_bonus: float = 10.0,
        failure_penalty: float = -3.0,
        safety_penalty: float = -10.0,
        timeout_penalty: float = -1.0,
        step_penalty: float = -0.01,
        action_l2_scale: float = -0.001,
        safety_failures: Sequence[str] = tuple(DEFAULT_SAFETY_FAILURES),
    ) -> None:
        self.success_stages = tuple(success_stages)
        if not self.success_stages:
            raise ValueError("success_stages must not be empty")
        self.progress_scale = float(progress_scale)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.safety_penalty = float(safety_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.step_penalty = float(step_penalty)
        self.action_l2_scale = float(action_l2_scale)
        self.safety_failures = frozenset(safety_failures)
        self._previous_potential = 0.0

    def reset(self) -> None:
        self._previous_potential = 0.0

    def state_dict(self) -> dict[str, float | str]:
        """Return the minimal state required for bit-exact reward resume."""

        return {
            "schema_version": REWARD_SCHEMA_VERSION,
            "previous_potential": self._previous_potential,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore reward potential after validating its schema and range."""

        if state.get("schema_version") != REWARD_SCHEMA_VERSION:
            raise ValueError("unsupported Dynamic Benchmark reward checkpoint schema")
        previous = float(state["previous_potential"])
        if not np.isfinite(previous) or not 0.0 <= previous <= 1.0:
            raise ValueError("reward previous_potential must be finite and in [0, 1]")
        self._previous_potential = previous

    def potential(
        self, *, event_names: Sequence[str], active_stage_progress: float
    ) -> float:
        completed = 0
        observed = set(event_names)
        for stage in self.success_stages:
            if stage not in observed:
                break
            completed += 1
        if completed == len(self.success_stages):
            return 1.0
        active = float(np.clip(active_stage_progress, 0.0, 1.0))
        return (completed + active) / len(self.success_stages)

    def step(
        self,
        *,
        action: np.ndarray,
        event_names: Sequence[str],
        active_stage_progress: float,
        success: bool,
        terminated: bool,
        truncated: bool,
        termination_reason: str | None,
    ) -> tuple[float, dict[str, float]]:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (7,) or not np.all(np.isfinite(action_array)):
            raise ValueError("reward action must be a finite E7 vector")
        current_potential = self.potential(
            event_names=event_names,
            active_stage_progress=active_stage_progress,
        )
        components = {
            "progress": self.progress_scale
            * (current_potential - self._previous_potential),
            "step": self.step_penalty,
            "action_l2": self.action_l2_scale * float(np.square(action_array).sum()),
            "success": self.success_bonus if success else 0.0,
            "failure": 0.0,
            "safety": 0.0,
            "timeout": 0.0,
        }
        if terminated and not success:
            if termination_reason in self.safety_failures:
                components["safety"] = self.safety_penalty
            else:
                components["failure"] = self.failure_penalty
        elif truncated:
            components["timeout"] = self.timeout_penalty
        self._previous_potential = current_potential
        total = float(sum(components.values()))
        components["total"] = total
        components["potential"] = current_potential
        return total, components

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": REWARD_SCHEMA_VERSION,
            "success_stages": list(self.success_stages),
            "progress_scale": self.progress_scale,
            "success_bonus": self.success_bonus,
            "failure_penalty": self.failure_penalty,
            "safety_penalty": self.safety_penalty,
            "timeout_penalty": self.timeout_penalty,
            "step_penalty": self.step_penalty,
            "action_l2_scale": self.action_l2_scale,
            "safety_failures": sorted(self.safety_failures),
        }


__all__ = [
    "DEFAULT_SAFETY_FAILURES",
    "DynamicBenchmarkReward",
    "REWARD_SCHEMA_VERSION",
]
