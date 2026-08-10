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

"""Config-driven dense reward components for Dynamic Benchmark experts.

The registry is the single place where optional reward terms are defined.  Every
component is a pure function of a per-step ``inputs`` mapping (assembled by the
environment from the current observation, the executed action, and previously
recorded potential values), so each term is independently recomputable and has
no access to future information.

Default: the registry is empty (``{}``), which adds no reward and changes no
baseline semantics.  Each enabled component must have a non-negative finite
``weight``; the component's canonical raw value carries its own sign (bonus >= 0
or penalty <= 0).  A weight of ``0.0`` disables the component entirely.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

REWARD_REGISTRY_SCHEMA_VERSION = "rlinf-dynamic-benchmark-reward-registry-v0.1"


def _finite_float(value: Any, *, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


@dataclass(frozen=True)
class RewardComponent:
    """One configured component with its weight and validated parameters."""

    name: str
    weight: float
    params: Mapping[str, Any]

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight, **dict(self.params)}


def _r_ori_geodesic(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> float:
    error = inputs.get("geodesic_error_rad")
    if error is None or not math.isfinite(float(error)):
        return 0.0
    scale = float(params["scale_rad"])
    return -min(float(error) / scale, 1.0)


def _r_rel_pose(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> float:
    translation = inputs.get("relative_translation_error_m")
    rotation = inputs.get("relative_rotation_error_rad")
    if translation is None or rotation is None:
        return 0.0
    if not math.isfinite(float(translation)) or not math.isfinite(float(rotation)):
        return 0.0
    return -min(
        float(translation) / float(params["scale_pos_m"])
        + float(rotation) / float(params["scale_rot_rad"]),
        1.0,
    )


def _r_effort(inputs: Mapping[str, Any], _params: Mapping[str, Any]) -> float:
    action_l2 = inputs.get("action_l2")
    if action_l2 is None or not math.isfinite(float(action_l2)):
        return 0.0
    return -float(action_l2)


def _r_completion_shaping(
    inputs: Mapping[str, Any], params: Mapping[str, Any]
) -> float:
    completion = inputs.get("completion")
    if completion is None or not math.isfinite(float(completion)):
        return 0.0
    threshold = float(params["near_threshold"])
    scale = float(params["scale"])
    if float(completion) < threshold:
        return 0.0
    return -scale * max(0.0, 1.0 - float(completion))


def _r_vel_align(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> float:
    velocity_norm = inputs.get("relative_velocity_norm_m_s")
    if velocity_norm is None or not math.isfinite(float(velocity_norm)):
        return 0.0
    return -float(velocity_norm) / float(params["speed_scale_m_s"])


def _r_timing(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> float:
    distance = inputs.get("distance_m")
    time_to_goal = inputs.get("time_to_goal_s")
    if distance is None or time_to_goal is None:
        return 0.0
    if not math.isfinite(float(distance)) or not math.isfinite(float(time_to_goal)):
        return 0.0
    if float(distance) > float(params["dist_threshold_m"]):
        return 0.0
    deviation = abs(float(time_to_goal) - float(params["target_ttc_s"]))
    return -min(deviation / float(params["ttc_scale_s"]), 1.0)


def _r_stage(inputs: Mapping[str, Any], _params: Mapping[str, Any]) -> float:
    current = inputs.get("stage_progress")
    previous = inputs.get("previous_stage_progress")
    if current is None or previous is None:
        return 0.0
    if not math.isfinite(float(current)) or not math.isfinite(float(previous)):
        return 0.0
    return float(current) - float(previous)


COMPONENT_IMPL: dict[str, Any] = {
    "r_ori_geodesic": _r_ori_geodesic,
    "r_rel_pose": _r_rel_pose,
    "r_effort": _r_effort,
    "r_completion_shaping": _r_completion_shaping,
    "r_vel_align": _r_vel_align,
    "r_timing": _r_timing,
    "r_stage": _r_stage,
}

COMPONENT_DEFAULT_PARAMS: dict[str, Mapping[str, Any]] = {
    "r_ori_geodesic": {"scale_rad": 1.0},
    "r_rel_pose": {"scale_pos_m": 0.1, "scale_rot_rad": 1.0},
    "r_effort": {},
    "r_completion_shaping": {"near_threshold": 0.9, "scale": 1.0},
    "r_vel_align": {"speed_scale_m_s": 1.0},
    "r_timing": {"dist_threshold_m": 0.05, "target_ttc_s": 0.1, "ttc_scale_s": 0.1},
    "r_stage": {},
}


def _validate_params(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    defaults = dict(COMPONENT_DEFAULT_PARAMS[name])
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise ValueError(f"unknown parameters for reward component {name!r}: {unknown}")
    merged = dict(defaults)
    for key, value in params.items():
        merged[key] = _finite_float(value, context=f"{name}.{key}")
    for key in ("near_threshold",):
        if key in merged and not 0.0 <= float(merged[key]) <= 1.0:
            raise ValueError(f"{name}.{key} must lie in [0, 1]")
    for key in ("dist_threshold_m", "ttc_scale_s", "scale_rad", "scale_pos_m",
                "scale_rot_rad", "speed_scale_m_s", "scale", "target_ttc_s"):
        if key in merged and float(merged[key]) <= 0.0:
            raise ValueError(f"{name}.{key} must be positive")
    return merged


def _parse_component_config(name: str, value: Any) -> RewardComponent:
    if name not in COMPONENT_IMPL:
        raise ValueError(
            f"unknown reward component {name!r}; available={sorted(COMPONENT_IMPL)}"
        )
    if isinstance(value, bool) or value is None:
        raise ValueError(f"reward component {name!r} requires an explicit weight")
    if isinstance(value, (int, float)):
        weight = _finite_float(value, context=f"{name}.weight")
        params: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        if "weight" not in value:
            raise ValueError(f"reward component {name!r} requires a 'weight'")
        weight = _finite_float(value["weight"], context=f"{name}.weight")
        params = {k: v for k, v in value.items() if k != "weight"}
    else:
        raise ValueError(f"reward component {name!r} config must be a number or mapping")
    if weight < 0.0:
        raise ValueError(
            f"reward component {name!r} weight must be non-negative "
            "(raw values are already signed)"
        )
    validated = _validate_params(name, params)
    return RewardComponent(name=name, weight=weight, params=validated)


class RewardRegistry:
    """Per-environment dense reward components with exact resume state."""

    def __init__(self, components: Mapping[str, RewardComponent] | None = None) -> None:
        self.components = dict(components or {})
        self._previous_stage_progress = 0.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> RewardRegistry:
        if config is None:
            return cls({})
        if not isinstance(config, Mapping):
            raise ValueError("reward_components config must be a mapping")
        components = {
            name: _parse_component_config(name, value)
            for name, value in sorted(config.items())
        }
        return cls(components)

    @property
    def enabled_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, component in sorted(self.components.items())
            if component.weight > 0.0
        )

    @property
    def is_empty(self) -> bool:
        return not self.enabled_names

    def reset(self) -> None:
        self._previous_stage_progress = 0.0

    def step(
        self, inputs: Mapping[str, Any]
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Compute enabled terms and the registry total.

        Returns ``(total, component_values, recorded_inputs)``.  ``recorded_inputs``
        contains every scalar needed for an independent recompute, including the
        previous stage progress consumed by :func:`_r_stage`.
        """

        effective = dict(inputs)
        effective["previous_stage_progress"] = self._previous_stage_progress
        recorded = dict(effective)
        values: dict[str, float] = {}
        total = 0.0
        for name in self.enabled_names:
            component = self.components[name]
            raw = COMPONENT_IMPL[name](effective, component.params)
            if not math.isfinite(raw):
                raise RuntimeError(f"reward component {name!r} produced a non-finite value")
            value = component.weight * raw
            values[name] = float(value)
            total += float(value)
        if inputs.get("stage_progress") is not None:
            self._previous_stage_progress = float(inputs["stage_progress"])
        return float(total), values, recorded

    @staticmethod
    def recompute(
        components: Mapping[str, RewardComponent],
        recorded: Mapping[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """Recompute the registry total from recorded per-step inputs."""

        total = 0.0
        values: dict[str, float] = {}
        for name, component in sorted(components.items()):
            if component.weight <= 0.0:
                continue
            raw = COMPONENT_IMPL[name](recorded, component.params)
            if not math.isfinite(raw):
                raise RuntimeError(
                    f"reward component {name!r} recompute produced a non-finite value"
                )
            value = component.weight * raw
            values[name] = value
            total += value
        return float(total), values

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REWARD_REGISTRY_SCHEMA_VERSION,
            "previous_stage_progress": self._previous_stage_progress,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != REWARD_REGISTRY_SCHEMA_VERSION:
            raise ValueError("unsupported reward registry checkpoint schema")
        value = float(state["previous_stage_progress"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("reward registry previous_stage_progress must be in [0, 1]")
        self._previous_stage_progress = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REWARD_REGISTRY_SCHEMA_VERSION,
            "components": {
                name: component.identity()
                for name, component in sorted(self.components.items())
            },
        }

    def identity_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "COMPONENT_DEFAULT_PARAMS",
    "COMPONENT_IMPL",
    "REWARD_REGISTRY_SCHEMA_VERSION",
    "RewardComponent",
    "RewardRegistry",
]
