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

"""Frozen privileged-state vector schema for Dynamic Benchmark experts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SCHEMA_VERSION = "rlinf-dynamic-benchmark-state-v0.1"

# Explicit allowlisting makes future-oracle additions fail closed: a newly
# added privileged field is not silently exposed to an expert policy.
ALLOWED_PROPRIO_KEYS = ("robot0_proprio_state",)
ALLOWED_PRIVILEGED_KEYS = (
    "object_pose_wxyz",
    "object_twist_world",
    "belt_surface_velocity_geom",
    "goal_pose_wxyz",
    "goal_twist_world",
    "buzz_frame_pose_wxyz",
    "buzz_frame_phase_speed",
    "ramp_pose_wxyz",
    "ramp_angle_rad",
    "ramp_hidden_physics",
    "ramp_catch_deck_pose_wxyz",
    "ramp_catch_region_bounds_xy_m",
    "ramp_exit_region_bounds_xy_m",
    "t5_striker_pose_wxyz",
    "t5_striker_joint_state",
    "t5_event_state",
    "driver_reference_pose_wxyz",
    "driver_reference_twist_world",
    "driver_applied_wrench_world",
    "eef_pose_xyzw",
    "left_fingerpad_center_world",
    "right_fingerpad_center_world",
    "fingerpad_closing_axis_world",
    "fingerpad_contact_flags",
    "capture_post_hold_contact_valid",
    "capture_bilateral_steps",
    "capture_max_bilateral_steps",
    "unsafe_occluder_contact",
    "placement_full_geometry_contained",
    "placement_entire_object_below_entrance",
    "placement_released",
    "placement_relative_speed",
    "placement_minimum_containment_clearance_m",
    "buzz_path_progress",
    "buzz_loop_distances_m",
    "buzz_loop_orientation_error_rad",
    "buzz_wire_clearance_m",
    "buzz_wire_contact",
    "ramp_capture_state",
    "ramp_regime_diagnostics",
    "t5_capture_state",
    "t5_event_diagnostics",
    "action_timing_counts",
)
DEFAULT_CATEGORICAL_FACTORS = {"speed_class": ("normal", "high")}


@dataclass(frozen=True)
class StateField:
    source: str
    name: str
    shape: tuple[int, ...]
    size: int


def _numeric_array(value: Any, *, context: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains NaN or Inf")
    return array


def _canonicalize_quaternion(name: str, value: np.ndarray) -> np.ndarray:
    canonical = np.array(value, dtype=np.float64, copy=True)
    if name.endswith("pose_wxyz"):
        poses = canonical.reshape(-1, 7)
        poses[poses[:, 3] < 0.0, 3:] *= -1.0
    elif name == "eef_pose_xyzw":
        poses = canonical.reshape(-1, 7)
        poses[poses[:, 6] < 0.0, 3:] *= -1.0
    elif name.endswith("quat_wxyz"):
        quaternions = canonical.reshape(-1, 4)
        quaternions[quaternions[:, 0] < 0.0] *= -1.0
    return canonical


class DynamicBenchmarkStateSchema:
    """Task-aware state vector with explicit fields, factors, and validity masks."""

    def __init__(
        self,
        *,
        task_id: str,
        task_ids: Sequence[str],
        fields: Sequence[StateField],
        factor_fields: Sequence[StateField],
        categorical_factors: Mapping[str, Sequence[str]],
    ) -> None:
        if task_id not in task_ids:
            raise ValueError(
                f"task {task_id!r} is absent from the frozen task vocabulary"
            )
        self.task_id = task_id
        self.task_ids = tuple(task_ids)
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")
        self.fields = tuple(fields)
        self.factor_fields = tuple(factor_fields)
        self.categorical_factors = {
            name: tuple(values) for name, values in sorted(categorical_factors.items())
        }
        names = [
            (field.source, field.name) for field in (*self.fields, *self.factor_fields)
        ]
        if len(set(names)) != len(names):
            raise ValueError("state schema fields must be unique")
        self.value_dim = (
            len(self.task_ids)
            + 2
            + sum(field.size for field in self.fields)
            + sum(field.size for field in self.factor_fields)
        )
        self.mask_dim = len(self.fields) + len(self.factor_fields)
        self.state_dim = self.value_dim + self.mask_dim

    @classmethod
    def from_observation(
        cls,
        *,
        task_id: str,
        task_ids: Sequence[str],
        observation: Any,
        factors: Mapping[str, Any],
        categorical_factors: Mapping[str, Sequence[str]] | None = None,
    ) -> DynamicBenchmarkStateSchema:
        categories = dict(DEFAULT_CATEGORICAL_FACTORS)
        if categorical_factors is not None:
            categories.update(
                {name: tuple(values) for name, values in categorical_factors.items()}
            )
        fields = []
        for source, values, allowed in (
            ("proprio", observation.proprio, ALLOWED_PROPRIO_KEYS),
            ("privileged", observation.privileged, ALLOWED_PRIVILEGED_KEYS),
        ):
            for name in allowed:
                if name not in values:
                    continue
                array = _numeric_array(values[name], context=f"{source}.{name}")
                fields.append(
                    StateField(source, name, tuple(array.shape), int(array.size))
                )
        required = {
            ("proprio", "robot0_proprio_state"),
            ("privileged", "object_pose_wxyz"),
        }
        present = {(field.source, field.name) for field in fields}
        if not required <= present:
            raise ValueError(
                f"observation is missing required state fields: {sorted(required - present)}"
            )

        factor_fields = []
        for name, value in sorted(factors.items()):
            if isinstance(value, str):
                values = categories.get(name)
                if values is None or value not in values:
                    raise ValueError(
                        f"categorical factor {name!r}={value!r} lacks a frozen vocabulary"
                    )
                factor_fields.append(
                    StateField("factor", name, (len(values),), len(values))
                )
            else:
                array = _numeric_array(value, context=f"factor.{name}")
                factor_fields.append(
                    StateField("factor", name, tuple(array.shape), int(array.size))
                )
        return cls(
            task_id=task_id,
            task_ids=task_ids,
            fields=fields,
            factor_fields=factor_fields,
            categorical_factors=categories,
        )

    def _encode_field(self, field: StateField, value: Any) -> np.ndarray:
        array = _numeric_array(value, context=f"{field.source}.{field.name}")
        if tuple(array.shape) != field.shape:
            raise ValueError(
                f"{field.source}.{field.name} shape changed from {field.shape} to {array.shape}"
            )
        return _canonicalize_quaternion(field.name, array).reshape(-1)

    def _encode_factor(self, field: StateField, value: Any) -> np.ndarray:
        if field.name in self.categorical_factors:
            vocabulary = self.categorical_factors[field.name]
            if value not in vocabulary:
                raise ValueError(
                    f"factor {field.name!r}={value!r} is outside vocabulary {vocabulary}"
                )
            encoded = np.zeros(len(vocabulary), dtype=np.float64)
            encoded[vocabulary.index(value)] = 1.0
            return encoded
        return self._encode_field(field, value)

    def encode(
        self,
        *,
        observation: Any,
        factors: Mapping[str, Any],
        horizon_steps: int,
    ) -> np.ndarray:
        if observation.task_id != self.task_id:
            raise ValueError(
                f"schema task {self.task_id!r} cannot encode {observation.task_id!r}"
            )
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        task_one_hot = np.zeros(len(self.task_ids), dtype=np.float64)
        task_one_hot[self.task_ids.index(self.task_id)] = 1.0
        progress = float(np.clip(observation.policy_step / horizon_steps, 0.0, 1.0))
        parts = [task_one_hot, np.asarray([progress, 1.0 - progress])]
        masks = []
        sources = {
            "proprio": observation.proprio,
            "privileged": observation.privileged,
        }
        for field in self.fields:
            values = sources[field.source]
            if field.name in values:
                parts.append(self._encode_field(field, values[field.name]))
                masks.append(1.0)
            else:
                parts.append(np.zeros(field.size, dtype=np.float64))
                masks.append(0.0)
        for field in self.factor_fields:
            if field.name in factors:
                parts.append(self._encode_factor(field, factors[field.name]))
                masks.append(1.0)
            else:
                parts.append(np.zeros(field.size, dtype=np.float64))
                masks.append(0.0)
        parts.append(np.asarray(masks, dtype=np.float64))
        state = np.concatenate(parts).astype(np.float32, copy=False)
        if state.shape != (self.state_dim,) or not np.all(np.isfinite(state)):
            raise RuntimeError(
                f"encoded state violates schema: shape={state.shape}, expected={self.state_dim}"
            )
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "task_ids": list(self.task_ids),
            "fields": [asdict(field) for field in self.fields],
            "factor_fields": [asdict(field) for field in self.factor_fields],
            "categorical_factors": {
                name: list(values) for name, values in self.categorical_factors.items()
            },
            "value_dim": self.value_dim,
            "mask_dim": self.mask_dim,
            "state_dim": self.state_dim,
        }


__all__ = [
    "ALLOWED_PRIVILEGED_KEYS",
    "ALLOWED_PROPRIO_KEYS",
    "DEFAULT_CATEGORICAL_FACTORS",
    "DynamicBenchmarkStateSchema",
    "SCHEMA_VERSION",
    "StateField",
]
