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

SCHEMA_VERSION = "rlinf-dynamic-benchmark-state-v0.2"

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

# Current-state derived features.  Each entry maps a feature name to a
# (shape, callable) pair; the callable takes the current observation and
# returns a finite np.ndarray of that shape.  Features whose prerequisites
# are missing are encoded as zeros with a zero mask entry (the same masking
# contract as optional privileged fields).
def _eef_to_object_pose(observation: Any) -> np.ndarray:
    from .geometry import object_in_eef_pose_wxyz

    return object_in_eef_pose_wxyz(
        observation.privileged["object_pose_wxyz"],
        observation.privileged["eef_pose_xyzw"],
    )


def _eef_to_object_distance(observation: Any) -> np.ndarray:
    pose = _eef_to_object_pose(observation)
    return np.asarray([float(np.linalg.norm(pose[:3]))], dtype=np.float64)


def _fingerpad_midpoint_world(observation: Any) -> np.ndarray:
    left = np.asarray(
        observation.privileged["left_fingerpad_center_world"], dtype=np.float64
    ).reshape(-1)
    right = np.asarray(
        observation.privileged["right_fingerpad_center_world"], dtype=np.float64
    ).reshape(-1)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("fingerpad centers must be length 3")
    return 0.5 * (left + right)


def _grasp_point_offset_world(observation: Any) -> np.ndarray:
    object_pose = np.asarray(
        observation.privileged["object_pose_wxyz"], dtype=np.float64
    )
    return np.asarray(object_pose[:3], dtype=np.float64) - _fingerpad_midpoint_world(
        observation
    )


def _closing_axis_object_alignment(observation: Any) -> np.ndarray:
    from .geometry import closing_axis_object_alignment_rad

    return np.asarray(
        [
            closing_axis_object_alignment_rad(
                observation.privileged["object_pose_wxyz"],
                observation.privileged["fingerpad_closing_axis_world"],
            )
        ],
        dtype=np.float64,
    )


def _object_vertical_position(observation: Any) -> np.ndarray:
    return np.asarray(
        [float(np.asarray(observation.privileged["object_pose_wxyz"])[2])],
        dtype=np.float64,
    )


def _eef_vertical_position(observation: Any) -> np.ndarray:
    return np.asarray(
        [float(np.asarray(observation.privileged["eef_pose_xyzw"])[2])],
        dtype=np.float64,
    )


def _object_yaw_rad(observation: Any) -> np.ndarray:
    from .geometry import quaternion_yaw_wxyz

    return np.asarray(
        [quaternion_yaw_wxyz(np.asarray(observation.privileged["object_pose_wxyz"])[3:])],
        dtype=np.float64,
    )


def _object_xy_velocity(observation: Any) -> np.ndarray:
    twist = np.asarray(
        observation.privileged["object_twist_world"], dtype=np.float64
    ).reshape(-1)
    if twist.shape != (6,):
        raise ValueError("object_twist_world must be length 6")
    return np.asarray(twist[:2], dtype=np.float64)


def _object_angular_velocity_z(observation: Any) -> np.ndarray:
    twist = np.asarray(
        observation.privileged["object_twist_world"], dtype=np.float64
    ).reshape(-1)
    if twist.shape != (6,):
        raise ValueError("object_twist_world must be length 6")
    return np.asarray([twist[5]], dtype=np.float64)


DERIVED_FEATURES: dict[str, tuple[tuple[int, ...], Any]] = {
    "eef_to_object_pose_wxyz": ((7,), _eef_to_object_pose),
    "eef_to_object_distance_m": ((1,), _eef_to_object_distance),
    "fingerpad_midpoint_world": ((3,), _fingerpad_midpoint_world),
    "grasp_point_offset_world_m": ((3,), _grasp_point_offset_world),
    "closing_axis_object_alignment_rad": ((1,), _closing_axis_object_alignment),
    "object_vertical_position_m": ((1,), _object_vertical_position),
    "eef_vertical_position_m": ((1,), _eef_vertical_position),
    "object_yaw_rad": ((1,), _object_yaw_rad),
    "object_xy_velocity_m_s": ((2,), _object_xy_velocity),
    "object_angular_velocity_z_rad_s": ((1,), _object_angular_velocity_z),
}

DERIVED_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "eef_to_object_pose_wxyz": ("object_pose_wxyz", "eef_pose_xyzw"),
    "eef_to_object_distance_m": ("object_pose_wxyz", "eef_pose_xyzw"),
    "fingerpad_midpoint_world": (
        "left_fingerpad_center_world",
        "right_fingerpad_center_world",
    ),
    "grasp_point_offset_world_m": (
        "object_pose_wxyz",
        "left_fingerpad_center_world",
        "right_fingerpad_center_world",
    ),
    "closing_axis_object_alignment_rad": (
        "object_pose_wxyz",
        "fingerpad_closing_axis_world",
    ),
    "object_vertical_position_m": ("object_pose_wxyz",),
    "eef_vertical_position_m": ("eef_pose_xyzw",),
    "object_yaw_rad": ("object_pose_wxyz",),
    "object_xy_velocity_m_s": ("object_twist_world",),
    "object_angular_velocity_z_rad_s": ("object_twist_world",),
}


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
        derived_fields: Sequence[StateField] = (),
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
        for field in derived_fields:
            if field.source != "derived":
                raise ValueError(
                    f"derived field {field.name!r} must use source 'derived'"
                )
        self.derived_fields = tuple(derived_fields)
        self.factor_fields = tuple(factor_fields)
        self.categorical_factors = {
            name: tuple(values) for name, values in sorted(categorical_factors.items())
        }
        names = [
            (field.source, field.name)
            for field in (*self.fields, *self.derived_fields, *self.factor_fields)
        ]
        if len(set(names)) != len(names):
            raise ValueError("state schema fields must be unique")
        self.value_dim = (
            len(self.task_ids)
            + 2
            + sum(field.size for field in self.fields)
            + sum(field.size for field in self.derived_fields)
            + sum(field.size for field in self.factor_fields)
        )
        self.mask_dim = (
            len(self.fields) + len(self.derived_fields) + len(self.factor_fields)
        )
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
        derived_features: Sequence[str] = (),
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

        derived_fields = []
        for name in derived_features:
            if name not in DERIVED_FEATURES:
                raise ValueError(f"unknown derived state feature {name!r}")
            shape, _ = DERIVED_FEATURES[name]
            derived_fields.append(StateField("derived", name, shape, int(np.prod(shape))))

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
            derived_fields=derived_fields,
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

    def _encode_derived(
        self, field: StateField, observation: Any
    ) -> tuple[np.ndarray, bool]:
        prerequisites = DERIVED_PREREQUISITES.get(field.name, ())
        privileged = getattr(observation, "privileged", {})
        if any(name not in privileged for name in prerequisites):
            return np.zeros(field.size, dtype=np.float64), False
        try:
            _, function = DERIVED_FEATURES[field.name]
            value = _numeric_array(
                function(observation), context=f"derived.{field.name}"
            )
        except (KeyError, ValueError, TypeError):
            return np.zeros(field.size, dtype=np.float64), False
        if value.shape != field.shape:
            raise ValueError(
                f"derived.{field.name} shape changed from {field.shape} to {value.shape}"
            )
        return value.reshape(-1), True

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
        for field in self.derived_fields:
            value, available = self._encode_derived(field, observation)
            parts.append(value)
            masks.append(1.0 if available else 0.0)
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
            "derived_fields": [asdict(field) for field in self.derived_fields],
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
    "DERIVED_FEATURES",
    "DERIVED_PREREQUISITES",
    "DynamicBenchmarkStateSchema",
    "SCHEMA_VERSION",
    "StateField",
]
