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

"""Config-driven, fixed-length state features for Dynamic Benchmark experts.

The registry appends optional derived features *after* the frozen base state
vector.  When the registry is empty the encoded vector is byte-identical to the
legacy schema; when features are enabled the composed vector is a new identity
whose feature set, shapes, and mask layout are recorded in the checkpoint.

Every feature consumes only the current observation, the executed action
history (strictly in the past), and previously observed end-effector poses.
There is no access to future state, hidden event times, or offline labels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import (
    object_in_eef_pose_wxyz,
    quaternion_geodesic_wxyz,
)

FEATURE_REGISTRY_SCHEMA_VERSION = "rlinf-dynamic-benchmark-feature-registry-v0.1"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    shape: tuple[int, ...]
    size: int
    params: Mapping[str, Any]
    prerequisites: tuple[str, ...] = ()
    needs_history: bool = False


def _feature_relative_pose(
    observation: Any, _params: Mapping[str, Any], _extra: Mapping[str, Any]
) -> np.ndarray:
    return object_in_eef_pose_wxyz(
        observation.privileged["object_pose_wxyz"],
        observation.privileged["eef_pose_xyzw"],
    )


def _feature_geodesic_error(
    observation: Any, _params: Mapping[str, Any], _extra: Mapping[str, Any]
) -> np.ndarray:
    return np.asarray(
        [
            quaternion_geodesic_wxyz(
                observation.privileged["object_pose_wxyz"][3:],
                observation.privileged["goal_pose_wxyz"][3:],
            )
        ],
        dtype=np.float64,
    )


def _feature_object_vel(
    observation: Any, _params: Mapping[str, Any], _extra: Mapping[str, Any]
) -> np.ndarray:
    return np.asarray(
        observation.privileged["object_twist_world"], dtype=np.float64
    ).reshape(-1)


def _feature_ee_vel(
    _observation: Any, _params: Mapping[str, Any], extra: Mapping[str, Any]
) -> np.ndarray:
    velocity = extra.get("ee_velocity")
    if velocity is None:
        raise ValueError("ee_velocity is unavailable")
    return np.asarray(velocity, dtype=np.float64).reshape(-1)


def _feature_relative_vel(
    observation: Any, _params: Mapping[str, Any], extra: Mapping[str, Any]
) -> np.ndarray:
    object_vel = np.asarray(
        observation.privileged["object_twist_world"], dtype=np.float64
    ).reshape(-1)
    velocity = extra.get("ee_velocity")
    if velocity is None:
        raise ValueError("ee_velocity is unavailable")
    ee_velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    return ee_velocity - object_vel


def _feature_belt_speed(
    observation: Any, _params: Mapping[str, Any], _extra: Mapping[str, Any]
) -> np.ndarray:
    value = np.asarray(
        observation.privileged["belt_surface_velocity_geom"], dtype=np.float64
    ).reshape(-1)
    return np.asarray([float(np.linalg.norm(value))], dtype=np.float64)


def _feature_time_to_goal(
    _observation: Any, _params: Mapping[str, Any], extra: Mapping[str, Any]
) -> np.ndarray:
    time_to_goal = extra.get("time_to_goal_s")
    if time_to_goal is None:
        raise ValueError("time_to_goal is unavailable")
    return np.asarray([float(time_to_goal)], dtype=np.float64)


def _feature_stage(
    _observation: Any, _params: Mapping[str, Any], extra: Mapping[str, Any]
) -> np.ndarray:
    progress = extra.get("stage_progress")
    if progress is None:
        raise ValueError("stage_progress is unavailable")
    return np.asarray([float(progress)], dtype=np.float64)


def _feature_action_history(
    _observation: Any, params: Mapping[str, Any], extra: Mapping[str, Any]
) -> np.ndarray:
    k = int(params["k"])
    history = list(extra.get("action_history", ()))[-k:]
    rows = []
    for action in history:
        array = np.asarray(action, dtype=np.float64).reshape(-1)
        if array.shape != (7,):
            raise ValueError("action history entries must be length 7")
        rows.append(array)
    while len(rows) < k:
        rows.insert(0, np.zeros(7, dtype=np.float64))
    return np.concatenate(rows)


def _feature_goal_error(
    observation: Any, _params: Mapping[str, Any], _extra: Mapping[str, Any]
) -> np.ndarray:
    object_pose = np.asarray(
        observation.privileged["object_pose_wxyz"], dtype=np.float64
    ).reshape(-1)
    goal_pose = np.asarray(
        observation.privileged["goal_pose_wxyz"], dtype=np.float64
    ).reshape(-1)
    return np.asarray(
        [float(np.linalg.norm(object_pose[:3] - goal_pose[:3]))], dtype=np.float64
    )


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "relative_pose": FeatureSpec(
        name="relative_pose",
        shape=(7,),
        size=7,
        params={},
        prerequisites=("object_pose_wxyz", "eef_pose_xyzw"),
    ),
    "geodesic_error": FeatureSpec(
        name="geodesic_error",
        shape=(1,),
        size=1,
        params={},
        prerequisites=("object_pose_wxyz", "goal_pose_wxyz"),
    ),
    "object_vel": FeatureSpec(
        name="object_vel",
        shape=(6,),
        size=6,
        params={},
        prerequisites=("object_twist_world",),
    ),
    "ee_vel": FeatureSpec(
        name="ee_vel",
        shape=(6,),
        size=6,
        params={},
        prerequisites=("eef_pose_xyzw",),
        needs_history=True,
    ),
    "relative_vel": FeatureSpec(
        name="relative_vel",
        shape=(6,),
        size=6,
        params={},
        prerequisites=("object_twist_world", "eef_pose_xyzw"),
        needs_history=True,
    ),
    "belt_speed": FeatureSpec(
        name="belt_speed",
        shape=(1,),
        size=1,
        params={},
        prerequisites=("belt_surface_velocity_geom",),
    ),
    "time_to_goal": FeatureSpec(
        name="time_to_goal",
        shape=(1,),
        size=1,
        params={},
        prerequisites=("object_pose_wxyz", "eef_pose_xyzw", "object_twist_world"),
        needs_history=True,
    ),
    "stage": FeatureSpec(
        name="stage",
        shape=(1,),
        size=1,
        params={},
    ),
    "action_history": FeatureSpec(
        name="action_history",
        shape=(7,),
        size=7,
        params={"k": 3},
        needs_history=True,
    ),
    "goal_error": FeatureSpec(
        name="goal_error",
        shape=(1,),
        size=1,
        params={},
        prerequisites=("object_pose_wxyz", "goal_pose_wxyz"),
    ),
}

FEATURE_IMPL: dict[str, Any] = {
    "relative_pose": _feature_relative_pose,
    "geodesic_error": _feature_geodesic_error,
    "object_vel": _feature_object_vel,
    "ee_vel": _feature_ee_vel,
    "relative_vel": _feature_relative_vel,
    "belt_speed": _feature_belt_speed,
    "time_to_goal": _feature_time_to_goal,
    "stage": _feature_stage,
    "action_history": _feature_action_history,
    "goal_error": _feature_goal_error,
}

FEATURE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    name: spec.prerequisites for name, spec in FEATURE_SPECS.items()
}


def _finite_array(value: Any, *, context: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains NaN or Inf")
    return array


def _parse_feature_config(name: str, value: Any) -> FeatureSpec:
    if name not in FEATURE_SPECS:
        raise ValueError(
            f"unknown state feature {name!r}; available={sorted(FEATURE_SPECS)}"
        )
    spec = FEATURE_SPECS[name]
    if isinstance(value, bool):
        if not value:
            raise ValueError(f"state feature {name!r} uses False; omit it instead")
        params: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        params = dict(value)
    else:
        raise ValueError(f"state feature {name!r} config must be true or a mapping")
    if name == "action_history":
        k = int(params.pop("k", spec.params.get("k", 3)))
        if not 1 <= k <= 8:
            raise ValueError("action_history k must lie in [1, 8]")
        params = {"k": k}
    unknown = sorted(set(params) - set(spec.params))
    if unknown:
        raise ValueError(f"unknown parameters for state feature {name!r}: {unknown}")
    merged = dict(spec.params)
    for key, item in params.items():
        if not isinstance(item, (int, float)) or not np.isfinite(item):
            raise ValueError(f"state feature {name}.{key} must be a finite number")
        merged[key] = item
    shape = list(spec.shape)
    if name == "action_history":
        shape[0] = 7 * int(merged["k"])
    return FeatureSpec(
        name=name,
        shape=tuple(shape),
        size=int(np.prod(shape)),
        params=merged,
        prerequisites=spec.prerequisites,
        needs_history=spec.needs_history,
    )


class FeatureRegistry:
    """Fixed-length derived state features appended after the base vector."""

    def __init__(self, features: Mapping[str, FeatureSpec] | None = None) -> None:
        self.features = dict(features or {})
        self.value_dim = sum(spec.size for spec in self.features.values())
        self.mask_dim = len(self.features)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> FeatureRegistry:
        if config is None:
            return cls({})
        if not isinstance(config, Mapping):
            raise ValueError("features config must be a mapping")
        return cls(
            {
                name: _parse_feature_config(name, value)
                for name, value in sorted(config.items())
            }
        )

    @property
    def is_empty(self) -> bool:
        return not self.features

    def encode(
        self,
        *,
        observation: Any,
        extra: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(values, masks)`` for all enabled features.

        ``extra`` supplies environment-computed quantities: ``ee_velocity``,
        ``time_to_goal_s``, ``stage_progress``, and ``action_history``.
        Features whose prerequisites are missing return zeros with a zero mask
        (the same masking contract as optional base fields).
        """

        parts: list[np.ndarray] = []
        masks: list[float] = []
        for name, spec in self.features.items():
            available = all(
                key in getattr(observation, "privileged", {})
                for key in spec.prerequisites
            )
            if available and name == "action_history":
                available = bool(extra.get("action_history"))
            if available and spec.needs_history and name in {
                "ee_vel",
                "relative_vel",
                "time_to_goal",
            }:
                available = extra.get("ee_velocity") is not None
            if available and name == "time_to_goal":
                available = extra.get("time_to_goal_s") is not None
            if available and name == "stage":
                available = extra.get("stage_progress") is not None
            if not available:
                parts.append(np.zeros(spec.size, dtype=np.float64))
                masks.append(0.0)
                continue
            try:
                value = _finite_array(
                    FEATURE_IMPL[name](observation, spec.params, extra),
                    context=f"feature.{name}",
                )
            except (KeyError, TypeError, ValueError):
                parts.append(np.zeros(spec.size, dtype=np.float64))
                masks.append(0.0)
                continue
            if value.shape != spec.shape:
                raise ValueError(
                    f"feature.{name} shape changed from {spec.shape} to {value.shape}"
                )
            parts.append(value.reshape(-1))
            masks.append(1.0)
        if not parts:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        return np.concatenate(parts), np.asarray(masks, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FEATURE_REGISTRY_SCHEMA_VERSION,
            "features": {
                name: {
                    "shape": list(spec.shape),
                    "size": spec.size,
                    "params": dict(spec.params),
                    "prerequisites": list(spec.prerequisites),
                    "needs_history": spec.needs_history,
                }
                for name, spec in sorted(self.features.items())
            },
            "value_dim": self.value_dim,
            "mask_dim": self.mask_dim,
        }

    def identity_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FEATURE_IMPL",
    "FEATURE_PREREQUISITES",
    "FEATURE_REGISTRY_SCHEMA_VERSION",
    "FEATURE_SPECS",
    "FeatureRegistry",
    "FeatureSpec",
]
