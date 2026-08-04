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

"""Probe the RLOPT shared infrastructure for a given YAML config.

Prints (and optionally writes as JSON):

* enabled features with shapes, masks, and identity SHA-256;
* enabled reward components with weights/parameters and identity SHA-256;
* a no-future-leakage check on the feature registry;
* an independent reward recompute check (max absolute error < 1e-7);
* the composed state dimension for the given feature set.

The probe is CPU-only and requires no benchmark runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from rlinf.envs.dynamic_benchmark.feature_registry import FeatureRegistry
from rlinf.envs.dynamic_benchmark.reward_registry import RewardRegistry


def _synthetic_observation() -> SimpleNamespace:
    return SimpleNamespace(
        task_id="t1_so3",
        policy_step=4,
        proprio={"robot0_proprio_state": np.asarray([0.1, -0.2])},
        privileged={
            "object_pose_wxyz": np.asarray([0.5, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
            "eef_pose_xyzw": np.asarray([0.4, 0.2, 0.25, 0.0, 0.0, 0.0, 1.0]),
            "object_twist_world": np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "goal_pose_wxyz": np.asarray([0.5, 0.2, 0.35, 1.0, 0.0, 0.0, 0.0]),
            "belt_surface_velocity_geom": np.asarray([0.05, 0.0, 0.0]),
        },
    )


def _synthetic_extra() -> dict[str, Any]:
    return {
        "ee_velocity": np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "time_to_goal_s": 0.5,
        "stage_progress": 0.75,
        "action_history": [
            np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
    }


def _synthetic_reward_inputs() -> dict[str, Any]:
    return {
        "action_l2": 1.0,
        "completion": 0.95,
        "stage_progress": 0.6,
        "geodesic_error_rad": 0.25,
        "relative_translation_error_m": 0.02,
        "relative_rotation_error_rad": 0.1,
        "relative_velocity_norm_m_s": 0.3,
        "distance_m": 0.03,
        "time_to_goal_s": 0.15,
    }


def _no_leakage_check(features: FeatureRegistry) -> dict[str, Any]:
    current = _synthetic_observation()
    future = SimpleNamespace(
        task_id=current.task_id,
        policy_step=current.policy_step,
        proprio=dict(current.proprio),
        privileged=dict(current.privileged),
    )
    future.privileged["future_object_pose_wxyz"] = np.ones(7)
    future.privileged["future_object_twist_world"] = np.ones(6)
    future.privileged["hidden_event_time_s"] = np.asarray([3.0])
    current_values, current_masks = features.encode(
        observation=current, extra=_synthetic_extra()
    )
    future_values, future_masks = features.encode(
        observation=future, extra=_synthetic_extra()
    )
    return {
        "values_identical": bool(np.array_equal(current_values, future_values)),
        "masks_identical": bool(np.array_equal(current_masks, future_masks)),
    }


def _recompute_check(rewards: RewardRegistry) -> dict[str, Any]:
    inputs = _synthetic_reward_inputs()
    total, values, recorded = rewards.step(inputs)
    recomputed_total, recomputed_values = RewardRegistry.recompute(
        rewards.components, recorded
    )
    max_abs_error = max(
        [abs(total - recomputed_total)]
        + [
            abs(values[name] - recomputed_values[name])
            for name in sorted(set(values) | set(recomputed_values))
        ]
    )
    return {
        "total": total,
        "max_abs_error": float(max_abs_error),
        "passes_1e_7": bool(max_abs_error < 1e-7),
        "components": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML file containing optional 'features' and 'reward_components' sections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path for the probe report.",
    )
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a YAML mapping")
    features = FeatureRegistry.from_config(payload.get("features", {}))
    rewards = RewardRegistry.from_config(payload.get("reward_components", {}))

    report: dict[str, Any] = {
        "config": str(args.config.resolve()),
        "features": {
            "enabled": {
                name: {
                    "shape": list(spec.shape),
                    "size": spec.size,
                    "params": dict(spec.params),
                }
                for name, spec in features.features.items()
            },
            "value_dim": features.value_dim,
            "mask_dim": features.mask_dim,
            "identity_sha256": features.identity_sha256(),
            "no_future_leakage": _no_leakage_check(features),
        },
        "reward_components": {
            "enabled": {
                name: {
                    "weight": component.weight,
                    "params": dict(component.params),
                }
                for name, component in rewards.components.items()
                if component.weight > 0.0
            },
            "identity_sha256": rewards.identity_sha256(),
            "independent_recompute": _recompute_check(rewards),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
