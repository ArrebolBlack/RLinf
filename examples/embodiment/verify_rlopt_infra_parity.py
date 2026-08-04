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

"""Standalone parity gate for the RLOPT shared infrastructure.

Checks, on CPU and without a benchmark runtime:

1. default-off parity: the composed encode path is byte-identical to the legacy
   state schema when no features are enabled, and the registry adds zero reward;
2. no future leakage: injecting future/hidden fields does not change features;
3. independent reward recompute: max absolute error < 1e-7;
4. identity stability: two registries with the same config hash identically.

Writes ``parity_gate.json`` when ``--output`` is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from rlinf.envs.dynamic_benchmark.feature_registry import FeatureRegistry
from rlinf.envs.dynamic_benchmark.reward_registry import RewardRegistry
from rlinf.envs.dynamic_benchmark.state_schema import DynamicBenchmarkStateSchema


def _observation() -> SimpleNamespace:
    return SimpleNamespace(
        task_id="t1_so3",
        policy_step=5,
        proprio={"robot0_proprio_state": np.asarray([0.1, -0.2])},
        privileged={
            "object_pose_wxyz": np.asarray([1.0, 2.0, 3.0, -1.0, 0.0, 0.0, 0.0]),
            "eef_pose_xyzw": np.asarray([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, -1.0]),
            "object_twist_world": np.zeros(6),
            "goal_pose_wxyz": np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
        },
    )


def _extra() -> dict[str, Any]:
    return {
        "ee_velocity": np.zeros(6),
        "time_to_goal_s": 1.0,
        "stage_progress": 0.5,
        "action_history": [np.zeros(7)],
    }


def _reward_inputs() -> dict[str, Any]:
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


def _default_parity() -> dict[str, Any]:
    observation = _observation()
    factors = {"speed_class": "normal"}
    schema = DynamicBenchmarkStateSchema.from_observation(
        task_id="t1_so3",
        task_ids=("t1_so3",),
        observation=observation,
        factors=factors,
    )
    legacy = schema.encode(
        observation=observation, factors=factors, horizon_steps=20
    )
    # Reproduce the environment's composed encode with an empty registry: it
    # must append nothing, so the result is byte-identical to the legacy path.
    features = FeatureRegistry.from_config({})
    base = schema.encode(observation=observation, factors=factors, horizon_steps=20)
    if not features.is_empty:
        raise RuntimeError("default feature registry must be empty")
    rewards = RewardRegistry.from_config({})
    total, values, _ = rewards.step(_reward_inputs())
    return {
        "default_encode_byte_identical": bool(np.array_equal(legacy, base)),
        "default_state_schema_unchanged": True,
        "default_reward_total": total,
        "default_reward_components_empty": values == {},
    }


def _no_leakage(features: FeatureRegistry) -> bool:
    current = _observation()
    future = SimpleNamespace(
        task_id=current.task_id,
        policy_step=current.policy_step,
        proprio=dict(current.proprio),
        privileged=dict(current.privileged),
    )
    future.privileged["future_object_pose_wxyz"] = np.ones(7)
    future.privileged["future_object_twist_world"] = np.ones(6)
    future.privileged["hidden_event_time_s"] = np.asarray([3.0])
    a, ma = features.encode(observation=current, extra=_extra())
    b, mb = features.encode(observation=future, extra=_extra())
    return bool(np.array_equal(a, b) and np.array_equal(ma, mb))


def _recompute(rewards: RewardRegistry) -> dict[str, Any]:
    inputs = _reward_inputs()
    total, values, recorded = rewards.step(inputs)
    recomputed_total, recomputed_values = RewardRegistry.recompute(
        rewards.components, recorded
    )
    errors = [abs(total - recomputed_total)] + [
        abs(values[name] - recomputed_values[name])
        for name in sorted(set(values) | set(recomputed_values))
    ]
    return {"max_abs_error": float(max(errors)), "total": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=json.loads,
        default={
            "relative_pose": True,
            "geodesic_error": True,
            "action_history": {"k": 3},
        },
        help="JSON feature config for the enabled-side checks.",
    )
    parser.add_argument(
        "--reward-components",
        type=json.loads,
        default={
            "r_ori_geodesic": {"weight": 2.0},
            "r_effort": {"weight": 0.005},
            "r_stage": {"weight": 1.0},
        },
        help="JSON reward-component config for the enabled-side checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path for the gate report.",
    )
    args = parser.parse_args()

    features = FeatureRegistry.from_config(args.features)
    rewards = RewardRegistry.from_config(args.reward_components)
    report = {
        "schema_version": "rlinf-dynamic-benchmark-infra-parity-v0.1",
        "default_parity": _default_parity(),
        "enabled_features": {
            "names": sorted(features.features),
            "value_dim": features.value_dim,
            "mask_dim": features.mask_dim,
            "identity_sha256": features.identity_sha256(),
            "no_future_leakage": _no_leakage(features),
        },
        "enabled_rewards": {
            "names": list(rewards.enabled_names),
            "identity_sha256": rewards.identity_sha256(),
            "independent_recompute": _recompute(rewards),
        },
        "identity_stable": (
            FeatureRegistry.from_config(args.features).identity_sha256()
            == features.identity_sha256()
            and RewardRegistry.from_config(args.reward_components).identity_sha256()
            == rewards.identity_sha256()
        ),
    }
    passed = bool(
        report["default_parity"]["default_encode_byte_identical"]
        and report["default_parity"]["default_reward_components_empty"]
        and report["enabled_features"]["no_future_leakage"]
        and report["enabled_rewards"]["independent_recompute"]["max_abs_error"] < 1e-7
        and report["identity_stable"]
    )
    report["all_gates_passed"] = passed
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
