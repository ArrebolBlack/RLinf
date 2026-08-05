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

"""Default-off parity and checkpoint/resume identity gates for the RLOPT infra."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv
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


def _bare_env() -> DynamicBenchmarkEnv:
    env = object.__new__(DynamicBenchmarkEnv)
    env.num_envs = 1
    env.task_id = "t1_so3"
    env.split_name = "train"
    env.base_manifest_seed = 20261050
    env.manifest_size = 1024
    env.image_size = 64
    env.camera_observations = False
    env.worker_threads = 1
    env.worker_processes = 0
    env.process_start_method = "spawn"
    env.horizon_steps = 20
    env.feature_registry = FeatureRegistry.from_config({})
    env.reward_registries = [RewardRegistry.from_config({})]
    env._state_schema = DynamicBenchmarkStateSchema.from_observation(
        task_id="t1_so3",
        task_ids=("t1_so3",),
        observation=_observation(),
        factors={"speed_class": "normal"},
    )
    env._action_histories = [[]]
    env._previous_eef_poses = [None]
    env._ee_velocities = [None]
    env._time_to_goals = [None]
    env._distances = [None]
    env._relative_velocity_norms = [None]
    env._stage_progresses = [None]
    return env


def test_default_encode_is_byte_identical_to_legacy_schema() -> None:
    env = _bare_env()
    observation = _observation()
    factors = {"speed_class": "normal"}
    legacy = env._state_schema.encode(
        observation=observation, factors=factors, horizon_steps=20
    )
    encoded = env._encode(observation, SimpleNamespace(factors=factors))
    assert encoded.dtype == np.float32
    assert np.array_equal(encoded, legacy)
    assert env.state_schema == env._state_schema.to_dict()
    assert env.infra_is_default


def test_enabled_features_change_identity_and_compose_fixed_length() -> None:
    env = _bare_env()
    env.feature_registry = FeatureRegistry.from_config(
        {
            "relative_pose": True,
            "geodesic_error": True,
            "object_vel": True,
            "action_history": {"k": 2},
        }
    )
    env._ee_velocities[0] = np.zeros(6)
    env._stage_progresses[0] = 0.5
    env._time_to_goals[0] = 1.0
    env._action_histories[0] = [np.zeros(7)]
    schema = env.state_schema
    assert "feature_registry" in schema
    assert schema["state_dim"] == env._state_schema.state_dim + 7 + 1 + 6 + 14 + 4
    assert schema["mask_dim"] == env._state_schema.mask_dim + 4
    observation = _observation()
    encoded = env._encode(observation, SimpleNamespace(factors={"speed_class": "normal"}))
    assert encoded.shape == (schema["state_dim"],)
    assert not env.infra_is_default


def test_checkpoint_identity_records_infra_only_when_enabled() -> None:
    default_env = _bare_env()
    assert "infra_identity" not in default_env._checkpoint_identity()

    enabled_env = _bare_env()
    enabled_env.feature_registry = FeatureRegistry.from_config({"relative_pose": True})
    enabled_env.reward_registries = [
        RewardRegistry.from_config({"r_effort": {"weight": 0.005}})
    ]
    identity = enabled_env._checkpoint_identity()
    assert "infra_identity" in identity
    assert identity["infra_identity"]["features_sha256"]
    assert identity["infra_identity"]["reward_components_sha256"]

    # A different feature set must yield a different identity hash.
    other_env = _bare_env()
    other_env.feature_registry = FeatureRegistry.from_config({"goal_error": True})
    assert enabled_env._checkpoint_identity() != other_env._checkpoint_identity()


def test_checkpoint_identity_records_lift_target_when_shaping_is_enabled() -> None:
    env = _bare_env()
    env.cfg = {
        "reward_lift_shaping_weight": 2.0,
        "reward_orientation_shaping_weight": 0.0,
        "reward_lift_target_m": 0.12,
    }

    identity = env._checkpoint_identity()

    assert identity["reward_lift_shaping_weight"] == 2.0
    assert identity["reward_lift_target_m"] == 0.12


def test_enabled_feature_reencode_is_deterministic_after_restore() -> None:
    """Simulate load_checkpoint_state's re-encode after restoring derived state."""

    env = _bare_env()
    env.feature_registry = FeatureRegistry.from_config(
        {
            "relative_pose": True,
            "time_to_goal": True,
            "stage": True,
            "action_history": {"k": 2},
        }
    )
    observation = _observation()
    factors = {"speed_class": "normal"}
    request = SimpleNamespace(factors=factors)
    env._ee_velocities[0] = np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    env._time_to_goals[0] = 2.5
    env._distances[0] = 0.3
    env._relative_velocity_norms[0] = 0.1
    env._stage_progresses[0] = 0.7
    env._action_histories[0] = [np.zeros(7), np.ones(7)]
    original = env._encode(observation, request, env_index=0)

    # The checkpoint loader restores derived fields, then re-encodes from the
    # raw observation; the result must be byte-identical.
    env._ee_velocities = [None]
    env._time_to_goals = [None]
    env._distances = [None]
    env._relative_velocity_norms = [None]
    env._stage_progresses = [None]
    env._action_histories = [[]]
    env._ee_velocities[0] = np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    env._time_to_goals[0] = 2.5
    env._stage_progresses[0] = 0.7
    env._action_histories[0] = [np.zeros(7), np.ones(7)]
    restored = env._encode(observation, request, env_index=0)
    assert np.array_equal(original, restored)


def test_legacy_v0_2_checkpoint_is_rejected_for_non_default_infra() -> None:
    env = _bare_env()
    env.feature_registry = FeatureRegistry.from_config({"relative_pose": True})
    with pytest.raises(ValueError, match="legacy v0.2 checkpoint"):
        env.load_checkpoint_state({"schema_version": "rlinf-dynamic-benchmark-checkpoint-v0.2"})


def test_trainer_infra_helpers_and_yaml_parsing(tmp_path: Path) -> None:
    import examples.embodiment.train_dynamic_benchmark_expert as trainer

    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "task: t1_so3\n"
        "algorithm: residual_rlpd\n"
        "features:\n"
        "  relative_pose: true\n"
        "  action_history: {k: 3}\n"
        "reward_components:\n"
        "  r_ori_geodesic: {weight: 2.0}\n",
        encoding="utf-8",
    )
    args = trainer._parse_args(
        [
            "--config",
            str(config_path),
            "--task",
            "t1_so3",
            "--rlinf-commit",
            "a" * 40,
            "--benchmark-commit",
            "b" * 40,
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert args.features == {"relative_pose": True, "action_history": {"k": 3}}
    assert args.reward_components == {"r_ori_geodesic": {"weight": 2.0}}
    config = trainer._config(args)
    infra = trainer._infra_identity(config)
    assert infra is not None
    assert infra["features_sha256"] != infra["reward_components_sha256"]
    identity = trainer._demo_replay_identity(config, {"state_dim": 1})
    assert identity["infra_identity"] == infra

    default_args = trainer._parse_args(
        [
            "--task",
            "t1_so3",
            "--rlinf-commit",
            "a" * 40,
            "--benchmark-commit",
            "b" * 40,
            "--output",
            str(tmp_path / "out2"),
        ]
    )
    default = trainer._config(default_args)
    assert trainer._infra_identity(default) is None
    assert trainer._configs_equal(
        {key: value for key, value in asdict(default).items() if key not in {
            "features", "reward_components"
        }},
        asdict(default),
    )
    assert not trainer._configs_equal(
        dict(asdict(default), reward_components={"r_effort": {"weight": 0.005}}),
        asdict(default),
    )


def test_example_configs_parse_and_match_registry_schema() -> None:
    import yaml

    for name in ("rlopt_t1_so3", "rlopt_t2_se3", "rlopt_p0_grasp"):
        path = Path("examples/embodiment/config") / f"{name}.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        features = FeatureRegistry.from_config(payload["features"])
        rewards = RewardRegistry.from_config(payload["reward_components"])
        assert not features.is_empty
        assert not rewards.is_empty
        assert features.to_dict()["schema_version"].startswith(
            "rlinf-dynamic-benchmark-feature-registry"
        )
        assert rewards.to_dict()["schema_version"].startswith(
            "rlinf-dynamic-benchmark-reward-registry"
        )
        # Deterministic canonical identity is JSON-stable.
        first = json.dumps(
            features.to_dict(), sort_keys=True, separators=(",", ":")
        )
        second = json.dumps(
            features.to_dict(), sort_keys=True, separators=(",", ":")
        )
        assert first == second
