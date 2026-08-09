# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

import pytest

from examples.embodiment.train_dynamic_benchmark_expert import _config, _parse_args

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPO_ROOT / "configs" / "rld2_qa"
RLINF_COMMIT = "a" * 40
BENCHMARK_COMMIT = "b" * 40

FROZEN_CORE = {
    "algorithm": "residual_rlpd",
    "num_envs": 2,
    "eval_num_envs": 2,
    "demo_num_envs": 2,
    "total_env_steps": 20_000,
    "random_env_steps": 2_000,
    "demo_episodes": 16,
    "demo_max_attempts": 160,
    "demo_ratio": 0.5,
    "bc_steps": 2_000,
    "batch_size": 512,
    "replay_capacity": 250_000,
    "updates_per_vector_step": 1,
    "q_heads": 10,
    "q_target_subset": 2,
    "actor_bc_weight": 100.0,
    "initial_alpha": 0.01,
    "eval_interval": 2_000,
    "eval_episodes": 8,
    "checkpoint_interval": 2_000,
    "validation_manifest_seed": 20_261_450,
}


@pytest.mark.parametrize(
    (
        "recipe_name",
        "task",
        "residual_scale",
        "safety_penalty",
        "action_rate_weight",
        "features",
        "stage_weight",
    ),
    [
        (
            "quality_v2_final_source_common.yaml",
            "p0_grasp",
            0.25,
            -10.0,
            0.10,
            {"action_history": {"k": 3}},
            None,
        ),
        (
            "quality_v2_final_source_t2_se3.yaml",
            "t2_se3",
            0.10,
            -10.0,
            0.10,
            {"action_history": {"k": 3}},
            None,
        ),
        (
            "quality_v2_final_source_t3_phase.yaml",
            "t3_phase",
            0.125,
            -30.0,
            0.20,
            {"action_history": {"k": 3}, "stage": {}},
            None,
        ),
        (
            "quality_v2_final_source_t3_full.yaml",
            "t3_full",
            0.125,
            -30.0,
            0.20,
            {"action_history": {"k": 3}, "stage": {}},
            0.50,
        ),
    ],
)
def test_final_source_recipe_parses_through_training_entrypoint(
    tmp_path: Path,
    recipe_name: str,
    task: str,
    residual_scale: float,
    safety_penalty: float,
    action_rate_weight: float,
    features: dict[str, object],
    stage_weight: float | None,
) -> None:
    recipe = RECIPE_ROOT / recipe_name
    args = _parse_args(
        [
            "--config",
            str(recipe),
            "--task",
            task,
            "--rlinf-commit",
            RLINF_COMMIT,
            "--benchmark-commit",
            BENCHMARK_COMMIT,
            "--output",
            str(tmp_path / task),
            "--seed",
            "2",
            "--demo-seed",
            "2",
        ]
    )
    config = _config(args)

    for field_name, expected in FROZEN_CORE.items():
        assert getattr(config, field_name) == expected
    assert config.task == task
    assert config.seed == config.demo_seed == 2
    assert config.rlinf_commit == config.demo_rlinf_commit == RLINF_COMMIT
    assert config.benchmark_commit == BENCHMARK_COMMIT
    assert config.residual_scale == residual_scale
    assert config.reward_safety_penalty == safety_penalty
    assert config.features == features
    assert config.reward_components["r_action_rate"] == {
        "weight": action_rate_weight,
        "scale": 1.0,
    }
    if stage_weight is None:
        assert "r_stage" not in config.reward_components
    else:
        assert config.reward_components["r_stage"] == {"weight": stage_weight}


def test_final_source_recipe_rejects_unknown_training_key(tmp_path: Path) -> None:
    recipe = tmp_path / "unknown-final-source-key.yaml"
    recipe.write_text(
        "algorithm: residual_rlpd\nunknown_final_source_key: unsafe\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown Dynamic Benchmark config keys"):
        _parse_args(["--config", str(recipe)])
