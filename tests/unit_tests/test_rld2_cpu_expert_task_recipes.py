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

from examples.embodiment.resolve_dynamic_benchmark_expert_recipe import (
    DEFAULT_MANIFEST,
    EXACT_TASKS,
    load_recipe_map,
    resolve_recipe,
)
from examples.embodiment.train_dynamic_benchmark_expert import _config, _parse_args


RLINF_COMMIT = "a" * 40
BENCHMARK_COMMIT = "b" * 40
SO3_FEATURES = (
    "eef_to_object_pose_wxyz",
    "eef_to_object_distance_m",
    "fingerpad_midpoint_world",
    "grasp_point_offset_world_m",
    "closing_axis_object_alignment_rad",
    "object_vertical_position_m",
    "eef_vertical_position_m",
    "object_yaw_rad",
    "object_xy_velocity_m_s",
    "object_angular_velocity_z_rad_s",
)


def parse_recipe(task: str, tmp_path: Path):
    resolved = resolve_recipe(task)
    args = _parse_args(
        [
            "--config",
            str(resolved.recipe),
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
    return resolved, _config(args)


def test_task_recipe_map_is_exact_and_specializations_cannot_fall_back() -> None:
    payload = load_recipe_map(DEFAULT_MANIFEST)
    assert tuple(payload["tasks"]) == EXACT_TASKS
    assert set(payload["specialized_tasks"]) == {
        "t1_so3",
        "t2_se3",
        "t3_phase",
        "t3_full",
        "t4_sphere",
    }
    ordinary = payload["ordinary_recipe"]
    assert all(
        payload["tasks"][task]["recipe"] != ordinary
        for task in payload["specialized_tasks"]
    )


@pytest.mark.parametrize("task", EXACT_TASKS)
def test_every_task_recipe_parses_through_trainer(task: str, tmp_path: Path) -> None:
    resolved, config = parse_recipe(task, tmp_path)
    assert resolved.recipe.is_file()
    assert config.task == task
    assert config.seed == config.demo_seed == 2
    assert config.num_envs == config.eval_num_envs == config.demo_num_envs == 2
    assert config.env_worker_processes == config.eval_worker_processes == 2
    assert config.persistent_eval_workers is True
    assert config.eval_planner_in_processes is True


def test_t1_so3_retains_complete_historical_a4_observation_core(
    tmp_path: Path,
) -> None:
    resolved, config = parse_recipe("t1_so3", tmp_path)
    assert resolved.role == "historical_winner_core_transfer"
    assert resolved.current_source_matched_confirmation_required is True
    assert config.state_derived_features == SO3_FEATURES
    assert config.total_env_steps == 20_000
    assert config.residual_scale == 0.25
    assert config.features == {"action_history": {"k": 3}}


def test_t2_se3_retains_historical_d1_residual_envelope(tmp_path: Path) -> None:
    resolved, config = parse_recipe("t2_se3", tmp_path)
    assert resolved.role == "historical_winner_retained"
    assert config.residual_scale == 0.10
    assert config.total_env_steps == 20_000


def test_t4_sphere_retains_historical_long_budget_core(tmp_path: Path) -> None:
    resolved, config = parse_recipe("t4_sphere", tmp_path)
    assert resolved.role == "historical_winner_core_transfer"
    assert config.total_env_steps == 100_000
    assert config.random_env_steps == 5_000
    assert config.demo_episodes == 64
    assert config.bc_steps == 5_000
    assert config.actor_bc_weight == 50.0
    assert config.residual_scale == 1.0
    assert config.eval_episodes == 40


def test_ordinary_tasks_share_only_the_ordinary_recipe() -> None:
    payload = load_recipe_map(DEFAULT_MANIFEST)
    ordinary_recipe = payload["ordinary_recipe"]
    ordinary_tasks = {
        task for task, entry in payload["tasks"].items() if entry["role"] == "ordinary"
    }
    assert ordinary_tasks == set(EXACT_TASKS) - set(payload["specialized_tasks"])
    assert all(payload["tasks"][task]["recipe"] == ordinary_recipe for task in ordinary_tasks)

