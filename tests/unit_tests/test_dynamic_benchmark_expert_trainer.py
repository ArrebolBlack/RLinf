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

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.embodiment.benchmark_dynamic_benchmark_throughput import _full_commit
from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    TransitionReplay,
    _compose_residual_actions,
    _config,
    _demo_replay_identity,
    _env_cfg,
    _load_demo_replay_cache,
    _load_demo_replay_cache_for_training,
    _parse_args,
    _rng_state,
    _save_demo_replay_cache,
    _score,
)


def test_running_normalizer_standardizes_values_but_preserves_mask() -> None:
    normalizer = RunningNormalizer(dimension=4, mask_dim=2)
    normalizer.update(
        torch.tensor(
            [
                [1.0, 10.0, 0.0, 1.0],
                [3.0, 14.0, 1.0, 0.0],
            ]
        )
    )

    normalized = normalizer.normalize(
        torch.tensor([[2.0, 12.0, 1.0, 0.0]]),
        torch.device("cpu"),
    )

    assert torch.allclose(normalized[:, :2], torch.zeros(1, 2))
    assert torch.equal(normalized[:, 2:], torch.tensor([[1.0, 0.0]]))


def test_transition_replay_round_trip_preserves_data_cursor_and_sampling_rng() -> None:
    replay = TransitionReplay(capacity=4, state_dim=2, seed=17)
    states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    replay.add(
        states,
        torch.arange(42, dtype=torch.float32).reshape(6, 7),
        torch.arange(6, dtype=torch.float32),
        states + 1.0,
        torch.tensor([False, False, True, False, False, True]),
        torch.zeros(6, dtype=torch.bool),
    )
    checkpoint = replay.state_dict()
    restored = TransitionReplay(capacity=4, state_dim=2, seed=999)
    restored.load_state_dict(checkpoint)

    assert restored.size == replay.size == 4
    assert restored.cursor == replay.cursor == 2
    assert torch.equal(
        replay.states,
        torch.tensor([[8.0, 9.0], [10.0, 11.0], [4.0, 5.0], [6.0, 7.0]]),
    )
    original_sample = replay.sample(16)
    restored_sample = restored.sample(16)
    for name in TransitionReplay.FIELDS:
        assert torch.equal(original_sample[name], restored_sample[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-backed replay gate")
@pytest.mark.parametrize("storage", ["pinned_cpu", "gpu"])
def test_transition_replay_storage_paths_preserve_cpu_sampling(storage: str) -> None:
    states = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    payload = (
        states,
        torch.arange(42, dtype=torch.float32).reshape(6, 7),
        torch.arange(6, dtype=torch.float32),
        states + 1.0,
        torch.tensor([False, False, True, False, False, True]),
        torch.zeros(6, dtype=torch.bool),
    )
    baseline = TransitionReplay(capacity=16, state_dim=4, seed=17)
    candidate = TransitionReplay(
        capacity=16,
        state_dim=4,
        seed=17,
        storage=storage,
        device=torch.device("cuda:0"),
    )
    baseline.add(*payload)
    candidate.add(*payload)

    baseline_sample = baseline.sample(32)
    candidate_sample = candidate.sample(32)

    for name in TransitionReplay.FIELDS:
        assert torch.equal(baseline_sample[name], candidate_sample[name].cpu())
    if storage == "pinned_cpu":
        assert all(value.is_pinned() for value in candidate_sample.values())


def test_policy_score_is_success_then_safety_lexicographic() -> None:
    baseline = {
        "success_rate": 0.5,
        "safety_failure_rate": 0.0,
        "mean_completion": 0.9,
        "mean_return": 10.0,
        "mean_duration_steps": 20.0,
        "mean_action_l2_sum": 5.0,
    }
    safer_but_less_complete = dict(
        baseline,
        safety_failure_rate=0.0,
        mean_completion=0.1,
    )
    unsafe = dict(baseline, safety_failure_rate=0.1, mean_completion=1.0)
    more_success = dict(baseline, success_rate=0.6, safety_failure_rate=1.0)

    assert _score(safer_but_less_complete) > _score(unsafe)
    assert _score(more_success) > _score(baseline)


def test_throughput_probe_requires_full_frozen_source_commits() -> None:
    commit = "a" * 40

    assert _full_commit("source", commit) == commit
    with pytest.raises(ValueError, match="full lowercase"):
        _full_commit("source", "abc123")


def test_recipe_yaml_sets_defaults_but_keeps_run_identity_explicit(tmp_path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "task: t2_trans\nalgorithm: rlpd\nnum_envs: 2\n", encoding="utf-8"
    )
    args = _parse_args(
        [
            "--config",
            str(recipe),
            "--rlinf-commit",
            "a" * 40,
            "--benchmark-commit",
            "b" * 40,
            "--output",
            str(tmp_path / "run"),
        ]
    )

    assert args.task == "t2_trans"
    assert args.algorithm == "rlpd"
    assert args.num_envs == 2


@pytest.mark.parametrize(
    "key", ["rlinf_commit", "demo_rlinf_commit", "demo_seed", "demo_replay_in"]
)
def test_recipe_yaml_rejects_run_specific_provenance(tmp_path, key: str) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(f"task: t2_trans\n{key}: unsafe\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run-specific"):
        _parse_args(["--config", str(recipe)])


def test_actor_bc_regularization_is_explicit_and_non_negative(tmp_path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "task: t2_trans\nalgorithm: rlpd\nactor_bc_weight: 2.5\n",
        encoding="utf-8",
    )
    common = [
        "--config",
        str(recipe),
        "--rlinf-commit",
        "a" * 40,
        "--benchmark-commit",
        "b" * 40,
        "--output",
        str(tmp_path / "run"),
    ]

    assert _config(_parse_args(common)).actor_bc_weight == 2.5
    with pytest.raises(ValueError, match="non-negative"):
        _config(_parse_args([*common, "--actor-bc-weight", "-1"]))


def test_process_worker_configuration_is_explicit_and_thread_exclusive(
    tmp_path,
) -> None:
    common = [
        "--task",
        "t4_sphere",
        "--algorithm",
        "residual_rlpd",
        "--rlinf-commit",
        "a" * 40,
        "--benchmark-commit",
        "b" * 40,
        "--output",
        str(tmp_path / "run"),
        "--env-worker-processes",
        "4",
        "--eval-worker-processes",
        "8",
    ]

    config = _config(_parse_args(common))
    env_cfg = _env_cfg(
        config,
        split="validation",
        seed=17,
        num_envs=8,
        worker_threads=1,
        worker_processes=config.eval_worker_processes,
        process_start_method=config.process_start_method,
    )

    assert config.env_worker_processes == 4
    assert config.eval_worker_processes == 8
    assert config.process_start_method == "spawn"
    assert env_cfg["worker_processes"] == 8
    assert env_cfg["process_start_method"] == "spawn"
    with pytest.raises(ValueError, match="threads=1"):
        _config(_parse_args([*common, "--env-worker-threads", "2"]))


def test_safety_penalty_is_explicit_and_reaches_environment_and_cache_identity(
    tmp_path,
) -> None:
    common = [
        "--task",
        "t4_sphere",
        "--algorithm",
        "residual_rlpd",
        "--rlinf-commit",
        "a" * 40,
        "--benchmark-commit",
        "b" * 40,
        "--output",
        str(tmp_path / "run"),
        "--reward-safety-penalty",
        "-30",
    ]
    config = _config(_parse_args(common))

    assert config.reward_safety_penalty == -30.0
    assert (
        _env_cfg(
            config,
            split="train",
            seed=17,
            num_envs=2,
            worker_threads=1,
        )["reward_safety_penalty"]
        == -30.0
    )
    assert (
        _demo_replay_identity(
            config,
            {"state_dim": 2, "mask_dim": 0, "fields": ["fixture"]},
        )["reward_safety_penalty"]
        == -30.0
    )
    with pytest.raises(ValueError, match="non-positive"):
        _config(_parse_args([*common, "--reward-safety-penalty", "1"]))
    with pytest.raises(ValueError, match="finite"):
        _config(_parse_args([*common, "--reward-safety-penalty", "nan"]))


def test_residual_action_composition_is_scaled_and_clamped() -> None:
    planner = torch.tensor([[0.9, -0.9, 0.0, 0.1, -0.1, 0.2, -0.2]])
    residual = torch.tensor([[1.0, -1.0, 0.4, -0.4, 0.0, 0.8, -0.8]])

    composed = _compose_residual_actions(planner, residual, 0.25)

    assert torch.allclose(
        composed,
        torch.tensor([[1.0, -1.0, 0.1, 0.0, -0.1, 0.4, -0.4]]),
    )
    with pytest.raises(ValueError, match="residual_scale"):
        _compose_residual_actions(planner, residual, 0.0)


def test_demo_replay_cache_round_trip_and_identity_gate(tmp_path) -> None:
    args = _parse_args(
        [
            "--task",
            "t2_trans",
            "--algorithm",
            "rlpd",
            "--rlinf-commit",
            "a" * 40,
            "--benchmark-commit",
            "b" * 40,
            "--output",
            str(tmp_path / "run"),
            "--seed",
            "3",
            "--num-envs",
            "2",
            "--demo-episodes",
            "2",
            "--demo-max-attempts",
            "4",
            "--batch-size",
            "2",
            "--replay-capacity",
            "8",
        ]
    )
    config = _config(args)
    assert config.demo_seed == config.seed == 3
    state_schema = {"state_dim": 2, "mask_dim": 0, "fields": ["fixture"]}
    identity = _demo_replay_identity(config, state_schema)
    assert _demo_replay_identity(replace(config, num_envs=32), state_schema) == identity
    assert _demo_replay_identity(replace(config, seed=4), state_schema) == identity
    assert _demo_replay_identity(replace(config, demo_seed=4), state_schema) != identity
    assert (
        _demo_replay_identity(replace(config, demo_rlinf_commit="c" * 40), state_schema)
        != identity
    )
    replay = TransitionReplay(capacity=8, state_dim=2, seed=14)
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    replay.add(
        states,
        torch.arange(14, dtype=torch.float32).reshape(2, 7),
        torch.tensor([1.0, 2.0]),
        states + 0.5,
        torch.tensor([False, True]),
        torch.zeros(2, dtype=torch.bool),
    )
    normalizer = RunningNormalizer(dimension=2, mask_dim=0)
    normalizer.update(states)
    summary = {
        "accepted_episodes": 2,
        "attempted_episodes": 2,
        "successful_attempts": 2,
        "success_rate": 1.0,
        "transitions": 2,
    }
    cache_path = tmp_path / "demo_replay.pt"
    cache_sha256 = _save_demo_replay_cache(
        cache_path,
        identity,
        summary,
        replay,
        normalizer,
    )

    restored_replay = TransitionReplay(capacity=8, state_dim=2, seed=999)
    restored_normalizer = RunningNormalizer(dimension=2, mask_dim=0)
    restored_summary, restored_sha256 = _load_demo_replay_cache(
        cache_path,
        identity,
        restored_replay,
        restored_normalizer,
    )

    assert restored_sha256 == cache_sha256
    assert restored_summary == summary
    assert restored_replay.size == replay.size
    for name in TransitionReplay.FIELDS:
        assert torch.equal(
            restored_replay.state_dict()["data"][name],
            replay.state_dict()["data"][name],
        )
    assert restored_normalizer.state_dict()["count"] == 2

    legacy_identity = dict(identity)
    legacy_identity.pop("reward_safety_penalty")
    legacy_path = tmp_path / "legacy_demo_replay.pt"
    _save_demo_replay_cache(
        legacy_path,
        legacy_identity,
        summary,
        replay,
        normalizer,
    )
    _load_demo_replay_cache(
        legacy_path,
        identity,
        restored_replay,
        restored_normalizer,
    )
    with pytest.raises(ValueError, match="identity"):
        _load_demo_replay_cache(
            legacy_path,
            dict(identity, reward_safety_penalty=-20.0),
            restored_replay,
            restored_normalizer,
        )

    mismatched_identity = dict(identity, seed=4)
    with pytest.raises(ValueError, match="identity"):
        _load_demo_replay_cache(
            cache_path,
            mismatched_identity,
            restored_replay,
            restored_normalizer,
        )

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    multiseed_replay = TransitionReplay(capacity=8, state_dim=2, seed=52)
    learner_rng_state = _rng_state()
    learner_replay_generator_state = multiseed_replay.generator.get_state().clone()
    multiseed_normalizer = RunningNormalizer(dimension=2, mask_dim=0)
    _load_demo_replay_cache_for_training(
        cache_path,
        identity,
        multiseed_replay,
        multiseed_normalizer,
        training_seed=41,
        demo_seed=config.demo_seed,
    )
    restored_rng_state = _rng_state()
    assert restored_rng_state["python"] == learner_rng_state["python"]
    assert np.array_equal(
        restored_rng_state["numpy"][1], learner_rng_state["numpy"][1]
    )
    assert torch.equal(
        restored_rng_state["torch_cpu"], learner_rng_state["torch_cpu"]
    )
    assert torch.equal(
        multiseed_replay.generator.get_state(), learner_replay_generator_state
    )
    assert multiseed_replay.size == replay.size
