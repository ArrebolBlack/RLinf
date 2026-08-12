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

import copy
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import examples.embodiment.train_dynamic_benchmark_expert as expert_trainer
from examples.embodiment.benchmark_dynamic_benchmark_throughput import _full_commit
from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    TransitionReplay,
    _BorrowedEvaluationRuntime,
    _checkpoint_selection_run_identity,
    _CheckpointSelectionLedger,
    _compose_residual_actions,
    _config,
    _config_artifact_identity,
    _demo_replay_identity,
    _env_cfg,
    _EvaluationRuntime,
    _load_demo_replay_cache,
    _load_demo_replay_cache_for_training,
    _make_exact_zero_residual_policy,
    _overlap_sample_and_update,
    _parse_args,
    _rng_state,
    _save_demo_replay_cache,
    _score,
    _successful_task_quality_diagnostics,
)


def test_config_artifact_identity_matches_promotion_canonical_payload(tmp_path) -> None:
    payload = {"seed": 2, "task": "p0_grasp", "weights": [1.0, 2.0]}
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    identity = _config_artifact_identity(config_path, payload)
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()

    assert identity["config_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert (
        identity["config_file_sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    assert identity["config_sha256"] != identity["config_file_sha256"]


def test_config_artifact_identity_accepts_json_array_for_runtime_tuple(
    tmp_path,
) -> None:
    payload = {
        "seed": 1,
        "task": "t2_se3",
        "state_derived_features": ("goal_planar_error", "eef_speed"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    identity = _config_artifact_identity(config_path, payload)

    assert (
        identity["config_sha256"]
        == hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    )


def test_resume_metrics_identity_accepts_json_tuple_roundtrip() -> None:
    checkpoint_metrics = {
        "success_rate": 1.0,
        "worker_pids": (101, 102),
    }
    manifest_metrics = json.loads(json.dumps(checkpoint_metrics))

    assert expert_trainer._canonical_json_sha256(
        checkpoint_metrics
    ) == expert_trainer._canonical_json_sha256(manifest_metrics)
    manifest_metrics["worker_pids"][1] = 103
    assert expert_trainer._canonical_json_sha256(
        checkpoint_metrics
    ) != expert_trainer._canonical_json_sha256(manifest_metrics)


def test_episode_logging_checkpoint_state_roundtrips_exactly_and_immutably() -> None:
    recent_episodes = [
        {
            "success": True,
            "trajectory_completion": 1.0,
            "return": 7.5,
            "duration_steps": 4,
        }
    ]
    episode_returns = torch.tensor([1.25, -2.5], dtype=torch.float32)
    episode_steps = torch.tensor([3, 6], dtype=torch.int64)

    state = expert_trainer._checkpoint_episode_logging_state(
        recent_episodes,
        episode_returns,
        episode_steps,
    )
    recent_episodes[0]["return"] = 999.0
    episode_returns.zero_()
    episode_steps.zero_()
    restored_recent, restored_returns, restored_steps = (
        expert_trainer._restore_episode_logging_state(state, num_envs=2)
    )

    assert restored_recent[0]["return"] == 7.5
    assert torch.equal(restored_returns, torch.tensor([1.25, -2.5]))
    assert torch.equal(restored_steps, torch.tensor([3, 6], dtype=torch.int64))
    restored_recent[0]["return"] = -999.0
    restored_returns.zero_()
    restored_steps.zero_()
    assert state["recent_episodes"][0]["return"] == 7.5
    assert torch.equal(state["episode_returns"], torch.tensor([1.25, -2.5]))
    assert torch.equal(
        state["episode_steps"], torch.tensor([3, 6], dtype=torch.int64)
    )


def test_episode_logging_checkpoint_state_fails_closed_on_invalid_tensors() -> None:
    state = expert_trainer._checkpoint_episode_logging_state(
        [],
        torch.tensor([1.0, 2.0]),
        torch.tensor([3, 4], dtype=torch.int64),
    )
    wrong_shape = copy.deepcopy(state)
    wrong_shape["episode_returns"] = torch.tensor([1.0])
    with pytest.raises(ValueError, match="shape does not match num_envs"):
        expert_trainer._restore_episode_logging_state(wrong_shape, num_envs=2)

    nonfinite = copy.deepcopy(state)
    nonfinite["episode_returns"][1] = float("nan")
    with pytest.raises(ValueError, match="returns must be finite"):
        expert_trainer._restore_episode_logging_state(nonfinite, num_envs=2)

    negative_steps = copy.deepcopy(state)
    negative_steps["episode_steps"][0] = -1
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        expert_trainer._restore_episode_logging_state(negative_steps, num_envs=2)


@pytest.mark.parametrize(
    "rendered_features",
    [
        ["eef_speed", "goal_planar_error"],
        ["goal_planar_error"],
        ["goal_planar_error", "eef_speed", "unexpected"],
    ],
)
def test_config_artifact_identity_rejects_semantic_sequence_tampering(
    tmp_path, rendered_features
) -> None:
    payload = {
        "seed": 1,
        "task": "t2_se3",
        "state_derived_features": ("goal_planar_error", "eef_speed"),
    }
    rendered = dict(payload)
    rendered["state_derived_features"] = rendered_features
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(rendered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="rendered training config does not match the resolved payload"
    ):
        _config_artifact_identity(config_path, payload)


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


def test_evaluation_vector_width_cannot_expand_declared_episode_count(
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
        "--eval-num-envs",
        "32",
        "--eval-episodes",
        "8",
    ]

    with pytest.raises(ValueError, match="cannot exceed eval_episodes"):
        _config(_parse_args(common))


def _task_quality_summary(value: float) -> dict[str, object]:
    return {
        "schema_version": "db0-episode-task-quality-v2",
        "episode_id": "fixture",
        "task_id": "t2_se3",
        "evaluator_backend_id": "fixture-backend",
        "schema_sha256": "a" * 64,
        "physics_sample_count": 25,
        "terminal": True,
        "components": {
            "terminal_goal_orientation_error_rad": {
                "value": value,
                "direction": "minimize",
                "unit": "rad",
                "scientific_resolution": 1e-6,
                "reducer": "terminal",
            }
        },
        "summary_sha256": "b" * 64,
    }


def test_successful_task_quality_is_diagnostic_and_success_conditioned() -> None:
    diagnostics = _successful_task_quality_diagnostics(
        [
            {"success": True, "task_quality": _task_quality_summary(0.2)},
            {"success": False, "task_quality": _task_quality_summary(99.0)},
            {"success": True, "task_quality": _task_quality_summary(0.4)},
        ]
    )

    assert diagnostics["status"] == "complete"
    assert diagnostics["successful_episode_count"] == 2
    assert diagnostics["summarized_episode_count"] == 2
    assert diagnostics["components"]["terminal_goal_orientation_error_rad"] == {
        "direction": "minimize",
        "unit": "rad",
        "scientific_resolution": 1e-6,
        "reducer": "terminal",
        "mean": pytest.approx(0.3),
    }


def test_task_quality_unavailable_does_not_change_selector() -> None:
    diagnostics = _successful_task_quality_diagnostics([{"success": True}])

    assert diagnostics == {
        "status": "unavailable",
        "successful_episode_count": 1,
        "summarized_episode_count": 0,
        "components": {},
    }
    core_metrics = _selection_metrics(success=1.0, safety=0.0)
    with_diagnostics = dict(core_metrics, successful_task_quality=diagnostics)
    assert _score(with_diagnostics) == _score(core_metrics)


def test_task_quality_diagnostics_fail_closed_on_partial_or_identity_drift() -> None:
    with pytest.raises(RuntimeError, match="missing for some successful"):
        _successful_task_quality_diagnostics(
            [
                {"success": True, "task_quality": _task_quality_summary(0.2)},
                {"success": True},
            ]
        )

    drifted = _task_quality_summary(0.4)
    drifted["schema_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="identity changed"):
        _successful_task_quality_diagnostics(
            [
                {"success": True, "task_quality": _task_quality_summary(0.2)},
                {"success": True, "task_quality": drifted},
            ]
        )


def _selection_metrics(
    *,
    success: float,
    safety: float,
    completion: float = 0.5,
    mean_return: float = 1.0,
    duration: float = 20.0,
    effort: float = 5.0,
) -> dict[str, float]:
    return {
        "episodes": 20,
        "success_rate": success,
        "safety_failure_rate": safety,
        "mean_completion": completion,
        "mean_return": mean_return,
        "mean_duration_steps": duration,
        "mean_action_l2_sum": effort,
    }


def _write_selection_policy(
    path,
    config,
    state_schema,
    metrics,
    env_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "rlinf-dynamic-benchmark-expert-policy-v0.1",
            "config": asdict(config),
            "model": {"fixture": torch.tensor([float(env_steps)])},
            "normalizer": {"fixture": True},
            "state_schema": state_schema,
            "validation": metrics,
            "env_steps": env_steps,
        },
        path,
    )


def _selection_ledger(tmp_path, planner_safety: float):
    output = tmp_path / "run"
    output.mkdir()
    config = _config(
        _parse_args(
            [
                "--task",
                "t3_full",
                "--algorithm",
                "residual_rlpd",
                "--rlinf-commit",
                "a" * 40,
                "--benchmark-commit",
                "b" * 40,
                "--output",
                str(output),
            ]
        )
    )
    state_schema = {"state_dim": 3, "mask_dim": 0, "fields": ["fixture"]}
    planner_metrics = _selection_metrics(
        success=0.5,
        safety=planner_safety,
        completion=0.8,
    )
    initial_policy = output / "initial_policy.pt"
    _write_selection_policy(
        initial_policy,
        config,
        state_schema,
        planner_metrics,
        0,
    )
    run_identity = _checkpoint_selection_run_identity(
        config,
        state_schema,
        "c" * 64,
    )
    ledger = _CheckpointSelectionLedger.create(
        output,
        run_identity,
        planner_metrics,
        initial_policy,
    )
    return ledger, config, state_schema, run_identity


def _record_selection_candidate(
    ledger,
    config,
    state_schema,
    metrics,
    env_steps: int,
):
    path = ledger.output / "policy_snapshots" / f"policy_step_{env_steps:012d}.pt"
    _write_selection_policy(path, config, state_schema, metrics, env_steps)
    return ledger.record_existing_snapshot(path, metrics, env_steps)


def test_checkpoint_selector_rejects_unsafe_high_success(tmp_path) -> None:
    ledger, config, state_schema, _ = _selection_ledger(tmp_path, 0.1)
    safe = _selection_metrics(success=0.2, safety=0.1, completion=0.4)
    unsafe = _selection_metrics(success=1.0, safety=0.15, completion=1.0)

    safe_row = _record_selection_candidate(ledger, config, state_schema, safe, 100)
    unsafe_row = _record_selection_candidate(ledger, config, state_schema, unsafe, 200)

    assert safe_row["eligible"] is True
    assert unsafe_row["eligible"] is False
    assert ledger.best_metrics == safe
    assert (
        ledger.manifest["selection"]["selected_snapshot_identity"]["env_steps"] == 100
    )
    assert (
        hashlib.sha256((ledger.output / "best_policy.pt").read_bytes()).hexdigest()
        == safe_row["policy"]["sha256"]
    )


def test_checkpoint_selector_zero_ceiling_allows_only_numerical_zero(tmp_path) -> None:
    ledger, config, state_schema, _ = _selection_ledger(tmp_path, 0.0)
    zero = _selection_metrics(success=0.1, safety=0.0)
    nonzero = _selection_metrics(success=1.0, safety=1e-6)

    zero_row = _record_selection_candidate(ledger, config, state_schema, zero, 100)
    nonzero_row = _record_selection_candidate(
        ledger, config, state_schema, nonzero, 200
    )

    assert zero_row["eligible"] is True
    assert nonzero_row["eligible"] is False
    assert ledger.best_metrics == zero


def test_checkpoint_selector_without_safe_candidate_uses_planner_fallback(
    tmp_path,
) -> None:
    ledger, config, state_schema, _ = _selection_ledger(tmp_path, 0.0)
    unsafe = _selection_metrics(success=1.0, safety=0.05)

    row = _record_selection_candidate(ledger, config, state_schema, unsafe, 100)

    assert row["eligible"] is False
    assert ledger.best_score is None
    assert ledger.best_metrics is None
    assert not (ledger.output / "best_policy.pt").exists()
    assert ledger.manifest["selection"]["status"] == "planner_fallback_no_eligible"
    assert ledger.manifest["selection"]["selected_snapshot_identity"] is None
    assert (
        ledger.manifest["selection"]["planner_fallback_policy"]["path"]
        == "initial_policy.pt"
    )


def test_checkpoint_selector_exact_tie_keeps_earlier_snapshot(tmp_path) -> None:
    ledger, config, state_schema, _ = _selection_ledger(tmp_path, 0.0)
    tied = _selection_metrics(success=0.7, safety=0.0, completion=0.9)

    first = _record_selection_candidate(ledger, config, state_schema, tied, 100)
    second = _record_selection_candidate(ledger, config, state_schema, tied, 200)

    assert (
        ledger.manifest["selection"]["selected_snapshot_identity"]["env_steps"] == 100
    )
    assert ledger.manifest["evaluated_snapshots"][0]["selected"] is True
    assert ledger.manifest["evaluated_snapshots"][1]["selected"] is False
    assert first["policy"]["sha256"] != second["policy"]["sha256"]
    assert (
        hashlib.sha256((ledger.output / "best_policy.pt").read_bytes()).hexdigest()
        == first["policy"]["sha256"]
    )


def test_checkpoint_selector_capture_is_immutable_and_unique_per_step(tmp_path) -> None:
    ledger, config, state_schema, _ = _selection_ledger(tmp_path, 0.0)
    metrics = _selection_metrics(success=0.5, safety=0.0)
    model = torch.nn.Linear(3, 7)
    normalizer = RunningNormalizer(dimension=3, mask_dim=0)
    normalizer.update(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))

    first = ledger.capture(
        config,
        model,
        normalizer,
        state_schema,
        metrics,
        100,
    )
    first_bytes = (ledger.output / first["policy"]["path"]).read_bytes()
    with torch.no_grad():
        model.weight.fill_(99.0)
    repeated = ledger.capture(
        config,
        model,
        normalizer,
        state_schema,
        metrics,
        100,
    )

    assert repeated == first
    assert (ledger.output / first["policy"]["path"]).read_bytes() == first_bytes
    changed_metrics = dict(metrics, success_rate=0.6)
    with pytest.raises(ValueError, match="changed its metrics identity"):
        ledger.capture(
            config,
            model,
            normalizer,
            state_schema,
            changed_metrics,
            100,
        )


@pytest.mark.parametrize("tamper_target", ["manifest", "snapshot"])
def test_checkpoint_selector_resume_rejects_tampered_evidence(
    tmp_path,
    tamper_target: str,
) -> None:
    ledger, config, state_schema, run_identity = _selection_ledger(tmp_path, 0.0)
    metrics = _selection_metrics(success=0.7, safety=0.0)
    row = _record_selection_candidate(ledger, config, state_schema, metrics, 100)
    checkpoint_state = ledger.checkpoint_state()
    if tamper_target == "manifest":
        payload = json.loads(ledger.manifest_path.read_text(encoding="utf-8"))
        payload["evaluated_snapshots"][0]["eligible"] = False
        ledger.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        with (ledger.output / row["policy"]["path"]).open("ab") as stream:
            stream.write(b"tamper")

    with pytest.raises(ValueError, match="checkpoint-selection|policy"):
        _CheckpointSelectionLedger.resume(
            ledger.output,
            run_identity,
            checkpoint_state,
        )


def test_checkpoint_selector_resume_preserves_existing_snapshot_identities(
    tmp_path,
) -> None:
    ledger, config, state_schema, run_identity = _selection_ledger(tmp_path, 0.0)
    first_metrics = _selection_metrics(success=0.2, safety=0.0)
    _record_selection_candidate(ledger, config, state_schema, first_metrics, 100)
    first_identity = json.loads(
        json.dumps(ledger.manifest["evaluated_snapshots"][0], sort_keys=True)
    )
    first_identity.pop("selected")

    resumed = _CheckpointSelectionLedger.resume(
        ledger.output,
        run_identity,
        ledger.checkpoint_state(),
    )
    second_metrics = _selection_metrics(success=0.3, safety=0.0)
    _record_selection_candidate(
        resumed,
        config,
        state_schema,
        second_metrics,
        200,
    )

    resumed_first = copy.deepcopy(resumed.manifest["evaluated_snapshots"][0])
    resumed_first.pop("selected")
    assert resumed_first == first_identity
    assert len(resumed.manifest["evaluated_snapshots"]) == 2
    bad_state = resumed.checkpoint_state()
    bad_state["manifest_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identities diverged"):
        _CheckpointSelectionLedger.resume(
            resumed.output,
            run_identity,
            bad_state,
        )


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
    config = _config(args)
    assert config.env_worker_processes == 2
    assert config.eval_worker_processes == 32
    assert config.persistent_eval_workers is True
    assert config.eval_planner_in_processes is False


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


def test_exact_cpu_process_recipe_is_enabled_by_default(tmp_path) -> None:
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
    ]

    config = _config(_parse_args(common))

    assert config.num_envs == 32
    assert config.eval_num_envs == 32
    assert config.eval_episodes == 32
    assert config.env_worker_processes == 32
    assert config.eval_worker_processes == 32
    assert config.process_start_method == expert_trainer._default_process_start_method()
    assert config.eval_planner_in_processes is True
    assert config.persistent_eval_workers is True
    assert config.borrow_training_env_for_eval is False
    assert config.sampler_learner_overlap is False

    serial = _config(
        _parse_args(
            [
                *common,
                "--env-worker-processes",
                "0",
                "--eval-worker-processes",
                "0",
                "--no-eval-planner-in-processes",
                "--no-persistent-eval-workers",
            ]
        )
    )
    assert serial.env_worker_processes == 0
    assert serial.eval_worker_processes == 0
    assert serial.eval_planner_in_processes is False
    assert serial.persistent_eval_workers is False


def test_residual_rlpd_is_the_default_algorithm(tmp_path) -> None:
    config = _config(
        _parse_args(
            [
                "--task",
                "t4_slider",
                "--rlinf-commit",
                "a" * 40,
                "--benchmark-commit",
                "b" * 40,
                "--output",
                str(tmp_path / "run"),
            ]
        )
    )
    assert config.algorithm == "residual_rlpd"


def test_process_worker_configuration_is_overridable_and_thread_exclusive(
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
    assert config.process_start_method == expert_trainer._default_process_start_method()
    assert env_cfg["worker_processes"] == 8
    assert (
        env_cfg["process_start_method"]
        == expert_trainer._default_process_start_method()
    )
    assert env_cfg["process_residual_planner"] is False
    assert config.sampler_learner_overlap is False
    assert config.eval_planner_in_processes is True
    assert config.persistent_eval_workers is True
    assert config.borrow_training_env_for_eval is False
    overlap_config = _config(_parse_args([*common, "--sampler-learner-overlap"]))
    assert overlap_config.sampler_learner_overlap is True
    with pytest.raises(ValueError, match="threads=1"):
        _config(_parse_args([*common, "--env-worker-threads", "2"]))
    with pytest.raises(ValueError, match="requires training process workers"):
        _config(
            _parse_args(
                [*common, "--env-worker-processes", "0", "--sampler-learner-overlap"]
            )
        )
    planner_config = _config(_parse_args([*common, "--eval-planner-in-processes"]))
    planner_env_cfg = _env_cfg(
        planner_config,
        split="validation",
        seed=17,
        num_envs=8,
        worker_threads=1,
        worker_processes=planner_config.eval_worker_processes,
        process_start_method=planner_config.process_start_method,
        process_residual_planner=planner_config.eval_planner_in_processes,
    )
    assert planner_config.eval_planner_in_processes is True
    assert planner_env_cfg["process_residual_planner"] is True
    with pytest.raises(ValueError, match="process evaluation planner requires"):
        _config(
            _parse_args(
                [
                    *common,
                    "--eval-worker-processes",
                    "0",
                    "--eval-planner-in-processes",
                ]
            )
        )
    persistent_config = _config(_parse_args([*common, "--persistent-eval-workers"]))
    assert persistent_config.persistent_eval_workers is True
    with pytest.raises(ValueError, match="persistent evaluation requires"):
        _config(
            _parse_args(
                [
                    *common,
                    "--eval-worker-processes",
                    "0",
                    "--persistent-eval-workers",
                ]
            )
        )
    borrowed_common = [
        "--task",
        "t4_sphere",
        "--algorithm",
        "residual_rlpd",
        "--rlinf-commit",
        "a" * 40,
        "--benchmark-commit",
        "b" * 40,
        "--output",
        str(tmp_path / "borrowed"),
        "--num-envs",
        "4",
        "--eval-num-envs",
        "4",
        "--env-worker-processes",
        "4",
        "--eval-worker-processes",
        "4",
        "--borrow-training-env-for-eval",
    ]
    borrowed_config = _config(_parse_args(borrowed_common))
    assert borrowed_config.borrow_training_env_for_eval is True
    with pytest.raises(ValueError, match="exclusive with persistent"):
        _config(_parse_args([*borrowed_common, "--persistent-eval-workers"]))
    with pytest.raises(ValueError, match="matching train/eval vector width"):
        _config(_parse_args([*borrowed_common, "--eval-num-envs", "8"]))
    with pytest.raises(ValueError, match="matching train/eval worker topology"):
        _config(_parse_args([*borrowed_common, "--eval-worker-processes", "2"]))
    with pytest.raises(ValueError, match="does not support process evaluation planner"):
        _config(_parse_args([*borrowed_common, "--eval-planner-in-processes"]))


def test_persistent_evaluation_runtime_restores_one_frozen_checkpoint(
    tmp_path, monkeypatch
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
        "--eval-worker-processes",
        "2",
        "--persistent-eval-workers",
    ]
    config = _config(_parse_args(common))

    class FakeEnv:
        num_envs = 2

        def __init__(self) -> None:
            self.rewind_payloads = []
            self.closed = False

        def checkpoint_state(self):
            return {"nested": {"values": [1, 2, 3]}}

        def close(self) -> None:
            self.closed = True

    env = FakeEnv()
    teachers = [object(), object()]
    resets = []
    monkeypatch.setattr(
        expert_trainer,
        "_build_evaluation_environment",
        lambda _config: (env, teachers),
    )
    monkeypatch.setattr(
        expert_trainer,
        "_rewind_evaluation_environment",
        lambda target_env, state: target_env.rewind_payloads.append(state),
    )
    monkeypatch.setattr(
        expert_trainer,
        "_reset_planner_teachers",
        lambda task, target_env, target_teachers, indices: resets.append(
            (task, target_env, target_teachers, indices)
        ),
    )

    runtime = _EvaluationRuntime()
    first = runtime.prepare(config)
    assert first[0] is env
    assert first[1] is teachers
    assert first[2] > 0.0
    assert first[3] == 0.0
    assert first[4] == 0
    assert first[5] == "initial_construction"
    runtime.mark_complete()

    second = runtime.prepare(config)
    assert second[0] is env
    assert second[2] == 0.0
    assert second[3] > 0.0
    assert second[4] == 1
    assert second[5] == "manifest_reset"
    assert env.rewind_payloads == [{"nested": {"values": [1, 2, 3]}}]
    assert runtime.initial_checkpoint == {"nested": {"values": [1, 2, 3]}}
    assert resets == [("t4_sphere", env, teachers, [0, 1])]

    runtime.close()
    runtime.close()
    assert env.closed is True
    assert runtime.closed is True


def test_borrowed_evaluation_runtime_restores_training_state_and_teachers(
    tmp_path, monkeypatch
) -> None:
    config = _config(
        _parse_args(
            [
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
                "--num-envs",
                "2",
                "--eval-num-envs",
                "2",
                "--env-worker-processes",
                "2",
                "--eval-worker-processes",
                "2",
                "--borrow-training-env-for-eval",
            ]
        )
    )

    class FakeEnv:
        num_envs = 2
        seed_offset = 0

        def __init__(self) -> None:
            self.split_name = "train"
            self.base_manifest_seed = config.train_manifest_seed
            self._last_obs = {"states": torch.tensor([[1.0], [2.0]])}
            self.loaded = 0
            self.loaded_without_refresh = 0
            self.manifest_context_builds = 0
            self.manifest_cache_loads = 0

        def checkpoint_state(self):
            return {
                "identity": {
                    "split": self.split_name,
                    "base_manifest_seed": self.base_manifest_seed,
                },
                "states": self._last_obs["states"].clone(),
            }

        def set_manifest_context(self, *, split_name, base_manifest_seed) -> None:
            self.split_name = split_name
            self.base_manifest_seed = base_manifest_seed
            self.manifest_context_builds += 1

        def manifest_cache_state(self):
            return {
                "split_name": self.split_name,
                "base_manifest_seed": self.base_manifest_seed,
            }

        def load_manifest_cache_state(self, state) -> None:
            self.split_name = state["split_name"]
            self.base_manifest_seed = state["base_manifest_seed"]
            self.manifest_cache_loads += 1

        def reset(self, *, options):
            assert options == {"env_idx": [0, 1]}
            self._last_obs = {"states": torch.tensor([[7.0], [8.0]])}
            return self._last_obs, {}

        def load_checkpoint_state(self, checkpoint, *, refresh_manifest=True) -> None:
            self.loaded += 1
            self.loaded_without_refresh += int(not refresh_manifest)
            self._last_obs = {"states": checkpoint["states"].clone()}

    env = FakeEnv()
    training_teachers = [SimpleNamespace(value=1), SimpleNamespace(value=2)]
    validation_teachers = [SimpleNamespace(value=7), SimpleNamespace(value=8)]
    monkeypatch.setattr(
        expert_trainer,
        "_make_planner_teachers",
        lambda task, target_env: validation_teachers,
    )
    runtime = _BorrowedEvaluationRuntime(
        env=env,
        training_teachers=training_teachers,
    )

    prepared = runtime.prepare(config)
    assert prepared[0] is env
    assert prepared[1] is validation_teachers
    assert prepared[4] == 0
    assert prepared[5] == "borrow_training_pool"
    assert runtime.validation_manifest_cache_hit is False
    assert env.split_name == "validation"
    assert torch.equal(env._last_obs["states"], torch.tensor([[7.0], [8.0]]))
    training_teachers[0].value = 99

    restore_s = runtime.finish(completed=True)
    assert restore_s >= 0.0
    assert runtime.evaluations == 1
    assert env.loaded == 1
    assert env.loaded_without_refresh == 1
    assert env.manifest_context_builds == 1
    assert env.manifest_cache_loads == 1
    assert env.split_name == "train"
    assert env.base_manifest_seed == config.train_manifest_seed
    assert torch.equal(env._last_obs["states"], torch.tensor([[1.0], [2.0]]))
    assert [teacher.value for teacher in training_teachers] == [1, 2]
    assert runtime.training_checkpoint is None

    repeated = runtime.prepare(config)
    assert repeated[5] == "borrow_training_pool"
    assert runtime.validation_manifest_cache_hit is True
    assert env.manifest_context_builds == 1
    assert env.manifest_cache_loads == 2
    runtime.finish(completed=True)
    assert env.loaded_without_refresh == 2
    assert env.manifest_cache_loads == 3


def test_borrowed_evaluation_prepare_failure_restores_training_state(
    tmp_path, monkeypatch
) -> None:
    config = _config(
        _parse_args(
            [
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
                "--borrow-training-env-for-eval",
            ]
        )
    )

    class FailingEnv:
        num_envs = 2
        seed_offset = 0

        def __init__(self) -> None:
            self.split_name = "train"
            self.base_manifest_seed = config.train_manifest_seed
            self._last_obs = {"states": torch.tensor([[1.0], [2.0]])}

        def checkpoint_state(self):
            return {
                "identity": {
                    "split": self.split_name,
                    "base_manifest_seed": self.base_manifest_seed,
                },
                "states": self._last_obs["states"].clone(),
            }

        def set_manifest_context(self, *, split_name, base_manifest_seed) -> None:
            self.split_name = split_name
            self.base_manifest_seed = base_manifest_seed

        def manifest_cache_state(self):
            return {
                "split_name": self.split_name,
                "base_manifest_seed": self.base_manifest_seed,
            }

        def load_manifest_cache_state(self, state) -> None:
            self.split_name = state["split_name"]
            self.base_manifest_seed = state["base_manifest_seed"]

        def reset(self, *, options):
            raise RuntimeError("validation reset failed")

        def load_checkpoint_state(self, checkpoint, *, refresh_manifest=True) -> None:
            assert refresh_manifest is False
            self._last_obs = {"states": checkpoint["states"].clone()}

    env = FailingEnv()
    monkeypatch.setattr(
        expert_trainer,
        "_make_planner_teachers",
        lambda task, target_env: pytest.fail("planner creation must not run"),
    )
    runtime = _BorrowedEvaluationRuntime(env=env)
    with pytest.raises(RuntimeError, match="validation reset failed"):
        runtime.prepare(config)
    assert env.split_name == "train"
    assert env.base_manifest_seed == config.train_manifest_seed
    assert torch.equal(env._last_obs["states"], torch.tensor([[1.0], [2.0]]))
    assert runtime.training_checkpoint is None


@pytest.mark.parametrize(
    "checkpoint_schema",
    (
        "rlinf-dynamic-benchmark-checkpoint-v0.2",
        "rlinf-dynamic-benchmark-checkpoint-v0.3",
    ),
)
def test_evaluation_rewind_loads_the_exact_frozen_checkpoint(
    checkpoint_schema: str,
) -> None:
    identity = {"task_id": "t4_sphere", "num_envs": 2}
    request_rows = [
        {
            "episode_id": f"episode-{index}",
            "task_id": "t4_sphere",
            "split": "validation",
            "seed": 100 + index,
            "action_mode": "joint_delta",
            "observation_track": "state",
            "object_mode": "default",
            "reset_mode": "canonical",
            "factors": {"index": index},
            "api_version": "v1",
        }
        for index in range(2)
    ]

    def request(row):
        return SimpleNamespace(
            episode_id=row["episode_id"],
            task_id=row["task_id"],
            split=SimpleNamespace(value=row["split"]),
            seed=row["seed"],
            action_mode=SimpleNamespace(value=row["action_mode"]),
            observation_track=SimpleNamespace(value=row["observation_track"]),
            object_mode=row["object_mode"],
            reset_mode=row["reset_mode"],
            factors=row["factors"],
            api_version=row["api_version"],
        )

    class FakeEnv:
        num_envs = 2

        def __init__(self) -> None:
            self._manifest_generation = 99
            self._manifest_cursor = 99
            self._requests = []
            self.loaded_checkpoint = None

        def _checkpoint_identity(self):
            return identity

        def load_checkpoint_state(self, state):
            self.loaded_checkpoint = state
            self._manifest_generation = int(state["manifest_generation"])
            self._requests = [request(row) for row in request_rows]
            self._manifest_cursor = int(state["manifest_cursor"])

    checkpoint = {
        "schema_version": checkpoint_schema,
        "identity": identity,
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "manifest_generation": 3,
        "manifest_cursor": 2,
        "requests": request_rows,
        "needs_reset": np.zeros(2, dtype=bool),
        "elapsed_steps": torch.zeros(2, dtype=torch.int32),
        "prev_step_reward": torch.zeros(2),
        "returns": torch.zeros(2),
        "success_once": torch.zeros(2, dtype=torch.bool),
        "is_start": True,
    }
    env = FakeEnv()
    expert_trainer._rewind_evaluation_environment(env, checkpoint)
    assert env._manifest_generation == 3
    assert env._manifest_cursor == 2
    assert env.loaded_checkpoint is checkpoint


def test_sampler_learner_overlap_runs_updates_while_sampling_is_in_flight() -> None:
    sample_started = threading.Event()
    release_sample = threading.Event()

    def sample() -> str:
        sample_started.set()
        assert release_sample.wait(timeout=2.0)
        time.sleep(0.01)
        return "sample"

    def update() -> str:
        assert sample_started.wait(timeout=2.0)
        time.sleep(0.01)
        release_sample.set()
        time.sleep(0.01)
        return "update"

    with ThreadPoolExecutor(max_workers=1) as executor:
        sample_result, update_result, timing = _overlap_sample_and_update(
            executor, sample, update
        )

    assert sample_result == "sample"
    assert update_result == "update"
    assert timing["environment_step_s"] > 0.0
    assert timing["sampler_learner_overlap_s"] > 0.0
    assert timing["overlapped_update_wall_s"] > 0.0
    assert timing["sampler_wait_after_update_s"] >= 0.0


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


def test_exact_zero_residual_policy_matches_planner_for_every_state() -> None:
    model = SimpleNamespace(actor_mean=torch.nn.Linear(3, 7))
    with torch.no_grad():
        model.actor_mean.weight.fill_(2.0)
        model.actor_mean.bias.fill_(3.0)

    _make_exact_zero_residual_policy(model)

    assert torch.equal(model.actor_mean(torch.randn(5, 3)), torch.zeros(5, 7))


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
    assert np.array_equal(restored_rng_state["numpy"][1], learner_rng_state["numpy"][1])
    assert torch.equal(restored_rng_state["torch_cpu"], learner_rng_state["torch_cpu"])
    assert torch.equal(
        multiseed_replay.generator.get_state(), learner_replay_generator_state
    )
    assert multiseed_replay.size == replay.size
