#!/usr/bin/env python3
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

"""Evaluate the frozen privileged planner on paired Dynamic Benchmark resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from examples.embodiment.dynamic_benchmark_evaluation_attempt import (
    load_quality_v4_rollout_reference,
    materialize_evaluation_attempt,
    materialize_quality_v4_fresh_replay_attempt,
    recursive_output_checksums,
    validate_formal_quality_v2_thresholds,
)
from examples.embodiment.evaluate_dynamic_benchmark_expert import (
    _atomic_json,
    _expected_sha256,
    _full_commit,
    _latency_summary,
    _payload_sha256,
    _sha256,
    _task_quality_from_terminal_infos,
    _task_quality_identity,
    _validate_task_quality_summary,
)

EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-planner-evaluation-v0.1"
FORMAL_EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-planner-evaluation-v0.2"
QUALITY_V4_EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-planner-evaluation-v0.3"
TASK_QUALITY_BACKEND_ID = "mujoco311-rs140-v1-rld2-quality"


def _trajectory_quality_v2_from_rollout(
    observations: list[Any],
    action_array: np.ndarray,
    *,
    task_id: str,
    task_config: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Compute replay-bound quality-v2 measurements for planner calibration."""

    from se3_wam.benchmark.trajectory_quality import (
        trajectory_quality_v2_from_observations,
    )

    return trajectory_quality_v2_from_observations(
        observations,
        action_array,
        task_id=task_id,
        task_config=task_config,
        sample_period_s=0.05,
        continuous_dimensions=max(1, int(action_array.shape[1]) - 1),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test_id", "test_ood"),
        required=True,
    )
    parser.add_argument("--manifest-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--quality-v2-thresholds", type=Path)
    parser.add_argument("--expected-quality-v2-thresholds-sha256")
    parser.add_argument("--quality-v4-thresholds", type=Path)
    parser.add_argument("--expected-quality-v4-thresholds-sha256")
    parser.add_argument("--quality-v4-reference-root", type=Path)
    return parser


def _planner_action_values(values: Any) -> tuple[torch.Tensor, np.ndarray]:
    env_actions = torch.as_tensor(
        np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)[None],
        dtype=torch.float32,
    )
    recorded_values = np.asarray(env_actions[0], dtype=np.float64)
    return env_actions, recorded_values


class _ArmedResetReplayEnv:
    """Expose raw replay while restoring Dynamic Benchmark hidden reset state."""

    def __init__(
        self,
        vector_env: Any,
        raw_env: Any,
        *,
        capture_quality_v4: bool = False,
    ) -> None:
        self._vector_env = vector_env
        self._raw_env = raw_env
        self._terminal_task_quality: Any | None = None
        self._capture_quality_v4 = capture_quality_v4
        self._observations: list[Any] = []
        self._outcomes: list[tuple[Any, ...]] = []
        self._rewards: list[float] = []
        self._physics_samples: list[Any] = []

    def reset(self, request: Any) -> Any:
        observation = self._raw_env.reset(request)
        self._vector_env._arm_hidden_t5_event(self._raw_env, request)
        if self._capture_quality_v4:
            self._observations = [observation]
            self._physics_samples = [
                self._raw_env.quality_v4_current_physics_source_sample()
            ]
        return observation

    def step(self, action: Any) -> Any:
        result = self._raw_env.step(action)
        if bool(getattr(result, "terminated", False)) or bool(
            getattr(result, "truncated", False)
        ):
            self._terminal_task_quality = getattr(result, "task_quality", None)
        if self._capture_quality_v4:
            self._observations.append(result.observation)
            self._outcomes.append(
                (
                    bool(result.terminated),
                    bool(result.truncated),
                    bool(result.success),
                    result.termination_reason,
                    float(result.active_stage_progress),
                )
            )
            self._rewards.append(float(result.reward))
            self._physics_samples.extend(self._raw_env.quality_v4_last_physics_trace())
        return result

    def save_state(self) -> bytes:
        return self._raw_env.save_state()

    @property
    def terminal_task_quality(self) -> Any | None:
        """Return the terminal task-quality summary from independent replay."""

        return self._terminal_task_quality

    def quality_v4_capture(self) -> dict[str, Any]:
        """Return replay-owned raw sources before the fresh environment closes."""

        if not self._capture_quality_v4 or not self._observations:
            raise RuntimeError("Qv4 replay capture was not enabled")
        # The benchmark property already returns detached read-only records.
        # Qv4 converts this snapshot to its hashed plain action-history schema.
        history = getattr(self._raw_env, "canonical_action_history", None)
        return {
            "raw_env": SimpleNamespace(canonical_action_history=history),
            "observations": tuple(self._observations),
            "outcomes": tuple(self._outcomes),
            "rewards": tuple(self._rewards),
            "physics_samples": tuple(self._physics_samples),
            "events": tuple(self._raw_env._ledger.events),
        }


def _replay_actions_on_fresh_env(
    *,
    vector_env: Any,
    task_id: str,
    request: Any,
    expected_observations: tuple[Any, ...],
    actions: tuple[Any, ...],
    expected_outcomes: tuple[Any, ...],
    expected_final_state: bytes,
    expected_task_quality: Mapping[str, Any] | None = None,
    task_quality_identity: Mapping[str, Any] | None = None,
    replay_fn: Any | None = None,
    capture_quality_v4: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Replay one action tape on an independent canonical raw environment."""
    if replay_fn is None:
        from se3_wam.benchmark.evaluation import replay_actions

        replay_fn = replay_actions
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import (
        _task_quality_make_kwargs,
    )

    task_quality_kwargs = _task_quality_make_kwargs(
        getattr(vector_env, "task_quality_schema_version", None),
        getattr(vector_env, "task_quality_evaluator_backend_id", None),
    )
    raw_env = vector_env._make_mujoco_env(
        task_id,
        image_size=vector_env.image_size,
        camera_observations=vector_env.camera_observations,
        **task_quality_kwargs,
    )
    try:
        replay_env = _ArmedResetReplayEnv(
            vector_env,
            raw_env,
            capture_quality_v4=capture_quality_v4,
        )
        validation = replay_fn(
            replay_env,
            request=request,
            expected_observations=expected_observations,
            actions=actions,
            expected_outcomes=expected_outcomes,
            expected_final_state=expected_final_state,
        )
        if not isinstance(validation, Mapping):
            raise ValueError("planner replay validation must be a mapping")
        result = dict(validation)
        if task_quality_identity is not None:
            episode_id = getattr(request, "episode_id", None)
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("planner replay request episode identity is missing")
            if expected_task_quality is None:
                result["task_quality_exact"] = replay_env.terminal_task_quality is None
                result["task_quality_summary_sha256"] = None
            else:
                recorded = _validate_task_quality_summary(
                    expected_task_quality,
                    identity=task_quality_identity,
                    task_id=task_id,
                    episode_id=episode_id,
                )
                replayed = _validate_task_quality_summary(
                    replay_env.terminal_task_quality,
                    identity=task_quality_identity,
                    task_id=task_id,
                    episode_id=episode_id,
                )
                result["task_quality_exact"] = replayed == recorded
                result["task_quality_summary_sha256"] = replayed["summary_sha256"]
            result["passed"] = bool(
                result.get("passed") is True and result["task_quality_exact"]
            )
        elif expected_task_quality is not None:
            raise ValueError("expected task quality requires its canonical identity")
        if capture_quality_v4:
            return result, replay_env.quality_v4_capture()
        return result
    finally:
        raw_env.close()


def _reset_rollout_on_fresh_env(*, vector_env: Any, request: Any) -> Any:
    """Start one planner rollout from a newly constructed raw environment.

    The vector adapter normally advances a manifest by repeatedly resetting the
    same raw simulator. Exact-replay validation instead constructs a new raw
    simulator. Keeping those lifecycle semantics asymmetric can expose hidden
    carry-over after later resets, so the planner evaluator replaces its sole raw
    member for every manifest row and resets that member exactly once.
    """
    if int(vector_env.num_envs) != 1:
        raise ValueError("planner evaluation requires exactly one vector member")
    if request.task_id != vector_env.task_id:
        raise ValueError(
            "planner evaluation request task does not match the environment"
        )
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import (
        _task_quality_make_kwargs,
    )

    task_quality_kwargs = _task_quality_make_kwargs(
        getattr(vector_env, "task_quality_schema_version", None),
        getattr(vector_env, "task_quality_evaluator_backend_id", None),
    )
    raw_env = vector_env._make_mujoco_env(
        vector_env.task_id,
        image_size=vector_env.image_size,
        camera_observations=vector_env.camera_observations,
        **task_quality_kwargs,
    )
    try:
        if int(raw_env.horizon_steps) != int(vector_env.horizon_steps):
            raise RuntimeError("fresh planner raw environment changed the horizon")
        observation = raw_env.reset(request)
        vector_env._arm_hidden_t5_event(raw_env, request)
        state = vector_env._encode(observation, request)
    except BaseException:
        raw_env.close()
        raise

    previous_env = vector_env.envs[0]
    try:
        previous_env.close()
    except BaseException:
        raw_env.close()
        raise
    vector_env.envs[0] = raw_env
    vector_env._reset_metrics(np.asarray([0], dtype=np.int64))
    vector_env._raw_observations[0] = observation
    vector_env._requests[0] = request
    vector_env._needs_reset[0] = False
    vector_env._last_obs = {"states": torch.as_tensor(state[None], dtype=torch.float32)}
    vector_env._is_start = True
    return observation


def _episode(
    *,
    env: Any,
    task_id: str,
    task_quality_identity: Mapping[str, Any],
    quality_v2_thresholds: Mapping[str, object] | None = None,
    quality_v2_thresholds_sha256: str | None = None,
    attempt_output: Path | None = None,
    attempt_index: int | None = None,
    quality_v4_thresholds: Mapping[str, Any] | None = None,
    quality_v4_reference: Mapping[str, Any] | None = None,
    quality_v4_output: Path | None = None,
) -> tuple[dict[str, Any], list[float]]:
    from se3_wam.benchmark.metrics import (
        completion_time_from_events,
        hierarchical_task_completion,
        validate_stage_event_order,
    )
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    request = env._requests[0]
    observation = env._raw_observations[0]
    if request is None or observation is None:
        raise RuntimeError("planner evaluation environment is not initialized")
    matches = [
        row
        for row in env._manifest_rows
        if row.request.episode_id == request.episode_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"manifest does not uniquely contain episode {request.episode_id!r}"
        )
    row = matches[0]
    teacher, preparation = make_privileged_teacher(task_id, request=request)
    if hasattr(teacher, "reset"):
        teacher.reset()
    raw_env = env.envs[0]
    quality_v4_arguments = (
        quality_v4_thresholds,
        quality_v4_reference,
        quality_v4_output,
    )
    if any(value is not None for value in quality_v4_arguments) and not all(
        value is not None for value in quality_v4_arguments
    ):
        raise ValueError("Qv4 evaluator arguments must be supplied together")
    quality_v4_enabled = all(value is not None for value in quality_v4_arguments)
    physics_samples = (
        [raw_env.quality_v4_current_physics_source_sample()]
        if quality_v4_enabled
        else []
    )
    observations = [observation]
    if env._last_obs is None:
        raise RuntimeError("planner evaluation environment lost its encoded state")
    states = [np.asarray(env._last_obs["states"][0], dtype=np.float32)]
    actions = []
    policy_action_values = []
    outcomes = []
    rewards = []
    terminated_rows = []
    truncated_rows = []
    latencies_s = []
    result_info: dict[str, Any] | None = None
    terminated_value = False
    truncated_value = False
    while not (terminated_value or truncated_value):
        started = time.perf_counter()
        teacher_action = teacher.act(observation)
        latencies_s.append(time.perf_counter() - started)
        env_actions, values = _planner_action_values(teacher_action.values)
        action = env._ActionCommand(
            mode=request.action_mode,
            values=values,
            policy_step=observation.policy_step,
        )
        next_obs, reward, terminated, truncated, infos = env.step(
            env_actions, auto_reset=False
        )
        next_observation = env._raw_observations[0]
        if next_observation is None:
            raise RuntimeError(
                "planner evaluation environment lost its raw observation"
            )
        terminated_value = bool(terminated[0])
        truncated_value = bool(truncated[0])
        reason = infos["termination_reason"][0]
        active_progress = float(infos["reward_inputs"]["active_stage_progress"][0])
        observations.append(next_observation)
        states.append(np.asarray(next_obs["states"][0], dtype=np.float32))
        actions.append(action)
        policy_action_values.append(np.asarray(env_actions[0], dtype=np.float32))
        outcomes.append(
            (
                terminated_value,
                truncated_value,
                bool(infos["success"][0]),
                reason,
                active_progress,
            )
        )
        rewards.append(float(reward[0]))
        if quality_v4_enabled:
            physics_samples.extend(raw_env.quality_v4_last_physics_trace())
        terminated_rows.append(terminated_value)
        truncated_rows.append(truncated_value)
        result_info = {
            "success": bool(infos["success"][0]),
            "termination_reason": reason,
            "active_stage_progress": active_progress,
        }
        if terminated_value or truncated_value:
            result_info["task_quality"] = _task_quality_from_terminal_infos(
                infos,
                identity=task_quality_identity,
                task_id=task_id,
                episode_id=request.episode_id,
                success=result_info["success"],
            )
        observation = next_observation
        if len(actions) > int(env.horizon_steps):
            raise RuntimeError("planner rollout exceeded the environment horizon")
    if result_info is None:
        raise RuntimeError("planner produced no action")

    events = tuple(raw_env._ledger.events)
    final_state = raw_env.save_state()
    task = env._get_task_spec(task_id)
    completed = validate_stage_event_order(task, events)
    completion = hierarchical_task_completion(
        task,
        completed,
        result_info["active_stage_progress"],
    )
    completion_time = (
        completion_time_from_events(
            events,
            start_event=task.task_start_event,
            success_event=task.success_stages[-1],
        )
        if result_info["success"]
        else None
    )
    replay_result = _replay_actions_on_fresh_env(
        vector_env=env,
        task_id=task_id,
        request=request,
        expected_observations=tuple(observations),
        actions=tuple(actions),
        expected_outcomes=tuple(outcomes),
        expected_final_state=final_state,
        expected_task_quality=result_info["task_quality"],
        task_quality_identity=task_quality_identity,
        capture_quality_v4=quality_v4_enabled,
    )
    if quality_v4_enabled:
        if not isinstance(replay_result, tuple) or len(replay_result) != 2:
            raise RuntimeError("Qv4 fresh replay did not return its source capture")
        replay_validation, replay_capture = replay_result
    else:
        if not isinstance(replay_result, Mapping):
            raise RuntimeError("planner replay validation has an invalid type")
        replay_validation = dict(replay_result)
        replay_capture = None
    if not replay_validation["passed"] and not quality_v4_enabled:
        raise RuntimeError(
            "planner rollout replay failed: "
            f"{request.episode_id}: {json.dumps(replay_validation, sort_keys=True)}"
        )
    action_array = np.stack([action.values for action in actions])
    quality_v2 = _trajectory_quality_v2_from_rollout(
        observations,
        action_array,
        task_id=task_id,
        task_config=getattr(raw_env, "task_config", None),
    )
    quality_v2_gate = None
    if quality_v2_thresholds is not None:
        if quality_v2_thresholds_sha256 is None:
            raise ValueError("quality-v2 gate is missing its threshold SHA-256")
        from se3_wam.benchmark.trajectory_quality import evaluate_quality_v2_gate

        quality_v2_gate = evaluate_quality_v2_gate(
            quality_v2,
            quality_v2_thresholds,
            task_id=task_id,
        )
        quality_v2_gate["contract_sha256"] = quality_v2_thresholds_sha256
    safety_failures = set(env.reward_schema["safety_failures"])
    record = {
        "episode_id": request.episode_id,
        "task_id": task_id,
        "seed": request.seed,
        "factors": dict(request.factors),
        "source_group_id": row.source_group_id,
        "pair_id": row.pair_id,
        "pair_member_id": row.pair_member_id,
        "candidate_index": row.candidate_index,
        "success": result_info["success"],
        "safety_failure": result_info["termination_reason"] in safety_failures,
        "reward_schema_safety_failures": sorted(safety_failures),
        "termination_reason": result_info["termination_reason"],
        "trajectory_completion": completion,
        "completion_time_s": completion_time,
        "return": float(sum(rewards)),
        "control_steps": len(actions),
        "action_l2_sum": float(np.square(action_array).sum()),
        "task_quality": result_info["task_quality"],
        "quality_v2": quality_v2,
        "quality_v2_sha256": _payload_sha256(quality_v2),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(action_array).tobytes()
        ).hexdigest(),
        "actions": action_array.tolist(),
        "teacher_preparation": preparation,
        "replay_validation": replay_validation,
        "events": [event.name for event in events],
    }
    if quality_v2_gate is not None:
        record["quality_v2_gate"] = quality_v2_gate
    producer_arguments = (
        quality_v2_thresholds is not None,
        quality_v2_thresholds_sha256 is not None,
        attempt_output is not None,
        attempt_index is not None,
    )
    if any(producer_arguments) and not all(producer_arguments):
        raise ValueError("formal attempt producer arguments must be supplied together")
    if all(producer_arguments):
        assert attempt_output is not None
        assert attempt_index is not None
        assert quality_v2_thresholds_sha256 is not None
        record = materialize_evaluation_attempt(
            attempt_output,
            record,
            candidate_index=attempt_index,
            raw_env=raw_env,
            observations=observations,
            states=states,
            policy_actions=policy_action_values,
            rewards=rewards,
            terminated=terminated_rows,
            truncated=truncated_rows,
            quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
        )
    if quality_v4_enabled:
        assert quality_v4_output is not None
        assert quality_v4_thresholds is not None
        assert quality_v4_reference is not None
        assert replay_capture is not None
        record["quality_v4_attempt"] = materialize_quality_v4_fresh_replay_attempt(
            output=quality_v4_output,
            record=record,
            raw_env=raw_env,
            observations=observations,
            actions=action_array,
            rewards=rewards,
            outcomes=outcomes,
            physics_samples=physics_samples,
            events=events,
            thresholds=quality_v4_thresholds,
            reference_contract=quality_v4_reference,
            base_replay_validation=replay_validation,
            replay_capture=replay_capture,
        )
    return record, latencies_s


def _task_summary(task_id: str, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    from se3_wam.benchmark.evaluation import summarize_task_records

    summary = summarize_task_records([task_id], records)
    summary[task_id].update(
        safety_failure_rate=float(np.mean([row["safety_failure"] for row in records])),
        mean_return=float(np.mean([row["return"] for row in records])),
        mean_action_l2_sum=float(np.mean([row["action_l2_sum"] for row in records])),
    )
    return summary


def _evaluation_schema(*, formal_attempts: bool, quality_v4: bool = False) -> str:
    """Select the legacy, Qv3-formal, or parallel Qv4 container schema."""

    if quality_v4:
        return QUALITY_V4_EVALUATION_SCHEMA
    return FORMAL_EVALUATION_SCHEMA if formal_attempts else EVALUATION_SCHEMA


def _evaluation_checksums(
    output: Path,
    *,
    result_path: Path,
    reset_manifest_path: Path,
    formal_attempts: bool,
) -> str:
    """Seal formal tapes recursively or enforce the two-file calibration contract."""

    if formal_attempts:
        return recursive_output_checksums(output)
    owned = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if owned != {"evaluation.json", "reset_manifest.jsonl"}:
        raise ValueError(
            "metric-calibration planner output must contain exactly its two data files"
        )
    return (
        f"{_sha256(result_path)}  evaluation.json\n"
        f"{_sha256(reset_manifest_path)}  reset_manifest.jsonl\n"
    )


def main() -> None:
    from se3_wam.benchmark.contracts import canonical_json
    from se3_wam.benchmark.evaluation import manifest_record
    from se3_wam.benchmark.task_quality import task_quality_schema_manifest

    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.image_size < 64:
        raise ValueError("--image-size must be at least 64")
    evaluator_commit = _full_commit("evaluator_commit", args.evaluator_commit)
    benchmark_commit = _full_commit("benchmark_commit", args.benchmark_commit)
    if (args.quality_v2_thresholds is None) != (
        args.expected_quality_v2_thresholds_sha256 is None
    ):
        raise ValueError(
            "quality-v2 thresholds and expected SHA-256 must be supplied together"
        )
    quality_v2_thresholds: Mapping[str, object] | None = None
    quality_v2_thresholds_sha256: str | None = None
    if args.quality_v2_thresholds is not None:
        if not args.quality_v2_thresholds.is_file():
            raise FileNotFoundError(args.quality_v2_thresholds)
        quality_v2_thresholds_sha256 = _sha256(args.quality_v2_thresholds)
        expected_thresholds_sha256 = _expected_sha256(
            str(args.expected_quality_v2_thresholds_sha256)
        )
        if quality_v2_thresholds_sha256 != expected_thresholds_sha256:
            raise ValueError(
                "quality-v2 threshold SHA-256 does not match the expected identity"
            )
        loaded_thresholds = json.loads(
            args.quality_v2_thresholds.read_text(encoding="utf-8")
        )
        if not isinstance(loaded_thresholds, Mapping):
            raise ValueError("quality-v2 threshold contract must be a mapping")
        quality_v2_thresholds = dict(loaded_thresholds)
        validate_formal_quality_v2_thresholds(
            quality_v2_thresholds,
            task_id=args.task,
            thresholds_sha256=quality_v2_thresholds_sha256,
        )
    quality_v4_arguments = (
        args.quality_v4_thresholds,
        args.expected_quality_v4_thresholds_sha256,
        args.quality_v4_reference_root,
    )
    if any(value is not None for value in quality_v4_arguments) and not all(
        value is not None for value in quality_v4_arguments
    ):
        raise ValueError(
            "Qv4 thresholds, expected file SHA-256, and reference root must be supplied together"
        )
    quality_v4_thresholds: Mapping[str, Any] | None = None
    quality_v4_threshold_validation: dict[str, Any] | None = None
    if args.quality_v4_thresholds is not None:
        from examples.embodiment.dynamic_benchmark_quality_v4 import (
            load_quality_v4_thresholds,
        )

        if args.quality_v4_reference_root.is_symlink() or not (
            args.quality_v4_reference_root.is_dir()
        ):
            raise FileNotFoundError(args.quality_v4_reference_root)
        quality_v4_thresholds, quality_v4_threshold_validation = (
            load_quality_v4_thresholds(
                args.quality_v4_thresholds,
                expected_file_sha256=_expected_sha256(
                    str(args.expected_quality_v4_thresholds_sha256)
                ),
                require_formal_freeze=False,
            )
        )
    task_quality_identity = _task_quality_identity(args.task)
    manifest_size = max(args.episodes, 2)
    if manifest_size % 2:
        manifest_size += 1
    env = DynamicBenchmarkEnv(
        cfg={
            "task_id": args.task,
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "manifest_size": manifest_size,
            "image_size": args.image_size,
            "camera_observations": False,
            "auto_reset": False,
            "ignore_terminations": False,
            "group_size": 1,
            "task_quality_schema_version": task_quality_schema_manifest(args.task)[
                "schema_version"
            ],
            "task_quality_evaluator_backend_id": TASK_QUALITY_BACKEND_ID,
        },
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        rows = list(env._manifest_rows[: args.episodes])
        if len(rows) != args.episodes:
            raise RuntimeError("planner evaluation manifest is shorter than requested")
        args.output.mkdir(parents=True)
        reset_manifest_path = args.output / "reset_manifest.jsonl"
        reset_manifest_path.write_text(
            "".join(canonical_json(manifest_record(row)) + "\n" for row in rows),
            encoding="utf-8",
        )
        started = time.time()
        records = []
        latencies_s: list[float] = []
        for episode_index, row in enumerate(rows):
            _reset_rollout_on_fresh_env(vector_env=env, request=row.request)
            quality_v4_reference = (
                None
                if quality_v4_thresholds is None
                else load_quality_v4_rollout_reference(
                    args.quality_v4_reference_root,
                    task_id=args.task,
                    episode_id=row.request.episode_id,
                    expected_state_schema_sha256=_payload_sha256(env.state_schema),
                )
            )
            record, episode_latencies = _episode(
                env=env,
                task_id=args.task,
                task_quality_identity=task_quality_identity,
                quality_v2_thresholds=quality_v2_thresholds,
                quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
                attempt_output=(
                    args.output if quality_v2_thresholds is not None else None
                ),
                attempt_index=(
                    episode_index if quality_v2_thresholds is not None else None
                ),
                quality_v4_thresholds=quality_v4_thresholds,
                quality_v4_reference=quality_v4_reference,
                quality_v4_output=(
                    args.output if quality_v4_thresholds is not None else None
                ),
            )
            if record["episode_id"] != row.request.episode_id:
                raise RuntimeError(
                    "planner rollout order diverged from the frozen reset manifest"
                )
            records.append(record)
            latencies_s.extend(episode_latencies)
            print(
                json.dumps(
                    {
                        "episode_id": record["episode_id"],
                        "success": record["success"],
                        "safety_failure": record["safety_failure"],
                        "trajectory_completion": record["trajectory_completion"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        latency = _latency_summary(latencies_s)
        result = {
            "schema_version": _evaluation_schema(
                formal_attempts=quality_v2_thresholds is not None,
                quality_v4=quality_v4_thresholds is not None,
            ),
            "planner_identity": {"task": args.task, "kind": "privileged_teacher"},
            "source_identity": {
                "evaluator_rlinf_commit": evaluator_commit,
                "benchmark_commit": benchmark_commit,
            },
            **(
                {
                    "task_quality_identity": task_quality_identity,
                    "quality_v2_threshold_identity": {
                        "schema_version": quality_v2_thresholds.get("schema_version"),
                        "sha256": quality_v2_thresholds_sha256,
                    },
                    "all_successful_quality_v2_gates_passed": all(
                        bool(row["quality_v2_gate"]["passed"])
                        for row in records
                        if row["success"]
                    ),
                }
                if quality_v2_thresholds is not None
                else {}
            ),
            "quality_v4_threshold_identity": (
                None
                if quality_v4_threshold_validation is None
                else dict(quality_v4_threshold_validation)
            ),
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "reset_manifest_sha256": _sha256(reset_manifest_path),
            "episodes": args.episodes,
            "records": records,
            "task_summary": _task_summary(args.task, records),
            "decision_latency": latency,
            "all_replays_passed": all(
                row["replay_validation"]["passed"] for row in records
            ),
            "all_quality_v4_layer1_gates_passed": (
                None
                if quality_v4_thresholds is None
                else all(row["quality_v4_attempt"]["layer1_passed"] for row in records)
            ),
            "all_quality_v4_layer2_gates_passed": (
                None
                if quality_v4_thresholds is None
                else all(row["quality_v4_attempt"]["layer2_passed"] for row in records)
            ),
            "started_unix_s": started,
            "finished_unix_s": time.time(),
        }
        if (
            args.quality_v2_thresholds is not None
            and _sha256(args.quality_v2_thresholds) != quality_v2_thresholds_sha256
        ):
            raise RuntimeError("quality-v2 threshold file changed during evaluation")
        if (
            args.quality_v4_thresholds is not None
            and quality_v4_threshold_validation is not None
            and _sha256(args.quality_v4_thresholds)
            != quality_v4_threshold_validation["file_sha256"]
        ):
            raise RuntimeError("Qv4 threshold file changed during evaluation")
        result["payload_sha256"] = _payload_sha256(result)
        result_path = args.output / "evaluation.json"
        _atomic_json(result_path, result)
        checksums = _evaluation_checksums(
            args.output,
            result_path=result_path,
            reset_manifest_path=reset_manifest_path,
            formal_attempts=(
                quality_v2_thresholds is not None or quality_v4_thresholds is not None
            ),
        )
        (args.output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
        print(
            json.dumps(
                {
                    "evaluation_sha256": _sha256(result_path),
                    "payload_sha256": result["payload_sha256"],
                    "all_replays_passed": result["all_replays_passed"],
                    "decision_latency": latency,
                    "task_summary": result["task_summary"],
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
