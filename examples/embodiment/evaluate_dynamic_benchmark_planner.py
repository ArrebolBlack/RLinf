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
from typing import Any

import numpy as np
import torch

from examples.embodiment.evaluate_dynamic_benchmark_expert import (
    _atomic_json,
    _full_commit,
    _latency_summary,
    _payload_sha256,
    _sha256,
)

EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-planner-evaluation-v0.1"


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

    def __init__(self, vector_env: Any, raw_env: Any) -> None:
        self._vector_env = vector_env
        self._raw_env = raw_env

    def reset(self, request: Any) -> Any:
        observation = self._raw_env.reset(request)
        self._vector_env._arm_hidden_t5_event(self._raw_env, request)
        return observation

    def step(self, action: Any) -> Any:
        return self._raw_env.step(action)

    def save_state(self) -> bytes:
        return self._raw_env.save_state()


def _replay_actions_on_fresh_env(
    *,
    vector_env: Any,
    task_id: str,
    request: Any,
    expected_observations: tuple[Any, ...],
    actions: tuple[Any, ...],
    expected_outcomes: tuple[Any, ...],
    expected_final_state: bytes,
    replay_fn: Any | None = None,
) -> dict[str, Any]:
    """Replay one action tape on an independent canonical raw environment."""
    if replay_fn is None:
        from se3_wam.benchmark.evaluation import replay_actions

        replay_fn = replay_actions
    raw_env = vector_env._make_mujoco_env(
        task_id,
        image_size=vector_env.image_size,
        camera_observations=vector_env.camera_observations,
    )
    try:
        return replay_fn(
            _ArmedResetReplayEnv(vector_env, raw_env),
            request=request,
            expected_observations=expected_observations,
            actions=actions,
            expected_outcomes=expected_outcomes,
            expected_final_state=expected_final_state,
        )
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
        raise ValueError("planner evaluation request task does not match the environment")
    raw_env = vector_env._make_mujoco_env(
        vector_env.task_id,
        image_size=vector_env.image_size,
        camera_observations=vector_env.camera_observations,
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
    vector_env._last_obs = {
        "states": torch.as_tensor(state[None], dtype=torch.float32)
    }
    vector_env._is_start = True
    return observation


def _episode(*, env: Any, task_id: str) -> tuple[dict[str, Any], list[float]]:
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
    observations = [observation]
    actions = []
    outcomes = []
    rewards = []
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
        _, reward, terminated, truncated, infos = env.step(
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
        actions.append(action)
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
        result_info = {
            "success": bool(infos["success"][0]),
            "termination_reason": reason,
            "active_stage_progress": active_progress,
        }
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
    replay_validation = _replay_actions_on_fresh_env(
        vector_env=env,
        task_id=task_id,
        request=request,
        expected_observations=tuple(observations),
        actions=tuple(actions),
        expected_outcomes=tuple(outcomes),
        expected_final_state=final_state,
    )
    if not replay_validation["passed"]:
        raise RuntimeError(
            "planner rollout replay failed: "
            f"{request.episode_id}: {json.dumps(replay_validation, sort_keys=True)}"
        )
    action_array = np.stack([action.values for action in actions])
    safety_failures = set(env.reward_schema["safety_failures"])
    return (
        {
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
            "termination_reason": result_info["termination_reason"],
            "trajectory_completion": completion,
            "completion_time_s": completion_time,
            "return": float(sum(rewards)),
            "control_steps": len(actions),
            "action_l2_sum": float(np.square(action_array).sum()),
            "action_sha256": hashlib.sha256(
                np.ascontiguousarray(action_array).tobytes()
            ).hexdigest(),
            "actions": action_array.tolist(),
            "teacher_preparation": preparation,
            "replay_validation": replay_validation,
            "events": [event.name for event in events],
        },
        latencies_s,
    )


def _task_summary(task_id: str, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    from se3_wam.benchmark.evaluation import summarize_task_records

    summary = summarize_task_records([task_id], records)
    summary[task_id].update(
        safety_failure_rate=float(np.mean([row["safety_failure"] for row in records])),
        mean_return=float(np.mean([row["return"] for row in records])),
        mean_action_l2_sum=float(np.mean([row["action_l2_sum"] for row in records])),
    )
    return summary


def main() -> None:
    from se3_wam.benchmark.contracts import canonical_json
    from se3_wam.benchmark.evaluation import manifest_record

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
            record, episode_latencies = _episode(env=env, task_id=args.task)
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
            "schema_version": EVALUATION_SCHEMA,
            "planner_identity": {"task": args.task, "kind": "privileged_teacher"},
            "source_identity": {
                "evaluator_rlinf_commit": evaluator_commit,
                "benchmark_commit": benchmark_commit,
            },
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
            "started_unix_s": started,
            "finished_unix_s": time.time(),
        }
        result["payload_sha256"] = _payload_sha256(result)
        result_path = args.output / "evaluation.json"
        _atomic_json(result_path, result)
        (args.output / "SHA256SUMS").write_text(
            (
                f"{_sha256(result_path)}  evaluation.json\n"
                f"{_sha256(reset_manifest_path)}  reset_manifest.jsonl\n"
            ),
            encoding="utf-8",
        )
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
