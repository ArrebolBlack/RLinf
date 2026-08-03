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

"""Evaluate a frozen Dynamic Benchmark expert on deterministic test resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    _compose_residual_actions,
    _planner_actions,
    _policy_action,
)

POLICY_SCHEMA = "rlinf-dynamic-benchmark-expert-policy-v0.1"
EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-expert-evaluation-v0.1"
SUPPORTED_ALGORITHMS = frozenset({"bc", "sac", "rlpd", "residual_rlpd", "ppo"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test_id", "test_ood"),
        required=True,
    )
    parser.add_argument("--manifest-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    from se3_wam.benchmark.contracts import canonical_json

    return hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def _full_commit(name: str, value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


def _expected_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("expected policy SHA-256 must be 64 lowercase hexadecimal characters")
    return normalized


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _model_kwargs(config: Mapping[str, Any], state_dim: int) -> dict[str, Any]:
    algorithm = str(config.get("algorithm", ""))
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported policy algorithm {algorithm!r}")
    if state_dim < 1:
        raise ValueError("policy state dimension must be positive")
    if algorithm == "ppo":
        return {
            "obs_dim": state_dim,
            "action_dim": 7,
            "num_action_chunks": 1,
            "add_value_head": True,
            "add_q_head": False,
            "num_q_heads": 2,
        }
    q_heads = int(config.get("q_heads", 2))
    if q_heads < 2:
        raise ValueError("SAC-family policy must declare at least two Q heads")
    return {
        "obs_dim": state_dim,
        "action_dim": 7,
        "num_action_chunks": 1,
        "add_value_head": False,
        "add_q_head": True,
        "q_head_type": "default",
        "critic_obs_dim": state_dim,
        "num_q_heads": q_heads,
    }


class _InferenceMLPPolicy(nn.Module):
    """Actor-only reconstruction of the training MLP for frozen inference.

    Importing ``rlinf.models`` also imports the distributed scheduler and Ray.
    Evaluation and trajectory export only call ``_sample_actions``, so keep this
    path independent of the training stack while validating the omitted head
    weights in :func:`_load_inference_policy`.
    """

    def __init__(self, state_dim: int, algorithm: str) -> None:
        super().__init__()
        self.independent_std = algorithm == "ppo"
        self.final_tanh = algorithm != "ppo"
        self.logstd_range = (-5.0, 2.0)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, 7)
        if self.independent_std:
            self.actor_logstd = nn.Parameter(torch.empty(1, 7))
        else:
            self.actor_logstd = nn.Linear(256, 7)

    def _sample_actions(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        mean = self.actor_mean(features)
        if self.independent_std:
            log_std = self.actor_logstd.expand_as(mean)
        else:
            log_std = self.actor_logstd(features)
            log_std = torch.tanh(log_std)
            low, high = self.logstd_range
            log_std = low + 0.5 * (high - low) * (log_std + 1.0)
        return mean, log_std


def _load_inference_policy(
    config: Mapping[str, Any],
    state_dim: int,
    state_dict: Mapping[str, Any],
    device: torch.device,
) -> _InferenceMLPPolicy:
    """Strictly validate a training checkpoint and load its inference actor."""

    kwargs = _model_kwargs(config, state_dim)
    algorithm = str(config["algorithm"])
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("policy model state must be a non-empty mapping")
    if any(not isinstance(key, str) for key in state_dict):
        raise ValueError("policy model state keys must be strings")
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("policy model state values must be tensors")

    model = _InferenceMLPPolicy(state_dim, algorithm)
    actor_keys = set(model.state_dict())
    available = set(state_dict)
    missing = actor_keys - available
    if missing:
        raise ValueError(f"policy actor state is missing keys: {sorted(missing)}")
    auxiliary = available - actor_keys

    if algorithm == "ppo":
        if not auxiliary or any(not key.startswith("value_head.") for key in auxiliary):
            raise ValueError("PPO checkpoint must contain only actor and value-head state")
    else:
        buffer_keys = {"action_scale", "action_bias"}
        q_keys = auxiliary - buffer_keys
        if auxiliary & buffer_keys != buffer_keys or not q_keys:
            raise ValueError("SAC-family checkpoint is missing Q-head or action buffers")
        if any(not key.startswith("q_head.qs.") for key in q_keys):
            raise ValueError("SAC-family checkpoint contains unsupported model state")
        q_indices: set[int] = set()
        for key in q_keys:
            parts = key.split(".")
            if len(parts) < 4 or not parts[2].isdigit():
                raise ValueError("SAC-family Q-head key has an invalid structure")
            q_indices.add(int(parts[2]))
        if q_indices != set(range(int(kwargs["num_q_heads"]))):
            raise ValueError("SAC-family checkpoint Q-head count does not match config")
        for key, expected in (("action_scale", 1.0), ("action_bias", 0.0)):
            value = state_dict[key]
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError(f"policy {key} buffer must be one finite scalar")
            if float(value.detach().cpu().reshape(())) != expected:
                raise ValueError(f"policy {key} buffer has unsupported bounds")

    actor_state = {key: state_dict[key] for key in actor_keys}
    try:
        model.load_state_dict(actor_state, strict=True)
    except RuntimeError as error:
        raise ValueError("policy actor tensor shapes do not match its schema") from error
    model.to(device)
    model.eval()
    return model


def _validate_policy_payload(
    payload: Mapping[str, Any],
    *,
    rlinf_commit: str,
    benchmark_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if payload.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported Dynamic Benchmark policy schema")
    config = payload.get("config")
    state_schema = payload.get("state_schema")
    if not isinstance(config, dict) or not isinstance(state_schema, dict):
        raise ValueError("policy config/state schema must be mappings")
    if config.get("rlinf_commit") != rlinf_commit:
        raise ValueError("policy RLinf commit does not match the requested source")
    if config.get("benchmark_commit") != benchmark_commit:
        raise ValueError("policy benchmark commit does not match the requested source")
    if not isinstance(config.get("task"), str) or not config["task"]:
        raise ValueError("policy task identity is missing")
    state_dim = int(state_schema.get("state_dim", 0))
    mask_dim = int(state_schema.get("mask_dim", -1))
    if state_dim < 1 or not 0 <= mask_dim <= state_dim:
        raise ValueError("policy state schema dimensions are invalid")
    if not isinstance(payload.get("model"), dict) or not isinstance(
        payload.get("normalizer"), dict
    ):
        raise ValueError("policy model/normalizer state is missing")
    _model_kwargs(config, state_dim)
    return config, state_schema


def _latency_summary(latencies_s: list[float]) -> dict[str, Any]:
    values = np.asarray(latencies_s, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("decision latency samples must be a finite non-empty vector")
    return {
        "sample_count": int(values.size),
        "mean_ms": float(values.mean() * 1000.0),
        "p50_ms": float(np.percentile(values, 50) * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
        "p99_ms": float(np.percentile(values, 99) * 1000.0),
        "max_ms": float(values.max() * 1000.0),
        "p95_meets_20hz": bool(np.percentile(values, 95) <= 0.05),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _manifest_row(env: Any, episode_id: str) -> Any:
    matches = [row for row in env._manifest_rows if row.request.episode_id == episode_id]
    if len(matches) != 1:
        raise RuntimeError(f"manifest does not uniquely contain episode {episode_id!r}")
    return matches[0]


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
    """Replay one expert action tape on an independent canonical raw environment."""
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
    """Start one expert rollout from a newly constructed raw environment."""
    if int(vector_env.num_envs) != 1:
        raise ValueError("expert evaluation requires exactly one vector member")
    if request.task_id != vector_env.task_id:
        raise ValueError("expert evaluation request task does not match the environment")
    raw_env = vector_env._make_mujoco_env(
        vector_env.task_id,
        image_size=vector_env.image_size,
        camera_observations=vector_env.camera_observations,
    )
    try:
        if int(raw_env.horizon_steps) != int(vector_env.horizon_steps):
            raise RuntimeError("fresh expert raw environment changed the horizon")
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


def _make_teacher(task: str, request: Any) -> Any:
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    teacher, _ = make_privileged_teacher(task, request=request)
    if hasattr(teacher, "reset"):
        teacher.reset()
    return teacher


def _episode(
    *,
    env: Any,
    model: Any,
    normalizer: RunningNormalizer,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[float]]:
    from se3_wam.benchmark.metrics import (
        completion_time_from_events,
        hierarchical_task_completion,
        validate_stage_event_order,
    )

    request = env._requests[0]
    observation = env._raw_observations[0]
    obs = env._last_obs
    if request is None or observation is None or obs is None:
        raise RuntimeError("evaluation environment is not initialized")
    task_id = str(config["task"])
    row = _manifest_row(env, request.episode_id)
    raw_env = env.envs[0]
    residual = str(config["algorithm"]) == "residual_rlpd"
    teacher = _make_teacher(task_id, request) if residual else None
    residual_scale = float(config.get("residual_scale", 0.25))
    observations = [observation]
    actions = []
    policy_action_values = []
    outcomes = []
    rewards = []
    latencies_s = []
    result_info: dict[str, Any] | None = None
    terminated_value = False
    truncated_value = False
    model.eval()
    while not (terminated_value or truncated_value):
        _sync(device)
        started = time.perf_counter()
        with torch.inference_mode():
            policy_actions, _ = _policy_action(
                model,
                normalizer,
                obs["states"],
                device,
                stochastic=False,
            )
        policy_actions = policy_actions.cpu()
        env_actions = policy_actions
        if teacher is not None:
            env_actions = _compose_residual_actions(
                _planner_actions(env, [teacher]),
                policy_actions,
                residual_scale,
            )
        _sync(device)
        latencies_s.append(time.perf_counter() - started)
        values = np.clip(
            np.asarray(env_actions[0], dtype=np.float64),
            -1.0,
            1.0,
        )
        action = env._ActionCommand(
            mode=request.action_mode,
            values=values,
            policy_step=observation.policy_step,
        )
        next_obs, reward, terminated, truncated, infos = env.step(
            env_actions,
            auto_reset=False,
        )
        next_observation = env._raw_observations[0]
        if next_observation is None:
            raise RuntimeError("evaluation environment lost its raw observation")
        terminated_value = bool(terminated[0])
        truncated_value = bool(truncated[0])
        reason = infos["termination_reason"][0]
        active_progress = float(infos["reward_inputs"]["active_stage_progress"][0])
        observations.append(next_observation)
        actions.append(action)
        policy_action_values.append(np.asarray(policy_actions[0], dtype=np.float64))
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
            "trajectory_completion": float(infos["trajectory_completion"][0]),
        }
        observation = next_observation
        obs = next_obs
        if len(actions) > int(env.horizon_steps):
            raise RuntimeError("evaluation rollout exceeded the environment horizon")
    if result_info is None:
        raise RuntimeError("evaluation policy produced no action")

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
            "expert rollout replay failed: "
            f"{request.episode_id}: {json.dumps(replay_validation, sort_keys=True)}"
        )
    action_array = np.stack([action.values for action in actions])
    policy_action_array = np.stack(policy_action_values)
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
        "termination_reason": result_info["termination_reason"],
        "trajectory_completion": completion,
        "completion_time_s": completion_time,
        "return": float(sum(rewards)),
        "control_steps": len(actions),
        "action_l2_sum": float(np.square(action_array).sum()),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(action_array).tobytes()
        ).hexdigest(),
        "policy_action_sha256": hashlib.sha256(
            np.ascontiguousarray(policy_action_array).tobytes()
        ).hexdigest(),
        "actions": action_array.tolist(),
        "replay_validation": replay_validation,
        "events": [event.name for event in events],
    }
    return record, latencies_s


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    from se3_wam.benchmark.contracts import canonical_json
    from se3_wam.benchmark.evaluation import manifest_record, summarize_task_records

    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    evaluator_commit = _full_commit("evaluator_commit", args.evaluator_commit)
    rlinf_commit = _full_commit("rlinf_commit", args.rlinf_commit)
    benchmark_commit = _full_commit("benchmark_commit", args.benchmark_commit)
    expected_policy_sha256 = _expected_sha256(args.expected_policy_sha256)
    if not args.policy.is_file():
        raise FileNotFoundError(args.policy)
    policy_sha256 = _sha256(args.policy)
    if policy_sha256 != expected_policy_sha256:
        raise ValueError("policy SHA-256 does not match the expected identity")
    payload = torch.load(args.policy, map_location="cpu", weights_only=False)
    config, state_schema = _validate_policy_payload(
        payload,
        rlinf_commit=rlinf_commit,
        benchmark_commit=benchmark_commit,
    )
    device = _device(args.device)
    state_dim = int(state_schema["state_dim"])
    model = _load_inference_policy(config, state_dim, payload["model"], device)
    normalizer = RunningNormalizer(state_dim, int(state_schema["mask_dim"]))
    normalizer.load_state_dict(payload["normalizer"])

    manifest_size = max(args.episodes, 2)
    if manifest_size % 2:
        manifest_size += 1
    env = DynamicBenchmarkEnv(
        cfg={
            "task_id": config["task"],
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "manifest_size": manifest_size,
            "image_size": int(config.get("image_size", 64)),
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
        if env.state_schema != state_schema:
            raise ValueError("evaluation environment state schema does not match the policy")
        rows = list(env._manifest_rows[: args.episodes])
        if len(rows) != args.episodes:
            raise RuntimeError("evaluation manifest is shorter than requested")
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
            record, episode_latencies = _episode(
                env=env,
                model=model,
                normalizer=normalizer,
                config=config,
                device=device,
            )
            if record["episode_id"] != row.request.episode_id:
                raise RuntimeError("rollout order diverged from the frozen reset manifest")
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
        task_id = str(config["task"])
        task_summary = summarize_task_records([task_id], records)
        task_summary[task_id].update(
            safety_failure_rate=float(np.mean([row["safety_failure"] for row in records])),
            mean_return=float(np.mean([row["return"] for row in records])),
            mean_action_l2_sum=float(np.mean([row["action_l2_sum"] for row in records])),
        )
        latency = _latency_summary(latencies_s)
        result = {
            "schema_version": EVALUATION_SCHEMA,
            "policy_identity": {
                "path": str(args.policy.resolve()),
                "sha256": policy_sha256,
                "schema_version": payload["schema_version"],
                "task": task_id,
                "algorithm": config["algorithm"],
                "training_seed": config["seed"],
                "training_env_steps": payload["env_steps"],
                "validation": payload["validation"],
            },
            "source_identity": {
                "evaluator_rlinf_commit": evaluator_commit,
                "policy_rlinf_commit": rlinf_commit,
                "benchmark_commit": benchmark_commit,
            },
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "reset_manifest_sha256": _sha256(reset_manifest_path),
            "episodes": args.episodes,
            "device": str(device),
            "state_schema": state_schema,
            "records": records,
            "task_summary": task_summary,
            "decision_latency": latency,
            "all_replays_passed": all(
                row["replay_validation"]["passed"] for row in records
            ),
            "started_unix_s": started,
            "finished_unix_s": time.time(),
        }
        if _sha256(args.policy) != policy_sha256:
            raise RuntimeError("policy file changed during evaluation")
        result["payload_sha256"] = _payload_sha256(result)
        result_path = args.output / "evaluation.json"
        _atomic_json(result_path, result)
        (args.output / "SHA256SUMS").write_text(
            (
                f"{_sha256(result_path)}  evaluation.json\n"
                f"{_sha256(reset_manifest_path)}  reset_manifest.jsonl\n"
                f"{policy_sha256}  {args.policy.resolve()}\n"
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
                    "task_summary": task_summary,
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
