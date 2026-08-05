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

"""Verify real vector reward recomputation and exact checkpoint resume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv
from rlinf.envs.dynamic_benchmark.reward import DynamicBenchmarkReward


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="t2_trans")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--prefix-steps", type=int, default=8)
    parser.add_argument("--suffix-steps", type=int, default=8)
    parser.add_argument("--chunk-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _cfg(task: str, num_envs: int) -> dict[str, Any]:
    return {
        "task_id": task,
        "split": "train",
        "manifest_seed": 20261050,
        "manifest_size": max(8, num_envs * 2),
        "image_size": 64,
        "auto_reset": True,
        "ignore_terminations": False,
        "group_size": 1,
        "task_prompt": f"Solve Dynamic Benchmark task {task}.",
    }


def _actions(num_envs: int, steps: int, seed: int) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    return [
        torch.as_tensor(rng.uniform(-0.1, 0.1, size=(num_envs, 7)), dtype=torch.float32)
        for _ in range(steps)
    ]


def _independent_trackers(env: DynamicBenchmarkEnv) -> list[DynamicBenchmarkReward]:
    schema = dict(env.reward_schema)
    return [
        DynamicBenchmarkReward(
            success_stages=schema["success_stages"],
            progress_scale=schema["progress_scale"],
            success_bonus=schema["success_bonus"],
            failure_penalty=schema["failure_penalty"],
            safety_penalty=schema["safety_penalty"],
            timeout_penalty=schema["timeout_penalty"],
            step_penalty=schema["step_penalty"],
            action_l2_scale=schema["action_l2_scale"],
            lift_shaping_weight=schema.get("lift_shaping_weight", 0.0),
            orientation_shaping_weight=schema.get(
                "orientation_shaping_weight", 0.0
            ),
            lift_target_m=schema.get("lift_target_m", 0.08),
            lift_hold_event=schema.get("lift_hold_event", "bilateral_hold"),
            safety_failures=schema["safety_failures"],
        )
        for _ in range(env.num_envs)
    ]


def _verify_reward(
    trackers: list[DynamicBenchmarkReward],
    reward: torch.Tensor,
    infos: dict[str, Any],
) -> float:
    maximum_error = 0.0
    inputs = infos["reward_inputs"]
    components = infos["reward_components"]
    for index, tracker in enumerate(trackers):
        if not bool(inputs["stepped"][index]):
            continue
        object_z_m = inputs.get("object_z_m")
        alignment_error_rad = inputs.get("alignment_error_rad")
        object_z_value = (
            None
            if object_z_m is None or not np.isfinite(float(object_z_m[index]))
            else float(object_z_m[index])
        )
        alignment_value = (
            None
            if alignment_error_rad is None
            or not np.isfinite(float(alignment_error_rad[index]))
            else float(alignment_error_rad[index])
        )
        total, expected = tracker.step(
            action=inputs["action"][index].numpy(),
            event_names=inputs["event_names"][index],
            active_stage_progress=float(inputs["active_stage_progress"][index]),
            success=bool(inputs["success"][index]),
            terminated=bool(inputs["terminated"][index]),
            truncated=bool(inputs["truncated"][index]),
            termination_reason=inputs["termination_reason"][index],
            object_z_m=object_z_value,
            alignment_error_rad=alignment_value,
        )
        maximum_error = max(maximum_error, abs(total - float(reward[index])))
        for name, value in expected.items():
            maximum_error = max(
                maximum_error, abs(value - float(components[name][index]))
            )
    if maximum_error > 1e-6:
        raise RuntimeError(f"independent reward mismatch: max_error={maximum_error}")
    return maximum_error


def _step_digest(result: tuple[Any, ...]) -> str:
    obs, reward, terminated, truncated, infos = result
    payload = {
        "states": obs["states"].numpy().tolist(),
        "reward": reward.numpy().tolist(),
        "terminated": terminated.numpy().tolist(),
        "truncated": truncated.numpy().tolist(),
        "success": infos["success"].numpy().tolist(),
        "completion": infos["trajectory_completion"].numpy().tolist(),
        "reason": infos["termination_reason"],
        "components": {
            key: value.numpy().tolist()
            for key, value in sorted(infos["reward_components"].items())
        },
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def _checkpoint_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(state["identity_sha256"].encode())
    for env_state in state["env_states"]:
        digest.update(env_state)
    for observation_sha256 in state["raw_observation_sha256"]:
        digest.update(observation_sha256.encode())
    digest.update(state["last_obs_sha256"].encode())
    for key in ("manifest_generation", "manifest_cursor"):
        digest.update(str(state[key]).encode())
    return digest.hexdigest()


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if min(args.num_envs, args.prefix_steps, args.suffix_steps, args.chunk_steps) < 1:
        raise ValueError("all vector and step counts must be positive")
    cfg = _cfg(args.task, args.num_envs)
    env = DynamicBenchmarkEnv(
        cfg=cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    restored = None
    chunk_env = None
    try:
        observation, _ = env.reset()
        if observation["states"].shape != (args.num_envs, env.state_schema["state_dim"]):
            raise RuntimeError("vector reset state shape mismatch")
        episode_ids = [request.episode_id for request in env._requests]
        if len(set(episode_ids)) != args.num_envs:
            raise RuntimeError("vector reset reused an episode identity")
        trackers = _independent_trackers(env)
        maximum_reward_error = 0.0
        for action in _actions(args.num_envs, args.prefix_steps, 20261051):
            result = env.step(action, auto_reset=False)
            maximum_reward_error = max(
                maximum_reward_error, _verify_reward(trackers, result[1], result[4])
            )
        checkpoint = env.checkpoint_state()
        checkpoint_sha256 = _checkpoint_sha256(checkpoint)
        suffix_actions = _actions(args.num_envs, args.suffix_steps, 20261052)
        uninterrupted = [env.step(action, auto_reset=False) for action in suffix_actions]

        restored = DynamicBenchmarkEnv(
            cfg=cfg,
            num_envs=args.num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        restored.load_checkpoint_state(checkpoint)
        resumed = [restored.step(action, auto_reset=False) for action in suffix_actions]
        uninterrupted_digests = [_step_digest(result) for result in uninterrupted]
        resumed_digests = [_step_digest(result) for result in resumed]
        if uninterrupted_digests != resumed_digests:
            raise RuntimeError("checkpoint/resume trajectory digests do not match")

        chunk_env = DynamicBenchmarkEnv(
            cfg=cfg,
            num_envs=args.num_envs,
            seed_offset=1,
            total_num_processes=1,
            worker_info=None,
        )
        chunk_actions = torch.stack(
            _actions(args.num_envs, args.chunk_steps, 20261053), dim=1
        )
        obs_list, rewards, terminations, truncations, infos_list = chunk_env.chunk_step(
            chunk_actions
        )
        chunk_shapes = {
            "obs_steps": len(obs_list),
            "rewards": list(rewards.shape),
            "terminations": list(terminations.shape),
            "truncations": list(truncations.shape),
            "info_steps": len(infos_list),
        }
        expected_shape = [args.num_envs, args.chunk_steps]
        if (
            chunk_shapes["rewards"] != expected_shape
            or chunk_shapes["terminations"] != expected_shape
            or chunk_shapes["truncations"] != expected_shape
        ):
            raise RuntimeError(f"chunk_step shape mismatch: {chunk_shapes}")
        result = {
            "schema_version": "rlinf-dynamic-benchmark-runtime-gate-v0.1",
            "task_id": args.task,
            "num_envs": args.num_envs,
            "prefix_steps": args.prefix_steps,
            "suffix_steps": args.suffix_steps,
            "chunk_steps": args.chunk_steps,
            "state_schema": env.state_schema,
            "reward_schema": dict(env.reward_schema),
            "episode_ids": episode_ids,
            "maximum_reward_recompute_abs_error": maximum_reward_error,
            "checkpoint_sha256": checkpoint_sha256,
            "uninterrupted_step_digests": uninterrupted_digests,
            "resumed_step_digests": resumed_digests,
            "checkpoint_resume_exact": uninterrupted_digests == resumed_digests,
            "chunk_shapes": chunk_shapes,
            "all_gates_passed": True,
        }
    finally:
        env.close()
        if restored is not None:
            restored.close()
        if chunk_env is not None:
            chunk_env.close()
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
