#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
"""Small real-runtime Dynamic Benchmark reset/step/replay smoke for RLD2-QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="p0_grasp")
    parser.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--manifest-seed", type=int, default=20262150)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    from se3_wam.benchmark.task_quality import task_quality_schema_manifest

    from examples.embodiment.prepare_dynamic_benchmark_rld2_launch_gate import (
        BACKEND_ID,
    )
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    config = {
        "task_id": args.task,
        "split": args.split,
        "manifest_seed": args.manifest_seed,
        "manifest_size": 2,
        "image_size": 64,
        "camera_observations": False,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
        "task_quality_schema_version": task_quality_schema_manifest(args.task)[
            "schema_version"
        ],
        "task_quality_evaluator_backend_id": BACKEND_ID,
        "reward_components": {"r_action_rate": {"weight": 0.10, "scale": 1.0}},
        "features": {"action_history": {"k": 3}},
    }
    env = DynamicBenchmarkEnv(
        cfg=config,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    actions = []
    observations = []
    rewards = []
    try:
        observation, _ = env.reset()
        observations.append(env._raw_observations[0])
        for _ in range(args.steps):
            action = torch.zeros((1, 7), dtype=torch.float32)
            observations_next, reward, terminated, truncated, infos = env.step(
                action,
                auto_reset=False,
            )
            observations.append(env._raw_observations[0])
            actions.append(action.numpy()[0])
            rewards.append(float(reward[0]))
            if bool(terminated[0]) or bool(truncated[0]):
                break
        if not actions:
            raise RuntimeError("runtime smoke produced no action step")
        if not np.all(np.isfinite(np.asarray(actions))):
            raise RuntimeError("runtime smoke actions are not finite")
        if not np.all(np.isfinite(np.asarray(rewards))):
            raise RuntimeError("runtime smoke rewards are not finite")
        request = env._requests[0]
        if request is None:
            raise RuntimeError("runtime smoke lost reset request")
        privileged = getattr(env._raw_observations[0], "privileged", {})
        initial_pose = np.asarray(
            getattr(observations[0], "privileged", {})["eef_pose_xyzw"],
            dtype=np.float64,
        )
        x, y, z, w = initial_pose[3:]
        initial_rotation = np.asarray(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )
        payload = {
            "task": args.task,
            "split": args.split,
            "episode_id": request.episode_id,
            "steps": len(actions),
            "reward_sum": float(np.sum(rewards, dtype=np.float64)),
            "terminated": bool(terminated[0]),
            "truncated": bool(truncated[0]),
            "state_dim": int(observation["states"].shape[-1]),
            "task_quality_schema_version": env.task_quality_schema_version,
            "task_quality_evaluator_backend_id": env.task_quality_evaluator_backend_id,
            "privileged_keys": sorted(str(key) for key in privileged),
            "initial_eef_pose_xyzw": initial_pose.tolist(),
            "initial_eef_rotation_columns": initial_rotation.T.tolist(),
            "task_quality_present": bool(infos["task_quality"][0] is not None),
            "reward_components": sorted(infos["reward_components"]),
            "reset_replay_ready": len(observations) == len(actions) + 1,
        }
    finally:
        env.close()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
