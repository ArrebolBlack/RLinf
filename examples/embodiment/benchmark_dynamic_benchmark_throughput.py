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

"""Measure real Dynamic Benchmark construction and steady vector-step throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="t2_trans")
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--camera-observations", action="store_true")
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _full_commit(name: str, value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


def main() -> None:
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num_envs and steps must be positive")
    rlinf_commit = _full_commit("rlinf_commit", args.rlinf_commit)
    benchmark_commit = _full_commit("benchmark_commit", args.benchmark_commit)
    cfg: dict[str, Any] = {
        "task_id": args.task,
        "split": "train",
        "manifest_seed": 20261250,
        "manifest_size": max(32, args.num_envs * 2),
        "image_size": args.image_size,
        "camera_observations": args.camera_observations,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
    }
    started = time.perf_counter()
    env = DynamicBenchmarkEnv(
        cfg=cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    construction_seconds = time.perf_counter() - started
    try:
        if env._last_obs is None:
            raise RuntimeError("environment did not initialize its state")
        actions = torch.zeros((args.num_envs, 7), dtype=torch.float32)
        rollout_started = time.perf_counter()
        terminal_count = 0
        for step in range(args.steps):
            _, _, terminated, truncated, _ = env.step(actions, auto_reset=False)
            if terminated.any() or truncated.any():
                terminal_count += int(torch.logical_or(terminated, truncated).sum())
                raise RuntimeError(
                    "throughput window reached a terminal state: "
                    f"step={step}, count={terminal_count}"
                )
        rollout_seconds = time.perf_counter() - rollout_started
        transitions = args.num_envs * args.steps
        result = {
            "schema_version": "rlinf-dynamic-benchmark-throughput-v0.1",
            "task_id": args.task,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "transitions": transitions,
            "camera_observations": args.camera_observations,
            "image_size": args.image_size,
            "construction_seconds": construction_seconds,
            "rollout_seconds": rollout_seconds,
            "vector_steps_per_second": args.steps / rollout_seconds,
            "transitions_per_second": transitions / rollout_seconds,
            "seconds_per_transition": rollout_seconds / transitions,
            "terminal_count": terminal_count,
            "state_schema": env.state_schema,
            "reward_schema": dict(env.reward_schema),
            "rlinf_commit": rlinf_commit,
            "benchmark_commit": benchmark_commit,
            "resource_id": os.environ.get("SE3WAM_RESOURCE_ID"),
            "lane_id": os.environ.get("SE3WAM_LANE_ID"),
            "expected_gpu_uuid": os.environ.get("SE3WAM_EXPECTED_GPU_UUID"),
            "cpu_set": os.environ.get("SE3WAM_CPU_SET"),
        }
    finally:
        env.close()
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
