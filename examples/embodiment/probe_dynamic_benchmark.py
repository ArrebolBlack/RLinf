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

"""Probe one real Dynamic Benchmark reset and emit the RLinf state contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest-seed", type=int, default=20261050)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    # Keep this low-level runtime probe independent from Hydra/OmegaConf.  The
    # adapter deliberately accepts plain mappings so that environment and
    # reward correctness can be checked before the full RLinf scheduler stack
    # is installed on a compute node.
    cfg = {
        "task_id": args.task,
        "split": args.split,
        "manifest_seed": args.manifest_seed,
        "manifest_size": 2,
        "image_size": args.image_size,
        "auto_reset": True,
        "ignore_terminations": False,
        "group_size": 1,
        "task_prompt": f"Solve Dynamic Benchmark task {args.task}.",
    }
    env = DynamicBenchmarkEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        observation, _ = env.reset()
        action = env.sample_action_space().unsqueeze(0) * 0.0
        _, reward, terminated, truncated, infos = env.step(action, auto_reset=False)
        result = {
            "schema_version": "rlinf-dynamic-benchmark-probe-v0.1",
            "task_id": args.task,
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "image_size": args.image_size,
            "state_schema": env.state_schema,
            "reward_schema": dict(env.reward_schema),
            "observation_shape": list(observation["states"].shape),
            "step": {
                "reward": float(reward[0]),
                "terminated": bool(terminated[0]),
                "truncated": bool(truncated[0]),
                "trajectory_completion": float(infos["trajectory_completion"][0]),
            },
        }
    finally:
        env.close()
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
