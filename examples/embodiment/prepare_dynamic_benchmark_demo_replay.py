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

"""Materialize one source-bound planner demonstration cache without training."""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    TransitionReplay,
    _atomic_json,
    _collect_demos,
    _config,
    _demo_replay_identity,
    _env_cfg,
    _file_sha256,
    _parse_args,
    _save_demo_replay_cache,
)


def main() -> None:
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    args = _parse_args()
    config = _config(args)
    if config.algorithm not in {"bc", "rlpd", "residual_rlpd"}:
        raise ValueError("demo materialization requires a demonstration-based algorithm")
    if args.resume is not None or args.demo_replay_in is not None:
        raise ValueError("demo materialization creates a fresh cache and cannot resume/load one")
    if config.demo_seed != config.seed or config.demo_rlinf_commit != config.rlinf_commit:
        raise ValueError("fresh demo identity must match the explicit producer seed/source")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(config.demo_seed)
    np.random.seed(config.demo_seed)
    torch.manual_seed(config.demo_seed)
    probe = DynamicBenchmarkEnv(
        cfg=_env_cfg(
            config,
            split="train",
            seed=config.train_manifest_seed + 700_001,
            num_envs=1,
            worker_threads=1,
        ),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        state_schema = probe.state_schema
    finally:
        probe.close()
    state_dim = int(state_schema["state_dim"])
    replay = TransitionReplay(
        config.replay_capacity,
        state_dim,
        config.demo_seed + 11,
        storage="pageable_cpu",
    )
    normalizer = RunningNormalizer(state_dim, int(state_schema["mask_dim"]))
    summary = _collect_demos(config, replay, normalizer)
    identity = _demo_replay_identity(config, state_schema)
    cache_path = args.output / "demo_replay.pt"
    cache_sha256 = _save_demo_replay_cache(
        cache_path,
        identity,
        summary,
        replay,
        normalizer,
    )
    receipt = {
        "schema_version": "rlinf-dynamic-benchmark-demo-materialization-v0.1",
        "status": "complete",
        "config": asdict(config),
        "identity": identity,
        "demo_summary": summary,
        "cache_path": cache_path.name,
        "cache_sha256": cache_sha256,
        "producer_pid": os.getpid(),
    }
    _atomic_json(args.output / "demo_materialization.json", receipt)
    if _file_sha256(cache_path) != cache_sha256:
        raise RuntimeError("demo cache changed after sealing")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
