#!/usr/bin/env python3
"""Verify exact and faster residual-planner validation in environment processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    _config,
    _env_cfg,
    _evaluate,
    _parse_args,
)


class _ZeroResidualPolicy:
    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None

    def _sample_actions(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (states.shape[0], 7)
        return (
            torch.zeros(shape, dtype=torch.float32, device=states.device),
            torch.full(shape, -20.0, dtype=torch.float32, device=states.device),
        )


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="t4_sphere")
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--worker-processes", type=int, default=8)
    parser.add_argument("--eval-num-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--manifest-seed", type=int, default=20262150)
    parser.add_argument("--manifest-size", type=int, default=64)
    parser.add_argument("--process-start-method", default="spawn")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    args = _parser().parse_args()
    if args.worker_processes < 1 or args.eval_num_envs < 1 or args.eval_episodes < 1:
        raise ValueError("worker, vector, and episode counts must be positive")
    if args.eval_num_envs > args.eval_episodes:
        raise ValueError("eval_num_envs cannot exceed eval_episodes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing existing output {args.output}")

    base = _config(
        _parse_args(
            [
                "--task",
                args.task,
                "--algorithm",
                "residual_rlpd",
                "--rlinf-commit",
                args.rlinf_commit,
                "--benchmark-commit",
                args.benchmark_commit,
                "--output",
                str(args.output.parent / "unused-trainer-output"),
                "--eval-num-envs",
                str(args.eval_num_envs),
                "--eval-worker-processes",
                str(args.worker_processes),
                "--eval-episodes",
                str(args.eval_episodes),
                "--validation-manifest-seed",
                str(args.manifest_seed),
                "--manifest-size",
                str(args.manifest_size),
                "--process-start-method",
                args.process_start_method,
            ]
        )
    )
    control_config = replace(base, eval_planner_in_processes=False)
    process_config = replace(base, eval_planner_in_processes=True)

    probe = DynamicBenchmarkEnv(
        cfg=_env_cfg(
            base,
            split="validation",
            seed=args.manifest_seed,
            num_envs=1,
            worker_threads=1,
        ),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        schema = probe.state_schema
    finally:
        probe.close()
    normalizer = RunningNormalizer(
        dimension=int(schema["state_dim"]),
        mask_dim=int(schema["mask_dim"]),
    )
    policy = _ZeroResidualPolicy()
    device = torch.device("cpu")

    control = _evaluate(control_config, policy, normalizer, device)
    candidate = _evaluate(process_config, policy, normalizer, device)
    exact_keys = (
        "episodes",
        "success_rate",
        "safety_failure_rate",
        "mean_completion",
        "mean_return",
        "mean_duration_steps",
        "mean_action_l2_sum",
        "action_digest_sha256",
        "state_digest_sha256",
        "records",
    )
    mismatches = {
        key: {"control": control[key], "candidate": candidate[key]}
        for key in exact_keys
        if control[key] != candidate[key]
    }
    if mismatches:
        raise RuntimeError(
            "process planner evaluation differs from main-process planner: "
            + json.dumps(mismatches, sort_keys=True)
        )

    payload = {
        "schema_version": "rle-u1-process-planner-eval-gate-v0.1",
        "task_id": args.task,
        "rlinf_commit": args.rlinf_commit,
        "benchmark_commit": args.benchmark_commit,
        "pid": os.getpid(),
        "worker_processes": args.worker_processes,
        "eval_num_envs": args.eval_num_envs,
        "eval_episodes": args.eval_episodes,
        "manifest_seed": args.manifest_seed,
        "episode_order_exact": True,
        "science_metrics_exact": True,
        "action_digest_exact": True,
        "state_digest_exact": True,
        "control": control,
        "candidate": candidate,
        "wall_speedup": control["wall_time_s"] / candidate["wall_time_s"],
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
