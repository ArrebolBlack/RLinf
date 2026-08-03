#!/usr/bin/env python3
"""Verify exact persistent evaluation reuse, speedup, and failure cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    _config,
    _env_cfg,
    _evaluate,
    _EvaluationRuntime,
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


class _FailingPolicy(_ZeroResidualPolicy):
    def _sample_actions(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del states
        raise RuntimeError("intentional persistent evaluation failure")


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
    parser.add_argument("--cleanup-timeout-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _live_pids(pids: tuple[int, ...]) -> list[int]:
    live = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        live.append(pid)
    return live


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

    config = _config(
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
                "--persistent-eval-workers",
            ]
        )
    )
    probe = DynamicBenchmarkEnv(
        cfg=_env_cfg(
            config,
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

    recreated = _evaluate(config, policy, normalizer, device)
    runtime = _EvaluationRuntime()
    first = _evaluate(config, policy, normalizer, device, runtime=runtime)
    first_pids = tuple(first["worker_pids"])
    second = _evaluate(config, policy, normalizer, device, runtime=runtime)
    second_pids = tuple(second["worker_pids"])
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
    recreate_mismatches = {
        key: {"recreated": recreated[key], "persistent": first[key]}
        for key in exact_keys
        if recreated[key] != first[key]
    }
    repeat_mismatches = {
        key: {"first": first[key], "second": second[key]}
        for key in exact_keys
        if first[key] != second[key]
    }
    if recreate_mismatches or repeat_mismatches:
        runtime.close()
        raise RuntimeError(
            "persistent evaluation parity failed: "
            + json.dumps(
                {
                    "recreated": recreate_mismatches,
                    "repeated": repeat_mismatches,
                },
                sort_keys=True,
            )
        )
    if first_pids != second_pids:
        runtime.close()
        raise RuntimeError("persistent evaluation worker PIDs changed across evaluations")
    if (
        first["persistent_evaluation_index"] != 0
        or second["persistent_evaluation_index"] != 1
        or first["environment_construction_s"] <= 0.0
        or first["environment_restore_s"] != 0.0
        or second["environment_construction_s"] != 0.0
        or second["environment_restore_s"] <= 0.0
    ):
        runtime.close()
        raise RuntimeError("persistent evaluation lifecycle timings are inconsistent")

    failure_cleanup_passed = False
    try:
        _evaluate(
            config,
            _FailingPolicy(),
            normalizer,
            device,
            runtime=runtime,
        )
    except RuntimeError as exc:
        if str(exc) != "intentional persistent evaluation failure":
            raise
        deadline = time.monotonic() + args.cleanup_timeout_s
        while _live_pids(first_pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        failure_cleanup_passed = runtime.closed and not _live_pids(first_pids)
    if not failure_cleanup_passed:
        runtime.close()
        raise RuntimeError("persistent evaluation failure cleanup left live workers")

    payload = {
        "schema_version": "rle-u1-persistent-eval-gate-v0.1",
        "task_id": args.task,
        "rlinf_commit": args.rlinf_commit,
        "benchmark_commit": args.benchmark_commit,
        "pid": os.getpid(),
        "worker_processes": args.worker_processes,
        "eval_num_envs": args.eval_num_envs,
        "eval_episodes": args.eval_episodes,
        "manifest_seed": args.manifest_seed,
        "recreated_persistent_exact": True,
        "repeated_evaluation_exact": True,
        "worker_pids_stable": True,
        "failure_cleanup_passed": True,
        "recreated": recreated,
        "persistent_first": first,
        "persistent_second": second,
        "steady_state_wall_speedup": (
            recreated["wall_time_s"] / second["wall_time_s"]
        ),
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
