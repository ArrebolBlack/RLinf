#!/usr/bin/env python3
"""Gate validation borrowed from the training process pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    _BorrowedEvaluationRuntime,
    _config,
    _env_cfg,
    _evaluate,
    _make_planner_teachers,
    _parse_args,
)
from examples.embodiment.verify_dynamic_benchmark_runtime import (
    _checkpoint_sha256,
    _step_digest,
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
        raise RuntimeError("intentional borrowed evaluation failure")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="t4_sphere")
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--worker-processes", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--train-manifest-seed", type=int, default=20262050)
    parser.add_argument("--validation-manifest-seed", type=int, default=20262150)
    parser.add_argument("--manifest-size", type=int, default=64)
    parser.add_argument("--process-start-method", default="spawn")
    parser.add_argument("--cleanup-timeout-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_nested_equal(expected: Any, observed: Any, path: str = "checkpoint") -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(observed, torch.Tensor) or not torch.equal(expected, observed):
            raise RuntimeError(f"{path} tensor diverged")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(observed, np.ndarray) or not np.array_equal(expected, observed):
            raise RuntimeError(f"{path} array diverged")
        return
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(expected) != set(observed):
            raise RuntimeError(f"{path} mapping keys diverged")
        for key in expected:
            _assert_nested_equal(expected[key], observed[key], f"{path}.{key}")
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(observed, Sequence) or len(expected) != len(observed):
            raise RuntimeError(f"{path} sequence shape diverged")
        for index, (expected_item, observed_item) in enumerate(
            zip(expected, observed, strict=True)
        ):
            _assert_nested_equal(expected_item, observed_item, f"{path}[{index}]")
        return
    if expected != observed:
        raise RuntimeError(f"{path} value diverged")


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
    if args.output.exists():
        raise FileExistsError(f"refusing existing output {args.output}")
    if min(args.worker_processes, args.num_envs, args.eval_episodes) < 1:
        raise ValueError("worker, vector, and episode counts must be positive")
    if args.worker_processes > args.num_envs:
        raise ValueError("worker_processes may not exceed num_envs")
    if args.num_envs > args.eval_episodes:
        raise ValueError("num_envs may not exceed eval_episodes")

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
                "--num-envs",
                str(args.num_envs),
                "--eval-num-envs",
                str(args.num_envs),
                "--env-worker-processes",
                str(args.worker_processes),
                "--eval-worker-processes",
                str(args.worker_processes),
                "--eval-episodes",
                str(args.eval_episodes),
                "--train-manifest-seed",
                str(args.train_manifest_seed),
                "--validation-manifest-seed",
                str(args.validation_manifest_seed),
                "--manifest-size",
                str(args.manifest_size),
                "--process-start-method",
                args.process_start_method,
                "--borrow-training-env-for-eval",
            ]
        )
    )
    env = DynamicBenchmarkEnv(
        cfg=_env_cfg(
            config,
            split="train",
            seed=args.train_manifest_seed,
            num_envs=args.num_envs,
            worker_threads=1,
            worker_processes=args.worker_processes,
            process_start_method=args.process_start_method,
        ),
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=args.worker_processes,
        worker_info=None,
    )
    reference = None
    worker_pids = tuple(env.process_worker_pids)
    try:
        normalizer = RunningNormalizer(
            dimension=int(env.state_schema["state_dim"]),
            mask_dim=int(env.state_schema["mask_dim"]),
        )
        policy = _ZeroResidualPolicy()
        device = torch.device("cpu")
        training_teachers = _make_planner_teachers(config.task, env)
        runtime = _BorrowedEvaluationRuntime(
            env=env,
            training_teachers=training_teachers,
        )
        training_checkpoint = env.checkpoint_state()
        checkpoint_sha256 = _checkpoint_sha256(training_checkpoint)

        recreated = _evaluate(config, policy, normalizer, device)
        first = _evaluate(config, policy, normalizer, device, runtime=runtime)
        after_first = env.checkpoint_state()
        _assert_nested_equal(training_checkpoint, after_first)
        second = _evaluate(config, policy, normalizer, device, runtime=runtime)
        after_second = env.checkpoint_state()
        _assert_nested_equal(training_checkpoint, after_second)

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
            key: {"recreated": recreated[key], "borrowed": first[key]}
            for key in exact_keys
            if recreated[key] != first[key]
        }
        repeat_mismatches = {
            key: {"first": first[key], "second": second[key]}
            for key in exact_keys
            if first[key] != second[key]
        }
        if recreate_mismatches or repeat_mismatches:
            raise RuntimeError(
                "borrowed evaluation parity failed: "
                + json.dumps(
                    {"recreated": recreate_mismatches, "repeated": repeat_mismatches},
                    sort_keys=True,
                )
            )
        if tuple(first["worker_pids"]) != worker_pids:
            raise RuntimeError("borrowed evaluation did not use the training worker PIDs")
        if first["evaluation_rewind_mode"] != "borrow_training_pool":
            raise RuntimeError("borrowed evaluation mode was not reported")

        failure_restore_passed = False
        try:
            _evaluate(config, _FailingPolicy(), normalizer, device, runtime=runtime)
        except RuntimeError as exc:
            if str(exc) != "intentional borrowed evaluation failure":
                raise
            _assert_nested_equal(training_checkpoint, env.checkpoint_state())
            failure_restore_passed = not runtime.closed
        if not failure_restore_passed:
            raise RuntimeError("borrowed evaluation failure did not restore training state")

        reference = DynamicBenchmarkEnv(
            cfg=_env_cfg(
                config,
                split="train",
                seed=args.train_manifest_seed,
                num_envs=args.num_envs,
                worker_threads=1,
                worker_processes=args.worker_processes,
                process_start_method=args.process_start_method,
            ),
            num_envs=args.num_envs,
            seed_offset=0,
            total_num_processes=args.worker_processes,
            worker_info=None,
        )
        reference.load_checkpoint_state(training_checkpoint)
        actions = torch.zeros((args.num_envs, 7), dtype=torch.float32)
        continued_digest = _step_digest(env.step(actions, auto_reset=False))
        reference_digest = _step_digest(reference.step(actions, auto_reset=False))
        if continued_digest != reference_digest:
            raise RuntimeError("post-evaluation training continuation diverged")

        result = {
            "schema_version": "rle-u1-borrowed-eval-gate-v0.1",
            "task_id": args.task,
            "rlinf_commit": args.rlinf_commit,
            "benchmark_commit": args.benchmark_commit,
            "pid": os.getpid(),
            "worker_processes": args.worker_processes,
            "num_envs": args.num_envs,
            "eval_episodes": args.eval_episodes,
            "worker_pids": worker_pids,
            "checkpoint_sha256": checkpoint_sha256,
            "recreated_borrowed_exact": True,
            "repeated_evaluation_exact": True,
            "checkpoint_restore_exact": True,
            "failure_restore_passed": True,
            "continuation_exact": True,
            "continued_step_digest": continued_digest,
            "recreated": recreated,
            "borrowed_first": first,
            "borrowed_second": second,
        }
        runtime.close()
    finally:
        if reference is not None:
            reference.close()
        env.close()

    deadline = time.monotonic() + args.cleanup_timeout_s
    while _live_pids(worker_pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    result["worker_cleanup_passed"] = not _live_pids(worker_pids)
    if not result["worker_cleanup_passed"]:
        raise RuntimeError("borrowed training pool left live workers after close")
    result["all_gates_passed"] = True
    result["payload_sha256"] = _payload_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
