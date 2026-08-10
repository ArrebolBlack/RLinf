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

"""Gate ordered Dynamic Benchmark subprocess stepping against the serial backend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from examples.embodiment.verify_dynamic_benchmark_runtime import (
    _actions,
    _checkpoint_sha256,
    _step_digest,
)
from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="t4_sphere")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--worker-processes", type=int, default=4)
    parser.add_argument("--prefix-steps", type=int, default=8)
    parser.add_argument("--suffix-steps", type=int, default=8)
    parser.add_argument(
        "--process-start-method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
    )
    parser.add_argument("--process-timeout-s", type=float, default=120.0)
    parser.add_argument("--skip-failure-cleanup", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _cfg(args: argparse.Namespace, *, worker_processes: int) -> dict[str, Any]:
    return {
        "task_id": args.task,
        "split": "validation",
        "manifest_seed": 20262150,
        "manifest_size": max(16, args.num_envs * 2),
        "image_size": 64,
        "camera_observations": False,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
        "worker_threads": 1,
        "worker_processes": worker_processes,
        "process_start_method": args.process_start_method,
        "process_timeout_s": args.process_timeout_s,
    }


def _make_env(
    args: argparse.Namespace, *, worker_processes: int
) -> DynamicBenchmarkEnv:
    return DynamicBenchmarkEnv(
        cfg=_cfg(args, worker_processes=worker_processes),
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=max(1, worker_processes),
        worker_info=None,
    )


def _reset_digest(env: DynamicBenchmarkEnv) -> str:
    if env._last_obs is None:
        raise RuntimeError("process runtime gate environment did not initialize")
    payload = {
        "episode_ids": [request.episode_id for request in env._requests],
        "states": env._last_obs["states"].numpy().tolist(),
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        min(
            args.num_envs,
            args.worker_processes,
            args.prefix_steps,
            args.suffix_steps,
        )
        < 1
    ):
        raise ValueError("environment, process, and step counts must be positive")
    if args.worker_processes > args.num_envs:
        raise ValueError("worker_processes may not exceed num_envs in the runtime gate")

    serial = _make_env(args, worker_processes=0)
    processes = _make_env(args, worker_processes=args.worker_processes)
    restored = None
    failure_env = None
    try:
        serial_reset_digest = _reset_digest(serial)
        process_reset_digest = _reset_digest(processes)
        if serial_reset_digest != process_reset_digest:
            raise RuntimeError("serial/process reset digests do not match")

        actions = _actions(args.num_envs, args.prefix_steps, 20263061)
        serial_digests = [
            _step_digest(serial.step(action, auto_reset=False)) for action in actions
        ]
        process_digests = [
            _step_digest(processes.step(action, auto_reset=False)) for action in actions
        ]
        if serial_digests != process_digests:
            raise RuntimeError("serial/process step digests do not match")

        checkpoint = processes.checkpoint_state()
        checkpoint_sha256 = _checkpoint_sha256(checkpoint)
        suffix_actions = _actions(args.num_envs, args.suffix_steps, 20263062)
        uninterrupted = [
            _step_digest(processes.step(action, auto_reset=False))
            for action in suffix_actions
        ]
        restored = _make_env(args, worker_processes=args.worker_processes)
        restored.load_checkpoint_state(checkpoint)
        resumed = [
            _step_digest(restored.step(action, auto_reset=False))
            for action in suffix_actions
        ]
        if uninterrupted != resumed:
            raise RuntimeError("process checkpoint/resume digests do not match")

        process_vector = processes._process_vector
        if process_vector is None:
            raise RuntimeError("process runtime gate did not create subprocesses")
        worker_pids = process_vector.worker_pids
        failure_cleanup = {"exercised": False, "passed": True}
        if not args.skip_failure_cleanup:
            failure_env = _make_env(args, worker_processes=args.worker_processes)
            failure_vector = failure_env._process_vector
            if failure_vector is None:
                raise RuntimeError("failure cleanup gate did not create subprocesses")
            failure_pids = failure_vector.worker_pids
            try:
                failure_vector.crash_worker_for_test()
            except RuntimeError as error:
                failure_cleanup = {
                    "exercised": True,
                    "passed": failure_vector.closed and not failure_vector.alive_pids,
                    "worker_pids": failure_pids,
                    "error": str(error).splitlines()[0],
                }
            else:
                raise RuntimeError("intentional worker crash did not fail closed")
            if not failure_cleanup["passed"]:
                raise RuntimeError(
                    "worker failure did not clean up every process shard"
                )

        result = {
            "schema_version": "rlinf-dynamic-benchmark-process-runtime-gate-v0.1",
            "task_id": args.task,
            "num_envs": args.num_envs,
            "worker_processes": args.worker_processes,
            "process_start_method": args.process_start_method,
            "worker_pids": worker_pids,
            "serial_reset_digest": serial_reset_digest,
            "process_reset_digest": process_reset_digest,
            "reset_digest_exact": serial_reset_digest == process_reset_digest,
            "serial_step_digests": serial_digests,
            "process_step_digests": process_digests,
            "serial_process_exact": serial_digests == process_digests,
            "checkpoint_sha256": checkpoint_sha256,
            "uninterrupted_step_digests": uninterrupted,
            "resumed_step_digests": resumed,
            "checkpoint_resume_exact": uninterrupted == resumed,
            "failure_cleanup": failure_cleanup,
            "all_gates_passed": True,
        }
    finally:
        serial.close()
        processes.close()
        if restored is not None:
            restored.close()
        if failure_env is not None:
            failure_env.close()

    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
