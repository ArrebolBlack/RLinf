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

"""Launch the measured J32W2 whole-host residual-RLPD throughput profile."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.embodiment.run_dynamic_benchmark_cpu_production import (
    _discover_numa_cpus,
)

SCHEMA_VERSION = "rlinf-dynamic-benchmark-residual-wave-v0.1"
JOB_COUNT = 32
GPU_COUNT = 8
JOBS_PER_GPU = 4
WORKERS_PER_JOB = 2


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-manifest", type=Path, required=True)
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/rld2_qa/cpu_residual_throughput_j32w2.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _full_commit(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"job manifest must use {SCHEMA_VERSION}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != JOB_COUNT:
        raise ValueError(f"J32W2 requires exactly {JOB_COUNT} jobs")
    outputs: set[Path] = set()
    cpu_inventory: set[int] = set()
    gpu_counts = dict.fromkeys(range(GPU_COUNT), 0)
    gpu_nodes = {index: set() for index in range(GPU_COUNT)}
    normalized = []
    for index, row in enumerate(jobs):
        if not isinstance(row, dict) or set(row) != {
            "task",
            "seed",
            "demo_seed",
            "demo_replay",
            "output",
            "gpu",
            "cpu_affinity",
            "numa_node",
        }:
            raise ValueError(f"job {index} field inventory is invalid")
        gpu = row["gpu"]
        cpus = row["cpu_affinity"]
        numa_node = row["numa_node"]
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu not in gpu_counts:
            raise ValueError(f"job {index} GPU index is invalid")
        if (
            not isinstance(cpus, list)
            or len(cpus) != WORKERS_PER_JOB
            or any(isinstance(cpu, bool) or not isinstance(cpu, int) for cpu in cpus)
            or len(set(cpus)) != WORKERS_PER_JOB
        ):
            raise ValueError(f"job {index} must have two unique logical CPUs")
        if isinstance(numa_node, bool) or not isinstance(numa_node, int):
            raise ValueError(f"job {index} NUMA node is invalid")
        overlap = cpu_inventory.intersection(cpus)
        if overlap:
            raise ValueError(f"CPU affinity overlaps across jobs: {sorted(overlap)}")
        output = Path(row["output"]).resolve()
        demo_replay = Path(row["demo_replay"]).resolve()
        if output in outputs:
            raise ValueError("job outputs must be unique")
        if not demo_replay.is_file():
            raise FileNotFoundError(demo_replay)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"job output is not empty: {output}")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise ValueError(f"job {index} seed is invalid")
        if isinstance(row["demo_seed"], bool) or not isinstance(row["demo_seed"], int):
            raise ValueError(f"job {index} demo_seed is invalid")
        gpu_counts[gpu] += 1
        gpu_nodes[gpu].add(numa_node)
        cpu_inventory.update(cpus)
        outputs.add(output)
        normalized.append(
            dict(row, output=output, demo_replay=demo_replay, cpu_affinity=cpus)
        )
    if set(gpu_counts.values()) != {JOBS_PER_GPU}:
        raise ValueError(f"J32W2 requires four jobs per GPU: {gpu_counts}")
    if any(len(nodes) != 1 for nodes in gpu_nodes.values()):
        raise ValueError(f"J32W2 requires each GPU to stay on one NUMA node: {gpu_nodes}")
    return normalized


def _validate_profile(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("J32W2 profile must be a YAML mapping")
    required = {
        "algorithm": "residual_rlpd",
        "num_envs": 2,
        "eval_num_envs": 2,
        "env_worker_processes": 2,
        "eval_worker_processes": 2,
        "updates_per_vector_step": 1,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in required.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"J32W2 profile topology drifted: {mismatches}")
    if payload.get("sampler_learner_overlap", False):
        raise ValueError("J32W2 profile must keep sampler/learner overlap off")


def _command(
    job: dict[str, Any],
    *,
    config: Path,
    rlinf_commit: str,
    benchmark_commit: str,
) -> list[str]:
    trainer = Path(__file__).resolve().parent / "train_dynamic_benchmark_expert.py"
    return [
        "numactl",
        f"--cpunodebind={job['numa_node']}",
        f"--membind={job['numa_node']}",
        "taskset",
        "-c",
        ",".join(str(cpu) for cpu in job["cpu_affinity"]),
        sys.executable,
        str(trainer),
        "--config",
        str(config),
        "--task",
        str(job["task"]),
        "--seed",
        str(job["seed"]),
        "--demo-seed",
        str(job["demo_seed"]),
        "--demo-replay-in",
        str(job["demo_replay"]),
        "--demo-rlinf-commit",
        rlinf_commit,
        "--rlinf-commit",
        rlinf_commit,
        "--benchmark-commit",
        benchmark_commit,
        "--output",
        str(job["output"]),
    ]


def _stop(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _wait_fail_fast(processes: list[subprocess.Popen[bytes]]) -> list[int]:
    """Poll the whole wave and cancel every live job on the first failure."""

    statuses: list[int | None] = [None] * len(processes)
    remaining = set(range(len(processes)))
    while remaining:
        for index in list(remaining):
            status = processes[index].poll()
            if status is None:
                continue
            statuses[index] = status
            remaining.remove(index)
            if status != 0:
                _stop(processes)
                raise RuntimeError(f"J32W2 job {index} failed: {status}")
        if remaining:
            time.sleep(0.1)
    return [int(status) for status in statuses]


def main() -> None:
    args = _parser().parse_args()
    if not sys.platform.startswith("linux"):
        raise RuntimeError("J32W2 launcher requires Linux process affinity")
    rlinf_commit = _full_commit(args.rlinf_commit, "rlinf_commit")
    benchmark_commit = _full_commit(args.benchmark_commit, "benchmark_commit")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    _validate_profile(args.config)
    for executable in ("numactl", "taskset"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"J32W2 launcher requires {executable}")
    jobs = _load_jobs(args.job_manifest)
    numa_cpus = _discover_numa_cpus()
    allowed = set(os.sched_getaffinity(0))
    for job in jobs:
        node = job["numa_node"]
        if node not in numa_cpus or not set(job["cpu_affinity"]).issubset(
            set(numa_cpus[node]).intersection(allowed)
        ):
            raise ValueError(
                f"job {job['task']}/seed{job['seed']} CPU affinity is outside NUMA node {node}"
            )
    commands = [
        _command(
            job,
            config=args.config.resolve(),
            rlinf_commit=rlinf_commit,
            benchmark_commit=benchmark_commit,
        )
        for job in jobs
    ]
    if args.dry_run:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "commands": commands}, indent=2))
        return

    processes: list[subprocess.Popen[bytes]] = []
    streams = []
    try:
        for job, command in zip(jobs, commands, strict=True):
            log = Path(str(job["output"]) + ".launch.log")
            stream = log.open("xb")
            streams.append(stream)
            env = dict(os.environ)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                    "MUJOCO_GL": "osmesa",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            )
        statuses = _wait_fail_fast(processes)
    except BaseException:
        _stop(processes)
        raise
    finally:
        for stream in streams:
            stream.close()
    print(json.dumps({"schema_version": SCHEMA_VERSION, "statuses": statuses}, indent=2))


if __name__ == "__main__":
    main()
