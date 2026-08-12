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

"""Run the canonical CPU-only Dynamic Benchmark trajectory production topology.

One production job is sixteen exact fresh-process shards (W16) pinned to one
NUMA node.  A campaign manifest automatically runs one W16 job per NUMA node,
which gives the measured W16+W16 whole-host topology on a two-socket host.
Every shard has external phase/execution sidecars and supports exact resume.
Datasets use accepted-prefix sealing by default; fixed-reset sealing remains an
explicit scientific-contract option. CUDA is hidden from every child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "rlinf-dynamic-benchmark-cpu-production-v0.1"
DEFAULT_WORKERS_PER_JOB = 16
DEFAULT_WORKER_VIRTUAL_MEMORY_GIB = 16


@dataclass(frozen=True)
class ProductionJob:
    """One independently sealed trajectory dataset."""

    job_id: str
    output: Path
    export_args: tuple[str, ...]
    numa_node: int | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--output", type=Path, help="single-job final dataset root")
    source.add_argument(
        "--campaign-manifest",
        type=Path,
        help="JSON manifest containing independent jobs for automatic W16+W16",
    )
    parser.add_argument("--workers-per-job", type=int, default=DEFAULT_WORKERS_PER_JOB)
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        help="defaults to one job per discovered NUMA node",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume existing sibling work roots; sealed outputs are never overwritten",
    )
    parser.add_argument(
        "--fixed-reset-workload",
        action="store_true",
        help=(
            "retain all max-resets and require exactly accepted-episodes winners; "
            "default production keeps the exact accepted prefix"
        ),
    )
    parser.add_argument(
        "--worker-virtual-memory-gib",
        type=int,
        default=DEFAULT_WORKER_VIRTUAL_MEMORY_GIB,
    )
    parser.add_argument(
        "export_args",
        nargs=argparse.REMAINDER,
        help="single-job exporter arguments after -- (do not include --output/shard flags)",
    )
    return parser


def _payload_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_cpu_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _discover_numa_cpus(
    sysfs_root: Path = Path("/sys/devices/system"),
    affinity: set[int] | None = None,
) -> dict[int, list[int]]:
    """Return online CPUs visible to this process, grouped by NUMA node."""

    allowed = set(os.sched_getaffinity(0)) if affinity is None else set(affinity)
    result: dict[int, list[int]] = {}
    node_root = sysfs_root / "node"
    if node_root.is_dir():
        for path in sorted(node_root.glob("node[0-9]*")):
            cpulist = path / "cpulist"
            if not cpulist.is_file():
                continue
            node = int(path.name[4:])
            cpus = [cpu for cpu in _parse_cpu_list(cpulist.read_text()) if cpu in allowed]
            if cpus:
                result[node] = cpus
    if not result and allowed:
        result[0] = sorted(allowed)
    if not result:
        raise RuntimeError("no online CPUs are available to the production supervisor")
    return result


def _select_physical_first_cpus(
    cpus: list[int],
    count: int,
    sysfs_root: Path = Path("/sys/devices/system"),
) -> list[int]:
    """Select one logical CPU per physical core before using SMT siblings."""

    if count < 1 or len(cpus) < count:
        raise ValueError(f"NUMA scope has {len(cpus)} CPUs but W{count} was requested")
    primary: list[int] = []
    siblings: list[int] = []
    seen: set[tuple[str, str]] = set()
    for cpu in sorted(cpus):
        topology = sysfs_root / "cpu" / f"cpu{cpu}" / "topology"
        try:
            key = (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        except OSError:
            key = ("unknown", str(cpu))
        if key in seen:
            siblings.append(cpu)
        else:
            seen.add(key)
            primary.append(cpu)
    selected = (primary + siblings)[:count]
    if len(selected) != count:
        raise RuntimeError("failed to construct the requested CPU worker scope")
    return selected


def _argument_value(arguments: tuple[str, ...], name: str) -> str:
    matches = [index for index, value in enumerate(arguments) if value == name]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise ValueError(f"production export requires exactly one {name} VALUE")
    return arguments[matches[0] + 1]


def _normalized_export_args(arguments: tuple[str, ...]) -> tuple[str, ...]:
    args = tuple(value for value in arguments if value != "--")
    forbidden = {
        "--output",
        "--shard-count",
        "--shard-index",
        "--phase-profile-json",
        "--execution-receipt-json",
        "--resume",
    }
    present = sorted(forbidden.intersection(args))
    if present:
        raise ValueError(f"supervisor owns these exporter arguments: {present}")
    _argument_value(args, "--accepted-episodes")
    _argument_value(args, "--max-resets")
    if "--candidate-search-mode" not in args:
        args += ("--candidate-search-mode", "full-pool")
    if "--selection-mode" not in args:
        args += ("--selection-mode", "planner-pareto")
    return args


def _load_jobs(args: argparse.Namespace) -> list[ProductionJob]:
    if args.output is not None:
        return [
            ProductionJob(
                job_id=args.output.name,
                output=args.output.resolve(),
                export_args=_normalized_export_args(tuple(args.export_args)),
            )
        ]
    if args.export_args:
        raise ValueError("export arguments are only valid with --output")
    payload = json.loads(args.campaign_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"campaign manifest must use {SCHEMA_VERSION}")
    rows = payload.get("jobs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("campaign manifest jobs must be a non-empty list")
    jobs: list[ProductionJob] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - {
            "job_id",
            "output",
            "export_args",
            "numa_node",
        }:
            raise ValueError("campaign job has an invalid field inventory")
        job_id = row.get("job_id")
        output = row.get("output")
        export_args = row.get("export_args")
        numa_node = row.get("numa_node")
        if not isinstance(job_id, str) or not job_id or "/" in job_id or "\\" in job_id:
            raise ValueError("campaign job_id must be a non-empty path-free string")
        if not isinstance(output, str) or not isinstance(export_args, list) or not all(
            isinstance(value, str) for value in export_args
        ):
            raise ValueError(f"campaign job {job_id} has invalid output/export_args")
        if numa_node is not None and (isinstance(numa_node, bool) or not isinstance(numa_node, int)):
            raise ValueError(f"campaign job {job_id} has invalid numa_node")
        jobs.append(
            ProductionJob(
                job_id=job_id,
                output=Path(output).resolve(),
                export_args=_normalized_export_args(tuple(export_args)),
                numa_node=numa_node,
            )
        )
    if len({job.job_id for job in jobs}) != len(jobs) or len(
        {job.output for job in jobs}
    ) != len(jobs):
        raise ValueError("campaign job IDs and outputs must be unique")
    return jobs


def _child_preexec(cpu: int, virtual_memory_gib: int) -> Any:
    def configure() -> None:
        import resource

        os.sched_setaffinity(0, {cpu})
        limit = virtual_memory_gib * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return configure


def _run_logged(
    command: list[str],
    log: Path,
    *,
    cancel_event: threading.Event | None = None,
    **kwargs: Any,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            **kwargs,
        )
        try:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    _terminate_processes([process])
                    raise RuntimeError("campaign cancelled while sealing another job")
                time.sleep(0.1)
            return int(process.returncode)
        except BaseException:
            _terminate_processes([process])
            raise


def _merge_command(
    merger: Path,
    shards: Path,
    output: Path,
    accepted: str,
    *,
    fixed_reset_workload: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(merger),
        "--root",
        str(shards),
        "--output",
        str(output),
        "--accepted-episodes",
        accepted,
    ]
    if fixed_reset_workload:
        command.append("--require-max-resets")
    return command


def _run_job(
    job: ProductionJob,
    *,
    numa_node: int,
    cpus: list[int],
    workers: int,
    resume: bool,
    virtual_memory_gib: int,
    fixed_reset_workload: bool,
    cancel_event: threading.Event,
) -> dict[str, Any]:
    output = job.output
    work = output.parent / f".{output.name}.cpu-production"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed output {output}")
    if work.exists() and not resume:
        raise FileExistsError(f"work root exists; pass --resume after inspection: {work}")
    work.mkdir(parents=True, exist_ok=True)
    shards = work / "shards"
    shards.mkdir(exist_ok=True)
    attempt = work / "sidecars" / f"attempt-{time.time_ns()}"
    attempt.mkdir(parents=True)

    here = Path(__file__).resolve().parent
    exporter = here / "export_dynamic_benchmark_optimal_trajectories.py"
    merger = here / "merge_optimal_export_shards.py"
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "MUJOCO_GL": "osmesa",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    commands: list[list[str]] = []
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    try:
        for index, cpu in enumerate(cpus):
            if cancel_event.is_set():
                raise RuntimeError("campaign cancelled after another job failed")
            shard = shards / f"shard-{index:02d}"
            if (shard / "shard_complete.json").is_file():
                continue
            command = [
                "numactl",
                f"--membind={numa_node}",
                f"--cpunodebind={numa_node}",
                "taskset",
                "-c",
                str(cpu),
                sys.executable,
                str(exporter),
                *job.export_args,
                "--output",
                str(shards),
                "--shard-count",
                str(workers),
                "--shard-index",
                str(index),
                "--phase-profile-json",
                str(attempt / f"shard-{index:02d}-phases.json"),
                "--execution-receipt-json",
                str(attempt / f"shard-{index:02d}-execution.json"),
            ]
            if shard.exists():
                command.append("--resume")
            commands.append(command)
            log_path = attempt / f"shard-{index:02d}.log"
            stream = log_path.open("wb")
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=_child_preexec(cpu, virtual_memory_gib),
                start_new_session=True,
            )
            processes.append((index, process, stream))

        failures: list[tuple[int, int]] = []
        remaining = {index: process for index, process, _ in processes}
        while remaining:
            for index, process in list(remaining.items()):
                status = process.poll()
                if status is None:
                    continue
                remaining.pop(index)
                if status:
                    failures.append((index, status))
                    cancel_event.set()
            if failures or cancel_event.is_set():
                _terminate_processes([item[1] for item in processes])
                break
            time.sleep(0.1)
        if failures:
            raise RuntimeError(f"job {job.job_id} shard failures: {failures}")
        if cancel_event.is_set():
            raise RuntimeError("campaign cancelled after another job failed")
    except BaseException:
        cancel_event.set()
        _terminate_processes([item[1] for item in processes])
        raise
    finally:
        for _, _, stream in processes:
            stream.close()

    accepted = _argument_value(job.export_args, "--accepted-episodes")
    merge_command = _merge_command(
        merger,
        shards,
        output,
        accepted,
        fixed_reset_workload=fixed_reset_workload,
    )
    merge_status = _run_logged(
        merge_command,
        attempt / "merge.log",
        cancel_event=cancel_event,
        env=env,
    )
    if merge_status:
        cancel_event.set()
        raise RuntimeError(f"job {job.job_id} merge failed: {merge_status}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id,
        "output": str(output),
        "numa_node": numa_node,
        "worker_cpus": cpus,
        "workers": workers,
        "cpu_only": True,
        "merge_semantics": (
            "fixed_reset_workload" if fixed_reset_workload else "accepted_prefix"
        ),
        "export_args_sha256": _payload_sha256(list(job.export_args)),
        "commands": commands,
    }
    (attempt / "production_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    """Boundedly stop exact child process groups and leave no shard workers."""

    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30.0
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _run_node_queue(
    assignments: list[tuple[ProductionJob, int, list[int]]],
    **run_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run one NUMA node's jobs serially so worker scopes never overlap."""

    results = []
    for job, node, cpus in assignments:
        results.append(
            _run_job(
                job,
                numa_node=node,
                cpus=cpus,
                **run_kwargs,
            )
        )
    return results


def main() -> None:
    args = _parser().parse_args()
    if not sys.platform.startswith("linux"):
        raise RuntimeError("CPU production topology requires Linux affinity and NUMA")
    if args.workers_per_job != DEFAULT_WORKERS_PER_JOB:
        raise ValueError("canonical production topology is W16; overrides are not accepted")
    if args.worker_virtual_memory_gib < 1:
        raise ValueError("worker virtual-memory limit must be positive")
    for executable in ("numactl", "taskset"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"CPU production supervisor requires {executable}")
    jobs = _load_jobs(args)
    numa_cpus = _discover_numa_cpus()
    nodes = sorted(numa_cpus)
    max_concurrent = args.max_concurrent_jobs or len(nodes)
    if not 1 <= max_concurrent <= len(nodes):
        raise ValueError("max-concurrent-jobs must be within discovered NUMA capacity")

    assigned: list[tuple[ProductionJob, int, list[int]]] = []
    for index, job in enumerate(jobs):
        node = job.numa_node if job.numa_node is not None else nodes[index % max_concurrent]
        if node not in numa_cpus:
            raise ValueError(f"job {job.job_id} requests unavailable NUMA node {node}")
        cpus = _select_physical_first_cpus(numa_cpus[node], args.workers_per_job)
        assigned.append((job, node, cpus))

    queues = {
        node: [assignment for assignment in assigned if assignment[1] == node]
        for node in sorted({assignment[1] for assignment in assigned})
    }
    if len(queues) > max_concurrent:
        raise ValueError("explicit NUMA assignments exceed max-concurrent-jobs")
    results: list[dict[str, Any]] = []
    cancel_event = threading.Event()
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        pending = {
            executor.submit(
                _run_node_queue,
                queue,
                workers=args.workers_per_job,
                resume=args.resume,
                virtual_memory_gib=args.worker_virtual_memory_gib,
                fixed_reset_workload=args.fixed_reset_workload,
                cancel_event=cancel_event,
            ): node
            for node, queue in queues.items()
        }
        for future in as_completed(pending):
            results.extend(future.result())
    print(json.dumps({"schema_version": SCHEMA_VERSION, "jobs": results}, indent=2))


if __name__ == "__main__":
    main()
