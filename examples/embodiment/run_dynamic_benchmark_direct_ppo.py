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

"""Run the frozen GPUENV0 visual Direct-PPO and throughput protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-nvml-process-count", type=_positive_int, default=1)
    parser.add_argument("--se3-source", type=Path, required=True)
    parser.add_argument("--se3-commit", required=True)
    parser.add_argument("--se3-tree", required=True)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--rlinf-tree", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--expected-cpuset", required=True)
    parser.add_argument("--worker-ready", type=Path, required=True)
    parser.add_argument("--launch-release", type=Path, required=True)
    parser.add_argument("--cuda-ready", type=Path, required=True)
    parser.add_argument("--science-release", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--hold-timeout-s", type=float, default=3600.0)
    parser.add_argument("--explore-profile")
    parser.add_argument("--explore-num-envs", type=int)
    parser.add_argument("--explore-rollout-horizon", type=int)
    parser.add_argument("--explore-minibatch-size", type=int)
    parser.add_argument("--explore-ppo-epochs", type=int)
    parser.add_argument("--explore-encoder-batch-size", type=int)
    parser.add_argument("--explore-cohorts", type=int, default=3)
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _full_object(value: str, name: str) -> str:
    if len(value) != 40 or value.lower() != value or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a full lowercase Git object id")
    return value


def _parse_cpuset(value: str) -> set[int]:
    result = set()
    for part in value.split(","):
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            result.update(range(start, stop + 1))
        else:
            result.add(int(part))
    if not result or min(result) < 0:
        raise ValueError("expected cpuset is empty or invalid")
    return result


def _source_identity(root: Path, commit: str, tree: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    observed_commit = _git(root, "rev-parse", "HEAD")
    observed_tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1")
    if observed_commit != commit or observed_tree != tree or dirty:
        raise RuntimeError(
            f"dirty or mismatched source {root}: commit={observed_commit}, "
            f"tree={observed_tree}, dirty={bool(dirty)}"
        )
    return {
        "path": str(root),
        "commit": observed_commit,
        "tree": observed_tree,
        "tracked_worktree_clean": True,
    }


def _wait_for_release(path: Path, timeout_s: float, heartbeat: Path, phase: str) -> None:
    started = time.monotonic()
    while not path.exists():
        if time.monotonic() - started > timeout_s:
            raise TimeoutError(f"timed out waiting for {phase} release")
        _atomic_json(
            heartbeat,
            {
                "schema_version": "gpuenv0-direct-ppo-heartbeat-v1",
                "pid": os.getpid(),
                "phase": phase,
                "time_epoch_s": time.time(),
            },
        )
        time.sleep(0.25)


def _wait_for_expected_nvml_processes(
    *,
    query_pids: Callable[[], list[int]],
    expected_count: int,
    timeout_s: float,
    heartbeat: Path | None,
    poll_interval_s: float = 0.25,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int]:
    """Wait until the leased GPU exposes the complete expected process set."""
    if expected_count < 1:
        raise ValueError("expected NVML process count must be positive")
    started = monotonic()
    while True:
        pids = sorted(int(pid) for pid in query_pids())
        if any(pid < 1 for pid in pids) or len(set(pids)) != len(pids):
            raise RuntimeError("NVML returned an invalid compute PID set")
        if len(pids) > expected_count:
            raise RuntimeError(
                "leased GPU has more NVML compute processes than admitted: "
                f"expected={expected_count}, observed={len(pids)}"
            )
        if len(pids) == expected_count:
            return pids
        if monotonic() - started >= timeout_s:
            raise TimeoutError(
                "timed out waiting for all admitted NVML compute processes: "
                f"expected={expected_count}, observed={len(pids)}"
            )
        if heartbeat is not None:
            _atomic_json(
                heartbeat,
                {
                    "schema_version": "gpuenv0-direct-ppo-heartbeat-v1",
                    "pid": os.getpid(),
                    "phase": "cuda_admission",
                    "expected_nvml_process_count": expected_count,
                    "observed_nvml_pids": pids,
                    "time_epoch_s": time.time(),
                },
            )
        sleep(poll_interval_s)


def _pid_namespace_ids(path: Path = Path("/proc/self/status")) -> tuple[int, ...]:
    """Return Linux PID namespace IDs, or an empty tuple when unavailable."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ()
    for line in lines:
        if line.startswith("NSpid:"):
            try:
                return tuple(int(value) for value in line.split()[1:])
            except ValueError:
                return ()
    return ()


def _compatible_nvml_pid(
    nvml_pids: list[int], *, control_pid: int, namespace_pids: tuple[int, ...]
) -> tuple[int, str]:
    """Map this runner to one NVML PID while retaining the legacy scalar field."""
    if control_pid in nvml_pids:
        return control_pid, "direct_pid_match"
    if len(nvml_pids) == 1:
        return nvml_pids[0], "exclusive_uuid_transition_pid_namespace_hidden"
    matches = sorted(set(nvml_pids).intersection(namespace_pids))
    if len(matches) == 1:
        return matches[0], "pid_namespace_mapping"
    # Container PID namespaces may hide every host NVML PID.  The complete,
    # sorted ``nvml_pids`` set is the compound-job identity; retain its first
    # member only as a deterministic compatibility scalar for older reports.
    return nvml_pids[0], "compound_nvml_pid_set"


class ResourceMonitor:
    """Sample physical GPU and cpuset-wide CPU utilization for this job."""

    def __init__(self, *, gpu_uuid: str, cpus: set[int], heartbeat: Path) -> None:
        import psutil
        import pynvml

        self.psutil = psutil
        self.pynvml = pynvml
        pynvml.nvmlInit()
        self.handle = None
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            observed = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(observed, bytes):
                observed = observed.decode()
            if observed.lower() == gpu_uuid.lower():
                self.handle = handle
                break
        if self.handle is None:
            raise RuntimeError("NVML cannot find the leased physical GPU UUID")
        self.gpu_uuid = gpu_uuid
        self.cpus = tuple(sorted(cpus))
        self.heartbeat = heartbeat
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._previous_cpu = None

    def start(self) -> None:
        self._thread.start()

    def _cpu_utilization(self) -> float | None:
        current = self.psutil.cpu_times(percpu=True)
        selected = [current[index] for index in self.cpus]
        if self._previous_cpu is None:
            self._previous_cpu = selected
            return None
        total_delta = 0.0
        idle_delta = 0.0
        for previous, value in zip(self._previous_cpu, selected, strict=True):
            total_delta += sum(value) - sum(previous)
            idle_delta += (value.idle + getattr(value, "iowait", 0.0)) - (
                previous.idle + getattr(previous, "iowait", 0.0)
            )
        self._previous_cpu = selected
        if total_delta <= 0.0:
            return None
        return 100.0 * (total_delta - idle_delta) / total_delta

    def _run(self) -> None:
        while not self._stop.is_set():
            utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            sample = {
                "time_epoch_s": time.time(),
                "gpu_utilization_percent": float(utilization.gpu),
                "gpu_memory_used_bytes": int(memory.used),
                "cpuset_total_utilization_percent": self._cpu_utilization(),
            }
            self.samples.append(sample)
            _atomic_json(
                self.heartbeat,
                {
                    "schema_version": "gpuenv0-direct-ppo-heartbeat-v1",
                    "pid": os.getpid(),
                    "phase": "science",
                    "physical_gpu_uuid": self.gpu_uuid,
                    "sample": sample,
                },
            )
            self._stop.wait(0.5)

    def summary(
        self, *, start_epoch_s: float | None = None, end_epoch_s: float | None = None
    ) -> dict[str, Any]:
        selected = [
            row
            for row in self.samples
            if (start_epoch_s is None or row["time_epoch_s"] >= start_epoch_s)
            and (end_epoch_s is None or row["time_epoch_s"] <= end_epoch_s)
        ]

        def metrics(name: str) -> dict[str, float | None]:
            values = [float(row[name]) for row in selected if row[name] is not None]
            if not values:
                return {"median": None, "min": None, "max": None, "mean": None}
            return {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
            }

        return {
            "sample_count": len(selected),
            "physical_gpu_uuid": self.gpu_uuid,
            "cpuset": list(self.cpus),
            "cpuset_cpu_count": len(self.cpus),
            "cpu_scope_definition": "100 percent means every CPU in the job affinity is busy",
            "gpu_utilization_percent": metrics("gpu_utilization_percent"),
            "gpu_memory_used_bytes": metrics("gpu_memory_used_bytes"),
            "cpuset_total_utilization_percent": metrics(
                "cpuset_total_utilization_percent"
            ),
        }

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self.pynvml.nvmlShutdown()


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_model_state(path: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError(f"model checkpoint {path} lacks a model state")
    return payload["model"]


def _episode_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    episodes = report["train"]["episodes"]
    successes = [row for row in episodes if row["success"]]
    reasons = Counter(row["termination_reason"] for row in episodes)
    returns = [float(row["reward"]["return"]) for row in episodes]
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / max(1, len(episodes)),
        "termination_reasons": dict(sorted(reasons.items())),
        "mean_return": statistics.fmean(returns) if returns else None,
        "min_return": min(returns) if returns else None,
        "max_return": max(returns) if returns else None,
        "successful_task_quality": [row["task_quality"] for row in successes],
    }


def _run_simulator_only(
    *,
    contract: Mapping[str, Any],
    source: Any,
    num_envs: int,
    output: Path,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import GpuNativeTensorBackendEnv

    throughput = contract["throughput"]
    window_cohorts = int(throughput["window_cohorts"])
    warmup_cohorts = int(throughput["warmup_windows"]) * window_cohorts
    measured_cohorts = int(throughput["measurement_windows"]) * window_cohorts
    total_cohorts = warmup_cohorts + measured_cohorts
    manifest = contract["manifests"]["simulator_only"]
    evaluator = contract["evaluator"]
    env = GpuNativeTensorBackendEnv(
        task_id=str(contract["task_id"]),
        num_envs=num_envs,
        export_dir=source.export_dir,
        expected_gpu_uuid=source.expected_gpu_uuid,
        expected_se3_source_commit=source.se3_commit,
        expected_se3_source_tree=source.se3_tree,
        image_size=64,
        split=str(manifest["split"]),
        manifest_seed=int(manifest["seed"]),
        manifest_size=int(manifest["size"]),
        manifest_sha256=str(manifest["sha256"]),
        task_quality_schema_version=str(evaluator["task_quality_schema_version"]),
        task_quality_evaluator_backend_id=str(
            evaluator["task_quality_evaluator_backend_id"]
        ),
        observation_track="state",
    )
    cohort_rows = []
    episode_rows = 0
    action = torch.zeros((num_envs, 7), dtype=torch.float32, device=env.device)
    try:
        for cohort in range(total_cohorts):
            torch.cuda.synchronize(env.device)
            started_epoch_s = time.time()
            started = time.perf_counter()
            reset = env.reset()
            active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
            valid_steps = torch.zeros((), dtype=torch.int64, device=env.device)
            for _step in range(env.cohort_horizon_steps):
                valid_steps += active.sum()
                result = env.step(action)
                active &= ~result.done
            if bool(active.any()):
                raise RuntimeError("simulator-only cohort failed to terminate")
            health = env.materialize_health_audit()
            ledger = env.materialize_terminal_ledger_once(
                tuple(range(num_envs)), reset.episode_ids
            )
            if len(ledger) != num_envs or any(np.asarray(health["overflow"]).reshape(-1)):
                raise RuntimeError("simulator-only ledger or overflow audit failed")
            episode_rows += len(ledger)
            torch.cuda.synchronize(env.device)
            ended_epoch_s = time.time()
            wall_seconds = time.perf_counter() - started
            valid = int(valid_steps)
            cohort_rows.append(
                {
                    "cohort": cohort,
                    "started_at_epoch_s": started_epoch_s,
                    "ended_at_epoch_s": ended_epoch_s,
                    "valid_env_steps": valid,
                    "wall_seconds": wall_seconds,
                    "valid_env_steps_per_s": valid / wall_seconds,
                }
            )
        windows = []
        measured = cohort_rows[warmup_cohorts:]
        for index in range(int(throughput["measurement_windows"])):
            rows = measured[index * window_cohorts : (index + 1) * window_cohorts]
            valid = sum(row["valid_env_steps"] for row in rows)
            wall = sum(row["wall_seconds"] for row in rows)
            windows.append(
                {
                    "window": index,
                    "valid_env_steps": valid,
                    "wall_seconds": wall,
                    "valid_env_steps_per_s": valid / wall,
                    "resources": monitor.summary(
                        start_epoch_s=rows[0]["started_at_epoch_s"],
                        end_epoch_s=rows[-1]["ended_at_epoch_s"],
                    ),
                }
            )
        values = [row["valid_env_steps_per_s"] for row in windows]
        report = {
            "schema_version": "rlinf-gpuenv0-simulator-only-throughput-v1",
            "status": "passed",
            "num_envs": num_envs,
            "manifest_sha256": env.manifest_sha256,
            "backend_id": env.provenance.backend_id,
            "render_enabled": False,
            "policy_or_learner_included": False,
            "cpu_env_or_physics_fallback": False,
            "episode_ledger_rows": episode_rows,
            "cohorts": cohort_rows,
            "windows": windows,
            "simulator_only_valid_env_steps_per_s": {
                "median": statistics.median(values),
                "min": min(values),
            },
        }
        _atomic_json(output / "report.json", report)
        return report
    finally:
        env.close()


def _throughput_windows(
    report: Mapping[str, Any], contract: Mapping[str, Any], monitor: ResourceMonitor
) -> list[dict[str, Any]]:
    throughput = contract["throughput"]
    window_cohorts = int(throughput["window_cohorts"])
    warmup_cohorts = int(throughput["warmup_windows"]) * window_cohorts
    measured = report["train"]["cohorts"][warmup_cohorts:]
    windows = []
    for index in range(int(throughput["measurement_windows"])):
        rows = measured[index * window_cohorts : (index + 1) * window_cohorts]
        valid = sum(row["valid_env_steps"] for row in rows)
        frames = sum(row["rendered_frames"] for row in rows)
        updates = sum(row["optimizer_updates"] for row in rows)
        samples = sum(row["learner_samples"] for row in rows)
        wall = sum(row["wall_seconds"] for row in rows)
        windows.append(
            {
                "window": index,
                "valid_env_steps": valid,
                "rendered_frames": frames,
                "learner_samples": samples,
                "optimizer_updates": updates,
                "wall_seconds": wall,
                "render_enabled_frames_per_s": frames / wall,
                "render_enabled_valid_env_steps_per_s": valid / wall,
                "learner_samples_per_s": samples / wall,
                "optimizer_updates_per_s": updates / wall,
                "end_to_end_valid_env_steps_per_s": valid / wall,
                "resources": monitor.summary(
                    start_epoch_s=rows[0]["started_at_epoch_s"],
                    end_epoch_s=rows[-1]["ended_at_epoch_s"],
                ),
            }
        )
    return windows


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"fail-if-exists: output already exists: {args.output}")
    contract_path = args.contract.resolve(strict=True)
    if _file_sha256(contract_path) != args.contract_sha256:
        raise RuntimeError("frozen Direct PPO contract SHA-256 mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "gpuenv0-direct-ppo-throughput-contract-v1":
        raise RuntimeError("frozen Direct PPO contract schema mismatch")
    runtime_path = args.runtime_manifest.resolve(strict=True)
    if _file_sha256(runtime_path) != args.runtime_manifest_sha256:
        raise RuntimeError("runtime manifest SHA-256 mismatch")
    cpus = _parse_cpuset(args.expected_cpuset)
    if not hasattr(os, "sched_getaffinity") or set(os.sched_getaffinity(0)) != cpus:
        raise RuntimeError("job CPU affinity differs from the expected cpuset")
    source_payload = {
        "se3_wam": _source_identity(
            args.se3_source,
            _full_object(args.se3_commit, "se3_commit"),
            _full_object(args.se3_tree, "se3_tree"),
        ),
        "rlinf": _source_identity(
            args.rlinf_source,
            _full_object(args.rlinf_commit, "rlinf_commit"),
            _full_object(args.rlinf_tree, "rlinf_tree"),
        ),
    }
    _atomic_json(
        args.worker_ready,
        {
            "schema_version": "gpuenv0-direct-ppo-worker-ready-v1",
            "pid": os.getpid(),
            "contract_sha256": args.contract_sha256,
            "runtime_manifest_sha256": args.runtime_manifest_sha256,
            "sources": source_payload,
            "cpuset": sorted(cpus),
            "cuda_initialized": False,
            "expected_nvml_process_count": args.expected_nvml_process_count,
        },
    )
    _wait_for_release(
        args.launch_release, args.hold_timeout_s, args.heartbeat, "launch_hold"
    )

    import numpy as np
    import pynvml
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Direct PPO requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize(0)
    pynvml.nvmlInit()
    try:
        observed_uuid = None
        nvml_handle = None
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            value = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(value, bytes):
                value = value.decode()
            if value.lower() == args.expected_gpu_uuid.lower():
                observed_uuid = value
                nvml_handle = handle
                break
        if observed_uuid is None or nvml_handle is None:
            raise RuntimeError("NVML cannot find the leased physical GPU UUID")
        nvml_pids = _wait_for_expected_nvml_processes(
            query_pids=lambda: [
                int(process.pid)
                for process in pynvml.nvmlDeviceGetComputeRunningProcesses(nvml_handle)
            ],
            expected_count=args.expected_nvml_process_count,
            timeout_s=args.hold_timeout_s,
            heartbeat=args.heartbeat,
        )
        nvml_pid, pid_identity_mode = _compatible_nvml_pid(
            nvml_pids,
            control_pid=os.getpid(),
            namespace_pids=_pid_namespace_ids(),
        )
    finally:
        pynvml.nvmlShutdown()
    _atomic_json(
        args.cuda_ready,
        {
            "schema_version": "gpuenv0-direct-ppo-cuda-ready-v1",
            "control_pid": os.getpid(),
            "nvml_pid": nvml_pid,
            "nvml_pids": nvml_pids,
            "pid_identity_mode": pid_identity_mode,
            "physical_gpu_uuid": observed_uuid,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "expected_nvml_process_count": args.expected_nvml_process_count,
            "science_started": False,
        },
    )
    _wait_for_release(
        args.science_release, args.hold_timeout_s, args.heartbeat, "science_hold"
    )

    from rlinf.runners.dynamic_benchmark_direct_ppo_runner import (
        DirectPPORunConfig,
        DirectPPORunner,
        DirectPPOSourceIdentity,
    )

    import rlinf
    import se3_wam

    if not Path(rlinf.__file__).resolve().is_relative_to(args.rlinf_source.resolve()):
        raise RuntimeError("loaded RLinf module is outside the frozen source")
    if not Path(se3_wam.__file__).resolve().is_relative_to(args.se3_source.resolve()):
        raise RuntimeError("loaded SE3-WAM module is outside the frozen source")
    source = DirectPPOSourceIdentity(
        se3_commit=args.se3_commit,
        se3_tree=args.se3_tree,
        rlinf_commit=args.rlinf_commit,
        rlinf_tree=args.rlinf_tree,
        expected_gpu_uuid=args.expected_gpu_uuid,
        export_dir=str(args.export_dir.resolve(strict=True)),
    )
    args.output.mkdir(parents=True)
    monitor = ResourceMonitor(
        gpu_uuid=args.expected_gpu_uuid, cpus=cpus, heartbeat=args.heartbeat
    )
    monitor.start()
    try:
        if args.explore_profile is not None:
            overrides = (
                args.explore_num_envs,
                args.explore_rollout_horizon,
                args.explore_minibatch_size,
                args.explore_ppo_epochs,
                args.explore_encoder_batch_size,
                args.explore_cohorts,
            )
            if any(value is None or value < 1 for value in overrides):
                raise ValueError("exploratory profile requires positive PPO shape arguments")
            config = DirectPPORunConfig(
                name=args.explore_profile,
                seed=20261600 + int(args.explore_num_envs),
                num_envs=args.explore_num_envs,
                cohorts=args.explore_cohorts,
                rollout_horizon=args.explore_rollout_horizon,
                minibatch_size=args.explore_minibatch_size,
                ppo_epochs=args.explore_ppo_epochs,
                encoder_batch_size=args.explore_encoder_batch_size,
                manifest_name="train",
                checkpoint_every_cohorts=args.explore_cohorts,
            )
            _seed_all(config.seed)
            started = time.time()
            runner = DirectPPORunner(
                contract=contract,
                contract_path=contract_path,
                source=source,
                config=config,
                output=args.output / "profile",
                verify_ledger_double_consume=True,
            )
            try:
                report = runner.run()
            finally:
                runner.close()
            report["resources"] = monitor.summary(start_epoch_s=started)
            _atomic_json(args.output / "profile" / "report.json", report)
            measured = report["train"]["cohorts"][1:]
            if not measured:
                raise RuntimeError("exploratory profile produced no post-warmup cohort")
            rates = [row["end_to_end_valid_env_steps_per_s"] for row in measured]
            summary = {
                "schema_version": "gpuenv0-direct-ppo-exploratory-profile-v1",
                "status": "passed",
                "job_id": contract["job_id"],
                "profile": asdict(config),
                "contract_sha256": args.contract_sha256,
                "sources": source_payload,
                "control_pid": os.getpid(),
                "nvml_pid": nvml_pid,
                "nvml_pids": nvml_pids,
                "pid_identity_mode": pid_identity_mode,
                "physical_gpu_uuid": observed_uuid,
                "warmup_cohorts": 1,
                "measurement_cohorts": len(measured),
                "end_to_end_valid_env_steps_per_s": {
                    "median": statistics.median(rates),
                    "min": min(rates),
                },
                "render_enabled_frames_per_s": {
                    "median": statistics.median(
                        row["render_enabled_frames_per_s"] for row in measured
                    ),
                    "min": min(row["render_enabled_frames_per_s"] for row in measured),
                },
                "timing": report["timing"],
                "resources": report["resources"],
                "terminal_ledger": report["terminal_ledger"],
                "checkpoint": report["checkpoint"],
                "report": str(args.output / "profile" / "report.json"),
            }
            _atomic_json(args.output / "summary.json", summary)
            return 0
        summary: dict[str, Any] = {
            "schema_version": "gpuenv0-direct-ppo-throughput-full-report-v1",
            "status": "running",
            "job_id": contract["job_id"],
            "contract": {
                "path": str(contract_path),
                "sha256": args.contract_sha256,
            },
            "sources": source_payload,
            "runtime_manifest": {
                "path": str(runtime_path),
                "sha256": args.runtime_manifest_sha256,
            },
            "control_pid": os.getpid(),
            "nvml_pid": nvml_pid,
            "nvml_pids": nvml_pids,
            "pid_identity_mode": pid_identity_mode,
            "physical_gpu_uuid": observed_uuid,
        }
        _seed_all(int(contract["canary"]["num_envs"]) + 20261201)
        canary_config = DirectPPORunConfig(
            name="canary",
            seed=20261201,
            num_envs=int(contract["canary"]["num_envs"]),
            cohorts=int(contract["canary"]["cohorts"]),
            rollout_horizon=int(contract["canary"]["rollout_horizon"]),
            minibatch_size=int(contract["canary"]["minibatch_size"]),
            ppo_epochs=int(contract["canary"]["ppo_epochs"]),
            encoder_batch_size=1,
            manifest_name="canary",
            checkpoint_every_cohorts=1,
        )
        canary_started = time.time()
        canary = DirectPPORunner(
            contract=contract,
            contract_path=contract_path,
            source=source,
            config=canary_config,
            output=args.output / "canary",
            verify_ledger_double_consume=True,
        )
        try:
            canary_report = canary.run()
        finally:
            canary.close()
        canary_report["resources"] = monitor.summary(start_epoch_s=canary_started)
        _atomic_json(args.output / "canary" / "report.json", canary_report)
        summary["canary"] = {
            "status": canary_report["status"],
            "report": str(args.output / "canary" / "report.json"),
        }
        _atomic_json(args.output / "summary.partial.json", summary)

        base_profile = next(
            profile
            for profile in contract["throughput"]["profiles"]
            if profile["name"] == "b4_h32_m128_e2_x4"
        )
        training_config = DirectPPORunConfig(
            name="frozen_train",
            seed=20261231,
            num_envs=int(base_profile["num_envs"]),
            cohorts=int(contract["training"]["cohorts"]),
            rollout_horizon=int(base_profile["rollout_horizon"]),
            minibatch_size=int(base_profile["minibatch_size"]),
            ppo_epochs=int(base_profile["ppo_epochs"]),
            encoder_batch_size=int(base_profile["encoder_batch_size"]),
            manifest_name="train",
            checkpoint_every_cohorts=int(
                contract["training"]["checkpoint_every_cohorts"]
            ),
        )
        _seed_all(training_config.seed)
        training_started = time.time()
        training = DirectPPORunner(
            contract=contract,
            contract_path=contract_path,
            source=source,
            config=training_config,
            output=args.output / "training",
        )
        try:
            training_report = training.run()
        finally:
            training.close()
        training_report["resources"] = monitor.summary(start_epoch_s=training_started)
        _atomic_json(args.output / "training" / "report.json", training_report)
        summary["training"] = {
            "status": training_report["status"],
            "report": str(args.output / "training" / "report.json"),
            "episodes": _episode_summary(training_report),
        }
        _atomic_json(args.output / "summary.partial.json", summary)

        evaluation_num_envs = 4
        evaluation_cohorts = math.ceil(
            int(contract["training"]["held_out_episodes"]) / evaluation_num_envs
        )
        initial_state = _load_model_state(args.output / "training" / "initial_policy.pt")
        final_state = _load_model_state(args.output / "training" / "checkpoint_latest.pt")
        evaluations = {}
        for offset, (name, mode, model_state) in enumerate(
            (
                ("random", "random", None),
                ("initial", "deterministic", initial_state),
                ("final", "deterministic", final_state),
            )
        ):
            config = DirectPPORunConfig(
                name=f"heldout_{name}",
                seed=20261310 + offset,
                num_envs=evaluation_num_envs,
                cohorts=evaluation_cohorts,
                rollout_horizon=32,
                minibatch_size=128,
                ppo_epochs=2,
                encoder_batch_size=4,
                manifest_name="validation",
                checkpoint_every_cohorts=evaluation_cohorts,
            )
            _seed_all(config.seed)
            evaluation = DirectPPORunner(
                contract=contract,
                contract_path=contract_path,
                source=source,
                config=config,
                output=args.output / "heldout" / name,
                model_state=model_state,
                policy_mode=mode,
                train=False,
            )
            evaluation_started = time.time()
            try:
                report = evaluation.run()
            finally:
                evaluation.close()
            report["resources"] = monitor.summary(start_epoch_s=evaluation_started)
            _atomic_json(args.output / "heldout" / name / "report.json", report)
            evaluations[name] = {
                "report": str(args.output / "heldout" / name / "report.json"),
                **_episode_summary(report),
            }
        summary["heldout"] = evaluations
        _atomic_json(args.output / "summary.partial.json", summary)

        simulator_reports = []
        for num_envs in sorted(
            {int(profile["num_envs"]) for profile in contract["throughput"]["profiles"]}
        ):
            simulator_reports.append(
                _run_simulator_only(
                    contract=contract,
                    source=source,
                    num_envs=num_envs,
                    output=args.output / "throughput" / "simulator_only" / f"b{num_envs}",
                    monitor=monitor,
                )
            )
        summary["simulator_only"] = [
            {
                "num_envs": report["num_envs"],
                **report["simulator_only_valid_env_steps_per_s"],
            }
            for report in simulator_reports
        ]
        _atomic_json(args.output / "summary.partial.json", summary)

        throughput = contract["throughput"]
        total_sweep_cohorts = (
            int(throughput["warmup_windows"])
            + int(throughput["measurement_windows"])
        ) * int(throughput["window_cohorts"])
        profile_reports = []
        for profile_index, profile in enumerate(throughput["profiles"]):
            config = DirectPPORunConfig(
                name=str(profile["name"]),
                seed=20261400 + profile_index,
                num_envs=int(profile["num_envs"]),
                cohorts=total_sweep_cohorts,
                rollout_horizon=int(profile["rollout_horizon"]),
                minibatch_size=int(profile["minibatch_size"]),
                ppo_epochs=int(profile["ppo_epochs"]),
                encoder_batch_size=int(profile["encoder_batch_size"]),
                manifest_name="train",
                checkpoint_every_cohorts=total_sweep_cohorts,
            )
            _seed_all(config.seed)
            runner = DirectPPORunner(
                contract=contract,
                contract_path=contract_path,
                source=source,
                config=config,
                output=args.output / "throughput" / "profiles" / config.name,
            )
            try:
                report = runner.run()
            finally:
                runner.close()
            windows = _throughput_windows(report, contract, monitor)
            values = [row["end_to_end_valid_env_steps_per_s"] for row in windows]
            median = statistics.median(values)
            minimum = min(values)
            stable = (
                len(windows) == int(throughput["measurement_windows"])
                and minimum >= 0.90 * median
                and report["terminal_ledger"]["rows"]
                == config.num_envs * config.cohorts
            )
            profile_summary = {
                "name": config.name,
                "config": asdict(config),
                "status": "stable" if stable else "unstable",
                "windows": windows,
                "median_end_to_end_valid_env_steps_per_s": median,
                "min_end_to_end_valid_env_steps_per_s": minimum,
                "report": str(
                    args.output / "throughput" / "profiles" / config.name / "report.json"
                ),
            }
            _atomic_json(
                args.output
                / "throughput"
                / "profiles"
                / config.name
                / "throughput_summary.json",
                profile_summary,
            )
            profile_reports.append(profile_summary)
            summary["throughput_profiles"] = profile_reports
            _atomic_json(args.output / "summary.partial.json", summary)
        stable_profiles = [row for row in profile_reports if row["status"] == "stable"]
        if not stable_profiles:
            raise RuntimeError("throughput sweep found no stable render-enabled PPO profile")
        best = max(
            stable_profiles,
            key=lambda row: row["median_end_to_end_valid_env_steps_per_s"],
        )
        best_profile = next(
            profile for profile in throughput["profiles"] if profile["name"] == best["name"]
        )
        sustained_config = DirectPPORunConfig(
            name=f"sustained_{best['name']}",
            seed=20261500,
            num_envs=int(best_profile["num_envs"]),
            cohorts=int(throughput["sustained_training_cohorts"]),
            rollout_horizon=int(best_profile["rollout_horizon"]),
            minibatch_size=int(best_profile["minibatch_size"]),
            ppo_epochs=int(best_profile["ppo_epochs"]),
            encoder_batch_size=int(best_profile["encoder_batch_size"]),
            manifest_name="train",
            checkpoint_every_cohorts=int(throughput["sustained_training_cohorts"]),
        )
        _seed_all(sustained_config.seed)
        sustained = DirectPPORunner(
            contract=contract,
            contract_path=contract_path,
            source=source,
            config=sustained_config,
            output=args.output / "throughput" / "sustained",
        )
        sustained_started = time.time()
        try:
            sustained_report = sustained.run()
        finally:
            sustained.close()
        sustained_report["resources"] = monitor.summary(start_epoch_s=sustained_started)
        _atomic_json(
            args.output / "throughput" / "sustained" / "report.json",
            sustained_report,
        )
        summary["maximum_stable_throughput"] = best
        summary["sustained_training"] = {
            "config": asdict(sustained_config),
            "status": sustained_report["status"],
            "report": str(args.output / "throughput" / "sustained" / "report.json"),
        }
        summary["resources_all"] = monitor.summary()
        summary["status"] = "passed"
        _atomic_json(args.output / "summary.json", summary)
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": "gpuenv0-direct-ppo-failure-v1",
            "status": "failed",
            "type": type(exc).__name__,
            "message": str(exc),
            "time_epoch_s": time.time(),
        }
        _atomic_json(args.output / "failure.json", failure)
        raise
    finally:
        monitor.stop()


if __name__ == "__main__":
    raise SystemExit(main())
