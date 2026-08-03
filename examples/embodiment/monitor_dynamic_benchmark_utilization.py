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

"""Supervise one Dynamic Benchmark utilization run and collect 1 Hz telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--outer-launcher", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--heartbeat-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--termination-grace-seconds", type=float, default=120.0)
    parser.add_argument("trainer_command", nargs=argparse.REMAINDER)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def _gpu_sample(index: int, expected_uuid: str) -> dict[str, Any]:
    fields = (
        "timestamp,index,uuid,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            "-i",
            str(index),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != 8:
        raise RuntimeError(f"unexpected nvidia-smi row: {completed.stdout!r}")
    if values[2] != expected_uuid:
        raise RuntimeError(
            f"GPU UUID drift: index={index}, expected={expected_uuid}, actual={values[2]}"
        )
    return {
        "sample_time_unix_s": time.time(),
        "nvidia_timestamp": values[0],
        "gpu_index": int(values[1]),
        "gpu_uuid": values[2],
        "gpu_utilization_percent": float(values[3]),
        "gpu_memory_used_mib": float(values[4]),
        "gpu_memory_total_mib": float(values[5]),
        "gpu_temperature_c": float(values[6]),
        "gpu_power_w": float(values[7]),
    }


def _proc_stat(pid: int) -> tuple[int, int, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split()
    return int(fields[1]), int(fields[11]), int(fields[12])


def _process_tree(root_pid: int) -> list[int]:
    parent_by_pid = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        stat = _proc_stat(int(entry.name))
        if stat is not None:
            parent_by_pid[int(entry.name)] = stat[0]
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parent_by_pid.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sorted(pid for pid in selected if pid in parent_by_pid)


def _status_value(pid: int, key: str) -> str | None:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def _cpu_sample(
    root_pid: int,
    previous_ticks: int | None,
    previous_time: float | None,
) -> tuple[dict[str, Any], int, float]:
    now = time.monotonic()
    pids = _process_tree(root_pid)
    ticks = 0
    threads = 0
    rss_kib = 0
    for pid in pids:
        stat = _proc_stat(pid)
        if stat is not None:
            ticks += stat[1] + stat[2]
        thread_value = _status_value(pid, "Threads")
        rss_value = _status_value(pid, "VmRSS")
        threads += int(thread_value) if thread_value is not None else 0
        rss_kib += int(rss_value.split()[0]) if rss_value is not None else 0
    cpu_percent = None
    if previous_ticks is not None and previous_time is not None and now > previous_time:
        clock_ticks = os.sysconf("SC_CLK_TCK")
        cpu_percent = max(
            0.0,
            100.0 * (ticks - previous_ticks) / clock_ticks / (now - previous_time),
        )
    payload = {
        "sample_time_unix_s": time.time(),
        "root_pid": root_pid,
        "process_count": len(pids),
        "pids": pids,
        "cpu_percent": cpu_percent,
        "threads": threads,
        "rss_mib": rss_kib / 1024.0,
        "cpu_affinity": _status_value(root_pid, "Cpus_allowed_list"),
        "numa_mem_nodes": _status_value(root_pid, "Mems_allowed_list"),
    }
    return payload, ticks, now


def _gpu_fd_pids(index: int) -> list[int]:
    target = f"/dev/nvidia{index}"
    owners = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fd_root = entry / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                if os.readlink(descriptor) == target:
                    owners.append(int(entry.name))
                    break
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
    return sorted(owners)


def _start_nvidia_monitor(command: list[str], path: Path) -> tuple[subprocess.Popen[str], Any]:
    stream = path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, stream


def _stop_monitor(process: subprocess.Popen[str], stream: Any) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    stream.close()


def _trainer_output(command: list[str]) -> Path:
    try:
        index = command.index("--output")
        return Path(command[index + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise ValueError("trainer command must contain --output PATH") from exc


def _artifact_hashes(trainer_root: Path) -> dict[str, str]:
    names = (
        "config.json",
        "summary.json",
        "best_policy.pt",
        "final_policy.pt",
        "checkpoint_latest.pt",
        "demo_replay.pt",
        "heartbeat.json",
        "metrics.jsonl",
    )
    return {
        name: _sha256(trainer_root / name)
        for name in names
        if (trainer_root / name).is_file()
    }


def _log_anomalies(paths: list[Path]) -> list[str]:
    patterns = {
        "oom": ("out of memory", "cuda out of memory", "std::bad_alloc"),
        "io": ("input/output error", "no space left on device", "disk quota exceeded"),
        "worker_crash": ("brokenprocesspool", "worker exited", "segmentation fault"),
    }
    found = set()
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for name, needles in patterns.items():
            if any(needle in text for needle in needles):
                found.add(name)
    return sorted(found)


def _combined_anomalies(paths: list[Path], watchdog_triggered: bool) -> list[str]:
    """Combine log classifications with supervisor-detected failures."""
    found = set(_log_anomalies(paths))
    if watchdog_triggered:
        found.add("hang")
    return sorted(found)


def main() -> None:
    args = _parser().parse_args()
    command = list(args.trainer_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("trainer command is required after --")
    if (
        args.gpu_index < 0
        or args.sample_seconds <= 0.0
        or args.heartbeat_timeout_seconds <= 0.0
        or args.max_wall_seconds <= 0.0
        or args.termination_grace_seconds <= 0.0
    ):
        raise ValueError("GPU index, sample interval, heartbeat, or wall limit is invalid")
    if not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("gpu_uuid must be an NVML UUID")
    if args.run_root.exists() and any(args.run_root.iterdir()):
        raise FileExistsError(f"refusing non-empty run root {args.run_root}")
    if not args.outer_launcher.is_file():
        raise FileNotFoundError(args.outer_launcher)
    trainer_root = _trainer_output(command)
    expected_trainer_root = (args.run_root / "trainer").resolve()
    if trainer_root != expected_trainer_root:
        raise ValueError(
            f"trainer --output must be {expected_trainer_root}, got {trainer_root}"
        )
    args.run_root.mkdir(parents=True, exist_ok=True)
    monitor_root = args.run_root / "monitor"
    monitor_root.mkdir()
    gpu_path = monitor_root / "gpu_1hz.jsonl"
    cpu_path = monitor_root / "process_1hz.jsonl"
    supervisor_path = Path(__file__).resolve()
    metadata = {
        "schema_version": "rlinf-dynamic-benchmark-utilization-run-v0.1",
        "run_root": str(args.run_root.resolve()),
        "trainer_root": str(trainer_root),
        "trainer_command": command,
        "supervisor_sha256": _sha256(supervisor_path),
        "outer_launcher": str(args.outer_launcher.resolve()),
        "outer_launcher_sha256": _sha256(args.outer_launcher),
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resource_id": os.environ.get("SE3WAM_RESOURCE_ID"),
        "lane_id": os.environ.get("SE3WAM_LANE_ID"),
        "expected_gpu_uuid_env": os.environ.get("SE3WAM_EXPECTED_GPU_UUID"),
        "cpu_set_env": os.environ.get("SE3WAM_CPU_SET"),
        "supervisor_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu_fd_pids_before": _gpu_fd_pids(args.gpu_index),
        "started_at_unix_s": time.time(),
        "heartbeat_timeout_seconds": args.heartbeat_timeout_seconds,
        "max_wall_seconds": args.max_wall_seconds,
        "termination_grace_seconds": args.termination_grace_seconds,
    }
    _gpu_sample(args.gpu_index, args.gpu_uuid)
    _atomic_json(args.run_root / "run_metadata.json", metadata)
    dmon, dmon_stream = _start_nvidia_monitor(
        [
            "nvidia-smi",
            "dmon",
            "-i",
            str(args.gpu_index),
            "-s",
            "pucvmet",
            "-d",
            "1",
            "-o",
            "DT",
        ],
        monitor_root / "nvidia_smi_dmon.log",
    )
    try:
        pmon, pmon_stream = _start_nvidia_monitor(
            [
                "nvidia-smi",
                "pmon",
                "-i",
                str(args.gpu_index),
                "-s",
                "um",
                "-d",
                "1",
            ],
            monitor_root / "nvidia_smi_pmon.log",
        )
    except BaseException:
        _stop_monitor(dmon, dmon_stream)
        raise
    try:
        stdout = (monitor_root / "trainer.stdout.log").open("w", encoding="utf-8")
        try:
            stderr = (monitor_root / "trainer.stderr.log").open(
                "w", encoding="utf-8"
            )
        except BaseException:
            stdout.close()
            raise
    except BaseException:
        _stop_monitor(dmon, dmon_stream)
        _stop_monitor(pmon, pmon_stream)
        raise
    try:
        trainer = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        _stop_monitor(dmon, dmon_stream)
        _stop_monitor(pmon, pmon_stream)
        stdout.close()
        stderr.close()
        raise

    gpu_samples: list[dict[str, Any]] = []
    cpu_samples: list[dict[str, Any]] = []
    sampling_errors: list[str] = []
    previous_ticks: int | None = None
    previous_time: float | None = None
    watchdog_triggered = False
    wall_time_limit_triggered = False
    termination_reason: str | None = None
    termination_requested_at: float | None = None
    termination_escalated = False

    def request_termination(reason: str, signum: int = signal.SIGTERM) -> None:
        nonlocal termination_reason, termination_requested_at
        if trainer.poll() is not None or termination_requested_at is not None:
            return
        termination_reason = reason
        termination_requested_at = time.monotonic()
        try:
            os.killpg(trainer.pid, signum)
        except ProcessLookupError:
            pass

    def forward_stop(signum: int, _frame: Any) -> None:
        request_termination(f"external_signal_{signum}", signum)

    signal.signal(signal.SIGTERM, forward_stop)
    signal.signal(signal.SIGINT, forward_stop)
    heartbeat_path = trainer_root / "heartbeat.json"
    try:
        while trainer.poll() is None:
            sample_started = time.monotonic()
            try:
                gpu = _gpu_sample(args.gpu_index, args.gpu_uuid)
                gpu_samples.append(gpu)
                _append_jsonl(gpu_path, gpu)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                sampling_errors.append(f"gpu: {type(exc).__name__}: {exc}")
            cpu, previous_ticks, previous_time = _cpu_sample(
                trainer.pid,
                previous_ticks,
                previous_time,
            )
            cpu_samples.append(cpu)
            _append_jsonl(cpu_path, cpu)
            heartbeat_age = (
                time.time() - heartbeat_path.stat().st_mtime
                if heartbeat_path.is_file()
                else time.time() - metadata["started_at_unix_s"]
            )
            if (
                heartbeat_age > args.heartbeat_timeout_seconds
                and not watchdog_triggered
            ):
                watchdog_triggered = True
                sampling_errors.append(
                    "hang: trainer heartbeat absent or stale for "
                    f"{heartbeat_age:.1f} seconds"
                )
                request_termination("heartbeat_timeout")
            wall_time_s = time.time() - metadata["started_at_unix_s"]
            if wall_time_s >= args.max_wall_seconds and not wall_time_limit_triggered:
                wall_time_limit_triggered = True
                request_termination("wall_time_limit")
            if (
                termination_requested_at is not None
                and not termination_escalated
                and time.monotonic() - termination_requested_at
                >= args.termination_grace_seconds
            ):
                termination_escalated = True
                sampling_errors.append(
                    "hang: trainer did not exit within termination grace; sent SIGKILL"
                )
                try:
                    os.killpg(trainer.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            remaining = args.sample_seconds - (time.monotonic() - sample_started)
            if remaining > 0.0:
                time.sleep(remaining)
        return_code = trainer.wait()
    finally:
        _stop_monitor(dmon, dmon_stream)
        _stop_monitor(pmon, pmon_stream)
        stdout.close()
        stderr.close()

    trainer_summary = None
    summary_path = trainer_root / "summary.json"
    if summary_path.is_file():
        trainer_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trainer_status = (
        trainer_summary.get("status") if isinstance(trainer_summary, dict) else None
    )
    if return_code == 0 and trainer_status == "complete" and not watchdog_triggered:
        status = "complete"
    elif (
        return_code == 0
        and trainer_status == "stopped"
        and wall_time_limit_triggered
        and not watchdog_triggered
    ):
        status = "time_limit"
    else:
        status = "failed"
    monitor_hashes = {
        path.name: _sha256(path)
        for path in monitor_root.iterdir()
        if path.is_file()
    }
    monitor_hashes["run_metadata.json"] = _sha256(
        args.run_root / "run_metadata.json"
    )
    gpu_utilization = [row["gpu_utilization_percent"] for row in gpu_samples]
    gpu_memory = [row["gpu_memory_used_mib"] for row in gpu_samples]
    cpu_utilization = [
        row["cpu_percent"] for row in cpu_samples if row["cpu_percent"] is not None
    ]
    result = {
        "schema_version": "rlinf-dynamic-benchmark-utilization-summary-v0.1",
        "status": status,
        "trainer_return_code": return_code,
        "trainer_status": trainer_status,
        "wall_time_limit_triggered": wall_time_limit_triggered,
        "termination_reason": termination_reason,
        "termination_escalated": termination_escalated,
        "wall_time_s": time.time() - metadata["started_at_unix_s"],
        "metadata": metadata,
        "gpu_utilization_percent": _distribution(gpu_utilization),
        "gpu_memory_used_mib": _distribution(gpu_memory),
        "cpu_percent": _distribution(cpu_utilization),
        "process_threads": _distribution(
            [float(row["threads"]) for row in cpu_samples]
        ),
        "process_rss_mib": _distribution(
            [float(row["rss_mib"]) for row in cpu_samples]
        ),
        "cpu_affinities": sorted(
            {row["cpu_affinity"] for row in cpu_samples if row["cpu_affinity"]}
        ),
        "numa_mem_nodes": sorted(
            {row["numa_mem_nodes"] for row in cpu_samples if row["numa_mem_nodes"]}
        ),
        "sampling_errors": sampling_errors,
        "anomalies": _combined_anomalies(
            [
                monitor_root / "trainer.stdout.log",
                monitor_root / "trainer.stderr.log",
                monitor_root / "nvidia_smi_dmon.log",
                monitor_root / "nvidia_smi_pmon.log",
            ],
            watchdog_triggered,
        ),
        "monitor_sha256": monitor_hashes,
        "nvidia_monitor_return_codes": {
            "dmon": dmon.returncode,
            "pmon": pmon.returncode,
        },
        "gpu_fd_pids_after": _gpu_fd_pids(args.gpu_index),
        "artifact_sha256": _artifact_hashes(trainer_root),
        "trainer_summary": trainer_summary,
    }
    rendered = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["payload_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
    _atomic_json(args.run_root / "utilization_summary.json", result)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
