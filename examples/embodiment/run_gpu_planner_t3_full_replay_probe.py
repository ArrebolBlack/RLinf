# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run an admitted B=1 T3-full live-teacher/fresh-replay engineering probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

TASK_ID = "t3_full"
BACKEND_ID = "mjwarp_gpu_v1"
HORIZON_CONTROL_STEPS = 640
SCHEMA_VERSION = "gpuplan0-t3-full-b1-replay-probe-v2"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    enum_value = getattr(value, "value", value)
    if enum_value is not value:
        return _jsonable(enum_value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__
        }
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo_identity(root: Path, *, expected_commit: str, expected_tree: str) -> dict[str, str]:
    root = root.resolve(strict=True)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source repository is not clean: {root}: {status}")
    identity = {
        "path": str(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "show", "-s", "--format=%T", "HEAD"),
    }
    if identity["commit"] != expected_commit or identity["tree"] != expected_tree:
        raise RuntimeError(f"source identity mismatch: expected commit/tree, got {identity}")
    return identity


def _gitlink(root: Path, relative: str) -> str:
    rows = _git(root, "ls-tree", "HEAD", relative).splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"source gitlink is missing or ambiguous: {root}/{relative}")
    fields = rows[0].split()
    if len(fields) < 3:
        raise RuntimeError(f"malformed source gitlink row: {rows[0]!r}")
    return fields[2]


def _require_sha256(name: str, value: str) -> str:
    if len(value) != 64 or value.lower() != value:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value


def _validate_admission(args: argparse.Namespace) -> dict[str, Any]:
    if args.queue_job_id != args.runtime_ledger_job_id:
        raise RuntimeError("Queue ready job and runtime-ledger lease job IDs differ")
    expires_at = datetime.fromisoformat(args.runtime_ledger_lease_expires_at)
    if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc):
        raise RuntimeError("runtime-ledger exact lease is missing a future timezone-aware expiry")
    return {
        "queue_job_id": args.queue_job_id,
        "runtime_ledger_job_id": args.runtime_ledger_job_id,
        "resource_unit": args.resource_unit,
        "lease_expires_at": expires_at.isoformat(),
        "launch_token_sha256": _require_sha256(
            "runtime_ledger_launch_token_sha256",
            args.runtime_ledger_launch_token_sha256,
        ),
    }


def _gpu_identity(expected_uuid: str) -> dict[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    matches = [row for row in rows if expected_uuid in row]
    if len(matches) != 1:
        raise RuntimeError(f"exact leased GPU UUID is not uniquely visible: {rows}")
    return {"uuid": expected_uuid, "query_row": matches[0]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", required=True, type=Path)
    parser.add_argument("--candidate-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--se3-root", required=True, type=Path)
    parser.add_argument("--rlinf-root", required=True, type=Path)
    parser.add_argument("--dynamic-root", required=True, type=Path)
    parser.add_argument("--expected-se3-commit", required=True)
    parser.add_argument("--expected-se3-tree", required=True)
    parser.add_argument("--expected-rlinf-commit", required=True)
    parser.add_argument("--expected-rlinf-tree", required=True)
    parser.add_argument("--expected-dynamic-commit", required=True)
    parser.add_argument("--expected-dynamic-tree", required=True)
    parser.add_argument("--expected-mjwarp-gitlink", required=True)
    parser.add_argument("--expected-dynamic-rlinf-gitlink", required=True)
    parser.add_argument("--research-commit", required=True)
    parser.add_argument("--research-tree", required=True)
    parser.add_argument("--queue-job-id", required=True)
    parser.add_argument("--runtime-ledger-job-id", required=True)
    parser.add_argument("--runtime-ledger-lease-expires-at", required=True)
    parser.add_argument("--runtime-ledger-launch-token-sha256", required=True)
    parser.add_argument("--resource-unit", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    return parser


def _run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    admission = _validate_admission(args)

    se3_root = args.se3_root.resolve(strict=True)
    rlinf_root = args.rlinf_root.resolve(strict=True)
    dynamic_root = args.dynamic_root.resolve(strict=True)
    source = {
        "research": {"commit": args.research_commit, "tree": args.research_tree},
        "se3_wam": _repo_identity(
            se3_root,
            expected_commit=args.expected_se3_commit,
            expected_tree=args.expected_se3_tree,
        ),
        "rlinf": _repo_identity(
            rlinf_root,
            expected_commit=args.expected_rlinf_commit,
            expected_tree=args.expected_rlinf_tree,
        ),
        "dynamic_benchmark": _repo_identity(
            dynamic_root,
            expected_commit=args.expected_dynamic_commit,
            expected_tree=args.expected_dynamic_tree,
        ),
    }
    if _gitlink(se3_root, "third_party/mujoco_warp") != args.expected_mjwarp_gitlink:
        raise RuntimeError("MJWarp gitlink identity mismatch")
    if (
        _gitlink(dynamic_root, "third_party/RLinf")
        != args.expected_dynamic_rlinf_gitlink
    ):
        raise RuntimeError("Dynamic Benchmark RLinf gitlink identity mismatch")

    sys.path[:0] = [str(se3_root / "src"), str(rlinf_root)]

    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    from rlinf.envs.dynamic_benchmark.gpu_backend import GpuNativeBackendEnv
    from rlinf.envs.dynamic_benchmark.gpu_planner import (
        GpuCurrentStatePlanner,
        GpuPlannerReplayError,
    )
    from rlinf.envs.dynamic_benchmark.t3_full_export import (
        load_t3_full_export_row,
    )
    export = load_t3_full_export_row(
        args.export_manifest,
        candidate_index=args.candidate_index,
    )
    gpu = _gpu_identity(args.expected_gpu_uuid)

    common = {
        "task_id": TASK_ID,
        "num_envs": 1,
        "export_dir": str(export.export_dir),
        "device_ordinal": 0,
        "image_size": args.image_size,
        "observation_track": "state",
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "expected_se3_source_commit": args.expected_se3_commit,
        "expected_se3_source_tree": args.expected_se3_tree,
        "task_quality_schema_version": "db0-episode-task-quality-v1",
        "task_quality_evaluator_backend_id": BACKEND_ID,
        "require_exact_export_identity": True,
    }

    def planner_factory(task_id: str, request: Any) -> Any:
        return make_privileged_teacher(task_id, request=request, image_size=args.image_size)

    online = None
    replay_backend = None
    try:
        online = GpuNativeBackendEnv(**common)
        online.validate_frozen_request(export.request, exact_episode_id=True)
        planner = GpuCurrentStatePlanner(
            backend=online,
            task_id=TASK_ID,
            planner_factory=planner_factory,
            max_control_steps=HORIZON_CONTROL_STEPS,
            capture_replay_probe=True,
        )
        tape = planner.rollout(export.request)
        _write_json(output / "live_tape.json", tape.to_dict())
        online.close()
        online = None

        replay_backend = GpuNativeBackendEnv(**common)
        replay_backend.validate_frozen_request(export.request, exact_episode_id=True)
        replay_runner = GpuCurrentStatePlanner(
            backend=replay_backend,
            task_id=TASK_ID,
            planner_factory=planner_factory,
            max_control_steps=HORIZON_CONTROL_STEPS,
            capture_replay_probe=True,
        )
        try:
            replay = replay_runner.replay(tape, backend=replay_backend)
        except GpuPlannerReplayError as exc:
            _write_json(
                output / "replay_divergence.json",
                {
                    "error": str(exc),
                    "evidence": dict(exc.evidence),
                    "first_divergence_eliminated": False,
                    "replay_gate_relaxed": False,
                },
            )
            raise
        _write_json(output / "replay.json", replay)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "claim_scope": "B=1 engineering replay probe only; not baseline, D16, MC64, or held-out evidence",
            "d16_completed": 0,
            "task_id": TASK_ID,
            "candidate_index": export.candidate_index,
            "episode_id": export.request.episode_id,
            "manifest_seed": export.manifest_seed,
            "export_manifest": {
                "path": str(export.manifest_path),
                "file_sha256": export.manifest_file_sha256,
                "payload_sha256": export.payload_sha256,
            },
            "reset_identity": export.row,
            "admission": admission,
            "gpu": gpu,
            "source": source,
            "clock": {
                "physics_hz": 500,
                "controller_hz": 20,
                "sensor_hz": 20,
                "physics_steps_per_control": 25,
                "horizon_control_steps": HORIZON_CONTROL_STEPS,
                "preroll_s": 0.4,
            },
            "backend_id": BACKEND_ID,
            "observation_track": "state",
            "action_mode": "E7",
            "replay": replay,
        }
        _write_json(output / "receipt.json", receipt)
        return 0
    finally:
        if online is not None:
            online.close()
        if replay_backend is not None:
            replay_backend.close()


def main() -> int:
    args = _parser().parse_args()
    try:
        return _run(args)
    except BaseException as exc:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        failure = output / "failure.json"
        if not failure.exists():
            _write_json(
                failure,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "d16_completed": 0,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
