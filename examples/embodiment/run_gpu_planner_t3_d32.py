# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the frozen ``t3_phase`` D32 development batch on the CUDA backend.

Online success and task-quality receipts are authoritative for this
exploration round.  Fresh-backend action-tape replay is retained as an audit
field and never blocks the online result when it diverges.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from examples.embodiment.run_gpu_planner_t3_e0 import (
    BACKEND_ID,
    TASK_ID,
    TASK_QUALITY_SCHEMA_VERSION,
    _check_gpu,
    _encode_gif,
    _gitlink,
    _jsonable,
    _repo_identity,
    _sha256,
    _terminal_dict,
    _write_checksums,
    _write_json,
)

D32_SCHEMA_VERSION = "se3wam-gpu-planner-t3-d32-manifest-v1"
TOTAL_EPISODES = 32
NUM_ENVS = 8
COHORTS = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--resource-unit", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--runtime-ledger-job-id", required=True)
    parser.add_argument("--research-commit", required=True)
    parser.add_argument("--research-tree", required=True)
    parser.add_argument("--expected-se3-commit", required=True)
    parser.add_argument("--expected-se3-tree", required=True)
    parser.add_argument("--expected-rlinf-commit", required=True)
    parser.add_argument("--expected-rlinf-tree", required=True)
    parser.add_argument("--expected-dynamic-commit", required=True)
    parser.add_argument("--expected-dynamic-tree", required=True)
    parser.add_argument("--expected-mjwarp-gitlink", required=True)
    parser.add_argument("--expected-dynamic-rlinf-gitlink", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    return parser


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != D32_SCHEMA_VERSION:
        raise RuntimeError("D32 manifest schema identity mismatch")
    if manifest.get("task_id") != TASK_ID:
        raise RuntimeError("D32 manifest task identity mismatch")
    if manifest.get("total_episodes") != TOTAL_EPISODES:
        raise RuntimeError("D32 manifest must contain exactly 32 episodes")
    if manifest.get("num_envs") != NUM_ENVS or manifest.get("cohorts") != COHORTS:
        raise RuntimeError("D32 manifest must use four cohorts of eight GPU lanes")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != TOTAL_EPISODES:
        raise RuntimeError("D32 manifest episode list is not exactly 32 rows")
    identities = []
    for index, row in enumerate(episodes):
        if not isinstance(row, dict):
            raise RuntimeError(f"D32 manifest episode {index} is not an object")
        if row.get("index") != index:
            raise RuntimeError(f"D32 manifest episode index mismatch at {index}")
        if row.get("cohort") != index // NUM_ENVS or row.get("lane") != index % NUM_ENVS:
            raise RuntimeError(f"D32 manifest cohort/lane mismatch at {index}")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise RuntimeError(f"D32 manifest episode id missing at {index}")
        identities.append(episode_id)
    if len(set(identities)) != TOTAL_EPISODES:
        raise RuntimeError("D32 manifest episode identities are not unique")
    return manifest, _sha256(path)


def _source_identity(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve(strict=True)
    se3_root = source_root / "SE3-WAM"
    dynamic_root = source_root / "se3-wam-dynamic-benchmark"
    rlinf_root = dynamic_root / "third_party" / "RLinf"
    source = {
        "research": {"commit": args.research_commit, "tree": args.research_tree},
        "se3_wam": _repo_identity(se3_root),
        "rlinf": _repo_identity(rlinf_root),
        "dynamic_benchmark": _repo_identity(dynamic_root),
        "mjwarp_gitlink": _gitlink(se3_root, "third_party/mujoco_warp"),
        "dynamic_rlinf_gitlink": _gitlink(dynamic_root, "third_party/RLinf"),
    }
    expected = {
        "se3_wam": (args.expected_se3_commit, args.expected_se3_tree),
        "rlinf": (args.expected_rlinf_commit, args.expected_rlinf_tree),
        "dynamic_benchmark": (args.expected_dynamic_commit, args.expected_dynamic_tree),
    }
    for name, (commit, tree) in expected.items():
        if source[name]["commit"] != commit or source[name]["tree"] != tree:
            raise RuntimeError(f"{name} source identity mismatch: {source[name]}")
    if source["mjwarp_gitlink"] != args.expected_mjwarp_gitlink:
        raise RuntimeError("MJWarp gitlink identity mismatch")
    if source["dynamic_rlinf_gitlink"] != args.expected_dynamic_rlinf_gitlink:
        raise RuntimeError("Dynamic Benchmark RLinf gitlink identity mismatch")
    return source


def _quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "minimum_wire_clearance_m",
        "maximum_wire_impulse_n_s",
        "maximum_wire_penetration_m",
        "maximum_traversal_path_distance_m",
        "minimum_progress_backtrack",
    )
    summary: dict[str, Any] = {}
    for key in keys:
        values = [
            float(row["task_quality"][key])
            for row in rows
            if isinstance(row.get("task_quality"), dict) and key in row["task_quality"]
        ]
        summary[key] = None if not values else {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }
    return summary


def _replay_audit(
    *,
    tape: Any,
    common: dict[str, Any],
    planner_factory: Any,
    image_size: int,
) -> dict[str, Any]:
    replay_backend = None
    try:
        from rlinf.envs.dynamic_benchmark.gpu_backend import GpuNativeBackendEnv
        from rlinf.envs.dynamic_benchmark.gpu_planner import GpuCurrentStatePlanner

        replay_backend = GpuNativeBackendEnv(**{**common, "num_envs": 1})
        replay_planner = GpuCurrentStatePlanner(
            backend=replay_backend,
            task_id=TASK_ID,
            planner_factory=planner_factory,
            max_control_steps=420,
            evaluator_backend_id=BACKEND_ID,
            quality_schema_version=TASK_QUALITY_SCHEMA_VERSION,
        )
        replay = replay_planner.replay(tape, backend=replay_backend)
        return {"passed": True, "audit": _jsonable(replay)}
    except BaseException as exc:
        return {
            "passed": False,
            "audit": {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }
    finally:
        if replay_backend is not None:
            replay_backend.close()


def _run(args: argparse.Namespace) -> int:
    source_root = args.source_root.resolve(strict=True)
    export_dir = args.export_dir.resolve(strict=True)
    run_root = args.run_root.resolve()
    manifest, manifest_sha256 = _load_manifest(args.manifest)
    if not export_dir.is_dir():
        raise FileNotFoundError(f"frozen GPU artifact directory is missing: {export_dir}")
    bundle_dir = run_root / "review-bundle" / TASK_ID / args.bundle_id
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty bundle directory {bundle_dir}")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_bytes(args.manifest.resolve(strict=True).read_bytes())
    (bundle_dir / "episodes").mkdir()
    (bundle_dir / "media").mkdir()

    source = _source_identity(args)

    gpu = _check_gpu(args.expected_gpu_uuid)

    import sys

    se3_root = source_root / "SE3-WAM"
    dynamic_root = source_root / "se3-wam-dynamic-benchmark"
    rlinf_root = dynamic_root / "third_party" / "RLinf"
    sys.path[:0] = [str(se3_root / "src"), str(rlinf_root)]
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("D32 requires real CUDA; CPU fallback is forbidden")
    from se3_wam.benchmark.contracts import ObservationTrack
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    from rlinf.envs.dynamic_benchmark.gpu_backend import GpuNativeBackendEnv
    from rlinf.envs.dynamic_benchmark.gpu_planner import GpuCurrentStatePlanner

    common = {
        "task_id": TASK_ID,
        "num_envs": NUM_ENVS,
        "export_dir": str(export_dir),
        "device_ordinal": 0,
        "image_size": args.image_size,
        "observation_track": ObservationTrack.HYBRID,
        "task_quality_schema_version": TASK_QUALITY_SCHEMA_VERSION,
        "task_quality_evaluator_backend_id": BACKEND_ID,
    }

    def planner_factory(task_id: str, request: Any) -> Any:
        return make_privileged_teacher(task_id, request=request, image_size=args.image_size)

    wall_start = time.perf_counter()
    online = None
    episode_records: list[dict[str, Any]] = []
    try:
        online = GpuNativeBackendEnv(**common)
        actual_manifest_sha = getattr(online.provenance, "manifest_sha256", None)
        expected_manifest_sha = manifest.get("base_artifact_sha256")
        if expected_manifest_sha is not None and actual_manifest_sha != expected_manifest_sha:
            raise RuntimeError(
                f"D32 base artifact identity mismatch: expected {expected_manifest_sha}, "
                f"got {actual_manifest_sha}"
            )
        if manifest.get("base_request_sha256") is not None:
            actual_request_sha = getattr(online.provenance, "request_sha256", None)
            if actual_request_sha != manifest["base_request_sha256"]:
                raise RuntimeError(
                    f"D32 base request identity mismatch: expected {manifest['base_request_sha256']}, "
                    f"got {actual_request_sha}"
                )
        planner = GpuCurrentStatePlanner(
            backend=online,
            task_id=TASK_ID,
            planner_factory=planner_factory,
            max_control_steps=420,
            evaluator_backend_id=BACKEND_ID,
            quality_schema_version=TASK_QUALITY_SCHEMA_VERSION,
        )

        for cohort in range(COHORTS):
            cohort_rows = manifest["episodes"][cohort * NUM_ENVS : (cohort + 1) * NUM_ENVS]
            requests = tuple(
                replace(
                    online.next_request(),
                    episode_id=row["episode_id"],
                )
                for row in cohort_rows
            )
            tapes = planner.rollout_batch(requests)
            for row, tape in zip(cohort_rows, tapes, strict=True):
                terminal = _terminal_dict(tape.terminal_row)
                replay = _replay_audit(
                    tape=tape,
                    common=common,
                    planner_factory=planner_factory,
                    image_size=args.image_size,
                )
                episode_id = str(row["episode_id"])
                episode_dir = bundle_dir / "episodes" / episode_id
                media_dir = bundle_dir / "media" / episode_id
                episode_dir.mkdir()
                media_dir.mkdir()
                action_path = episode_dir / "action_tape.json"
                trajectory_path = episode_dir / "trajectory_tape.json"
                replay_path = episode_dir / "replay.json"
                _write_json(
                    action_path,
                    {
                        "action_tape_sha256": tape.action_tape_sha256,
                        "actions": _jsonable(tape.actions),
                    },
                )
                _write_json(trajectory_path, tape.to_dict())
                _write_json(replay_path, replay)
                scene_media = _encode_gif(
                    [observation.rgb["agentview"] for observation in tape.observations],
                    media_dir / "scene.gif",
                )
                wrist_media = _encode_gif(
                    [observation.rgb["robot0_eye_in_hand"] for observation in tape.observations],
                    media_dir / "wrist.gif",
                )
                scene_media["path"] = f"media/{episode_id}/scene.gif"
                wrist_media["path"] = f"media/{episode_id}/wrist.gif"
                episode_records.append(
                    {
                        "index": row["index"],
                        "cohort": row["cohort"],
                        "lane": row["lane"],
                        "episode_id": episode_id,
                        "terminal": terminal,
                        "result_sha256": _sha256(terminal),
                        "action_tape": {
                            "path": str(action_path.relative_to(bundle_dir)),
                            "sha256": _sha256(action_path),
                        },
                        "trajectory_tape": {
                            "path": str(trajectory_path.relative_to(bundle_dir)),
                            "sha256": _sha256(trajectory_path),
                        },
                        "replay": {
                            "path": str(replay_path.relative_to(bundle_dir)),
                            "sha256": _sha256(replay_path),
                            "passed": bool(replay["passed"]),
                        },
                        "scene": scene_media,
                        "wrist": wrist_media,
                    }
                )
            _write_json(
                run_root / "progress.json",
                {
                    "phase": "d32",
                    "completed": len(episode_records),
                    "total": TOTAL_EPISODES,
                    "success": sum(bool(item["terminal"]["success"]) for item in episode_records),
                    "drop": sum(item["terminal"]["termination_reason"] == "drop" for item in episode_records),
                    "replay_audit_failed": sum(not item["replay"]["passed"] for item in episode_records),
                    "cohorts_completed": cohort + 1,
                },
            )

        online_provenance = _jsonable(online.provenance)
        online.close()
        online = None

        terminal_rows = [item["terminal"] for item in episode_records]
        termination_counts = dict(sorted(Counter(row["termination_reason"] for row in terminal_rows).items()))
        success_count = sum(bool(row["success"]) for row in terminal_rows)
        drop_count = sum(row["termination_reason"] == "drop" for row in terminal_rows)
        replay_failed = sum(not item["replay"]["passed"] for item in episode_records)
        aggregate = {
            "phase": "development_d32",
            "completed": len(episode_records),
            "total": TOTAL_EPISODES,
            "success_count": success_count,
            "success_rate": success_count / TOTAL_EPISODES,
            "drop_count": drop_count,
            "drop_rate": drop_count / TOTAL_EPISODES,
            "failure_count": TOTAL_EPISODES - success_count,
            "termination_reason_counts": termination_counts,
            "replay_audit": {
                "passed_count": TOTAL_EPISODES - replay_failed,
                "failed_count": replay_failed,
                "blocking": False,
            },
            "task_quality_schema": TASK_QUALITY_SCHEMA_VERSION,
            "task_quality_summary": _quality_summary(terminal_rows),
            "wall_seconds": time.perf_counter() - wall_start,
            "episodes_per_second": TOTAL_EPISODES / max(time.perf_counter() - wall_start, 1e-12),
            "rendered_frame_count": sum(item["scene"]["frame_count"] + item["wrist"]["frame_count"] for item in episode_records),
            "machine_decision": "not_evaluable",
        }
        _write_json(bundle_dir / "aggregate.json", aggregate)
        (bundle_dir / "import_instructions.md").write_text(
            "# GPUPLAN0 t3_phase D32 import\n\n"
            "Online success/drop/quality are the development metrics. Fresh replay differences are retained "
            "as audit evidence and are non-blocking for this exploration round. Verify the frozen manifest, "
            "source/runtime identity, SHA256SUMS, and released ledger lease before serial import.\n",
            encoding="utf-8",
        )
        checksum_path = _write_checksums(bundle_dir)
        bundle = {
            "schema_version": "se3wam-gpu-planner-review-bundle-v1",
            "bundle_status": "complete",
            "bundle_id": args.bundle_id,
            "task_id": TASK_ID,
            "route": {
                "environment": "gpu",
                "policy_family": "planner",
                "backend_id": BACKEND_ID,
                "physics_device": "cuda",
                "render_device": "cuda",
                "planner_compute_device": "cpu",
            },
            "claim_scope": "D32 development: frozen current candidate, 32 independent episodes in four 8-lane CUDA cohorts; online current-observation CPU Planner metrics are primary, replay is non-blocking audit, no formal or held-out claim.",
            "source": source,
            "runtime": {
                "real_cuda": True,
                "cpu_planner_compute_allowed": True,
                "cpu_environment_or_physics_fallback": False,
                "current_observation_closed_loop": True,
                "frozen_action_replay_used_as_planner": False,
                "gpu_rendering": True,
                "physical_gpu_uuid": args.expected_gpu_uuid,
                "cuda_version": str(torch.version.cuda),
                "driver_version": gpu["driver_version"],
                "gpu_query_row": gpu["query_row"],
                "resource_unit": args.resource_unit,
                "runtime_ledger_job_id": args.runtime_ledger_job_id,
                "container_name": args.container_name,
                "source_root": str(source_root),
                "export_dir": str(export_dir),
                "online_provenance": online_provenance,
                "clock": {
                    "physics_hz": 500,
                    "controller_hz": 20,
                    "sensor_hz": 20,
                    "physics_steps_per_control": 25,
                    "horizon_control_steps": 420,
                },
            },
            "evaluation": {
                "phase": "development_d32",
                "split": "registered_t3_phase_surface",
                "expected_episode_count": TOTAL_EPISODES,
                "completed_episode_count": len(episode_records),
                "resampled": False,
                "manifest_sha256": manifest_sha256,
                "machine_decision": "not_evaluable",
                "machine_decision_reasons": ["development D32; no promotion gate"],
                "aggregate": aggregate,
            },
            "manifest": {
                "path": "manifest.json",
                "sha256": _sha256(bundle_dir / "manifest.json"),
                "candidate_id": manifest["candidate_id"],
                "base_artifact_sha256": manifest.get("base_artifact_sha256"),
                "base_request_sha256": manifest.get("base_request_sha256"),
            },
            "episodes": episode_records,
            "owner": {
                "status": "pending_owner_review",
                "promotion_authorized": False,
                "training_data_authorized": False,
                "policy_replacement_authorized": False,
            },
            "checksums": {"path": "SHA256SUMS", "sha256": _sha256(checksum_path)},
            "import_instructions": {
                "path": "import_instructions.md",
                "sha256": _sha256(bundle_dir / "import_instructions.md"),
            },
        }
        _write_json(bundle_dir / "bundle.json", bundle)
        print(
            json.dumps(
                {
                    "bundle_dir": str(bundle_dir),
                    "bundle_json": str(bundle_dir / "bundle.json"),
                    "completed": len(episode_records),
                    "total": TOTAL_EPISODES,
                    "success": success_count,
                    "drop": drop_count,
                    "termination_reason_counts": termination_counts,
                    "replay_audit_failed": replay_failed,
                    "machine_decision": "not_evaluable",
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        if online is not None:
            online.close()


def main() -> int:
    args = _parser().parse_args()
    try:
        return _run(args)
    except BaseException as exc:
        args.run_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.run_root / "failure.json",
            {
                "schema_version": "se3wam-gpu-planner-d32-failure-v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
