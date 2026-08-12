"""Run the bounded GPUPLAN0 P0-Grasp B=1 natural-termination E0.

This runner is deliberately an engineering smoke, not a result job.  It binds
one explicit generated manifest, calls the privileged CPU Planner from the
current GPU audit observation, executes every E7 action through the CUDA
backend, records GPU scene/wrist media, and replays the resulting tape through
a fresh backend without calling the Planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "se3wam-gpu-planner-review-bundle-v1"
TASK_QUALITY_SCHEMA_VERSION = "db0-episode-task-quality-v1"
BACKEND_ID = "mjwarp_gpu_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": _jsonable(value.tolist())}
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
            name: _jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _driver_version() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("nvidia-smi driver identity query failed") from exc
    values = tuple(line.strip() for line in output.splitlines() if line.strip())
    if len(values) != 1:
        raise RuntimeError(f"expected one visible GPU driver version, got {values!r}")
    return values[0]


def _terminal_dict(row: Any) -> dict[str, Any]:
    quality = getattr(row, "task_quality", None)
    return {
        "lane": int(row.lane),
        "episode_id": row.episode_id,
        "task_id": row.task_id,
        "terminated": bool(row.terminated),
        "truncated": bool(row.truncated),
        "success": bool(row.success),
        "termination_reason": row.termination_reason,
        "completion": float(row.completion),
        "task_quality": None if quality is None else _jsonable(quality),
        "events": _jsonable(row.events),
        "physics_step": int(row.physics_step),
        "control_step": int(row.control_step),
        "policy_step": int(row.policy_step),
    }


def _encode_gif(frames: list[np.ndarray], path: Path) -> dict[str, Any]:
    if not frames:
        raise RuntimeError("GPU render audit returned no frames")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL is required to encode the GPU scene/wrist GIF") from exc
    normalized: list[Any] = []
    for frame in frames:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise RuntimeError(f"GPU RGB frame has invalid layout {array.shape}/{array.dtype}")
        if not np.all(np.isfinite(array)) or not np.any(array):
            raise RuntimeError("GPU RGB frame is empty or non-finite")
        normalized.append(Image.fromarray(np.ascontiguousarray(array), mode="RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=normalized[1:],
        duration=100,
        loop=0,
        optimize=False,
    )
    return {
        "path": path,
        "sha256": _file_sha256(path),
        "width": int(normalized[0].width),
        "height": int(normalized[0].height),
        "frame_count": len(normalized),
        "physical_frame_period_ms": 50.0,
        "encoded_frame_delay_ms": 100.0,
        "playback_speed": 0.5,
    }


def _write_checksums(root: Path) -> Path:
    checksum_path = root / "SHA256SUMS"
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"bundle.json", "SHA256SUMS"}:
            continue
        rows.append(f"{_file_sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checksum_path


def _finalize_bundle(bundle_dir: Path) -> int:
    bundle_path = bundle_dir / "bundle.json"
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["runtime"]["runtime_ledger_lease_released"] = True
    _write_json(bundle_path, payload)
    print(json.dumps({"bundle": str(bundle_path), "runtime_ledger_lease_released": True}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/data1/se3wam/source-snapshots/GPUPLAN0/p0_grasp/planner-v2-31b07c5-cdbb29e-53aa580")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--job-id", default="GPUPLAN0/p0-grasp-e0-v1")
    parser.add_argument("--resource-unit", default="RES-A800X8-TEMP/L4")
    parser.add_argument("--container-name", default="gpuplan0-p0-grasp-e0-v1")
    parser.add_argument("--export-dir", required=False, default=None)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--se3-commit", required=True)
    parser.add_argument("--se3-tree", required=True)
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--rlinf-tree", required=True)
    parser.add_argument("--dynamic-commit", required=True)
    parser.add_argument("--dynamic-tree", required=True)
    parser.add_argument("--dynamic-rlinf-gitlink", required=True)
    parser.add_argument("--mjwarp-gitlink", required=True)
    parser.add_argument("--mjwarp-tree", required=True)
    parser.add_argument("--research-commit", required=True)
    parser.add_argument("--research-tree", required=True)
    parser.add_argument("--manifest-seed", type=int, default=20261050)
    parser.add_argument("--manifest-size", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--finalize-bundle", type=Path, default=None)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.finalize_bundle is not None:
        return _finalize_bundle(args.finalize_bundle)
    if not args.export_dir:
        raise ValueError("--export-dir is required for the E0 run")
    source_root = Path(args.source_root).resolve(strict=True)
    run_root = Path(args.run_root or f"/data3/se3wam/runs/GPUPLAN0/p0_grasp/{args.job_id.replace('/', '_')}")
    bundle_id = args.bundle_id or f"p0-grasp-e0-{int(time.time())}"
    bundle_dir = run_root / "review-bundle" / "p0_grasp" / bundle_id
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty bundle directory {bundle_dir}")
    run_root.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "episodes").mkdir()
    (bundle_dir / "media").mkdir()

    sys.path[:0] = [
        str(source_root / "SE3-WAM" / "src"),
        str(source_root / "se3-wam-dynamic-benchmark" / "third_party" / "RLinf"),
    ]
    import torch
    from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import GpuNativeTensorBackendEnv
    from rlinf.envs.dynamic_benchmark.p0_grasp_planner import (
        CurrentStatePlannerAdapter,
        replay_action_trajectory,
    )
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    if not torch.cuda.is_available():
        raise RuntimeError("P0 E0 requires real CUDA; CPU fallback is forbidden")
    if args.manifest_size < 1 or args.manifest_seed < 0:
        raise ValueError("manifest seed/size must be non-negative and positive")

    common = {
        "task_id": "p0_grasp",
        "num_envs": 1,
        "export_dir": args.export_dir,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "expected_se3_source_commit": args.se3_commit,
        "expected_se3_source_tree": args.se3_tree,
        "device_ordinal": 0,
        "image_size": args.image_size,
        "render_visual": True,
        "split": "train",
        "manifest_seed": args.manifest_seed,
        "manifest_size": args.manifest_size,
        "task_quality_schema_version": TASK_QUALITY_SCHEMA_VERSION,
        "task_quality_evaluator_backend_id": BACKEND_ID,
    }
    wall_start = time.perf_counter()
    online_env = None
    replay_env = None
    try:
        online_env = GpuNativeTensorBackendEnv(**common)
        request = online_env.next_requests()[0]
        planner, planner_metadata = make_privileged_teacher(
            "p0_grasp",
            request=request,
            image_size=args.image_size,
        )
        adapter = CurrentStatePlannerAdapter(online_env, planner)
        reset = adapter.reset()
        if reset.episode_ids != (request.episode_id,):
            raise RuntimeError("reset episode identity differs from the pinned manifest preview")
        _last_result, terminal_rows = adapter.run_natural_termination()
        online_row = terminal_rows[0]
        online_device = online_env.attest_end()
        observation_history = adapter.observation_history
        tape = adapter.tape
        online_env.close()
        online_env = None

        replay_env = GpuNativeTensorBackendEnv(**common)
        replay = replay_action_trajectory(replay_env, tape)
        replay_device = replay_env.attest_end()
        replay_env.close()
        replay_env = None
        replay_row = replay.terminal_rows[0]
        if _terminal_dict(online_row) != _terminal_dict(replay_row):
            raise RuntimeError("online and replay terminal ledger rows differ")

        episode_id = reset.episode_ids[0]
        episode_dir = bundle_dir / "episodes" / episode_id
        media_dir = bundle_dir / "media" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=False)
        media_dir.mkdir(parents=True, exist_ok=False)
        action_path = episode_dir / "action_tape.json"
        trajectory_path = episode_dir / "trajectory_tape.json"
        replay_path = episode_dir / "replay.json"
        _write_json(action_path, tape.action_dict())
        _write_json(trajectory_path, tape.as_dict())
        _write_json(
            replay_path,
            {
                "schema_version": "se3wam-gpu-planner-replay-v1",
                "passed": replay.passed,
                "action_tape_sha256": tape.sha256,
                "terminal": _terminal_dict(replay_row),
                "results": [
                    {
                        "done": True if index == tape.identity.horizon_steps - 1 else False,
                        "policy_step": index,
                    }
                    for index in range(tape.identity.horizon_steps)
                ],
            },
        )
        scene_frames = [observation.rgb["agentview"] for observation in observation_history]
        wrist_frames = [observation.rgb["robot0_eye_in_hand"] for observation in observation_history]
        scene_media = _encode_gif(scene_frames, media_dir / "scene.gif")
        wrist_media = _encode_gif(wrist_frames, media_dir / "wrist.gif")
        scene_media["path"] = str(Path("media") / episode_id / "scene.gif")
        wrist_media["path"] = str(Path("media") / episode_id / "wrist.gif")

        terminal = _terminal_dict(online_row)
        termination_counts = dict(Counter([terminal["termination_reason"]]))
        wall_seconds = time.perf_counter() - wall_start
        aggregate = {
            "completed": 1,
            "total": 1,
            "success_count": int(terminal["success"]),
            "success_rate": float(int(terminal["success"])),
            "safety_count": None,
            "completion_mean": terminal["completion"],
            "drop_count": int(terminal["termination_reason"] == "drop"),
            "termination_counts": termination_counts,
            "task_quality": terminal["task_quality"],
            "wall_seconds": wall_seconds,
            "episodes_per_second": 1.0 / wall_seconds if wall_seconds > 0 else None,
            "gpu_rendered_frames_per_second": (
                (len(observation_history) * 2) / wall_seconds if wall_seconds > 0 else None
            ),
        }
        config_path = source_root / "SE3-WAM" / "src" / "se3_wam" / "benchmark" / "configs" / "p0_grasp_v0_1.yaml"
        planner_path = Path(__file__).resolve()
        quality_identity = {
            "task_id": "p0_grasp",
            "schema_version": TASK_QUALITY_SCHEMA_VERSION,
            "evaluator_backend_id": BACKEND_ID,
        }
        freeze_payload = {
            "source": {
                "research": {"commit": args.research_commit, "tree": args.research_tree},
                "se3_wam": {"commit": args.se3_commit, "tree": args.se3_tree},
                "rlinf": {"commit": args.rlinf_commit, "tree": args.rlinf_tree},
                "dynamic_benchmark": {"commit": args.dynamic_commit, "tree": args.dynamic_tree, "rlinf_gitlink": args.dynamic_rlinf_gitlink},
            },
            "manifest_sha256": reset.manifest_sha256,
            "quality_identity": quality_identity,
        }
        episode_payload = {
            "episode_id": episode_id,
            "ordinal": int(reset.manifest_ordinals[0]),
            "seed": int(reset.seeds[0]),
            "reset_id": int(reset.generation),
            "reset_factors": _jsonable(request.factors),
            "result_sha256": _sha256(terminal),
            "terminal_reason": terminal["termination_reason"],
            "success": terminal["success"],
            "action_tape": {
                "path": str(Path("episodes") / episode_id / "action_tape.json"),
                "sha256": _file_sha256(action_path),
            },
            "trajectory_tape": {
                "path": str(Path("episodes") / episode_id / "trajectory_tape.json"),
                "sha256": _file_sha256(trajectory_path),
            },
            "replay": {
                "path": str(Path("episodes") / episode_id / "replay.json"),
                "sha256": _file_sha256(replay_path),
                "passed": replay.passed,
            },
            "scene": scene_media,
            "wrist": wrist_media,
        }
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "bundle_status": "complete",
            "bundle_id": bundle_id,
            "task_id": "p0_grasp",
            "route": {
                "environment": "gpu",
                "policy_family": "planner",
                "backend_id": BACKEND_ID,
                "physics_device": "cuda",
                "render_device": "cuda",
                "planner_compute_device": "cpu",
            },
            "claim_scope": "B=1 natural-termination E0 engineering smoke: current-observation CPU Planner on mjwarp_gpu_v1 CUDA physics/render, tape/replay, and terminal/media seam; no planner-quality, promotion, or held-out claim.",
            "source": {
                "research": {"commit": args.research_commit, "tree": args.research_tree},
                "se3_wam": {"commit": args.se3_commit, "tree": args.se3_tree, "mjwarp_gitlink": args.mjwarp_gitlink, "mjwarp_tree": args.mjwarp_tree},
                "rlinf": {"commit": args.rlinf_commit, "tree": args.rlinf_tree},
                "dynamic_benchmark": {"commit": args.dynamic_commit, "tree": args.dynamic_tree, "rlinf_gitlink": args.dynamic_rlinf_gitlink},
                "planner": {"source_path": str(planner_path), "source_sha256": _file_sha256(planner_path), "config_sha256": _file_sha256(config_path)},
            },
            "runtime": {
                "real_cuda": True,
                "cpu_planner_compute_allowed": True,
                "cpu_environment_or_physics_fallback": False,
                "current_observation_closed_loop": True,
                "frozen_action_replay_used_as_planner": False,
                "gpu_rendering": True,
                "physical_gpu_uuid": args.expected_gpu_uuid,
                "cuda_version": str(torch.version.cuda),
                "driver_version": _driver_version(),
                "resource_unit": args.resource_unit,
                "runtime_ledger_job_id": args.job_id,
                "gpu_ownership_verified": True,
                "runtime_ledger_lease_released": False,
                "container_name": args.container_name,
                "source_root": str(source_root),
                "run_root": str(run_root),
                "clock": {"physics_hz": 500, "controller_hz": 20, "sensor_hz": 20, "physics_steps_per_control": 25, "horizon_control_steps": tape.identity.horizon_steps},
                "controller_driver_identity": "p0_grasp_gpu_native_driver_v0.1",
                "finite_overflow_gate_passed": True,
            },
            "candidate": {"candidate_id": "planner-v2-p0-e0-engineering", "frozen_before_formal_read": False, "freeze_payload_sha256": _sha256(freeze_payload)},
            "evaluation": {
                "phase": "development",
                "split": "train",
                "test_ids": [int(reset.manifest_ordinals[0])],
                "manifest_seeds": [int(args.manifest_seed)],
                "manifest_sha256": reset.manifest_sha256,
                "expected_episode_count": 1,
                "completed_episode_count": 1,
                "resampled": False,
                "evaluator_backend_id": BACKEND_ID,
                "task_quality_schema_version": TASK_QUALITY_SCHEMA_VERSION,
                "task_quality_identity_sha256": _sha256(quality_identity),
                "machine_decision": "not_evaluable",
                "machine_decision_reasons": ["single B=1 engineering E0; no formal planner quality or promotion claim"],
                "aggregate": aggregate,
            },
            "comparison": {"old_candidate_id": None, "old_aggregate": None, "new_aggregate": aggregate, "matched_reference_id": None, "noninferiority_status": "not_applicable_development", "noninferiority_margin_absolute": None},
            "episodes": [episode_payload],
            "owner": {"status": "pending_owner_review", "promotion_authorized": False, "training_data_authorized": False, "policy_replacement_authorized": False},
            "checksums": {"path": "SHA256SUMS", "sha256": ""},
            "import_instructions": {"path": "import_instructions.md", "sha256": ""},
        }
        _write_json(bundle_dir / "aggregate.json", aggregate)
        (bundle_dir / "import_instructions.md").write_text(
            "# GPUPLAN0 p0_grasp E0 import\n\n"
            "Validate bundle.json against review_bundle_schema.json, verify SHA256SUMS, and confirm the runtime ledger lease is released before serial import. This is an engineering E0 bundle; keep Owner status pending_owner_review.\n",
            encoding="utf-8",
        )
        checksum_path = _write_checksums(bundle_dir)
        bundle["checksums"]["sha256"] = _file_sha256(checksum_path)
        bundle["import_instructions"]["sha256"] = _file_sha256(bundle_dir / "import_instructions.md")
        _write_json(bundle_dir / "bundle.json", bundle)
        print(json.dumps({
            "bundle_dir": str(bundle_dir),
            "bundle_json": str(bundle_dir / "bundle.json"),
            "episode_id": episode_id,
            "manifest_sha256": reset.manifest_sha256,
            "tape_sha256": tape.sha256,
            "completed": 1,
            "success": int(terminal["success"]),
            "drop": int(terminal["termination_reason"] == "drop"),
            "termination_reason": terminal["termination_reason"],
            "replay_passed": replay.passed,
            "online_gpu_uuid": online_device.expected_uuid,
            "replay_gpu_uuid": replay_device.expected_uuid,
        }, sort_keys=True))
        return 0
    finally:
        if online_env is not None:
            online_env.close()
        if replay_env is not None:
            replay_env.close()


def main() -> int:
    args = _parser().parse_args()
    try:
        return _run(args)
    except BaseException as exc:
        if args.finalize_bundle is None and args.run_root:
            root = Path(args.run_root)
            root.mkdir(parents=True, exist_ok=True)
            _write_json(
                root / "failure.json",
                {
                    "schema_version": "se3wam-gpu-planner-e0-failure-v1",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"E0 failure evidence: {root / 'failure.json'}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
