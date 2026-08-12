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

"""Run the bounded GPUPLAN0 ``t3_phase`` B=1 natural-termination E0.

This is an engineering smoke only.  The Planner runs on the host from the
current observation, while every state transition, terminal receipt, and
rendered frame comes from the CUDA ``mjwarp_gpu_v1`` backend.  The action tape
is produced after the live rollout and is used only for an independent replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "se3wam-gpu-planner-review-bundle-v1"
TASK_ID = "t3_phase"
TASK_QUALITY_SCHEMA_VERSION = "db0-episode-task-quality-v1"
BACKEND_ID = "mjwarp_gpu_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _jsonable(value.tolist()),
        }
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
    digest = hashlib.sha256()
    if isinstance(value, Path):
        with value.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        digest.update(_canonical(value))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), sort_keys=True, indent=2, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo_identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    status = _git(path, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source repository is not clean: {path}: {status}")
    return {
        "path": str(path),
        "commit": _git(path, "rev-parse", "HEAD"),
        "tree": _git(path, "show", "-s", "--format=%T", "HEAD"),
    }


def _gitlink(path: Path, relative: str) -> str:
    rows = _git(path, "ls-tree", "HEAD", relative).splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"source gitlink is missing or ambiguous: {path}/{relative}")
    fields = rows[0].split()
    if len(fields) < 3:
        raise RuntimeError(f"source gitlink row is malformed: {rows[0]!r}")
    return fields[2]


def _driver_version() -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    values = tuple(line.strip() for line in output.splitlines() if line.strip())
    if len(values) != 1:
        raise RuntimeError(f"expected one visible GPU driver version, got {values!r}")
    return values[0]


def _check_gpu(expected_uuid: str) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or expected_uuid not in rows[0]:
        raise RuntimeError(f"physical GPU identity mismatch: expected {expected_uuid}, got {rows}")
    return {
        "query_row": rows[0],
        "uuid": expected_uuid,
        "driver_version": _driver_version(),
    }


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
    from PIL import Image

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
        "path": str(path),
        "sha256": _sha256(path),
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
        rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checksum_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--bundle-id", required=True)
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


def _run(args: argparse.Namespace) -> int:
    source_root = args.source_root.resolve(strict=True)
    export_dir = args.export_dir.resolve(strict=True)
    run_root = args.run_root.resolve()
    if not export_dir.is_dir():
        raise FileNotFoundError(f"frozen GPU artifact directory is missing: {export_dir}")
    bundle_dir = run_root / "review-bundle" / TASK_ID / args.bundle_id
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty bundle directory {bundle_dir}")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = bundle_dir / "episodes"
    media_dir = bundle_dir / "media"
    episode_dir.mkdir()
    media_dir.mkdir()

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

    gpu = _check_gpu(args.expected_gpu_uuid)

    import sys

    sys.path[:0] = [str(se3_root / "src"), str(rlinf_root)]
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("E0 requires real CUDA; CPU fallback is forbidden")
    from se3_wam.benchmark.contracts import ObservationTrack
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    from rlinf.envs.dynamic_benchmark.gpu_backend import GpuNativeBackendEnv
    from rlinf.envs.dynamic_benchmark.gpu_planner import GpuCurrentStatePlanner

    common = {
        "task_id": TASK_ID,
        "num_envs": 1,
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
    replay_backend = None
    try:
        online = GpuNativeBackendEnv(**common)
        request = online.next_request()
        planner = GpuCurrentStatePlanner(
            backend=online,
            task_id=TASK_ID,
            planner_factory=planner_factory,
            max_control_steps=420,
            evaluator_backend_id=BACKEND_ID,
            quality_schema_version=TASK_QUALITY_SCHEMA_VERSION,
        )
        tape = planner.rollout(request)
        online_provenance = _jsonable(tape.source_identity)
        online.close()
        online = None

        replay_backend = GpuNativeBackendEnv(**common)
        replay_planner = GpuCurrentStatePlanner(
            backend=replay_backend,
            task_id=TASK_ID,
            planner_factory=planner_factory,
            max_control_steps=420,
            evaluator_backend_id=BACKEND_ID,
            quality_schema_version=TASK_QUALITY_SCHEMA_VERSION,
        )
        replay = _jsonable(replay_planner.replay(tape, backend=replay_backend))
        replay_backend.close()
        replay_backend = None

        terminal = _terminal_dict(tape.terminal_row)
        episode_id = str(tape.request.episode_id)
        episode_path = episode_dir / episode_id
        media_path = media_dir / episode_id
        episode_path.mkdir()
        media_path.mkdir()
        action_path = episode_path / "action_tape.json"
        trajectory_path = episode_path / "trajectory_tape.json"
        replay_path = episode_path / "replay.json"
        _write_json(action_path, {"action_tape_sha256": tape.action_tape_sha256, "actions": _jsonable(tape.actions)})
        _write_json(trajectory_path, tape.to_dict())
        _write_json(replay_path, replay)
        scene_frames = [observation.rgb["agentview"] for observation in tape.observations]
        wrist_frames = [observation.rgb["robot0_eye_in_hand"] for observation in tape.observations]
        scene_media = _encode_gif(scene_frames, media_path / "scene.gif")
        wrist_media = _encode_gif(wrist_frames, media_path / "wrist.gif")
        scene_media["path"] = str(Path("media") / episode_id / "scene.gif")
        wrist_media["path"] = str(Path("media") / episode_id / "wrist.gif")

        aggregate = {
            "completed": 1,
            "total": 1,
            "success_count": int(terminal["success"]),
            "success_rate": float(int(terminal["success"])),
            "drop_count": int(terminal["termination_reason"] == "drop"),
            "termination_reason": terminal["termination_reason"],
            "task_quality": terminal["task_quality"],
            "wall_seconds": time.perf_counter() - wall_start,
            "machine_decision": "not_evaluable",
        }
        _write_json(bundle_dir / "aggregate.json", aggregate)
        (bundle_dir / "import_instructions.md").write_text(
            "# GPUPLAN0 t3_phase E0 import\n\n"
            "Verify SHA256SUMS, source/runtime identity, and the released ledger lease before serial import. "
            "This B=1 bundle is engineering evidence only; keep Owner status pending_owner_review.\n",
            encoding="utf-8",
        )
        checksum_path = _write_checksums(bundle_dir)
        bundle = {
            "schema_version": SCHEMA_VERSION,
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
            "claim_scope": "B=1 natural-termination E0 engineering smoke: current-observation CPU Planner on mjwarp_gpu_v1 CUDA physics/render, tape/replay, terminal/media seam; no planner-quality, promotion, or held-out claim.",
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
                "phase": "engineering_e0",
                "split": getattr(tape.request.split, "value", tape.request.split),
                "expected_episode_count": 1,
                "completed_episode_count": 1,
                "resampled": False,
                "manifest_sha256": getattr(tape.request, "manifest_sha256", None),
                "machine_decision": "not_evaluable",
                "machine_decision_reasons": ["single B=1 engineering smoke"],
                "aggregate": aggregate,
            },
            "episodes": [
                {
                    "episode_id": episode_id,
                    "result_sha256": _sha256(terminal),
                    "terminal": terminal,
                    "action_tape": {"path": str(action_path.relative_to(bundle_dir)), "sha256": _sha256(action_path)},
                    "trajectory_tape": {"path": str(trajectory_path.relative_to(bundle_dir)), "sha256": _sha256(trajectory_path)},
                    "replay": {"path": str(replay_path.relative_to(bundle_dir)), "sha256": _sha256(replay_path), "passed": bool(replay["passed"])},
                    "scene": scene_media,
                    "wrist": wrist_media,
                }
            ],
            "owner": {
                "status": "pending_owner_review",
                "promotion_authorized": False,
                "training_data_authorized": False,
                "policy_replacement_authorized": False,
            },
            "checksums": {"path": "SHA256SUMS", "sha256": _sha256(checksum_path)},
            "import_instructions": {"path": "import_instructions.md", "sha256": _sha256(bundle_dir / "import_instructions.md")},
        }
        _write_json(bundle_dir / "bundle.json", bundle)
        print(json.dumps({
            "bundle_dir": str(bundle_dir),
            "bundle_json": str(bundle_dir / "bundle.json"),
            "episode_id": episode_id,
            "action_tape_sha256": tape.action_tape_sha256,
            "completed": 1,
            "success": int(terminal["success"]),
            "drop": int(terminal["termination_reason"] == "drop"),
            "termination_reason": terminal["termination_reason"],
            "replay_passed": bool(replay["passed"]),
        }, sort_keys=True))
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
        args.run_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.run_root / "failure.json",
            {
                "schema_version": "se3wam-gpu-planner-e0-failure-v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
