# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the bounded GPUPLAN0 P0-Grasp B=1 natural-termination E0.

This is an engineering smoke, not a result job. It binds one deterministic
manifest, calls the privileged CPU Planner from each current GPU STATE audit,
executes every E7 action through CUDA, records GPU scene/wrist media, and
replays the complete tape through a fresh backend without calling the Planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "se3wam-gpu-planner-review-bundle-v1"
SCHEMA_ID = "https://se3-wam.local/schemas/gpu-planner-review-bundle-v1.json"
TASK_QUALITY_SCHEMA_VERSION = "db0-episode-task-quality-v2"
BACKEND_ID = "mjwarp_gpu_v1"
JOB_ID = "GPUPLAN0/p0-grasp-e0-v1"
_RUN_ARGUMENTS = (
    "source_root",
    "run_root",
    "bundle_id",
    "resource_unit",
    "container_name",
    "export_dir",
    "expected_gpu_uuid",
    "se3_commit",
    "se3_tree",
    "rlinf_commit",
    "rlinf_tree",
    "dynamic_commit",
    "dynamic_tree",
    "dynamic_rlinf_gitlink",
    "mjwarp_gitlink",
    "mjwarp_tree",
    "research_commit",
    "research_tree",
    "review_schema",
)


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
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
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
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


def _repo_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    status = _git(resolved, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source repository is not clean: {resolved}: {status}")
    return {
        "commit": _git(resolved, "rev-parse", "HEAD"),
        "tree": _git(resolved, "show", "-s", "--format=%T", "HEAD"),
    }


def _git_object(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase full Git object id")
    return value


def _gitlink(path: Path, relative: str) -> str:
    rows = _git(path, "ls-tree", "HEAD", relative).splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"source gitlink is missing or ambiguous: {path}/{relative}")
    fields = rows[0].split()
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        raise RuntimeError(f"source gitlink row is malformed: {rows[0]!r}")
    return fields[2]


def _verify_source(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Path]]:
    root = args.source_root.resolve(strict=True)
    paths = {
        "research": root / "se3-wam-research",
        "se3_wam": root / "SE3-WAM",
        "dynamic_benchmark": root / "se3-wam-dynamic-benchmark",
    }
    paths["rlinf"] = paths["dynamic_benchmark"] / "third_party" / "RLinf"
    paths["mjwarp"] = paths["se3_wam"] / "third_party" / "mujoco_warp"
    identities = {name: _repo_identity(path) for name, path in paths.items()}
    expected = {
        "research": (
            _git_object("research_commit", args.research_commit),
            _git_object("research_tree", args.research_tree),
        ),
        "se3_wam": (
            _git_object("se3_commit", args.se3_commit),
            _git_object("se3_tree", args.se3_tree),
        ),
        "rlinf": (
            _git_object("rlinf_commit", args.rlinf_commit),
            _git_object("rlinf_tree", args.rlinf_tree),
        ),
        "dynamic_benchmark": (
            _git_object("dynamic_commit", args.dynamic_commit),
            _git_object("dynamic_tree", args.dynamic_tree),
        ),
        "mjwarp": (
            _git_object("mjwarp_gitlink", args.mjwarp_gitlink),
            _git_object("mjwarp_tree", args.mjwarp_tree),
        ),
    }
    for name, (commit, tree) in expected.items():
        if identities[name] != {"commit": commit, "tree": tree}:
            raise RuntimeError(
                f"{name} source identity mismatch: {identities[name]} != "
                f"{{'commit': {commit!r}, 'tree': {tree!r}}}"
            )
    mjwarp_gitlink = _gitlink(paths["se3_wam"], "third_party/mujoco_warp")
    dynamic_rlinf_gitlink = _gitlink(paths["dynamic_benchmark"], "third_party/RLinf")
    if mjwarp_gitlink != identities["mjwarp"]["commit"]:
        raise RuntimeError(
            "SE3-WAM MJWarp gitlink differs from the checked-out submodule"
        )
    if dynamic_rlinf_gitlink != identities["rlinf"]["commit"]:
        raise RuntimeError(
            "Dynamic Benchmark RLinf gitlink differs from the checked-out submodule"
        )
    expected_dynamic_rlinf_gitlink = _git_object(
        "dynamic_rlinf_gitlink",
        args.dynamic_rlinf_gitlink,
    )
    if dynamic_rlinf_gitlink != expected_dynamic_rlinf_gitlink:
        raise RuntimeError(
            "Dynamic Benchmark RLinf gitlink differs from the frozen tuple"
        )
    source = {
        "research": identities["research"],
        "se3_wam": {
            **identities["se3_wam"],
            "mjwarp_gitlink": mjwarp_gitlink,
            "mjwarp_tree": identities["mjwarp"]["tree"],
        },
        "rlinf": identities["rlinf"],
        "dynamic_benchmark": {
            **identities["dynamic_benchmark"],
            "rlinf_gitlink": dynamic_rlinf_gitlink,
        },
    }
    return source, paths


def _request_payload(request: Any) -> dict[str, Any]:
    return {
        "episode_id": request.episode_id,
        "task_id": request.task_id,
        "split": getattr(request.split, "value", request.split),
        "seed": int(request.seed),
        "action_mode": getattr(request.action_mode, "value", request.action_mode),
        "observation_track": getattr(
            request.observation_track,
            "value",
            request.observation_track,
        ),
        "object_mode": request.object_mode,
        "reset_mode": request.reset_mode,
        "factors": _jsonable(request.factors),
        "api_version": request.api_version,
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


def _encode_gif(frames: tuple[np.ndarray, ...], path: Path) -> dict[str, Any]:
    if not frames:
        raise RuntimeError("GPU render audit returned no frames")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL is required to encode scene/wrist GIFs") from exc
    normalized = []
    shape = None
    for frame in frames:
        array = np.asarray(frame)
        if (
            array.ndim != 3
            or array.shape[-1] != 3
            or array.dtype != np.uint8
            or not np.all(np.isfinite(array))
            or not np.any(array)
        ):
            raise RuntimeError(
                f"GPU RGB frame has invalid layout/content {array.shape}/{array.dtype}"
            )
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise RuntimeError("GPU RGB frame dimensions changed during the episode")
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
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"bundle.json", "SHA256SUMS"}:
            continue
        rows.append(f"{_file_sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checksum_path


def _verify_checksums(root: Path) -> None:
    expected = {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"bundle.json", "SHA256SUMS"}
    }
    if not expected:
        raise RuntimeError("bundle has no checksum-covered artifacts")
    rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if not rows:
        raise RuntimeError("SHA256SUMS is empty")
    observed = {}
    for row in rows:
        digest, separator, relative = row.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
            or relative not in expected
            or relative in observed
            or expected.get(relative) != digest
        ):
            raise RuntimeError(f"checksum verification failed for {relative or row}")
        observed[relative] = digest
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        raise RuntimeError(f"SHA256SUMS does not cover the complete bundle: {missing}")


def _finalize_bundle(bundle_dir: Path) -> int:
    root = bundle_dir.resolve(strict=True)
    bundle_path = root / "bundle.json"
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("bundle schema identity differs during lease finalization")
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("runtime_ledger_lease_released") is not False
    ):
        raise RuntimeError("bundle is not awaiting one runtime-ledger release")
    _verify_checksums(root)
    checksums = payload.get("checksums")
    instructions = payload.get("import_instructions")
    if checksums != {
        "path": "SHA256SUMS",
        "sha256": _file_sha256(root / "SHA256SUMS"),
    }:
        raise RuntimeError("bundle checksum receipt differs from SHA256SUMS")
    if instructions != {
        "path": "import_instructions.md",
        "sha256": _file_sha256(root / "import_instructions.md"),
    }:
        raise RuntimeError(
            "bundle import-instruction receipt differs from its artifact"
        )
    runtime["runtime_ledger_lease_released"] = True
    _write_json(bundle_path, payload)
    print(
        json.dumps(
            {
                "bundle": str(bundle_path),
                "runtime_ledger_lease_released": True,
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--bundle-id")
    parser.add_argument("--job-id", choices=(JOB_ID,), default=JOB_ID)
    parser.add_argument("--resource-unit")
    parser.add_argument("--container-name")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--se3-commit")
    parser.add_argument("--se3-tree")
    parser.add_argument("--rlinf-commit")
    parser.add_argument("--rlinf-tree")
    parser.add_argument("--dynamic-commit")
    parser.add_argument("--dynamic-tree")
    parser.add_argument("--dynamic-rlinf-gitlink")
    parser.add_argument("--mjwarp-gitlink")
    parser.add_argument("--mjwarp-tree")
    parser.add_argument("--research-commit")
    parser.add_argument("--research-tree")
    parser.add_argument("--review-schema", type=Path)
    parser.add_argument("--manifest-seed", type=int, default=20261050)
    parser.add_argument("--manifest-size", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--prior-failure", type=Path, action="append", default=[])
    parser.add_argument("--finalize-bundle", type=Path)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.finalize_bundle is not None:
        return _finalize_bundle(args.finalize_bundle)
    if args.manifest_size < 1 or args.manifest_seed < 0:
        raise ValueError("manifest seed/size must be non-negative and positive")
    if args.image_size < 1:
        raise ValueError("image-size must be positive")
    run_root = args.run_root.resolve()
    bundle_dir = run_root / "review-bundle" / "p0_grasp" / args.bundle_id
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty bundle directory {bundle_dir}"
        )
    export_dir = args.export_dir.resolve(strict=True)
    schema_path = args.review_schema.resolve(strict=True)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if (
        schema.get("$id") != SCHEMA_ID
        or schema.get("properties", {}).get("schema_version", {}).get("const")
        != SCHEMA_VERSION
    ):
        raise RuntimeError("review bundle schema identity differs from GPUPLAN0 v1")
    source, source_paths = _verify_source(args)
    expected_schema_path = (
        source_paths["research"]
        / "docs"
        / "experiments"
        / "GPUPLAN0"
        / "design"
        / "review_bundle_schema.json"
    ).resolve(strict=True)
    if schema_path != expected_schema_path:
        raise RuntimeError("review schema is not the exact research-source artifact")

    run_root.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "episodes").mkdir()
    (bundle_dir / "media").mkdir()
    candidate_dir = bundle_dir / "candidate"
    schema_dir = bundle_dir / "schema"
    candidate_dir.mkdir()
    schema_dir.mkdir()
    shutil.copyfile(schema_path, schema_dir / "review_bundle_schema.json")

    rlinf_root = source_paths["rlinf"]
    se3_root = source_paths["se3_wam"]
    planner_source = (
        rlinf_root / "rlinf" / "envs" / "dynamic_benchmark" / "p0_grasp_planner.py"
    )
    backend_source = (
        rlinf_root / "rlinf" / "envs" / "dynamic_benchmark" / "gpu_tensor_backend.py"
    )
    runner_source = rlinf_root / "examples" / "embodiment" / Path(__file__).name
    task_config = (
        se3_root / "src" / "se3_wam" / "benchmark" / "configs" / "p0_grasp_v0_1.yaml"
    )
    shutil.copyfile(planner_source, candidate_dir / planner_source.name)
    shutil.copyfile(backend_source, candidate_dir / backend_source.name)
    shutil.copyfile(runner_source, candidate_dir / runner_source.name)
    shutil.copyfile(task_config, candidate_dir / task_config.name)

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(se3_root / "src"), str(rlinf_root)]
    import torch
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import (
        GpuNativeTensorBackendEnv,
    )
    from rlinf.envs.dynamic_benchmark.p0_grasp_planner import (
        CurrentStatePlannerAdapter,
        replay_action_trajectory,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("P0 E0 requires real CUDA; CPU fallback is forbidden")
    common = {
        "task_id": "p0_grasp",
        "num_envs": 1,
        "export_dir": str(export_dir),
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "expected_se3_source_commit": args.se3_commit,
        "expected_se3_source_tree": args.se3_tree,
        "device_ordinal": 0,
        "image_size": args.image_size,
        "render_observations": True,
        "split": "train",
        "manifest_seed": args.manifest_seed,
        "manifest_size": args.manifest_size,
        "task_quality_schema_version": TASK_QUALITY_SCHEMA_VERSION,
        "task_quality_evaluator_backend_id": BACKEND_ID,
        "observation_track": "state",
    }

    wall_start = time.perf_counter()
    online_env = None
    replay_env = None
    try:
        online_env = GpuNativeTensorBackendEnv(**common)
        manifest_requests = online_env.sequence_requests
        request = online_env.next_requests()[0]
        planner, planner_metadata = make_privileged_teacher(
            "p0_grasp",
            request=request,
            image_size=args.image_size,
        )
        adapter = CurrentStatePlannerAdapter(online_env, planner)
        reset = adapter.reset()
        if reset.episode_ids != (request.episode_id,):
            raise RuntimeError(
                "reset episode identity differs from the pinned manifest preview"
            )
        if reset.manifest_sha256 != online_env.manifest_sha256:
            raise RuntimeError("reset manifest identity differs from the backend")
        _last_result, terminal_rows = adapter.run_natural_termination()
        online_row = terminal_rows[0]
        online_device = online_env.attest_end()
        tape = adapter.tape
        scene_frames = adapter.scene_frames
        wrist_frames = adapter.wrist_frames
        online_teacher_audits = online_env.teacher_audit_materializations
        online_transport_checks = online_env.transport_checks
        if (
            online_teacher_audits != tape.identity.horizon_steps
            or online_transport_checks != tape.identity.horizon_steps
        ):
            raise RuntimeError(
                "online current-observation/action transport counts differ from the tape"
            )
        online_env.close()
        online_env = None

        replay_env = GpuNativeTensorBackendEnv(**common)
        replay = replay_action_trajectory(replay_env, tape)
        replay_device = replay_env.attest_end()
        replay_teacher_audits = replay_env.teacher_audit_materializations
        replay_transport_checks = replay_env.transport_checks
        if (
            replay_teacher_audits != tape.identity.horizon_steps
            or replay_transport_checks != tape.identity.horizon_steps
        ):
            raise RuntimeError(
                "fresh replay audit/transport counts differ from the tape"
            )
        replay_env.close()
        replay_env = None
        replay_row = replay.terminal_rows[0]
        terminal = _terminal_dict(online_row)
        replay_terminal = _terminal_dict(replay_row)
        if terminal != replay_terminal:
            raise RuntimeError("online and replay terminal ledger rows differ")
        if online_device != replay_device:
            raise RuntimeError("online and replay CUDA device identities differ")
        source_end, _source_paths_end = _verify_source(args)
        if source_end != source:
            raise RuntimeError(
                "clean source identity changed during online/replay execution"
            )

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
        replay_payload = {
            "schema_version": "se3wam-gpu-planner-replay-v2",
            "passed": replay.passed,
            "fresh_backend": True,
            "planner_called": False,
            "action_tape_sha256": _file_sha256(action_path),
            "trajectory_tape_sha256": _file_sha256(trajectory_path),
            "trajectory_payload_sha256": tape.sha256,
            "backend_identity_sha256": replay.backend_identity_sha256,
            "online_teacher_audit_count": online_teacher_audits,
            "online_action_transport_count": online_transport_checks,
            "replay_teacher_audit_count": replay_teacher_audits,
            "replay_action_transport_count": replay_transport_checks,
            "terminal_ledger_materializations": {"online": 1, "replay": 1},
            "terminal": replay_terminal,
            "steps": [_jsonable(step) for step in replay.steps],
        }
        _write_json(replay_path, replay_payload)
        scene_media = _encode_gif(scene_frames, media_dir / "scene.gif")
        wrist_media = _encode_gif(wrist_frames, media_dir / "wrist.gif")
        scene_media["path"] = str(Path("media") / episode_id / "scene.gif")
        wrist_media["path"] = str(Path("media") / episode_id / "wrist.gif")

        manifest_payload = {
            "schema_version": "se3wam-p0-grasp-e0-manifest-v1",
            "task_id": "p0_grasp",
            "split": "train",
            "manifest_seed": args.manifest_seed,
            "manifest_size": args.manifest_size,
            "manifest_sha256": reset.manifest_sha256,
            "requests": [_request_payload(value) for value in manifest_requests],
            "selected_ordinal": int(reset.manifest_ordinals[0]),
            "selected_episode_id": episode_id,
        }
        _write_json(bundle_dir / "manifest.json", manifest_payload)

        config_payload = {
            "schema_version": "se3wam-p0-grasp-e0-candidate-config-v1",
            "task_id": "p0_grasp",
            "backend_id": BACKEND_ID,
            "observation_track": "state",
            "action_mode": "E7",
            "planner_compute_device": "cpu_numpy",
            "planner_metadata": _jsonable(planner_metadata),
            "render_observations": True,
            "num_envs": 1,
            "image_size": args.image_size,
            "clock": {
                "physics_hz": 500,
                "controller_hz": 20,
                "sensor_hz": 20,
                "physics_steps_per_control": 25,
                "horizon_control_steps": tape.identity.max_horizon_steps,
            },
            "manifest": {
                "seed": args.manifest_seed,
                "size": args.manifest_size,
                "sha256": reset.manifest_sha256,
                "selected_ordinal": int(reset.manifest_ordinals[0]),
            },
            "task_config_sha256": _file_sha256(task_config),
            "planner_source_sha256": _file_sha256(planner_source),
            "tensor_backend_source_sha256": _file_sha256(backend_source),
            "runner_source_sha256": _file_sha256(runner_source),
            "review_schema_sha256": _file_sha256(schema_path),
            "source_reverified_after_replay": True,
        }
        _write_json(bundle_dir / "candidate_config.json", config_payload)

        prior_failures = []
        for failure_path in args.prior_failure:
            resolved = failure_path.resolve(strict=True)
            prior_failures.append(
                {
                    "path": str(resolved),
                    "sha256": _file_sha256(resolved),
                    "payload": json.loads(resolved.read_text(encoding="utf-8")),
                }
            )
        _write_json(
            bundle_dir / "failure_facts.json",
            {
                "schema_version": "se3wam-gpu-planner-e0-failure-facts-v1",
                "current_attempt_failure": None,
                "prior_attempts": prior_failures,
            },
        )

        wall_seconds = time.perf_counter() - wall_start
        termination_counts = dict(Counter([terminal["termination_reason"]]))
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
                (len(scene_frames) + len(wrist_frames)) / wall_seconds
                if wall_seconds > 0
                else None
            ),
        }
        _write_json(bundle_dir / "aggregate.json", aggregate)

        quality_identity = {
            "task_id": "p0_grasp",
            "schema_version": TASK_QUALITY_SCHEMA_VERSION,
            "evaluator_backend_id": BACKEND_ID,
            "task_config_sha256": _file_sha256(task_config),
        }
        source["planner"] = {
            "source_path": "candidate/p0_grasp_planner.py",
            "source_sha256": _file_sha256(candidate_dir / "p0_grasp_planner.py"),
            "config_sha256": _file_sha256(candidate_dir / "p0_grasp_v0_1.yaml"),
        }
        freeze_payload = {
            "source": source,
            "candidate_config": config_payload,
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
                "path": str(action_path.relative_to(bundle_dir)),
                "sha256": _file_sha256(action_path),
            },
            "trajectory_tape": {
                "path": str(trajectory_path.relative_to(bundle_dir)),
                "sha256": _file_sha256(trajectory_path),
            },
            "replay": {
                "path": str(replay_path.relative_to(bundle_dir)),
                "sha256": _file_sha256(replay_path),
                "passed": replay.passed,
            },
            "scene": scene_media,
            "wrist": wrist_media,
        }
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "bundle_status": "complete",
            "bundle_id": args.bundle_id,
            "task_id": "p0_grasp",
            "route": {
                "environment": "gpu",
                "policy_family": "planner",
                "backend_id": BACKEND_ID,
                "physics_device": "cuda",
                "render_device": "cuda",
                "planner_compute_device": "cpu",
            },
            "claim_scope": (
                "B=1 natural-termination E0 engineering smoke: current STATE "
                "observation to CPU Planner to E7 on mjwarp_gpu_v1 CUDA "
                "physics/render, exact-once terminal ledger, complete tape, and "
                "fresh-backend replay; no success-rate, promotion, or held-out claim."
            ),
            "source": source,
            "runtime": {
                "real_cuda": True,
                "cpu_planner_compute_allowed": True,
                "cpu_environment_or_physics_fallback": False,
                "current_observation_closed_loop": True,
                "frozen_action_replay_used_as_planner": False,
                "gpu_rendering": True,
                "physical_gpu_uuid": online_device.expected_uuid,
                "cuda_version": str(torch.version.cuda),
                "driver_version": online_device.driver_version,
                "resource_unit": args.resource_unit,
                "runtime_ledger_job_id": args.job_id,
                "gpu_ownership_verified": True,
                "runtime_ledger_lease_released": False,
                "container_name": args.container_name,
                "source_root": str(args.source_root.resolve()),
                "run_root": str(run_root),
                "clock": {
                    "physics_hz": 500,
                    "controller_hz": 20,
                    "sensor_hz": 20,
                    "physics_steps_per_control": 25,
                    "horizon_control_steps": tape.identity.max_horizon_steps,
                },
                "controller_driver_identity": (
                    "p0_grasp_gpu_native_driver_v0.1+planner_tape_v2"
                ),
                "finite_overflow_gate_passed": True,
            },
            "candidate": {
                "candidate_id": "planner-v2-p0-e0-review-v2",
                "frozen_before_formal_read": False,
                "freeze_payload_sha256": _sha256(freeze_payload),
            },
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
                "machine_decision_reasons": [
                    "single B=1 engineering E0; no Planner quality or promotion claim"
                ],
                "aggregate": aggregate,
            },
            "comparison": {
                "old_candidate_id": None,
                "old_aggregate": None,
                "new_aggregate": aggregate,
                "matched_reference_id": None,
                "noninferiority_status": "not_applicable_development",
                "noninferiority_margin_absolute": None,
            },
            "episodes": [episode_payload],
            "owner": {
                "status": "pending_owner_review",
                "promotion_authorized": False,
                "training_data_authorized": False,
                "policy_replacement_authorized": False,
            },
            "checksums": {"path": "SHA256SUMS", "sha256": ""},
            "import_instructions": {
                "path": "import_instructions.md",
                "sha256": "",
            },
        }
        (bundle_dir / "import_instructions.md").write_text(
            "# GPUPLAN0 p0_grasp E0 import\n\n"
            "Validate `bundle.json` against `schema/review_bundle_schema.json`, "
            "verify `SHA256SUMS`, source/gitlink identities, singleton CUDA UUID, "
            "released runtime lease, exact-once terminal ledger, full tape, and "
            "fresh replay before the unique serial gallery import. Scene and wrist "
            "contain the reset frame plus every post-control frame at nominal 50 ms "
            "physical cadence, encoded at 100 ms per frame (0.5x playback); the "
            f"terminal occurred at physics step {terminal['physics_step']} "
            f"({terminal['physics_step'] / 500.0:.6f} s). Keep Owner status "
            "`pending_owner_review`; this bundle authorizes neither promotion, "
            "training data, nor policy replacement.\n\n"
            "After the runtime-ledger lease is actually released, finalize only "
            "the lease receipt with `python candidate/run_gpu_planner_p0_e0.py "
            "--finalize-bundle <bundle-directory>`, then validate the finalized "
            "`bundle.json` against the copied schema.\n",
            encoding="utf-8",
        )
        checksum_path = _write_checksums(bundle_dir)
        bundle["checksums"]["sha256"] = _file_sha256(checksum_path)
        bundle["import_instructions"]["sha256"] = _file_sha256(
            bundle_dir / "import_instructions.md"
        )
        _write_json(bundle_dir / "bundle.json", bundle)
        print(
            json.dumps(
                {
                    "bundle_dir": str(bundle_dir),
                    "bundle_json": str(bundle_dir / "bundle.json"),
                    "episode_id": episode_id,
                    "manifest_sha256": reset.manifest_sha256,
                    "tape_sha256": tape.sha256,
                    "completed": 1,
                    "success": int(terminal["success"]),
                    "drop": int(terminal["termination_reason"] == "drop"),
                    "termination_reason": terminal["termination_reason"],
                    "terminal_physics_step": terminal["physics_step"],
                    "control_steps": tape.identity.horizon_steps,
                    "replay_passed": replay.passed,
                    "online_gpu_uuid": online_device.expected_uuid,
                    "replay_gpu_uuid": replay_device.expected_uuid,
                    "scene_frames": len(scene_frames),
                    "wrist_frames": len(wrist_frames),
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        if online_env is not None:
            online_env.close()
        if replay_env is not None:
            replay_env.close()


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.finalize_bundle is None:
        missing = [name for name in _RUN_ARGUMENTS if getattr(args, name) is None]
        if missing:
            parser.error(
                "normal E0 execution requires: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
    try:
        return _run(args)
    except BaseException as exc:
        if args.finalize_bundle is None:
            root = args.run_root.resolve()
            root.mkdir(parents=True, exist_ok=True)
            _write_json(
                root / "failure.json",
                {
                    "schema_version": "se3wam-gpu-planner-e0-failure-v1",
                    "job_id": args.job_id,
                    "bundle_id": args.bundle_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"E0 failure evidence: {root / 'failure.json'}",
                file=sys.stderr,
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
