# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run strict GPUPLAN0 T1-SO3 B=1 natural-termination E0 evidence.

The manifest must be sealed from the exact export request before launch. The
online CPU/NumPy Planner reads the current CUDA STATE observation at every
control boundary. A complete review bundle is written only after exact-once
terminal materialization, quality-v2 evaluation, dual-view GPU media capture,
and strict replay on a fresh ``mjwarp_gpu_v1`` backend all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from run_gpu_planner_p0_e0 import (
        BACKEND_ID,
        SCHEMA_ID,
        SCHEMA_VERSION,
        TASK_QUALITY_SCHEMA_VERSION,
        _encode_gif,
        _file_sha256,
        _jsonable,
        _request_payload,
        _review_contract_payloads,
        _sha256,
        _terminal_dict,
        _verify_source,
        _write_json,
    )
except ModuleNotFoundError:
    from examples.embodiment.run_gpu_planner_p0_e0 import (
        BACKEND_ID,
        SCHEMA_ID,
        SCHEMA_VERSION,
        TASK_QUALITY_SCHEMA_VERSION,
        _encode_gif,
        _file_sha256,
        _jsonable,
        _request_payload,
        _review_contract_payloads,
        _sha256,
        _terminal_dict,
        _verify_source,
        _write_json,
    )

TASK_ID = "t1_so3"
EVALUATION_PHASE = "development"
FROZEN_MANIFEST_SCHEMA_VERSION = "se3wam-t1-so3-e0-frozen-manifest-v2"
EXPECTED_HORIZON_STEPS = 160
_RUN_ARGUMENTS = (
    "source_root",
    "run_root",
    "bundle_id",
    "job_id",
    "resource_unit",
    "container_name",
    "export_dir",
    "frozen_manifest",
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


def _manifest_sha256(requests: tuple[Any, ...]) -> str:
    encoded = json.dumps(
        [_request_payload(request) for request in requests],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frozen_manifest_payload(request: Any) -> dict[str, Any]:
    """Build the B=1 sealing payload from one exact exported reset request."""

    request_payload = _request_payload(request)
    if (
        request_payload.get("task_id") != TASK_ID
        or request_payload.get("split") != "test_id"
        or request_payload.get("action_mode") != "E7"
        or request_payload.get("observation_track") != "state"
    ):
        raise ValueError(
            "T1-SO3 manifest requires the exact test_id STATE/E7 export request"
        )
    return {
        "schema_version": FROZEN_MANIFEST_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "batch_size": 1,
        "request_sha256": _sha256(request_payload),
        "manifest_sha256": _manifest_sha256((request,)),
        "request": request_payload,
    }


def _validate_frozen_manifest_payload(
    payload: Any,
    request: Any,
) -> Mapping[str, Any]:
    """Bind a pre-launch seal to the immutable request loaded from the export."""

    required = {
        "schema_version",
        "task_id",
        "batch_size",
        "request_sha256",
        "manifest_sha256",
        "request",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("frozen T1-SO3 manifest has an unexpected field set")
    expected = _frozen_manifest_payload(request)
    if dict(payload) != expected:
        raise ValueError("frozen T1-SO3 manifest differs from the exact export request")
    return expected


def _load_frozen_manifest(path: Path, request: Any) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return _validate_frozen_manifest_payload(payload, request)


def _without_episode_id(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("episode_id", None)
    return normalized


def _validate_review_bundle(
    schema: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError(
            "jsonschema is required before a complete review bundle can be emitted"
        ) from exc
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(value) for value in first.absolute_path) or "<root>"
        raise RuntimeError(
            f"review bundle schema validation failed at {location}: {first.message}"
        )


def _write_checksums(root: Path) -> Path:
    checksum_path = root / "SHA256SUMS"
    control_files = {"bundle.json", "bundle.pending.json", "SHA256SUMS"}
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in control_files:
            continue
        rows.append(f"{_file_sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checksum_path


def _verify_checksums(root: Path) -> None:
    control_files = {"bundle.json", "bundle.pending.json", "SHA256SUMS"}
    expected = {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in control_files
    }
    if not expected:
        raise RuntimeError("bundle has no checksum-covered artifacts")
    rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
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
        raise RuntimeError(f"SHA256SUMS does not cover the complete payload: {missing}")


def _finalize_bundle(bundle_dir: Path) -> int:
    """Publish bundle.json only after the external runtime lease is released."""

    root = bundle_dir.resolve(strict=True)
    pending_path = root / "bundle.pending.json"
    bundle_path = root / "bundle.json"
    if bundle_path.exists() or not pending_path.is_file():
        raise RuntimeError(
            "finalization requires exactly one unpublished pending bundle"
        )
    payload = json.loads(pending_path.read_text(encoding="utf-8"))
    runtime = payload.get("runtime")
    episodes = payload.get("episodes")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("bundle_status") != "complete"
        or not isinstance(runtime, dict)
        or runtime.get("runtime_ledger_lease_released") is not False
        or not isinstance(episodes, list)
        or len(episodes) != 1
        or episodes[0].get("replay", {}).get("passed") is not True
    ):
        raise RuntimeError("pending bundle is not eligible for lease finalization")
    _verify_checksums(root)
    checksums = {"path": "SHA256SUMS", "sha256": _file_sha256(root / "SHA256SUMS")}
    instructions = {
        "path": "import_instructions.md",
        "sha256": _file_sha256(root / "import_instructions.md"),
    }
    identity = payload.get("source_bundle_identity")
    if (
        payload.get("checksums") != checksums
        or payload.get("import_instructions") != instructions
        or not isinstance(identity, dict)
        or identity.get("checksums") != checksums
        or identity.get("import_instructions") != instructions
    ):
        raise RuntimeError("pending bundle file receipts differ from the payload")
    schema_path = root / "schema" / "review_bundle_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if (
        schema.get("$id") != SCHEMA_ID
        or schema.get("properties", {}).get("schema_version", {}).get("const")
        != SCHEMA_VERSION
    ):
        raise RuntimeError("copied review schema identity differs during finalization")
    runtime["runtime_ledger_lease_released"] = True
    _validate_review_bundle(schema, payload)
    _write_json(pending_path, payload)
    pending_path.replace(bundle_path)
    _verify_checksums(root)
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
    parser.add_argument("--job-id")
    parser.add_argument("--resource-unit")
    parser.add_argument("--container-name")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--frozen-manifest", type=Path)
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
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--prior-failure", type=Path, action="append", default=[])
    parser.add_argument("--finalize-bundle", type=Path)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.finalize_bundle is not None:
        return _finalize_bundle(args.finalize_bundle)
    if args.image_size < 1:
        raise ValueError("image-size must be positive")
    run_root = args.run_root.resolve()
    bundle_dir = run_root / "review-bundle" / TASK_ID / args.bundle_id
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty bundle directory {bundle_dir}"
        )
    export_dir = args.export_dir.resolve(strict=True)
    manifest_path = args.frozen_manifest.resolve(strict=True)
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

    rlinf_root = source_paths["rlinf"]
    se3_root = source_paths["se3_wam"]
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(se3_root / "src"), str(rlinf_root)]
    import torch
    from se3_wam.benchmark.gpu_native.p0_grasp_engine import (
        load_p0_grasp_artifacts,
    )
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import (
        GpuNativeTensorBackendEnv,
    )
    from rlinf.envs.dynamic_benchmark.t1_so3_planner import (
        CurrentStatePlannerAdapter,
        causal_observation_fingerprint,
        replay_action_trajectory,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("T1-SO3 E0 requires real CUDA; CPU fallback is forbidden")
    artifacts = load_p0_grasp_artifacts(str(export_dir))
    request = artifacts.reset_request
    frozen_manifest = _load_frozen_manifest(manifest_path, request)
    manifest_sha256 = str(frozen_manifest["manifest_sha256"])
    request_seed = int(request.seed)
    common = {
        "task_id": TASK_ID,
        "num_envs": 1,
        "export_dir": str(export_dir),
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "expected_se3_source_commit": args.se3_commit,
        "expected_se3_source_tree": args.se3_tree,
        "device_ordinal": 0,
        "image_size": args.image_size,
        "render_observations": True,
        "split": "test_id",
        "manifest_seed": request_seed,
        "manifest_size": 1,
        "manifest_requests": (request,),
        "manifest_sha256": manifest_sha256,
        "task_quality_schema_version": TASK_QUALITY_SCHEMA_VERSION,
        "task_quality_evaluator_backend_id": BACKEND_ID,
        "observation_track": "state",
    }

    wall_start = time.perf_counter()
    online_env = None
    replay_env = None
    try:
        online_env = GpuNativeTensorBackendEnv(**common)
        if online_env.cohort_horizon_steps != EXPECTED_HORIZON_STEPS:
            raise RuntimeError("T1-SO3 backend horizon differs from frozen 160 steps")
        active_request = online_env.next_requests()[0]
        if _without_episode_id(_request_payload(active_request)) != _without_episode_id(
            frozen_manifest["request"]
        ):
            raise RuntimeError("active request differs from the frozen export manifest")
        planner, planner_metadata = make_privileged_teacher(
            TASK_ID,
            request=active_request,
            image_size=args.image_size,
        )
        adapter = CurrentStatePlannerAdapter(online_env, planner)
        reset = adapter.reset()
        if reset.episode_ids != (active_request.episode_id,):
            raise RuntimeError(
                "reset episode identity differs from the pinned manifest preview"
            )
        if reset.manifest_sha256 != manifest_sha256:
            raise RuntimeError(
                "reset manifest identity differs from the pre-launch seal"
            )
        _last_result, terminal_rows = adapter.run_natural_termination()
        online_row = terminal_rows[0]
        terminal_observations = online_env.materialize_teacher_observations((0,))
        if len(terminal_observations) != 1:
            raise RuntimeError("online terminal observation audit did not return B=1")
        terminal_fingerprint = causal_observation_fingerprint(terminal_observations[0])
        online_device = online_env.attest_end()
        tape = adapter.tape
        scene_frames = adapter.scene_frames
        wrist_frames = adapter.wrist_frames
        online_teacher_audits = online_env.teacher_audit_materializations
        online_transport_checks = online_env.transport_checks
        if (
            online_teacher_audits != tape.identity.horizon_steps + 1
            or online_transport_checks != tape.identity.horizon_steps
            or len(scene_frames) != tape.identity.horizon_steps + 1
            or len(wrist_frames) != tape.identity.horizon_steps + 1
        ):
            raise RuntimeError(
                "online observation/action/media counts differ from the natural tape"
            )
        online_env.close()
        online_env = None

        replay_env = GpuNativeTensorBackendEnv(**common)
        replay = replay_action_trajectory(replay_env, tape)
        if replay.passed is not True:
            raise RuntimeError("fresh replay did not return a strict pass receipt")
        replay_terminal_observations = replay_env.materialize_teacher_observations((0,))
        if len(replay_terminal_observations) != 1:
            raise RuntimeError("replay terminal observation audit did not return B=1")
        replay_terminal_fingerprint = causal_observation_fingerprint(
            replay_terminal_observations[0]
        )
        if replay_terminal_fingerprint != terminal_fingerprint:
            raise RuntimeError("fresh replay terminal observation fingerprint differs")
        replay_device = replay_env.attest_end()
        replay_teacher_audits = replay_env.teacher_audit_materializations
        replay_transport_checks = replay_env.transport_checks
        if (
            replay_teacher_audits != tape.identity.horizon_steps + 1
            or replay_transport_checks != tape.identity.horizon_steps
        ):
            raise RuntimeError(
                "fresh replay audit/transport counts differ from the tape"
            )
        replay_env.close()
        replay_env = None

        terminal = _terminal_dict(online_row)
        replay_terminal = _terminal_dict(replay.terminal_rows[0])
        if terminal != replay_terminal:
            raise RuntimeError("online and replay terminal ledger rows differ")
        if online_device != replay_device:
            raise RuntimeError("online and replay CUDA device identities differ")
        source_end, _ = _verify_source(args)
        if source_end != source:
            raise RuntimeError("clean source identity changed during execution/replay")

        run_root.mkdir(parents=True, exist_ok=True)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        episode_id = reset.episode_ids[0]
        episode_dir = bundle_dir / "episodes" / episode_id
        media_dir = bundle_dir / "media" / episode_id
        candidate_dir = bundle_dir / "candidate"
        schema_dir = bundle_dir / "schema"
        for directory in (episode_dir, media_dir, candidate_dir, schema_dir):
            directory.mkdir(parents=True, exist_ok=False)

        planner_source = (
            rlinf_root / "rlinf" / "envs" / "dynamic_benchmark" / "t1_so3_planner.py"
        )
        backend_source = (
            rlinf_root
            / "rlinf"
            / "envs"
            / "dynamic_benchmark"
            / "gpu_tensor_backend.py"
        )
        runner_source = rlinf_root / "examples" / "embodiment" / Path(__file__).name
        task_config = (
            se3_root / "src" / "se3_wam" / "benchmark" / "configs" / "t1_so3_v0_16.yaml"
        )
        for source_path, destination in (
            (planner_source, candidate_dir / planner_source.name),
            (backend_source, candidate_dir / backend_source.name),
            (runner_source, candidate_dir / runner_source.name),
            (task_config, candidate_dir / task_config.name),
            (manifest_path, candidate_dir / "frozen_manifest.json"),
            (schema_path, schema_dir / "review_bundle_schema.json"),
        ):
            shutil.copyfile(source_path, destination)

        action_path = episode_dir / "action_tape.json"
        trajectory_path = episode_dir / "trajectory_tape.json"
        replay_path = episode_dir / "replay.json"
        action_payload, trajectory_payload, replay_payload = _review_contract_payloads(
            tape=tape,
            terminal=terminal,
            replay=replay,
            terminal_observation_fingerprint=terminal_fingerprint,
            online_teacher_audits=online_teacher_audits,
            online_transport_checks=online_transport_checks,
            replay_teacher_audits=replay_teacher_audits,
            replay_transport_checks=replay_transport_checks,
        )
        _write_json(action_path, action_payload)
        _write_json(trajectory_path, trajectory_payload)
        replay_payload["trajectory_tape_sha256"] = _file_sha256(trajectory_path)
        _write_json(replay_path, replay_payload)
        scene_media = _encode_gif(scene_frames, media_dir / "scene.gif")
        wrist_media = _encode_gif(wrist_frames, media_dir / "wrist.gif")
        scene_media["path"] = str(Path("media") / episode_id / "scene.gif")
        wrist_media["path"] = str(Path("media") / episode_id / "wrist.gif")

        manifest_payload = {
            **dict(frozen_manifest),
            "source_path": str(manifest_path),
            "source_file_sha256": _file_sha256(manifest_path),
            "active_episode_id": episode_id,
            "active_manifest_ordinal": int(reset.manifest_ordinals[0]),
        }
        _write_json(bundle_dir / "manifest.json", manifest_payload)
        config_payload = {
            "schema_version": "se3wam-t1-so3-e0-candidate-config-v2",
            "task_id": TASK_ID,
            "backend_id": BACKEND_ID,
            "observation_track": "state",
            "action_mode": "E7",
            "planner_compute_device": "cpu_numpy",
            "planner_metadata": _jsonable(planner_metadata),
            "planner_runtime_overrides": {},
            "render_observations": True,
            "num_envs": 1,
            "image_size": args.image_size,
            "clock": {
                "physics_hz": 500,
                "controller_hz": 20,
                "sensor_hz": 20,
                "physics_steps_per_control": 25,
                "horizon_control_steps": EXPECTED_HORIZON_STEPS,
            },
            "manifest": {
                "size": 1,
                "sha256": manifest_sha256,
                "request_sha256": frozen_manifest["request_sha256"],
                "selected_ordinal": int(reset.manifest_ordinals[0]),
            },
            "task_quality_schema_version": TASK_QUALITY_SCHEMA_VERSION,
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
        diagnostics = [entry.diagnostics for entry in tape.entries]
        axis_errors = [
            abs(float(row["modulo_pi_axis_error_rad"]))
            for row in diagnostics
            if row.get("modulo_pi_axis_error_rad") is not None
        ]
        aggregate = {
            "completed": 1,
            "total": 1,
            "success_count": int(terminal["success"]),
            "success_rate": float(int(terminal["success"])),
            "safety_count": None,
            "completion_mean": terminal["completion"],
            "drop_count": int(terminal["termination_reason"] == "drop"),
            "termination_counts": dict(Counter([terminal["termination_reason"]])),
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
        diagnostics_summary = {
            "schema_version": "se3wam-t1-so3-planner-diagnostics-v2",
            "planner_phases": dict(
                Counter(str(row.get("phase")) for row in diagnostics)
            ),
            "max_modulo_pi_axis_error_rad": max(axis_errors) if axis_errors else None,
            "natural_control_steps": tape.identity.horizon_steps,
            "observation_fingerprint_count": tape.identity.horizon_steps + 1,
        }
        diagnostics_path = bundle_dir / "t1_so3_diagnostics.json"
        _write_json(diagnostics_path, diagnostics_summary)

        quality_identity = {
            "task_id": TASK_ID,
            "schema_version": TASK_QUALITY_SCHEMA_VERSION,
            "evaluator_backend_id": BACKEND_ID,
            "task_config_sha256": _file_sha256(task_config),
        }
        source["planner"] = {
            "source_path": "candidate/t1_so3_planner.py",
            "source_sha256": _file_sha256(candidate_dir / "t1_so3_planner.py"),
            "config_sha256": _file_sha256(candidate_dir / "t1_so3_v0_16.yaml"),
        }
        freeze_payload = {
            "source": source,
            "candidate_config": config_payload,
            "manifest_sha256": manifest_sha256,
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
            "terminal": terminal,
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
                "passed": True,
            },
            "scene": scene_media,
            "wrist": wrist_media,
        }
        development_limitations = [
            {
                "code": "b1_engineering_e0_only",
                "statement": "Only one B=1 engineering E0 episode was executed.",
            },
            {
                "code": "candidate_not_frozen",
                "statement": "The development candidate is not a formal frozen candidate.",
            },
            {
                "code": "comparison_not_performed",
                "statement": "No matched comparison or noninferiority test was performed.",
            },
            {
                "code": "d32_not_run",
                "statement": "The D32 diagnostic gate was not run.",
            },
            {
                "code": "mc64_not_run",
                "statement": "The MC64 calibration gate was not run.",
            },
            {
                "code": "held_out_manifest_not_provided",
                "statement": "No formal held-out manifest was provided or consumed.",
            },
            {
                "code": "formal_qualification_not_performed",
                "statement": "Formal qualification was not performed.",
            },
        ]
        source_bundle_path = bundle_dir / "source_bundle.json"
        _write_json(
            source_bundle_path,
            {
                "schema_version": "se3wam-gpu-planner-source-bundle-v1",
                "bundle_id": args.bundle_id,
                "task_id": TASK_ID,
                "source": source,
                "candidate_config": {
                    "path": "candidate_config.json",
                    "sha256": _file_sha256(bundle_dir / "candidate_config.json"),
                },
                "manifest": {
                    "path": "manifest.json",
                    "sha256": _file_sha256(bundle_dir / "manifest.json"),
                },
                "aggregate": {
                    "path": "aggregate.json",
                    "sha256": _file_sha256(bundle_dir / "aggregate.json"),
                },
                "diagnostics": {
                    "path": "t1_so3_diagnostics.json",
                    "sha256": _file_sha256(diagnostics_path),
                },
                "episode": episode_payload,
                "development_limitations": development_limitations,
                "owner_status": "pending_owner_review",
            },
        )
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
            "claim_scope": (
                "B=1 engineering E0 with natural termination: current STATE "
                "observation to default T1-SO3 CPU Planner to E7 on mjwarp_gpu_v1 CUDA "
                "physics/render, quality-v2, exact-once terminal ledger, complete "
                "tape, and strict fresh replay; no qualification, promotion, "
                "training-data, policy-replacement, D32, MC64, or held-out claim."
            ),
            "development_limitations": development_limitations,
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
                    "horizon_control_steps": EXPECTED_HORIZON_STEPS,
                },
                "controller_driver_identity": (
                    "t1_so3_gpu_native_driver_v0.2+planner_tape_v2"
                ),
                "finite_overflow_gate_passed": True,
            },
            "candidate": {
                "candidate_id": "gpu-planner-t1-so3-strict-evidence-v2",
                "frozen_before_formal_read": False,
                "freeze_payload_sha256": _sha256(freeze_payload),
            },
            "evaluation": {
                "phase": EVALUATION_PHASE,
                "split": "test_id",
                "test_ids": [int(reset.manifest_ordinals[0])],
                "manifest_seeds": [request_seed],
                "manifest_sha256": manifest_sha256,
                "expected_episode_count": 1,
                "completed_episode_count": 1,
                "resampled": False,
                "evaluator_backend_id": BACKEND_ID,
                "task_quality_schema_version": TASK_QUALITY_SCHEMA_VERSION,
                "task_quality_identity_sha256": _sha256(quality_identity),
                "machine_decision": "not_evaluable",
                "machine_decision_reasons": [
                    "single B=1 development E0; no formal Planner quality or promotion claim"
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
            "# GPUPLAN0 t1_so3 strict E0 import\n\n"
            "Verify `SHA256SUMS`, the copied review schema, the pre-launch frozen "
            "manifest, all source/gitlink identities, singleton CUDA UUID, "
            "quality-v2 identity, exact-once terminal rows, reset-through-terminal "
            "tape/media coverage, and strict fresh replay. Scene and wrist frames "
            "have nominal 50 ms physical cadence and are encoded at 100 ms per "
            "frame (0.5x playback). Keep Owner status `pending_owner_review`; this "
            "bundle authorizes neither promotion, training data, nor policy "
            "replacement.\n\n"
            "After the exact runtime-ledger lease is released, finalize only the "
            "lease receipt with `python candidate/run_gpu_planner_t1_so3_e0.py "
            "--finalize-bundle <bundle-directory>`. A failed replay must remain "
            "failure evidence and must never be imported as a complete bundle.\n",
            encoding="utf-8",
        )
        checksum_path = _write_checksums(bundle_dir)
        _verify_checksums(bundle_dir)
        bundle["checksums"]["sha256"] = _file_sha256(checksum_path)
        bundle["import_instructions"]["sha256"] = _file_sha256(
            bundle_dir / "import_instructions.md"
        )
        payload_member_count = len(
            [
                row
                for row in checksum_path.read_text(encoding="utf-8").splitlines()
                if row
            ]
        )
        bundle["source_bundle_identity"] = {
            "bundle_id": args.bundle_id,
            "bundle_json": {
                "path": "source_bundle.json",
                "sha256": _file_sha256(source_bundle_path),
            },
            "checksums": dict(bundle["checksums"]),
            "import_instructions": dict(bundle["import_instructions"]),
            "payload_member_count": payload_member_count,
        }
        validation_payload = json.loads(json.dumps(_jsonable(bundle)))
        validation_payload["runtime"]["runtime_ledger_lease_released"] = True
        _validate_review_bundle(schema, validation_payload)
        pending_path = bundle_dir / "bundle.pending.json"
        _write_json(pending_path, bundle)
        print(
            json.dumps(
                {
                    "bundle_dir": str(bundle_dir),
                    "bundle_pending": str(pending_path),
                    "lease_finalization_required": True,
                    "episode_id": episode_id,
                    "manifest_sha256": manifest_sha256,
                    "tape_sha256": tape.sha256,
                    "completed": 1,
                    "success": int(terminal["success"]),
                    "termination_reason": terminal["termination_reason"],
                    "terminal_physics_step": terminal["physics_step"],
                    "control_steps": tape.identity.horizon_steps,
                    "replay_passed": True,
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
                    "task_id": TASK_ID,
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
