#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the resumable, accepted-prefix t1_xyz A100 production canary.

The supervisor executes one sealed E0 reset at a time, retains fail-closed
attempt evidence, materializes canonical HDF5 only for rows that pass every
strict gate, and stops when the first 50 accepted rows have been published.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

JOB_SCHEMA = "gpu-planner-t1-xyz-production-canary-job-v1"
PROGRESS_SCHEMA = "gpu-planner-t1-xyz-production-canary-progress-v1"
CARD_SCHEMA = "gpu-planner-t1-xyz-production-canary-dataset-v1"
TARGET_ACCEPTED = 50
TASK_ID = "t1_xyz"
KNOWN_REJECTION_SUFFIXES = (
    ".semantic-replay-failure.json",
    ".first-acquisition-failure.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--research-source-root", type=Path, required=True)
    parser.add_argument("--se3-source-root", type=Path, required=True)
    parser.add_argument("--mjwarp-source-root", type=Path, required=True)
    parser.add_argument("--rlinf-source-root", type=Path, required=True)
    parser.add_argument("--dynamic-source-root", type=Path, required=True)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def _load_strict_contract() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/envs/dynamic_benchmark/t1_xyz_strict_evidence.py"
    )
    name = "_t1_xyz_production_canary_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load strict evidence contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _load_job_spec(
    path: Path, strict: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    attempts = payload.get("attempts")
    if (
        payload.get("schema_version") != JOB_SCHEMA
        or payload.get("task_id") != TASK_ID
        or payload.get("target_accepted") != TARGET_ACCEPTED
        or not isinstance(attempts, list)
        or len(attempts) < TARGET_ACCEPTED
    ):
        raise ValueError("production canary job spec identity or target is invalid")
    source_identity = payload.get("source_identity_sha256")
    if not isinstance(source_identity, str) or len(source_identity) != 64:
        raise ValueError("production canary job spec source identity is invalid")
    episode_ids: set[str] = set()
    reset_seeds: set[int] = set()
    verified: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping) or attempt.get("attempt_index") != index:
            raise ValueError(f"production attempt {index} identity is invalid")
        manifest_path = Path(str(attempt.get("manifest")))
        if not manifest_path.is_absolute() or not manifest_path.is_file():
            raise ValueError(f"production attempt {index} manifest is unavailable")
        if _sha256(manifest_path) != attempt.get("manifest_file_sha256"):
            raise ValueError(f"production attempt {index} manifest file digest drifted")
        manifest = strict.load_frozen_manifest(
            manifest_path, expected_phase="e0", verify_exports=True
        )
        row = manifest.row(0)
        request = row["request"]
        if (
            manifest.source_identity_sha256 != source_identity
            or manifest.manifest_sha256 != attempt.get("manifest_sha256")
            or request.get("episode_id") != attempt.get("episode_id")
            or request.get("seed") != attempt.get("reset_seed")
        ):
            raise ValueError(f"production attempt {index} sealed identity drifted")
        episode_id = str(request["episode_id"])
        reset_seed = int(request["seed"])
        if episode_id in episode_ids or reset_seed in reset_seeds:
            raise ValueError("production job contains duplicate reset identities")
        episode_ids.add(episode_id)
        reset_seeds.add(reset_seed)
        verified.append({**dict(attempt), "manifest": str(manifest_path)})
    return payload, verified


def _attempt_paths(run_root: Path, index: int) -> dict[str, Path]:
    root = run_root / "attempts" / f"attempt-{index:03d}"
    result = root / "result.json"
    return {
        "root": root,
        "result": result,
        "tape": root / "result.tape.npz",
        "visual": root / "result.scene-wrist.gif",
        "receipt": root / "episode-materialization.json",
        "status": root / "attempt-status.json",
        "stdout": root / "runner.stdout.log",
        "stderr": root / "runner.stderr.log",
        "semantic_failure": result.with_name(
            f"{result.stem}.semantic-replay-failure.json"
        ),
        "acquisition_failure": result.with_name(
            f"{result.stem}.first-acquisition-failure.json"
        ),
    }


def _run_command(command: list[str], *, stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    if stdout.exists() or stderr.exists():
        raise FileExistsError("refusing to overwrite production attempt logs")
    with stdout.open("wb") as stdout_stream, stderr.open("wb") as stderr_stream:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
        )
    return int(completed.returncode)


def _materialize_command(
    *, args: argparse.Namespace, attempt: Mapping[str, Any], paths: Mapping[str, Path]
) -> list[str]:
    script = Path(__file__).with_name(
        "materialize_gpu_planner_t1_xyz_exploratory_episode.py"
    )
    return [
        sys.executable,
        str(script),
        "--manifest",
        str(attempt["manifest"]),
        "--row-index",
        "0",
        "--result",
        str(paths["result"]),
        "--tape",
        str(paths["tape"]),
        "--dataset-root",
        str(args.dataset_root),
        "--receipt",
        str(paths["receipt"]),
    ]


def _validate_accepted(
    *,
    args: argparse.Namespace,
    strict: Any,
    attempt: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    manifest = strict.load_frozen_manifest(
        Path(str(attempt["manifest"])), expected_phase="e0", verify_exports=True
    )
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    strict.validate_result_for_row(result, manifest=manifest, row=manifest.row(0))
    completed = subprocess.run(
        _materialize_command(args=args, attempt=attempt, paths=paths),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "accepted production row failed canonical materialization: "
            f"{completed.stderr.strip()}"
        )
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "accepted_exploratory_episode_materialized"
        or receipt.get("episode_id") != attempt["episode_id"]
        or receipt.get("source_identity_sha256") != manifest.source_identity_sha256
        or receipt.get("gpu_uuid") != args.expected_gpu_uuid
        or receipt.get("cpu_physics_or_env_fallback") is not False
    ):
        raise ValueError("accepted production row materialization receipt drifted")
    return receipt


def _classify_existing(
    *,
    args: argparse.Namespace,
    strict: Any,
    attempt: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[str, dict[str, Any]] | None:
    if paths["result"].exists():
        return "accepted", _validate_accepted(
            args=args, strict=strict, attempt=attempt, paths=paths
        )
    if paths["status"].exists():
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
        if (
            status.get("status") != "rejected_hard_gate"
            or status.get("episode_id") != attempt["episode_id"]
        ):
            raise ValueError("existing production rejection status drifted")
        failure_path = Path(str(status.get("failure_evidence")))
        if not failure_path.is_file() or _sha256(failure_path) != status.get(
            "failure_evidence_sha256"
        ):
            raise ValueError("existing production rejection evidence drifted")
        return "rejected", status
    failures = [
        path
        for path in (paths["semantic_failure"], paths["acquisition_failure"])
        if path.exists()
    ]
    if len(failures) == 1:
        failure_payload = json.loads(failures[0].read_text(encoding="utf-8"))
        if failure_payload.get("status") not in {
            "blocked_semantic_fresh_replay",
            "blocked_first_acquisition",
        }:
            raise ValueError("recoverable production rejection taxonomy drifted")
        status = {
            "schema_version": PROGRESS_SCHEMA,
            "status": "rejected_hard_gate",
            "attempt_index": attempt["attempt_index"],
            "episode_id": attempt["episode_id"],
            "reset_seed": attempt["reset_seed"],
            "failure_status": failure_payload["status"],
            "failure_evidence": str(failures[0].resolve()),
            "failure_evidence_sha256": _sha256(failures[0]),
            "runner_return_code": None,
            "recovered_after_supervisor_seam": True,
        }
        _atomic_json(paths["status"], status)
        return "rejected", status
    existing = list(paths["root"].glob("*")) if paths["root"].exists() else []
    if existing:
        raise RuntimeError(
            f"production attempt {attempt['attempt_index']} has an incomplete seam: {existing}"
        )
    return None


def _execute_attempt(
    *,
    args: argparse.Namespace,
    strict: Any,
    attempt: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[str, dict[str, Any]]:
    runner = Path(__file__).with_name("run_gpu_planner_t1_xyz_e0.py")
    command = [
        sys.executable,
        str(runner),
        "--manifest",
        str(attempt["manifest"]),
        "--row-index",
        "0",
        "--phase",
        "e0",
        "--output",
        str(paths["result"]),
        "--tape-output",
        str(paths["tape"]),
        "--visual-gif",
        str(paths["visual"]),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "--research-source-root",
        str(args.research_source_root),
        "--se3-source-root",
        str(args.se3_source_root),
        "--mjwarp-source-root",
        str(args.mjwarp_source_root),
        "--rlinf-source-root",
        str(args.rlinf_source_root),
        "--dynamic-source-root",
        str(args.dynamic_source_root),
        "--device-ordinal",
        str(args.device_ordinal),
        "--image-size",
        str(args.image_size),
    ]
    return_code = _run_command(command, stdout=paths["stdout"], stderr=paths["stderr"])
    if return_code == 0:
        return "accepted", _validate_accepted(
            args=args, strict=strict, attempt=attempt, paths=paths
        )
    failures = [
        path
        for path in (paths["semantic_failure"], paths["acquisition_failure"])
        if path.exists()
    ]
    if len(failures) != 1:
        raise RuntimeError(
            f"production attempt {attempt['attempt_index']} failed outside an allowed "
            "trajectory hard-gate rejection"
        )
    failure_payload = json.loads(failures[0].read_text(encoding="utf-8"))
    if failure_payload.get("status") not in {
        "blocked_semantic_fresh_replay",
        "blocked_first_acquisition",
    }:
        raise ValueError("production rejection failure taxonomy drifted")
    status = {
        "schema_version": PROGRESS_SCHEMA,
        "status": "rejected_hard_gate",
        "attempt_index": attempt["attempt_index"],
        "episode_id": attempt["episode_id"],
        "reset_seed": attempt["reset_seed"],
        "failure_status": failure_payload["status"],
        "failure_evidence": str(failures[0].resolve()),
        "failure_evidence_sha256": _sha256(failures[0]),
        "runner_return_code": return_code,
    }
    _atomic_json(paths["status"], status)
    return "rejected", status


def _write_progress(
    path: Path,
    *,
    job: Mapping[str, Any],
    job_spec_sha256: str,
    attempts: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    status: str,
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": PROGRESS_SCHEMA,
            "status": status,
            "job_id": job["job_id"],
            "task_id": TASK_ID,
            "job_spec_sha256": job_spec_sha256,
            "source_identity_sha256": job["source_identity_sha256"],
            "target_accepted": TARGET_ACCEPTED,
            "attempts_completed": len(attempts),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "attempt_rows": attempts,
        },
    )


def _finalize_dataset(
    *,
    args: argparse.Namespace,
    job: Mapping[str, Any],
    job_spec_sha256: str,
    attempts: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(accepted) != TARGET_ACCEPTED:
        raise ValueError("cannot finalize before exactly 50 accepted rows")
    winner_rows = []
    for receipt in accepted:
        episode_dir = args.dataset_root / receipt["relative_episode_dir"]
        record = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
        winner_rows.append(
            {
                **record,
                "exploratory_canary": True,
                "production_receipt_sha256": _payload_sha256(receipt),
            }
        )
    winner_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in winner_rows
    )
    attempts_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in attempts
    )
    _atomic_text(args.dataset_root / "winner_manifest.jsonl", winner_text)
    _atomic_text(args.dataset_root / "attempts.jsonl", attempts_text)
    card = {
        "schema_version": CARD_SCHEMA,
        "status": "exploratory_canary_complete",
        "job_id": job["job_id"],
        "task_id": TASK_ID,
        "job_spec_sha256": job_spec_sha256,
        "source_identity_sha256": job["source_identity_sha256"],
        "planner_profile_id": job["planner_profile_id"],
        "backend_id": "mjwarp_gpu_v1",
        "gpu_uuid": args.expected_gpu_uuid,
        "target_accepted": TARGET_ACCEPTED,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "attempts_completed": len(attempts),
        "accepted_prefix": True,
        "only_accepted_rows_have_canonical_hdf5": True,
        "image_size": args.image_size,
        "cameras": ["agentview", "robot0_eye_in_hand"],
        "semantic_fresh_replay_required": True,
        "one_shot_acquisition_required": True,
        "terminal_exact_once_required": True,
        "quality_v2_required": True,
        "cpu_physics_or_env_fallback": False,
        "owner_decision_sha256": job["owner_decision_sha256"],
        "exploratory_canary": True,
        "formal_qualification": False,
        "formal_rld3_release": False,
        "quality_v4_complete": False,
        "visual_training_allowed": False,
    }
    _atomic_json(args.dataset_root / "dataset_card.json", card)
    checksum_path = args.dataset_root / "SHA256SUMS"
    members = sorted(
        path
        for path in args.dataset_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_text = "".join(
        f"{_sha256(path)}  {path.relative_to(args.dataset_root).as_posix()}\n"
        for path in members
    )
    _atomic_text(checksum_path, checksum_text)
    return {
        **card,
        "dataset_card_sha256": _sha256(args.dataset_root / "dataset_card.json"),
        "sha256sums_sha256": _sha256(checksum_path),
        "sha256_member_count": len(members),
        "persistent_file_count": len(members) + 1,
        "persistent_bytes": sum(path.stat().st_size for path in members)
        + checksum_path.stat().st_size,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.image_size != 224:
        raise ValueError("production canary requires exact 224x224 observations")
    if args.device_ordinal < 0 or not args.expected_gpu_uuid.strip():
        raise ValueError("production canary GPU identity is invalid")
    strict = _load_strict_contract()
    job, attempt_specs = _load_job_spec(args.job_spec, strict)
    job_spec_sha256 = _sha256(args.job_spec)
    if Path(__file__).resolve().parents[2] != args.rlinf_source_root.resolve(
        strict=True
    ):
        raise RuntimeError(
            "production supervisor is outside the sealed RLinf source root"
        )
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.run_root / "progress.json"
    attempts: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for attempt in attempt_specs:
        if len(accepted) >= TARGET_ACCEPTED:
            break
        paths = _attempt_paths(args.run_root, int(attempt["attempt_index"]))
        classified = _classify_existing(
            args=args,
            strict=strict,
            attempt=attempt,
            paths=paths,
        )
        if classified is None:
            classified = _execute_attempt(
                args=args,
                strict=strict,
                attempt=attempt,
                paths=paths,
            )
        disposition, evidence = classified
        attempt_row = {
            "attempt_index": attempt["attempt_index"],
            "episode_id": attempt["episode_id"],
            "reset_seed": attempt["reset_seed"],
            "disposition": disposition,
            "evidence_sha256": _payload_sha256(evidence),
        }
        attempts.append(attempt_row)
        if disposition == "accepted":
            accepted.append(evidence)
        else:
            rejected.append(evidence)
        _write_progress(
            progress_path,
            job=job,
            job_spec_sha256=job_spec_sha256,
            attempts=attempts,
            accepted=accepted,
            rejected=rejected,
            status="running" if len(accepted) < TARGET_ACCEPTED else "target_reached",
        )
    if len(accepted) != TARGET_ACCEPTED:
        _write_progress(
            progress_path,
            job=job,
            job_spec_sha256=job_spec_sha256,
            attempts=attempts,
            accepted=accepted,
            rejected=rejected,
            status="blocked_attempt_inventory_exhausted",
        )
        raise RuntimeError(
            f"production canary exhausted {len(attempt_specs)} resets with "
            f"only {len(accepted)} accepted"
        )
    final = _finalize_dataset(
        args=args,
        job=job,
        job_spec_sha256=job_spec_sha256,
        attempts=attempts,
        accepted=accepted,
        rejected=rejected,
    )
    _write_progress(
        progress_path,
        job=job,
        job_spec_sha256=job_spec_sha256,
        attempts=attempts,
        accepted=accepted,
        rejected=rejected,
        status="complete",
    )
    print(json.dumps(final, sort_keys=True))


if __name__ == "__main__":
    main()
