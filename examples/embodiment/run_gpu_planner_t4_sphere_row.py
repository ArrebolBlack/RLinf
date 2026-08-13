#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run one exact T4-sphere row with blocking terminal ledger and fresh replay.

The runner always selects B=1.  It first validates the entire preregistered
32-row manifest, the selected row's dedicated SE3 export, the current candidate
repository tuple, the request-sensitive teacher reset, and the frozen task/clock
contract.  ``--phase identity_smoke`` stops there without creating CUDA state.
A D32 row report is written only after the primary terminal ledger and an
independent fresh-backend replay both pass with exact-once negative witnesses.

This entry point does not acquire a GPU lease or authorize D32.  Callers must
obtain the exact Queue/runtime-ledger admission before invoking it with
``--phase d32``.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlinf.envs.dynamic_benchmark.gpu_backend import (
    GpuNativePlannerAdapter,
    assert_terminal_ledger_exact_once,
    replay_recorded_tape,
)
from rlinf.envs.dynamic_benchmark.t4_sphere_d32 import (
    BACKEND_ID,
    D32_ROW_REPORT_SCHEMA_VERSION,
    EXECUTION_CONTRACT,
    TASK_ID,
    jsonable,
    load_candidate_identity,
    load_d32_manifest,
    sha256_file,
    teacher_reset_identity,
    validate_blocking_replay,
    validate_repository_tuple,
    validate_request_identity,
    validate_scientific_contract,
    validate_terminal_row,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--se3-source-root", type=Path, required=True)
    parser.add_argument("--rlinf-source-root", type=Path, required=True)
    parser.add_argument("--dynamic-source-root", type=Path, required=True)
    parser.add_argument("--expected-device-uuid")
    parser.add_argument(
        "--phase",
        choices=("identity_smoke", "d32"),
        default="identity_smoke",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument(
        "--evaluator-backend-id",
        default=EXECUTION_CONTRACT["evaluator_backend_id"],
    )
    parser.add_argument(
        "--task-quality-schema-version",
        default=EXECUTION_CONTRACT["task_quality_schema_version"],
    )
    return parser


def _module_under(module: Any, root: Path, description: str) -> str:
    module_path = Path(module.__file__).resolve()
    resolved_root = root.resolve(strict=True)
    try:
        module_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            f"imported {description} module is outside its frozen source root"
        ) from exc
    return str(module_path)


def _validate_import_roots(*, se3_root: Path, rlinf_root: Path) -> dict[str, str]:
    import se3_wam

    import rlinf

    return {
        "rlinf_module": _module_under(rlinf, rlinf_root, "RLinf"),
        "se3_wam_module": _module_under(se3_wam, se3_root, "SE3-WAM"),
    }


def _validate_export_metadata(
    export_dir: Path,
    row: Mapping[str, Any],
    *,
    expected_task_config_sha256: str,
) -> dict[str, Any]:
    """Reject a stale/row-zero export before importing any CUDA runtime."""

    request_path = export_dir / "request.json"
    report_path = export_dir / "export_report.json"
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    export_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(request_payload, Mapping) or not isinstance(
        export_report, Mapping
    ):
        raise RuntimeError("selected export metadata must contain JSON objects")
    request_fields = (
        "action_mode",
        "candidate_index",
        "episode_id",
        "factors",
        "object_mode",
        "observation_track",
        "pair_id",
        "pair_member_id",
        "reset_mode",
        "seed",
        "source_group_id",
        "split",
        "task_id",
    )
    mismatches = {
        name: {"expected": row.get(name), "actual": request_payload.get(name)}
        for name in request_fields
        if request_payload.get(name) != row.get(name)
    }
    if mismatches:
        raise RuntimeError(f"selected export request.json identity drift: {mismatches}")
    if (
        export_report.get("task_id") != TASK_ID
        or export_report.get("episode_id") != row["episode_id"]
        or export_report.get("row_index") != row["candidate_index"]
        or export_report.get("task_config_sha256") != expected_task_config_sha256
    ):
        raise RuntimeError("selected export report identity/configuration drift")
    return {
        "export_report_sha256": sha256_file(report_path),
        "request_json_sha256": sha256_file(request_path),
        "sha256sums_sha256": sha256_file(export_dir / "SHA256SUMS"),
        "task_config_sha256": export_report["task_config_sha256"],
    }


def _provenance_payload(value: Any) -> dict[str, Any]:
    names = (
        "backend_id",
        "bundle_sha256",
        "config_sha256",
        "device_name",
        "device_ordinal",
        "device_platform",
        "git_commit",
        "git_tree",
        "implementation_version",
        "manifest_sha256",
        "model_sha256",
        "physical_device_identity_source",
        "physical_device_pci_bus_id",
        "physical_device_uuid",
        "precision",
        "request_sha256",
        "runtime_versions",
    )
    return {name: jsonable(getattr(value, name, None)) for name in names}


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_device_uuid: str,
    expected_source_commit: str,
    expected_source_tree: str,
) -> None:
    if provenance.get("backend_id") != BACKEND_ID:
        raise RuntimeError("row execution backend is not mjwarp_gpu_v1")
    if provenance.get("device_platform") not in {"cuda", "gpu"}:
        raise RuntimeError("row execution physics is not CUDA/GPU")
    if provenance.get("physical_device_uuid") != expected_device_uuid:
        raise RuntimeError("row execution resolved to a different physical GPU UUID")
    if (
        provenance.get("git_commit") != expected_source_commit
        or provenance.get("git_tree") != expected_source_tree
    ):
        raise RuntimeError("row execution SE3 source provenance drifted")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.image_size < 16:
        raise ValueError("image-size must be at least 16")
    if (
        isinstance(args.device_ordinal, bool)
        or not isinstance(args.device_ordinal, int)
        or args.device_ordinal < 0
    ):
        raise ValueError("device-ordinal must be a non-negative integer")
    if args.expected_device_uuid is not None and (
        not isinstance(args.expected_device_uuid, str)
        or not args.expected_device_uuid.strip()
    ):
        raise ValueError("expected-device-uuid must be a non-empty physical UUID")
    if args.phase == "d32" and args.expected_device_uuid is None:
        raise ValueError("D32 execution requires an exact leased physical GPU UUID")
    if args.evaluator_backend_id != EXECUTION_CONTRACT["evaluator_backend_id"]:
        raise ValueError(
            "evaluator-backend-id differs from the frozen candidate identity"
        )
    if (
        args.task_quality_schema_version
        != EXECUTION_CONTRACT["task_quality_schema_version"]
    ):
        raise ValueError(
            "task-quality-schema-version differs from the frozen candidate identity"
        )

    candidate = load_candidate_identity(args.candidate_identity)
    repositories = validate_repository_tuple(
        candidate,
        se3_root=args.se3_source_root,
        rlinf_root=args.rlinf_source_root,
        dynamic_root=args.dynamic_source_root,
    )
    import_roots = _validate_import_roots(
        se3_root=args.se3_source_root,
        rlinf_root=args.rlinf_source_root,
    )
    scientific_contract = validate_scientific_contract(candidate)
    manifest = load_d32_manifest(args.manifest, candidate=candidate)
    row = manifest.row(args.row_index)
    export_dir = manifest.export_dir(row)
    export_identity = _validate_export_metadata(
        export_dir,
        row,
        expected_task_config_sha256=candidate.task_config_sha256,
    )

    # This path loads only the sealed host artifacts and the current teacher.
    # It deliberately runs before any GpuNativePlannerAdapter/CUDA construction.
    from se3_wam.benchmark.gpu_native.p0_grasp_engine import load_p0_grasp_artifacts
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    artifacts = load_p0_grasp_artifacts(export_dir)
    request_payload = validate_request_identity(
        artifacts.reset_request,
        row,
        actual_task_config_sha256=artifacts.config_sha256,
        candidate=candidate,
    )
    teacher, preflight_teacher_metadata = make_privileged_teacher(
        TASK_ID,
        request=artifacts.reset_request,
    )
    teacher.reset()
    preflight_teacher_identity = teacher_reset_identity(preflight_teacher_metadata)
    if preflight_teacher_identity != row["teacher_reset_identity"]:
        raise RuntimeError(
            "selected row teacher reset identity differs from the frozen full ResetRequest"
        )

    common_report = {
        "schema_version": D32_ROW_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "phase": args.phase,
        "task_id": TASK_ID,
        "batch_size": 1,
        "identity": {
            "candidate_index": int(row["candidate_index"]),
            "candidate_sha256": candidate.candidate_sha256,
            "episode_id": row["episode_id"],
            "manifest_sha256": manifest.manifest_sha256,
            "seed": int(row["seed"]),
            "task_config_sha256": candidate.task_config_sha256,
        },
        "request": request_payload,
        "execution": dict(EXECUTION_CONTRACT),
        "repositories": repositories,
        "import_roots": import_roots,
        "scientific_contract": scientific_contract,
        "export": {
            "path": str(export_dir),
            **export_identity,
        },
        "teacher": {
            "full_reset_request_bound": True,
            "reset_identity": preflight_teacher_identity,
        },
    }
    if args.phase == "identity_smoke":
        return {
            **common_report,
            "sample_count": 0,
            "counts_as_d32_result": False,
            "closed_loop_planner": False,
            "frozen_action_replay": False,
            "resource": {
                "gpu_created": False,
                "expected_device_uuid": args.expected_device_uuid,
                "observed_device_uuid": None,
            },
            "terminal_ledger": {
                "blocking": True,
                "executed": False,
                "passed": None,
                "exact_once_second_consumption_rejected": None,
                "status": "not_run_without_d32_admission",
            },
            "replay": {
                "blocking": True,
                "executed": False,
                "passed": None,
                "exact_once_second_consumption_rejected": None,
                "status": "not_run_without_d32_admission",
            },
        }

    se3_identity = candidate.repositories["se3_wam"]
    adapter = GpuNativePlannerAdapter(
        task_id=TASK_ID,
        num_envs=1,
        export_dir=str(export_dir),
        device_ordinal=args.device_ordinal,
        image_size=args.image_size,
        observation_track=EXECUTION_CONTRACT["observation_track"],
        evaluator_backend_id=args.evaluator_backend_id,
        schema_version=args.task_quality_schema_version,
        expected_gpu_uuid=args.expected_device_uuid,
        expected_se3_source_commit=se3_identity["commit"],
        expected_se3_source_tree=se3_identity["tree"],
    )
    primary_terminal_payload: dict[str, Any]
    primary_fingerprints: tuple[str, ...]
    action_tape: tuple[Mapping[str, Any], ...]
    teacher_metadata: Mapping[str, Any]
    provenance: dict[str, Any]
    action_tape_sha256: str
    primary_exact_once_witnesses: tuple[str, ...]
    try:
        if adapter.num_envs != 1:
            raise RuntimeError("T4-sphere row runner must remain B=1")
        if adapter.observation_track.value != EXECUTION_CONTRACT["observation_track"]:
            raise RuntimeError(
                "T4-sphere row adapter did not preserve STATE observations"
            )
        requests = adapter.frozen_requests
        if len(requests) != 1:
            raise RuntimeError("T4-sphere export did not bind exactly one ResetRequest")
        adapter_request_payload = validate_request_identity(
            requests[0],
            row,
            actual_task_config_sha256=export_identity["task_config_sha256"],
            candidate=candidate,
        )
        if adapter_request_payload != request_payload:
            raise RuntimeError(
                "CUDA adapter request differs from the preflight ResetRequest"
            )
        adapter.reset(requests)
        if len(adapter.teacher_metadata) != 1:
            raise RuntimeError(
                "T4-sphere B=1 did not build exactly one per-reset teacher"
            )
        teacher_metadata = dict(adapter.teacher_metadata[0])
        if teacher_reset_identity(teacher_metadata) != preflight_teacher_identity:
            raise RuntimeError("CUDA per-reset teacher changed the frozen row identity")
        for _ in range(EXECUTION_CONTRACT["horizon_control_steps"]):
            adapter.step()
            if not adapter.active_mask.any():
                break
        if adapter.active_mask.any():
            raise RuntimeError("T4-sphere row did not terminate within horizon 120")
        terminal_rows = adapter.terminal_rows
        if len(terminal_rows) != 1:
            raise RuntimeError(
                "blocking terminal ledger did not return exactly one row"
            )
        primary_terminal_payload = validate_terminal_row(
            terminal_rows[0],
            episode_id=str(row["episode_id"]),
        )
        primary_exact_once_witnesses = adapter.assert_terminal_ledger_exact_once()
        primary_fingerprints = adapter.observation_fingerprints[0]
        action_tape = adapter.action_tapes[0]
        if len(primary_fingerprints) != len(action_tape) + 1:
            raise RuntimeError("primary trajectory/action tape cardinality drifted")
        action_tape_sha256 = adapter.action_tape_sha256
        provenance = _provenance_payload(adapter.provenance)
        _validate_provenance(
            provenance,
            expected_device_uuid=args.expected_device_uuid,
            expected_source_commit=se3_identity["commit"],
            expected_source_tree=se3_identity["tree"],
        )
        replay_backend_factory = adapter.new_replay_backend
    finally:
        adapter.close()

    # Build only after the primary engine has closed.  Replay is therefore a
    # genuinely fresh CUDA environment, not a reset/continuation of primary.
    replay_backend = replay_backend_factory()
    try:
        replay = replay_recorded_tape(
            replay_backend,
            requests,
            (action_tape,),
        )
        if len(replay.terminal_rows) != 1 or len(replay.observation_fingerprints) != 1:
            raise RuntimeError("fresh replay changed the B=1 cardinality")
        replay_receipt = validate_blocking_replay(
            primary_terminal=primary_terminal_payload,
            replay_terminal=replay.terminal_rows[0],
            primary_observation_fingerprints=primary_fingerprints,
            replay_observation_fingerprints=replay.observation_fingerprints[0],
        )
        replay_exact_once_witnesses = assert_terminal_ledger_exact_once(
            replay_backend,
            replay.terminal_rows,
        )
        replay_receipt["exact_once_second_consumption_rejected"] = True
        replay_receipt["exact_once_negative_witnesses"] = list(
            replay_exact_once_witnesses
        )
    finally:
        replay_backend.close()

    return {
        **common_report,
        "sample_count": 1,
        "counts_as_d32_result": True,
        "closed_loop_planner": True,
        "frozen_action_replay": False,
        "resource": {
            "gpu_created": True,
            "device_ordinal": args.device_ordinal,
            "expected_device_uuid": args.expected_device_uuid,
            "observed_device_uuid": provenance["physical_device_uuid"],
        },
        "provenance": provenance,
        "teacher": {
            "full_reset_request_bound": True,
            "reset_identity": preflight_teacher_identity,
            "metadata": jsonable(teacher_metadata),
        },
        "terminal_ledger": {
            "blocking": True,
            "passed": True,
            "row_count": 1,
            "exact_once_second_consumption_rejected": True,
            "exact_once_negative_witnesses": list(primary_exact_once_witnesses),
        },
        "terminal": primary_terminal_payload,
        "replay": replay_receipt,
        "action_tape": {
            "records": jsonable(action_tape),
            "sha256": action_tape_sha256,
        },
        "trajectory_tape": {
            "observation_fingerprints": list(primary_fingerprints),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = _run(args)
    except BaseException as exc:
        failure = {
            "schema_version": f"{D32_ROW_REPORT_SCHEMA_VERSION}-failure",
            "status": "failed",
            "phase": getattr(args, "phase", None),
            "task_id": TASK_ID,
            "batch_size": 1,
            "sample_count": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        args.output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
