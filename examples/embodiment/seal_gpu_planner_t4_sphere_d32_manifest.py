#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Seal 32 independently exported T4-sphere rows into the v2 D32 manifest.

Create every ``rowNNNN`` directory first with the current SE3 exporter and the
explicit flags ``--manifest-seed 20261040 --split test_id
--observation-track state``.  This command refuses historical row-zero reuse,
missing rows, stale task configs, or any per-field ResetRequest mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rlinf.envs.dynamic_benchmark.t4_sphere_d32 import (
    BACKEND_ID,
    D32_EPISODES,
    D32_MANIFEST_SCHEMA_VERSION,
    D32_MANIFEST_SEED,
    D32_SEED_SET_SHA256,
    EXECUTION_CONTRACT,
    TASK_ID,
    assert_seed_disjointness,
    canonical_json_bytes,
    load_candidate_identity,
    load_d32_manifest,
    request_identity,
    sha256_file,
    teacher_reset_identity,
    validate_repository_tuple,
    validate_request_identity,
    validate_scientific_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-identity", type=Path, required=True)
    parser.add_argument("--se3-source-root", type=Path, required=True)
    parser.add_argument("--rlinf-source-root", type=Path, required=True)
    parser.add_argument("--dynamic-source-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--mc64-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _row_payload(
    source: Any, *, export_dir: Path, output_parent: Path
) -> dict[str, Any]:
    request = source.request
    relative_export = Path(os.path.relpath(export_dir, output_parent)).as_posix()
    return {
        "action_mode": request.action_mode.value,
        "api_version": request.api_version,
        "candidate_index": int(source.candidate_index),
        "episode_id": request.episode_id,
        "export_dir": relative_export,
        "export_sha256": sha256_file(export_dir / "SHA256SUMS"),
        "factors": dict(request.factors),
        "object_mode": request.object_mode,
        "observation_track": request.observation_track.value,
        "pair_id": source.pair_id,
        "pair_member_id": source.pair_member_id,
        "request_json_sha256": sha256_file(export_dir / "request.json"),
        "reset_mode": request.reset_mode,
        "seed": int(request.seed),
        "source_group_id": source.source_group_id,
        "split": request.split.value,
        "task_id": request.task_id,
    }


def build_manifest(
    *,
    candidate_path: Path,
    se3_root: Path,
    rlinf_root: Path,
    dynamic_root: Path,
    export_root: Path,
    output_path: Path,
    mc64_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build a sealed D32 manifest from exactly matching current-source exports."""

    candidate = load_candidate_identity(candidate_path)
    se3_src = str(se3_root.resolve(strict=True) / "src")
    sys.path = [se3_src, *[value for value in sys.path if value != se3_src]]
    validate_repository_tuple(
        candidate,
        se3_root=se3_root,
        rlinf_root=rlinf_root,
        dynamic_root=dynamic_root,
    )
    validate_scientific_contract(candidate)
    from se3_wam.benchmark.api import Split
    from se3_wam.benchmark.contracts import ObservationTrack
    from se3_wam.benchmark.dataset_manifest import (
        DATASET_MANIFEST_VERSION,
        make_dataset_candidate_manifest,
    )
    from se3_wam.benchmark.gpu_native.p0_grasp_engine import load_p0_grasp_artifacts
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    source_rows = make_dataset_candidate_manifest(
        split=Split.TEST_ID,
        attempts_per_task=D32_EPISODES,
        manifest_seed=D32_MANIFEST_SEED,
        tasks=(TASK_ID,),
    )
    if len(source_rows) != D32_EPISODES:
        raise RuntimeError(
            "current SE3 source did not produce the frozen 32-row surface"
        )
    source_rows = tuple(
        replace(
            source,
            request=replace(
                source.request,
                observation_track=ObservationTrack.STATE,
            ),
        )
        for source in source_rows
    )
    export_root = export_root.resolve(strict=True)
    output_parent = output_path.resolve().parent
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        if int(source.candidate_index) != index:
            raise RuntimeError("current SE3 candidate indices are not contiguous 0..31")
        export_dir = export_root / f"row{index:04d}"
        if not export_dir.is_dir():
            raise FileNotFoundError(export_dir)
        for required_name in ("SHA256SUMS", "request.json", "export_report.json"):
            if not (export_dir / required_name).is_file():
                raise RuntimeError(f"row {index} export lacks {required_name}")
        artifacts = load_p0_grasp_artifacts(export_dir)
        expected_row = _row_payload(
            source,
            export_dir=export_dir,
            output_parent=output_parent,
        )
        validate_request_identity(
            artifacts.reset_request,
            expected_row,
            actual_task_config_sha256=artifacts.config_sha256,
            candidate=candidate,
        )
        teacher, teacher_metadata = make_privileged_teacher(
            TASK_ID,
            request=artifacts.reset_request,
        )
        teacher.reset()
        expected_row["teacher_reset_identity"] = teacher_reset_identity(
            teacher_metadata
        )
        request_json = json.loads(
            (export_dir / "request.json").read_text(encoding="utf-8")
        )
        expected_request = request_identity(source.request)
        for name, expected in expected_request.items():
            if name == "api_version":
                continue
            if request_json.get(name) != expected:
                raise RuntimeError(f"row {index} request.json field {name} drifted")
        export_report = json.loads(
            (export_dir / "export_report.json").read_text(encoding="utf-8")
        )
        if export_report.get("task_config_sha256") != candidate.task_config_sha256:
            raise RuntimeError(f"row {index} export task config is stale")
        rows.append(expected_row)

    payload: dict[str, Any] = {
        "schema_version": D32_MANIFEST_SCHEMA_VERSION,
        "phase_id": "GPUPLAN0/t4-sphere-valid-d32-v2",
        "task_id": TASK_ID,
        "backend_id": BACKEND_ID,
        "split": "test_id",
        "manifest_seed": D32_MANIFEST_SEED,
        "dataset_manifest_version": DATASET_MANIFEST_VERSION,
        "episode_count": D32_EPISODES,
        "row_execution_batch_size": 1,
        "candidate_indices": [row["candidate_index"] for row in rows],
        "episode_ids": [row["episode_id"] for row in rows],
        "seed_set_sha256": D32_SEED_SET_SHA256,
        "candidate_sha256": candidate.candidate_sha256,
        "task_config_sha256": candidate.task_config_sha256,
        "observation_track": EXECUTION_CONTRACT["observation_track"],
        "action_mode": EXECUTION_CONTRACT["action_mode"],
        "rows": rows,
    }
    if mc64_manifest_path is not None:
        other = json.loads(
            mc64_manifest_path.resolve(strict=True).read_text(encoding="utf-8")
        )
        payload["mc64_seed_non_overlap"] = assert_seed_disjointness(
            rows,
            other,
            other_name="historical_blocked_mc64",
        )
    payload["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_manifest(
        candidate_path=args.candidate_identity,
        se3_root=args.se3_source_root,
        rlinf_root=args.rlinf_source_root,
        dynamic_root=args.dynamic_source_root,
        export_root=args.export_root,
        output_path=args.output,
        mc64_manifest_path=args.mc64_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate = load_candidate_identity(args.candidate_identity)
    load_d32_manifest(args.output, candidate=candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
