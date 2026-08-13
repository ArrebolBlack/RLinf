#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run a sealed ``t1_xyz`` D32 without row reuse or audit-valid outcomes.

The supervisor consumes an already sealed 32-row manifest.  Every row runs
through the strict B=1 result path with its own export-bound ResetRequest.  A
candidate result exists only when all 32 row receipts independently pass
source, quality-v2, terminal-ledger, evidence-export, and fresh replay gates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_strict_contract() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/envs/dynamic_benchmark/t1_xyz_strict_evidence.py"
    )
    name = "_t1_xyz_strict_evidence_contract"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load strict evidence contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_STRICT = _load_strict_contract()
load_frozen_manifest = _STRICT.load_frozen_manifest
summarize_d32_results = _STRICT.summarize_d32_results
validate_repository_tuple = _STRICT.validate_repository_tuple


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--research-source-root", type=Path, required=True)
    parser.add_argument("--se3-source-root", type=Path, required=True)
    parser.add_argument("--mjwarp-source-root", type=Path, required=True)
    parser.add_argument("--rlinf-source-root", type=Path, required=True)
    parser.add_argument("--dynamic-source-root", type=Path, required=True)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    return parser


def _summary(
    results: list[dict[str, Any]],
    manifest: Any | None = None,
) -> dict[str, Any]:
    """Aggregate only receipts accepted by the strict evidence contract."""

    return summarize_d32_results(results, manifest=manifest)


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = _parser().parse_args()
    if args.image_size < 64:
        raise ValueError("--image-size must be at least 64")
    if args.device_ordinal < 0:
        raise ValueError("--device-ordinal must be nonnegative")
    if not args.expected_gpu_uuid.strip():
        raise ValueError("--expected-gpu-uuid must be non-empty")
    if Path(__file__).resolve().parents[2] != args.rlinf_source_root.resolve(
        strict=True
    ):
        raise RuntimeError("D32 supervisor is not executing from the sealed RLinf root")

    # Validate the whole cohort, every export, and every source checkout before
    # the first row process is allowed to construct a CUDA backend.
    manifest = load_frozen_manifest(
        args.manifest,
        expected_phase="d32",
        verify_exports=True,
    )
    validate_repository_tuple(
        manifest,
        research_root=args.research_source_root,
        se3_root=args.se3_source_root,
        mjwarp_root=args.mjwarp_source_root,
        rlinf_root=args.rlinf_source_root,
        dynamic_root=args.dynamic_source_root,
    )
    args.run_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.run_root / "d32_summary.json"
    if summary_path.exists():
        raise FileExistsError("refusing to reuse an existing D32 summary")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    e0_runner = Path(__file__).with_name("run_gpu_planner_t1_xyz_e0.py")
    results: list[dict[str, Any]] = []

    for index in range(32):
        row_root = args.run_root / "rows" / f"row-{index:02d}"
        output_path = row_root / "result.json"
        tape_path = row_root / "result.tape.npz"
        visual_path = row_root / "result.scene-wrist.gif"
        log_path = row_root / "runner.log"
        if any(
            path.exists() for path in (output_path, tape_path, visual_path, log_path)
        ):
            raise FileExistsError(f"refusing to reuse D32 row {index} artifacts")
        row_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(e0_runner),
            "--manifest",
            str(manifest.path),
            "--row-index",
            str(index),
            "--phase",
            "d32",
            "--output",
            str(output_path),
            "--tape-output",
            str(tape_path),
            "--visual-gif",
            str(visual_path),
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
        with log_path.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=args.rlinf_source_root,
                env=os.environ.copy(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not output_path.is_file():
            results.append(
                {
                    "manifest_index": index,
                    "status": "engineering_failure",
                    "stage": "strict_row_runner",
                    "returncode": completed.returncode,
                    "log": str(log_path),
                }
            )
            summary = _summary(results, manifest)
            _write_summary(summary_path, summary)
            raise RuntimeError(f"strict D32 row {index} failed; cohort is incomplete")
        result = json.loads(output_path.read_text(encoding="utf-8"))
        results.append(result)
        summary = _summary(results, manifest)
        _write_summary(summary_path, summary)
        print(
            json.dumps(
                {
                    "received_rows": summary["received_rows"],
                    "valid_completed": summary["valid_completed"],
                    "complete_cohort": summary["complete_cohort"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = _summary(results, manifest)
    _write_summary(summary_path, summary)
    if summary["complete_cohort"] is not True:
        raise RuntimeError("strict D32 cohort did not produce 32/32 valid receipts")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
