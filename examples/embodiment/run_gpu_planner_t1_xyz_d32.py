#!/usr/bin/env python3
"""Run the exploratory t1_xyz GPU Planner D32 in four cohorts of eight.

Each row gets its own frozen export identity and a fresh CUDA backend through
the existing E0 runner.  The Planner is called from the current observation;
the action tape is retained only for replay audit.  This small supervisor
keeps engineering failures separate from online task outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TASK_ID = "t1_xyz"
MANIFEST_SEED = 20261040


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--replay-policy", choices=("audit", "strict"), default="audit")
    return parser


def _request_payload(request: Any) -> dict[str, Any]:
    return {
        "api_version": request.api_version,
        "episode_id": request.episode_id,
        "task_id": request.task_id,
        "split": request.split.value,
        "seed": request.seed,
        "action_mode": request.action_mode.value,
        "observation_track": request.observation_track.value,
        "object_mode": request.object_mode,
        "reset_mode": request.reset_mode,
        "factors": dict(request.factors),
    }


def _write_manifest(path: Path, rows: tuple[Any, ...]) -> None:
    payload = {
        "schema_version": "gpu-planner-t1-xyz-d32-manifest-v1",
        "task_id": TASK_ID,
        "split": "test_id",
        "manifest_seed": MANIFEST_SEED,
        "attempts_per_task": len(rows),
        "rows": [
            {
                "source_group_id": row.source_group_id,
                "pair_id": row.pair_id,
                "candidate_index": row.candidate_index,
                "pair_member_id": row.pair_member_id,
                "request": _request_payload(row.request),
            }
            for row in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    online = [row for row in results if row.get("status") != "engineering_failure"]
    successes = [row for row in online if bool(row.get("success"))]
    reasons = Counter(str(row.get("termination_reason")) for row in online)
    qualities: dict[str, list[float]] = {}
    for row in online:
        for ledger_row in row.get("terminal_ledger", []):
            quality = ledger_row.get("task_quality") or {}
            for name, component in quality.get("components", {}).items():
                value = component.get("value")
                if isinstance(value, (int, float)):
                    qualities.setdefault(name, []).append(float(value))
    quality_summary = {
        name: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        for name, values in sorted(qualities.items())
    }
    return {
        "schema_version": "gpu-planner-t1-xyz-d32-summary-v1",
        "task_id": TASK_ID,
        "split": "test_id",
        "manifest_seed": MANIFEST_SEED,
        "requested_total": len(results),
        "online_completed": len(online),
        "engineering_failures": len(results) - len(online),
        "online_successes": len(successes),
        "online_failures": len(online) - len(successes),
        "success_rate": (len(successes) / len(online)) if online else None,
        "drop_rate": ((len(online) - len(successes)) / len(online)) if online else None,
        "termination_reasons": dict(sorted(reasons.items())),
        "quality": quality_summary,
        "results": results,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.episodes != 32:
        raise ValueError("D32 requires exactly 32 episodes")
    args.run_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    from se3_wam.benchmark.dataset_manifest import (
        Split,
        make_dataset_candidate_manifest,
    )

    candidate_rows = make_dataset_candidate_manifest(
        split=Split.TEST_ID,
        attempts_per_task=args.episodes,
        manifest_seed=MANIFEST_SEED,
        # t1_xyz is the first member of the registered t1_xyz/t1_so3 pair;
        # generate the complete pair so the factory's pair-integrity gate is
        # exercised, then select the t1_xyz half for this D32 run.
        tasks=("t1_xyz", "t1_so3"),
    )
    rows = tuple(row for row in candidate_rows if row.request.task_id == TASK_ID)
    if len(rows) != args.episodes:
        raise RuntimeError(f"expected {args.episodes} t1_xyz rows, got {len(rows)}")
    manifest_path = args.run_root / "t1_xyz_d32_manifest.json"
    _write_manifest(manifest_path, rows)

    source_root = Path(__file__).resolve().parents[3]
    se3_root = source_root / "SE3-WAM"
    e0_runner = Path(__file__).with_name("run_gpu_planner_t1_xyz_e0.py")
    exporter = se3_root / "scripts" / "gpuenv_export_p0_grasp_seed.py"
    export_root = args.run_root / "exports"
    episode_root = args.run_root / "episodes"
    results: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        cohort = index // 8
        row_root = export_root / f"cohort-{cohort:03d}" / f"row-{index:02d}-{TASK_ID}"
        output_root = episode_root / f"cohort-{cohort:03d}" / f"row-{index:02d}"
        output_path = output_root / "e0.json"
        export_log = output_root / "export.log"
        run_log = output_root / "e0.log"
        output_root.mkdir(parents=True, exist_ok=True)

        # Row 0 is the already completed B=1 online sample and is reused as
        # the first D32 result; rows 1..31 receive fresh registered exports.
        if index == 0:
            existing = Path(
                "/vepfs-mlp2/mlp-public/haoce/yjq/se3wam-a100/runs/GPUPLAN0/t1_xyz/"
                "t1-xyz-d32-v1/cohort-000-row0-final/e0.json"
            )
            if existing.exists():
                results.append({"manifest_index": index, **json.loads(existing.read_text())})
                print(json.dumps({"completed": len(results), "total": 32, "status": "reused_row0"}), flush=True)
                continue

        if not (row_root / "export_report.json").exists():
            export_command = [
                sys.executable,
                str(exporter),
                "--output",
                str(row_root),
                "--row-index",
                str(index),
                "--image-size",
                str(args.image_size),
                "--task-id",
                TASK_ID,
                "--observation-track",
                "hybrid",
            ]
            with export_log.open("w", encoding="utf-8") as handle:
                export_result = subprocess.run(
                    export_command,
                    cwd=se3_root,
                    env=os.environ.copy(),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if export_result.returncode != 0:
                failure = {
                    "manifest_index": index,
                    "status": "engineering_failure",
                    "stage": "export",
                    "returncode": export_result.returncode,
                    "log": str(export_log),
                }
                results.append(failure)
                print(json.dumps({"completed": len(results), "total": 32, **failure}), flush=True)
                continue

        command = [
            sys.executable,
            str(e0_runner),
            "--export-dir",
            str(row_root),
            "--output",
            str(output_path),
            "--tape-output",
            str(output_root / "e0.tape.npz"),
            "--visual-gif",
            str(output_root / "e0.scene-wrist.gif"),
            "--expected-gpu-uuid",
            args.expected_gpu_uuid,
            "--device-ordinal",
            str(args.device_ordinal),
            "--image-size",
            str(args.image_size),
            "--observation-track",
            "hybrid",
            "--replay-policy",
            args.replay_policy,
        ]
        with run_log.open("w", encoding="utf-8") as handle:
            run_result = subprocess.run(
                command,
                cwd=source_root / "RLinf",
                env=os.environ.copy(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if run_result.returncode == 0 and output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            results.append({"manifest_index": index, **result})
            status = result.get("status")
        else:
            failure = {
                "manifest_index": index,
                "status": "engineering_failure",
                "stage": "online_runner",
                "returncode": run_result.returncode,
                "log": str(run_log),
            }
            results.append(failure)
            status = failure["status"]
        summary = _summary(results)
        (args.run_root / "d32_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "completed": len(results),
                    "total": 32,
                    "manifest_index": index,
                    "status": status,
                    "online_successes": summary["online_successes"],
                    "engineering_failures": summary["engineering_failures"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = _summary(results)
    summary["manifest_file"] = str(manifest_path)
    (args.run_root / "d32_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
