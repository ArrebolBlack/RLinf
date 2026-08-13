#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Freeze the clean current-main T4-sphere candidate repository tuple."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rlinf.envs.dynamic_benchmark.t4_sphere_d32 import (
    BACKEND_ID,
    CANDIDATE_SCHEMA_VERSION,
    EXECUTION_CONTRACT,
    TASK_ID,
    canonical_json_bytes,
    load_candidate_identity,
    validate_repository_tuple,
    validate_scientific_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--se3-source-root", type=Path, required=True)
    parser.add_argument("--mjwarp-source-root", type=Path, required=True)
    parser.add_argument("--rlinf-source-root", type=Path, required=True)
    parser.add_argument("--dynamic-source-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-id",
        default="GPUPLAN0/t4-sphere-valid-d32-v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _checkout(root: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError(f"candidate source checkout is dirty: {root}")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
    }


def _gitlink(root: Path, path: str) -> str:
    fields = _git(root, "ls-tree", "HEAD", "--", path).split()
    if len(fields) < 4 or fields[:2] != ["160000", "commit"]:
        raise RuntimeError(f"candidate source lacks gitlink {path}")
    return fields[2]


def build_candidate_identity(
    *,
    candidate_id: str,
    se3_root: Path,
    mjwarp_root: Path,
    rlinf_root: Path,
    dynamic_root: Path,
) -> dict[str, Any]:
    """Build and self-validate one hash-sealed candidate identity."""

    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate-id must be a non-empty string")
    roots = {
        "se3_wam": se3_root.resolve(strict=True),
        "mujoco_warp": mjwarp_root.resolve(strict=True),
        "rlinf": rlinf_root.resolve(strict=True),
        "dynamic_benchmark": dynamic_root.resolve(strict=True),
    }
    repositories: dict[str, dict[str, str]] = {
        name: _checkout(root) for name, root in roots.items()
    }
    repositories["se3_wam"]["mujoco_warp_gitlink"] = _gitlink(
        roots["se3_wam"], "third_party/mujoco_warp"
    )
    repositories["dynamic_benchmark"]["rlinf_gitlink"] = _gitlink(
        roots["dynamic_benchmark"], "third_party/RLinf"
    )
    if (
        repositories["se3_wam"]["mujoco_warp_gitlink"]
        != repositories["mujoco_warp"]["commit"]
    ):
        raise RuntimeError("SE3-WAM checkout does not pin the supplied MJWarp checkout")
    if (
        repositories["dynamic_benchmark"]["rlinf_gitlink"]
        != repositories["rlinf"]["commit"]
    ):
        raise RuntimeError(
            "Dynamic Benchmark checkout does not pin the supplied RLinf checkout"
        )

    se3_src = str(roots["se3_wam"] / "src")
    sys.path = [se3_src, *[value for value in sys.path if value != se3_src]]
    from se3_wam.benchmark.config import task_config_sha256

    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "task_id": TASK_ID,
        "backend_id": BACKEND_ID,
        "repositories": repositories,
        "task_config_sha256": task_config_sha256(TASK_ID),
        "execution": dict(EXECUTION_CONTRACT),
    }
    payload["candidate_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_candidate_identity(
        candidate_id=args.candidate_id,
        se3_root=args.se3_source_root,
        mjwarp_root=args.mjwarp_source_root,
        rlinf_root=args.rlinf_source_root,
        dynamic_root=args.dynamic_source_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate = load_candidate_identity(args.output)
    validate_repository_tuple(
        candidate,
        se3_root=args.se3_source_root,
        rlinf_root=args.rlinf_source_root,
        dynamic_root=args.dynamic_source_root,
    )
    validate_scientific_contract(candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
