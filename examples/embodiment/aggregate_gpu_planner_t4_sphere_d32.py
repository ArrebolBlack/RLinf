#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Aggregate T4-sphere D32 only after all 32 exact row receipts complete."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rlinf.envs.dynamic_benchmark.t4_sphere_d32 import (
    aggregate_d32_reports,
    load_candidate_identity,
    load_d32_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        required=True,
        help="one completed B=1 D32 row report; pass exactly 32 times",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidate = load_candidate_identity(args.candidate_identity)
    manifest = load_d32_manifest(
        args.manifest,
        candidate=candidate,
        verify_exports=False,
    )
    reports = [
        json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
        for path in args.report
    ]
    aggregate = aggregate_d32_reports(
        candidate=candidate,
        manifest=manifest,
        reports=reports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
