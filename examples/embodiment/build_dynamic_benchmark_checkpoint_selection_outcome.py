#!/usr/bin/env python3
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

"""Build a portable, authoritative trainer checkpoint-selection outcome receipt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.embodiment.dynamic_benchmark_checkpoint_admission import (
    build_checkpoint_selection_outcome_payload,
    checkpoint_selection_outcome_versioned_path,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-rlinf-source-root", type=Path, required=True)
    parser.add_argument("--verifier-rlinf-source-root", type=Path, required=True)
    parser.add_argument("--evaluator-rlinf-source-root", type=Path, required=True)
    parser.add_argument("--expected-policy-rlinf-commit", required=True)
    parser.add_argument("--expected-verifier-rlinf-commit", required=True)
    parser.add_argument("--expected-evaluator-rlinf-commit", required=True)
    parser.add_argument("--expected-benchmark-commit", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-checkpoint-selection-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-metrics-sha256", required=True)
    parser.add_argument("--expected-initial-policy-sha256", required=True)
    parser.add_argument("--expected-best-policy-sha256")
    return parser.parse_args(argv)


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint outcome {path}")
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        # Keep the staging basename short: the two full source commits already make
        # the canonical version directory long, and Windows still commonly enforces
        # MAX_PATH for temporary-file creation.
        prefix=".o-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        linked = True
    finally:
        temporary.unlink(missing_ok=True)
    if not linked:
        raise RuntimeError("checkpoint outcome was not published")


def _prepare_output_parent(run_root: Path, output: Path) -> None:
    """Create only the canonical non-symlink provenance directory chain."""

    relative_parent = output.parent.relative_to(run_root)
    current = run_root
    for part in relative_parent.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        if (
            current.is_symlink()
            or not current.is_dir()
            or current.resolve(strict=True) != current
        ):
            raise ValueError(
                "checkpoint outcome provenance path contains a symlink or "
                "non-canonical directory"
            )


def write_checkpoint_selection_outcome(
    *,
    run_root: Path,
    output_path: Path,
    policy_rlinf_source_root: Path,
    verifier_rlinf_source_root: Path,
    evaluator_rlinf_source_root: Path,
    expected_policy_rlinf_commit: str,
    expected_verifier_rlinf_commit: str,
    expected_evaluator_rlinf_commit: str,
    expected_benchmark_commit: str,
    expected_summary_sha256: str,
    expected_checkpoint_selection_sha256: str,
    expected_config_sha256: str,
    expected_metrics_sha256: str,
    expected_initial_policy_sha256: str,
    expected_best_policy_sha256: str | None = None,
) -> Path:
    raw_root = Path(run_root)
    if raw_root.is_symlink():
        raise ValueError("trainer run root must not be a symlink")
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    resolved_root = raw_root.resolve(strict=True)
    raw_output = Path(output_path)
    output = Path(os.path.abspath(raw_output))
    if raw_output != output or ".." in raw_output.parts:
        raise ValueError(
            "checkpoint outcome output must be an absolute canonical path without "
            "traversal"
        )
    expected_output = checkpoint_selection_outcome_versioned_path(
        resolved_root,
        expected_verifier_rlinf_commit,
        expected_evaluator_rlinf_commit,
    )
    if output != expected_output:
        raise ValueError(
            "checkpoint outcome output must be the exact source-versioned path"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint outcome {output}")
    payload = build_checkpoint_selection_outcome_payload(
        run_root=resolved_root,
        policy_rlinf_source_root=policy_rlinf_source_root,
        verifier_rlinf_source_root=verifier_rlinf_source_root,
        evaluator_rlinf_source_root=evaluator_rlinf_source_root,
        expected_policy_rlinf_commit=expected_policy_rlinf_commit,
        expected_verifier_rlinf_commit=expected_verifier_rlinf_commit,
        expected_evaluator_rlinf_commit=expected_evaluator_rlinf_commit,
        expected_benchmark_commit=expected_benchmark_commit,
        expected_summary_sha256=expected_summary_sha256,
        expected_checkpoint_selection_sha256=(expected_checkpoint_selection_sha256),
        expected_config_sha256=expected_config_sha256,
        expected_metrics_sha256=expected_metrics_sha256,
        expected_initial_policy_sha256=expected_initial_policy_sha256,
        expected_best_policy_sha256=expected_best_policy_sha256,
    )
    _prepare_output_parent(resolved_root, output)
    _exclusive_json(output, payload)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = write_checkpoint_selection_outcome(
        run_root=args.run_root,
        output_path=args.output,
        policy_rlinf_source_root=args.policy_rlinf_source_root,
        verifier_rlinf_source_root=args.verifier_rlinf_source_root,
        evaluator_rlinf_source_root=args.evaluator_rlinf_source_root,
        expected_policy_rlinf_commit=args.expected_policy_rlinf_commit,
        expected_verifier_rlinf_commit=args.expected_verifier_rlinf_commit,
        expected_evaluator_rlinf_commit=args.expected_evaluator_rlinf_commit,
        expected_benchmark_commit=args.expected_benchmark_commit,
        expected_summary_sha256=args.expected_summary_sha256,
        expected_checkpoint_selection_sha256=(
            args.expected_checkpoint_selection_sha256
        ),
        expected_config_sha256=args.expected_config_sha256,
        expected_metrics_sha256=args.expected_metrics_sha256,
        expected_initial_policy_sha256=args.expected_initial_policy_sha256,
        expected_best_policy_sha256=args.expected_best_policy_sha256,
    )
    print(output)


if __name__ == "__main__":
    main()
