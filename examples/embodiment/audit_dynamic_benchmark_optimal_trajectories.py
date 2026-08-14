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

"""Independently audit a best-known Dynamic Benchmark trajectory export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

try:
    from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
        validate_compatibility_evidence,
    )
except ModuleNotFoundError:
    from build_dynamic_benchmark_rld2_evidence import validate_compatibility_evidence

CANDIDATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-candidates-v0.1"
CANDIDATE_SCHEMA_V2 = "rlinf-dynamic-benchmark-optimal-candidates-v0.2"
CANDIDATE_SCHEMAS = (CANDIDATE_SCHEMA, CANDIDATE_SCHEMA_V2)
EVALUATOR_IDENTITY_SCHEMA = "rlinf-dynamic-benchmark-quality-evaluator-identity-v0.1"
CALIBRATION_EVIDENCE_SCHEMA = (
    "rlinf-dynamic-benchmark-planner-calibration-evidence-v0.1"
)
CANDIDATE_RELEASE_SCHEMA = "rlinf-dynamic-benchmark-rld2-candidate-release-v0.2"
EXPORT_SCHEMA = "rlinf-dynamic-benchmark-optimal-export-v0.1"
ATTEMPT_SCHEMA = "rlinf-dynamic-benchmark-optimal-attempt-v0.3"
HISTORICAL_ATTEMPT_SCHEMAS = frozenset(
    {
        "rlinf-dynamic-benchmark-optimal-attempt-v0.1",
        "rlinf-dynamic-benchmark-optimal-attempt-v0.2",
    }
)
LEGACY_ATTEMPT_SCHEMA = "rlinf-dynamic-benchmark-optimal-attempt-v0.1"
AUDIT_SCHEMA = "rlinf-dynamic-benchmark-optimal-audit-v0.1"
STATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-export-state-v0.1"
PROGRESS_SCHEMA = "rlinf-dynamic-benchmark-optimal-progress-v0.1"
RENDER_PARITY_SKIP_SCHEMA = "rlinf-dynamic-benchmark-render-parity-skip-v0.1"
LEGACY_SELECTION_MODE = "legacy-lexicographic"
PLANNER_PARETO_SELECTION_MODE = "planner-pareto"
SELECTION_MODES = (LEGACY_SELECTION_MODE, PLANNER_PARETO_SELECTION_MODE)
FIRST_ELIGIBLE_SEARCH_MODE = "first-eligible"
FULL_POOL_SEARCH_MODE = "full-pool"
CANDIDATE_SEARCH_MODES = (FIRST_ELIGIBLE_SEARCH_MODE, FULL_POOL_SEARCH_MODE)
PLANNER_DOMINANCE_SCHEMA = "rlinf-dynamic-benchmark-planner-dominance-v0.1"
QUALITY_V2_THRESHOLDS_SCHEMA = "se3-wam-trajectory-quality-v2-thresholds-v0.3"
QUALITY_V2_SUMMARY_SCHEMA = "se3-wam-trajectory-quality-v2"
QUALITY_V2_GATE_SCHEMA = "se3-wam-trajectory-quality-v2-gate-v0.1"
QUALITY_V2_DOMINANCE_SCHEMA = "rlinf-dynamic-benchmark-quality-v2-dominance-v0.1"
QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA = (
    "rld2-qa-planner-calibration-wave-receipt-v0.1"
)
QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES = 20
QUALITY_V2_MINIMUM_SUCCESSFUL_EPISODES = 8
SELECTION_CONTRACT = (
    "success,safety,trajectory_completion,return,-control_steps,-action_l2_sum"
)
PLANNER_PARETO_SELECTION_CONTRACT = (
    "success,safety,t5-replan-causal-timing,quality-v2-absolute-gate,"
    "planner-pareto(trajectory_completion,"
    "task_quality.*,-completion_time_s,-control_steps,"
    "quality-v2.threshold-checks,-t5-impact-to-applied-correction-s,"
    "-action_l2_sum);return=diagnostic-only"
)
T5_ACTION_HISTORY_SCHEMA = "se3-wam-t5-issued-applied-action-history-v0.1"
T5_ACTION_VALUE_SEMANTIC_LABELS = (
    "arm_translation_x",
    "arm_translation_y",
    "arm_translation_z",
    "arm_rotation_x",
    "arm_rotation_y",
    "arm_rotation_z",
    "gripper",
)
T5_TIMING_VALUE_SEMANTIC_LABELS = (
    "impact_end_time_s",
    "first_contact_time_s",
    "control_hz",
)
T5_TIMING_COUNT_SEMANTIC_LABELS = (
    "expected_issued_action_count",
    "expected_action_delay_steps",
)
T5_CAUSAL_TAPE_INVENTORY = frozenset(
    {
        "t5_action_history_schema",
        "action_value_semantic_labels",
        "issued_action_values",
        "issued_policy_step",
        "issued_time_s",
        "scheduled_apply_policy_step",
        "scheduled_apply_time_s",
        "applied_action_values",
        "applied_issue_policy_step",
        "actual_apply_policy_step",
        "actual_apply_time_s",
        "t5_timing_value_semantic_labels",
        "t5_timing_values",
        "t5_timing_count_semantic_labels",
        "t5_timing_counts",
    }
)
EXACT_TASKS = (
    "p0_grasp",
    "t1_xyz",
    "t1_belt",
    "t1_so3",
    "t1_occ",
    "t2_trans",
    "t2_se3",
    "t3_phase",
    "t3_full",
    "t4_sphere",
    "t4_sphere_tabletop",
    "t4_slider",
    "t4_can",
    "t5_replan",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-dataset-card-sha256", required=True)
    parser.add_argument("--expected-checksums-sha256", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-quality-v2-thresholds-sha256", required=True)
    parser.add_argument("--auditor-commit", required=True)
    parser.add_argument(
        "--qv4-disabled-nonblocking",
        action="store_true",
        help=(
            "Audit a release that intentionally did not export Qv4. The winner "
            "must remain successful and label-valid, with null Qv4 validation and "
            "behavior-cloning eligibility disabled; Qv2 remains fully audited."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _full_commit(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("payload_sha256", None)
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} is not a mapping")
            rows.append(value)
    return rows


def _safe_dataset_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe dataset-relative path {relative!r}")
    target = root.joinpath(*pure.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"dataset-relative path escapes the root: {relative!r}")
    return target


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _quality_v2_calibration_receipt_binding(
    thresholds: Mapping[str, Any],
) -> dict[str, str]:
    wave = thresholds.get("calibration_wave_receipt")
    if not isinstance(wave, Mapping) or wave.get("binding_status") != "bound":
        raise ValueError(
            "quality-v2 threshold has no bound calibration receipt artifact"
        )
    relative = wave.get("relative_path")
    if not isinstance(relative, str):
        raise ValueError("quality-v2 calibration receipt relative path is missing")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or pure.as_posix() != relative
        or "\\" in relative
        or pure.parts[0] != "provenance"
        or pure.name != "wave_receipt.json"
    ):
        raise ValueError(f"unsafe quality-v2 calibration receipt path: {relative!r}")
    file_sha256 = _expected_sha256(
        wave.get("file_sha256"),
        "quality-v2 calibration receipt file SHA-256",
    )
    payload_sha256 = _expected_sha256(
        wave.get("payload_sha256"),
        "quality-v2 calibration receipt payload SHA-256",
    )
    legacy_sha256 = _expected_sha256(
        wave.get("sha256"),
        "quality-v2 calibration receipt compatibility SHA-256",
    )
    if file_sha256 != payload_sha256 or file_sha256 != legacy_sha256:
        raise ValueError(
            "canonical quality-v2 calibration receipt file/payload identities disagree"
        )
    return {
        "relative_path": relative,
        "file_sha256": file_sha256,
        "payload_sha256": payload_sha256,
    }


def _audit_quality_v2_calibration_receipt_artifact(
    root: Path,
    thresholds: Mapping[str, Any],
    *,
    expected_benchmark_commit: str | None = None,
) -> dict[str, str]:
    """Reopen the dataset-local receipt instead of trusting threshold metadata."""

    from examples.embodiment.dynamic_benchmark_calibration_projection import (
        validate_projection_artifact,
    )

    binding = _quality_v2_calibration_receipt_binding(thresholds)
    receipt_path = _safe_dataset_path(root, binding["relative_path"])
    projection = validate_projection_artifact(
        thresholds,
        receipt_path,
        expected_sha256=None,
        expected_benchmark_commit=expected_benchmark_commit,
    )
    if projection is not None:
        return {
            "relative_path": str(binding["relative_path"]),
            "file_sha256": str(projection["file_sha256"]),
            "payload_sha256": str(projection["payload_sha256"]),
        }
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(
            "quality-v2 calibration receipt artifact is missing or symlinked"
        )
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != binding["file_sha256"]:
        raise ValueError("quality-v2 calibration receipt file SHA-256 mismatch")
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("quality-v2 calibration receipt is not UTF-8 JSON") from error
    if not isinstance(receipt, Mapping):
        raise TypeError("quality-v2 calibration receipt must be a mapping")
    canonical_bytes = _canonical_json_bytes(receipt)
    if receipt_bytes != canonical_bytes:
        raise ValueError("quality-v2 calibration receipt is not canonical JSON")
    if hashlib.sha256(canonical_bytes).hexdigest() != binding["payload_sha256"]:
        raise ValueError("quality-v2 calibration receipt payload SHA-256 mismatch")

    wave = thresholds["calibration_wave_receipt"]
    assert isinstance(wave, Mapping)
    expected_top_level = {
        "schema_version": QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA,
        "scientific_partition": "metric_calibration",
        "transport_split": "validation",
        "task_count": len(EXACT_TASKS),
        "episodes_per_task": QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES,
        "total_reset_count": len(EXACT_TASKS) * QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES,
        "task_order": list(EXACT_TASKS),
    }
    for key, value in expected_top_level.items():
        if receipt.get(key) != value or wave.get(key) != value:
            raise ValueError(f"quality-v2 calibration receipt/threshold {key} mismatch")
    for key in (
        "manifest_seed",
        "wave_contract_sha256",
        "predeclaration_receipt_sha256",
        "source_identity",
        "disjointness",
    ):
        if wave.get(key) != receipt.get(key):
            raise ValueError(f"quality-v2 calibration receipt/threshold {key} mismatch")
    source_identity = receipt.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("quality-v2 calibration receipt source identity is missing")
    receipt_benchmark_commit = _full_commit(
        "quality-v2 calibration receipt benchmark commit",
        source_identity.get("benchmark_commit"),
    )
    if (
        expected_benchmark_commit is not None
        and receipt_benchmark_commit
        != _full_commit(
            "expected quality-v2 calibration benchmark commit",
            expected_benchmark_commit,
        )
    ):
        raise ValueError(
            "quality-v2 calibration receipt benchmark commit differs from the "
            "authenticated evaluator benchmark commit"
        )
    raw_receipt_tasks = receipt.get("tasks")
    raw_binding_tasks = wave.get("tasks")
    if (
        not isinstance(raw_receipt_tasks, list)
        or not isinstance(raw_binding_tasks, list)
        or len(raw_receipt_tasks) != len(EXACT_TASKS)
        or len(raw_binding_tasks) != len(EXACT_TASKS)
    ):
        raise ValueError("quality-v2 calibration receipt task inventory is not exact14")
    identity_keys = (
        "task_contract_sha256",
        "task_receipt_sha256",
        "task_config_sha256",
        "task_quality_schema_version",
        "task_quality_schema_sha256",
        "reset_manifest_relative_path",
        "reset_manifest_sha256",
        "reset_identity_set_sha256",
        "reset_row_set_sha256",
        "evaluation_relative_path",
        "evaluation_sha256",
        "evaluation_payload_sha256",
    )
    for ordinal, (task_id, receipt_task, binding_task) in enumerate(
        zip(EXACT_TASKS, raw_receipt_tasks, raw_binding_tasks, strict=True)
    ):
        if not isinstance(receipt_task, Mapping) or not isinstance(
            binding_task, Mapping
        ):
            raise TypeError("quality-v2 calibration receipt task row must be a mapping")
        if (
            receipt_task.get("ordinal") != ordinal
            or binding_task.get("ordinal") != ordinal
            or receipt_task.get("task_id") != task_id
            or binding_task.get("task_id") != task_id
            or receipt_task.get("reset_count") != QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES
            or binding_task.get("reset_identity_count")
            != QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES
        ):
            raise ValueError("quality-v2 calibration receipt task order/count mismatch")
        for key in identity_keys:
            if binding_task.get(key) != receipt_task.get(key):
                raise ValueError(
                    f"quality-v2 calibration task {task_id} {key} mismatch"
                )
        for key in (
            "task_contract_sha256",
            "task_receipt_sha256",
            "task_config_sha256",
            "task_quality_schema_sha256",
            "reset_manifest_sha256",
            "reset_identity_set_sha256",
            "reset_row_set_sha256",
            "evaluation_sha256",
            "evaluation_payload_sha256",
        ):
            _expected_sha256(
                receipt_task.get(key),
                f"quality-v2 calibration task {task_id} {key}",
            )
    return binding


def _verify_root_checksums(root: Path, expected_sha256: str) -> int:
    checksum_path = root / "SHA256SUMS"
    if _sha256(checksum_path) != expected_sha256:
        raise ValueError("root SHA256SUMS identity mismatch")
    declared: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"SHA256SUMS:{line_number} is malformed")
        digest = _expected_sha256(parts[0], f"SHA256SUMS:{line_number} digest")
        relative = parts[1]
        if relative in declared:
            raise ValueError(f"SHA256SUMS repeats {relative!r}")
        path = _safe_dataset_path(root, relative)
        if not path.is_file() or path.name == "SHA256SUMS":
            raise ValueError(f"SHA256SUMS target is missing or forbidden: {relative!r}")
        if _sha256(path) != digest:
            raise ValueError(f"dataset file checksum mismatch: {relative!r}")
        declared[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and ".staging" not in path.parts
    }
    if set(declared) != actual:
        missing = sorted(actual - set(declared))
        extra = sorted(set(declared) - actual)
        raise ValueError(
            f"root checksum inventory mismatch: missing={missing}, extra={extra}"
        )
    return len(declared)


def _candidate_rows(
    payload: Mapping[str, Any],
    *,
    card: Mapping[str, Any],
) -> list[dict[str, Any]]:
    schema_version = payload.get("schema_version")
    if schema_version not in CANDIDATE_SCHEMAS:
        raise ValueError("candidate manifest schema mismatch")
    if payload.get("task") != card.get("task"):
        raise ValueError("candidate manifest task mismatch")
    source = card.get("source_identity")
    if not isinstance(source, dict):
        raise ValueError("dataset card source identity is missing")
    if schema_version == CANDIDATE_SCHEMA:
        if set(source) != {
            "evaluator_rlinf_commit",
            "policy_rlinf_commit",
            "benchmark_commit",
        }:
            raise ValueError("legacy source identity inventory mismatch")
        if payload.get("rlinf_commit") != source.get("policy_rlinf_commit"):
            raise ValueError("candidate manifest RLinf source mismatch")
        if payload.get("benchmark_commit") != source.get("benchmark_commit"):
            raise ValueError("candidate manifest benchmark source mismatch")
    else:
        if set(payload) != {
            "schema_version",
            "task",
            "evaluator_identity",
            "policy_rlinf_commits",
            "policy_benchmark_commits",
            "candidates",
            "planner_dominance",
        }:
            raise ValueError("candidate schema v0.2 top-level inventory mismatch")
        if set(source) != {
            "evaluator_rlinf_commit",
            "evaluator_benchmark_commit",
            "policy_rlinf_commits",
            "policy_benchmark_commits",
        }:
            raise ValueError("candidate schema v0.2 source identity inventory mismatch")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or len(rows) < int(card["max_k"]):
        raise ValueError("candidate manifest is shorter than max_k")
    ids = []
    planner_count = 0
    policy_source_commits: set[str] = set()
    policy_benchmark_commits: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - {
            "candidate_id",
            "kind",
            "policy_path",
            "policy_sha256",
            "stochastic",
            "exploration_seed_offset",
            "residual_scale",
            "provenance",
        }:
            raise ValueError(f"candidate {index} is not a mapping")
        candidate_id = row.get("candidate_id")
        kind = row.get("kind")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"candidate {index} has no stable ID")
        if kind not in {"planner", "policy"}:
            raise ValueError(f"candidate {candidate_id!r} has invalid kind")
        if not isinstance(row.get("stochastic", False), bool):
            raise ValueError(
                f"candidate {candidate_id!r} stochastic flag is not boolean"
            )
        seed_offset = row.get("exploration_seed_offset", 0)
        if (
            isinstance(seed_offset, bool)
            or not isinstance(seed_offset, int)
            or not 0 <= seed_offset < 2**31
        ):
            raise ValueError(f"candidate {candidate_id!r} exploration seed is invalid")
        planner_count += int(kind == "planner")
        provenance = row.get("provenance")
        if provenance is not None:
            if not isinstance(provenance, Mapping) or not provenance:
                raise ValueError(f"candidate {candidate_id!r} provenance is invalid")
            try:
                json.dumps(provenance, allow_nan=False, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"candidate {candidate_id!r} provenance is not canonical-JSON-safe"
                ) from error
        if schema_version == CANDIDATE_SCHEMA_V2 and provenance is None:
            raise ValueError("candidate schema v0.2 has a provenance gap")
        if kind == "planner":
            if (
                row.get("policy_path") is not None
                or row.get("policy_sha256") is not None
            ):
                raise ValueError("planner candidate unexpectedly names a policy file")
            if schema_version == CANDIDATE_SCHEMA_V2:
                assert isinstance(provenance, Mapping)
                planner_source = provenance.get("source")
                planner_runtime = provenance.get("runtime")
                planner_benchmark = provenance.get("benchmark")
                if (
                    not isinstance(planner_source, Mapping)
                    or planner_source.get("rlinf_commit")
                    != source["evaluator_rlinf_commit"]
                    or not isinstance(planner_runtime, Mapping)
                    or planner_runtime.get("evaluator_rlinf_commit")
                    != source["evaluator_rlinf_commit"]
                    or not isinstance(planner_benchmark, Mapping)
                    or planner_benchmark.get("commit")
                    != source["evaluator_benchmark_commit"]
                ):
                    raise ValueError("planner provenance is not evaluator-bound")
        else:
            if not isinstance(row.get("policy_path"), str) or not row["policy_path"]:
                raise ValueError(f"policy candidate {candidate_id!r} has no path")
            policy_sha256 = _expected_sha256(
                row.get("policy_sha256"),
                f"candidate {candidate_id} policy SHA-256",
            )
            if provenance is not None:
                source_row = provenance.get("source")
                checkpoint = provenance.get("checkpoint")
                benchmark = provenance.get("benchmark")
                if not isinstance(source_row, Mapping):
                    raise ValueError(
                        f"candidate {candidate_id!r} provenance has no source"
                    )
                policy_source_commits.add(
                    _full_commit(
                        f"candidate {candidate_id!r} source RLinf commit",
                        source_row.get("rlinf_commit"),
                    )
                )
                if (
                    not isinstance(checkpoint, Mapping)
                    or checkpoint.get("path") != row["policy_path"]
                    or checkpoint.get("sha256") != policy_sha256
                ):
                    raise ValueError(
                        f"candidate {candidate_id!r} checkpoint provenance mismatch"
                    )
                if not isinstance(benchmark, Mapping):
                    raise ValueError(
                        f"candidate {candidate_id!r} benchmark provenance is missing"
                    )
                policy_benchmark_commits.add(
                    _full_commit(
                        f"candidate {candidate_id!r} benchmark commit",
                        benchmark.get("commit"),
                    )
                )
            else:
                if schema_version == CANDIDATE_SCHEMA_V2:
                    raise ValueError(
                        "candidate schema v0.2 policy authority is missing"
                    )
        ids.append(candidate_id)
    if len(ids) != len(set(ids)) or planner_count != 1:
        raise ValueError("candidate IDs must be unique and contain exactly one planner")
    if rows[0].get("kind") != "planner":
        raise ValueError("the frozen candidate pool must put its planner at index zero")
    if schema_version == CANDIDATE_SCHEMA_V2:
        expected_rlinf = sorted(policy_source_commits)
        expected_benchmark = sorted(policy_benchmark_commits)
        if (
            payload.get("policy_rlinf_commits") != expected_rlinf
            or source.get("policy_rlinf_commits") != expected_rlinf
        ):
            raise ValueError("policy RLinf commit inventory does not recompute")
        if (
            payload.get("policy_benchmark_commits") != expected_benchmark
            or source.get("policy_benchmark_commits") != expected_benchmark
        ):
            raise ValueError("policy benchmark commit inventory does not recompute")
        evaluator_identity = payload.get("evaluator_identity")
        if not isinstance(evaluator_identity, Mapping) or (
            evaluator_identity.get("evaluator_rlinf_commit")
            != source["evaluator_rlinf_commit"]
            or evaluator_identity.get("evaluator_benchmark_commit")
            != source["evaluator_benchmark_commit"]
        ):
            raise ValueError("candidate evaluator identity differs from dataset source")
    return rows


def _quality_score(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(bool(record["success"])),
        float(not bool(record["safety_failure"])),
        float(record["trajectory_completion"]),
        float(record["return"]),
        -float(record["control_steps"]),
        -float(record["action_l2_sum"]),
    )


def _eligible(record: Mapping[str, Any]) -> bool:
    replay = record.get("replay_validation")
    for key in ("success", "safety_failure", "finite_and_bounded"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"attempt {key} must be boolean")
    if not isinstance(replay, Mapping) or not isinstance(replay.get("passed"), bool):
        raise ValueError("attempt replay-validation passed flag must be boolean")
    quality_gate = record.get("quality_v2_gate")
    if quality_gate is None and record.get("schema_version") == LEGACY_ATTEMPT_SCHEMA:
        quality_passed = True
    else:
        if not isinstance(quality_gate, Mapping) or not isinstance(
            quality_gate.get("passed"), bool
        ):
            raise ValueError("attempt quality-v2 gate passed flag must be boolean")
        quality_passed = bool(quality_gate["passed"])
    causal_timing_passed = True
    if record.get("schema_version") == ATTEMPT_SCHEMA:
        issued_equals_applied = record.get("issued_equals_applied")
        if not isinstance(issued_equals_applied, bool):
            raise ValueError(
                "attempt issued_equals_applied declaration must be boolean"
            )
        if record.get("task_id") == "t5_replan":
            if issued_equals_applied:
                raise ValueError(
                    "T5 Replan must preserve distinct issued/applied histories"
                )
            raw_causal_gate = record.get("t5_replan_causal_timing_passed")
            if not isinstance(raw_causal_gate, bool):
                raise ValueError("T5 Replan causal-timing gate must be boolean")
            causal_timing_passed = raw_causal_gate
        elif not issued_equals_applied:
            raise ValueError("non-T5 attempt must declare issued_equals_applied=true")
    return bool(
        record.get("success")
        and not record.get("safety_failure")
        and record.get("finite_and_bounded")
        and replay.get("passed")
        and quality_passed
        and causal_timing_passed
    )


def _candidate_search_mode(payload: Mapping[str, Any]) -> str:
    mode = payload.get("candidate_search_mode", FIRST_ELIGIBLE_SEARCH_MODE)
    if mode not in CANDIDATE_SEARCH_MODES:
        raise ValueError(f"unsupported candidate search mode {mode!r}")
    return str(mode)


def _selection_contract(selection_mode: str) -> str:
    if selection_mode == LEGACY_SELECTION_MODE:
        return SELECTION_CONTRACT
    if selection_mode == PLANNER_PARETO_SELECTION_MODE:
        return PLANNER_PARETO_SELECTION_CONTRACT
    raise ValueError(f"unsupported selection mode {selection_mode!r}")


def _selection_mode(payload: Mapping[str, Any]) -> str:
    mode = payload.get("selection_mode", LEGACY_SELECTION_MODE)
    if mode not in SELECTION_MODES:
        raise ValueError(f"unsupported selection mode {mode!r}")
    return str(mode)


def _metric_contract(
    value: Any,
    *,
    metric_name: str,
    direction: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"planner-dominance metric {metric_name!r} must be a mapping")
    common = {
        "direction",
        "max_observed_replay_drift",
        "scientific_resolution",
    }
    expected = (
        common | {"numeric_floor_absolute", "numeric_floor_relative"}
        if metric_name == "action_l2_sum"
        else common | {"numeric_floor"}
    )
    if set(value) != expected or value.get("direction") != direction:
        raise ValueError(f"planner-dominance metric {metric_name!r} contract mismatch")
    drift = _finite_number(
        value["max_observed_replay_drift"],
        f"planner-dominance metric {metric_name!r} drift",
    )
    resolution = _finite_number(
        value["scientific_resolution"],
        f"planner-dominance metric {metric_name!r} scientific resolution",
    )
    if drift < 0.0:
        raise ValueError(f"planner-dominance metric {metric_name!r} drift is invalid")
    if resolution <= 0.0:
        raise ValueError(
            f"planner-dominance metric {metric_name!r} resolution is invalid"
        )
    normalized = {
        "direction": direction,
        "max_observed_replay_drift": drift,
        "scientific_resolution": resolution,
    }
    if metric_name == "action_l2_sum":
        absolute = _finite_number(
            value["numeric_floor_absolute"],
            "action_l2_sum absolute numeric floor",
        )
        relative = _finite_number(
            value["numeric_floor_relative"],
            "action_l2_sum relative numeric floor",
        )
        if absolute != 1.0e-6 or relative != 1.0e-6:
            raise ValueError("action_l2_sum numeric floor contract mismatch")
        normalized.update(
            numeric_floor_absolute=absolute,
            numeric_floor_relative=relative,
        )
    else:
        floor = _finite_number(
            value["numeric_floor"],
            f"planner-dominance metric {metric_name!r} numeric floor",
        )
        expected_floor = 0.0 if metric_name == "control_steps" else 1.0e-6
        if floor != expected_floor:
            raise ValueError(f"planner-dominance metric {metric_name!r} floor mismatch")
        if metric_name == "control_steps" and resolution < 1.0:
            raise ValueError("control_steps scientific resolution must be at least one")
        if metric_name == "completion_time_s" and resolution != 0.002:
            raise ValueError(
                "completion_time_s scientific resolution must be one 0.002 s physics step"
            )
        normalized["numeric_floor"] = floor
    return normalized


def _planner_dominance_contract(
    candidate_payload: Mapping[str, Any],
    *,
    task: str,
    selection_mode: str,
) -> dict[str, Any] | None:
    raw = candidate_payload.get("planner_dominance")
    if selection_mode == LEGACY_SELECTION_MODE:
        if raw is not None:
            raise ValueError(
                "legacy candidate manifest unexpectedly declares planner dominance"
            )
        return None
    if selection_mode != PLANNER_PARETO_SELECTION_MODE or not isinstance(raw, Mapping):
        raise ValueError("planner-pareto candidate manifest has no dominance contract")
    if (
        set(raw)
        != {
            "schema_version",
            "task",
            "backend_id",
            "quality_schema",
            "calibration",
            "metrics",
            "tie_break_order",
        }
        or raw.get("schema_version") != PLANNER_DOMINANCE_SCHEMA
    ):
        raise ValueError("planner-dominance schema or field inventory mismatch")
    if raw.get("task") != task:
        raise ValueError("planner-dominance task mismatch")
    backend_id = raw.get("backend_id")
    if (
        not isinstance(backend_id, str)
        or not backend_id
        or backend_id.strip() != backend_id
    ):
        raise ValueError("planner-dominance backend identity is missing")
    quality_schema = raw.get("quality_schema")
    if not isinstance(quality_schema, Mapping) or set(quality_schema) != {
        "schema_version",
        "task_id",
        "task_config_sha256",
        "components",
        "schema_sha256",
    }:
        raise ValueError("planner-dominance quality schema inventory is invalid")
    quality_schema_version = quality_schema.get("schema_version")
    if not isinstance(quality_schema_version, str) or not quality_schema_version:
        raise ValueError("planner-dominance quality schema version is missing")
    if quality_schema.get("task_id") != task:
        raise ValueError("planner-dominance quality schema task identity mismatch")
    task_config_sha256 = _expected_sha256(
        quality_schema.get("task_config_sha256"),
        "planner-dominance task config SHA-256",
    )
    quality_schema_sha256 = _expected_sha256(
        quality_schema.get("schema_sha256"),
        "planner-dominance quality schema SHA-256",
    )
    components = quality_schema.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("planner-dominance quality components are missing")
    normalized_components: list[dict[str, Any]] = []
    component_names: set[str] = set()
    for metadata in components:
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "name",
            "direction",
            "unit",
            "scientific_resolution",
            "reducer",
            "source",
            "description",
        }:
            raise ValueError("planner-dominance quality component metadata is invalid")
        name = metadata.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or "." in name
            or name in component_names
        ):
            raise ValueError("planner-dominance quality component name is invalid")
        direction = metadata.get("direction")
        unit = metadata.get("unit")
        reducer = metadata.get("reducer")
        source = metadata.get("source")
        description = metadata.get("description")
        resolution = _finite_number(
            metadata.get("scientific_resolution"),
            f"quality component {name!r} scientific resolution",
        )
        if direction not in {"minimize", "maximize"}:
            raise ValueError(f"quality component {name!r} direction is invalid")
        if not isinstance(unit, str) or not unit:
            raise ValueError(f"quality component {name!r} unit is missing")
        if reducer not in {"minimum", "maximum", "terminal"}:
            raise ValueError(f"quality component {name!r} reducer is invalid")
        if not isinstance(source, str) or not source or source.strip() != source:
            raise ValueError(f"quality component {name!r} source is missing")
        if (
            not isinstance(description, str)
            or not description
            or description.strip() != description
        ):
            raise ValueError(f"quality component {name!r} description is missing")
        if resolution <= 0.0:
            raise ValueError(f"quality component {name!r} resolution is invalid")
        component_names.add(name)
        # Preserve the exact upstream canonical row.  The ordered list and the
        # source/description strings are part of the schema digest.
        normalized_components.append(dict(metadata))
    normalized_quality_schema = {
        "schema_version": quality_schema_version,
        "task_id": task,
        "task_config_sha256": task_config_sha256,
        "components": normalized_components,
        "schema_sha256": quality_schema_sha256,
    }
    recomputed_quality_schema_sha256 = _payload_sha256(
        {
            "schema_version": quality_schema_version,
            "task_id": task,
            "task_config_sha256": task_config_sha256,
            "components": list(components),
        }
    )
    if quality_schema_sha256 != recomputed_quality_schema_sha256:
        raise ValueError("planner-dominance quality schema SHA-256 does not recompute")
    calibration = raw.get("calibration")
    if not isinstance(calibration, Mapping) or set(calibration) != {
        "replay_count",
        "reset_episode_id",
        "reset_manifest_sha256",
        "evidence_path",
        "evidence_sha256",
    }:
        raise ValueError("planner-dominance calibration inventory mismatch")
    replay_count = calibration.get("replay_count")
    reset_episode_id = calibration.get("reset_episode_id")
    if (
        isinstance(replay_count, bool)
        or not isinstance(replay_count, int)
        or replay_count < 3
    ):
        raise ValueError(
            "planner-dominance calibration requires at least three replays"
        )
    if not isinstance(reset_episode_id, str) or not reset_episode_id:
        raise ValueError("planner-dominance calibration reset identity is missing")
    evidence_path = calibration.get("evidence_path")
    if (
        not isinstance(evidence_path, str)
        or not evidence_path
        or evidence_path.strip() != evidence_path
    ):
        raise ValueError("planner-dominance calibration evidence path is missing")
    normalized_calibration = {
        "replay_count": replay_count,
        "reset_episode_id": reset_episode_id,
        "reset_manifest_sha256": _expected_sha256(
            calibration.get("reset_manifest_sha256"),
            "planner-dominance reset manifest SHA-256",
        ),
        "evidence_path": evidence_path,
        "evidence_sha256": _expected_sha256(
            calibration.get("evidence_sha256"),
            "planner-dominance evidence SHA-256",
        ),
    }
    metrics = raw.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "trajectory_completion",
        "task_quality",
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    }:
        raise ValueError("planner-dominance metric inventory mismatch")
    quality_metrics = metrics.get("task_quality")
    component_index = {row["name"]: row for row in normalized_components}
    if not isinstance(quality_metrics, Mapping) or set(quality_metrics) != set(
        component_index
    ):
        raise ValueError("planner-dominance quality mapping is incomplete")
    normalized_metrics = {
        "trajectory_completion": _metric_contract(
            metrics["trajectory_completion"],
            metric_name="trajectory_completion",
            direction="max",
        ),
        "task_quality": {
            name: _metric_contract(
                quality_metrics[name],
                metric_name=f"task_quality.{name}",
                direction=(
                    "max" if component_index[name]["direction"] == "maximize" else "min"
                ),
            )
            for name in component_index
        },
        "completion_time_s": _metric_contract(
            metrics["completion_time_s"],
            metric_name="completion_time_s",
            direction="min",
        ),
        "control_steps": _metric_contract(
            metrics["control_steps"],
            metric_name="control_steps",
            direction="min",
        ),
        "action_l2_sum": _metric_contract(
            metrics["action_l2_sum"],
            metric_name="action_l2_sum",
            direction="min",
        ),
    }
    for name in component_index:
        if (
            normalized_metrics["task_quality"][name]["scientific_resolution"]
            != component_index[name]["scientific_resolution"]
        ):
            raise ValueError(
                f"quality component {name!r} calibration resolution differs from schema"
            )
    metric_keys = [
        "trajectory_completion",
        *(f"task_quality.{name}" for name in component_index),
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    ]
    tie_break_order = raw.get("tie_break_order")
    if (
        not isinstance(tie_break_order, list)
        or len(tie_break_order) != len(metric_keys)
        or set(tie_break_order) != set(metric_keys)
    ):
        raise ValueError("planner-dominance tie-break order is incomplete")
    normalized = {
        "schema_version": PLANNER_DOMINANCE_SCHEMA,
        "task": task,
        "backend_id": backend_id,
        "quality_schema": normalized_quality_schema,
        "calibration": normalized_calibration,
        "metrics": normalized_metrics,
        "tie_break_order": list(tie_break_order),
    }
    normalized["payload_sha256"] = _payload_sha256(normalized)
    return normalized


def _metric_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        "trajectory_completion",
        *(
            f"task_quality.{component['name']}"
            for component in contract["quality_schema"]["components"]
        ),
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    )


def _metric_spec(contract: Mapping[str, Any], metric_name: str) -> Mapping[str, Any]:
    if metric_name.startswith("task_quality."):
        return contract["metrics"]["task_quality"][metric_name.split(".", 1)[1]]
    return contract["metrics"][metric_name]


def _metric_value(record: Mapping[str, Any], metric_name: str) -> float:
    if metric_name.startswith("task_quality."):
        component = metric_name.split(".", 1)[1]
        summary = record.get("task_quality")
        values = summary.get("components") if isinstance(summary, Mapping) else None
        component_row = values.get(component) if isinstance(values, Mapping) else None
        if not isinstance(component_row, Mapping) or "value" not in component_row:
            raise ValueError(
                f"task quality mapping gap: missing component {component!r}"
            )
        value = _finite_number(
            component_row["value"],
            f"task quality component {component!r} value",
        )
    else:
        raw_value = record.get(metric_name)
        if metric_name == "control_steps":
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 1
            ):
                raise ValueError(
                    "planner-dominance control_steps must be a positive integer"
                )
            value = float(raw_value)
        else:
            value = _finite_number(
                raw_value,
                f"planner-dominance metric {metric_name!r}",
            )
            if metric_name == "trajectory_completion" and not 0.0 <= value <= 1.0:
                raise ValueError("trajectory_completion must be in [0, 1]")
            if metric_name in {"completion_time_s", "action_l2_sum"} and value < 0.0:
                raise ValueError(
                    f"planner-dominance metric {metric_name!r} is negative"
                )
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number, not bool or string")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validate_attempt_quality(
    record: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    summary = record.get("task_quality")
    schema = contract["quality_schema"]
    if not isinstance(summary, Mapping):
        raise ValueError(
            "task quality mapping gap: attempt has no task_quality summary"
        )
    if set(summary) != {
        "schema_version",
        "episode_id",
        "task_id",
        "evaluator_backend_id",
        "schema_sha256",
        "physics_sample_count",
        "terminal",
        "components",
        "summary_sha256",
    }:
        raise ValueError("task quality summary field inventory mismatch")
    if (
        summary.get("schema_version") != schema["schema_version"]
        or summary.get("task_id") != contract["task"]
        or summary.get("schema_sha256") != schema["schema_sha256"]
        or summary.get("episode_id") != record.get("episode_id")
        or summary.get("evaluator_backend_id") != contract["backend_id"]
    ):
        raise ValueError("task quality summary identity mismatch")
    if summary.get("terminal") is not True:
        raise ValueError("task quality summary must be terminal")
    summary_sha256 = _expected_sha256(
        summary.get("summary_sha256"),
        "task quality summary SHA-256",
    )
    summary_payload = dict(summary)
    summary_payload.pop("summary_sha256")
    if summary_sha256 != _payload_sha256(summary_payload):
        raise ValueError("task quality summary SHA-256 does not recompute")
    sample_count = summary.get("physics_sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("task quality summary physics sample count is invalid")
    values = summary.get("components")
    schema_components = schema["components"]
    expected_names = [component["name"] for component in schema_components]
    expected = set(expected_names)
    if not isinstance(values, Mapping) or set(values) != expected:
        missing = sorted(
            expected - set(values) if isinstance(values, Mapping) else expected
        )
        extra = sorted(set(values) - expected if isinstance(values, Mapping) else set())
        raise ValueError(f"task quality mapping gap: missing={missing}, extra={extra}")
    for frozen_schema in schema_components:
        name = frozen_schema["name"]
        component = values[name]
        frozen = {
            key: frozen_schema[key]
            for key in ("direction", "unit", "scientific_resolution", "reducer")
        }
        if not isinstance(component, Mapping) or set(component) != {
            "value",
            "direction",
            "unit",
            "scientific_resolution",
            "reducer",
        }:
            raise ValueError(f"task quality component {name!r} inventory mismatch")
        _finite_number(
            component["scientific_resolution"],
            f"task quality component {name!r} scientific resolution",
        )
        if any(component.get(key) != frozen[key] for key in frozen):
            raise ValueError(f"task quality component {name!r} metadata mismatch")
        _metric_value(record, f"task_quality.{name}")


def _metric_thresholds(
    metric_name: str,
    reference_value: float,
    spec: Mapping[str, Any],
) -> tuple[float, float]:
    if metric_name == "action_l2_sum":
        floor = max(
            float(spec["numeric_floor_absolute"]),
            float(spec["numeric_floor_relative"]) * abs(reference_value),
        )
    else:
        floor = float(spec["numeric_floor"])
    epsilon = max(floor, 2.0 * float(spec["max_observed_replay_drift"]))
    strict_margin = max(float(spec["scientific_resolution"]), 2.0 * epsilon)
    if metric_name == "control_steps":
        strict_margin = max(1.0, strict_margin)
    return epsilon, strict_margin


def _quality_v2_dominance_contract(
    payload: Mapping[str, Any],
    *,
    task: str,
    thresholds_sha256: str,
    require_formal_freeze: bool = True,
) -> dict[str, Any]:
    """Independently derive paired Qv2 dimensions from the frozen checks."""

    threshold_sha256 = _expected_sha256(
        thresholds_sha256,
        "quality-v2 threshold SHA-256",
    )
    if payload.get("schema_version") != QUALITY_V2_THRESHOLDS_SCHEMA:
        raise ValueError(
            "planner-pareto requires the frozen quality-v2 threshold schema v0.3"
        )
    formal_freeze_eligible = payload.get("formal_freeze_eligible")
    if not isinstance(formal_freeze_eligible, bool):
        raise ValueError(
            "quality-v2 threshold formal-freeze eligibility must be boolean"
        )
    if require_formal_freeze and not formal_freeze_eligible:
        raise ValueError(
            "quality-v2 threshold contract is not eligible for formal freeze"
        )
    if require_formal_freeze:
        minimum_attempted = payload.get("minimum_attempted_episodes")
        minimum_successful = payload.get("minimum_successful_episodes")
        wave = payload.get("calibration_wave_receipt")
        if (
            payload.get("calibration_status") != "frozen"
            or isinstance(minimum_attempted, bool)
            or not isinstance(minimum_attempted, int)
            or minimum_attempted < QUALITY_V2_MINIMUM_ATTEMPTED_EPISODES
            or isinstance(minimum_successful, bool)
            or not isinstance(minimum_successful, int)
            or minimum_successful < QUALITY_V2_MINIMUM_SUCCESSFUL_EPISODES
            or not isinstance(wave, Mapping)
            or wave.get("binding_status") != "bound"
            or wave.get("schema_version") != QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA
            or wave.get("scientific_partition") != "metric_calibration"
            or wave.get("task_count") != 14
            or wave.get("episodes_per_task") != 20
            or wave.get("total_reset_count") != 280
        ):
            raise ValueError(
                "quality-v2 threshold contract has invalid formal calibration provenance"
            )
        _expected_sha256(
            wave.get("sha256"),
            "quality-v2 calibration wave receipt SHA-256",
        )
    tasks = payload.get("tasks")
    task_contract = tasks.get(task) if isinstance(tasks, Mapping) else None
    if not isinstance(task_contract, Mapping):
        raise ValueError(f"quality-v2 threshold contract has no task {task!r}")
    if require_formal_freeze:
        provenance = task_contract.get("provenance")
        attempted = (
            provenance.get("attempted_episode_count")
            if isinstance(provenance, Mapping)
            else None
        )
        successful = (
            provenance.get("successful_episode_count")
            if isinstance(provenance, Mapping)
            else None
        )
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("formal_freeze_eligible") is not True
            or isinstance(attempted, bool)
            or not isinstance(attempted, int)
            or attempted < minimum_attempted
            or isinstance(successful, bool)
            or not isinstance(successful, int)
            or successful < minimum_successful
        ):
            raise ValueError(
                f"quality-v2 threshold task {task!r} has invalid formal calibration provenance"
            )
    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal_exporter,
    )

    expected_specs, orientation_mode, jaw_axis_mode = (
        optimal_exporter._quality_v2_expected_check_specs(task_contract)
    )
    expected_by_identity = {
        (spec["phase"], spec["metric"]): spec for spec in expected_specs
    }
    raw_checks = task_contract.get("checks")
    if not isinstance(raw_checks, list) or len(raw_checks) != len(expected_specs):
        raise ValueError(
            f"quality-v2 checks for {task!r} must contain exactly "
            f"{len(expected_specs)} task-derived entries"
        )

    paired_fields = (
        "paired_nonworse_absolute_tolerance",
        "paired_nonworse_relative_tolerance",
        "paired_strict_improvement_absolute",
        "paired_strict_improvement_relative",
    )
    required_fields = {
        "phase",
        "metric",
        "max",
        "direction",
        "paired_comparison_family",
        *paired_fields,
    }
    metrics: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index, raw_check in enumerate(raw_checks):
        if not isinstance(raw_check, Mapping) or not required_fields.issubset(
            raw_check
        ):
            raise ValueError(
                f"quality-v2 check {index} is missing frozen paired-comparison metadata"
            )
        phase = raw_check.get("phase")
        metric = raw_check.get("metric")
        family = raw_check.get("paired_comparison_family")
        if (
            not isinstance(phase, str)
            or not phase
            or phase.strip() != phase
            or "." in phase
        ):
            raise ValueError(f"quality-v2 check {index} phase is invalid")
        if (
            not isinstance(metric, str)
            or not metric
            or metric.strip() != metric
            or any(not part for part in metric.split("."))
        ):
            raise ValueError(f"quality-v2 check {index} metric is invalid")
        if raw_check.get("direction") != "minimize":
            raise ValueError(
                f"quality-v2 check {phase}.{metric} must use direction='minimize'"
            )
        if not isinstance(family, str) or not family or family.strip() != family:
            raise ValueError(
                f"quality-v2 check {phase}.{metric} comparison family is invalid"
            )
        identity = (phase, metric)
        if identity in identities:
            raise ValueError(f"duplicate quality-v2 check {phase}.{metric}")
        identities.add(identity)
        expected_spec = expected_by_identity.get(identity)
        if expected_spec is None:
            raise ValueError(
                "quality-v2 v0.3 task-derived check inventory mismatch: "
                f"unexpected={identity!r}"
            )
        expected_family = expected_spec["paired_comparison_family"]
        if family != expected_family:
            raise ValueError(
                f"quality-v2 check {phase}.{metric} comparison family "
                f"must be {expected_family!r}"
            )
        maximum = _finite_number(
            raw_check.get("max"),
            f"quality-v2 check {phase}.{metric} maximum",
        )
        normalized_paired = {
            name: _finite_number(
                raw_check.get(name),
                f"quality-v2 check {phase}.{metric} {name}",
            )
            for name in paired_fields
        }
        if maximum < 0.0 or any(value < 0.0 for value in normalized_paired.values()):
            raise ValueError(
                f"quality-v2 check {phase}.{metric} thresholds must be non-negative"
            )
        if (
            normalized_paired["paired_strict_improvement_absolute"] == 0.0
            and normalized_paired["paired_strict_improvement_relative"] == 0.0
        ):
            raise ValueError(
                f"quality-v2 check {phase}.{metric} has no strict-improvement resolution"
            )
        metrics.append(
            {
                "name": f"quality_v2.{phase}.{metric}",
                "key": expected_spec["key"],
                "group": expected_spec["group"],
                "phase": phase,
                "metric": metric,
                "maximum": maximum,
                "direction": "minimize",
                "paired_comparison_family": expected_family,
                **normalized_paired,
            }
        )

    expected_identities = set(expected_by_identity)
    if identities != expected_identities:
        raise ValueError(
            "quality-v2 v0.3 task-derived check inventory mismatch: "
            f"missing={sorted(expected_identities - identities)}, "
            f"extra={sorted(identities - expected_identities)}"
        )
    normalized = {
        "schema_version": QUALITY_V2_DOMINANCE_SCHEMA,
        "threshold_schema_version": QUALITY_V2_THRESHOLDS_SCHEMA,
        "threshold_sha256": threshold_sha256,
        "formal_freeze_eligible": formal_freeze_eligible,
        "task": task,
        "orientation_mode": orientation_mode,
        "jaw_axis_mode": jaw_axis_mode,
        "metrics": metrics,
    }
    normalized["payload_sha256"] = _payload_sha256(normalized)
    return normalized


def _quality_v2_metric_value(
    record: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> float:
    summary = record.get("quality_v2")
    if not isinstance(summary, Mapping):
        raise ValueError("quality-v2 mapping gap: attempt has no quality_v2 summary")
    phase = str(spec["phase"])
    if phase == "full_episode":
        value: Any = summary
    else:
        phases = summary.get("phases")
        value = phases.get(phase) if isinstance(phases, Mapping) else None
        if not isinstance(value, Mapping):
            raise ValueError(
                f"quality-v2 mapping gap: attempt is missing phase {phase!r}"
            )
    metric = str(spec["metric"])
    for part in metric.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(
                f"quality-v2 mapping gap: attempt is missing metric {phase}.{metric}"
            )
        value = value[part]
    result = _finite_number(value, f"quality-v2 metric {phase}.{metric}")
    if result < 0.0:
        raise ValueError(f"quality-v2 metric {phase}.{metric} must be non-negative")
    return result


def _validate_quality_v2_attempt(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, float]:
    if record.get("schema_version") != ATTEMPT_SCHEMA:
        raise ValueError("planner-pareto requires attempt schema v0.3")
    if record.get("task_id") != contract.get("task"):
        raise ValueError("quality-v2 attempt task identity mismatch")
    summary = record.get("quality_v2")
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version") != QUALITY_V2_SUMMARY_SCHEMA
    ):
        raise ValueError("quality-v2 summary schema mismatch")
    summary_sha256 = _expected_sha256(
        record.get("quality_v2_sha256"),
        "quality-v2 summary SHA-256",
    )
    if summary_sha256 != _payload_sha256(summary):
        raise ValueError("quality-v2 summary SHA-256 does not recompute")

    gate = record.get("quality_v2_gate")
    expected_gate_keys = {
        "schema_version",
        "contract_schema_version",
        "contract_sha256",
        "task_id",
        "passed",
        "checks",
    }
    if not isinstance(gate, Mapping) or set(gate) != expected_gate_keys:
        raise ValueError("quality-v2 gate field inventory mismatch")
    if (
        gate.get("schema_version") != QUALITY_V2_GATE_SCHEMA
        or gate.get("contract_schema_version")
        != contract.get("threshold_schema_version")
        or gate.get("contract_sha256") != contract.get("threshold_sha256")
        or gate.get("task_id") != contract.get("task")
        or gate.get("passed") is not True
    ):
        raise ValueError("quality-v2 gate identity or absolute decision mismatch")
    raw_gate_checks = gate.get("checks")
    metric_specs = contract.get("metrics")
    if (
        not isinstance(raw_gate_checks, list)
        or not isinstance(metric_specs, list)
        or len(raw_gate_checks) != len(metric_specs)
    ):
        raise ValueError("quality-v2 gate check inventory mismatch")

    values: dict[str, float] = {}
    expected_check_keys = {"metric", "phase", "actual", "max", "passed"}
    for raw_gate_check, spec in zip(raw_gate_checks, metric_specs, strict=True):
        if (
            not isinstance(raw_gate_check, Mapping)
            or set(raw_gate_check) != expected_check_keys
        ):
            raise ValueError("quality-v2 gate check field inventory mismatch")
        actual = _quality_v2_metric_value(record, spec)
        gate_actual = _finite_number(
            raw_gate_check.get("actual"),
            f"quality-v2 gate actual {spec['name']}",
        )
        gate_maximum = _finite_number(
            raw_gate_check.get("max"),
            f"quality-v2 gate maximum {spec['name']}",
        )
        if (
            raw_gate_check.get("phase") != spec["phase"]
            or raw_gate_check.get("metric") != spec["metric"]
            or gate_actual != actual
            or gate_maximum != spec["maximum"]
            or raw_gate_check.get("passed") is not True
            or actual > float(spec["maximum"])
        ):
            raise ValueError(f"quality-v2 gate check {spec['name']} does not recompute")
        values[str(spec["name"])] = actual
    return values


def _t5_causal_latency(record: Mapping[str, Any]) -> float | None:
    """Return the required T5 causal latency metric; non-T5 tasks have none."""

    if record.get("task_id") != "t5_replan":
        if (
            record.get("impact_end_to_first_qualifying_applied_correction_s")
            is not None
        ):
            raise ValueError("non-T5 attempt cannot declare a T5 causal latency")
        return None
    if record.get("t5_replan_causal_timing_passed") is not True:
        raise ValueError("T5 planner comparison requires a passing causal-timing gate")
    value = _finite_number(
        record.get("impact_end_to_first_qualifying_applied_correction_s"),
        "T5 impact-end-to-applied-correction latency",
    )
    if value < 0.0:
        raise ValueError("T5 causal correction latency must be nonnegative")
    return value


def _quality_v2_metric_thresholds(
    reference_value: float,
    spec: Mapping[str, Any],
) -> tuple[float, float]:
    tolerance = max(
        float(spec["paired_nonworse_absolute_tolerance"]),
        float(spec["paired_nonworse_relative_tolerance"]) * abs(reference_value),
    )
    strict_margin = max(
        float(spec["paired_strict_improvement_absolute"]),
        float(spec["paired_strict_improvement_relative"]) * abs(reference_value),
        2.0 * tolerance,
    )
    return tolerance, strict_margin


def _planner_pareto_dominates(
    record: Mapping[str, Any],
    reference: Mapping[str, Any],
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
) -> bool:
    _validate_attempt_quality(record, contract)
    _validate_attempt_quality(reference, contract)
    quality_v2_values = _validate_quality_v2_attempt(record, quality_v2_contract)
    reference_quality_v2_values = _validate_quality_v2_attempt(
        reference, quality_v2_contract
    )
    strictly_better = False
    for metric_name in _metric_names(contract):
        candidate_value = _metric_value(record, metric_name)
        reference_value = _metric_value(reference, metric_name)
        spec = _metric_spec(contract, metric_name)
        epsilon, strict_margin = _metric_thresholds(metric_name, reference_value, spec)
        if spec["direction"] == "max":
            if candidate_value < reference_value - epsilon:
                return False
            strictly_better |= candidate_value > reference_value + strict_margin
        else:
            if candidate_value > reference_value + epsilon:
                return False
            strictly_better |= candidate_value < reference_value - strict_margin
    for spec in quality_v2_contract["metrics"]:
        metric_name = str(spec["name"])
        candidate_value = quality_v2_values[metric_name]
        reference_value = reference_quality_v2_values[metric_name]
        tolerance, strict_margin = _quality_v2_metric_thresholds(
            reference_value,
            spec,
        )
        if candidate_value > reference_value + tolerance:
            return False
        strictly_better |= candidate_value < reference_value - strict_margin
    candidate_causal_latency = _t5_causal_latency(record)
    reference_causal_latency = _t5_causal_latency(reference)
    if candidate_causal_latency is not None:
        assert reference_causal_latency is not None
        if candidate_causal_latency > reference_causal_latency + 1.0e-9:
            return False
        strictly_better |= candidate_causal_latency < reference_causal_latency - 1.0e-9
    return strictly_better


def _frontier(
    records: list[dict[str, Any]],
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if not any(
            other is not record
            and _planner_pareto_dominates(
                other,
                record,
                contract,
                quality_v2_contract,
            )
            for other in records
        )
    ]


def _pareto_tie_key(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
) -> tuple[float, ...]:
    values = []
    quality_v2_values = _validate_quality_v2_attempt(record, quality_v2_contract)
    quality_v2_tie_values = [
        -quality_v2_values[str(spec["name"])] for spec in quality_v2_contract["metrics"]
    ]
    for metric_name in contract["tie_break_order"]:
        if metric_name == "action_l2_sum":
            # Independently reproduce task utility -> duration -> Qv2
            # smoothness/path -> aggregate control-effort ordering.
            values.extend(quality_v2_tie_values)
            causal_latency = _t5_causal_latency(record)
            if causal_latency is not None:
                values.append(-causal_latency)
        value = _metric_value(record, metric_name)
        values.append(
            value
            if _metric_spec(contract, metric_name)["direction"] == "max"
            else -value
        )
    values.append(-float(int(record["candidate_index"])))
    return tuple(values)


def _selected(
    records: list[dict[str, Any]],
    *,
    selection_mode: str = LEGACY_SELECTION_MODE,
    planner_dominance: Mapping[str, Any] | None = None,
    quality_v2_dominance: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    eligible = [record for record in records if _eligible(record)]
    if not eligible:
        return None
    if selection_mode == LEGACY_SELECTION_MODE:
        if planner_dominance is not None or quality_v2_dominance is not None:
            raise ValueError(
                "legacy selection unexpectedly declares dominance contracts"
            )
        return max(
            eligible,
            key=lambda record: (
                _quality_score(record),
                -int(record["candidate_index"]),
            ),
        )
    elif selection_mode == PLANNER_PARETO_SELECTION_MODE:
        if planner_dominance is None or quality_v2_dominance is None:
            raise ValueError(
                "planner-pareto selection requires frozen task and quality-v2 dominance contracts"
            )
        planner = next(
            (record for record in records if int(record["candidate_index"]) == 0),
            None,
        )
        if planner is None:
            raise ValueError("planner-pareto selection requires candidate index zero")
        eligible_rl = [
            record for record in eligible if int(record["candidate_index"]) != 0
        ]
        for record in eligible_rl:
            _validate_attempt_quality(record, planner_dominance)
            _validate_quality_v2_attempt(record, quality_v2_dominance)
        if not _eligible(planner):
            selectable = eligible_rl
        else:
            _validate_attempt_quality(planner, planner_dominance)
            _validate_quality_v2_attempt(planner, quality_v2_dominance)
            selectable = [
                record
                for record in eligible_rl
                if _planner_pareto_dominates(
                    record,
                    planner,
                    planner_dominance,
                    quality_v2_dominance,
                )
            ]
            if not selectable:
                return planner
    else:
        raise ValueError(f"unsupported selection mode {selection_mode!r}")
    if not selectable:
        return None
    return max(
        _frontier(selectable, planner_dominance, quality_v2_dominance),
        key=lambda record: _pareto_tie_key(
            record,
            planner_dominance,
            quality_v2_dominance,
        ),
    )


def _selection_result(
    records: list[dict[str, Any]],
    winner: Mapping[str, Any] | None,
    *,
    selection_mode: str,
) -> dict[str, Any]:
    """Independently reproduce the exporter decision provenance."""

    planner = next(
        (record for record in records if int(record["candidate_index"]) == 0),
        None,
    )
    if planner is None:
        raise ValueError("selection result requires candidate index zero")
    if selection_mode == LEGACY_SELECTION_MODE:
        source_kind = "legacy" if winner is not None else "rejected"
    elif winner is None:
        source_kind = "rejected"
    elif int(winner["candidate_index"]) == 0:
        source_kind = "planner_fallback"
    else:
        source_kind = "expert_dominant"
    return {
        "source_kind": source_kind,
        "planner_eligible": _eligible(planner),
        "winner_candidate_id": None if winner is None else winner["candidate_id"],
        "winner_candidate_index": None
        if winner is None
        else int(winner["candidate_index"]),
    }


def _planner_metric_relations(
    record: Mapping[str, Any],
    planner: Mapping[str, Any],
    contract: Mapping[str, Any],
    quality_v2_contract: Mapping[str, Any],
) -> dict[str, str]:
    """Classify each accepted expert metric against its eligible planner."""

    _validate_attempt_quality(record, contract)
    _validate_attempt_quality(planner, contract)
    quality_v2_values = _validate_quality_v2_attempt(record, quality_v2_contract)
    planner_quality_v2_values = _validate_quality_v2_attempt(
        planner, quality_v2_contract
    )
    relations: dict[str, str] = {}
    for metric_name in _metric_names(contract):
        value = _metric_value(record, metric_name)
        reference = _metric_value(planner, metric_name)
        spec = _metric_spec(contract, metric_name)
        epsilon, strict_margin = _metric_thresholds(metric_name, reference, spec)
        if spec["direction"] == "max":
            regressed = value < reference - epsilon
            improved = value > reference + strict_margin
        else:
            regressed = value > reference + epsilon
            improved = value < reference - strict_margin
        if regressed:
            relations[metric_name] = "regressed"
        elif improved:
            relations[metric_name] = "strictly_improved"
        else:
            relations[metric_name] = "non_worse_not_strict"
    for spec in quality_v2_contract["metrics"]:
        metric_name = str(spec["name"])
        value = quality_v2_values[metric_name]
        reference = planner_quality_v2_values[metric_name]
        tolerance, strict_margin = _quality_v2_metric_thresholds(reference, spec)
        if value > reference + tolerance:
            relations[metric_name] = "regressed"
        elif value < reference - strict_margin:
            relations[metric_name] = "strictly_improved"
        else:
            relations[metric_name] = "non_worse_not_strict"
    causal_latency = _t5_causal_latency(record)
    planner_causal_latency = _t5_causal_latency(planner)
    if causal_latency is not None:
        assert planner_causal_latency is not None
        metric_name = "impact_end_to_first_qualifying_applied_correction_s"
        if causal_latency > planner_causal_latency + 1.0e-9:
            relations[metric_name] = "regressed"
        elif causal_latency < planner_causal_latency - 1.0e-9:
            relations[metric_name] = "strictly_improved"
        else:
            relations[metric_name] = "non_worse_not_strict"
    return relations


def _audit_provenance_file(
    root: Path,
    row: Any,
    *,
    expected_relative_path: str,
    expected_sha256: str,
) -> Path:
    if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256"}:
        raise ValueError("dataset provenance file inventory mismatch")
    if (
        row.get("relative_path") != expected_relative_path
        or row.get("sha256") != expected_sha256
    ):
        raise ValueError("dataset provenance file identity mismatch")
    path = _safe_dataset_path(root, expected_relative_path)
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("dataset provenance file checksum mismatch")
    return path


def _task_compatibility_inventory(
    candidate_payload: Mapping[str, Any], policy_benchmark_commit: str
) -> list[dict[str, Any]]:
    task = candidate_payload.get("task")
    candidates = candidate_payload.get("candidates")
    if not isinstance(task, str) or not isinstance(candidates, list):
        raise ValueError("candidate compatibility inventory is incomplete")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or candidate.get("kind") != "policy":
            continue
        provenance = candidate.get("provenance")
        source = provenance.get("source") if isinstance(provenance, Mapping) else None
        benchmark = (
            provenance.get("benchmark") if isinstance(provenance, Mapping) else None
        )
        state_schema = (
            provenance.get("state_schema") if isinstance(provenance, Mapping) else None
        )
        if (
            not isinstance(source, Mapping)
            or not isinstance(benchmark, Mapping)
            or benchmark.get("commit") != policy_benchmark_commit
            or not isinstance(state_schema, Mapping)
        ):
            continue
        state_dim = state_schema.get("state_dim")
        mask_dim = state_schema.get("mask_dim")
        if (
            isinstance(state_dim, bool)
            or not isinstance(state_dim, int)
            or state_dim < 1
            or isinstance(mask_dim, bool)
            or not isinstance(mask_dim, int)
            or mask_dim < 0
        ):
            raise ValueError("candidate compatibility state dimensions are invalid")
        policy_sha256 = _expected_sha256(
            candidate.get("policy_sha256"), "compatibility policy SHA-256"
        )
        row = {
            "task": task,
            "policy_sha256": policy_sha256,
            "policy_rlinf_commit": _full_commit(
                "compatibility policy RLinf commit", source.get("rlinf_commit")
            ),
            "policy_benchmark_commit": policy_benchmark_commit,
            "policy_state_schema_sha256": _expected_sha256(
                state_schema.get("sha256"), "compatibility policy state schema"
            ),
            "policy_state_dim": state_dim,
            "policy_mask_dim": mask_dim,
        }
        key = (task, policy_sha256)
        previous = rows.get(key)
        if previous is not None and previous != row:
            raise ValueError("candidate compatibility provenance is mixed")
        rows[key] = row
    return [rows[key] for key in sorted(rows)]


def _audit_evaluator_identity(
    root: Path,
    *,
    card: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    planner_dominance: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate_payload.get("schema_version") == CANDIDATE_SCHEMA:
        if card.get("evaluator_identity") is not None or card.get(
            "compatibility_evidence", []
        ) not in (None, []):
            raise ValueError(
                "legacy dataset unexpectedly declares v0.2 evaluator identity"
            )
        return None
    raw = candidate_payload.get("evaluator_identity")
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "evaluator_rlinf_commit",
        "evaluator_benchmark_commit",
        "backend_id",
        "policy_benchmark_relations",
    }:
        raise ValueError("candidate evaluator identity inventory mismatch")
    if raw.get("schema_version") != EVALUATOR_IDENTITY_SCHEMA:
        raise ValueError("candidate evaluator identity schema mismatch")
    evaluator_rlinf_commit = _full_commit(
        "evaluator RLinf commit",
        raw.get("evaluator_rlinf_commit"),
    )
    evaluator_benchmark_commit = _full_commit(
        "evaluator benchmark commit",
        raw.get("evaluator_benchmark_commit"),
    )
    backend_id = raw.get("backend_id")
    if (
        not isinstance(backend_id, str)
        or not backend_id
        or backend_id.strip() != backend_id
        or planner_dominance is None
        or planner_dominance.get("backend_id") != backend_id
    ):
        raise ValueError("candidate evaluator backend is invalid or unbound")
    source = card["source_identity"]
    if (
        source["evaluator_rlinf_commit"] != evaluator_rlinf_commit
        or source["evaluator_benchmark_commit"] != evaluator_benchmark_commit
    ):
        raise ValueError("dataset evaluator source identity mismatch")
    relations = raw.get("policy_benchmark_relations")
    policy_commits = candidate_payload.get("policy_benchmark_commits")
    if (
        not isinstance(relations, list)
        or not isinstance(policy_commits, list)
        or not relations
    ):
        raise ValueError("candidate benchmark relation inventory mismatch")
    expected_evidence: list[dict[str, str]] = []
    relation_commits: list[str] = []
    for relation in relations:
        if not isinstance(relation, Mapping) or set(relation) != {
            "policy_benchmark_commit",
            "relation",
            "evidence_path",
            "evidence_sha256",
        }:
            raise ValueError("candidate benchmark relation row mismatch")
        policy_commit = _full_commit(
            "policy benchmark relation commit",
            relation.get("policy_benchmark_commit"),
        )
        relation_commits.append(policy_commit)
        relation_name = relation.get("relation")
        evidence_path = relation.get("evidence_path")
        evidence_sha256 = relation.get("evidence_sha256")
        if relation_name == "identical":
            if (
                policy_commit != evaluator_benchmark_commit
                or evidence_path is not None
                or evidence_sha256 is not None
            ):
                raise ValueError("identical benchmark relation is inconsistent")
        elif relation_name == "checkpoint-compatible":
            if policy_commit == evaluator_benchmark_commit:
                raise ValueError(
                    "identical benchmark commits cannot claim compatibility"
                )
            digest = _expected_sha256(
                evidence_sha256,
                "benchmark compatibility evidence SHA-256",
            )
            expected_evidence.append(
                {
                    "policy_benchmark_commit": policy_commit,
                    "relative_path": f"provenance/evidence/{digest}.json",
                    "sha256": digest,
                }
            )
        else:
            raise ValueError(f"unsupported policy benchmark relation {relation_name!r}")
    if (
        relation_commits != sorted(set(relation_commits))
        or relation_commits != policy_commits
    ):
        raise ValueError("candidate benchmark relations are not canonical or complete")
    if card.get("evaluator_identity") != dict(raw):
        raise ValueError(
            "dataset-card evaluator identity differs from candidate manifest"
        )
    if card.get("compatibility_evidence") != expected_evidence:
        raise ValueError("dataset-card compatibility evidence inventory mismatch")
    for row in expected_evidence:
        proof_path = _audit_provenance_file(
            root,
            {
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
            },
            expected_relative_path=row["relative_path"],
            expected_sha256=row["sha256"],
        )
        proof = validate_compatibility_evidence(
            json.loads(proof_path.read_text(encoding="utf-8"))
        )
        if (
            proof["policy_benchmark_commit"] != row["policy_benchmark_commit"]
            or proof["evaluator_rlinf_commit"] != evaluator_rlinf_commit
            or proof["evaluator_benchmark_commit"] != evaluator_benchmark_commit
            or proof["backend_id"] != backend_id
        ):
            raise ValueError("compatibility evidence identity mismatch")
        expected_task_inventory = _task_compatibility_inventory(
            candidate_payload, row["policy_benchmark_commit"]
        )
        proof_task_inventory = [
            {
                "task": probe["task"],
                "policy_sha256": probe["policy_sha256"],
                "policy_rlinf_commit": probe["policy_rlinf_commit"],
                "policy_benchmark_commit": proof["policy_benchmark_commit"],
                "policy_state_schema_sha256": probe["policy_state_schema_sha256"],
                "policy_state_dim": probe["policy_state_dim"],
                "policy_mask_dim": probe["policy_mask_dim"],
            }
            for probe in proof["probes"]
            if probe["task"] == candidate_payload["task"]
        ]
        if proof_task_inventory != expected_task_inventory:
            raise ValueError(
                "compatibility evidence does not cover the task policy pool"
            )
    return dict(raw)


def _audit_candidate_release_chain(
    root: Path,
    *,
    card: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    candidate_manifest_sha256: str,
) -> None:
    if candidate_payload.get("schema_version") == CANDIDATE_SCHEMA:
        if card.get("candidate_release_manifest_sha256") is not None or card.get(
            "candidate_release_provenance", []
        ) not in (None, []):
            raise ValueError(
                "legacy dataset unexpectedly declares a v0.2 release chain"
            )
        return
    release_sha256 = _expected_sha256(
        card.get("candidate_release_manifest_sha256"),
        "candidate release manifest SHA-256",
    )
    provenance = card.get("candidate_release_provenance")
    if not isinstance(provenance, list) or len(provenance) != 2:
        raise ValueError("candidate release provenance inventory mismatch")
    release_path = _audit_provenance_file(
        root,
        provenance[0],
        expected_relative_path="provenance/candidate_release/release_manifest.json",
        expected_sha256=release_sha256,
    )
    sums_sha256 = _expected_sha256(
        provenance[1].get("sha256") if isinstance(provenance[1], Mapping) else None,
        "candidate release SHA256SUMS SHA-256",
    )
    _audit_provenance_file(
        root,
        provenance[1],
        expected_relative_path="provenance/candidate_release/SHA256SUMS",
        expected_sha256=sums_sha256,
    )
    release = json.loads(release_path.read_text(encoding="utf-8"))
    expected_release_keys = {
        "schema_version",
        "release_id",
        "candidate_schema_version",
        "evaluator_identity",
        "policy_rlinf_commits",
        "policy_benchmark_commits",
        "evaluator_evidence",
        "calibration_evidence",
        "tasks",
        "task_manifest_sha256",
        "candidate_count",
        "deduplicated",
        "input_spec_sha256",
        "input_inventory_sha256",
        "inputs_sha256_sha256",
        "production_validated",
        "payload_sha256",
    }
    if (
        not isinstance(release, Mapping)
        or set(release) != expected_release_keys
        or release.get("schema_version") != CANDIDATE_RELEASE_SCHEMA
        or release.get("release_id") != "RLD2"
        or release.get("candidate_schema_version") != CANDIDATE_SCHEMA_V2
        or release.get("production_validated") is not True
        or _payload_sha256(release) != release.get("payload_sha256")
        or tuple(release.get("tasks", [])) != EXACT_TASKS
    ):
        raise ValueError(
            "candidate release manifest is not a production exact-14 release"
        )
    task = str(card["task"])
    task_hashes = release.get("task_manifest_sha256")
    candidate_counts = release.get("candidate_count")
    if (
        not isinstance(task_hashes, Mapping)
        or set(task_hashes) != set(EXACT_TASKS)
        or task_hashes.get(task) != candidate_manifest_sha256
        or not isinstance(candidate_counts, Mapping)
        or set(candidate_counts) != set(EXACT_TASKS)
        or candidate_counts.get(task) != len(candidate_payload["candidates"])
    ):
        raise ValueError("candidate release does not bind this task manifest")
    if not set(candidate_payload["policy_rlinf_commits"]).issubset(
        set(release.get("policy_rlinf_commits", []))
    ) or not set(candidate_payload["policy_benchmark_commits"]).issubset(
        set(release.get("policy_benchmark_commits", []))
    ):
        raise ValueError("candidate release policy authority union is incomplete")
    release_identity = release.get("evaluator_identity")
    task_identity = candidate_payload.get("evaluator_identity")
    if not isinstance(release_identity, Mapping) or not isinstance(
        task_identity, Mapping
    ):
        raise ValueError("candidate release evaluator identity is missing")
    for key in (
        "schema_version",
        "evaluator_rlinf_commit",
        "evaluator_benchmark_commit",
        "backend_id",
    ):
        if release_identity.get(key) != task_identity.get(key):
            raise ValueError("candidate release evaluator identity mismatch")
    release_relations = release_identity.get("policy_benchmark_relations")
    task_relations = task_identity.get("policy_benchmark_relations")
    if not isinstance(release_relations, list) or not isinstance(task_relations, list):
        raise ValueError("candidate release benchmark relations are missing")
    release_by_commit = {
        row.get("policy_benchmark_commit"): row
        for row in release_relations
        if isinstance(row, Mapping)
    }
    for task_relation in task_relations:
        release_relation = release_by_commit.get(
            task_relation.get("policy_benchmark_commit")
        )
        if (
            not isinstance(release_relation, Mapping)
            or release_relation.get("relation") != task_relation.get("relation")
            or release_relation.get("evidence_sha256")
            != task_relation.get("evidence_sha256")
        ):
            raise ValueError("candidate release benchmark relation mismatch")
    calibration_rows = release.get("calibration_evidence")
    if not isinstance(calibration_rows, list):
        raise ValueError("candidate release calibration inventory is missing")
    calibration_by_task = {
        row.get("task"): row for row in calibration_rows if isinstance(row, Mapping)
    }
    task_calibration = candidate_payload["planner_dominance"]["calibration"]
    released_calibration = calibration_by_task.get(task)
    if (
        not isinstance(released_calibration, Mapping)
        or set(released_calibration) != {"task", "path", "sha256"}
        or released_calibration.get("sha256") != task_calibration["evidence_sha256"]
    ):
        raise ValueError("candidate release calibration evidence mismatch")


def _audit_calibration_evidence(
    root: Path,
    *,
    card: Mapping[str, Any],
    planner_dominance: Mapping[str, Any] | None,
    evaluator_identity: Mapping[str, Any] | None,
) -> dict[str, float]:
    if planner_dominance is None:
        if card.get("calibration_evidence") is not None:
            raise ValueError(
                "legacy dataset unexpectedly declares calibration evidence"
            )
        return {}
    if evaluator_identity is None:
        raise ValueError("planner calibration has no evaluator identity")
    calibration = planner_dominance["calibration"]
    digest = str(calibration["evidence_sha256"])
    relative_path = f"provenance/calibration/{digest}.json"
    path = _audit_provenance_file(
        root,
        card.get("calibration_evidence"),
        expected_relative_path=relative_path,
        expected_sha256=digest,
    )
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "schema_version",
        "task",
        "backend_id",
        "evaluator_identity_sha256",
        "split",
        "test_exposure",
        "reset_manifest_sha256",
        "replay_count",
        "replays",
        "payload_sha256",
    }:
        raise ValueError("planner calibration evidence inventory mismatch")
    if (
        evidence.get("schema_version") != CALIBRATION_EVIDENCE_SCHEMA
        or _payload_sha256(evidence) != evidence.get("payload_sha256")
        or evidence.get("task") != planner_dominance["task"]
        or evidence.get("backend_id") != planner_dominance["backend_id"]
        or evidence.get("evaluator_identity_sha256")
        != _payload_sha256(
            {
                "evaluator_rlinf_commit": evaluator_identity["evaluator_rlinf_commit"],
                "evaluator_benchmark_commit": evaluator_identity[
                    "evaluator_benchmark_commit"
                ],
                "backend_id": evaluator_identity["backend_id"],
            }
        )
    ):
        raise ValueError("planner calibration schema/evaluator identity mismatch")
    if evidence.get("split") not in {"train", "validation"} or evidence.get(
        "test_exposure"
    ) != {"test_id": False, "test_ood": False}:
        raise ValueError("planner calibration evidence used a formal test split")
    if evidence.get("reset_manifest_sha256") != calibration["reset_manifest_sha256"]:
        raise ValueError("planner calibration reset manifest mismatch")
    replay_count = evidence.get("replay_count")
    replays = evidence.get("replays")
    if (
        replay_count != calibration["replay_count"]
        or isinstance(replay_count, bool)
        or not isinstance(replay_count, int)
        or replay_count < 3
        or not isinstance(replays, list)
        or len(replays) != replay_count
    ):
        raise ValueError("planner calibration replay count mismatch")
    replay_keys = {
        "replay_index",
        "environment_instance_id",
        "episode_id",
        "reset_request_sha256",
        "action_sha256",
        "success",
        "safety_failure",
        "finite_and_bounded",
        "termination_reason",
        "trajectory_completion",
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
        "task_quality",
    }
    environment_ids: set[str] = set()
    frozen_identity: tuple[Any, ...] | None = None
    metric_rows: list[dict[str, Any]] = []
    for index, raw_replay in enumerate(replays):
        if not isinstance(raw_replay, Mapping) or set(raw_replay) != replay_keys:
            raise ValueError("planner calibration replay row inventory mismatch")
        environment_id = raw_replay.get("environment_instance_id")
        if (
            raw_replay.get("replay_index") != index
            or not isinstance(environment_id, str)
            or not environment_id
            or environment_id.strip() != environment_id
            or environment_id in environment_ids
        ):
            raise ValueError(
                "planner calibration did not use unique fresh environments"
            )
        environment_ids.add(environment_id)
        episode_id = raw_replay.get("episode_id")
        if episode_id != calibration["reset_episode_id"]:
            raise ValueError("planner calibration reset episode identity drifted")
        reset_request_sha256 = _expected_sha256(
            raw_replay.get("reset_request_sha256"),
            "planner calibration reset request SHA-256",
        )
        action_sha256 = _expected_sha256(
            raw_replay.get("action_sha256"),
            "planner calibration action SHA-256",
        )
        if (
            raw_replay.get("success") is not True
            or raw_replay.get("safety_failure") is not False
            or raw_replay.get("finite_and_bounded") is not True
        ):
            raise ValueError(
                "planner calibration replay is not successful, safe, and finite"
            )
        termination_reason = raw_replay.get("termination_reason")
        if not isinstance(termination_reason, str) or not termination_reason:
            raise ValueError("planner calibration termination reason is missing")
        completion = _finite_number(
            raw_replay["trajectory_completion"],
            "planner calibration trajectory completion",
        )
        completion_time = _finite_number(
            raw_replay["completion_time_s"],
            "planner calibration completion time",
        )
        control_steps = raw_replay.get("control_steps")
        action_l2_sum = _finite_number(
            raw_replay["action_l2_sum"],
            "planner calibration action L2 sum",
        )
        if (
            not 0.0 <= completion <= 1.0
            or completion_time <= 0.0
            or isinstance(control_steps, bool)
            or not isinstance(control_steps, int)
            or control_steps < 1
            or action_l2_sum < 0.0
        ):
            raise ValueError("planner calibration replay metrics are invalid")
        replay = dict(raw_replay)
        _validate_attempt_quality(replay, planner_dominance)
        identity = (
            episode_id,
            reset_request_sha256,
            action_sha256,
            termination_reason,
            control_steps,
        )
        if frozen_identity is None:
            frozen_identity = identity
        elif identity != frozen_identity:
            raise ValueError(
                "planner calibration reset/action/outcome identity drifted"
            )
        metric_rows.append(replay)
    observed_drifts: dict[str, float] = {}
    for metric_name in _metric_names(planner_dominance):
        values = [_metric_value(row, metric_name) for row in metric_rows]
        observed = max(values) - min(values)
        frozen = float(
            _metric_spec(planner_dominance, metric_name)["max_observed_replay_drift"]
        )
        if not math.isclose(observed, frozen, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError(
                f"planner calibration drift does not recompute for {metric_name!r}"
            )
        observed_drifts[metric_name] = observed
    return observed_drifts


def _render_parity_failure_reason(error: str) -> str | None:
    if "parity failed" in error:
        return "render_parity_failed"
    if "canonical replay contract" in error:
        return "canonical_replay_contract_failed"
    return None


def _render_parity_skip_events(recovery_events: Any) -> dict[int, dict[str, str]]:
    """Parse the frozen v0.1 string event used by already exported shards."""

    if not isinstance(recovery_events, list) or not all(
        isinstance(event, str) for event in recovery_events
    ):
        raise ValueError("recovery events must be a list of strings")
    skips: dict[int, dict[str, str]] = {}
    for event in recovery_events:
        if not event.startswith("render_parity_skip:"):
            continue
        parts = event.split(":", maxsplit=4)
        if len(parts) != 5 or parts[:2] != ["render_parity_skip", "reset"]:
            raise ValueError("render-parity recovery event is malformed")
        try:
            reset_index = int(parts[2])
        except ValueError as error:
            raise ValueError(
                "render-parity recovery reset index is malformed"
            ) from error
        episode_id, message = parts[3], parts[4]
        reason = _render_parity_failure_reason(message)
        if reset_index < 0 or not episode_id or reason is None:
            raise ValueError("render-parity recovery event is invalid")
        if reset_index in skips:
            raise ValueError("multiple render-parity recovery events name one reset")
        skips[reset_index] = {
            "episode_id": episode_id,
            "error": message,
            "reason": reason,
        }
    return skips


def _audit_render_parity_skip(
    result: Mapping[str, Any],
    selected: Mapping[str, Any],
    event: Mapping[str, str] | None,
) -> str:
    """Validate a rejected publication whose light winner failed render replay."""

    if event is None:
        raise ValueError("render-parity skip has no matching recovery event")
    if event.get("episode_id") != result.get("episode_id"):
        raise ValueError("render-parity recovery event episode identity mismatch")
    skip = result.get("render_parity_skip")
    if skip is None:
        return "legacy-v0.1"
    if not isinstance(skip, dict):
        raise ValueError("render-parity skip evidence is not a mapping")
    expected_keys = {
        "schema_version",
        "reason",
        "error_type",
        "error",
        "candidate_id",
        "candidate_index",
        "attempt_tape",
        "attempt_tape_sha256",
        "action_sha256",
    }
    if set(skip) != expected_keys:
        raise ValueError("render-parity skip evidence inventory mismatch")
    if skip.get("schema_version") != RENDER_PARITY_SKIP_SCHEMA:
        raise ValueError("render-parity skip evidence schema mismatch")
    if skip.get("error_type") not in {"RuntimeError", "ValueError"}:
        raise ValueError("render-parity skip error type is invalid")
    message = skip.get("error")
    if not isinstance(message, str):
        raise ValueError("render-parity skip error is missing")
    reason = _render_parity_failure_reason(message)
    if reason is None or skip.get("reason") != reason:
        raise ValueError("render-parity skip reason does not recompute")
    if event.get("error") != message or event.get("reason") != reason:
        raise ValueError("render-parity skip disagrees with its recovery event")
    selected_values = {
        "candidate_id": selected["candidate_id"],
        "candidate_index": int(selected["candidate_index"]),
        "attempt_tape": selected["attempt_tape"],
        "attempt_tape_sha256": selected["attempt_tape_sha256"],
        "action_sha256": selected["action_sha256"],
    }
    if any(skip.get(key) != value for key, value in selected_values.items()):
        raise ValueError("render-parity skip is not bound to the selected attempt")
    return "structured-v0.1"


def _audit_t5_replan_causal_history(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    control_steps: int,
) -> None:
    """Reconstruct T5 queue records from split arrays and rerun the validator."""

    from se3_wam.benchmark.config import load_task_config
    from se3_wam.benchmark.registry import get_task_spec
    from se3_wam.benchmark.t5_timing_contract import validate_t5_replan_timing

    if record.get("task_id") != "t5_replan":
        raise ValueError("T5 causal tape is bound to the wrong attempt task")
    if set(arrays) != set(T5_CAUSAL_TAPE_INVENTORY):
        raise ValueError("T5 causal tape array inventory mismatch")

    def require_array(
        name: str,
        *,
        dtype: np.dtype[Any],
        shape: tuple[int, ...],
    ) -> np.ndarray:
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(f"T5 causal tape {name} dtype or shape mismatch")
        return value

    schema = np.asarray(arrays["t5_action_history_schema"])
    if (
        schema.shape != ()
        or schema.dtype != np.dtype(f"<U{len(T5_ACTION_HISTORY_SCHEMA)}")
        or str(schema.item()) != T5_ACTION_HISTORY_SCHEMA
    ):
        raise ValueError("T5 causal tape semantic schema mismatch")
    action_labels = np.asarray(arrays["action_value_semantic_labels"])
    timing_value_labels = np.asarray(arrays["t5_timing_value_semantic_labels"])
    timing_count_labels = np.asarray(arrays["t5_timing_count_semantic_labels"])
    for labels, expected, expected_dtype, name in (
        (
            action_labels,
            T5_ACTION_VALUE_SEMANTIC_LABELS,
            np.dtype("<U32"),
            "action-value",
        ),
        (
            timing_value_labels,
            T5_TIMING_VALUE_SEMANTIC_LABELS,
            np.dtype("<U32"),
            "timing-value",
        ),
        (
            timing_count_labels,
            T5_TIMING_COUNT_SEMANTIC_LABELS,
            np.dtype("<U40"),
            "timing-count",
        ),
    ):
        if (
            labels.dtype != expected_dtype
            or labels.shape != (len(expected),)
            or tuple(str(value) for value in labels.tolist()) != expected
        ):
            raise ValueError(f"T5 causal tape {name} semantic labels mismatch")

    timing_counts = require_array(
        "t5_timing_counts",
        dtype=np.dtype(np.int64),
        shape=(2,),
    )
    expected_issued_action_count = int(timing_counts[0])
    expected_action_delay_steps = int(timing_counts[1])
    if expected_issued_action_count != control_steps:
        raise ValueError("T5 causal tape issued count differs from control_steps")
    if expected_action_delay_steps < 0:
        raise ValueError("T5 causal tape action delay must be nonnegative")
    timing_values = require_array(
        "t5_timing_values",
        dtype=np.dtype(np.float64),
        shape=(3,),
    )
    impact_end_time_s = None if np.isnan(timing_values[0]) else float(timing_values[0])
    first_contact_time_s = (
        None if np.isnan(timing_values[1]) else float(timing_values[1])
    )
    control_hz = float(timing_values[2])
    task = get_task_spec("t5_replan")
    configured_delay_s = float(
        load_task_config("t5_replan")["latency"]["sensor_to_actuation_delay_s"]
    )
    configured_delay_steps = configured_delay_s * task.clock.control_hz
    if (
        not math.isfinite(control_hz)
        or control_hz != float(task.clock.control_hz)
        or not math.isclose(
            configured_delay_steps,
            round(configured_delay_steps),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or expected_action_delay_steps != int(round(configured_delay_steps))
    ):
        raise ValueError("T5 causal tape control rate or configured delay mismatch")
    for value, label in (
        (impact_end_time_s, "impact end"),
        (first_contact_time_s, "first contact"),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"T5 causal tape {label} time is invalid")

    issued_action_values = require_array(
        "issued_action_values",
        dtype=np.dtype(np.float64),
        shape=(control_steps, 7),
    )
    issued_policy_step = require_array(
        "issued_policy_step",
        dtype=np.dtype(np.int64),
        shape=(control_steps,),
    )
    issued_time_s = require_array(
        "issued_time_s",
        dtype=np.dtype(np.float64),
        shape=(control_steps,),
    )
    scheduled_apply_policy_step = require_array(
        "scheduled_apply_policy_step",
        dtype=np.dtype(np.int64),
        shape=(control_steps,),
    )
    scheduled_apply_time_s = require_array(
        "scheduled_apply_time_s",
        dtype=np.dtype(np.float64),
        shape=(control_steps,),
    )
    expected_applied_count = max(0, control_steps - expected_action_delay_steps)
    applied_action_values = require_array(
        "applied_action_values",
        dtype=np.dtype(np.float64),
        shape=(expected_applied_count, 7),
    )
    applied_issue_policy_step = require_array(
        "applied_issue_policy_step",
        dtype=np.dtype(np.int64),
        shape=(expected_applied_count,),
    )
    actual_apply_policy_step = require_array(
        "actual_apply_policy_step",
        dtype=np.dtype(np.int64),
        shape=(expected_applied_count,),
    )
    actual_apply_time_s = require_array(
        "actual_apply_time_s",
        dtype=np.dtype(np.float64),
        shape=(expected_applied_count,),
    )
    expected_issued_steps = np.arange(control_steps, dtype=np.int64)
    expected_applied_issue_steps = np.arange(expected_applied_count, dtype=np.int64)
    if not np.array_equal(issued_policy_step, expected_issued_steps):
        raise ValueError(
            "T5 causal tape issued policy steps are not exact and contiguous"
        )
    if not np.array_equal(
        scheduled_apply_policy_step,
        expected_issued_steps + expected_action_delay_steps,
    ):
        raise ValueError("T5 causal tape scheduled delay does not match issued steps")
    if not np.array_equal(applied_issue_policy_step, expected_applied_issue_steps):
        raise ValueError("T5 causal tape applied rows do not uniquely cover due issues")
    if not np.array_equal(
        actual_apply_policy_step,
        expected_applied_issue_steps + expected_action_delay_steps,
    ):
        raise ValueError(
            "T5 causal tape actual apply steps do not preserve queue delay"
        )
    if (
        not np.allclose(
            issued_time_s,
            expected_issued_steps.astype(np.float64) / control_hz,
            rtol=0.0,
            atol=1.0e-9,
        )
        or not np.allclose(
            scheduled_apply_time_s,
            scheduled_apply_policy_step.astype(np.float64) / control_hz,
            rtol=0.0,
            atol=1.0e-9,
        )
        or not np.allclose(
            actual_apply_time_s,
            actual_apply_policy_step.astype(np.float64) / control_hz,
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ValueError("T5 causal tape action times are off the control grid")
    if (
        not np.all(np.isfinite(issued_action_values))
        or not np.all(np.isfinite(applied_action_values))
        or np.any(np.abs(issued_action_values) > 1.0)
        or np.any(np.abs(applied_action_values) > 1.0)
    ):
        raise ValueError("T5 causal tape actions are not finite bounded E7 values")
    if not np.array_equal(
        applied_action_values,
        issued_action_values[applied_issue_policy_step],
    ):
        raise ValueError(
            "T5 causal tape applied actions do not exactly link to issued values"
        )

    issued_actions = [
        {
            "policy_step": int(issued_policy_step[index]),
            "issue_time_s": float(issued_time_s[index]),
            "apply_policy_step": int(scheduled_apply_policy_step[index]),
            "apply_time_s": float(scheduled_apply_time_s[index]),
            "values": issued_action_values[index].tolist(),
        }
        for index in range(control_steps)
    ]
    applied_actions = [
        {
            **issued_actions[int(issue_step)],
            "actual_apply_policy_step": int(actual_apply_policy_step[index]),
            "actual_apply_time_s": float(actual_apply_time_s[index]),
        }
        for index, issue_step in enumerate(applied_issue_policy_step)
    ]
    report = validate_t5_replan_timing(
        issued_actions=issued_actions,
        applied_actions=applied_actions,
        impact_end_time_s=impact_end_time_s,
        first_contact_time_s=first_contact_time_s,
        expected_issued_action_count=expected_issued_action_count,
        expected_action_delay_steps=expected_action_delay_steps,
        control_hz=control_hz,
    )
    declared_gate = record.get("t5_replan_causal_timing_passed")
    if not isinstance(declared_gate, bool) or declared_gate is not report.passed:
        raise ValueError("T5 Replan causal-timing gate does not recompute from tape")
    declared_latency = record.get("impact_end_to_first_qualifying_applied_correction_s")
    if not report.passed:
        if declared_latency is not None:
            raise ValueError(
                "failed T5 causal timing cannot declare a correction latency"
            )
        return
    if impact_end_time_s is None or not report.qualifying_correction_steps:
        raise ValueError("passing T5 causal timing lacks a qualifying correction")
    first_qualifying_step = report.qualifying_correction_steps[0]
    applied_index = int(
        np.flatnonzero(applied_issue_policy_step == first_qualifying_step)[0]
    )
    recomputed_latency = float(actual_apply_time_s[applied_index]) - impact_end_time_s
    stored_latency = _finite_number(
        declared_latency,
        "T5 impact-end-to-applied-correction latency",
    )
    if recomputed_latency < 0.0 or not math.isclose(
        stored_latency,
        recomputed_latency,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("T5 causal correction latency does not recompute from tape")


def _audit_attempt_tape(
    root: Path,
    record: Mapping[str, Any],
    *,
    expected_task: str,
    quality_v2_thresholds: Mapping[str, object] | None = None,
    quality_v2_thresholds_sha256: str | None = None,
) -> None:
    schema_version = record.get("schema_version")
    if (
        schema_version != ATTEMPT_SCHEMA
        and schema_version not in HISTORICAL_ATTEMPT_SCHEMAS
    ):
        raise ValueError("attempt schema mismatch")
    if record.get("task_id") != expected_task:
        raise ValueError("attempt task mismatch")
    for boolean_key in ("success", "safety_failure", "finite_and_bounded"):
        if not isinstance(record.get(boolean_key), bool):
            raise ValueError(f"attempt {boolean_key} must be boolean")
    completion = _finite_number(
        record.get("trajectory_completion"),
        "attempt trajectory completion",
    )
    if not 0.0 <= completion <= 1.0:
        raise ValueError("attempt trajectory completion must be in [0, 1]")
    completion_time = record.get("completion_time_s")
    if record.get("success") is True:
        if _finite_number(completion_time, "attempt completion time") <= 0.0:
            raise ValueError("successful attempt completion time must be positive")
    elif completion_time is not None:
        raise ValueError("unsuccessful attempt must not declare completion time")
    relative = record.get("attempt_tape")
    if not isinstance(relative, str):
        raise ValueError("attempt tape path is missing")
    path = _safe_dataset_path(root, relative)
    if _sha256(path) != record.get("attempt_tape_sha256"):
        raise ValueError("attempt tape checksum mismatch")
    with np.load(path, allow_pickle=False) as tape:
        base_inventory = {
            "states",
            "policy_actions",
            "actions",
            "rewards",
            "terminated",
            "truncated",
        }
        quality_inventory = {
            "eef_pose_xyzw",
            "fingerpad_closing_axis_world",
            "object_pose_wxyz",
            "fingerpad_contact_flags",
        }
        inventory = set(tape.files)
        if schema_version == LEGACY_ATTEMPT_SCHEMA:
            valid_inventory = frozenset(inventory) in {
                frozenset(base_inventory),
                frozenset(base_inventory | quality_inventory),
            }
        elif schema_version == ATTEMPT_SCHEMA and expected_task == "t5_replan":
            valid_inventory = inventory == (
                base_inventory | quality_inventory | T5_CAUSAL_TAPE_INVENTORY
            )
        else:
            valid_inventory = inventory == base_inventory | quality_inventory
        if not valid_inventory:
            raise ValueError("attempt tape array inventory mismatch")
        states = np.asarray(tape["states"])
        policy_actions = np.asarray(tape["policy_actions"])
        actions = np.asarray(tape["actions"])
        rewards = np.asarray(tape["rewards"])
        terminated = np.asarray(tape["terminated"])
        truncated = np.asarray(tape["truncated"])
        eef_pose_xyzw = (
            np.asarray(tape["eef_pose_xyzw"]) if "eef_pose_xyzw" in tape.files else None
        )
        closing_axis_world = (
            np.asarray(tape["fingerpad_closing_axis_world"])
            if "fingerpad_closing_axis_world" in tape.files
            else None
        )
        object_pose_wxyz = (
            np.asarray(tape["object_pose_wxyz"])
            if "object_pose_wxyz" in tape.files
            else None
        )
        fingerpad_contact_flags = (
            np.asarray(tape["fingerpad_contact_flags"])
            if "fingerpad_contact_flags" in tape.files
            else None
        )
        t5_causal_arrays = (
            {name: np.asarray(tape[name]) for name in T5_CAUSAL_TAPE_INVENTORY}
            if T5_CAUSAL_TAPE_INVENTORY <= inventory
            else None
        )
    raw_steps = record.get("control_steps")
    if isinstance(raw_steps, bool) or not isinstance(raw_steps, int):
        raise ValueError("attempt control_steps must be an integer")
    steps = raw_steps
    if (
        steps < 1
        or states.ndim != 2
        or states.shape[0] != steps + 1
        or actions.shape != (steps, 7)
        or policy_actions.shape != (steps, 7)
        or rewards.shape != (steps,)
        or terminated.shape != (steps,)
        or truncated.shape != (steps,)
    ):
        raise ValueError("attempt tape array shapes do not align")
    if terminated.dtype != np.bool_ or truncated.dtype != np.bool_:
        raise ValueError("attempt termination tapes must be boolean")
    if np.any(terminated[:-1]) or np.any(truncated[:-1]):
        raise ValueError("attempt has a pre-terminal done flag")
    if bool(terminated[-1]) == bool(truncated[-1]):
        raise ValueError(
            "attempt must terminate or truncate exactly once at its final step"
        )
    if schema_version == ATTEMPT_SCHEMA:
        if expected_task == "t5_replan":
            if record.get("issued_equals_applied") is not False:
                raise ValueError("T5 Replan must declare issued_equals_applied=false")
            if t5_causal_arrays is None:
                raise ValueError("T5 Replan attempt is missing its causal history tape")
            _audit_t5_replan_causal_history(
                record,
                t5_causal_arrays,
                control_steps=steps,
            )
        else:
            if record.get("issued_equals_applied") is not True:
                raise ValueError(
                    "non-T5 attempt must declare issued_equals_applied=true"
                )
            if (
                record.get("t5_replan_causal_timing_passed") is not None
                or record.get("impact_end_to_first_qualifying_applied_correction_s")
                is not None
            ):
                raise ValueError(
                    "non-T5 attempt cannot declare T5 causal timing evidence"
                )
    finite = bool(
        np.all(np.isfinite(states))
        and np.all(np.isfinite(policy_actions))
        and np.all(np.isfinite(actions))
        and np.all(np.isfinite(rewards))
        and np.all(np.abs(policy_actions) <= 1.0)
        and np.all(np.abs(actions) <= 1.0)
    )
    if finite != bool(record["finite_and_bounded"]):
        raise ValueError("attempt finite/bounded status does not recompute")
    hashes = {
        "state_sha256": hashlib.sha256(
            np.ascontiguousarray(states).tobytes()
        ).hexdigest(),
        "policy_action_sha256": hashlib.sha256(
            np.ascontiguousarray(policy_actions).tobytes()
        ).hexdigest(),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest(),
        "reward_sha256": hashlib.sha256(
            np.ascontiguousarray(rewards).tobytes()
        ).hexdigest(),
    }
    if any(record.get(key) != value for key, value in hashes.items()):
        raise ValueError("attempt array content checksum does not recompute")
    quality_v2 = record.get("quality_v2")
    if schema_version != LEGACY_ATTEMPT_SCHEMA and quality_v2 is None:
        raise ValueError("attempt v0.2/v0.3 requires a quality-v2 summary")
    if quality_v2 is not None:
        if not isinstance(quality_v2, Mapping):
            raise ValueError("attempt quality-v2 summary must be a mapping")
        quality_v2_sha256 = _expected_sha256(
            record.get("quality_v2_sha256"),
            "attempt quality-v2 summary SHA-256",
        )
        if _payload_sha256(quality_v2) != quality_v2_sha256:
            raise ValueError("stored quality-v2 summary checksum does not recompute")
        if any(
            value is None
            for value in (
                eef_pose_xyzw,
                closing_axis_world,
                object_pose_wxyz,
                fingerpad_contact_flags,
            )
        ):
            raise ValueError("quality-v2 attempt is missing replay source tapes")
        assert eef_pose_xyzw is not None
        assert closing_axis_world is not None
        assert object_pose_wxyz is not None
        assert fingerpad_contact_flags is not None
        if (
            eef_pose_xyzw.shape != (steps + 1, 7)
            or closing_axis_world.shape != (steps + 1, 3)
            or object_pose_wxyz.shape != (steps + 1, 7)
            or fingerpad_contact_flags.shape != (steps + 1, 2)
        ):
            raise ValueError("quality-v2 pose tapes do not align with control steps")
        raw_events = record.get("quality_v2_events_by_observation")
        if (
            not isinstance(raw_events, list)
            or len(raw_events) != steps + 1
            or any(
                not isinstance(row, list)
                or any(not isinstance(name, str) or not name for name in row)
                for row in raw_events
            )
        ):
            raise ValueError("quality-v2 per-observation event tape is malformed")
        from se3_wam.benchmark.config import load_task_config
        from se3_wam.benchmark.trajectory_quality import (
            evaluate_quality_v2_gate,
            trajectory_quality_v2_from_observations,
        )

        observations = [
            SimpleNamespace(
                privileged={
                    "eef_pose_xyzw": eef_pose_xyzw[index],
                    "fingerpad_closing_axis_world": closing_axis_world[index],
                    "object_pose_wxyz": object_pose_wxyz[index],
                    "fingerpad_contact_flags": fingerpad_contact_flags[index],
                },
                events_since_last_observation=tuple(
                    SimpleNamespace(name=name) for name in raw_events[index]
                ),
            )
            for index in range(steps + 1)
        ]
        recomputed_quality_v2 = trajectory_quality_v2_from_observations(
            observations,
            actions,
            task_id=expected_task,
            task_config=load_task_config(expected_task),
            sample_period_s=0.05,
            continuous_dimensions=max(1, actions.shape[1] - 1),
        )
        if recomputed_quality_v2 != quality_v2:
            raise ValueError("stored quality-v2 summary differs from the replay tape")
        if _payload_sha256(recomputed_quality_v2) != quality_v2_sha256:
            raise ValueError("quality-v2 summary checksum does not recompute")
        if quality_v2_thresholds is None or quality_v2_thresholds_sha256 is None:
            raise ValueError("quality-v2 audit requires the frozen threshold contract")
        recomputed_gate = evaluate_quality_v2_gate(
            recomputed_quality_v2,
            quality_v2_thresholds,
            task_id=expected_task,
        )
        recomputed_gate["contract_sha256"] = quality_v2_thresholds_sha256
        if recomputed_gate != record.get("quality_v2_gate"):
            raise ValueError("quality-v2 gate does not recompute")
    elif any(
        value is not None
        for value in (
            eef_pose_xyzw,
            closing_axis_world,
            object_pose_wxyz,
            fingerpad_contact_flags,
        )
    ):
        raise ValueError("quality-v2 pose tapes require a quality-v2 summary")
    attempt_return = _finite_number(record.get("return"), "attempt return")
    if not math.isclose(
        attempt_return,
        float(rewards.sum(dtype=np.float64)),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("attempt return does not recompute")
    action_l2_sum = _finite_number(record.get("action_l2_sum"), "attempt action L2 sum")
    if not math.isclose(
        action_l2_sum,
        float(np.square(actions).sum(dtype=np.float64)),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("attempt action norm does not recompute")
    replay = record.get("replay_validation")
    if not isinstance(replay, dict):
        raise ValueError("attempt replay validation is missing")
    if not isinstance(replay.get("passed"), bool):
        raise ValueError("attempt replay-validation passed flag must be boolean")
    if _payload_sha256(replay) != record.get("replay_validation_sha256"):
        raise ValueError("attempt replay-validation checksum does not recompute")
    if list(_quality_score(record)) != record.get("quality_score"):
        raise ValueError("attempt quality score does not recompute")
    eligible_label = record.get("eligible")
    if not isinstance(eligible_label, bool) or _eligible(record) is not eligible_label:
        raise ValueError("attempt eligibility does not recompute")


def _audit_winner_release_eligibility(
    audit: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    qv4_disabled_nonblocking: bool,
) -> None:
    if not qv4_disabled_nonblocking:
        if not audit.get("eligible_for_behavior_cloning"):
            raise ValueError("winner episode is not behavior-cloning eligible")
        return

    metrics = metadata.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Qv4-disabled winner has no metrics mapping")
    if metadata.get("quality_v4_validation") is not None:
        raise ValueError("Qv4-disabled winner unexpectedly contains Qv4 validation")
    if audit.get("eligible_for_behavior_cloning") is not False:
        raise ValueError("Qv4-disabled winner unexpectedly claims audit BC eligibility")
    if (
        metrics.get("eligible_for_behavior_cloning") is not False
        or metrics.get("success") is not True
        or metrics.get("label_valid") is not True
        or metrics.get("termination_reason") != "success"
        or not math.isclose(
            _finite_number(
                metrics.get("trajectory_completion"),
                "Qv4-disabled winner trajectory completion",
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Qv4-disabled winner release eligibility mismatch")


def _audit_winner_episode(
    root: Path,
    winner: Mapping[str, Any],
    attempt: Mapping[str, Any],
    reset: Mapping[str, Any],
    card: Mapping[str, Any],
    planner_dominance: Mapping[str, Any] | None,
    *,
    qv4_disabled_nonblocking: bool,
) -> None:
    from se3_wam.benchmark.dataset import audit_episode

    relative = winner.get("relative_episode_dir")
    if not isinstance(relative, str):
        raise ValueError("winner episode directory is missing")
    episode_dir = _safe_dataset_path(root, relative)
    audit = audit_episode(episode_dir)
    if audit.get("episode_id") != attempt["episode_id"] or not audit.get("success"):
        raise ValueError("winner episode audit identity or success mismatch")
    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    _audit_winner_release_eligibility(
        audit,
        metadata,
        qv4_disabled_nonblocking=qv4_disabled_nonblocking,
    )
    if metadata != {key: winner[key] for key in metadata}:
        raise ValueError("winner manifest does not reproduce the episode record")
    request = metadata.get("request")
    if not isinstance(request, dict):
        raise ValueError("winner episode reset request is missing")
    for key in (
        "episode_id",
        "task_id",
        "split",
        "seed",
        "action_mode",
        "observation_track",
        "object_mode",
        "reset_mode",
        "factors",
    ):
        if request.get(key) != reset.get(key):
            raise ValueError(f"winner episode reset identity mismatch for {key}")
    teacher = metadata.get("teacher_preparation")
    if not isinstance(teacher, dict) or teacher.get("method") != (
        "rlinf_best_known_candidate_selection"
    ):
        raise ValueError("winner episode selection provenance is missing")
    candidate = teacher.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("winner candidate provenance is missing")
    candidate_search_mode = _candidate_search_mode(card)
    selection_mode = _selection_mode(card)
    if (
        candidate.get("candidate_id") != attempt["candidate_id"]
        or int(teacher.get("candidate_index", -1)) != int(attempt["candidate_index"])
        or int(teacher.get("budget_used", -1)) != int(winner["budget_used"])
        or teacher.get("candidate_manifest_sha256") != card["candidate_manifest_sha256"]
        or teacher.get("selection_contract") != _selection_contract(selection_mode)
        or teacher.get("winner_quality_score") != list(_quality_score(attempt))
        or (
            attempt.get("quality_v2_sha256") is not None
            and teacher.get("quality_v2_sha256") != attempt["quality_v2_sha256"]
        )
        or teacher.get("lightweight_action_sha256") != attempt["action_sha256"]
        or teacher.get("source_identity") != card["source_identity"]
        or teacher.get("quality_v2_threshold_identity")
        != card.get("quality_v2_threshold_identity")
        or teacher.get("planner_dominance") != planner_dominance
        or teacher.get("evaluator_identity") != card.get("evaluator_identity")
        or teacher.get("compatibility_evidence") != card.get("compatibility_evidence")
        or teacher.get("calibration_evidence") != card.get("calibration_evidence")
        or teacher.get("candidate_release_manifest_sha256")
        != card.get("candidate_release_manifest_sha256")
        or teacher.get("selection_result") != winner.get("selection_result")
    ):
        raise ValueError("winner episode selection provenance does not recompute")
    if (
        "candidate_search_mode" in card
        and teacher.get("candidate_search_mode") != candidate_search_mode
    ):
        raise ValueError("winner episode candidate-search provenance mismatch")
    if "selection_mode" in card and teacher.get("selection_mode") != selection_mode:
        raise ValueError("winner episode selection-mode provenance mismatch")
    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        action_values = np.asarray(handle["actions/values"])
    rendered_action_sha256 = hashlib.sha256(
        np.ascontiguousarray(action_values).tobytes()
    ).hexdigest()
    if rendered_action_sha256 != attempt["action_sha256"]:
        raise ValueError(
            "winner RGB-D trajectory differs from its selected lightweight action tape"
        )


def _audit_export_state_and_progress(
    root: Path,
    *,
    card: Mapping[str, Any],
    candidate_rows: list[dict[str, Any]],
    reset_rows: list[dict[str, Any]],
    planner_dominance: Mapping[str, Any] | None,
) -> None:
    state_path = root / "export_state.json"
    progress_path = root / "progress.json"
    if _sha256(state_path) != card.get("export_state_sha256"):
        raise ValueError("export-state file identity mismatch")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA or (
        _payload_sha256(state) != state.get("payload_sha256")
    ):
        raise ValueError("export-state schema or payload checksum mismatch")
    card_search_mode = _candidate_search_mode(card)
    state_search_mode = _candidate_search_mode(state)
    if state_search_mode != card_search_mode:
        raise ValueError("export-state candidate search mode differs from dataset card")
    card_selection_mode = _selection_mode(card)
    state_selection_mode = _selection_mode(state)
    if state_selection_mode != card_selection_mode:
        raise ValueError("export-state selection settings differ from dataset card")
    if card.get("planner_dominance") != planner_dominance:
        raise ValueError("dataset-card planner-dominance contract mismatch")
    if state.get("planner_dominance") != planner_dominance:
        raise ValueError("export-state planner-dominance contract mismatch")
    for key in (
        "candidate_schema_version",
        "evaluator_identity",
        "compatibility_evidence",
        "calibration_evidence",
        "candidate_release_manifest_sha256",
        "candidate_release_provenance",
    ):
        if state.get(key) != card.get(key):
            raise ValueError(f"export-state {key} differs from dataset card")
    expected_selection_contract = _selection_contract(card_selection_mode)
    if card.get("selection_contract") != expected_selection_contract:
        raise ValueError("dataset-card selection contract does not match its mode")
    if (
        "selection_contract" in state
        and state.get("selection_contract") != expected_selection_contract
    ):
        raise ValueError("export-state selection contract does not match its mode")
    for payload_name, payload in (("dataset card", card), ("export state", state)):
        if "candidate_pool_size" in payload and int(
            payload["candidate_pool_size"]
        ) != len(candidate_rows):
            raise ValueError(f"{payload_name} candidate pool size mismatch")
        if (
            card_search_mode == FULL_POOL_SEARCH_MODE
            and "candidate_pool_size" not in payload
        ):
            raise ValueError(f"full-pool {payload_name} is missing candidate pool size")
    expected_state_values = {
        "task": card["task"],
        "split": card["split"],
        "manifest_seed": card["manifest_seed"],
        "accepted_target": card["accepted_target"],
        "initial_k": card["initial_k"],
        "max_k": card["max_k"],
        "budget_sequence": card["budget_sequence"],
        "image_size": card["image_size"],
        "device": card["device"],
        "candidate_manifest_sha256": card["candidate_manifest_sha256"],
        "reset_manifest_sha256": card["reset_manifest_sha256"],
        "source_identity": card["source_identity"],
        "quality_v2_threshold_identity": card["quality_v2_threshold_identity"],
    }
    if any(state.get(key) != value for key, value in expected_state_values.items()):
        raise ValueError("export-state identity does not match the dataset card")
    if int(state.get("max_resets", -1)) != len(reset_rows):
        raise ValueError("export-state max_resets does not match reset manifest")
    state_candidates = state.get("candidates")
    if not isinstance(state_candidates, list) or len(state_candidates) != len(
        candidate_rows
    ):
        raise ValueError("export-state candidate inventory mismatch")
    for index, (state_row, manifest_row) in enumerate(
        zip(state_candidates, candidate_rows, strict=True)
    ):
        if not isinstance(state_row, dict):
            raise ValueError(f"export-state candidate {index} is not a mapping")
        for key in (
            "candidate_id",
            "kind",
            "policy_sha256",
            "stochastic",
            "exploration_seed_offset",
            "residual_scale",
            "provenance",
        ):
            default = (
                False
                if key == "stochastic"
                else 0
                if key == "exploration_seed_offset"
                else None
            )
            if state_row.get(key) != manifest_row.get(key, default):
                raise ValueError(f"export-state candidate {index} differs for {key}")
        policy_path = state_row.get("policy_path")
        if manifest_row["kind"] == "policy" and not isinstance(policy_path, str):
            raise ValueError(
                f"export-state candidate {index} has no resolved policy path"
            )
        if manifest_row["kind"] == "planner" and policy_path is not None:
            raise ValueError("export-state planner unexpectedly has a policy path")

    if _sha256(progress_path) != card.get("progress_sha256"):
        raise ValueError("progress file identity mismatch")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("schema_version") != PROGRESS_SCHEMA or (
        _payload_sha256(progress) != progress.get("payload_sha256")
    ):
        raise ValueError("progress schema or payload checksum mismatch")
    if progress.get("export_state_sha256") != card.get("export_state_sha256"):
        raise ValueError("progress references a different export state")
    progress_values = {
        "started_unix_s": card["started_unix_s"],
        "next_reset_index": card["attempted_reset_count"],
        "accepted_count": card["accepted_count"],
        "candidate_attempt_count": card["candidate_attempt_count"],
        "budget_histogram": card["budget_histogram"],
        "resume_count": card["resume_count"],
        "recovery_events": card["recovery_events"],
    }
    if any(progress.get(key) != value for key, value in progress_values.items()):
        raise ValueError("progress counters do not match the dataset card")
    boundaries = progress.get("file_boundaries")
    if not isinstance(boundaries, dict):
        raise ValueError("progress file boundaries are missing")
    for name in ("attempts.jsonl", "reset_results.jsonl", "winner_manifest.jsonl"):
        path = root / name
        boundary = boundaries.get(name)
        if not isinstance(boundary, dict) or boundary != {
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }:
            raise ValueError(f"progress boundary does not match final {name}")


def _audit_quality_v4_full_exports(
    root: Path, winner_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Independently re-gate every Qv4 winner HDF5 when the new path exists."""

    from examples.embodiment.dynamic_benchmark_quality_v4 import (
        QUALITY_V4_FULL_EXPORT_SUBDIRECTORY,
        audit_quality_v4_full_export,
        dataset_quality_v4_validation,
    )

    directory = root / QUALITY_V4_FULL_EXPORT_SUBDIRECTORY
    if not directory.exists():
        return {
            "enabled": False,
            "audited_count": 0,
            "thresholds_sha256": None,
            "orientation_contract_sha256": None,
            "gate_sha256": [],
        }
    episode_ids = []
    for winner in winner_rows:
        request = winner.get("request")
        episode_id = request.get("episode_id") if isinstance(request, Mapping) else None
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Qv4 winner has no episode identity")
        episode_ids.append(episode_id)
    expected_files = {
        *(f"{episode_id}.h5" for episode_id in episode_ids),
        *(f"{episode_id}.gate.json" for episode_id in episode_ids),
    }
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("Qv4 full-export file inventory mismatch")
    gate_hashes = []
    threshold_hashes = set()
    orientation_hashes = set()
    for episode_id in episode_ids:
        export_path = directory / f"{episode_id}.h5"
        recorded_gate = json.loads(
            (directory / f"{episode_id}.gate.json").read_text(encoding="utf-8")
        )
        recomputed_gate = audit_quality_v4_full_export(export_path)
        if recorded_gate != recomputed_gate:
            raise ValueError("Qv4 recorded full-export gate does not recompute")
        dataset_gate = dataset_quality_v4_validation(recomputed_gate)
        if not dataset_gate["passed"]:
            raise ValueError("Qv4 winner is not behavior-cloning eligible")
        gate_hashes.append(recomputed_gate["gate_sha256"])
        threshold_hashes.add(recomputed_gate["thresholds_sha256"])
        orientation_hashes.add(recomputed_gate["orientation_contract_sha256"])
    if len(threshold_hashes) != 1 or len(orientation_hashes) != 1:
        raise ValueError("Qv4 winners mix threshold or orientation contracts")
    return {
        "enabled": True,
        "audited_count": len(episode_ids),
        "thresholds_sha256": next(iter(threshold_hashes)),
        "orientation_contract_sha256": next(iter(orientation_hashes)),
        "gate_sha256": gate_hashes,
    }


def _audit_dataset(
    *,
    root: Path,
    expected_card_sha256: str,
    expected_checksums_sha256: str,
    expected_candidate_sha256: str,
    expected_quality_v2_thresholds_sha256: str,
    qv4_disabled_nonblocking: bool = False,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    card_path = root / "dataset_card.json"
    if _sha256(card_path) != expected_card_sha256:
        raise ValueError("dataset-card file identity mismatch")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("schema_version") != EXPORT_SCHEMA:
        raise ValueError("dataset-card schema mismatch")
    if _payload_sha256(card) != card.get("payload_sha256"):
        raise ValueError("dataset-card payload checksum mismatch")
    if card.get("status") != "complete" or card.get("training_eligible") is not False:
        raise ValueError("only complete, unaudited exports can enter independent audit")
    checksum_entries = _verify_root_checksums(root, expected_checksums_sha256)
    candidate_path = root / "candidate_manifest.json"
    if _sha256(candidate_path) != expected_candidate_sha256:
        raise ValueError("candidate-manifest file identity mismatch")
    if card.get("candidate_manifest_sha256") != expected_candidate_sha256:
        raise ValueError("dataset card candidate-manifest identity mismatch")
    thresholds_path = root / "quality_v2_thresholds.json"
    if _sha256(thresholds_path) != expected_quality_v2_thresholds_sha256:
        raise ValueError("quality-v2 threshold file identity mismatch")
    quality_v2_thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if not isinstance(quality_v2_thresholds, Mapping):
        raise ValueError("quality-v2 threshold contract must be a mapping")
    if quality_v2_thresholds.get("formal_freeze_eligible") is not True:
        raise ValueError(
            "quality-v2 threshold contract is not eligible for formal freeze"
        )
    expected_threshold_identity = {
        "schema_version": quality_v2_thresholds.get("schema_version"),
        "sha256": expected_quality_v2_thresholds_sha256,
    }
    if card.get("quality_v2_threshold_identity") != expected_threshold_identity:
        raise ValueError("dataset card quality-v2 threshold identity mismatch")
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = _candidate_rows(candidate_payload, card=card)
    candidate_search_mode = _candidate_search_mode(card)
    selection_mode = _selection_mode(card)
    planner_dominance = _planner_dominance_contract(
        candidate_payload,
        task=str(card["task"]),
        selection_mode=selection_mode,
    )
    quality_v2_dominance = (
        _quality_v2_dominance_contract(
            quality_v2_thresholds,
            task=str(card["task"]),
            thresholds_sha256=expected_quality_v2_thresholds_sha256,
        )
        if selection_mode == PLANNER_PARETO_SELECTION_MODE
        else None
    )
    if (
        candidate_payload.get("schema_version") == CANDIDATE_SCHEMA_V2
        or "candidate_schema_version" in card
    ) and card.get("candidate_schema_version") != candidate_payload.get(
        "schema_version"
    ):
        raise ValueError("dataset-card candidate schema version mismatch")
    evaluator_identity = _audit_evaluator_identity(
        root,
        card=card,
        candidate_payload=candidate_payload,
        planner_dominance=planner_dominance,
    )
    source_identity = card["source_identity"]
    if not isinstance(source_identity, Mapping):
        raise ValueError("dataset source identity is missing")
    calibration_benchmark_commit = (
        _full_commit(
            "dataset benchmark commit",
            source_identity.get("benchmark_commit"),
        )
        if evaluator_identity is None
        else evaluator_identity["evaluator_benchmark_commit"]
    )
    calibration_receipt_identity = _audit_quality_v2_calibration_receipt_artifact(
        root,
        quality_v2_thresholds,
        expected_benchmark_commit=calibration_benchmark_commit,
    )
    _audit_candidate_release_chain(
        root,
        card=card,
        candidate_payload=candidate_payload,
        candidate_manifest_sha256=expected_candidate_sha256,
    )
    calibration_drifts = _audit_calibration_evidence(
        root,
        card=card,
        planner_dominance=planner_dominance,
        evaluator_identity=evaluator_identity,
    )
    if (
        selection_mode == PLANNER_PARETO_SELECTION_MODE
        and candidate_search_mode != FULL_POOL_SEARCH_MODE
    ):
        raise ValueError("planner-pareto dataset did not use full-pool search")
    selection_contract = _selection_contract(selection_mode)
    if card.get("selection_contract") != selection_contract:
        raise ValueError("dataset-card selection contract mismatch")

    reset_rows = _jsonl(root / "reset_manifest.jsonl")
    attempt_rows = _jsonl(root / "attempts.jsonl")
    reset_results = _jsonl(root / "reset_results.jsonl")
    winner_rows = _jsonl(root / "winner_manifest.jsonl")
    _audit_export_state_and_progress(
        root,
        card=card,
        candidate_rows=candidates,
        reset_rows=reset_rows,
        planner_dominance=planner_dominance,
    )
    if _sha256(root / "reset_manifest.jsonl") != card.get("reset_manifest_sha256"):
        raise ValueError("reset-manifest identity mismatch")
    attempted_reset_count = int(card["attempted_reset_count"])
    if (
        len(reset_rows) < attempted_reset_count
        or len(reset_results) != attempted_reset_count
    ):
        raise ValueError("reset manifest/results count mismatch")
    if len(attempt_rows) != int(card["candidate_attempt_count"]):
        raise ValueError("attempt count does not match the dataset card")
    if len(winner_rows) != int(card["accepted_count"]):
        raise ValueError("winner count does not match the dataset card")
    if int(card["accepted_count"]) != int(card["accepted_target"]):
        raise ValueError("completed export did not meet its accepted target")
    budgets = [int(value) for value in card.get("budget_sequence", [])]
    if (
        not budgets
        or budgets[0] != int(card["initial_k"])
        or budgets[-1] != int(card["max_k"])
    ):
        raise ValueError("dataset-card budget sequence is malformed")
    if candidate_search_mode == FIRST_ELIGIBLE_SEARCH_MODE:
        if any(
            right != min(int(card["max_k"]), left * 2)
            for left, right in zip(budgets, budgets[1:])
        ):
            raise ValueError("dataset-card budget escalation is malformed")
    elif budgets != [len(candidates)]:
        raise ValueError("full-pool dataset did not freeze the complete candidate pool")

    reset_by_episode: dict[str, dict[str, Any]] = {}
    for row in reset_rows:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in reset_by_episode:
            raise ValueError("reset episode IDs must be non-empty and unique")
        if row.get("task_id") != card["task"] or row.get("split") != card["split"]:
            raise ValueError("reset task/split identity mismatch")
        reset_by_episode[episode_id] = row
    attempts_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    referenced_tapes: set[str] = set()
    for record in attempt_rows:
        episode_id = record.get("episode_id")
        if episode_id not in reset_by_episode:
            raise ValueError("attempt references an unknown reset")
        reset = reset_by_episode[episode_id]
        candidate_index = int(record.get("candidate_index", -1))
        if not 0 <= candidate_index < len(candidates):
            raise ValueError("attempt candidate index is outside the frozen pool")
        candidate = candidates[candidate_index]
        if (
            record.get("candidate_id") != candidate["candidate_id"]
            or record.get("candidate_kind") != candidate["kind"]
        ):
            raise ValueError("attempt candidate identity mismatch")
        for key in (
            "episode_id",
            "task_id",
            "seed",
            "factors",
            "source_group_id",
            "pair_id",
            "pair_member_id",
        ):
            if record.get(key) != reset.get(key):
                raise ValueError(f"attempt reset identity mismatch for {key}")
        if record.get("candidate_manifest_index") != reset.get("candidate_index"):
            raise ValueError("attempt reset candidate-index provenance mismatch")
        _audit_attempt_tape(
            root,
            record,
            expected_task=str(card["task"]),
            quality_v2_thresholds=quality_v2_thresholds,
            quality_v2_thresholds_sha256=expected_quality_v2_thresholds_sha256,
        )
        tape = str(record["attempt_tape"])
        if tape in referenced_tapes:
            raise ValueError("multiple attempts reference the same lightweight tape")
        referenced_tapes.add(tape)
        attempts_by_episode[str(episode_id)].append(record)
    lightweight_root = root / "lightweight"
    actual_lightweight_files = (
        {
            path.relative_to(root).as_posix()
            for path in lightweight_root.rglob("*")
            if path.is_file()
        }
        if lightweight_root.exists()
        else set()
    )
    if actual_lightweight_files != referenced_tapes:
        raise ValueError(
            "lightweight tape inventory contains missing or unreferenced files"
        )

    winner_by_episode: dict[str, dict[str, Any]] = {}
    for winner in winner_rows:
        request = winner.get("request")
        episode_id = request.get("episode_id") if isinstance(request, dict) else None
        if not isinstance(episode_id, str) or episode_id in winner_by_episode:
            raise ValueError("winner episode IDs must be non-empty and unique")
        winner_by_episode[episode_id] = winner
    quality_v4_full_exports = _audit_quality_v4_full_exports(root, winner_rows)

    accepted = 0
    render_parity_skips: Counter[str] = Counter()
    skip_events = _render_parity_skip_events(card.get("recovery_events"))
    consumed_skip_events: set[int] = set()
    budget_histogram: Counter[str] = Counter()
    selection_source_counts: Counter[str] = Counter()
    accepted_selection_source_counts: Counter[str] = Counter()
    strict_improvement_counts: Counter[str] = Counter()
    non_strict_non_worse_counts: Counter[str] = Counter()
    for reset_index, result in enumerate(reset_results):
        reset = reset_rows[reset_index]
        episode_id = reset["episode_id"]
        if (
            result.get("reset_index") != reset_index
            or result.get("episode_id") != episode_id
        ):
            raise ValueError("reset-result order or identity mismatch")
        for key in ("source_group_id",):
            if result.get(key) != reset.get(key):
                raise ValueError(f"reset-result identity mismatch for {key}")
        records = attempts_by_episode.get(episode_id, [])
        candidate_count = int(result["candidate_count"])
        budget_used = int(result["budget_used"])
        if candidate_count != budget_used or budget_used not in budgets:
            raise ValueError("reset-result candidate count/budget mismatch")
        if (
            "candidate_search_mode" in result
            and result.get("candidate_search_mode") != candidate_search_mode
        ):
            raise ValueError("reset-result candidate-search mode mismatch")
        if (
            "selection_mode" in result
            and result.get("selection_mode") != selection_mode
        ):
            raise ValueError("reset-result selection mode mismatch")
        if candidate_search_mode == FULL_POOL_SEARCH_MODE and (
            result.get("candidate_search_mode") != FULL_POOL_SEARCH_MODE
            or candidate_count != len(candidates)
        ):
            raise ValueError("full-pool reset did not attempt every candidate")
        if len(records) != candidate_count:
            raise ValueError("attempt rows do not match reset-result candidate count")
        if [int(row["candidate_index"]) for row in records] != list(
            range(candidate_count)
        ):
            raise ValueError("attempted candidates are not the frozen pool prefix")
        budget_position = budgets.index(budget_used)
        if candidate_search_mode == FIRST_ELIGIBLE_SEARCH_MODE:
            for previous_budget in budgets[:budget_position]:
                if (
                    _selected(
                        records[:previous_budget],
                        selection_mode=selection_mode,
                        planner_dominance=planner_dominance,
                        quality_v2_dominance=quality_v2_dominance,
                    )
                    is not None
                ):
                    raise ValueError(
                        "candidate search escalated after already finding an eligible winner"
                    )
        selected = _selected(
            records,
            selection_mode=selection_mode,
            planner_dominance=planner_dominance,
            quality_v2_dominance=quality_v2_dominance,
        )
        expected_selection_result = _selection_result(
            records,
            selected,
            selection_mode=selection_mode,
        )
        if (
            selection_mode == PLANNER_PARETO_SELECTION_MODE
            or "selection_result" in result
        ) and result.get("selection_result") != expected_selection_result:
            raise ValueError("reset-result selection provenance does not recompute")
        selection_source_counts[expected_selection_result["source_kind"]] += 1
        budget_histogram[str(budget_used)] += 1
        winner = winner_by_episode.get(episode_id)
        accepted_label = result.get("accepted")
        if not isinstance(accepted_label, bool):
            raise ValueError("reset-result accepted must be boolean")
        if selected is None:
            if accepted_label or result.get("render_parity_skip") is not None:
                raise ValueError("reset-result acceptance does not recompute")
            if reset_index in skip_events:
                raise ValueError("render-parity recovery event has no selected attempt")
            if budget_used != budgets[-1] or winner is not None:
                raise ValueError(
                    "rejected reset did not exhaust max_k or unexpectedly has a winner"
                )
            if (
                result.get("winner_candidate_id") is not None
                or result.get("winner_candidate_index") is not None
            ):
                raise ValueError("rejected reset names a winner")
            continue
        if not accepted_label:
            if winner is not None:
                raise ValueError(
                    "render-parity skipped reset unexpectedly has a winner"
                )
            if (
                result.get("winner_candidate_id") is not None
                or result.get("winner_candidate_index") is not None
            ):
                raise ValueError("render-parity skipped reset names a winner")
            protocol = _audit_render_parity_skip(
                result,
                selected,
                skip_events.get(reset_index),
            )
            render_parity_skips[protocol] += 1
            consumed_skip_events.add(reset_index)
            continue
        if result.get("render_parity_skip") is not None or reset_index in skip_events:
            raise ValueError(
                "accepted reset unexpectedly carries render-parity skip evidence"
            )
        if (
            result.get("winner_candidate_id") != selected["candidate_id"]
            or int(result.get("winner_candidate_index", -1))
            != int(selected["candidate_index"])
            or winner is None
            or winner.get("candidate_id") != selected["candidate_id"]
            or int(winner.get("candidate_index", -1))
            != int(selected["candidate_index"])
            or int(winner.get("candidate_count", -1)) != candidate_count
            or int(winner.get("budget_used", -1)) != budget_used
            or winner.get("selection_contract") != selection_contract
            or winner.get("quality_score") != list(_quality_score(selected))
            or winner.get("lightweight_attempt_tape") != selected["attempt_tape"]
            or winner.get("lightweight_attempt_tape_sha256")
            != selected["attempt_tape_sha256"]
        ):
            raise ValueError(
                "published winner does not match independently selected attempt"
            )
        if (
            selection_mode == PLANNER_PARETO_SELECTION_MODE
            or "selection_result" in winner
        ) and winner.get("selection_result") != expected_selection_result:
            raise ValueError("published winner selection provenance mismatch")
        if (
            "candidate_search_mode" in winner
            and winner.get("candidate_search_mode") != candidate_search_mode
        ):
            raise ValueError("published winner candidate-search mode mismatch")
        if (
            "selection_mode" in winner
            and winner.get("selection_mode") != selection_mode
        ):
            raise ValueError("published winner selection mode mismatch")
        if winner.get("planner_dominance") != planner_dominance:
            raise ValueError("published winner planner-dominance contract mismatch")
        for key in (
            "evaluator_identity",
            "compatibility_evidence",
            "calibration_evidence",
            "candidate_release_manifest_sha256",
        ):
            if winner.get(key) != card.get(key):
                raise ValueError(f"published winner {key} mismatch")
        source_kind = expected_selection_result["source_kind"]
        accepted_selection_source_counts[source_kind] += 1
        if (
            planner_dominance is not None
            and source_kind == "expert_dominant"
            and expected_selection_result["planner_eligible"]
        ):
            planner = records[0]
            relations = _planner_metric_relations(
                selected,
                planner,
                planner_dominance,
                quality_v2_dominance,
            )
            if "regressed" in relations.values():
                raise ValueError("accepted expert regresses an eligible planner metric")
            for metric_name, relation in relations.items():
                if relation == "strictly_improved":
                    strict_improvement_counts[metric_name] += 1
                else:
                    non_strict_non_worse_counts[metric_name] += 1
        _audit_winner_episode(
            root,
            winner,
            selected,
            reset,
            card,
            planner_dominance,
            qv4_disabled_nonblocking=qv4_disabled_nonblocking,
        )
        accepted += 1
    if consumed_skip_events != set(skip_events):
        raise ValueError("render-parity recovery event inventory mismatch")
    if accepted != len(winner_rows) or set(winner_by_episode) != {
        row["episode_id"] for row in reset_results if row["accepted"]
    }:
        raise ValueError("winner/reset-result inventory mismatch")
    if dict(budget_histogram) != {
        str(key): int(value) for key, value in card["budget_histogram"].items()
    }:
        raise ValueError("budget histogram does not recompute")
    actual_episode_dirs = {
        path.parent.relative_to(root).as_posix()
        for path in (root / "episodes").rglob("episode.json")
    }
    declared_episode_dirs = {
        str(winner["relative_episode_dir"]) for winner in winner_rows
    }
    if actual_episode_dirs != declared_episode_dirs:
        raise ValueError("winner episode directory inventory mismatch")
    staging_root = root / ".staging"
    if staging_root.exists() and any(staging_root.iterdir()):
        raise ValueError("dataset contains unpublished staging content")
    return {
        "dataset_card_payload_sha256": card["payload_sha256"],
        "task": card["task"],
        "split": card["split"],
        "source_identity": card["source_identity"],
        "quality_v2_threshold_identity": expected_threshold_identity,
        "quality_v2_calibration_wave_receipt_identity": (calibration_receipt_identity),
        "quality_v4_full_exports": quality_v4_full_exports,
        "quality_v4_policy": (
            "nonblocking_not_exported"
            if qv4_disabled_nonblocking
            else "behavior_cloning_required"
        ),
        "accepted_count": accepted,
        "attempted_reset_count": len(reset_results),
        "candidate_attempt_count": len(attempt_rows),
        "candidate_pool_size": len(candidates),
        "candidate_search_mode": candidate_search_mode,
        "selection_mode": selection_mode,
        "planner_dominance": planner_dominance,
        "candidate_schema_version": candidate_payload["schema_version"],
        "evaluator_identity": evaluator_identity,
        "candidate_release_manifest_sha256": card.get(
            "candidate_release_manifest_sha256"
        ),
        "calibration_observed_drifts": calibration_drifts,
        "selection_source_counts": dict(selection_source_counts),
        "accepted_selection_source_counts": dict(accepted_selection_source_counts),
        "strict_improvement_counts": dict(strict_improvement_counts),
        "non_strict_non_worse_counts": dict(non_strict_non_worse_counts),
        "checksum_entry_count": checksum_entries,
        "budget_histogram": dict(budget_histogram),
        "render_parity_skip_count": sum(render_parity_skips.values()),
        "render_parity_skip_protocols": dict(render_parity_skips),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    auditor_commit = _full_commit("auditor_commit", args.auditor_commit)
    expected_card = _expected_sha256(
        args.expected_dataset_card_sha256,
        "expected dataset-card SHA-256",
    )
    expected_checksums = _expected_sha256(
        args.expected_checksums_sha256,
        "expected SHA256SUMS SHA-256",
    )
    expected_candidate = _expected_sha256(
        args.expected_candidate_manifest_sha256,
        "expected candidate-manifest SHA-256",
    )
    expected_quality_v2_thresholds = _expected_sha256(
        args.expected_quality_v2_thresholds_sha256,
        "expected quality-v2 threshold SHA-256",
    )
    started = time.time()
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_card_sha256": expected_card,
        "checksums_sha256": expected_checksums,
        "candidate_manifest_sha256": expected_candidate,
        "quality_v2_thresholds_sha256": expected_quality_v2_thresholds,
        "auditor_commit": auditor_commit,
        "quality_v4_policy": (
            "nonblocking_not_exported"
            if args.qv4_disabled_nonblocking
            else "behavior_cloning_required"
        ),
        "started_unix_s": started,
    }
    try:
        summary = _audit_dataset(
            root=args.dataset_root.resolve(),
            expected_card_sha256=expected_card,
            expected_checksums_sha256=expected_checksums,
            expected_candidate_sha256=expected_candidate,
            expected_quality_v2_thresholds_sha256=expected_quality_v2_thresholds,
            qv4_disabled_nonblocking=args.qv4_disabled_nonblocking,
        )
        report.update(
            status="passed",
            training_eligible=True,
            training_eligibility_reason="independent audit passed",
            summary=summary,
        )
    except Exception as error:
        report.update(
            status="failed",
            training_eligible=False,
            training_eligibility_reason="independent audit failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        report["finished_unix_s"] = time.time()
        report["payload_sha256"] = _payload_sha256(report)
        _atomic_json(args.output, report)
        raise
    report["finished_unix_s"] = time.time()
    report["payload_sha256"] = _payload_sha256(report)
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
