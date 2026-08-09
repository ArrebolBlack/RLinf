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

"""Export the fixed RLD2-QA paired owner-review cohort.

This entrypoint is deliberately separate from production trajectory generation.
It always evaluates exactly twenty validation-domain resets in the ``review``
partition, runs the planner and every promoted policy deterministically on every
reset, keeps every lightweight tape, and then selects six category-specific
planner/comparison pairs.  A missing category seals an incomplete diagnostic
export and exits non-zero; it never samples replacement resets.

Heavy RLinf/SE3-WAM imports are intentionally delayed until :func:`main` so the
selection and provenance contracts remain usable as pure unit-test fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REVIEW_SCHEMA = "rlinf-dynamic-benchmark-rld2-paired-review-v0.2"
HISTORICAL_REVIEW_SCHEMA = "rlinf-dynamic-benchmark-rld2-paired-review-v0.1"
REVIEW_CANDIDATE_CONTRACT_SCHEMA = "rld2-qa-review-candidate-contract-v0.1"
PROMOTION_SCHEMA = "rld2-qa-policy-promotion-v0.2"
HISTORICAL_PROMOTION_SCHEMA = "rld2-qa-policy-promotion-v0.1"
PROMOTION_EVIDENCE_SCHEMA = "rld2-qa-policy-selection-evidence-v0.1"
QUALITY_V2_THRESHOLD_SCHEMA = "se3-wam-trajectory-quality-v2-thresholds-v0.3"
ATTEMPT_INDEX_SCHEMA = "rlinf-dynamic-benchmark-rld2-review-attempt-v0.1"
PAIR_TRACE_SCHEMA = "rlinf-dynamic-benchmark-rld2-paired-trace-v0.1"
PARTITION = "review"
ENVIRONMENT_SPLIT = "validation"
MANIFEST_SEED = 20261250
RESET_COUNT = 20
INFERENCE_MODE = "deterministic_mean"
SEARCH_MODE = "full-pool"
T5_CAUSAL_LATENCY_TOLERANCE_S = 1.0e-9
RELEASE_HANDOFF_BLOCKERS: tuple[Mapping[str, str], ...] = ()
CATEGORIES = (
    "planner_anchor",
    "learned_representative",
    "smoothness_boundary",
    "orientation_boundary",
    "rejected_jitter_or_tilt",
    "planner_rl_disagreement",
)
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_PROMOTION_KEYS = {
    "schema_version",
    "task_id",
    "candidate_id",
    "policy",
    "inference",
    "validation_receipt",
    "quality_v2_calibration_wave_receipt",
    "selection",
    "source_identity",
    "image_identity",
    "payload_sha256",
}
_CALIBRATION_RECEIPT_KEYS = {
    "path",
    "sha256",
    "schema_version",
    "payload_sha256",
    "dataset_relative_path",
}
_POLICY_KEYS = {
    "path",
    "sha256",
    "seed",
    "run_tag",
    "checkpoint_role",
    "env_steps",
    "metadata_path",
    "metadata_sha256",
    "metadata_payload_sha256",
}
_INFERENCE_KEYS = {"residual_scale", "deterministic", "action_noise"}
_VALIDATION_KEYS = {
    "path",
    "sha256",
    "payload_sha256",
    "evaluator_schema",
    "evaluator_source_sha256",
    "partition",
    "reset_manifest_path",
    "reset_manifest_sha256",
    "test_exposure",
    "review_exposure",
    "calibration_exposure",
    "quality_threshold_schema",
    "quality_threshold_sha256",
    "attempt_schema_version",
    "all_successful_quality_gates_passed",
    "all_successful_t5_causal_gates_passed",
}
_SELECTION_KEYS = {
    "decision",
    "reason",
    "planner_nonworse_all_dimensions",
    "strict_improvement_dimensions",
    "rejection_reasons",
    "selector_contract_sha256",
    "selector_contract_path",
    "planner_evaluation_path",
    "planner_evaluation_sha256",
    "planner_evaluation_payload_sha256",
    "attempt_artifacts_payload_sha256",
    "evidence_path",
    "evidence_sha256",
    "evidence_payload_sha256",
}
_SOURCE_IDENTITY_KEYS = {
    "rlinf_commit",
    "benchmark_commit",
    "evaluator_rlinf_commit",
    "files",
    "sha256",
}
_SOURCE_FILE_KEYS = {
    "policy_evaluator",
    "planner_evaluator",
    "qv3_comparator",
    "attempt_auditor",
    "promotion_builder",
}
_SOURCE_FILE_IDENTITY_KEYS = {"path", "sha256"}
_IMAGE_IDENTITY_KEYS = {"reference", "sha256"}
_PAIRED_THRESHOLD_FIELDS = (
    "paired_nonworse_absolute_tolerance",
    "paired_nonworse_relative_tolerance",
    "paired_strict_improvement_absolute",
    "paired_strict_improvement_relative",
)
_REVIEW_CONTRACT_KEYS = {
    "schema_version",
    "partition",
    "environment_split",
    "manifest_seed",
    "reset_count",
    "inference_mode",
    "candidate_search_mode",
    "full_generation",
    "quality_v2_threshold_schema",
    "quality_v2_thresholds_sha256",
    "evaluator_commit",
    "evaluator_benchmark_commit",
    "evaluator_source_sha256",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(
    value: Any, label: str, keys: set[str] | None = None
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if keys is not None and not keys.issubset(value):
        raise ValueError(f"{label} field inventory mismatch")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_sha256(value: Any, label: str) -> str:
    result = _require_string(value, label)
    if _HEX_64.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _require_commit(value: Any, label: str) -> str:
    result = _require_string(value, label)
    if _HEX_40.fullmatch(result) is None:
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return result


def _require_optional_scale(value: Any, label: str) -> float | None:
    if value is None:
        return None
    result = _require_number(value, label)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{label} must be in (0, 1]")
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(_canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    _canonical_json(payload)
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid {label} JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} must be an object")
        rows.append(row)
    return rows


def _resolve_reference(authority_path: Path, reference: str) -> Path:
    value = Path(_require_string(reference, "referenced path"))
    return value if value.is_absolute() else (authority_path.parent / value).resolve()


def _validate_promotion_receipt_payload(
    payload: Mapping[str, Any],
    *,
    task: str,
    candidate_id: str,
    policy_path: str,
    policy_sha256: str,
    residual_scale: float | None,
    rlinf_commit: str,
    benchmark_commit: str,
    threshold_schema: str,
    threshold_sha256: str,
    calibration_receipt_identity: Mapping[str, str],
    evaluator_rlinf_commit: str,
    evaluator_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the immutable policy-level promotion decision without I/O."""

    receipt = dict(_require_mapping(payload, "promotion receipt", _PROMOTION_KEYS))
    if receipt.get("schema_version") == HISTORICAL_PROMOTION_SCHEMA:
        raise ValueError("historical promotion schema v0.1 is not production evidence")
    if receipt.get("schema_version") != PROMOTION_SCHEMA:
        raise ValueError("promotion receipt schema mismatch")
    if receipt.get("task_id") != task or receipt.get("candidate_id") != candidate_id:
        raise ValueError("promotion receipt task/candidate identity mismatch")
    receipt_hash = _require_sha256(
        receipt.get("payload_sha256"), "promotion payload SHA-256"
    )
    if receipt_hash != _payload_sha256(receipt):
        raise ValueError("promotion receipt payload SHA-256 does not recompute")

    policy = dict(
        _require_mapping(receipt.get("policy"), "promotion policy", _POLICY_KEYS)
    )
    if policy.get("path") != policy_path:
        raise ValueError("promotion policy path mismatch")
    if (
        _require_sha256(policy.get("sha256"), "promotion policy SHA-256")
        != policy_sha256
    ):
        raise ValueError("promotion policy SHA-256 mismatch")
    _require_int(policy.get("seed"), "promotion policy seed")
    _require_string(policy.get("run_tag"), "promotion policy run tag")
    _require_string(policy.get("metadata_path"), "promotion policy metadata path")
    _require_sha256(policy.get("metadata_sha256"), "promotion policy metadata SHA-256")
    _require_sha256(
        policy.get("metadata_payload_sha256"),
        "promotion policy metadata payload SHA-256",
    )
    if policy.get("checkpoint_role") != "best":
        raise ValueError("only checkpoint_role=best can be promoted for review")
    if "final" in Path(policy_path).name.lower():
        raise ValueError("final checkpoints cannot be learned review candidates")
    _require_int(policy.get("env_steps"), "promotion policy env_steps", minimum=1)

    inference = dict(
        _require_mapping(
            receipt.get("inference"), "promotion inference", _INFERENCE_KEYS
        )
    )
    if (
        _require_bool(inference.get("deterministic"), "promotion deterministic")
        is not True
    ):
        raise ValueError("promoted review policies must be deterministic")
    if (
        _require_bool(inference.get("action_noise"), "promotion action_noise")
        is not False
    ):
        raise ValueError("promoted review policies must disable action noise")
    validated_scale = _require_optional_scale(
        inference.get("residual_scale"), "promotion residual_scale"
    )
    if validated_scale != residual_scale:
        raise ValueError(
            "candidate residual_scale was not validated by its promotion receipt"
        )

    validation = dict(
        _require_mapping(
            receipt.get("validation_receipt"),
            "promotion validation receipt",
            _VALIDATION_KEYS,
        )
    )
    _require_string(validation.get("path"), "validation receipt path")
    _require_sha256(validation.get("sha256"), "validation receipt SHA-256")
    _require_sha256(
        validation.get("payload_sha256"), "validation receipt payload SHA-256"
    )
    _require_string(validation.get("evaluator_schema"), "validation evaluator schema")
    validation_source = _require_sha256(
        validation.get("evaluator_source_sha256"),
        "validation evaluator source SHA-256",
    )
    if (
        evaluator_source_sha256 is not None
        and validation_source != evaluator_source_sha256
    ):
        raise ValueError("promotion evaluator source SHA-256 mismatch")
    if validation.get("partition") != "validation":
        raise ValueError("promotion must be selected on the validation partition")
    _require_string(
        validation.get("reset_manifest_path"), "selection reset manifest path"
    )
    _require_sha256(
        validation.get("reset_manifest_sha256"), "selection reset manifest SHA-256"
    )
    if _require_bool(validation.get("test_exposure"), "promotion test_exposure"):
        raise ValueError("test-exposed policies cannot be promoted for review")
    if _require_bool(validation.get("review_exposure"), "promotion review_exposure"):
        raise ValueError("review-exposed policies cannot be promoted for review")
    if _require_bool(
        validation.get("calibration_exposure"), "promotion calibration_exposure"
    ):
        raise ValueError("calibration-exposed policies cannot be promoted for review")
    if (
        validation.get("quality_threshold_schema") != threshold_schema
        or validation.get("quality_threshold_sha256") != threshold_sha256
    ):
        raise ValueError("promotion quality-threshold identity mismatch")
    from examples.embodiment import (
        audit_dynamic_benchmark_optimal_trajectories as optimal_auditor,
    )

    if validation.get("attempt_schema_version") != optimal_auditor.ATTEMPT_SCHEMA:
        raise ValueError("promotion attempt schema is not canonical v0.3")
    if (
        _require_bool(
            validation.get("all_successful_quality_gates_passed"),
            "promotion successful quality gates",
        )
        is not True
    ):
        raise ValueError(
            "promotion requires all successful validation quality gates to pass"
        )
    if (
        _require_bool(
            validation.get("all_successful_t5_causal_gates_passed"),
            "promotion successful T5 causal gates",
        )
        is not True
    ):
        raise ValueError("promotion requires all successful T5 causal gates to pass")

    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    calibration_receipt = dict(
        _require_mapping(
            receipt.get("quality_v2_calibration_wave_receipt"),
            "promotion quality-v2 calibration wave receipt",
            _CALIBRATION_RECEIPT_KEYS,
        )
    )
    if set(calibration_receipt) != _CALIBRATION_RECEIPT_KEYS:
        raise ValueError(
            "promotion quality-v2 calibration wave receipt field inventory mismatch"
        )
    _require_string(
        calibration_receipt.get("path"), "promotion calibration receipt path"
    )
    if (
        _require_sha256(
            calibration_receipt.get("sha256"),
            "promotion calibration receipt SHA-256",
        )
        != calibration_receipt_identity.get("file_sha256")
        or _require_sha256(
            calibration_receipt.get("payload_sha256"),
            "promotion calibration receipt payload SHA-256",
        )
        != calibration_receipt_identity.get("payload_sha256")
        or calibration_receipt.get("dataset_relative_path")
        != calibration_receipt_identity.get("relative_path")
        or calibration_receipt.get("schema_version")
        != optimal.QUALITY_V2_CALIBRATION_WAVE_RECEIPT_SCHEMA
    ):
        raise ValueError(
            "promotion calibration receipt identity does not match the review sidecar"
        )

    selection = dict(
        _require_mapping(
            receipt.get("selection"), "promotion selection", _SELECTION_KEYS
        )
    )
    if set(selection) != _SELECTION_KEYS:
        raise ValueError("promotion selection field inventory mismatch")
    reason = _require_string(selection.get("reason"), "promotion selection reason")
    rejection_reasons = selection.get("rejection_reasons")
    if not isinstance(rejection_reasons, list):
        raise ValueError("promotion selection rejection_reasons must be a list")
    if selection.get("decision") != "promote":
        raise ValueError("candidate has no promote decision")
    if reason != "strict_planner_nonworse_improvement":
        raise ValueError("promote decision has noncanonical selection reason")
    if rejection_reasons:
        raise ValueError("promoted candidate cannot carry formal rejection reasons")
    if (
        _require_bool(
            selection.get("planner_nonworse_all_dimensions"),
            "promotion planner nonworse decision",
        )
        is not True
    ):
        raise ValueError("promotion must be planner-nonworse on all frozen dimensions")
    improvements = selection.get("strict_improvement_dimensions")
    if (
        not isinstance(improvements, list)
        or not improvements
        or any(not isinstance(item, str) or not item.strip() for item in improvements)
        or len(improvements) != len(set(improvements))
    ):
        raise ValueError(
            "promotion requires non-empty unique strict improvement dimensions"
        )
    _require_sha256(
        selection.get("selector_contract_sha256"), "promotion selector contract SHA-256"
    )
    _require_string(
        selection.get("selector_contract_path"), "promotion selector contract path"
    )
    _require_string(
        selection.get("planner_evaluation_path"), "promotion planner evaluation path"
    )
    _require_sha256(
        selection.get("planner_evaluation_sha256"),
        "promotion planner evaluation SHA-256",
    )
    _require_sha256(
        selection.get("planner_evaluation_payload_sha256"),
        "promotion planner evaluation payload SHA-256",
    )
    _require_sha256(
        selection.get("attempt_artifacts_payload_sha256"),
        "promotion attempt artifacts payload SHA-256",
    )
    _require_string(selection.get("evidence_path"), "promotion evidence path")
    _require_sha256(selection.get("evidence_sha256"), "promotion evidence file SHA-256")
    _require_sha256(
        selection.get("evidence_payload_sha256"),
        "promotion evidence payload SHA-256",
    )

    source = dict(
        _require_mapping(
            receipt.get("source_identity"),
            "promotion source identity",
            _SOURCE_IDENTITY_KEYS,
        )
    )
    if set(source) != _SOURCE_IDENTITY_KEYS:
        raise ValueError("promotion source identity field inventory mismatch")
    if (
        _require_commit(source.get("rlinf_commit"), "promotion RLinf commit")
        != rlinf_commit
        or _require_commit(source.get("benchmark_commit"), "promotion benchmark commit")
        != benchmark_commit
        or _require_commit(
            source.get("evaluator_rlinf_commit"),
            "promotion evaluator RLinf commit",
        )
        != evaluator_rlinf_commit
    ):
        raise ValueError("promotion source commit mismatch")
    source_files = _require_mapping(source.get("files"), "promotion source files")
    if set(source_files) != _SOURCE_FILE_KEYS:
        raise ValueError("promotion source files field inventory mismatch")
    normalized_source_files: dict[str, dict[str, str]] = {}
    for name in sorted(_SOURCE_FILE_KEYS):
        identity = _require_mapping(
            source_files.get(name), f"promotion source file {name}"
        )
        if set(identity) != _SOURCE_FILE_IDENTITY_KEYS:
            raise ValueError(f"promotion source file {name} field inventory mismatch")
        normalized_source_files[name] = {
            "path": _require_string(
                identity.get("path"), f"promotion source file {name} path"
            ),
            "sha256": _require_sha256(
                identity.get("sha256"),
                f"promotion source file {name} SHA-256",
            ),
        }
    if (
        evaluator_source_sha256 is not None
        and normalized_source_files["policy_evaluator"]["sha256"]
        != evaluator_source_sha256
    ):
        raise ValueError("promotion policy evaluator source SHA-256 mismatch")
    source_sha = _require_sha256(source.get("sha256"), "promotion source SHA-256")
    unsigned_source = dict(source)
    unsigned_source.pop("sha256", None)
    if (
        source_sha
        != hashlib.sha256(_canonical_json(unsigned_source).encode("utf-8")).hexdigest()
    ):
        raise ValueError("promotion source SHA-256 does not recompute")
    image = dict(
        _require_mapping(
            receipt.get("image_identity"),
            "promotion image identity",
            _IMAGE_IDENTITY_KEYS,
        )
    )
    _require_string(image.get("reference"), "promotion image reference")
    _require_sha256(image.get("sha256"), "promotion image SHA-256")
    return receipt


def _validate_validation_receipt_file(
    promotion: Mapping[str, Any],
    *,
    promotion_path: Path,
    policy_sha256: str,
    policy_env_steps: int,
    task: str,
    rlinf_commit: str,
    evaluator_commit: str,
    evaluator_benchmark_commit: str,
    threshold_schema: str,
    threshold_sha256: str,
) -> tuple[Path, list[dict[str, Any]]]:
    """Verify the evaluator artifact and its exact checkpoint-selection resets."""

    validation = _require_mapping(promotion["validation_receipt"], "validation receipt")
    evaluation_path = _resolve_reference(promotion_path, str(validation["path"]))
    if (
        not evaluation_path.is_file()
        or _sha256(evaluation_path) != validation["sha256"]
    ):
        raise ValueError("validation receipt file is missing or has a SHA-256 mismatch")
    evaluation = _read_json(evaluation_path, "validation receipt")
    if evaluation.get("schema_version") != validation["evaluator_schema"]:
        raise ValueError("validation evaluator schema mismatch")
    if "payload_sha256" in evaluation and evaluation.get(
        "payload_sha256"
    ) != _payload_sha256(evaluation):
        raise ValueError("validation receipt payload SHA-256 does not recompute")
    if evaluation.get("payload_sha256") != validation.get("payload_sha256"):
        raise ValueError("validation receipt payload identity mismatch")
    policy = _require_mapping(
        evaluation.get("policy_identity"), "validation policy identity"
    )
    if (
        policy.get("sha256") != policy_sha256
        or policy.get("task") != task
        or policy.get("training_env_steps") != policy_env_steps
    ):
        raise ValueError("validation receipt policy identity mismatch")
    if evaluation.get("split") != ENVIRONMENT_SPLIT:
        raise ValueError("validation receipt used a non-validation split")
    source = _require_mapping(
        evaluation.get("source_identity"), "validation source identity"
    )
    if (
        source.get("evaluator_rlinf_commit") != evaluator_commit
        or source.get("policy_rlinf_commit") != rlinf_commit
        or source.get("benchmark_commit") != evaluator_benchmark_commit
    ):
        raise ValueError("validation receipt source identity mismatch")
    if evaluation.get("manifest_seed") == MANIFEST_SEED:
        raise ValueError("checkpoint selection reused the frozen review manifest seed")
    threshold = _require_mapping(
        evaluation.get("quality_v2_threshold_identity"), "validation threshold identity"
    )
    if (
        threshold.get("schema_version") != threshold_schema
        or threshold.get("sha256") != threshold_sha256
    ):
        raise ValueError("validation receipt threshold identity mismatch")
    if evaluation.get("all_replays_passed") is not True:
        raise ValueError("validation receipt did not pass all replays")
    if evaluation.get("all_successful_quality_v2_gates_passed") is not True:
        raise ValueError("validation receipt did not pass all successful quality gates")

    reset_path = _resolve_reference(
        evaluation_path, str(validation["reset_manifest_path"])
    )
    if (
        not reset_path.is_file()
        or _sha256(reset_path) != validation["reset_manifest_sha256"]
    ):
        raise ValueError(
            "selection reset manifest is missing or has a SHA-256 mismatch"
        )
    if evaluation.get("reset_manifest_sha256") != validation["reset_manifest_sha256"]:
        raise ValueError("validation receipt/reset manifest identity mismatch")
    rows = _read_jsonl(reset_path, "selection reset manifest")
    if not rows:
        raise ValueError("selection reset manifest is empty")
    for row in rows:
        if row.get("task_id") != task or row.get("split") != ENVIRONMENT_SPLIT:
            raise ValueError("selection reset manifest task/split mismatch")
    return reset_path, rows


def _validate_selection_evidence_file(
    promotion: Mapping[str, Any],
    *,
    promotion_path: Path,
    task: str,
    candidate_id: str,
    policy_path: str,
    policy_sha256: str,
    residual_scale: float | None,
    rlinf_commit: str,
    benchmark_commit: str,
    threshold_sha256: str,
    calibration_receipt_identity: Mapping[str, str],
    evaluator_source_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Recompute the promotion decision from every evidence-bound artifact."""

    from examples.embodiment import (
        build_dynamic_benchmark_rld2_promotion as promotion_builder,
    )

    selection = _require_mapping(promotion.get("selection"), "promotion selection")
    evidence_path = _resolve_reference(promotion_path, str(selection["evidence_path"]))
    if (
        not evidence_path.is_file()
        or _sha256(evidence_path) != selection["evidence_sha256"]
    ):
        raise ValueError("promotion evidence is missing or has a SHA-256 mismatch")
    evidence = _read_json(evidence_path, "policy selection evidence")
    if evidence.get("schema_version") != PROMOTION_EVIDENCE_SCHEMA:
        raise ValueError("promotion evidence schema mismatch")
    if evidence.get("payload_sha256") != selection["evidence_payload_sha256"]:
        raise ValueError("promotion evidence payload identity mismatch")
    validated = promotion_builder.validate_selection_evidence_artifacts(
        evidence,
        expected_task=task,
        expected_candidate_id=candidate_id,
        expected_policy_path=policy_path,
        expected_policy_sha256=policy_sha256,
        expected_threshold_sha256=threshold_sha256,
        expected_calibration_receipt_sha256=calibration_receipt_identity["file_sha256"],
        expected_selector_sha256=str(selection["selector_contract_sha256"]),
        expected_rlinf_commit=rlinf_commit,
        expected_benchmark_commit=benchmark_commit,
        expected_policy_evaluator_source_sha256=evaluator_source_sha256,
    )
    evidence_selection = _require_mapping(
        validated.get("selection"), "selection evidence decision"
    )
    if (
        evidence_selection.get("decision") != selection.get("decision")
        or evidence_selection.get("planner_nonworse_all_dimensions")
        != selection.get("planner_nonworse_all_dimensions")
        or evidence_selection.get("strict_improvement_dimensions")
        != selection.get("strict_improvement_dimensions")
    ):
        raise ValueError("promotion receipt decision differs from recomputed evidence")
    if validated.get("inference") != {
        "mode": "deterministic_mean",
        "deterministic": True,
        "action_noise": False,
        "residual_scale": residual_scale,
        "basis": validated["inference"]["basis"],
    }:
        raise ValueError("promotion evidence inference identity mismatch")
    if validated.get("source_identity") != promotion.get("source_identity"):
        raise ValueError("promotion receipt/evidence source identity mismatch")
    if validated.get("image_identity") != promotion.get("image_identity"):
        raise ValueError("promotion receipt/evidence image identity mismatch")
    inputs = _require_mapping(validated.get("inputs"), "selection evidence inputs")
    policy_evaluation = _require_mapping(
        inputs.get("policy_evaluation"), "selection evidence policy evaluation"
    )
    planner_evaluation = _require_mapping(
        inputs.get("planner_evaluation"), "selection evidence planner evaluation"
    )
    attempt_artifacts = _require_mapping(
        inputs.get("attempt_artifacts"), "selection evidence attempt artifacts"
    )
    reset_manifest = _require_mapping(
        inputs.get("reset_manifest"), "selection evidence reset manifest"
    )
    thresholds = _require_mapping(
        inputs.get("quality_v3_thresholds"), "selection evidence thresholds"
    )
    calibration_receipt = _require_mapping(
        inputs.get("quality_v2_calibration_wave_receipt"),
        "selection evidence calibration wave receipt",
    )
    selector = _require_mapping(
        inputs.get("selector_contract"), "selection evidence selector"
    )
    validation = _require_mapping(
        promotion.get("validation_receipt"), "promotion validation receipt"
    )
    expected_policy_evaluation = _resolve_reference(
        promotion_path, str(validation["path"])
    )
    expected_reset_manifest = _resolve_reference(
        expected_policy_evaluation, str(validation["reset_manifest_path"])
    )
    if (
        Path(str(policy_evaluation.get("path"))).resolve()
        != expected_policy_evaluation.resolve()
        or policy_evaluation.get("sha256") != validation.get("sha256")
        or policy_evaluation.get("payload_sha256") != validation.get("payload_sha256")
        or Path(str(reset_manifest.get("path"))).resolve()
        != expected_reset_manifest.resolve()
        or reset_manifest.get("sha256") != validation.get("reset_manifest_sha256")
        or thresholds.get("sha256") != validation.get("quality_threshold_sha256")
        or calibration_receipt != promotion.get("quality_v2_calibration_wave_receipt")
        or calibration_receipt.get("sha256")
        != calibration_receipt_identity["file_sha256"]
        or calibration_receipt.get("payload_sha256")
        != calibration_receipt_identity["payload_sha256"]
        or calibration_receipt.get("dataset_relative_path")
        != calibration_receipt_identity["relative_path"]
        or attempt_artifacts.get("schema_version")
        != validation.get("attempt_schema_version")
        or validated["aggregate"]["all_successful_policy_t5_causal_gates_passed"]
        != validation.get("all_successful_t5_causal_gates_passed")
        or selector.get("sha256") != selection.get("selector_contract_sha256")
        or Path(str(selector.get("path"))).resolve()
        != _resolve_reference(
            promotion_path, str(selection["selector_contract_path"])
        ).resolve()
        or Path(str(planner_evaluation.get("path"))).resolve()
        != _resolve_reference(
            promotion_path, str(selection["planner_evaluation_path"])
        ).resolve()
        or planner_evaluation.get("sha256")
        != selection.get("planner_evaluation_sha256")
        or planner_evaluation.get("payload_sha256")
        != selection.get("planner_evaluation_payload_sha256")
        or _payload_sha256(attempt_artifacts)
        != selection.get("attempt_artifacts_payload_sha256")
    ):
        raise ValueError(
            "promotion receipt does not bind the recomputed evidence inputs"
        )
    return evidence_path, validated


def _reset_identity(row: Mapping[str, Any]) -> str:
    """Identity a physical reset independently of its episode label/index."""

    required = (
        "task_id",
        "seed",
        "action_mode",
        "observation_track",
        "object_mode",
        "reset_mode",
        "factors",
    )
    if any(key not in row for key in required):
        raise ValueError("reset manifest row is missing physical reset identity fields")
    return hashlib.sha256(
        _canonical_json({key: row[key] for key in required}).encode("utf-8")
    ).hexdigest()


def _assert_disjoint_reset_rows(
    review_rows: Sequence[Mapping[str, Any]],
    selection_manifests: Sequence[Sequence[Mapping[str, Any]]],
    *,
    task: str,
) -> None:
    """Fail closed if review sees any checkpoint-selection reset."""

    if len(review_rows) != RESET_COUNT:
        raise ValueError(
            f"review reset manifest must contain exactly {RESET_COUNT} rows"
        )
    review_episode_ids: set[str] = set()
    review_identities: set[str] = set()
    for row in review_rows:
        if row.get("task_id") != task or row.get("split") != ENVIRONMENT_SPLIT:
            raise ValueError("review reset manifest task/split mismatch")
        episode_id = _require_string(row.get("episode_id"), "review episode_id")
        identity = _reset_identity(row)
        if episode_id in review_episode_ids or identity in review_identities:
            raise ValueError("review reset manifest contains duplicate resets")
        review_episode_ids.add(episode_id)
        review_identities.add(identity)
    for rows in selection_manifests:
        for row in rows:
            if row.get("task_id") != task or row.get("split") != ENVIRONMENT_SPLIT:
                raise ValueError(
                    "checkpoint-selection reset manifest task/split mismatch"
                )
            if (
                row.get("episode_id") in review_episode_ids
                or _reset_identity(row) in review_identities
            ):
                raise ValueError(
                    "review and checkpoint-selection reset manifests overlap"
                )


def _gate_checks(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    gate = _require_mapping(record.get("quality_v2_gate"), "attempt quality_v2_gate")
    checks = gate.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("attempt quality_v2_gate checks must be non-empty")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(checks):
        check = _require_mapping(value, f"attempt gate check {index}")
        phase = _require_string(check.get("phase"), f"attempt gate check {index} phase")
        metric = _require_string(
            check.get("metric"), f"attempt gate check {index} metric"
        )
        actual = _require_number(
            check.get("actual"), f"attempt gate check {index} actual"
        )
        maximum = _require_number(check.get("max"), f"attempt gate check {index} max")
        passed = _require_bool(
            check.get("passed"), f"attempt gate check {index} passed"
        )
        identity = (phase, metric)
        if identity in seen:
            raise ValueError("attempt quality gate contains duplicate checks")
        if maximum < 0.0 or actual < 0.0 or passed != (actual <= maximum):
            raise ValueError("attempt quality gate check is internally inconsistent")
        seen.add(identity)
        normalized.append(
            {
                "phase": phase,
                "metric": metric,
                "actual": actual,
                "max": maximum,
                "passed": passed,
            }
        )
    gate_passed = _require_bool(gate.get("passed"), "attempt quality gate passed")
    if gate_passed != all(check["passed"] for check in normalized):
        raise ValueError("attempt aggregate quality gate is internally inconsistent")
    return normalized


def _action_application_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the v0.3 issued/applied declaration without trusting eligibility."""

    issued_equals_applied = _require_bool(
        record.get("issued_equals_applied"), "attempt issued_equals_applied"
    )
    raw_gate = record.get("t5_replan_causal_timing_passed")
    raw_latency = record.get("impact_end_to_first_qualifying_applied_correction_s")
    if record.get("task_id") != "t5_replan":
        if not issued_equals_applied:
            raise ValueError("non-T5 attempt must declare issued_equals_applied=true")
        if raw_gate is not None or raw_latency is not None:
            raise ValueError("non-T5 attempt cannot declare T5 causal timing evidence")
        return {
            "issued_equals_applied": True,
            "t5_replan_causal_timing_passed": None,
            "impact_end_to_first_qualifying_applied_correction_s": None,
        }
    if issued_equals_applied:
        raise ValueError("T5 attempt must preserve distinct issued/applied histories")
    gate = _require_bool(raw_gate, "T5 causal-timing gate")
    if gate:
        latency = _require_number(raw_latency, "T5 causal correction latency")
        if latency < 0.0:
            raise ValueError("T5 causal correction latency must be non-negative")
    else:
        if raw_latency is not None:
            raise ValueError(
                "failed T5 causal timing cannot declare a correction latency"
            )
        latency = None
    return {
        "issued_equals_applied": False,
        "t5_replan_causal_timing_passed": gate,
        "impact_end_to_first_qualifying_applied_correction_s": latency,
    }


def _audit_review_attempt_tape(
    root: Path,
    record: Mapping[str, Any],
    *,
    task: str,
    quality_v2_thresholds: Mapping[str, Any],
    quality_v2_thresholds_sha256: str,
) -> None:
    """Run the canonical tape/Qv3/T5 audit before a review attempt is admitted."""

    from examples.embodiment import (
        audit_dynamic_benchmark_optimal_trajectories as optimal_auditor,
    )

    if record.get("schema_version") != optimal_auditor.ATTEMPT_SCHEMA:
        raise ValueError("paired review requires canonical attempt schema v0.3")
    optimal_auditor._audit_attempt_tape(
        root,
        record,
        expected_task=task,
        quality_v2_thresholds=quality_v2_thresholds,
        quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
    )
    causal = _action_application_evidence(record)
    if task == "t5_replan" and causal["t5_replan_causal_timing_passed"] is not True:
        raise ValueError("paired T5 review requires a passing causal-timing gate")


def _absolute_gate_eligible(record: Mapping[str, Any]) -> bool:
    replay = _require_mapping(
        record.get("replay_validation"), "attempt replay_validation"
    )
    causal = _action_application_evidence(record)
    causal_passed = causal["t5_replan_causal_timing_passed"] is not False
    return bool(
        _require_bool(record.get("success"), "attempt success")
        and not _require_bool(record.get("safety_failure"), "attempt safety_failure")
        and _require_bool(
            record.get("finite_and_bounded"), "attempt finite_and_bounded"
        )
        and _require_bool(replay.get("passed"), "attempt replay passed")
        and _require_bool(
            _require_mapping(record.get("quality_v2_gate"), "attempt quality gate").get(
                "passed"
            ),
            "attempt quality gate passed",
        )
        and causal_passed
    )


def _reviewable(record: Mapping[str, Any]) -> bool:
    replay = _require_mapping(
        record.get("replay_validation"), "attempt replay_validation"
    )
    _action_application_evidence(record)
    return bool(
        _require_bool(record.get("finite_and_bounded"), "attempt finite_and_bounded")
        and _require_bool(replay.get("passed"), "attempt replay passed")
    )


def _threshold_check_inventory(
    threshold_checks: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    inventory: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(threshold_checks):
        check = dict(_require_mapping(raw, f"threshold check {index}"))
        phase = _require_string(check.get("phase"), f"threshold check {index} phase")
        metric = _require_string(check.get("metric"), f"threshold check {index} metric")
        required = {
            "phase",
            "metric",
            "max",
            "direction",
            "paired_comparison_family",
            *_PAIRED_THRESHOLD_FIELDS,
        }
        if not required.issubset(check):
            raise ValueError(
                "threshold check is missing frozen paired-comparison metadata"
            )
        maximum = _require_number(check["max"], "threshold maximum")
        if maximum < 0.0 or check.get("direction") != "minimize":
            raise ValueError("quality-v2 review thresholds must be non-negative minima")
        _require_string(
            check.get("paired_comparison_family"), "threshold comparison family"
        )
        paired = {
            field: _require_number(check[field], f"threshold {field}")
            for field in _PAIRED_THRESHOLD_FIELDS
        }
        if any(value < 0.0 for value in paired.values()):
            raise ValueError("paired comparison thresholds must be non-negative")
        if (
            paired["paired_strict_improvement_absolute"] == 0.0
            and paired["paired_strict_improvement_relative"] == 0.0
        ):
            raise ValueError("paired comparison has no strict-improvement resolution")
        identity = (phase, metric)
        if identity in inventory:
            raise ValueError("threshold contract contains duplicate checks")
        inventory[identity] = check
    if not inventory:
        raise ValueError("threshold contract has no checks")
    return inventory


def _comparison_tolerance(
    check: Mapping[str, Any], reference_value: float
) -> tuple[float, float]:
    tolerance = max(
        _require_number(
            check["paired_nonworse_absolute_tolerance"],
            "threshold paired_nonworse_absolute_tolerance",
        ),
        _require_number(
            check["paired_nonworse_relative_tolerance"],
            "threshold paired_nonworse_relative_tolerance",
        )
        * abs(reference_value),
    )
    strict_margin = max(
        _require_number(
            check["paired_strict_improvement_absolute"],
            "threshold paired_strict_improvement_absolute",
        ),
        _require_number(
            check["paired_strict_improvement_relative"],
            "threshold paired_strict_improvement_relative",
        )
        * abs(reference_value),
        2.0 * tolerance,
    )
    if tolerance < 0.0 or strict_margin <= 0.0:
        raise ValueError("threshold comparison tolerances must be non-negative")
    return tolerance, strict_margin


def _task_quality_components(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    summary = record.get("task_quality")
    if summary is None:
        return {}
    summary = _require_mapping(summary, "attempt task_quality")
    components = _require_mapping(
        summary.get("components"), "attempt task_quality components"
    )
    return {
        str(name): _require_mapping(value, f"task quality {name}")
        for name, value in components.items()
    }


def _planner_comparison(
    learned: Mapping[str, Any],
    planner: Mapping[str, Any],
    threshold_checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the same-reset nonworse + strict-improvement rule dynamically."""

    if learned.get("task_id") != planner.get("task_id"):
        raise ValueError("learned/planner comparison task identity mismatch")
    _action_application_evidence(learned)
    _action_application_evidence(planner)
    threshold_inventory = _threshold_check_inventory(threshold_checks)
    learned_checks = {
        (row["phase"], row["metric"]): row for row in _gate_checks(learned)
    }
    planner_checks = {
        (row["phase"], row["metric"]): row for row in _gate_checks(planner)
    }
    if set(learned_checks) != set(threshold_inventory) or set(planner_checks) != set(
        threshold_inventory
    ):
        raise ValueError(
            "attempt gate checks do not match the frozen threshold inventory"
        )
    learned_gate = _require_bool(
        _require_mapping(learned.get("quality_v2_gate"), "learned quality gate").get(
            "passed"
        ),
        "learned quality gate passed",
    )
    planner_gate = _require_bool(
        _require_mapping(planner.get("quality_v2_gate"), "planner quality gate").get(
            "passed"
        ),
        "planner quality gate passed",
    )
    absolute_gate_nonworse = learned_gate or not planner_gate
    dimensions: list[dict[str, Any]] = [
        {
            "name": "quality_v2.absolute_gate",
            "direction": "max",
            "planner": planner_gate,
            "learned": learned_gate,
            "nonworse": absolute_gate_nonworse,
            "strictly_better": learned_gate and not planner_gate,
        }
    ]
    strict: list[str] = []
    nonworse = absolute_gate_nonworse

    if learned.get("task_id") == "t5_replan":
        from examples.embodiment import (
            export_dynamic_benchmark_optimal_trajectories as optimal,
        )

        learned_latency = optimal._t5_causal_latency(learned)
        planner_latency = optimal._t5_causal_latency(planner)
        if learned_latency is None or planner_latency is None:
            raise ValueError("T5 planner comparison is missing causal latency")
        dimension_name = "causal.impact_end_to_first_qualifying_applied_correction_s"
        dimension_nonworse = (
            learned_latency <= planner_latency + T5_CAUSAL_LATENCY_TOLERANCE_S
        )
        dimension_strict = (
            learned_latency < planner_latency - T5_CAUSAL_LATENCY_TOLERANCE_S
        )
        nonworse &= dimension_nonworse
        if dimension_strict:
            strict.append(dimension_name)
        dimensions.append(
            {
                "name": dimension_name,
                "direction": "min",
                "planner": planner_latency,
                "learned": learned_latency,
                "tolerance": T5_CAUSAL_LATENCY_TOLERANCE_S,
                "strict_margin": T5_CAUSAL_LATENCY_TOLERANCE_S,
                "nonworse": dimension_nonworse,
                "strictly_better": dimension_strict,
            }
        )

    for identity, contract in threshold_inventory.items():
        learned_value = float(learned_checks[identity]["actual"])
        planner_value = float(planner_checks[identity]["actual"])
        tolerance, strict_margin = _comparison_tolerance(contract, planner_value)
        direction = "min"
        dimension_nonworse = learned_value <= planner_value + tolerance
        dimension_strict = learned_value < planner_value - strict_margin
        name = f"quality_v2.{identity[0]}.{identity[1]}"
        nonworse &= dimension_nonworse
        metric_groups = _metric_groups(identity[1])
        if dimension_strict and "smoothness" in metric_groups:
            strict.append(f"control.{name}")
        elif dimension_strict and "path" in metric_groups:
            strict.append(f"path.{name}")
        dimensions.append(
            {
                "name": name,
                "direction": direction,
                "planner": planner_value,
                "learned": learned_value,
                "tolerance": tolerance,
                "strict_margin": strict_margin,
                "nonworse": dimension_nonworse,
                "strictly_better": dimension_strict,
            }
        )

    scalar_specs = (
        ("trajectory_completion", "max", 1.0e-12, 1.0e-6, "utility"),
        ("completion_time_s", "min", 1.0e-12, 1.0e-6, "duration"),
        ("control_steps", "min", 0.0, 1.0, "control"),
        ("action_l2_sum", "min", 1.0e-6, 1.0e-6, "control"),
    )
    for name, direction, tolerance, strict_margin, improvement_group in scalar_specs:
        learned_raw = learned.get(name)
        planner_raw = planner.get(name)
        if learned_raw is None or planner_raw is None:
            nonworse = False
            dimensions.append({"name": name, "nonworse": False, "reason": "missing"})
            continue
        learned_value = _require_number(learned_raw, f"learned {name}")
        planner_value = _require_number(planner_raw, f"planner {name}")
        if name == "action_l2_sum":
            tolerance = max(tolerance, 1.0e-6 * abs(planner_value))
            strict_margin = max(strict_margin, 2.0 * tolerance)
        if direction == "min":
            dimension_nonworse = learned_value <= planner_value + tolerance
            dimension_strict = learned_value < planner_value - strict_margin
        else:
            dimension_nonworse = learned_value >= planner_value - tolerance
            dimension_strict = learned_value > planner_value + strict_margin
        nonworse &= dimension_nonworse
        if dimension_strict:
            strict.append(f"{improvement_group}.{name}")
        dimensions.append(
            {
                "name": name,
                "direction": direction,
                "planner": planner_value,
                "learned": learned_value,
                "tolerance": tolerance,
                "strict_margin": strict_margin,
                "nonworse": dimension_nonworse,
                "strictly_better": dimension_strict,
            }
        )

    learned_quality = _task_quality_components(learned)
    planner_quality = _task_quality_components(planner)
    if set(learned_quality) != set(planner_quality):
        nonworse = False
        dimensions.append(
            {
                "name": "task_quality",
                "nonworse": False,
                "reason": "component inventory mismatch",
            }
        )
    else:
        for name in learned_quality:
            learned_component = learned_quality[name]
            planner_component = planner_quality[name]
            if learned_component.get("direction") != planner_component.get("direction"):
                raise ValueError("task-quality direction mismatch within a reset")
            raw_direction = learned_component.get("direction")
            direction = {"minimize": "min", "maximize": "max"}.get(
                str(raw_direction), str(raw_direction)
            )
            if direction not in {"min", "max"}:
                raise ValueError(
                    "task-quality direction must be min/max or minimize/maximize"
                )
            learned_value = _require_number(
                learned_component.get("value"), f"learned task quality {name}"
            )
            planner_value = _require_number(
                planner_component.get("value"), f"planner task quality {name}"
            )
            resolution = max(
                _require_number(
                    learned_component.get("scientific_resolution", 0.0),
                    f"task quality {name} resolution",
                ),
                0.0,
            )
            if direction == "min":
                dimension_nonworse = learned_value <= planner_value + resolution
                dimension_strict = learned_value < planner_value - resolution
            else:
                dimension_nonworse = learned_value >= planner_value - resolution
                dimension_strict = learned_value > planner_value + resolution
            nonworse &= dimension_nonworse
            if dimension_strict:
                strict.append(f"utility.task_quality.{name}")
            dimensions.append(
                {
                    "name": f"task_quality.{name}",
                    "direction": direction,
                    "planner": planner_value,
                    "learned": learned_value,
                    "tolerance": resolution,
                    "strict_margin": resolution,
                    "nonworse": dimension_nonworse,
                    "strictly_better": dimension_strict,
                }
            )
    return {
        "planner_nonworse_all_dimensions": bool(nonworse),
        "strict_improvement_dimensions": sorted(set(strict)),
        "planner_exact_tie": bool(nonworse and not strict),
        "dimensions": dimensions,
    }


def _decorate_trajectory_decisions(
    attempts: Sequence[Mapping[str, Any]],
    threshold_checks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return copies with absolute-gate and same-reset eligibility separated."""

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        grouped[_require_int(attempt.get("reset_index"), "attempt reset_index")].append(
            attempt
        )
    decorated: list[dict[str, Any]] = []
    for reset_index in sorted(grouped):
        rows = grouped[reset_index]
        planners = [row for row in rows if row.get("candidate_kind") == "planner"]
        if len(planners) != 1:
            raise ValueError("each review reset requires exactly one planner attempt")
        planner = planners[0]
        for row in rows:
            item = dict(row)
            base = _absolute_gate_eligible(row)
            item["absolute_gate_eligible"] = base
            if row.get("candidate_kind") == "planner":
                decision = {
                    "source": "planner",
                    "planner_nonworse_all_dimensions": True,
                    "strict_improvement_dimensions": [],
                    "planner_exact_tie": True,
                    "dimensions": [],
                }
                eligible = base
            elif row.get("candidate_kind") == "policy":
                if row.get("promotion_validated") is not True:
                    raise ValueError(
                        "learned review attempt lacks a validated promotion receipt"
                    )
                decision = _planner_comparison(row, planner, threshold_checks)
                decision["source"] = "learned"
                if not base:
                    decision["reason"] = "absolute_or_replay_eligibility_failed"
                eligible = bool(
                    base
                    and decision["planner_nonworse_all_dimensions"]
                    and decision["strict_improvement_dimensions"]
                )
            else:
                raise ValueError(
                    "review attempt candidate_kind must be planner or policy"
                )
            item["planner_comparison"] = decision
            item["trajectory_eligible"] = eligible
            item["review_selected"] = False
            item["review_categories"] = []
            decorated.append(item)
    return decorated


def _metric_groups(metric: str) -> set[str]:
    value = metric.lower()
    groups: set[str] = set()
    if any(
        token in value
        for token in ("approach", "jaw", "orientation", "rotation", "tilt", "roll")
    ):
        groups.add("orientation")
    if any(
        token in value
        for token in ("path_length", "corridor", "backtrack", "detour", "drift")
    ):
        groups.add("path")
    if any(
        token in value
        for token in (
            "action",
            "jerk",
            "acceleration",
            "total_variation",
            "second_difference",
            "smooth",
        )
    ):
        groups.add("smoothness")
    return groups or {"other"}


def _check_ratio(check: Mapping[str, Any]) -> float:
    maximum = float(check["max"])
    actual = float(check["actual"])
    if maximum == 0.0:
        return 0.0 if actual == 0.0 else math.inf
    return actual / maximum


def _group_boundary(
    record: Mapping[str, Any], groups: set[str]
) -> tuple[float, dict[str, Any]] | None:
    values = [
        check
        for check in _gate_checks(record)
        if _metric_groups(check["metric"]) & groups
    ]
    if not values:
        return None
    selected = max(
        values, key=lambda check: (_check_ratio(check), check["phase"], check["metric"])
    )
    return _check_ratio(selected), selected


def _allowed_rejection(
    record: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]]] | None:
    gate = _require_mapping(record.get("quality_v2_gate"), "attempt quality gate")
    if gate.get("passed") is not False:
        return None
    failed = []
    for check in _gate_checks(record):
        metric = check["metric"].lower()
        allowed = any(
            token in metric
            for token in (
                "action",
                "jerk",
                "approach",
                "jaw",
                "orientation",
                "rotation",
                "tilt",
                "roll",
            )
        )
        if not check["passed"] and allowed:
            failed.append(check)
    if not failed:
        return None
    return max(_check_ratio(check) for check in failed), failed


def _disagreement_score(
    planner: Mapping[str, Any], learned: Mapping[str, Any]
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if planner["trajectory_eligible"] != learned["trajectory_eligible"]:
        reasons.append("trajectory_eligibility")
    if planner["absolute_gate_eligible"] != learned["absolute_gate_eligible"]:
        reasons.append("absolute_gate")
    for key in ("success", "safety_failure", "termination_reason"):
        if planner.get(key) != learned.get(key):
            reasons.append(key)
    planner_checks = {
        (row["phase"], row["metric"]): row for row in _gate_checks(planner)
    }
    learned_checks = {
        (row["phase"], row["metric"]): row for row in _gate_checks(learned)
    }
    metric_delta = 0.0
    for identity in planner_checks.keys() & learned_checks.keys():
        denominator = max(float(planner_checks[identity]["max"]), 1.0e-12)
        metric_delta = max(
            metric_delta,
            abs(
                float(planner_checks[identity]["actual"])
                - float(learned_checks[identity]["actual"])
            )
            / denominator,
        )
    completion_delta = abs(
        float(planner.get("trajectory_completion", 0.0))
        - float(learned.get("trajectory_completion", 0.0))
    )
    step_delta = abs(
        float(planner.get("control_steps", 0)) - float(learned.get("control_steps", 0))
    )
    if metric_delta > 1.0e-12:
        reasons.append("quality_metric")
    if completion_delta > 1.0e-12:
        reasons.append("trajectory_completion")
    if step_delta > 0.0:
        reasons.append("control_steps")
    score = 1000.0 * len(
        set(reasons)
        & {"trajectory_eligibility", "absolute_gate", "success", "safety_failure"}
    )
    score += (
        10.0 * len(reasons)
        + metric_delta
        + completion_delta
        + min(step_delta / 1000.0, 1.0)
    )
    return score, sorted(set(reasons))


def _validate_attempt_coverage(
    attempts: Sequence[Mapping[str, Any]],
    *,
    review_rows: Sequence[Mapping[str, Any]],
    candidate_count: int,
) -> None:
    if len(review_rows) != RESET_COUNT:
        raise ValueError("review coverage requires the fixed twenty-reset manifest")
    if len(attempts) != RESET_COUNT * candidate_count:
        raise ValueError("review did not evaluate the complete reset x candidate pool")
    by_reset: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        by_reset[
            _require_int(attempt.get("reset_index"), "attempt reset_index")
        ].append(attempt)
    if set(by_reset) != set(range(RESET_COUNT)):
        raise ValueError("review attempt reset coverage is not exact")
    expected_candidates = set(range(candidate_count))
    for reset_index, expected_row in enumerate(review_rows):
        rows = by_reset[reset_index]
        indices = [
            _require_int(row.get("candidate_index"), "attempt candidate_index")
            for row in rows
        ]
        if len(indices) != len(set(indices)) or set(indices) != expected_candidates:
            raise ValueError(
                "review attempt candidate coverage is incomplete or duplicated"
            )
        if any(row.get("episode_id") != expected_row.get("episode_id") for row in rows):
            raise ValueError("review attempt/reset episode identity mismatch")


def _select_review_pairs(
    attempts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Select the best same-reset pair independently for each frozen category."""

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in attempts:
        grouped[_require_int(row.get("reset_index"), "attempt reset_index")].append(row)
    eligible_learned = [
        row
        for row in attempts
        if row.get("candidate_kind") == "policy"
        and row.get("trajectory_eligible") is True
    ]
    representative_gate_median = (
        median(
            [
                max((_check_ratio(check) for check in _gate_checks(row)), default=0.0)
                for row in eligible_learned
            ]
        )
        if eligible_learned
        else 0.0
    )
    representative_step_median = (
        median([float(row.get("control_steps", 0)) for row in eligible_learned])
        if eligible_learned
        else 0.0
    )

    options: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for reset_index in sorted(grouped):
        rows = grouped[reset_index]
        planners = [row for row in rows if row.get("candidate_kind") == "planner"]
        learned_rows = [row for row in rows if row.get("candidate_kind") == "policy"]
        if len(planners) != 1:
            raise ValueError("paired review requires exactly one planner per reset")
        planner = planners[0]
        if not _reviewable(planner):
            continue
        planner_ratio = max(
            (_check_ratio(check) for check in _gate_checks(planner)), default=0.0
        )
        for learned in learned_rows:
            if not _reviewable(learned):
                continue
            stable = (
                reset_index,
                int(learned["candidate_index"]),
                str(learned["candidate_id"]),
            )
            base = {
                "reset_index": reset_index,
                "episode_id": planner["episode_id"],
                "planner": planner,
                "comparison": learned,
            }
            if learned.get("episode_id") != planner.get("episode_id"):
                raise ValueError("paired attempts do not share one reset episode")
            options["planner_anchor"].append(
                {
                    **base,
                    "rank": (
                        0 if planner.get("trajectory_eligible") else 1,
                        abs(planner_ratio - 0.5),
                        0 if learned.get("trajectory_eligible") else 1,
                        *stable,
                    ),
                    "rationale": {"planner_gate_max_ratio": planner_ratio},
                }
            )
            if learned.get("trajectory_eligible") is True:
                learned_ratio = max(
                    (_check_ratio(check) for check in _gate_checks(learned)),
                    default=0.0,
                )
                options["learned_representative"].append(
                    {
                        **base,
                        "rank": (
                            abs(learned_ratio - representative_gate_median)
                            + abs(
                                float(learned.get("control_steps", 0))
                                - representative_step_median
                            )
                            / max(representative_step_median, 1.0),
                            *stable,
                        ),
                        "rationale": {
                            "gate_max_ratio": learned_ratio,
                            "cohort_gate_median": representative_gate_median,
                            "cohort_control_steps_median": representative_step_median,
                        },
                    }
                )
                smooth = _group_boundary(learned, {"smoothness", "path"})
                if smooth is not None:
                    ratio, check = smooth
                    options["smoothness_boundary"].append(
                        {
                            **base,
                            "rank": (-ratio, *stable),
                            "rationale": {
                                "boundary_ratio": ratio,
                                "boundary_check": check,
                            },
                        }
                    )
                orientation = _group_boundary(learned, {"orientation"})
                if orientation is not None:
                    ratio, check = orientation
                    options["orientation_boundary"].append(
                        {
                            **base,
                            "rank": (-ratio, *stable),
                            "rationale": {
                                "boundary_ratio": ratio,
                                "boundary_check": check,
                            },
                        }
                    )
            rejection = _allowed_rejection(learned)
            if rejection is not None:
                violation, failed = rejection
                options["rejected_jitter_or_tilt"].append(
                    {
                        **base,
                        "rank": (-violation, *stable),
                        "rationale": {
                            "violation_ratio": violation,
                            "failed_checks": failed,
                        },
                    }
                )
            disagreement, reasons = _disagreement_score(planner, learned)
            if disagreement > 0.0:
                options["planner_rl_disagreement"].append(
                    {
                        **base,
                        "rank": (-disagreement, *stable),
                        "rationale": {
                            "disagreement_score": disagreement,
                            "reasons": reasons,
                        },
                    }
                )

    selected = {
        category: min(options[category], key=lambda value: value["rank"])
        for category in CATEGORIES
        if options[category]
    }
    missing = tuple(category for category in CATEGORIES if category not in selected)
    cards = []
    for category in CATEGORIES:
        if category not in selected:
            continue
        option = selected[category]
        planner = option["planner"]
        learned = option["comparison"]
        if category == "rejected_jitter_or_tilt":
            gate = _require_mapping(
                learned.get("quality_v2_gate"), "rejected comparison gate"
            )
            if gate.get("passed") is not False or _allowed_rejection(learned) is None:
                raise AssertionError(
                    "rejected review card did not retain an allowed gate failure"
                )
        cards.append(
            {
                "category": category,
                "review_selected": True,
                "reset_index": int(option["reset_index"]),
                "episode_id": option["episode_id"],
                "planner_candidate_index": int(planner["candidate_index"]),
                "comparison_candidate_index": int(learned["candidate_index"]),
                "planner_candidate_id": planner["candidate_id"],
                "comparison_candidate_id": learned["candidate_id"],
                "planner_trajectory_eligible": bool(planner["trajectory_eligible"]),
                "comparison_trajectory_eligible": bool(learned["trajectory_eligible"]),
                "rationale": option["rationale"],
            }
        )
    return cards, missing


def _mark_review_selected(
    attempts: Sequence[Mapping[str, Any]], cards: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    selected: dict[tuple[int, int], set[str]] = defaultdict(set)
    for card in cards:
        for side in ("planner", "comparison"):
            identity = (
                int(card["reset_index"]),
                int(card[f"{side}_candidate_index"]),
            )
            selected[identity].add(str(card["category"]))
    return [
        {
            **dict(row),
            "review_selected": (int(row["reset_index"]), int(row["candidate_index"]))
            in selected,
            "review_categories": sorted(
                selected.get(
                    (int(row["reset_index"]), int(row["candidate_index"])), set()
                )
            ),
        }
        for row in attempts
    ]


def _validate_review_contract(
    payload: Mapping[str, Any],
    *,
    threshold_schema: str,
    threshold_sha256: str,
    evaluator_commit: str,
    evaluator_benchmark_commit: str,
) -> Mapping[str, Any]:
    contract = _require_mapping(
        payload.get("review_contract"),
        "candidate review contract",
        _REVIEW_CONTRACT_KEYS,
    )
    expected = {
        "schema_version": REVIEW_CANDIDATE_CONTRACT_SCHEMA,
        "partition": PARTITION,
        "environment_split": ENVIRONMENT_SPLIT,
        "manifest_seed": MANIFEST_SEED,
        "reset_count": RESET_COUNT,
        "inference_mode": INFERENCE_MODE,
        "candidate_search_mode": SEARCH_MODE,
        "full_generation": False,
        "quality_v2_threshold_schema": threshold_schema,
        "quality_v2_thresholds_sha256": threshold_sha256,
        "evaluator_commit": evaluator_commit,
        "evaluator_benchmark_commit": evaluator_benchmark_commit,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"candidate review contract mismatch for {key}")
    _require_sha256(
        contract.get("evaluator_source_sha256"), "review evaluator source SHA-256"
    )
    return contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--quality-v2-thresholds", type=Path, required=True)
    parser.add_argument("--expected-quality-v2-thresholds-sha256", required=True)
    parser.add_argument(
        "--quality-v2-calibration-wave-receipt",
        type=Path,
        required=True,
        help="authoritative receipt source copied into review provenance",
    )
    parser.add_argument(
        "--expected-quality-v2-calibration-wave-receipt-sha256",
        required=True,
    )
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--evaluator-benchmark-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=(PARTITION,), default=PARTITION)
    parser.add_argument("--manifest-seed", type=int, default=MANIFEST_SEED)
    parser.add_argument("--review-resets", type=int, default=RESET_COUNT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _load_thresholds(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str, str]:
    expected = _require_sha256(
        expected_sha256, "expected quality-v2 thresholds SHA-256"
    )
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError("quality-v2 threshold contract SHA-256 mismatch")
    payload = _read_json(path, "quality-v2 threshold contract")
    schema = _require_string(
        payload.get("schema_version"), "quality-v2 threshold schema"
    )
    if schema != QUALITY_V2_THRESHOLD_SCHEMA:
        raise ValueError(
            "paired production review requires quality-v2 threshold schema v0.3"
        )
    if payload.get("formal_freeze_eligible") is not True:
        raise ValueError(
            "quality-v2 threshold contract is provisional and not formal-freeze eligible"
        )
    tasks = _require_mapping(payload.get("tasks"), "quality-v2 threshold tasks")
    if not tasks:
        raise ValueError("quality-v2 threshold task inventory is empty")
    return payload, schema, expected


def _validate_quality_v2_calibration_wave_receipt(
    thresholds: Mapping[str, Any],
    receipt_path: Path,
    expected_sha256: str,
    *,
    expected_benchmark_commit: str,
) -> tuple[Any, dict[str, str]]:
    """Delegate the frozen exact-14 receipt contract to the optimal exporter."""

    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    provenance = optimal._validate_quality_v2_calibration_receipt_artifact(
        thresholds,
        receipt_path,
        expected_sha256=expected_sha256,
        expected_benchmark_commit=expected_benchmark_commit,
    )
    binding = optimal._quality_v2_calibration_receipt_binding(thresholds)
    if (
        provenance.relative_path != binding["relative_path"]
        or provenance.sha256 != binding["file_sha256"]
        or provenance.sha256 != binding["payload_sha256"]
    ):
        raise RuntimeError(
            "validated quality-v2 calibration receipt identity changed unexpectedly"
        )
    return provenance, dict(binding)


def _copy_quality_v2_calibration_wave_receipt(output: Path, provenance: Any) -> None:
    """Materialize the validated receipt through the canonical safe copier."""

    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    optimal._copy_provenance_file(
        output,
        provenance.source_path,
        provenance.relative_path,
        provenance.sha256,
    )


def _quality_v2_selection_manifest_provenance(
    *,
    threshold_schema: str,
    threshold_sha256: str,
    paired_dimension_count: int,
    calibration_receipt_identity: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Bind the frozen thresholds and their copied receipt sidecar together."""

    return {
        "quality_v2_thresholds": {
            "path": "quality_v2_thresholds.json",
            "schema_version": threshold_schema,
            "sha256": threshold_sha256,
            "paired_dimension_count": paired_dimension_count,
        },
        "quality_v2_calibration_wave_receipt_identity": dict(
            calibration_receipt_identity
        ),
    }


def _candidate_promotions(
    *,
    payload: Mapping[str, Any],
    manifest_path: Path,
    specs: Sequence[Any],
    task: str,
    rlinf_commit: str,
    benchmark_commit: str,
    evaluator_commit: str,
    evaluator_benchmark_commit: str,
    threshold_schema: str,
    threshold_sha256: str,
    calibration_receipt_identity: Mapping[str, str],
    evaluator_source_sha256: str,
) -> tuple[dict[int, dict[str, Any]], list[list[dict[str, Any]]]]:
    promotions: dict[int, dict[str, Any]] = {}
    selection_manifests: list[list[dict[str, Any]]] = []
    seen_selection_hashes: set[str] = set()
    for index, spec in enumerate(specs):
        if spec.kind == "planner":
            if index != 0 or spec.stochastic or spec.exploration_seed_offset:
                raise ValueError(
                    "review candidate index zero must be the deterministic planner"
                )
            continue
        if spec.stochastic or spec.exploration_seed_offset != 0:
            raise ValueError(
                "review candidate pool forbids stochastic/action-noise expansion"
            )
        provenance = _require_mapping(
            spec.provenance, f"candidate {spec.candidate_id} provenance"
        )
        receipt_identity = _require_mapping(
            provenance.get("promotion_receipt"),
            f"candidate {spec.candidate_id} promotion receipt identity",
        )
        if not {"path", "sha256"}.issubset(receipt_identity):
            raise ValueError(
                "candidate promotion receipt identity field inventory mismatch"
            )
        receipt_path = _resolve_reference(manifest_path, str(receipt_identity["path"]))
        expected_receipt_sha = _require_sha256(
            receipt_identity.get("sha256"), "candidate promotion receipt SHA-256"
        )
        if not receipt_path.is_file() or _sha256(receipt_path) != expected_receipt_sha:
            raise ValueError(
                "candidate promotion receipt is missing or has a SHA-256 mismatch"
            )
        receipt = _read_json(receipt_path, "policy promotion receipt")
        assert spec.policy_path is not None and spec.policy_sha256 is not None
        normalized = _validate_promotion_receipt_payload(
            receipt,
            task=task,
            candidate_id=spec.candidate_id,
            policy_path=str(spec.policy_path),
            policy_sha256=spec.policy_sha256,
            residual_scale=spec.residual_scale,
            rlinf_commit=rlinf_commit,
            benchmark_commit=benchmark_commit,
            threshold_schema=threshold_schema,
            threshold_sha256=threshold_sha256,
            calibration_receipt_identity=calibration_receipt_identity,
            evaluator_rlinf_commit=evaluator_commit,
            evaluator_source_sha256=evaluator_source_sha256,
        )
        evidence_path, evidence = _validate_selection_evidence_file(
            normalized,
            promotion_path=receipt_path,
            task=task,
            candidate_id=spec.candidate_id,
            policy_path=str(spec.policy_path),
            policy_sha256=spec.policy_sha256,
            residual_scale=spec.residual_scale,
            rlinf_commit=rlinf_commit,
            benchmark_commit=benchmark_commit,
            threshold_sha256=threshold_sha256,
            calibration_receipt_identity=calibration_receipt_identity,
            evaluator_source_sha256=evaluator_source_sha256,
        )
        selection_reset_path, rows = _validate_validation_receipt_file(
            normalized,
            promotion_path=receipt_path,
            policy_sha256=spec.policy_sha256,
            policy_env_steps=int(normalized["policy"]["env_steps"]),
            task=task,
            rlinf_commit=rlinf_commit,
            evaluator_commit=evaluator_commit,
            evaluator_benchmark_commit=evaluator_benchmark_commit,
            threshold_schema=threshold_schema,
            threshold_sha256=threshold_sha256,
        )
        reset_sha = str(normalized["validation_receipt"]["reset_manifest_sha256"])
        if reset_sha not in seen_selection_hashes:
            selection_manifests.append(rows)
            seen_selection_hashes.add(reset_sha)
        promotions[index] = {
            "candidate_id": spec.candidate_id,
            "candidate_index": index,
            "policy_path": str(spec.policy_path),
            "policy_sha256": spec.policy_sha256,
            "residual_scale": spec.residual_scale,
            "receipt_path": str(receipt_path),
            "receipt_sha256": expected_receipt_sha,
            "receipt_payload_sha256": normalized["payload_sha256"],
            "selection_evidence_path": str(evidence_path),
            "selection_evidence_sha256": normalized["selection"]["evidence_sha256"],
            "selection_evidence_payload_sha256": evidence["payload_sha256"],
            "quality_v2_calibration_wave_receipt": normalized[
                "quality_v2_calibration_wave_receipt"
            ],
            "env_steps": normalized["policy"]["env_steps"],
            "checkpoint_role": normalized["policy"]["checkpoint_role"],
            "validation_receipt_path": str(
                _resolve_reference(
                    receipt_path, normalized["validation_receipt"]["path"]
                )
            ),
            "validation_receipt_sha256": normalized["validation_receipt"]["sha256"],
            "selection_reset_manifest_path": str(selection_reset_path),
            "selection_reset_manifest_sha256": reset_sha,
            "selector_contract_sha256": normalized["selection"][
                "selector_contract_sha256"
            ],
            "source_identity": normalized["source_identity"],
            "image_identity": normalized["image_identity"],
        }
    if len(promotions) != len(specs) - 1 or not promotions:
        raise ValueError(
            "every and at least one learned candidate must have a valid promotion receipt"
        )
    return promotions, selection_manifests


def _media_from_trace(root: Path, trace: Any) -> list[dict[str, Any]]:
    """Write browser-ready per-camera GIFs without introducing a module import dependency."""

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for paired review media") from error
    observations = tuple(trace.observations)
    if not observations or not observations[0].rgb:
        raise ValueError("rendered review trace has no RGB cameras")
    cameras = tuple(observations[0].rgb)
    if len(cameras) != 2:
        raise ValueError("paired owner review requires exactly two RGB cameras")
    if any(tuple(observation.rgb) != cameras for observation in observations[1:]):
        raise ValueError("rendered review camera inventory drifted within a trace")
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for camera in cameras:
        if _SAFE_COMPONENT.fullmatch(camera) is None:
            raise ValueError(f"unsafe review camera name: {camera!r}")
        frames = []
        for observation in observations:
            frame = observation.rgb[camera]
            image = Image.fromarray(frame)
            if image.mode != "RGB":
                image = image.convert("RGB")
            frames.append(image)
        path = root / f"{camera}.gif"
        temporary = path.with_suffix(".gif.tmp")
        frames[0].save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
            optimize=False,
        )
        os.replace(temporary, path)
        rows.append(
            {
                "camera": camera,
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "frame_count": len(frames),
                "frame_period_s": 0.05,
            }
        )
    return rows


def _render_selected_pairs(
    *,
    output: Path,
    cards: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    candidates: Sequence[Any],
    candidate_env_keys: Sequence[str],
    policy_configs: Mapping[str, Mapping[str, Any] | None],
    make_render_env: Any,
    threshold_contract: Mapping[str, Any],
    threshold_schema: str,
    threshold_sha256: str,
    candidate_manifest_sha256: str,
    review_reset_manifest_sha256: str,
    source_identity: Mapping[str, Any],
    device: Any,
    rollout: Any,
    restore_candidate_start: Any,
    write_episode_atomic: Any,
) -> list[dict[str, Any]]:
    by_identity = {
        (int(row["reset_index"]), int(row["candidate_index"])): row for row in attempts
    }
    cards_by_reset: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for card in cards:
        cards_by_reset[int(card["reset_index"])].append(card)
    render_envs = {
        key: make_render_env(policy) for key, policy in policy_configs.items()
    }
    rendered_cards: dict[str, dict[str, Any]] = {}
    try:
        for reset_index in range(RESET_COUNT):
            for env in render_envs.values():
                request = env._requests[0]
                if request is None:
                    raise RuntimeError(
                        "render environment lost its review reset request"
                    )
            initial_states = {
                key: env.checkpoint_state() for key, env in render_envs.items()
            }
            for card in cards_by_reset.get(reset_index, ()):
                side_rows: dict[str, dict[str, Any]] = {}
                for side, index_key in (
                    ("planner", "planner_candidate_index"),
                    ("comparison", "comparison_candidate_index"),
                ):
                    candidate_index = int(card[index_key])
                    candidate = candidates[candidate_index]
                    env_key = candidate_env_keys[candidate_index]
                    env = render_envs[env_key]
                    restore_candidate_start(env, initial_states[env_key])
                    light = by_identity[(reset_index, candidate_index)]
                    tape_path = output / str(light["attempt_tape"])
                    import numpy as np

                    with np.load(tape_path, allow_pickle=False) as tape:
                        replay_actions = np.asarray(tape["actions"], dtype=np.float64)
                    render_record, _, trace = rollout(
                        env=env,
                        candidate=candidate,
                        device=device,
                        capture_trace=True,
                        replay_actions_array=replay_actions,
                        quality_v2_thresholds=threshold_contract,
                        quality_v2_thresholds_sha256=threshold_sha256,
                        trace_metadata={
                            "review_partition": PARTITION,
                            "review_category": card["category"],
                            "review_selected": True,
                            "trajectory_eligible": light["trajectory_eligible"],
                            "candidate_manifest_sha256": candidate_manifest_sha256,
                            "review_reset_manifest_sha256": review_reset_manifest_sha256,
                            "quality_v2_threshold_schema": threshold_schema,
                            "quality_v2_thresholds_sha256": threshold_sha256,
                            "policy_sha256": candidate.spec.policy_sha256,
                            "source_identity": dict(source_identity),
                            "inference_mode": INFERENCE_MODE,
                            "full_generation": False,
                        },
                    )
                    if trace is None:
                        raise RuntimeError(
                            "paired review render did not return an EpisodeTrace"
                        )
                    for key in (
                        "episode_id",
                        "candidate_id",
                        "candidate_index",
                        "success",
                        "safety_failure",
                        "termination_reason",
                        "trajectory_completion",
                        "completion_time_s",
                        "control_steps",
                        "action_l2_sum",
                        "action_sha256",
                        "quality_v2_sha256",
                        "quality_v2_gate",
                        "issued_equals_applied",
                        "t5_replan_causal_timing_passed",
                        "impact_end_to_first_qualifying_applied_correction_s",
                    ):
                        if render_record.get(key) != light.get(key):
                            raise RuntimeError(
                                f"paired review render parity failed for {key}"
                            )
                    side_root = output / "paired" / str(card["category"]) / side
                    episode_record = write_episode_atomic(side_root, trace)
                    episode_dir = side_root / episode_record["relative_episode_dir"]
                    media = _media_from_trace(
                        output / "paired" / str(card["category"]) / "media" / side,
                        trace,
                    )
                    for item in media:
                        item["path"] = Path(item["path"]).relative_to(output).as_posix()
                    side_rows[side] = {
                        "candidate_id": light["candidate_id"],
                        "candidate_index": candidate_index,
                        "candidate_kind": light["candidate_kind"],
                        "policy_sha256": candidate.spec.policy_sha256,
                        "residual_scale": candidate.spec.residual_scale,
                        "promotion": light["promotion"],
                        "review_selected": True,
                        "absolute_gate_eligible": light["absolute_gate_eligible"],
                        "trajectory_eligible": light["trajectory_eligible"],
                        "quality_v2_gate": light["quality_v2_gate"],
                        "planner_comparison": light["planner_comparison"],
                        "raw_metrics": {
                            "success": light["success"],
                            "safety_failure": light["safety_failure"],
                            "termination_reason": light["termination_reason"],
                            "trajectory_completion": light["trajectory_completion"],
                            "completion_time_s": light["completion_time_s"],
                            "control_steps": light["control_steps"],
                            "action_l2_sum": light["action_l2_sum"],
                            "task_quality": light["task_quality"],
                            "quality_v2": light["quality_v2"],
                            "quality_v2_sha256": light["quality_v2_sha256"],
                            "issued_equals_applied": light["issued_equals_applied"],
                            "t5_replan_causal_timing_passed": light[
                                "t5_replan_causal_timing_passed"
                            ],
                            "impact_end_to_first_qualifying_applied_correction_s": light[
                                "impact_end_to_first_qualifying_applied_correction_s"
                            ],
                        },
                        "attempt_tape": light["attempt_tape"],
                        "attempt_tape_sha256": light["attempt_tape_sha256"],
                        "episode_record_path": (episode_dir / "episode.json")
                        .relative_to(output)
                        .as_posix(),
                        "episode_record_sha256": _sha256(episode_dir / "episode.json"),
                        "trajectory_path": (episode_dir / "trajectory.h5")
                        .relative_to(output)
                        .as_posix(),
                        "trajectory_sha256": episode_record["trajectory_sha256"],
                        "media": media,
                    }
                trace_payload = {
                    "schema_version": PAIR_TRACE_SCHEMA,
                    "category": card["category"],
                    "partition": PARTITION,
                    "environment_split": ENVIRONMENT_SPLIT,
                    "reset_index": reset_index,
                    "episode_id": card["episode_id"],
                    "candidate_manifest_sha256": candidate_manifest_sha256,
                    "review_reset_manifest_sha256": review_reset_manifest_sha256,
                    "quality_v2_threshold_identity": {
                        "schema_version": threshold_schema,
                        "sha256": threshold_sha256,
                    },
                    "source_identity": dict(source_identity),
                    "full_generation": False,
                    "rationale": card["rationale"],
                    "sides": side_rows,
                }
                trace_payload["payload_sha256"] = _payload_sha256(trace_payload)
                trace_path = (
                    output / "paired" / str(card["category"]) / "pair_trace.json"
                )
                _atomic_json(trace_path, trace_payload)
                rendered_cards[str(card["category"])] = {
                    **dict(card),
                    "pair_trace_path": trace_path.relative_to(output).as_posix(),
                    "pair_trace_sha256": _sha256(trace_path),
                    "sides": side_rows,
                }
            if reset_index + 1 < RESET_COUNT:
                for env in render_envs.values():
                    env.reset(options={"env_idx": [0]})
    finally:
        for env in render_envs.values():
            env.close()
    return [rendered_cards[category] for category in CATEGORIES]


def main() -> None:
    started_unix_s = time.time()
    args = _parser().parse_args()
    if (
        args.partition != PARTITION
        or args.manifest_seed != MANIFEST_SEED
        or args.review_resets != RESET_COUNT
    ):
        raise ValueError(
            "RLD2-QA review partition/seed/reset count are frozen and cannot be overridden"
        )
    if args.image_size < 64:
        raise ValueError("image_size must be at least 64")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    candidate_manifest_sha256 = _require_sha256(
        args.expected_candidate_manifest_sha256, "expected candidate manifest SHA-256"
    )
    if (
        not args.candidate_manifest.is_file()
        or _sha256(args.candidate_manifest) != candidate_manifest_sha256
    ):
        raise ValueError("candidate manifest SHA-256 mismatch")
    threshold_contract, threshold_schema, threshold_sha256 = _load_thresholds(
        args.quality_v2_thresholds, args.expected_quality_v2_thresholds_sha256
    )
    evaluator_commit = _require_commit(args.evaluator_commit, "evaluator commit")
    evaluator_benchmark_commit = _require_commit(
        args.evaluator_benchmark_commit, "evaluator benchmark commit"
    )
    candidate_payload = _read_json(args.candidate_manifest, "review candidate manifest")
    if (
        candidate_payload.get("schema_version")
        != "rlinf-dynamic-benchmark-optimal-candidates-v0.1"
    ):
        raise ValueError(
            "paired review currently requires the portable deterministic v0.1 pool"
        )
    rlinf_commit = _require_commit(
        candidate_payload.get("rlinf_commit"), "policy RLinf commit"
    )
    benchmark_commit = _require_commit(
        candidate_payload.get("benchmark_commit"), "policy benchmark commit"
    )
    contract = _validate_review_contract(
        candidate_payload,
        threshold_schema=threshold_schema,
        threshold_sha256=threshold_sha256,
        evaluator_commit=evaluator_commit,
        evaluator_benchmark_commit=evaluator_benchmark_commit,
    )
    (
        quality_v2_calibration_receipt,
        quality_v2_calibration_receipt_identity,
    ) = _validate_quality_v2_calibration_wave_receipt(
        threshold_contract,
        args.quality_v2_calibration_wave_receipt,
        args.expected_quality_v2_calibration_wave_receipt_sha256,
        expected_benchmark_commit=evaluator_benchmark_commit,
    )
    task = _require_string(candidate_payload.get("task"), "candidate task")
    threshold_tasks = _require_mapping(
        threshold_contract.get("tasks"), "quality threshold tasks"
    )
    task_thresholds = _require_mapping(
        threshold_tasks.get(task), f"quality thresholds for {task}"
    )
    threshold_checks = task_thresholds.get("checks")
    if not isinstance(threshold_checks, list):
        raise ValueError("task quality threshold checks must be an array")
    _threshold_check_inventory(threshold_checks)

    from se3_wam.benchmark.contracts import canonical_json
    from se3_wam.benchmark.dataset import write_episode_atomic
    from se3_wam.benchmark.evaluation import manifest_record

    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    quality_v2_dominance = optimal._quality_v2_dominance_contract(
        threshold_contract,
        task=task,
        thresholds_sha256=threshold_sha256,
        require_formal_freeze=True,
    )
    if len(quality_v2_dominance["metrics"]) != len(threshold_checks):
        raise ValueError(
            "quality-v2 paired dimension inventory changed during validation"
        )

    validated_task, specs = optimal._validate_candidate_manifest(
        candidate_payload,
        manifest_path=args.candidate_manifest.resolve(),
        rlinf_commit=rlinf_commit,
        benchmark_commit=benchmark_commit,
        max_k=1,
    )
    if validated_task != task:
        raise ValueError("candidate task changed during validation")
    promotions, selection_manifests = _candidate_promotions(
        payload=candidate_payload,
        manifest_path=args.candidate_manifest.resolve(),
        specs=specs,
        task=task,
        rlinf_commit=rlinf_commit,
        benchmark_commit=benchmark_commit,
        evaluator_commit=evaluator_commit,
        evaluator_benchmark_commit=evaluator_benchmark_commit,
        threshold_schema=threshold_schema,
        threshold_sha256=threshold_sha256,
        calibration_receipt_identity=quality_v2_calibration_receipt_identity,
        evaluator_source_sha256=str(contract["evaluator_source_sha256"]),
    )
    device = optimal._device(args.device)
    candidates = optimal._load_candidates(
        specs,
        task=task,
        rlinf_commit=rlinf_commit,
        benchmark_commit=benchmark_commit,
        device=device,
    )

    def make_env(policy: Mapping[str, Any] | None, *, rendered: bool) -> Any:
        return optimal._make_env(
            task=task,
            split=ENVIRONMENT_SPLIT,
            manifest_seed=MANIFEST_SEED,
            manifest_size=RESET_COUNT,
            image_size=args.image_size if rendered else 64,
            camera_observations=rendered,
            policy=policy,
        )

    light_default = make_env(None, rendered=False)
    default_key = canonical_json(light_default.state_schema)
    policy_configs: dict[str, Mapping[str, Any] | None] = {default_key: None}
    for candidate in candidates:
        if candidate.state_schema is None:
            continue
        assert candidate.config is not None
        policy_configs.setdefault(
            canonical_json(candidate.state_schema), candidate.config
        )
    light_envs = {default_key: light_default}
    for key, policy in policy_configs.items():
        if key == default_key:
            continue
        env = make_env(policy, rendered=False)
        if canonical_json(env.state_schema) != key:
            env.close()
            raise ValueError(
                "review environment state schema does not match its promoted policy"
            )
        light_envs[key] = env
    candidate_env_keys = [
        default_key
        if candidate.state_schema is None
        else canonical_json(candidate.state_schema)
        for candidate in candidates
    ]

    try:
        environment_review_rows = [
            manifest_record(row) for row in light_default._manifest_rows[:RESET_COUNT]
        ]
        for env in light_envs.values():
            if [
                manifest_record(row) for row in env._manifest_rows[:RESET_COUNT]
            ] != environment_review_rows:
                raise RuntimeError(
                    "state-schema environments disagree on the frozen review manifest"
                )
        review_rows = [
            {
                **row,
                "partition": PARTITION,
                "manifest_seed": MANIFEST_SEED,
                "reset_index": index,
            }
            for index, row in enumerate(environment_review_rows)
        ]
        _assert_disjoint_reset_rows(review_rows, selection_manifests, task=task)
        args.output.mkdir(parents=True)
        _copy_quality_v2_calibration_wave_receipt(
            args.output, quality_v2_calibration_receipt
        )
        shutil.copyfile(
            args.candidate_manifest, args.output / "candidate_manifest.json"
        )
        shutil.copyfile(
            args.quality_v2_thresholds, args.output / "quality_v2_thresholds.json"
        )
        review_reset_manifest = args.output / "review_reset_manifest.jsonl"
        _write_jsonl(review_reset_manifest, review_rows)
        review_reset_manifest_sha256 = _sha256(review_reset_manifest)
        attempts: list[dict[str, Any]] = []
        for reset_index, review_row in enumerate(review_rows):
            for env in light_envs.values():
                request = env._requests[0]
                if request is None or request.episode_id != review_row["episode_id"]:
                    raise RuntimeError(
                        "rollout order diverged from the frozen review reset manifest"
                    )
            initial_states = {
                key: env.checkpoint_state() for key, env in light_envs.items()
            }
            for candidate in candidates:
                env_key = candidate_env_keys[candidate.index]
                env = light_envs[env_key]
                optimal._restore_candidate_start(env, initial_states[env_key])
                record, arrays, _ = optimal._rollout(
                    env=env,
                    candidate=candidate,
                    device=device,
                    capture_trace=False,
                    quality_v2_thresholds=threshold_contract,
                    quality_v2_thresholds_sha256=threshold_sha256,
                )
                relative, tape_sha256 = optimal._write_attempt_tape(
                    args.output,
                    episode_id=record["episode_id"],
                    candidate_index=candidate.index,
                    arrays=arrays,
                )
                promotion = promotions.get(candidate.index)
                attempt = {
                    **record,
                    "review_attempt_schema_version": ATTEMPT_INDEX_SCHEMA,
                    "partition": PARTITION,
                    "environment_split": ENVIRONMENT_SPLIT,
                    "manifest_seed": MANIFEST_SEED,
                    "reset_index": reset_index,
                    "candidate_manifest_sha256": candidate_manifest_sha256,
                    "review_reset_manifest_sha256": review_reset_manifest_sha256,
                    "quality_v2_thresholds_sha256": threshold_sha256,
                    "inference_mode": INFERENCE_MODE,
                    "promotion_validated": candidate.spec.kind == "planner"
                    or promotion is not None,
                    "promotion": promotion,
                    "attempt_tape": relative,
                    "attempt_tape_sha256": tape_sha256,
                }
                attempt["quality_score"] = list(optimal._quality_score(attempt))
                attempt["eligible"] = optimal._eligible(attempt)
                _audit_review_attempt_tape(
                    args.output,
                    attempt,
                    task=task,
                    quality_v2_thresholds=threshold_contract,
                    quality_v2_thresholds_sha256=threshold_sha256,
                )
                attempts.append(attempt)
            if reset_index + 1 < RESET_COUNT:
                for env in light_envs.values():
                    env.reset(options={"env_idx": [0]})
        _validate_attempt_coverage(
            attempts, review_rows=review_rows, candidate_count=len(candidates)
        )
        attempts = _decorate_trajectory_decisions(attempts, threshold_checks)
        cards, missing = _select_review_pairs(attempts)
        attempts = _mark_review_selected(attempts, cards)
        _write_jsonl(args.output / "attempts.jsonl", attempts)
    finally:
        for env in light_envs.values():
            env.close()

    status = (
        "complete" if not missing and len(cards) == len(CATEGORIES) else "incomplete"
    )
    selection_manifest: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA,
        "status": status,
        "partition": PARTITION,
        "environment_split": ENVIRONMENT_SPLIT,
        "manifest_seed": MANIFEST_SEED,
        "review_reset_count": RESET_COUNT,
        "candidate_search_mode": SEARCH_MODE,
        "early_stopping": False,
        "replacement_resets": False,
        "seed_fishing": False,
        "full_generation": False,
        "full_generation_reason": "owner paired review gate has not approved production generation",
        "candidate_manifest": {
            "path": "candidate_manifest.json",
            "sha256": candidate_manifest_sha256,
            "candidate_count": len(candidates),
        },
        "review_reset_manifest": {
            "path": "review_reset_manifest.jsonl",
            "sha256": review_reset_manifest_sha256,
        },
        **_quality_v2_selection_manifest_provenance(
            threshold_schema=threshold_schema,
            threshold_sha256=threshold_sha256,
            paired_dimension_count=len(threshold_checks),
            calibration_receipt_identity=(quality_v2_calibration_receipt_identity),
        ),
        "source_identity": {
            "evaluator_commit": evaluator_commit,
            "evaluator_benchmark_commit": evaluator_benchmark_commit,
            "evaluator_source_sha256": contract["evaluator_source_sha256"],
            "policy_rlinf_commit": rlinf_commit,
            "policy_benchmark_commit": benchmark_commit,
        },
        "promotion_receipts": [promotions[index] for index in sorted(promotions)],
        "selection_contract": {
            "categories": list(CATEGORIES),
            "cards_per_task": len(CATEGORIES),
            "category_selection_independent": True,
            "reset_reuse_across_categories_allowed": True,
            "planner_and_all_promoted_candidates_per_reset": True,
            "review_selected_is_not_trajectory_eligible": True,
            "learned_trajectory_rule": (
                "absolute gates and replay pass; same-reset planner-nonworse on every frozen "
                "dimension including T5 applied-action causal latency; at least one strict "
                "utility/path/duration/control/causal improvement; planner wins exact ties"
            ),
            "rejected_category_allowed_groups": [
                "action",
                "jerk",
                "approach",
                "jaw",
                "orientation",
                "rotation",
            ],
        },
        "handoff_blockers": [dict(row) for row in RELEASE_HANDOFF_BLOCKERS],
        "attempt_count": len(attempts),
        "all_lightweight_tapes_saved": True,
        "missing_categories": list(missing),
        "cards": cards,
        "started_unix_s": started_unix_s,
    }
    if status == "complete":

        def make_render_env(policy: Mapping[str, Any] | None) -> Any:
            return make_env(policy, rendered=True)

        rendered_cards = _render_selected_pairs(
            output=args.output,
            cards=cards,
            attempts=attempts,
            candidates=candidates,
            candidate_env_keys=candidate_env_keys,
            policy_configs=policy_configs,
            make_render_env=make_render_env,
            threshold_contract=threshold_contract,
            threshold_schema=threshold_schema,
            threshold_sha256=threshold_sha256,
            candidate_manifest_sha256=candidate_manifest_sha256,
            review_reset_manifest_sha256=review_reset_manifest_sha256,
            source_identity=selection_manifest["source_identity"],
            device=device,
            rollout=optimal._rollout,
            restore_candidate_start=optimal._restore_candidate_start,
            write_episode_atomic=write_episode_atomic,
        )
        selection_manifest["cards"] = rendered_cards
    selection_manifest["finished_unix_s"] = time.time()
    selection_manifest["payload_sha256"] = _payload_sha256(selection_manifest)
    _atomic_json(args.output / "selection_manifest.json", selection_manifest)
    optimal._root_checksums(args.output)
    print(
        json.dumps(
            {
                "status": status,
                "task": task,
                "review_resets": RESET_COUNT,
                "candidate_count": len(candidates),
                "attempts": len(attempts),
                "cards": len(cards),
                "missing_categories": list(missing),
                "full_generation": False,
                "selection_manifest_sha256": _sha256(
                    args.output / "selection_manifest.json"
                ),
            },
            sort_keys=True,
        )
    )
    if status != "complete":
        raise RuntimeError(
            "paired review is incomplete; missing frozen categories: "
            + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
