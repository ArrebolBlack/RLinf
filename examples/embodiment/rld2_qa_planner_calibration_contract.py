#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Freeze, run, and verify the exact RLD2-QA planner-calibration wave.

The evaluator currently accepts the benchmark ``validation`` split.  This
wrapper treats that value only as a deterministic manifest-generator transport;
the scientific partition is the separately named ``metric_calibration``
partition, with its own seed and immutable reset receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

WAVE_CONTRACT_SCHEMA = "rld2-qa-planner-calibration-wave-contract-v0.1"
TASK_CONTRACT_SCHEMA = "rld2-qa-planner-calibration-task-contract-v0.1"
PREDECLARATION_RECEIPT_SCHEMA = (
    "rld2-qa-planner-calibration-predeclaration-receipt-v0.1"
)
TASK_RECEIPT_SCHEMA = "rld2-qa-planner-calibration-task-receipt-v0.1"
WAVE_RECEIPT_SCHEMA = "rld2-qa-planner-calibration-wave-receipt-v0.1"
ATTEMPT_CONTRACT_SCHEMA = "rld2-qa-planner-calibration-attempt-v0.1"
ATTEMPT_STATUS_SCHEMA = "rld2-qa-planner-calibration-attempt-status-v0.1"
PLANNER_CALIBRATION_POLICY_SCHEMA = "rld2-qa-planner-dominance-calibration-policy-v0.1"
PLANNER_CALIBRATION_BINDING_SCHEMA = (
    "rld2-qa-planner-dominance-calibration-binding-v0.1"
)
CALIBRATION_EVIDENCE_SCHEMA = (
    "rlinf-dynamic-benchmark-planner-calibration-evidence-v0.1"
)
PLANNER_DOMINANCE_SCHEMA = "rlinf-dynamic-benchmark-planner-dominance-v0.1"

SCIENTIFIC_PARTITION = "metric_calibration"
TRANSPORT_SPLIT = "validation"
EPISODES_PER_TASK = 20
PLANNER_CALIBRATION_REPLAY_COUNT = 3
DEFAULT_MANIFEST_SEED = 20261350
DEFAULT_VALIDATION_SEED = 20261150
DEFAULT_REVIEW_SEED = 20261250
DEFAULT_TEST_ID_SEED = 20262040
DEFAULT_TEST_OOD_SEED = 20262041
EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-planner-evaluation-v0.1"
TASK_QUALITY_BACKEND_ID = "mujoco311-rs140-v1-rld2-quality"
QUALITY_V2_SCHEMA = "se3-wam-trajectory-quality-v2"

TASK_ORDER = (
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

SOURCE_FILES = {
    "launcher": "examples/embodiment/rld2_qa_planner_calibration.sh",
    "contract_helper": ("examples/embodiment/rld2_qa_planner_calibration_contract.py"),
    "planner_evaluator": ("examples/embodiment/evaluate_dynamic_benchmark_planner.py"),
    "dynamic_benchmark_adapter": (
        "rlinf/envs/dynamic_benchmark/dynamic_benchmark_env.py"
    ),
    "planner_calibration_evidence_builder": (
        "examples/embodiment/build_dynamic_benchmark_rld2_evidence.py"
    ),
    "planner_calibration_replay_runner": (
        "examples/embodiment/run_dynamic_benchmark_rld2_launch_gate.py"
    ),
    "planner_calibration_template_builder": (
        "examples/embodiment/prepare_dynamic_benchmark_rld2_launch_gate.py"
    ),
}

PLANNER_CALIBRATION_ARTIFACTS = (
    (
        "selected_reset_manifest.jsonl",
        "selected_reset_manifest_relative_path",
        "selected_reset_manifest_sha256",
    ),
    (
        "planner_actions.npy",
        "planner_actions_relative_path",
        "planner_actions_file_sha256",
    ),
    (
        "calibration_input.json",
        "calibration_input_relative_path",
        "calibration_input_sha256",
    ),
    (
        "calibration_evidence.json",
        "calibration_evidence_relative_path",
        "calibration_evidence_sha256",
    ),
    (
        "planner_dominance_contract.json",
        "planner_dominance_contract_relative_path",
        "planner_dominance_contract_sha256",
    ),
)

RUNTIME_MODULES = (
    "se3_wam",
    "se3_wam.benchmark",
    "se3_wam.benchmark.api",
    "se3_wam.benchmark.config",
    "se3_wam.benchmark.contracts",
    "se3_wam.benchmark.dataset_manifest",
    "se3_wam.benchmark.evaluation",
    "se3_wam.benchmark.metrics",
    "se3_wam.benchmark.p0_grasp_manifest",
    "se3_wam.benchmark.registry",
    "se3_wam.benchmark.task_quality",
    "se3_wam.benchmark.teacher_factory",
    "se3_wam.benchmark.trajectory_quality",
)

BENCHMARK_SOURCE_FILES = {
    module_name: f"src/{module_name.replace('.', '/')}/__init__.py"
    if module_name in {"se3_wam", "se3_wam.benchmark"}
    else f"src/{module_name.replace('.', '/')}.py"
    for module_name in RUNTIME_MODULES
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_BENCHMARK_CONTAINER_ROOT_RE = re.compile(
    r"^/workspace/runtime/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$"
)


class ContractError(RuntimeError):
    """Raised when any frozen calibration identity fails closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a full lowercase Git commit")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON value {token!r} is forbidden")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError(f"cannot read {label} {path}: {error}") from error
    value = _require_mapping(value, label)
    _canonical_json(value)
    return value


def _write_exact_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ContractError(f"refusing to change immutable artifact {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_exact_json(path: Path, value: Any) -> None:
    # Deliberately no trailing newline: the file digest is the canonical payload
    # digest used as the single receipt identity by downstream calibrators.
    _write_exact_bytes(path, _canonical_bytes(value))


def _write_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_exact_sha256sums(path: Path, rows: Sequence[tuple[str, str]]) -> None:
    for digest, relative_path in rows:
        _require_sha256(digest, f"digest for {relative_path}")
        if not relative_path or "\n" in relative_path or "\r" in relative_path:
            raise ContractError("invalid SHA256SUMS relative path")
    text = "".join(f"{digest}  {relative_path}\n" for digest, relative_path in rows)
    _write_exact_bytes(path, text.encode("utf-8"))


def _safe_id(value: str, label: str) -> str:
    if _SAFE_ID_RE.fullmatch(value) is None:
        raise ContractError(
            f"{label} must be 1-64 safe characters: letters, digits, dot, dash, underscore"
        )
    return value


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(token) for token in value.split(","))
    except ValueError as error:
        raise ContractError("GPU CSV must contain decimal lane indices") from error
    if not parsed or len(parsed) > 8:
        raise ContractError("GPU lane pool must contain between 1 and 8 lanes")
    if len(set(parsed)) != len(parsed) or any(gpu < 0 or gpu > 7 for gpu in parsed):
        raise ContractError("GPU lane pool must contain unique N0 indices in [0, 7]")
    return parsed


def _git_output(source_root: Path, arguments: Sequence[str], label: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContractError(
            f"cannot resolve {label} for {source_root}: {error}"
        ) from error
    return result.stdout.strip()


def _git_head(source_root: Path) -> str:
    return _require_commit(
        _git_output(source_root, ("rev-parse", "HEAD"), "Git HEAD"),
        "source Git HEAD",
    )


def _require_clean_worktree(source_root: Path, label: str) -> None:
    status = _git_output(
        source_root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        f"{label} worktree status",
    )
    if status:
        raise ContractError(f"{label} worktree is not clean at its declared Git HEAD")


def _benchmark_container_root(value: Any) -> str:
    if (
        not isinstance(value, str)
        or _BENCHMARK_CONTAINER_ROOT_RE.fullmatch(value) is None
        or ".." in value.split("/")
    ):
        raise ContractError(
            "benchmark container root must be a normalized child of /workspace/runtime"
        )
    return value


def _git_file_identities(
    source_root: Path,
    source_files: Mapping[str, str],
    label: str,
) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for name, relative_path in source_files.items():
        path = source_root / relative_path
        if not path.is_file():
            raise ContractError(f"required {label} source file is absent: {path}")
        status = _git_output(
            source_root,
            ("status", "--porcelain=v1", "--untracked-files=all", "--", relative_path),
            f"{label} source status",
        )
        if status:
            raise ContractError(
                f"{label} source file is not clean and tracked at HEAD: {relative_path}"
            )
        blob_id = _git_output(
            source_root,
            ("rev-parse", f"HEAD:{relative_path}"),
            f"{label} Git blob identity",
        )
        if _GIT_OBJECT_RE.fullmatch(blob_id) is None:
            raise ContractError(f"{label} source file has an invalid Git blob identity")
        identities[name] = {
            "relative_path": relative_path,
            "sha256": _sha256(path),
            "git_blob_id": blob_id,
        }
    return identities


def _reserved_partitions(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    seeds = {
        "validation": int(args.validation_seed),
        "review": int(args.review_seed),
        "test_id": int(args.test_id_seed),
        "test_ood": int(args.test_ood_seed),
    }
    if any(seed < 0 for seed in seeds.values()):
        raise ContractError("manifest seeds must be nonnegative")
    if len({int(args.manifest_seed), *seeds.values()}) != 5:
        raise ContractError(
            "metric_calibration, validation, review, test-ID, and test-OOD seeds must be distinct"
        )
    return {
        "metric_calibration": {
            "scientific_partition": SCIENTIFIC_PARTITION,
            "transport_split": TRANSPORT_SPLIT,
            "manifest_seed": int(args.manifest_seed),
            "materialization": "predeclared_exact20_per_task",
        },
        "validation": {
            "scientific_partition": "validation",
            "transport_split": "validation",
            "manifest_seed": seeds["validation"],
            "materialization": "identity_comparator_only",
        },
        "review": {
            "scientific_partition": "review",
            "transport_split": "validation",
            "manifest_seed": seeds["review"],
            "materialization": "identity_comparator_only",
        },
        "test_id": {
            "scientific_partition": "test_id",
            "transport_split": "test_id",
            "manifest_seed": seeds["test_id"],
            "materialization": "sealed_not_read",
        },
        "test_ood": {
            "scientific_partition": "test_ood",
            "transport_split": "test_ood",
            "manifest_seed": seeds["test_ood"],
            "materialization": "sealed_not_read",
        },
    }


def _planner_calibration_policy() -> dict[str, Any]:
    return {
        "schema_version": PLANNER_CALIBRATION_POLICY_SCHEMA,
        "selection_policy": "first_safe_success_in_predeclared_reset_order",
        "action_tape_source": "planner_evaluation_recorded_actions",
        "recorded_action_dtype": "float64",
        "fresh_environment_replay_count_per_task": PLANNER_CALIBRATION_REPLAY_COUNT,
        "total_fresh_environment_replay_count": (
            len(TASK_ORDER) * PLANNER_CALIBRATION_REPLAY_COUNT
        ),
        "image_size": 64,
        "same_selected_reset_and_action_required": True,
        "unique_environment_instance_ids_required": True,
        "calibration_evidence_schema_version": CALIBRATION_EVIDENCE_SCHEMA,
        "planner_dominance_schema_version": PLANNER_DOMINANCE_SCHEMA,
        "test_exposure": {"test_id": False, "test_ood": False},
    }


def _wave_contract(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    benchmark_source_root = args.benchmark_source_root.resolve()
    if source_root == benchmark_source_root:
        raise ContractError(
            "evaluator and benchmark source roots must be distinct Git worktrees"
        )
    expected_commit = _require_commit(args.evaluator_commit, "evaluator commit")
    _require_clean_worktree(source_root, "evaluator RLinf")
    _require_clean_worktree(benchmark_source_root, "benchmark SE3-WAM")
    actual_commit = _git_head(source_root)
    if actual_commit != expected_commit:
        raise ContractError(
            f"source Git HEAD {actual_commit} does not equal evaluator commit {expected_commit}"
        )
    benchmark_commit = _require_commit(args.benchmark_commit, "benchmark commit")
    actual_benchmark_commit = _git_head(benchmark_source_root)
    if actual_benchmark_commit != benchmark_commit:
        raise ContractError(
            "benchmark source Git HEAD "
            f"{actual_benchmark_commit} does not equal benchmark commit {benchmark_commit}"
        )
    benchmark_container_root = _benchmark_container_root(
        args.benchmark_source_container_root
    )
    if _IMAGE_ID_RE.fullmatch(args.image_id) is None:
        raise ContractError(
            "image ID must be an immutable sha256:<64 lowercase hex> identity"
        )
    wave_id = _safe_id(args.wave_id, "wave ID")
    lane_prefix = _safe_id(args.lane_prefix, "lane prefix")
    gpus = _parse_gpus(args.gpus)
    partitions = _reserved_partitions(args)
    assignments = [
        {
            "ordinal": ordinal,
            "task_id": task,
            "lane_prefix": lane_prefix,
            "gpu_index": gpus[ordinal % len(gpus)],
        }
        for ordinal, task in enumerate(TASK_ORDER)
    ]
    evaluator_files = _git_file_identities(
        source_root,
        SOURCE_FILES,
        "evaluator RLinf",
    )
    benchmark_files = _git_file_identities(
        benchmark_source_root,
        BENCHMARK_SOURCE_FILES,
        "benchmark SE3-WAM",
    )
    source_identity = {
        "evaluator_rlinf_commit": expected_commit,
        "benchmark_commit": benchmark_commit,
        "runtime_image": {"reference": args.image_ref, "id": args.image_id},
        "files": evaluator_files,
        "mounted_sources": {
            "evaluator_rlinf": {
                "git_commit": expected_commit,
                "container_root": "/workspace/SE3-WAM",
                "worktree_status": "clean_at_declared_git_head",
                "files": evaluator_files,
            },
            "benchmark_se3_wam": {
                "git_commit": benchmark_commit,
                "container_root": benchmark_container_root,
                "worktree_status": "clean_at_declared_git_head",
                "files": benchmark_files,
            },
        },
        "planner_evaluation_schema": EVALUATION_SCHEMA,
        "task_quality_evaluator_backend_id": TASK_QUALITY_BACKEND_ID,
        "quality_v2_schema_version": QUALITY_V2_SCHEMA,
    }
    return {
        "schema_version": WAVE_CONTRACT_SCHEMA,
        "wave_id": wave_id,
        "scientific_partition": SCIENTIFIC_PARTITION,
        "transport_split": TRANSPORT_SPLIT,
        "transport_compatibility_note": (
            "The evaluator's validation enum is used only to generate deterministic "
            "candidate rows; this wave is scientifically metric_calibration and is "
            "identified by its distinct seed, contracts, and receipts."
        ),
        "manifest_seed": int(args.manifest_seed),
        "task_count": len(TASK_ORDER),
        "episodes_per_task": EPISODES_PER_TASK,
        "total_reset_count": len(TASK_ORDER) * EPISODES_PER_TASK,
        "total_fresh_environment_replay_count": (
            len(TASK_ORDER) * PLANNER_CALIBRATION_REPLAY_COUNT
        ),
        "task_order": list(TASK_ORDER),
        "partitions": partitions,
        "planner_dominance_calibration": _planner_calibration_policy(),
        "sealed_test_policy": {
            "test_manifests_generated": False,
            "test_manifests_read": False,
            "test_rows_must_not_appear_in_wave_artifacts": True,
        },
        "lane_pool": {
            "resource": "RES-A800X16-TEMP/N0",
            "lane_prefix": lane_prefix,
            "gpu_indices": list(gpus),
            "scheduler_contract": "one-task-per-lane-sequential-round-robin-v0.1",
            "assignments": assignments,
        },
        "source_identity": source_identity,
        "output_layout": {
            "task_root_template": "tasks/{task_id}",
            "attempt_root_template": "tasks/{task_id}/attempts/attempt-{number:06d}",
        },
    }


def _validate_wave_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    wave = dict(value)
    if wave.get("schema_version") != WAVE_CONTRACT_SCHEMA:
        raise ContractError("unsupported wave contract schema")
    if wave.get("scientific_partition") != SCIENTIFIC_PARTITION:
        raise ContractError("wave is not the metric_calibration partition")
    if wave.get("transport_split") != TRANSPORT_SPLIT:
        raise ContractError("wave transport split must be validation")
    if tuple(wave.get("task_order", ())) != TASK_ORDER:
        raise ContractError("wave task order is not the canonical exact-14 order")
    if (
        wave.get("task_count") != len(TASK_ORDER)
        or wave.get("episodes_per_task") != EPISODES_PER_TASK
        or wave.get("total_reset_count") != len(TASK_ORDER) * EPISODES_PER_TASK
        or wave.get("total_fresh_environment_replay_count")
        != len(TASK_ORDER) * PLANNER_CALIBRATION_REPLAY_COUNT
    ):
        raise ContractError("wave exact-14/exact-20/exact-three cardinality drifted")
    if wave.get("planner_dominance_calibration") != _planner_calibration_policy():
        raise ContractError("planner-dominance calibration policy drifted")
    partitions = _require_mapping(wave.get("partitions"), "wave partitions")
    required_partition_names = {
        "metric_calibration",
        "validation",
        "review",
        "test_id",
        "test_ood",
    }
    if set(partitions) != required_partition_names:
        raise ContractError("wave partition declarations are incomplete")
    calibration = _require_mapping(
        partitions["metric_calibration"], "calibration partition"
    )
    if (
        calibration.get("scientific_partition") != SCIENTIFIC_PARTITION
        or calibration.get("transport_split") != TRANSPORT_SPLIT
        or calibration.get("manifest_seed") != wave.get("manifest_seed")
    ):
        raise ContractError("calibration partition identity drifted")
    seeds = []
    for name in sorted(required_partition_names):
        partition = _require_mapping(partitions[name], f"partition {name}")
        seed = partition.get("manifest_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ContractError(f"partition {name} has an invalid manifest seed")
        expected_transport = (
            TRANSPORT_SPLIT
            if name in {"metric_calibration", "validation", "review"}
            else name
        )
        expected_materialization = (
            "predeclared_exact20_per_task"
            if name == "metric_calibration"
            else (
                "sealed_not_read"
                if name in {"test_id", "test_ood"}
                else "identity_comparator_only"
            )
        )
        if (
            partition.get("scientific_partition") != name
            or partition.get("transport_split") != expected_transport
            or partition.get("materialization") != expected_materialization
        ):
            raise ContractError(f"partition {name} declaration drifted")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ContractError("calibration/validation/review/test seeds are not disjoint")
    for test_name in ("test_id", "test_ood"):
        if partitions[test_name].get("materialization") != "sealed_not_read":
            raise ContractError("test partitions must remain sealed and unread")
    sealed = _require_mapping(wave.get("sealed_test_policy"), "sealed test policy")
    if sealed != {
        "test_manifests_generated": False,
        "test_manifests_read": False,
        "test_rows_must_not_appear_in_wave_artifacts": True,
    }:
        raise ContractError("sealed-test policy drifted")
    source = _require_mapping(wave.get("source_identity"), "source identity")
    evaluator_commit = _require_commit(
        source.get("evaluator_rlinf_commit"), "evaluator commit"
    )
    benchmark_commit = _require_commit(
        source.get("benchmark_commit"), "benchmark commit"
    )
    image = _require_mapping(source.get("runtime_image"), "runtime image identity")
    if (
        not isinstance(image.get("reference"), str)
        or _IMAGE_ID_RE.fullmatch(str(image.get("id"))) is None
    ):
        raise ContractError("runtime image identity is incomplete")
    if (
        source.get("planner_evaluation_schema") != EVALUATION_SCHEMA
        or source.get("task_quality_evaluator_backend_id") != TASK_QUALITY_BACKEND_ID
        or source.get("quality_v2_schema_version") != QUALITY_V2_SCHEMA
    ):
        raise ContractError("evaluator/quality source identity drifted")
    files = _require_mapping(source.get("files"), "source file identities")
    if set(files) != set(SOURCE_FILES):
        raise ContractError("source file identity set drifted")
    for name, relative_path in SOURCE_FILES.items():
        item = _require_mapping(files[name], f"source file {name}")
        if item.get("relative_path") != relative_path:
            raise ContractError(f"source file path drift for {name}")
        _require_sha256(item.get("sha256"), f"source file {name}")
        if _GIT_OBJECT_RE.fullmatch(str(item.get("git_blob_id"))) is None:
            raise ContractError(f"source file Git blob identity drift for {name}")
    mounted = _require_mapping(
        source.get("mounted_sources"), "mounted source identities"
    )
    if set(mounted) != {"evaluator_rlinf", "benchmark_se3_wam"}:
        raise ContractError("mounted source role set drifted")
    evaluator_source = _require_mapping(
        mounted["evaluator_rlinf"], "mounted evaluator source"
    )
    if (
        evaluator_source.get("git_commit") != evaluator_commit
        or evaluator_source.get("container_root") != "/workspace/SE3-WAM"
        or evaluator_source.get("worktree_status") != "clean_at_declared_git_head"
        or evaluator_source.get("files") != files
    ):
        raise ContractError("mounted evaluator source identity drifted")
    benchmark_source = _require_mapping(
        mounted["benchmark_se3_wam"], "mounted benchmark source"
    )
    if (
        benchmark_source.get("git_commit") != benchmark_commit
        or benchmark_source.get("worktree_status") != "clean_at_declared_git_head"
    ):
        raise ContractError("mounted benchmark Git identity drifted")
    _benchmark_container_root(benchmark_source.get("container_root"))
    benchmark_files = _require_mapping(
        benchmark_source.get("files"), "benchmark source files"
    )
    if set(benchmark_files) != set(BENCHMARK_SOURCE_FILES):
        raise ContractError("benchmark source file identity set drifted")
    for name, relative_path in BENCHMARK_SOURCE_FILES.items():
        item = _require_mapping(benchmark_files[name], f"benchmark source file {name}")
        if item.get("relative_path") != relative_path:
            raise ContractError(f"benchmark source file path drift for {name}")
        _require_sha256(item.get("sha256"), f"benchmark source file {name}")
        if _GIT_OBJECT_RE.fullmatch(str(item.get("git_blob_id"))) is None:
            raise ContractError(f"benchmark source Git blob identity drift for {name}")
    lane_pool = _require_mapping(wave.get("lane_pool"), "lane pool")
    gpus = lane_pool.get("gpu_indices")
    if not isinstance(gpus, list):
        raise ContractError("lane pool GPU indices must be a list")
    parsed_gpus = _parse_gpus(",".join(str(value) for value in gpus))
    assignments = lane_pool.get("assignments")
    expected_assignments = [
        {
            "ordinal": ordinal,
            "task_id": task,
            "lane_prefix": lane_pool.get("lane_prefix"),
            "gpu_index": parsed_gpus[ordinal % len(parsed_gpus)],
        }
        for ordinal, task in enumerate(TASK_ORDER)
    ]
    if assignments != expected_assignments:
        raise ContractError("lane assignment map is not deterministic round-robin")
    return wave


def _load_wave(wave_root: Path) -> tuple[dict[str, Any], str]:
    path = wave_root / "wave_contract.json"
    wave = _validate_wave_contract(_load_json(path, "wave contract"))
    digest = _sha256(path)
    if digest != _payload_sha256(wave):
        raise ContractError("wave contract is not canonical JSON")
    sidecar = wave_root / "WAVE_CONTRACT.sha256"
    expected_sidecar = f"{digest}  wave_contract.json\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise ContractError("wave contract SHA256 sidecar is absent or invalid")
    return wave, digest


def _import_runtime() -> dict[str, Any]:
    try:
        from se3_wam.benchmark.api import Split
        from se3_wam.benchmark.config import task_config_sha256
        from se3_wam.benchmark.contracts import canonical_json
        from se3_wam.benchmark.dataset_manifest import make_dataset_candidate_manifest
        from se3_wam.benchmark.p0_grasp_manifest import make_p0_grasp_candidate_manifest
        from se3_wam.benchmark.registry import ACTIVE_TASK_IDS, RL_EXPERT_TASK_IDS
        from se3_wam.benchmark.task_quality import task_quality_schema_manifest
    except ImportError as error:
        raise ContractError(
            "cannot import the frozen SE3-WAM benchmark runtime; run inside the bound image"
        ) from error
    if set(RL_EXPERT_TASK_IDS) != set(TASK_ORDER) or len(RL_EXPERT_TASK_IDS) != len(
        TASK_ORDER
    ):
        raise ContractError(
            "runtime RL expert task set is not the canonical exact-14 set"
        )
    normalized_active_tasks = tuple(
        "p0_grasp" if task == "p0" else task for task in ACTIVE_TASK_IDS
    )
    if set(normalized_active_tasks) != set(TASK_ORDER) or len(
        normalized_active_tasks
    ) != len(TASK_ORDER):
        raise ContractError("runtime active task set is not the canonical exact-14 set")
    return {
        "Split": Split,
        "ACTIVE_TASK_IDS": tuple(ACTIVE_TASK_IDS),
        "canonical_json": canonical_json,
        "make_dataset_candidate_manifest": make_dataset_candidate_manifest,
        "make_p0_grasp_candidate_manifest": make_p0_grasp_candidate_manifest,
        "manifest_record": _manifest_record,
        "task_config_sha256": task_config_sha256,
        "task_quality_schema_manifest": task_quality_schema_manifest,
    }


def _runtime_module_identities(
    benchmark_source_root: Path,
    expected_identities: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    import importlib.util

    result: dict[str, dict[str, str]] = {}
    benchmark_source_root = benchmark_source_root.resolve()
    for module_name in RUNTIME_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            raw_path = getattr(module, "__file__", None)
        else:
            spec = importlib.util.find_spec(module_name)
            raw_path = None if spec is None else spec.origin
        if not isinstance(raw_path, str):
            raise ContractError(f"runtime module has no source file: {module_name}")
        path = Path(raw_path)
        if path.suffix == ".pyc" and path.with_suffix(".py").is_file():
            path = path.with_suffix(".py")
        if not path.is_file():
            raise ContractError(
                f"runtime module source is absent: {module_name}: {path}"
            )
        relative_path = BENCHMARK_SOURCE_FILES[module_name]
        expected_path = (benchmark_source_root / relative_path).resolve()
        if path.resolve() != expected_path:
            raise ContractError(
                f"runtime module {module_name} was imported from {path.resolve()}, "
                f"not the mounted benchmark source {expected_path}"
            )
        expected = _require_mapping(
            expected_identities.get(module_name),
            f"expected runtime module {module_name}",
        )
        identity = {
            "relative_path": relative_path,
            "sha256": _sha256(path),
            "git_blob_id": expected.get("git_blob_id"),
        }
        if identity != expected:
            raise ContractError(
                f"runtime module source identity drifted: {module_name}"
            )
        result[module_name] = identity
    return result


def _manifest_record(row: Any) -> dict[str, Any]:
    """Mirror the bound benchmark evaluator's dependency-light row encoding."""

    request = row.request
    return {
        "episode_id": request.episode_id,
        "task_id": request.task_id,
        "split": request.split.value,
        "seed": request.seed,
        "action_mode": request.action_mode.value,
        "observation_track": request.observation_track.value,
        "object_mode": request.object_mode,
        "reset_mode": request.reset_mode,
        "factors": dict(request.factors),
        "source_group_id": row.source_group_id,
        "pair_id": row.pair_id,
        "pair_member_id": row.pair_member_id,
        "candidate_index": row.candidate_index,
    }


def _manifest_payloads(
    runtime: Mapping[str, Any], task: str, seed: int
) -> list[dict[str, Any]]:
    split = runtime["Split"].VALIDATION
    if task == "p0_grasp":
        rows = runtime["make_p0_grasp_candidate_manifest"](
            split=split,
            attempts=EPISODES_PER_TASK,
            manifest_seed=seed,
        )
    else:
        all_rows = runtime["make_dataset_candidate_manifest"](
            split=split,
            attempts_per_task=EPISODES_PER_TASK,
            manifest_seed=seed,
            tasks=runtime["ACTIVE_TASK_IDS"],
        )
        rows = tuple(row for row in all_rows if row.request.task_id == task)
    payloads = [runtime["manifest_record"](row) for row in rows]
    if len(payloads) != EPISODES_PER_TASK:
        raise ContractError(f"{task} manifest is not exact-{EPISODES_PER_TASK}")
    for payload in payloads:
        if payload.get("task_id") != task or payload.get("split") != TRANSPORT_SPLIT:
            raise ContractError(f"{task} runtime manifest identity drifted")
        runtime["canonical_json"](payload)
    episode_ids = [payload.get("episode_id") for payload in payloads]
    if any(not isinstance(value, str) or not value for value in episode_ids):
        raise ContractError(f"{task} manifest has an invalid episode identity")
    if len(set(episode_ids)) != EPISODES_PER_TASK:
        raise ContractError(f"{task} reset episode identities are not unique")
    return payloads


def _manifest_bytes(payloads: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(_canonical_json(dict(payload)) + "\n" for payload in payloads)
    ).encode("utf-8")


def _reset_identity(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities = []
    for ordinal, payload in enumerate(payloads):
        identities.append(
            {
                "ordinal": ordinal,
                "episode_id": payload["episode_id"],
                "reset_row_sha256": _payload_sha256(dict(payload)),
            }
        )
    episode_ids = sorted({str(item["episode_id"]) for item in identities})
    row_hashes = sorted({str(item["reset_row_sha256"]) for item in identities})
    if len(episode_ids) != EPISODES_PER_TASK or len(row_hashes) != EPISODES_PER_TASK:
        raise ContractError("reset identity set is not exact-20 and unique")
    return {
        "reset_identities": identities,
        "reset_identity_set_sha256": _payload_sha256(episode_ids),
        "reset_row_set_sha256": _payload_sha256(row_hashes),
    }


def _task_contract(
    wave: Mapping[str, Any],
    wave_sha256: str,
    task: str,
    benchmark_source_root: Path,
) -> tuple[dict[str, Any], bytes]:
    if task not in TASK_ORDER:
        raise ContractError(f"unknown calibration task {task!r}")
    runtime = _import_runtime()
    seed = int(wave["manifest_seed"])
    payloads = _manifest_payloads(runtime, task, seed)
    manifest_bytes = _manifest_bytes(payloads)
    identity = _reset_identity(payloads)
    partitions = wave["partitions"]

    comparator_rows: dict[str, list[dict[str, Any]]] = {}
    for name in ("validation", "review"):
        comparator_rows[name] = _manifest_payloads(
            runtime,
            task,
            int(partitions[name]["manifest_seed"]),
        )
    calibration_hashes = {
        item["reset_row_sha256"] for item in identity["reset_identities"]
    }
    calibration_episode_ids = {
        item["episode_id"] for item in identity["reset_identities"]
    }
    comparisons: dict[str, Any] = {}
    for name, rows in comparator_rows.items():
        comparator_identity = _reset_identity(rows)
        comparator_hashes = {
            item["reset_row_sha256"] for item in comparator_identity["reset_identities"]
        }
        comparator_episode_ids = {
            item["episode_id"] for item in comparator_identity["reset_identities"]
        }
        row_overlap = sorted(calibration_hashes.intersection(comparator_hashes))
        identity_overlap = sorted(
            calibration_episode_ids.intersection(comparator_episode_ids)
        )
        if row_overlap or identity_overlap:
            raise ContractError(f"{task} calibration resets overlap {name}")
        comparisons[name] = {
            "scientific_partition": name,
            "transport_split": partitions[name]["transport_split"],
            "manifest_seed": partitions[name]["manifest_seed"],
            "reset_identity_set_sha256": comparator_identity[
                "reset_identity_set_sha256"
            ],
            "reset_row_set_sha256": comparator_identity["reset_row_set_sha256"],
            "reset_manifest_sha256": hashlib.sha256(_manifest_bytes(rows)).hexdigest(),
            "calibration_reset_identity_overlap_count": 0,
            "calibration_reset_row_overlap_count": 0,
        }

    quality_schema = runtime["task_quality_schema_manifest"](task)
    config_sha256 = runtime["task_config_sha256"](task)
    if quality_schema.get("task_config_sha256") != config_sha256:
        raise ContractError(f"{task} quality schema/config identity mismatch")
    schema_sha256 = _require_sha256(
        quality_schema.get("schema_sha256"),
        f"{task} quality schema",
    )
    task_quality_schema_version = quality_schema.get("schema_version")
    if (
        not isinstance(task_quality_schema_version, str)
        or not task_quality_schema_version
    ):
        raise ContractError(f"{task} quality schema version is absent")
    contract = {
        "schema_version": TASK_CONTRACT_SCHEMA,
        "wave_id": wave["wave_id"],
        "wave_contract_sha256": wave_sha256,
        "ordinal": TASK_ORDER.index(task),
        "task_id": task,
        "scientific_partition": SCIENTIFIC_PARTITION,
        "transport_split": TRANSPORT_SPLIT,
        "manifest_seed": seed,
        "reset_count": EPISODES_PER_TASK,
        "task_output_root_relative": f"tasks/{task}",
        "reset_manifest_relative_path": f"tasks/{task}/reset_manifest.jsonl",
        "reset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        **identity,
        "planner_dominance_calibration": wave["planner_dominance_calibration"],
        "disjointness": {
            "comparators": comparisons,
            "test_partitions": {
                name: {
                    **dict(partitions[name]),
                    "identity_disjointness_declaration": (
                        "reserved_distinct_manifest_seed; rows remain sealed"
                    ),
                    "reset_overlap_status": "not_inspected_sealed_test",
                }
                for name in ("test_id", "test_ood")
            },
        },
        "config_identity": {"task_config_sha256": config_sha256},
        "quality_identity": {
            "evaluator_backend_id": TASK_QUALITY_BACKEND_ID,
            "task_quality_schema": quality_schema,
            "task_quality_schema_version": task_quality_schema_version,
            "task_quality_schema_sha256": schema_sha256,
            "quality_v2_schema_version": QUALITY_V2_SCHEMA,
        },
        "source_identity": wave["source_identity"],
        "runtime_module_identities": _runtime_module_identities(
            benchmark_source_root,
            wave["source_identity"]["mounted_sources"]["benchmark_se3_wam"]["files"],
        ),
    }
    return contract, manifest_bytes


def _load_task_contract(
    wave_root: Path,
    wave: Mapping[str, Any],
    wave_sha256: str,
    task: str,
) -> tuple[dict[str, Any], str]:
    if task not in TASK_ORDER:
        raise ContractError(f"unknown calibration task {task!r}")
    task_root = wave_root / "tasks" / task
    contract_path = task_root / "contract.json"
    contract = _load_json(contract_path, f"{task} contract")
    if contract.get("schema_version") != TASK_CONTRACT_SCHEMA:
        raise ContractError(f"{task} contract schema drifted")
    expected_scalars = {
        "wave_id": wave["wave_id"],
        "wave_contract_sha256": wave_sha256,
        "ordinal": TASK_ORDER.index(task),
        "task_id": task,
        "scientific_partition": SCIENTIFIC_PARTITION,
        "transport_split": TRANSPORT_SPLIT,
        "manifest_seed": wave["manifest_seed"],
        "reset_count": EPISODES_PER_TASK,
        "task_output_root_relative": f"tasks/{task}",
        "reset_manifest_relative_path": f"tasks/{task}/reset_manifest.jsonl",
    }
    for name, expected in expected_scalars.items():
        if contract.get(name) != expected:
            raise ContractError(f"{task} contract {name} drifted")
    if contract.get("source_identity") != wave["source_identity"]:
        raise ContractError(f"{task} contract source identity drifted")
    if (
        contract.get("planner_dominance_calibration")
        != wave["planner_dominance_calibration"]
    ):
        raise ContractError(f"{task} planner-dominance calibration policy drifted")
    expected_runtime_modules = wave["source_identity"]["mounted_sources"][
        "benchmark_se3_wam"
    ]["files"]
    if contract.get("runtime_module_identities") != expected_runtime_modules:
        raise ContractError(f"{task} contract runtime module identities drifted")
    config_identity = _require_mapping(
        contract.get("config_identity"), f"{task} config identity"
    )
    config_sha256 = _require_sha256(
        config_identity.get("task_config_sha256"), f"{task} config identity"
    )
    quality_identity = _require_mapping(
        contract.get("quality_identity"), f"{task} quality identity"
    )
    quality_schema = _require_mapping(
        quality_identity.get("task_quality_schema"), f"{task} quality schema"
    )
    quality_schema_sha256 = _require_sha256(
        quality_identity.get("task_quality_schema_sha256"),
        f"{task} quality schema",
    )
    quality_schema_version = quality_identity.get("task_quality_schema_version")
    if (
        not isinstance(quality_schema_version, str)
        or not quality_schema_version
        or quality_identity.get("evaluator_backend_id") != TASK_QUALITY_BACKEND_ID
        or quality_identity.get("quality_v2_schema_version") != QUALITY_V2_SCHEMA
        or quality_schema.get("task_id") != task
        or quality_schema.get("task_config_sha256") != config_sha256
        or quality_schema.get("schema_version") != quality_schema_version
        or quality_schema.get("schema_sha256") != quality_schema_sha256
    ):
        raise ContractError(f"{task} quality/config identity drifted")
    disjointness = _require_mapping(
        contract.get("disjointness"), f"{task} disjointness"
    )
    comparators = _require_mapping(
        disjointness.get("comparators"), f"{task} disjointness comparators"
    )
    if set(comparators) != {"validation", "review"}:
        raise ContractError(f"{task} disjointness comparator set drifted")
    for partition_name in ("validation", "review"):
        comparator = _require_mapping(
            comparators[partition_name], f"{task} {partition_name} comparator"
        )
        partition = wave["partitions"][partition_name]
        if (
            comparator.get("scientific_partition") != partition_name
            or comparator.get("transport_split") != partition["transport_split"]
            or comparator.get("manifest_seed") != partition["manifest_seed"]
            or comparator.get("calibration_reset_identity_overlap_count") != 0
            or comparator.get("calibration_reset_row_overlap_count") != 0
        ):
            raise ContractError(f"{task} {partition_name} comparator drifted")
        for hash_name in (
            "reset_identity_set_sha256",
            "reset_row_set_sha256",
            "reset_manifest_sha256",
        ):
            _require_sha256(
                comparator.get(hash_name),
                f"{task} {partition_name} comparator {hash_name}",
            )
    test_partitions = _require_mapping(
        disjointness.get("test_partitions"), f"{task} sealed test partitions"
    )
    expected_test_partitions = {
        name: {
            **dict(wave["partitions"][name]),
            "identity_disjointness_declaration": (
                "reserved_distinct_manifest_seed; rows remain sealed"
            ),
            "reset_overlap_status": "not_inspected_sealed_test",
        }
        for name in ("test_id", "test_ood")
    }
    if test_partitions != expected_test_partitions:
        raise ContractError(f"{task} sealed test declarations drifted")
    identities = contract.get("reset_identities")
    if not isinstance(identities, list) or len(identities) != EPISODES_PER_TASK:
        raise ContractError(f"{task} contract reset identities are not exact-20")
    expected_ordinals = list(range(EPISODES_PER_TASK))
    if [
        item.get("ordinal") for item in identities if isinstance(item, dict)
    ] != expected_ordinals:
        raise ContractError(f"{task} reset identity order drifted")
    episode_ids = []
    row_hashes = []
    for item in identities:
        item = _require_mapping(item, f"{task} reset identity")
        episode_id = item.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ContractError(f"{task} reset episode identity is invalid")
        episode_ids.append(episode_id)
        row_hashes.append(_require_sha256(item.get("reset_row_sha256"), "reset row"))
    if (
        len(set(episode_ids)) != EPISODES_PER_TASK
        or len(set(row_hashes)) != EPISODES_PER_TASK
    ):
        raise ContractError(f"{task} reset identities are not unique")
    if contract.get("reset_identity_set_sha256") != _payload_sha256(
        sorted(episode_ids)
    ):
        raise ContractError(f"{task} reset identity-set hash drifted")
    if contract.get("reset_row_set_sha256") != _payload_sha256(sorted(row_hashes)):
        raise ContractError(f"{task} reset row-set hash drifted")
    manifest_path = task_root / "reset_manifest.jsonl"
    if _sha256(manifest_path) != contract.get("reset_manifest_sha256"):
        raise ContractError(f"{task} reset manifest hash mismatch")
    payloads = _load_jsonl(manifest_path, f"{task} reset manifest")
    if _reset_identity(payloads) != {
        "reset_identities": identities,
        "reset_identity_set_sha256": contract["reset_identity_set_sha256"],
        "reset_row_set_sha256": contract["reset_row_set_sha256"],
    }:
        raise ContractError(
            f"{task} reset manifest identities do not match its contract"
        )
    digest = _sha256(contract_path)
    if digest != _payload_sha256(contract):
        raise ContractError(f"{task} contract is not canonical JSON")
    expected_sums = (
        f"{digest}  contract.json\n"
        f"{contract['reset_manifest_sha256']}  reset_manifest.jsonl\n"
    )
    sums_path = task_root / "PREDECLARATION_SHA256SUMS"
    if (
        not sums_path.is_file()
        or sums_path.read_text(encoding="utf-8") != expected_sums
    ):
        raise ContractError(f"{task} predeclaration SHA256SUMS is invalid")
    return contract, digest


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ContractError(f"cannot read {label} {path}: {error}") from error
    rows = []
    for index, line in enumerate(lines):
        if not line:
            raise ContractError(f"{label} contains a blank row at {index + 1}")
        try:
            value = json.loads(line, parse_constant=_reject_nonfinite)
        except (json.JSONDecodeError, ValueError) as error:
            raise ContractError(
                f"{label} row {index + 1} is invalid: {error}"
            ) from error
        value = _require_mapping(value, f"{label} row {index + 1}")
        if line != _canonical_json(value):
            raise ContractError(f"{label} row {index + 1} is not canonical JSON")
        rows.append(value)
    return rows


def _verify_source_files(
    wave: Mapping[str, Any],
    source_root: Path,
    benchmark_source_root: Path,
) -> None:
    source_root = source_root.resolve()
    benchmark_source_root = benchmark_source_root.resolve()
    source = wave["source_identity"]
    _require_clean_worktree(source_root, "runtime evaluator RLinf")
    _require_clean_worktree(benchmark_source_root, "runtime benchmark SE3-WAM")
    if _git_head(source_root) != source["evaluator_rlinf_commit"]:
        raise ContractError("runtime evaluator Git commit drifted")
    if _git_head(benchmark_source_root) != source["benchmark_commit"]:
        raise ContractError("runtime benchmark Git commit drifted")
    actual = _git_file_identities(
        source_root,
        SOURCE_FILES,
        "evaluator RLinf",
    )
    if actual != source["files"]:
        raise ContractError("runtime evaluator/launcher/helper source bytes drifted")
    expected_benchmark = source["mounted_sources"]["benchmark_se3_wam"]["files"]
    actual_benchmark = _git_file_identities(
        benchmark_source_root,
        BENCHMARK_SOURCE_FILES,
        "benchmark SE3-WAM",
    )
    if actual_benchmark != expected_benchmark:
        raise ContractError("runtime benchmark source bytes drifted")


def _verify_runtime_task(
    wave_root: Path,
    wave: Mapping[str, Any],
    wave_sha256: str,
    task: str,
    source_root: Path,
    benchmark_source_root: Path,
) -> tuple[dict[str, Any], str]:
    _verify_source_files(wave, source_root, benchmark_source_root)
    contract, contract_sha256 = _load_task_contract(wave_root, wave, wave_sha256, task)
    expected, expected_manifest = _task_contract(
        wave,
        wave_sha256,
        task,
        benchmark_source_root,
    )
    if expected != contract:
        raise ContractError(f"{task} runtime contract/config/schema identity drifted")
    if (
        wave_root / "tasks" / task / "reset_manifest.jsonl"
    ).read_bytes() != expected_manifest:
        raise ContractError(f"{task} runtime reset identities drifted")
    return contract, contract_sha256


def _predeclaration_receipt(
    wave_root: Path,
    wave: Mapping[str, Any],
    wave_sha256: str,
) -> dict[str, Any]:
    tasks = []
    for task in TASK_ORDER:
        contract, contract_sha256 = _load_task_contract(
            wave_root, wave, wave_sha256, task
        )
        tasks.append(
            {
                "ordinal": TASK_ORDER.index(task),
                "task_id": task,
                "task_contract_sha256": contract_sha256,
                "reset_manifest_relative_path": contract[
                    "reset_manifest_relative_path"
                ],
                "reset_manifest_sha256": contract["reset_manifest_sha256"],
                "reset_identity_set_sha256": contract["reset_identity_set_sha256"],
                "reset_row_set_sha256": contract["reset_row_set_sha256"],
                "reset_count": contract["reset_count"],
                "task_config_sha256": contract["config_identity"]["task_config_sha256"],
                "task_quality_schema_sha256": contract["quality_identity"][
                    "task_quality_schema_sha256"
                ],
                "task_quality_schema_version": contract["quality_identity"][
                    "task_quality_schema_version"
                ],
            }
        )
    return {
        "schema_version": PREDECLARATION_RECEIPT_SCHEMA,
        "wave_id": wave["wave_id"],
        "wave_contract_sha256": wave_sha256,
        "scientific_partition": SCIENTIFIC_PARTITION,
        "transport_split": TRANSPORT_SPLIT,
        "manifest_seed": wave["manifest_seed"],
        "task_count": len(TASK_ORDER),
        "episodes_per_task": EPISODES_PER_TASK,
        "total_reset_count": len(TASK_ORDER) * EPISODES_PER_TASK,
        "total_fresh_environment_replay_count": (
            len(TASK_ORDER) * PLANNER_CALIBRATION_REPLAY_COUNT
        ),
        "task_order": list(TASK_ORDER),
        "planner_dominance_calibration": wave["planner_dominance_calibration"],
        "sealed_test_policy": wave["sealed_test_policy"],
        "tasks": tasks,
    }


def _validate_predeclaration_seal(
    wave_root: Path,
    wave: Mapping[str, Any],
    wave_sha256: str,
) -> tuple[dict[str, Any], str]:
    expected = _predeclaration_receipt(wave_root, wave, wave_sha256)
    path = wave_root / "predeclaration_receipt.json"
    actual = _load_json(path, "predeclaration receipt")
    if actual != expected:
        raise ContractError(
            "predeclaration receipt does not match all exact-14 task contracts"
        )
    digest = _sha256(path)
    if digest != _payload_sha256(actual):
        raise ContractError("predeclaration receipt is not canonical JSON")
    rows = [(wave_sha256, "wave_contract.json")]
    for item in expected["tasks"]:
        task = item["task_id"]
        rows.extend(
            (
                (item["task_contract_sha256"], f"tasks/{task}/contract.json"),
                (item["reset_manifest_sha256"], f"tasks/{task}/reset_manifest.jsonl"),
            )
        )
    rows.append((digest, "predeclaration_receipt.json"))
    sums_path = wave_root / "PREDECLARATION_SHA256SUMS"
    expected_sums = "".join(f"{sha}  {path}\n" for sha, path in rows)
    if (
        not sums_path.is_file()
        or sums_path.read_text(encoding="utf-8") != expected_sums
    ):
        raise ContractError("wave predeclaration SHA256SUMS is invalid")
    return actual, digest


def _verify_evaluator_sums(evaluator_root: Path) -> None:
    sums = evaluator_root / "SHA256SUMS"
    if not sums.is_file():
        raise ContractError("evaluator SHA256SUMS is absent")
    expected_names = {"evaluation.json", "reset_manifest.jsonl"}
    found: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None or match.group(2) in found:
            raise ContractError("evaluator SHA256SUMS has an invalid row")
        found[match.group(2)] = match.group(1)
    if set(found) != expected_names:
        raise ContractError("evaluator SHA256SUMS file set drifted")
    for name, expected in found.items():
        if _sha256(evaluator_root / name) != expected:
            raise ContractError(f"evaluator artifact hash mismatch: {name}")


def _validate_finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{label}[{index}]")


def _validate_evaluation(
    wave_root: Path,
    wave: Mapping[str, Any],
    task_contract: Mapping[str, Any],
    evaluator_root: Path,
) -> dict[str, Any]:
    _verify_evaluator_sums(evaluator_root)
    frozen_manifest = wave_root / task_contract["reset_manifest_relative_path"]
    evaluator_manifest = evaluator_root / "reset_manifest.jsonl"
    if evaluator_manifest.read_bytes() != frozen_manifest.read_bytes():
        raise ContractError(
            "evaluator reset manifest differs byte-for-byte from predeclaration"
        )
    evaluation_path = evaluator_root / "evaluation.json"
    evaluation = _load_json(evaluation_path, f"{task_contract['task_id']} evaluation")
    _validate_finite_json(evaluation, "evaluation")
    task = task_contract["task_id"]
    expected_scalars = {
        "schema_version": EVALUATION_SCHEMA,
        "planner_identity": {"task": task, "kind": "privileged_teacher"},
        "source_identity": {
            "evaluator_rlinf_commit": wave["source_identity"]["evaluator_rlinf_commit"],
            "benchmark_commit": wave["source_identity"]["benchmark_commit"],
        },
        "split": TRANSPORT_SPLIT,
        "manifest_seed": wave["manifest_seed"],
        "reset_manifest_sha256": task_contract["reset_manifest_sha256"],
        "episodes": EPISODES_PER_TASK,
        "all_replays_passed": True,
    }
    for name, expected in expected_scalars.items():
        if evaluation.get(name) != expected:
            raise ContractError(f"{task} evaluation field {name} drifted")
    payload_digest = evaluation.get("payload_sha256")
    _require_sha256(payload_digest, f"{task} evaluation payload")
    unsigned = dict(evaluation)
    unsigned.pop("payload_sha256", None)
    if _payload_sha256(unsigned) != payload_digest:
        raise ContractError(f"{task} evaluation payload hash mismatch")
    records = evaluation.get("records")
    if not isinstance(records, list) or len(records) != EPISODES_PER_TASK:
        raise ContractError(f"{task} evaluation records are not exact-20")
    manifest_rows = _load_jsonl(frozen_manifest, f"{task} frozen reset manifest")
    expected_episode_ids = [row["episode_id"] for row in manifest_rows]
    actual_episode_ids = [
        row.get("episode_id") if isinstance(row, dict) else None for row in records
    ]
    if actual_episode_ids != expected_episode_ids:
        raise ContractError(
            f"{task} evaluation record order diverged from reset identities"
        )
    for ordinal, (record, reset) in enumerate(zip(records, manifest_rows, strict=True)):
        record = _require_mapping(record, f"{task} evaluation record {ordinal}")
        for name in (
            "episode_id",
            "task_id",
            "seed",
            "factors",
            "source_group_id",
            "pair_id",
            "pair_member_id",
            "candidate_index",
        ):
            if record.get(name) != reset.get(name):
                raise ContractError(
                    f"{task} record {ordinal} reset field {name} drifted"
                )
        replay = _require_mapping(
            record.get("replay_validation"), f"{task} record {ordinal} replay"
        )
        if replay.get("passed") is not True:
            raise ContractError(f"{task} record {ordinal} replay did not pass")
        quality = _require_mapping(
            record.get("quality_v2"), f"{task} record {ordinal} quality_v2"
        )
        if quality.get("schema_version") != QUALITY_V2_SCHEMA:
            raise ContractError(f"{task} record {ordinal} quality-v2 schema drifted")
        quality_sha = _require_sha256(
            record.get("quality_v2_sha256"), f"{task} record {ordinal} quality-v2"
        )
        if _payload_sha256(quality) != quality_sha:
            raise ContractError(f"{task} record {ordinal} quality-v2 hash mismatch")
        task_quality = record.get("task_quality")
        if task_quality is not None:
            task_quality = _require_mapping(
                task_quality, f"{task} record {ordinal} task quality"
            )
            quality_identity = task_contract["quality_identity"]
            if (
                task_quality.get("task_id") != task
                or task_quality.get("evaluator_backend_id")
                != quality_identity["evaluator_backend_id"]
                or task_quality.get("schema_version")
                != quality_identity["task_quality_schema_version"]
                or task_quality.get("schema_sha256")
                != quality_identity["task_quality_schema_sha256"]
            ):
                raise ContractError(
                    f"{task} record {ordinal} task-quality identity drifted"
                )
            summary_sha = _require_sha256(
                task_quality.get("summary_sha256"),
                f"{task} record {ordinal} task-quality summary",
            )
            unsigned_quality = dict(task_quality)
            unsigned_quality.pop("summary_sha256", None)
            if _payload_sha256(unsigned_quality) != summary_sha:
                raise ContractError(
                    f"{task} record {ordinal} task-quality hash mismatch"
                )
    summary = _require_mapping(evaluation.get("task_summary"), f"{task} task summary")
    task_summary = _require_mapping(summary.get(task), f"{task} task summary row")
    if task_summary.get("episode_count") != EPISODES_PER_TASK:
        raise ContractError(f"{task} task summary is not exact-20")
    return evaluation


def _artifact_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Match the established RLD2 evidence builder's on-disk JSON encoding."""

    return (json.dumps(dict(value), allow_nan=False, indent=2) + "\n").encode("utf-8")


def _selected_planner_calibration_record(
    wave_root: Path,
    task_contract: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> tuple[int, dict[str, Any], dict[str, Any], Any]:
    """Select and validate the first safe success in frozen reset order."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - present in the formal image.
        raise ContractError("planner calibration requires NumPy") from error

    task = task_contract["task_id"]
    records = evaluation.get("records")
    if not isinstance(records, list) or len(records) != EPISODES_PER_TASK:
        raise ContractError(f"{task} evaluation is not exact-20")
    reset_rows = _load_jsonl(
        wave_root / task_contract["reset_manifest_relative_path"],
        f"{task} frozen reset manifest",
    )
    identities = task_contract["reset_identities"]
    for ordinal, raw_record in enumerate(records):
        record = _require_mapping(raw_record, f"{task} evaluation record {ordinal}")
        if (
            record.get("success") is not True
            or record.get("safety_failure") is not False
        ):
            continue
        task_quality = record.get("task_quality")
        if not isinstance(task_quality, Mapping):
            raise ContractError(
                f"{task} first safe success has no measured task-quality summary"
            )
        control_steps = record.get("control_steps")
        if (
            isinstance(control_steps, bool)
            or not isinstance(control_steps, int)
            or control_steps < 1
        ):
            raise ContractError(
                f"{task} selected action tape has invalid control_steps"
            )
        raw_actions = record.get("actions")
        if not isinstance(raw_actions, list) or len(raw_actions) != control_steps:
            raise ContractError(f"{task} selected action tape length drifted")
        for step, raw_action in enumerate(raw_actions):
            if not isinstance(raw_action, list) or not raw_action:
                raise ContractError(f"{task} selected action {step} is not a vector")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_action
            ):
                raise ContractError(f"{task} selected action {step} is not numeric")
        try:
            # The planner evaluator hashes its inline recorded NumPy action
            # array as float64. Down-casting the JSON values here would silently
            # change that calibration-only action identity before replay.
            actions = np.ascontiguousarray(np.asarray(raw_actions, dtype=np.float64))
        except (TypeError, ValueError, OverflowError) as error:
            raise ContractError(
                f"{task} selected action tape is invalid: {error}"
            ) from error
        if (
            actions.ndim != 2
            or actions.shape[0] != control_steps
            or actions.shape[1] < 1
            or not np.isfinite(actions).all()
            or float(np.max(np.abs(actions))) > 1.0
        ):
            raise ContractError(
                f"{task} selected action tape is not finite and bounded"
            )
        action_sha256 = _require_sha256(
            record.get("action_sha256"), f"{task} selected action tape"
        )
        if hashlib.sha256(actions.tobytes()).hexdigest() != action_sha256:
            raise ContractError(f"{task} selected action tape hash mismatch")
        recorded_action_l2 = record.get("action_l2_sum")
        recomputed_action_l2 = float(np.square(actions).sum())
        if (
            isinstance(recorded_action_l2, bool)
            or not isinstance(recorded_action_l2, (int, float))
            or not math.isfinite(float(recorded_action_l2))
            or not math.isclose(
                float(recorded_action_l2),
                recomputed_action_l2,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ContractError(f"{task} selected action L2 summary drifted")
        termination_reason = record.get("termination_reason")
        if not isinstance(termination_reason, str) or not termination_reason:
            raise ContractError(f"{task} selected discrete outcome is incomplete")
        reset = reset_rows[ordinal]
        identity = identities[ordinal]
        if (
            reset.get("episode_id") != record.get("episode_id")
            or identity.get("episode_id") != record.get("episode_id")
            or identity.get("reset_row_sha256") != _payload_sha256(reset)
        ):
            raise ContractError(f"{task} selected reset identity drifted")
        return ordinal, dict(record), reset, actions
    raise ContractError(f"{task} exact-20 wave has no safe successful planner reset")


def _calibration_evaluator_identity(wave: Mapping[str, Any]) -> dict[str, str]:
    return {
        "evaluator_rlinf_commit": wave["source_identity"]["evaluator_rlinf_commit"],
        "evaluator_benchmark_commit": wave["source_identity"]["benchmark_commit"],
        "backend_id": TASK_QUALITY_BACKEND_ID,
    }


def _planner_contract_template(task_contract: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from examples.embodiment.prepare_dynamic_benchmark_rld2_launch_gate import (
            _contract_template,
        )
    except ImportError as error:
        raise ContractError(
            "cannot import the frozen planner contract template builder"
        ) from error
    template = _contract_template(
        task_contract["task_id"],
        task_contract["quality_identity"]["task_quality_schema"],
        TASK_QUALITY_BACKEND_ID,
    )
    if (
        template.get("schema_version") != PLANNER_DOMINANCE_SCHEMA
        or template.get("quality_schema")
        != task_contract["quality_identity"]["task_quality_schema"]
    ):
        raise ContractError(
            f"{task_contract['task_id']} planner contract template identity drifted"
        )
    return template


def _fresh_environment_replays(
    wave: Mapping[str, Any],
    task_contract: Mapping[str, Any],
    *,
    selected_ordinal: int,
    selected_record: Mapping[str, Any],
    selected_reset: Mapping[str, Any],
    actions: Any,
) -> list[dict[str, Any]]:
    """Replay one recorded action tape in three newly constructed environments."""

    try:
        from se3_wam.benchmark.evaluation import manifest_record

        from examples.embodiment.run_dynamic_benchmark_rld2_launch_gate import (
            _replay_planner_actions,
        )
        from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import (
            DynamicBenchmarkEnv,
        )
    except ImportError as error:
        raise ContractError(
            "cannot import the frozen planner calibration replay runtime"
        ) from error

    task = task_contract["task_id"]
    reset_request_sha256 = _payload_sha256(dict(selected_reset))
    action_sha256 = selected_record["action_sha256"]
    quality_identity = task_contract["quality_identity"]
    environment_config = {
        "task_id": task,
        "split": TRANSPORT_SPLIT,
        "manifest_seed": wave["manifest_seed"],
        "manifest_size": EPISODES_PER_TASK,
        "image_size": wave["planner_dominance_calibration"]["image_size"],
        "camera_observations": False,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
        "task_quality_schema_version": quality_identity["task_quality_schema_version"],
        "task_quality_evaluator_backend_id": TASK_QUALITY_BACKEND_ID,
    }
    replays: list[dict[str, Any]] = []
    for replay_index in range(PLANNER_CALIBRATION_REPLAY_COUNT):
        try:
            env = DynamicBenchmarkEnv(
                cfg=environment_config,
                num_envs=1,
                seed_offset=0,
                total_num_processes=1,
                worker_info=None,
            )
        except Exception as error:
            raise ContractError(
                f"{task} cannot construct fresh calibration environment {replay_index}: {error}"
            ) from error
        try:
            rows = list(env._manifest_rows[:EPISODES_PER_TASK])
            if len(rows) != EPISODES_PER_TASK:
                raise ContractError(
                    f"{task} fresh environment manifest is not exact-20"
                )
            row = rows[selected_ordinal]
            actual_reset = dict(manifest_record(row))
            if actual_reset != dict(selected_reset):
                raise ContractError(
                    f"{task} fresh environment reset {replay_index} drifted from predeclaration"
                )
            replay = _replay_planner_actions(
                env=env,
                request=row.request,
                task=task,
                actions=actions,
                replay_index=replay_index,
                reset_request_sha256=reset_request_sha256,
                action_sha256=action_sha256,
            )
        except ContractError:
            raise
        except Exception as error:
            raise ContractError(
                f"{task} fresh environment replay {replay_index} failed: {error}"
            ) from error
        finally:
            env.close()
        if (
            replay.get("episode_id") != selected_record.get("episode_id")
            or replay.get("reset_request_sha256") != reset_request_sha256
            or replay.get("action_sha256") != action_sha256
            or replay.get("success") is not True
            or replay.get("safety_failure") is not False
            or replay.get("termination_reason")
            != selected_record.get("termination_reason")
            or replay.get("control_steps") != selected_record.get("control_steps")
        ):
            raise ContractError(
                f"{task} replay {replay_index} action/reset/discrete outcome drifted"
            )
        replays.append(dict(replay))
    environment_ids = [row.get("environment_instance_id") for row in replays]
    if (
        any(not isinstance(value, str) or not value for value in environment_ids)
        or len(set(environment_ids)) != PLANNER_CALIBRATION_REPLAY_COUNT
    ):
        raise ContractError(
            f"{task} calibration environments are not uniquely identified"
        )
    return replays


def _planner_calibration_root(attempt_root: Path) -> Path:
    return attempt_root / "planner_calibration"


def _reject_interrupted_planner_calibration_staging(calibration_root: Path) -> None:
    prefix = f".{calibration_root.name}."
    interrupted = sorted(
        path.name
        for path in calibration_root.parent.iterdir()
        if path.name.startswith(prefix) and path.name.endswith(".tmp")
    )
    if interrupted:
        raise ContractError(
            "ambiguous interrupted planner calibration staging exists: "
            + ", ".join(interrupted)
        )


def _build_planner_calibration(
    wave_root: Path,
    wave: Mapping[str, Any],
    task_contract: Mapping[str, Any],
    attempt_root: Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    calibration_root = _planner_calibration_root(attempt_root)
    _reject_interrupted_planner_calibration_staging(calibration_root)
    if calibration_root.exists():
        return _load_planner_calibration(
            wave_root,
            wave,
            task_contract,
            attempt_root,
            evaluation,
        )
    selected_ordinal, selected_record, selected_reset, actions = (
        _selected_planner_calibration_record(wave_root, task_contract, evaluation)
    )
    replays = _fresh_environment_replays(
        wave,
        task_contract,
        selected_ordinal=selected_ordinal,
        selected_record=selected_record,
        selected_reset=selected_reset,
        actions=actions,
    )
    evaluator_identity = _calibration_evaluator_identity(wave)
    reset_manifest_bytes = _manifest_bytes((selected_reset,))
    reset_manifest_sha256 = hashlib.sha256(reset_manifest_bytes).hexdigest()
    calibration_relative_root = calibration_root.relative_to(wave_root).as_posix()
    evidence_reference = f"{calibration_relative_root}/calibration_evidence.json"
    calibration_input = {
        "task": task_contract["task_id"],
        "backend_id": TASK_QUALITY_BACKEND_ID,
        "evaluator_identity": evaluator_identity,
        "split": TRANSPORT_SPLIT,
        "test_exposure": {"test_id": False, "test_ood": False},
        "reset_manifest_sha256": reset_manifest_sha256,
        "replays": replays,
    }
    try:
        from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
            build_calibration_evidence,
            validate_calibration_evidence,
        )

        evidence, planner_contract = build_calibration_evidence(
            calibration_input,
            contract_template=_planner_contract_template(task_contract),
            evidence_reference=evidence_reference,
        )
        validate_calibration_evidence(
            evidence,
            contract=planner_contract,
            evaluator_identity=evaluator_identity,
        )
    except (ImportError, TypeError, ValueError) as error:
        raise ContractError(
            f"{task_contract['task_id']} planner calibration evidence build failed: {error}"
        ) from error

    staging = calibration_root.parent / (
        f".{calibration_root.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_exact_bytes(
            staging / "selected_reset_manifest.jsonl", reset_manifest_bytes
        )
        action_buffer = io.BytesIO()
        import numpy as np

        np.save(action_buffer, actions, allow_pickle=False)
        _write_exact_bytes(staging / "planner_actions.npy", action_buffer.getvalue())
        _write_exact_bytes(
            staging / "calibration_input.json", _artifact_json_bytes(calibration_input)
        )
        _write_exact_bytes(
            staging / "calibration_evidence.json", _artifact_json_bytes(evidence)
        )
        _write_exact_bytes(
            staging / "planner_dominance_contract.json",
            _artifact_json_bytes(planner_contract),
        )
        _write_exact_sha256sums(
            staging / "SHA256SUMS",
            tuple(
                (_sha256(staging / filename), filename)
                for filename, _, _ in PLANNER_CALIBRATION_ARTIFACTS
            ),
        )
        os.replace(staging, calibration_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _load_planner_calibration(
        wave_root,
        wave,
        task_contract,
        attempt_root,
        evaluation,
    )


def _load_planner_calibration(
    wave_root: Path,
    wave: Mapping[str, Any],
    task_contract: Mapping[str, Any],
    attempt_root: Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        import numpy as np

        from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
            build_calibration_evidence,
            validate_calibration_evidence,
        )
    except ImportError as error:
        raise ContractError(
            "cannot import planner calibration validation runtime"
        ) from error

    task = task_contract["task_id"]
    calibration_root = _planner_calibration_root(attempt_root)
    _reject_interrupted_planner_calibration_staging(calibration_root)
    expected_names = {
        *(filename for filename, _, _ in PLANNER_CALIBRATION_ARTIFACTS),
        "SHA256SUMS",
    }
    if (
        not calibration_root.is_dir()
        or {path.name for path in calibration_root.iterdir()} != expected_names
    ):
        raise ContractError(f"{task} planner calibration artifact inventory drifted")
    artifact_hashes = {
        filename: _sha256(calibration_root / filename)
        for filename, _, _ in PLANNER_CALIBRATION_ARTIFACTS
    }
    expected_sums = "".join(
        f"{artifact_hashes[filename]}  {filename}\n"
        for filename, _, _ in PLANNER_CALIBRATION_ARTIFACTS
    )
    sums_path = calibration_root / "SHA256SUMS"
    if sums_path.read_text(encoding="utf-8") != expected_sums:
        raise ContractError(f"{task} planner calibration SHA256SUMS drifted")

    selected_ordinal, selected_record, selected_reset, selected_actions = (
        _selected_planner_calibration_record(wave_root, task_contract, evaluation)
    )
    reset_manifest_path = calibration_root / "selected_reset_manifest.jsonl"
    if reset_manifest_path.read_bytes() != _manifest_bytes((selected_reset,)):
        raise ContractError(f"{task} selected calibration reset manifest drifted")
    try:
        stored_actions = np.load(
            calibration_root / "planner_actions.npy", allow_pickle=False
        )
    except (OSError, ValueError) as error:
        raise ContractError(
            f"{task} cannot read stored planner action tape: {error}"
        ) from error
    if (
        stored_actions.dtype != np.dtype(np.float64)
        or not stored_actions.flags.c_contiguous
        or not np.array_equal(stored_actions, selected_actions)
        or hashlib.sha256(stored_actions.tobytes()).hexdigest()
        != selected_record["action_sha256"]
    ):
        raise ContractError(f"{task} stored planner action tape drifted")

    calibration_input = _load_json(
        calibration_root / "calibration_input.json", f"{task} calibration input"
    )
    evidence = _load_json(
        calibration_root / "calibration_evidence.json", f"{task} calibration evidence"
    )
    planner_contract = _load_json(
        calibration_root / "planner_dominance_contract.json",
        f"{task} planner-dominance contract",
    )
    calibration_relative_root = calibration_root.relative_to(wave_root).as_posix()
    evidence_reference = f"{calibration_relative_root}/calibration_evidence.json"
    evaluator_identity = _calibration_evaluator_identity(wave)
    reset_manifest_sha256 = artifact_hashes["selected_reset_manifest.jsonl"]
    if calibration_input != {
        "task": task,
        "backend_id": TASK_QUALITY_BACKEND_ID,
        "evaluator_identity": evaluator_identity,
        "split": TRANSPORT_SPLIT,
        "test_exposure": {"test_id": False, "test_ood": False},
        "reset_manifest_sha256": reset_manifest_sha256,
        "replays": evidence.get("replays"),
    }:
        raise ContractError(f"{task} planner calibration input identity drifted")
    try:
        expected_evidence, expected_contract = build_calibration_evidence(
            calibration_input,
            contract_template=_planner_contract_template(task_contract),
            evidence_reference=evidence_reference,
        )
        if evidence != expected_evidence or planner_contract != expected_contract:
            raise ContractError(
                f"{task} calibration evidence/contract differs from the frozen builder"
            )
        validate_calibration_evidence(
            evidence,
            contract=planner_contract,
            evaluator_identity=evaluator_identity,
        )
    except ContractError:
        raise
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"{task} planner calibration validation failed: {error}"
        ) from error
    if (
        evidence.get("schema_version") != CALIBRATION_EVIDENCE_SCHEMA
        or planner_contract.get("schema_version") != PLANNER_DOMINANCE_SCHEMA
        or planner_contract.get("quality_schema")
        != task_contract["quality_identity"]["task_quality_schema"]
        or planner_contract.get("calibration", {}).get("evidence_path")
        != evidence_reference
        or planner_contract.get("calibration", {}).get("evidence_sha256")
        != artifact_hashes["calibration_evidence.json"]
        or planner_contract.get("calibration", {}).get("reset_episode_id")
        != selected_record["episode_id"]
    ):
        raise ContractError(f"{task} planner calibration binding drifted")
    replays = evidence.get("replays")
    if (
        not isinstance(replays, list)
        or len(replays) != PLANNER_CALIBRATION_REPLAY_COUNT
    ):
        raise ContractError(f"{task} planner calibration is not exact-three replay")
    environment_ids = [row.get("environment_instance_id") for row in replays]
    if (
        any(not isinstance(value, str) or not value for value in environment_ids)
        or len(set(environment_ids)) != PLANNER_CALIBRATION_REPLAY_COUNT
    ):
        raise ContractError(
            f"{task} planner calibration environment IDs are not unique"
        )
    for replay_index, replay in enumerate(replays):
        if (
            replay.get("replay_index") != replay_index
            or replay.get("episode_id") != selected_record["episode_id"]
            or replay.get("reset_request_sha256") != _payload_sha256(selected_reset)
            or replay.get("action_sha256") != selected_record["action_sha256"]
            or replay.get("success") is not True
            or replay.get("safety_failure") is not False
            or replay.get("termination_reason")
            != selected_record.get("termination_reason")
            or replay.get("control_steps") != selected_record.get("control_steps")
        ):
            raise ContractError(
                f"{task} planner calibration replay {replay_index} identity drifted"
            )
    task_quality_summary_sha256s = [
        _require_sha256(
            _require_mapping(
                row.get("task_quality"), f"{task} replay task quality"
            ).get("summary_sha256"),
            f"{task} replay task-quality summary",
        )
        for row in replays
    ]
    relative_root = calibration_root.relative_to(wave_root).as_posix()
    binding = {
        "schema_version": PLANNER_CALIBRATION_BINDING_SCHEMA,
        "task_id": task,
        "selection_policy": wave["planner_dominance_calibration"]["selection_policy"],
        "selected_reset_ordinal": selected_ordinal,
        "selected_episode_id": selected_record["episode_id"],
        "selected_reset_row_sha256": _payload_sha256(selected_reset),
        "planner_action_dtype": "float64",
        "planner_action_content_sha256": selected_record["action_sha256"],
        "source_discrete_outcome": {
            "success": True,
            "safety_failure": False,
            "termination_reason": selected_record["termination_reason"],
            "control_steps": selected_record["control_steps"],
        },
        "replay_count": PLANNER_CALIBRATION_REPLAY_COUNT,
        "environment_instance_ids": environment_ids,
        "replay_task_quality_summary_sha256s": task_quality_summary_sha256s,
        "test_exposure": {"test_id": False, "test_ood": False},
        "selected_reset_manifest_relative_path": (
            f"{relative_root}/selected_reset_manifest.jsonl"
        ),
        "selected_reset_manifest_sha256": artifact_hashes[
            "selected_reset_manifest.jsonl"
        ],
        "planner_actions_relative_path": f"{relative_root}/planner_actions.npy",
        "planner_actions_file_sha256": artifact_hashes["planner_actions.npy"],
        "calibration_input_relative_path": f"{relative_root}/calibration_input.json",
        "calibration_input_sha256": artifact_hashes["calibration_input.json"],
        "calibration_input_payload_sha256": _payload_sha256(calibration_input),
        "calibration_evidence_relative_path": (
            f"{relative_root}/calibration_evidence.json"
        ),
        "calibration_evidence_sha256": artifact_hashes["calibration_evidence.json"],
        "calibration_evidence_payload_sha256": evidence["payload_sha256"],
        "planner_dominance_contract_relative_path": (
            f"{relative_root}/planner_dominance_contract.json"
        ),
        "planner_dominance_contract_sha256": artifact_hashes[
            "planner_dominance_contract.json"
        ],
        "planner_dominance_contract_payload_sha256": _payload_sha256(planner_contract),
        "sha256sums_relative_path": f"{relative_root}/SHA256SUMS",
        "sha256sums_sha256": _sha256(sums_path),
    }
    return binding


def _planner_calibration_sha256_rows(
    binding: Mapping[str, Any], *, task_relative: bool
) -> tuple[tuple[str, str], ...]:
    prefix = f"tasks/{binding.get('task_id', '')}/"
    rows = []
    for _, path_key, sha_key in PLANNER_CALIBRATION_ARTIFACTS:
        relative_path = str(binding[path_key])
        if task_relative:
            if not relative_path.startswith(prefix):
                raise ContractError(
                    "planner calibration artifact escaped its task root"
                )
            relative_path = relative_path.removeprefix(prefix)
        rows.append((binding[sha_key], relative_path))
    sums_relative_path = str(binding["sha256sums_relative_path"])
    if task_relative:
        if not sums_relative_path.startswith(prefix):
            raise ContractError("planner calibration SHA256SUMS escaped its task root")
        sums_relative_path = sums_relative_path.removeprefix(prefix)
    rows.append((binding["sha256sums_sha256"], sums_relative_path))
    return tuple(rows)


def _attempt_roots(task_root: Path) -> list[Path]:
    attempts_root = task_root / "attempts"
    if not attempts_root.exists():
        return []
    roots = []
    for path in attempts_root.iterdir():
        if path.is_dir() and re.fullmatch(r"attempt-[0-9]{6}", path.name):
            roots.append(path)
    return sorted(roots, key=lambda item: item.name)


def _completed_evaluator_root(attempt_root: Path) -> Path | None:
    evaluator_root = attempt_root / "evaluator"
    required = ("evaluation.json", "reset_manifest.jsonl", "SHA256SUMS")
    return (
        evaluator_root
        if all((evaluator_root / name).is_file() for name in required)
        else None
    )


def _task_receipt(
    wave_root: Path,
    wave: Mapping[str, Any],
    wave_sha256: str,
    task_contract: Mapping[str, Any],
    task_contract_sha256: str,
    attempt_root: Path,
) -> dict[str, Any]:
    evaluator_root = attempt_root / "evaluator"
    evaluation = _validate_evaluation(wave_root, wave, task_contract, evaluator_root)
    task = task_contract["task_id"]
    task_root = wave_root / "tasks" / task
    planner_calibration = _load_planner_calibration(
        wave_root,
        wave,
        task_contract,
        attempt_root,
        evaluation,
    )
    evaluation_relative = (
        evaluator_root.relative_to(wave_root).as_posix() + "/evaluation.json"
    )
    attempt_relative = attempt_root.relative_to(task_root).as_posix()
    return {
        "schema_version": TASK_RECEIPT_SCHEMA,
        "wave_id": wave["wave_id"],
        "wave_contract_sha256": wave_sha256,
        "task_contract_sha256": task_contract_sha256,
        "ordinal": task_contract["ordinal"],
        "task_id": task,
        "scientific_partition": SCIENTIFIC_PARTITION,
        "transport_split": TRANSPORT_SPLIT,
        "manifest_seed": wave["manifest_seed"],
        "reset_count": EPISODES_PER_TASK,
        "reset_manifest_relative_path": task_contract["reset_manifest_relative_path"],
        "reset_manifest_sha256": task_contract["reset_manifest_sha256"],
        "reset_identity_set_sha256": task_contract["reset_identity_set_sha256"],
        "reset_row_set_sha256": task_contract["reset_row_set_sha256"],
        "reset_identities": task_contract["reset_identities"],
        "attempt_relative_path": attempt_relative,
        "evaluation_relative_path": evaluation_relative,
        "evaluation_sha256": _sha256(evaluator_root / "evaluation.json"),
        "evaluation_payload_sha256": evaluation["payload_sha256"],
        "evaluator_sha256sums_sha256": _sha256(evaluator_root / "SHA256SUMS"),
        "quality_v2_schema_version": QUALITY_V2_SCHEMA,
        "quality_v2_record_count": EPISODES_PER_TASK,
        "task_config_sha256": task_contract["config_identity"]["task_config_sha256"],
        "task_quality_schema_version": task_contract["quality_identity"][
            "task_quality_schema_version"
        ],
        "task_quality_schema_sha256": task_contract["quality_identity"][
            "task_quality_schema_sha256"
        ],
        "all_replays_passed": True,
        "planner_calibration": planner_calibration,
        "source_identity": wave["source_identity"],
    }


def _load_task_receipt(
    wave_root: Path,
    wave: Mapping[str, Any],
    wave_sha256: str,
    task: str,
) -> tuple[dict[str, Any], str]:
    contract, contract_sha256 = _load_task_contract(wave_root, wave, wave_sha256, task)
    task_root = wave_root / "tasks" / task
    receipt_path = task_root / "receipt.json"
    receipt = _load_json(receipt_path, f"{task} receipt")
    if receipt.get("schema_version") != TASK_RECEIPT_SCHEMA:
        raise ContractError(f"{task} receipt schema drifted")
    attempt_relative = receipt.get("attempt_relative_path")
    if (
        not isinstance(attempt_relative, str)
        or re.fullmatch(r"attempts/attempt-[0-9]{6}", attempt_relative) is None
    ):
        raise ContractError(f"{task} receipt attempt path is invalid")
    expected = _task_receipt(
        wave_root,
        wave,
        wave_sha256,
        contract,
        contract_sha256,
        task_root / Path(attempt_relative),
    )
    if receipt != expected:
        raise ContractError(f"{task} receipt does not match its frozen outputs")
    digest = _sha256(receipt_path)
    if digest != _payload_sha256(receipt):
        raise ContractError(f"{task} receipt is not canonical JSON")
    evaluator_root = wave_root / receipt["evaluation_relative_path"]
    sums_rows = (
        (contract_sha256, "contract.json"),
        (contract["reset_manifest_sha256"], "reset_manifest.jsonl"),
        (
            receipt["evaluation_sha256"],
            receipt["evaluation_relative_path"].removeprefix(f"tasks/{task}/"),
        ),
        (
            receipt["evaluator_sha256sums_sha256"],
            f"{attempt_relative}/evaluator/SHA256SUMS",
        ),
        (
            _sha256(evaluator_root.parent / "reset_manifest.jsonl"),
            f"{attempt_relative}/evaluator/reset_manifest.jsonl",
        ),
        *_planner_calibration_sha256_rows(
            receipt["planner_calibration"], task_relative=True
        ),
        (digest, "receipt.json"),
    )
    expected_sums = "".join(f"{sha}  {path}\n" for sha, path in sums_rows)
    sums_path = task_root / "SHA256SUMS"
    if (
        not sums_path.is_file()
        or sums_path.read_text(encoding="utf-8") != expected_sums
    ):
        raise ContractError(f"{task} final SHA256SUMS is invalid")
    return receipt, digest


def _finalize_task(
    wave_root: Path,
    wave: Mapping[str, Any],
    wave_sha256: str,
    task: str,
    attempt_root: Path,
) -> dict[str, Any]:
    contract, contract_sha256 = _load_task_contract(wave_root, wave, wave_sha256, task)
    task_root = wave_root / "tasks" / task
    evaluator_root = attempt_root / "evaluator"
    evaluation = _validate_evaluation(wave_root, wave, contract, evaluator_root)
    _build_planner_calibration(
        wave_root,
        wave,
        contract,
        attempt_root,
        evaluation,
    )
    receipt = _task_receipt(
        wave_root,
        wave,
        wave_sha256,
        contract,
        contract_sha256,
        attempt_root,
    )
    receipt_path = task_root / "receipt.json"
    _write_exact_json(receipt_path, receipt)
    receipt_sha256 = _sha256(receipt_path)
    attempt_relative = receipt["attempt_relative_path"]
    _write_exact_sha256sums(
        task_root / "SHA256SUMS",
        (
            (contract_sha256, "contract.json"),
            (contract["reset_manifest_sha256"], "reset_manifest.jsonl"),
            (
                receipt["evaluation_sha256"],
                f"{attempt_relative}/evaluator/evaluation.json",
            ),
            (
                receipt["evaluator_sha256sums_sha256"],
                f"{attempt_relative}/evaluator/SHA256SUMS",
            ),
            (
                _sha256(evaluator_root / "reset_manifest.jsonl"),
                f"{attempt_relative}/evaluator/reset_manifest.jsonl",
            ),
            *_planner_calibration_sha256_rows(
                receipt["planner_calibration"], task_relative=True
            ),
            (receipt_sha256, "receipt.json"),
        ),
    )
    _load_task_receipt(wave_root, wave, wave_sha256, task)
    return receipt


def _wave_receipt(
    wave_root: Path,
    wave: Mapping[str, Any],
    wave_sha256: str,
    predeclaration_sha256: str,
) -> dict[str, Any]:
    task_rows = []
    comparison_rows = []
    for task in TASK_ORDER:
        contract, _ = _load_task_contract(wave_root, wave, wave_sha256, task)
        receipt, receipt_sha256 = _load_task_receipt(wave_root, wave, wave_sha256, task)
        task_rows.append(
            {
                "ordinal": receipt["ordinal"],
                "task_id": task,
                "task_contract_sha256": receipt["task_contract_sha256"],
                "task_receipt_sha256": receipt_sha256,
                "reset_manifest_relative_path": receipt["reset_manifest_relative_path"],
                "reset_manifest_sha256": receipt["reset_manifest_sha256"],
                "reset_identity_set_sha256": receipt["reset_identity_set_sha256"],
                "reset_row_set_sha256": receipt["reset_row_set_sha256"],
                "reset_identities": receipt["reset_identities"],
                "reset_count": receipt["reset_count"],
                "evaluation_relative_path": receipt["evaluation_relative_path"],
                "evaluation_sha256": receipt["evaluation_sha256"],
                "evaluation_payload_sha256": receipt["evaluation_payload_sha256"],
                "quality_v2_schema_version": receipt["quality_v2_schema_version"],
                "quality_v2_record_count": receipt["quality_v2_record_count"],
                "task_config_sha256": receipt["task_config_sha256"],
                "task_quality_schema_version": receipt["task_quality_schema_version"],
                "task_quality_schema_sha256": receipt["task_quality_schema_sha256"],
                "all_replays_passed": receipt["all_replays_passed"],
                "planner_calibration": receipt["planner_calibration"],
            }
        )
        comparison_rows.append(
            {
                "task_id": task,
                "metric_calibration_reset_identity_set_sha256": contract[
                    "reset_identity_set_sha256"
                ],
                "metric_calibration_reset_row_set_sha256": contract[
                    "reset_row_set_sha256"
                ],
                "validation_reset_row_overlap_count": 0,
                "review_reset_row_overlap_count": 0,
                "validation_reset_identity_overlap_count": 0,
                "review_reset_identity_overlap_count": 0,
                "validation": contract["disjointness"]["comparators"]["validation"],
                "review": contract["disjointness"]["comparators"]["review"],
            }
        )
    return {
        "schema_version": WAVE_RECEIPT_SCHEMA,
        "wave_id": wave["wave_id"],
        "wave_contract_sha256": wave_sha256,
        "predeclaration_receipt_sha256": predeclaration_sha256,
        "scientific_partition": SCIENTIFIC_PARTITION,
        "transport_split": TRANSPORT_SPLIT,
        "manifest_seed": wave["manifest_seed"],
        "task_count": len(TASK_ORDER),
        "episodes_per_task": EPISODES_PER_TASK,
        "total_reset_count": len(TASK_ORDER) * EPISODES_PER_TASK,
        "total_fresh_environment_replay_count": (
            len(TASK_ORDER) * PLANNER_CALIBRATION_REPLAY_COUNT
        ),
        "task_order": list(TASK_ORDER),
        "planner_dominance_calibration": wave["planner_dominance_calibration"],
        "source_identity": wave["source_identity"],
        "disjointness": {
            "partitions": wave["partitions"],
            "sealed_test_policy": wave["sealed_test_policy"],
            "task_comparisons": comparison_rows,
        },
        "tasks": task_rows,
    }


def _task_lock(wave_root: Path, task: str) -> Any:
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - the formal runtime is Linux.
        raise ContractError("task execution requires Linux advisory locks") from error
    lock_path = wave_root / ".locks" / f"{task}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise ContractError(f"another launcher owns task {task}") from error
    return stream


def _existing_completed_attempt(task_root: Path) -> Path | None:
    completed = [
        attempt
        for attempt in _attempt_roots(task_root)
        if _completed_evaluator_root(attempt) is not None
    ]
    if len(completed) > 1:
        raise ContractError(
            "multiple complete evaluator attempts exist; refusing ambiguity"
        )
    return completed[0] if completed else None


def _next_attempt_root(task_root: Path) -> Path:
    attempts = _attempt_roots(task_root)
    for attempt in attempts:
        if (
            _completed_evaluator_root(attempt) is None
            and not (attempt / "attempt_status.json").is_file()
        ):
            raise ContractError(
                f"ambiguous interrupted attempt {attempt}; wait for it or record its terminal status"
            )
    next_number = (
        1 if not attempts else int(attempts[-1].name.removeprefix("attempt-")) + 1
    )
    path = task_root / "attempts" / f"attempt-{next_number:06d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def cmd_list_tasks(args: argparse.Namespace) -> None:
    if args.format == "json":
        print(_canonical_json(list(TASK_ORDER)))
    else:
        print("\n".join(TASK_ORDER))


def cmd_init_wave(args: argparse.Namespace) -> None:
    if args.manifest_seed < 0:
        raise ContractError("manifest seed must be nonnegative")
    wave = _wave_contract(args)
    wave_root = args.wave_root.resolve()
    wave_root.mkdir(parents=True, exist_ok=True)
    contract_path = wave_root / "wave_contract.json"
    _write_exact_json(contract_path, wave)
    digest = _sha256(contract_path)
    _write_exact_sha256sums(
        wave_root / "WAVE_CONTRACT.sha256", ((digest, "wave_contract.json"),)
    )
    _load_wave(wave_root)
    print(
        _canonical_json({"wave_contract_sha256": digest, "wave_root": str(wave_root)})
    )


def cmd_predeclare_task(args: argparse.Namespace) -> None:
    wave_root = args.wave_root.resolve()
    wave, wave_sha256 = _load_wave(wave_root)
    _verify_source_files(
        wave,
        args.source_root.resolve(),
        args.benchmark_source_root.resolve(),
    )
    contract, manifest_bytes = _task_contract(
        wave,
        wave_sha256,
        args.task,
        args.benchmark_source_root.resolve(),
    )
    task_root = wave_root / "tasks" / args.task
    manifest_path = task_root / "reset_manifest.jsonl"
    contract_path = task_root / "contract.json"
    _write_exact_bytes(manifest_path, manifest_bytes)
    _write_exact_json(contract_path, contract)
    contract_sha256 = _sha256(contract_path)
    _write_exact_sha256sums(
        task_root / "PREDECLARATION_SHA256SUMS",
        (
            (contract_sha256, "contract.json"),
            (contract["reset_manifest_sha256"], "reset_manifest.jsonl"),
        ),
    )
    _load_task_contract(wave_root, wave, wave_sha256, args.task)
    print(
        _canonical_json(
            {
                "task_id": args.task,
                "task_contract_sha256": contract_sha256,
                "reset_manifest_sha256": contract["reset_manifest_sha256"],
                "reset_count": contract["reset_count"],
            }
        )
    )


def cmd_seal_wave(args: argparse.Namespace) -> None:
    wave_root = args.wave_root.resolve()
    wave, wave_sha256 = _load_wave(wave_root)
    _verify_source_files(
        wave,
        args.source_root.resolve(),
        args.benchmark_source_root.resolve(),
    )
    receipt = _predeclaration_receipt(wave_root, wave, wave_sha256)
    receipt_path = wave_root / "predeclaration_receipt.json"
    _write_exact_json(receipt_path, receipt)
    receipt_sha256 = _sha256(receipt_path)
    rows = [(wave_sha256, "wave_contract.json")]
    for item in receipt["tasks"]:
        task = item["task_id"]
        rows.extend(
            (
                (item["task_contract_sha256"], f"tasks/{task}/contract.json"),
                (item["reset_manifest_sha256"], f"tasks/{task}/reset_manifest.jsonl"),
            )
        )
    rows.append((receipt_sha256, "predeclaration_receipt.json"))
    _write_exact_sha256sums(wave_root / "PREDECLARATION_SHA256SUMS", rows)
    _validate_predeclaration_seal(wave_root, wave, wave_sha256)
    print(
        _canonical_json(
            {
                "predeclaration_receipt_sha256": receipt_sha256,
                "task_count": len(TASK_ORDER),
                "total_reset_count": len(TASK_ORDER) * EPISODES_PER_TASK,
            }
        )
    )


def cmd_verify_task_runtime(args: argparse.Namespace) -> None:
    wave_root = args.wave_root.resolve()
    wave, wave_sha256 = _load_wave(wave_root)
    _validate_predeclaration_seal(wave_root, wave, wave_sha256)
    contract, contract_sha256 = _verify_runtime_task(
        wave_root,
        wave,
        wave_sha256,
        args.task,
        args.source_root.resolve(),
        args.benchmark_source_root.resolve(),
    )
    print(
        _canonical_json(
            {
                "task_id": args.task,
                "task_contract_sha256": contract_sha256,
                "reset_manifest_sha256": contract["reset_manifest_sha256"],
                "runtime_verified": True,
            }
        )
    )


def cmd_run_task(args: argparse.Namespace) -> None:
    wave_root = args.wave_root.resolve()
    task = args.task
    lock = _task_lock(wave_root, task)
    try:
        wave, wave_sha256 = _load_wave(wave_root)
        _verify_source_files(
            wave,
            args.source_root.resolve(),
            args.benchmark_source_root.resolve(),
        )
        _validate_predeclaration_seal(wave_root, wave, wave_sha256)
        try:
            receipt, receipt_sha256 = _load_task_receipt(
                wave_root, wave, wave_sha256, task
            )
        except ContractError:
            if (wave_root / "tasks" / task / "receipt.json").exists():
                raise
        else:
            print(
                _canonical_json(
                    {
                        "status": "complete",
                        "task_id": task,
                        "task_receipt_sha256": receipt_sha256,
                        "evaluation_sha256": receipt["evaluation_sha256"],
                    }
                )
            )
            return
        contract, contract_sha256 = _verify_runtime_task(
            wave_root,
            wave,
            wave_sha256,
            task,
            args.source_root.resolve(),
            args.benchmark_source_root.resolve(),
        )
        task_root = wave_root / "tasks" / task
        completed_attempt = _existing_completed_attempt(task_root)
        if completed_attempt is not None:
            receipt = _finalize_task(
                wave_root,
                wave,
                wave_sha256,
                task,
                completed_attempt,
            )
            print(
                _canonical_json(
                    {
                        "status": "finalized_existing_attempt",
                        "task_id": task,
                        "evaluation_sha256": receipt["evaluation_sha256"],
                    }
                )
            )
            return
        attempt_root = _next_attempt_root(task_root)
        evaluator_path = args.source_root.resolve() / SOURCE_FILES["planner_evaluator"]
        evaluator_root = attempt_root / "evaluator"
        command = [
            sys.executable,
            str(evaluator_path),
            "--evaluator-commit",
            wave["source_identity"]["evaluator_rlinf_commit"],
            "--benchmark-commit",
            wave["source_identity"]["benchmark_commit"],
            "--output",
            str(evaluator_root),
            "--task",
            task,
            "--split",
            TRANSPORT_SPLIT,
            "--manifest-seed",
            str(wave["manifest_seed"]),
            "--episodes",
            str(EPISODES_PER_TASK),
            "--image-size",
            str(args.image_size),
        ]
        attempt_contract = {
            "schema_version": ATTEMPT_CONTRACT_SCHEMA,
            "wave_contract_sha256": wave_sha256,
            "task_contract_sha256": contract_sha256,
            "task_id": task,
            "scientific_partition": SCIENTIFIC_PARTITION,
            "transport_split": TRANSPORT_SPLIT,
            "manifest_seed": wave["manifest_seed"],
            "reset_manifest_sha256": contract["reset_manifest_sha256"],
            "episodes": EPISODES_PER_TASK,
            "image_size": args.image_size,
            "planner_dominance_calibration": wave["planner_dominance_calibration"],
            "evaluator_relative_path": SOURCE_FILES["planner_evaluator"],
            "evaluation_output_relative_path": (
                evaluator_root.relative_to(wave_root).as_posix()
            ),
        }
        _write_exact_json(attempt_root / "attempt_contract.json", attempt_contract)
        log_path = attempt_root / "evaluator.log"
        with log_path.open("xb") as log_stream:
            result = subprocess.run(
                command,
                cwd=args.source_root.resolve(),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_stream.flush()
            os.fsync(log_stream.fileno())
        status = {
            "schema_version": ATTEMPT_STATUS_SCHEMA,
            "task_id": task,
            "returncode": result.returncode,
            "evaluator_log_sha256": _sha256(log_path),
            "complete_evaluator_artifacts_present": (
                _completed_evaluator_root(attempt_root) is not None
            ),
        }
        _write_exact_json(attempt_root / "attempt_status.json", status)
        if result.returncode != 0:
            raise ContractError(
                f"{task} evaluator exited {result.returncode}; preserved {attempt_root}"
            )
        receipt = _finalize_task(
            wave_root,
            wave,
            wave_sha256,
            task,
            attempt_root,
        )
        print(
            _canonical_json(
                {
                    "status": "complete",
                    "task_id": task,
                    "evaluation_sha256": receipt["evaluation_sha256"],
                }
            )
        )
    finally:
        lock.close()


def cmd_finalize_wave(args: argparse.Namespace) -> None:
    wave_root = args.wave_root.resolve()
    wave, wave_sha256 = _load_wave(wave_root)
    _verify_source_files(
        wave,
        args.source_root.resolve(),
        args.benchmark_source_root.resolve(),
    )
    _, predeclaration_sha256 = _validate_predeclaration_seal(
        wave_root, wave, wave_sha256
    )
    receipt = _wave_receipt(
        wave_root,
        wave,
        wave_sha256,
        predeclaration_sha256,
    )
    receipt_path = wave_root / "wave_receipt.json"
    _write_exact_json(receipt_path, receipt)
    receipt_sha256 = _sha256(receipt_path)
    if receipt_sha256 != _payload_sha256(receipt):
        raise ContractError("wave receipt serialization is not canonical")
    _write_exact_sha256sums(
        wave_root / "WAVE_RECEIPT.sha256",
        ((receipt_sha256, "wave_receipt.json"),),
    )
    rows: list[tuple[str, str]] = [
        (wave_sha256, "wave_contract.json"),
        (predeclaration_sha256, "predeclaration_receipt.json"),
    ]
    for item in receipt["tasks"]:
        task = item["task_id"]
        rows.extend(
            (
                (item["task_contract_sha256"], f"tasks/{task}/contract.json"),
                (item["task_receipt_sha256"], f"tasks/{task}/receipt.json"),
                (
                    item["reset_manifest_sha256"],
                    item["reset_manifest_relative_path"],
                ),
                (item["evaluation_sha256"], item["evaluation_relative_path"]),
                *_planner_calibration_sha256_rows(
                    item["planner_calibration"], task_relative=False
                ),
            )
        )
    rows.append((receipt_sha256, "wave_receipt.json"))
    _write_exact_sha256sums(wave_root / "SHA256SUMS", rows)
    print(
        _canonical_json(
            {
                "status": "complete",
                "wave_receipt": str(receipt_path),
                "wave_receipt_sha256": receipt_sha256,
                "task_count": len(TASK_ORDER),
                "total_reset_count": len(TASK_ORDER) * EPISODES_PER_TASK,
            }
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_tasks = subparsers.add_parser("list-tasks")
    list_tasks.add_argument("--format", choices=("lines", "json"), default="lines")
    list_tasks.set_defaults(func=cmd_list_tasks)

    init = subparsers.add_parser("init-wave")
    init.add_argument("--wave-root", type=Path, required=True)
    init.add_argument("--wave-id", required=True)
    init.add_argument("--lane-prefix", required=True)
    init.add_argument("--gpus", required=True)
    init.add_argument("--source-root", type=Path, required=True)
    init.add_argument("--benchmark-source-root", type=Path, required=True)
    init.add_argument("--benchmark-source-container-root", required=True)
    init.add_argument("--image-ref", required=True)
    init.add_argument("--image-id", required=True)
    init.add_argument("--evaluator-commit", required=True)
    init.add_argument("--benchmark-commit", required=True)
    init.add_argument("--manifest-seed", type=int, default=DEFAULT_MANIFEST_SEED)
    init.add_argument("--validation-seed", type=int, default=DEFAULT_VALIDATION_SEED)
    init.add_argument("--review-seed", type=int, default=DEFAULT_REVIEW_SEED)
    init.add_argument("--test-id-seed", type=int, default=DEFAULT_TEST_ID_SEED)
    init.add_argument("--test-ood-seed", type=int, default=DEFAULT_TEST_OOD_SEED)
    init.set_defaults(func=cmd_init_wave)

    predeclare = subparsers.add_parser("predeclare-task")
    predeclare.add_argument("--wave-root", type=Path, required=True)
    predeclare.add_argument("--task", choices=TASK_ORDER, required=True)
    predeclare.add_argument("--source-root", type=Path, required=True)
    predeclare.add_argument("--benchmark-source-root", type=Path, required=True)
    predeclare.set_defaults(func=cmd_predeclare_task)

    seal = subparsers.add_parser("seal-wave")
    seal.add_argument("--wave-root", type=Path, required=True)
    seal.add_argument("--source-root", type=Path, required=True)
    seal.add_argument("--benchmark-source-root", type=Path, required=True)
    seal.set_defaults(func=cmd_seal_wave)

    verify = subparsers.add_parser("verify-task-runtime")
    verify.add_argument("--wave-root", type=Path, required=True)
    verify.add_argument("--task", choices=TASK_ORDER, required=True)
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--benchmark-source-root", type=Path, required=True)
    verify.set_defaults(func=cmd_verify_task_runtime)

    run = subparsers.add_parser("run-task")
    run.add_argument("--wave-root", type=Path, required=True)
    run.add_argument("--task", choices=TASK_ORDER, required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--benchmark-source-root", type=Path, required=True)
    run.add_argument("--image-size", type=int, default=64)
    run.set_defaults(func=cmd_run_task)

    finalize = subparsers.add_parser("finalize-wave")
    finalize.add_argument("--wave-root", type=Path, required=True)
    finalize.add_argument("--source-root", type=Path, required=True)
    finalize.add_argument("--benchmark-source-root", type=Path, required=True)
    finalize.set_defaults(func=cmd_finalize_wave)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "image_size", 64) != 64:
        raise ContractError("formal planner calibration image size must be exactly 64")
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except ContractError as error:
        print(f"planner calibration contract error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
