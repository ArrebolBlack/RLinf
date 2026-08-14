# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed identity contracts for GPUPLAN0 ``t1_xyz`` evidence.

The helpers in this module are deliberately host-only.  They validate a sealed
manifest, every export, the complete ``ResetRequest``, and the checked-out
source tuple before a result runner is allowed to construct a CUDA backend.
They do not reserve resources or create an environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_ID = "t1_xyz"
BACKEND_ID = "mjwarp_gpu_v1"
QUALITY_SCHEMA_VERSION = "db0-episode-task-quality-v2"
QUALITY_EVALUATOR_ID = BACKEND_ID
MANIFEST_SCHEMA_VERSION = "gpuplan0-t1-xyz-review-evidence-manifest-v3"
RESULT_SCHEMA_VERSION = "gpu-planner-t1-xyz-review-evidence-result-v3"
D32_SUMMARY_SCHEMA_VERSION = "gpu-planner-t1-xyz-d32-strict-summary-v2"

CANONICAL_REQUEST_FIELDS = (
    "action_mode",
    "api_version",
    "episode_id",
    "factors",
    "object_mode",
    "observation_track",
    "reset_mode",
    "seed",
    "split",
    "task_id",
)
SOURCE_REPOSITORIES = frozenset(
    {
        "dynamic_benchmark",
        "mujoco_warp",
        "research",
        "rlinf",
        "se3_wam",
    }
)
FAILURE_TERMINATION_REASONS = (
    "unsafe_contact",
    "workspace_exit",
    "downstream_exit",
    "drop",
    "driver_blocked",
    "timeout",
    "invalid_state",
)
EXECUTION_CONTRACT = {
    "action_mode": "E7",
    "backend_id": BACKEND_ID,
    "control_hz": 20,
    "fresh_replay_gate": "semantic_identity_outcome_quality_v1",
    "horizon_control_steps": 160,
    "observation_track": "state",
    "physics_hz": 500,
    "physics_steps_per_control": 25,
    "planner_observation_source": "current_observation_each_control_step",
    "quality_evaluator_id": QUALITY_EVALUATOR_ID,
    "quality_schema_version": QUALITY_SCHEMA_VERSION,
    "replay_intermediate_event_drift_blocking": False,
    "review_materialization": "independent_scene_wrist_render_v1",
    "replay_numeric_drift_blocking": False,
    "sensor_hz": 20,
    "terminal_ledger_exact_once": True,
}


@dataclass(frozen=True)
class T1XYZFrozenManifest:
    """One completely validated E0 or D32 frozen manifest."""

    path: Path
    payload: Mapping[str, Any]
    manifest_sha256: str
    source_identity_sha256: str
    rows: tuple[Mapping[str, Any], ...]

    @property
    def phase(self) -> str:
        return str(self.payload["phase"])

    def row(self, index: int) -> Mapping[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("manifest row index must be an integer")
        if not 0 <= index < len(self.rows):
            raise IndexError(f"manifest row index must be in [0, {len(self.rows)})")
        row = self.rows[index]
        if row["manifest_index"] != index:
            raise RuntimeError("validated manifest row order changed")
        return row

    def export_dir(self, row: Mapping[str, Any]) -> Path:
        path = Path(str(row["export_dir"]))
        if not path.is_absolute():
            path = (self.path.parent / path).resolve()
        return path


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON encoding used by this contract."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_git_oid(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase full Git object ID")
    return value


def _validate_repositories(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SOURCE_REPOSITORIES:
        raise ValueError("manifest must freeze the exact five-repository source tuple")
    for name in SOURCE_REPOSITORIES:
        identity = value[name]
        if not isinstance(identity, Mapping):
            raise ValueError(f"repositories.{name} must be an object")
        _require_git_oid(identity.get("commit"), f"repositories.{name}.commit")
        _require_git_oid(identity.get("tree"), f"repositories.{name}.tree")
    se3 = value["se3_wam"]
    dynamic = value["dynamic_benchmark"]
    _require_git_oid(
        se3.get("mujoco_warp_gitlink"),
        "repositories.se3_wam.mujoco_warp_gitlink",
    )
    _require_git_oid(
        dynamic.get("rlinf_gitlink"),
        "repositories.dynamic_benchmark.rlinf_gitlink",
    )
    if se3["mujoco_warp_gitlink"] != value["mujoco_warp"]["commit"]:
        raise ValueError("SE3-WAM MJWarp gitlink differs from the frozen commit")
    if dynamic["rlinf_gitlink"] != value["rlinf"]["commit"]:
        raise ValueError(
            "Dynamic Benchmark RLinf gitlink differs from the frozen commit"
        )
    return value


def _validate_request_payload(value: Any, index: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(CANONICAL_REQUEST_FIELDS):
        raise ValueError(
            f"manifest row {index} lacks the complete canonical ResetRequest"
        )
    expected = {
        "action_mode": "E7",
        "observation_track": "state",
        "task_id": TASK_ID,
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(
                f"manifest row {index} request {name} differs from {required!r}"
            )
    for name in ("api_version", "episode_id", "object_mode", "reset_mode", "split"):
        field = value.get(name)
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"manifest row {index} request {name} is invalid")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"manifest row {index} request seed is invalid")
    factors = value.get("factors")
    if not isinstance(factors, Mapping):
        raise ValueError(f"manifest row {index} request factors must be an object")
    canonical_json_bytes(factors)
    return value


def _validate_row(value: Any, index: int, source_sha256: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest row {index} must be an object")
    if value.get("manifest_index") != index:
        raise ValueError(f"manifest row {index} has a different manifest_index")
    candidate_index = value.get("candidate_index")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise ValueError(f"manifest row {index} candidate_index must be an integer")
    source_group_id = value.get("source_group_id")
    if not isinstance(source_group_id, str) or not source_group_id.strip():
        raise ValueError(f"manifest row {index} source_group_id is invalid")
    for name in ("pair_id", "pair_member_id"):
        if value.get(name) is not None and not isinstance(value.get(name), str):
            raise ValueError(f"manifest row {index} {name} must be a string or null")
    export_dir = value.get("export_dir")
    if not isinstance(export_dir, str) or not export_dir.strip():
        raise ValueError(f"manifest row {index} export_dir is invalid")
    for name in (
        "export_report_sha256",
        "request_json_sha256",
        "sha256sums_sha256",
        "task_config_sha256",
    ):
        _require_sha256(value.get(name), f"rows[{index}].{name}")
    if value.get("source_identity_sha256") != source_sha256:
        raise ValueError(f"manifest row {index} is not bound to the source tuple")
    _validate_request_payload(value.get("request"), index)
    return value


def load_frozen_manifest(
    path: Path,
    *,
    expected_phase: str | None = None,
    verify_exports: bool = True,
) -> T1XYZFrozenManifest:
    """Validate every row before a caller selects an execution row."""

    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("t1_xyz manifest must be a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("t1_xyz manifest schema is not the review evidence contract")
    if payload.get("task_id") != TASK_ID or payload.get("backend_id") != BACKEND_ID:
        raise ValueError("t1_xyz manifest task/backend identity mismatch")
    phase = payload.get("phase")
    if phase not in {"e0", "d32"} or (
        expected_phase is not None and phase != expected_phase
    ):
        raise ValueError(
            "t1_xyz manifest phase does not match the requested result path"
        )
    expected_count = 1 if phase == "e0" else 32
    expected_cohort_size = 1 if phase == "e0" else 8
    if (
        payload.get("episode_count") != expected_count
        or payload.get("cohort_size") != expected_cohort_size
        or payload.get("cohort_count") != expected_count // expected_cohort_size
    ):
        raise ValueError("t1_xyz manifest episode/cohort cardinality is inconsistent")
    if payload.get("execution") != EXECUTION_CONTRACT:
        raise ValueError("t1_xyz manifest execution/quality/replay contract drifted")
    repositories = _validate_repositories(payload.get("repositories"))
    source_sha256 = payload_sha256(repositories)
    if payload.get("source_identity_sha256") != source_sha256:
        raise ValueError("manifest source_identity_sha256 does not match repositories")
    task_config_sha256 = _require_sha256(
        payload.get("task_config_sha256"), "task_config_sha256"
    )
    rows_value = payload.get("rows")
    if not isinstance(rows_value, list) or len(rows_value) != expected_count:
        raise ValueError(
            f"t1_xyz {phase} manifest must contain exactly {expected_count} rows"
        )
    rows = tuple(
        _validate_row(row, index, source_sha256) for index, row in enumerate(rows_value)
    )
    episode_ids = [str(row["request"]["episode_id"]) for row in rows]
    seeds = [int(row["request"]["seed"]) for row in rows]
    export_dirs = [str(row["export_dir"]) for row in rows]
    candidate_indices = [int(row["candidate_index"]) for row in rows]
    if len(set(episode_ids)) != expected_count:
        raise ValueError("t1_xyz manifest episode IDs are not globally unique")
    if len(set(seeds)) != expected_count:
        raise ValueError("t1_xyz manifest seeds are not globally unique")
    if len(set(export_dirs)) != expected_count:
        raise ValueError("t1_xyz manifest does not bind one distinct export per row")
    if len(set(candidate_indices)) != expected_count:
        raise ValueError("t1_xyz manifest candidate indices are not globally unique")
    if any(row["task_config_sha256"] != task_config_sha256 for row in rows):
        raise ValueError("t1_xyz manifest rows do not share the frozen task config")
    if payload.get("episode_ids") != episode_ids:
        raise ValueError("t1_xyz manifest episode_ids drifted from the rows")
    if payload.get("candidate_indices") != candidate_indices:
        raise ValueError("t1_xyz manifest candidate_indices drifted from the rows")
    declared_manifest_sha = _require_sha256(
        payload.get("manifest_sha256"), "manifest_sha256"
    )
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if declared_manifest_sha != payload_sha256(unsigned):
        raise ValueError("manifest_sha256 does not match the frozen manifest body")
    manifest = T1XYZFrozenManifest(
        path=resolved,
        payload=payload,
        manifest_sha256=declared_manifest_sha,
        source_identity_sha256=source_sha256,
        rows=rows,
    )
    if verify_exports:
        for row in rows:
            validate_export_files(manifest, row)
    return manifest


def _parse_sha256sums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        _require_sha256(digest, f"SHA256SUMS entry {relative or line}")
        if not separator or not relative or relative in rows:
            raise ValueError(
                "export SHA256SUMS contains a malformed or duplicate entry"
            )
        target = (path.parent / relative).resolve()
        try:
            target.relative_to(path.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                "export SHA256SUMS entry escapes the export directory"
            ) from exc
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"export SHA256SUMS verification failed for {relative}")
        rows[relative] = digest
    if not rows or not {"request.json", "export_report.json"}.issubset(rows):
        raise ValueError("export SHA256SUMS lacks request/export-report coverage")
    return rows


def _expected_request_json(row: Mapping[str, Any]) -> dict[str, Any]:
    request = row["request"]
    return {
        "action_mode": request["action_mode"],
        "candidate_index": row["candidate_index"],
        "episode_id": request["episode_id"],
        "factors": dict(request["factors"]),
        "object_mode": request["object_mode"],
        "observation_track": request["observation_track"],
        "pair_id": row["pair_id"],
        "pair_member_id": row["pair_member_id"],
        "reset_mode": request["reset_mode"],
        "seed": request["seed"],
        "source_group_id": row["source_group_id"],
        "split": request["split"],
        "task_id": request["task_id"],
    }


def validate_export_files(
    manifest: T1XYZFrozenManifest,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate hashes and JSON identities without constructing a GPU backend."""

    export_dir = manifest.export_dir(row)
    if not export_dir.is_dir():
        raise FileNotFoundError(export_dir)
    sums_path = export_dir / "SHA256SUMS"
    request_path = export_dir / "request.json"
    report_path = export_dir / "export_report.json"
    for path in (sums_path, request_path, report_path):
        if not path.is_file():
            raise ValueError(f"frozen export is incomplete: {path}")
    expected_digests = {
        sums_path: row["sha256sums_sha256"],
        request_path: row["request_json_sha256"],
        report_path: row["export_report_sha256"],
    }
    for path, expected in expected_digests.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen export digest drifted: {path.name}")
    sums = _parse_sha256sums(sums_path)
    if sums["request.json"] != row["request_json_sha256"]:
        raise ValueError(
            "SHA256SUMS request.json identity differs from the manifest row"
        )
    if sums["export_report.json"] != row["export_report_sha256"]:
        raise ValueError(
            "SHA256SUMS export_report identity differs from the manifest row"
        )
    request_json = json.loads(request_path.read_text(encoding="utf-8"))
    if request_json != _expected_request_json(row):
        raise ValueError("export request.json differs from the frozen manifest row")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("export_report.json must be an object")
    request = row["request"]
    if (
        report.get("task_id") != TASK_ID
        or report.get("episode_id") != request["episode_id"]
        or report.get("row_index") != row["candidate_index"]
        or report.get("task_config_sha256") != row["task_config_sha256"]
    ):
        raise ValueError(
            "export report identity/configuration differs from the manifest row"
        )
    return {
        "export_dir": export_dir,
        "export_report": dict(report),
        "request_json": request_json,
        "sha256sums_entries": sums,
    }


def request_identity(request: Any) -> dict[str, Any]:
    """Materialize all canonical ``ResetRequest`` identity fields."""

    def enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    return {
        "action_mode": enum_value(request.action_mode),
        "api_version": request.api_version,
        "episode_id": request.episode_id,
        "factors": dict(request.factors),
        "object_mode": request.object_mode,
        "observation_track": enum_value(request.observation_track),
        "reset_mode": request.reset_mode,
        "seed": int(request.seed),
        "split": enum_value(request.split),
        "task_id": request.task_id,
    }


def preflight_export_request(
    manifest: T1XYZFrozenManifest,
    row: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Load the host artifact and prove its full request identity before physics."""

    export = validate_export_files(manifest, row)
    from se3_wam.benchmark.gpu_native.p0_grasp_engine import load_p0_grasp_artifacts

    artifacts = load_p0_grasp_artifacts(export["export_dir"])
    actual = request_identity(artifacts.reset_request)
    expected = dict(row["request"])
    if actual != expected:
        raise RuntimeError(
            f"export-bound canonical ResetRequest differs from manifest: {actual} != {expected}"
        )
    if artifacts.config_sha256 != row["task_config_sha256"]:
        raise RuntimeError("export artifact task config identity differs from manifest")
    return artifacts.reset_request, {
        "export_dir": str(export["export_dir"]),
        "export_report_sha256": row["export_report_sha256"],
        "request_json_sha256": row["request_json_sha256"],
        "sha256sums_sha256": row["sha256sums_sha256"],
        "task_config_sha256": artifacts.config_sha256,
    }


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _validate_checkout(
    root: Path,
    expected: Mapping[str, Any],
    *,
    gitlink: tuple[str, str] | None = None,
) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    actual = {
        "commit": _git(resolved, "rev-parse", "HEAD"),
        "tree": _git(resolved, "rev-parse", "HEAD^{tree}"),
    }
    if actual != {"commit": expected["commit"], "tree": expected["tree"]}:
        raise RuntimeError(f"source checkout identity mismatch: {resolved}")
    if _git(resolved, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError(f"source checkout is dirty: {resolved}")
    if gitlink is not None:
        relative, field = gitlink
        values = _git(resolved, "ls-tree", "HEAD", "--", relative).split()
        if (
            len(values) < 3
            or values[0] != "160000"
            or values[1] != "commit"
            or values[2] != expected[field]
        ):
            raise RuntimeError(
                f"source checkout gitlink mismatch: {resolved}/{relative}"
            )
        actual[field] = values[2]
    return {"path": str(resolved), **actual}


def validate_repository_tuple(
    manifest: T1XYZFrozenManifest,
    *,
    research_root: Path,
    se3_root: Path,
    mjwarp_root: Path,
    rlinf_root: Path,
    dynamic_root: Path,
) -> dict[str, Any]:
    """Require the exact clean source tuple and both submodule gitlinks."""

    expected = manifest.payload["repositories"]
    actual = {
        "research": _validate_checkout(research_root, expected["research"]),
        "se3_wam": _validate_checkout(
            se3_root,
            expected["se3_wam"],
            gitlink=("third_party/mujoco_warp", "mujoco_warp_gitlink"),
        ),
        "mujoco_warp": _validate_checkout(mjwarp_root, expected["mujoco_warp"]),
        "rlinf": _validate_checkout(rlinf_root, expected["rlinf"]),
        "dynamic_benchmark": _validate_checkout(
            dynamic_root,
            expected["dynamic_benchmark"],
            gitlink=("third_party/RLinf", "rlinf_gitlink"),
        ),
    }
    if actual["se3_wam"]["mujoco_warp_gitlink"] != actual["mujoco_warp"]["commit"]:
        raise RuntimeError("runtime SE3-WAM gitlink differs from checked-out MJWarp")
    if actual["dynamic_benchmark"]["rlinf_gitlink"] != actual["rlinf"]["commit"]:
        raise RuntimeError(
            "runtime Dynamic Benchmark gitlink differs from checked-out RLinf"
        )
    return actual


def validate_result_for_row(
    result: Mapping[str, Any],
    *,
    manifest: T1XYZFrozenManifest,
    row: Mapping[str, Any],
) -> None:
    """Reject any row that did not pass every required review evidence gate."""

    binding = result.get("manifest")
    replay = result.get("replay")
    quality = result.get("quality")
    terminal_gate = result.get("terminal_ledger_gate")
    source_gate = result.get("source_gate")
    evidence_export = result.get("evidence_export")
    provenance = result.get("provenance")
    required_replay = (
        "fresh_backend_distinct",
        "backend_identity_exact",
        "source_identity_exact",
        "reset_identity_exact",
        "semantic_action_identity_exact",
        "observation_semantic_structure_exact",
        "review_semantic_structure_exact",
        "semantic_outcomes_exact",
        "terminal_ledger_semantic_exact",
        "terminal_ledger_exact_once",
    )
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "completed_review_evidence"
        or result.get("evidence_passed") is not True
        or result.get("task_id") != TASK_ID
        or result.get("backend_id") != BACKEND_ID
        or result.get("phase") != manifest.phase
        or result.get("manifest_index") != row["manifest_index"]
        or result.get("online_planner") is not True
        or result.get("frozen_action_replay") is not False
        or result.get("cpu_physics_or_env_fallback") is not False
        or result.get("planner_observation_source")
        != EXECUTION_CONTRACT["planner_observation_source"]
        or result.get("planner_observation_track") != "state"
        or result.get("review_materialization")
        != EXECUTION_CONTRACT["review_materialization"]
    ):
        raise RuntimeError("row result is not completed t1_xyz review evidence")
    if not isinstance(binding, Mapping) or binding != {
        "candidate_index": row["candidate_index"],
        "episode_id": row["request"]["episode_id"],
        "manifest_index": row["manifest_index"],
        "manifest_sha256": manifest.manifest_sha256,
        "source_identity_sha256": manifest.source_identity_sha256,
    }:
        raise RuntimeError("row result manifest/source identity mismatch")
    if result.get("reset_request") != row["request"]:
        raise RuntimeError("row result canonical ResetRequest identity mismatch")
    if (
        not isinstance(source_gate, Mapping)
        or source_gate.get("passed") is not True
        or source_gate.get("repositories_exact") is not True
        or source_gate.get("source_identity_sha256") != manifest.source_identity_sha256
    ):
        raise RuntimeError("row result source checkout gate did not pass")
    source_repositories = source_gate.get("repositories")
    expected_repositories = manifest.payload["repositories"]
    if not isinstance(source_repositories, Mapping) or set(source_repositories) != set(
        expected_repositories
    ):
        raise RuntimeError("row result lacks the exact five-repository source tuple")
    for name, expected in expected_repositories.items():
        actual = source_repositories.get(name)
        if not isinstance(actual, Mapping) or any(
            actual.get(field) != value for field, value in expected.items()
        ):
            raise RuntimeError(f"row result source identity drifted for {name}")
    se3_identity = expected_repositories["se3_wam"]
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("backend_id") != BACKEND_ID
        or provenance.get("device_platform") not in {"cuda", "gpu"}
        or not isinstance(provenance.get("physical_device_uuid"), str)
        or not provenance["physical_device_uuid"].strip()
        or provenance.get("git_commit") != se3_identity["commit"]
        or provenance.get("git_tree") != se3_identity["tree"]
    ):
        raise RuntimeError("row result CUDA/backend/source provenance drifted")
    if quality != {
        "evaluator_backend_id": QUALITY_EVALUATOR_ID,
        "schema_version": QUALITY_SCHEMA_VERSION,
    }:
        raise RuntimeError("row result quality identity mismatch")
    if (
        not isinstance(terminal_gate, Mapping)
        or terminal_gate.get("passed") is not True
        or terminal_gate.get("exact_once_second_consumption_rejected") is not True
    ):
        raise RuntimeError("row result terminal ledger gate did not pass")
    if (
        not isinstance(replay, Mapping)
        or replay.get("mode") != "semantic_fresh_backend_v1"
        or replay.get("passed") is not True
        or any(replay.get(name) is not True for name in required_replay)
        or replay.get("primary_provenance") != provenance
        or replay.get("replay_provenance") != provenance
        or replay.get("first_divergence") is not None
    ):
        raise RuntimeError("row result fresh replay gate did not pass")
    for name in (
        "observation_event_drift",
        "observation_numeric_drift",
        "review_numeric_drift",
        "terminal_numeric_drift",
    ):
        report = replay.get(name)
        if not isinstance(report, Mapping) or report.get("blocking") is not False:
            raise RuntimeError("row result non-blocking replay diagnostics are invalid")
    for name in (
        "observation_event_sequence_exact",
        "observation_tape_exact",
        "review_tape_exact",
        "action_tape_exact",
        "outcomes_exact",
        "terminal_ledger_exact",
    ):
        if type(replay.get(name)) is not bool:
            raise RuntimeError("row result exact replay diagnostics are invalid")
    terminal_grace = replay.get("terminal_grace")
    if (
        not isinstance(terminal_grace, Mapping)
        or terminal_grace.get("schema_version") != "gpu-planner-terminal-grid-grace-v1"
        or terminal_grace.get("mode") != "zero_order_hold_last_primary_action_v1"
        or terminal_grace.get("max_control_steps") != 1
        or type(terminal_grace.get("attempted")) is not bool
        or type(terminal_grace.get("accepted")) is not bool
    ):
        raise RuntimeError("row result terminal-grid grace receipt is invalid")
    terminal_early = replay.get("terminal_early")
    if (
        not isinstance(terminal_early, Mapping)
        or terminal_early.get("schema_version") != "gpu-planner-terminal-grid-early-v1"
        or terminal_early.get("mode") != "natural_success_before_primary_tape_end_v1"
        or terminal_early.get("max_unexecuted_control_steps") != 1
        or type(terminal_early.get("attempted")) is not bool
        or type(terminal_early.get("accepted")) is not bool
    ):
        raise RuntimeError("row result early terminal-grid receipt is invalid")

    grace_unused = bool(
        terminal_grace.get("attempted") is False
        and terminal_grace.get("accepted") is False
        and terminal_grace.get("reason") == "not_required_or_not_admissible"
        and terminal_grace.get("held_action_values_exact") is False
        and terminal_grace.get("control_steps") == 0
        and terminal_grace.get("physics_steps") == 0
        and terminal_grace.get("stage_index_before") is None
        and terminal_grace.get("stage_index_after") is None
        and terminal_grace.get("outcome") is None
        and terminal_grace.get("observation_sha256") is None
        and terminal_grace.get("review_sha256") is None
    )
    early_unused = bool(
        terminal_early.get("attempted") is False
        and terminal_early.get("accepted") is False
        and terminal_early.get("reason") == "not_required_or_not_admissible"
        and terminal_early.get("executed_action_prefix_exact") is False
        and terminal_early.get("executed_control_steps") == 0
        and terminal_early.get("unexecuted_control_steps") == 0
        and terminal_early.get("unexecuted_action_payload_sha256") is None
        and terminal_early.get("stage_index") is None
        and terminal_early.get("outcome") is None
    )
    grace_accepted = bool(
        terminal_grace.get("attempted") is True
        and terminal_grace.get("accepted") is True
        and terminal_grace.get("reason") == "semantic_terminal_reached"
        and terminal_grace.get("held_action_values_exact") is True
        and terminal_grace.get("control_steps") == 1
        and not isinstance(terminal_grace.get("physics_steps"), bool)
        and isinstance(terminal_grace.get("physics_steps"), int)
        and 1
        <= terminal_grace["physics_steps"]
        <= EXECUTION_CONTRACT["physics_steps_per_control"]
        and terminal_grace.get("stage_index_before") == 4
        and terminal_grace.get("stage_index_after") == 5
        and terminal_grace.get("outcome") == [True, False, True, "success"]
        and _is_sha256(terminal_grace.get("observation_sha256"))
        and _is_sha256(terminal_grace.get("review_sha256"))
    )
    replay_stop = replay.get("replay_stop")
    early_executed_steps = terminal_early.get("executed_control_steps")
    early_accepted = bool(
        terminal_early.get("attempted") is True
        and terminal_early.get("accepted") is True
        and terminal_early.get("reason") == "semantic_terminal_reached_one_step_early"
        and terminal_early.get("executed_action_prefix_exact") is True
        and not isinstance(early_executed_steps, bool)
        and isinstance(early_executed_steps, int)
        and early_executed_steps >= 1
        and terminal_early.get("unexecuted_control_steps") == 1
        and _is_sha256(terminal_early.get("unexecuted_action_payload_sha256"))
        and terminal_early.get("stage_index") == 5
        and terminal_early.get("outcome") == [True, False, True, "success"]
        and isinstance(replay_stop, Mapping)
        and replay_stop.get("reason") == "natural_terminal_before_action_tape_end"
        and replay_stop.get("command_index") == early_executed_steps - 1
        and replay_stop.get("policy_step") == early_executed_steps - 1
        and replay_stop.get("submitted_action_count") == early_executed_steps
        and replay_stop.get("expected_action_count") == early_executed_steps + 1
        and replay_stop.get("unexecuted_action_count") == 1
    )
    if replay.get("outcomes_exact") is True:
        if (
            not grace_unused
            or not early_unused
            or replay.get("action_tape_exact") is not True
        ):
            raise RuntimeError("exact replay must not consume terminal-grid handling")
    elif grace_accepted:
        if not early_unused or replay.get("action_tape_exact") is not True:
            raise RuntimeError("late terminal-grid replay receipt is inconsistent")
    elif early_accepted:
        if (
            not grace_unused
            or replay.get("action_tape_exact") is not False
            or replay.get("semantic_action_identity_exact") is not True
        ):
            raise RuntimeError("early terminal-grid replay receipt is inconsistent")
    else:
        raise RuntimeError("non-exact replay lacks bounded terminal-grid evidence")
    ledger = result.get("terminal_ledger")
    if (
        not isinstance(ledger, list)
        or len(ledger) != 1
        or not isinstance(ledger[0], Mapping)
        or ledger[0].get("episode_id") != row["request"]["episode_id"]
        or ledger[0].get("task_id") != TASK_ID
    ):
        raise RuntimeError("row result terminal ledger identity/cardinality mismatch")
    terminal = ledger[0]
    success = result.get("success")
    termination_reason = result.get("termination_reason")
    if type(success) is not bool or not isinstance(termination_reason, str):
        raise RuntimeError("row result lacks exact success/termination fields")
    if (
        terminal.get("success") is not success
        or terminal.get("termination_reason") != termination_reason
    ):
        raise RuntimeError(
            "row result success/termination differs from terminal ledger"
        )
    if success:
        if (
            termination_reason != "success"
            or terminal.get("outcome") != "success"
            or terminal.get("terminated") is not True
            or terminal.get("truncated") is not False
        ):
            raise RuntimeError("successful row terminal outcome is inconsistent")
    elif termination_reason not in FAILURE_TERMINATION_REASONS:
        raise RuntimeError(
            "failed row termination reason is outside the frozen taxonomy"
        )
    elif termination_reason == "timeout":
        if (
            terminal.get("outcome") != "timeout"
            or terminal.get("terminated") is not False
            or terminal.get("truncated") is not True
        ):
            raise RuntimeError("timeout row terminal outcome is inconsistent")
    elif (
        terminal.get("outcome") != "failure"
        or terminal.get("terminated") is not True
        or terminal.get("truncated") is not False
    ):
        raise RuntimeError("failed row terminal outcome is inconsistent")
    terminal_quality = terminal.get("task_quality")
    if success and (
        not isinstance(terminal_quality, Mapping)
        or terminal_quality.get("episode_id") != row["request"]["episode_id"]
        or terminal_quality.get("task_id") != TASK_ID
        or terminal_quality.get("schema_version") != QUALITY_SCHEMA_VERSION
        or terminal_quality.get("evaluator_backend_id") != QUALITY_EVALUATOR_ID
        or terminal_quality.get("terminal") is not True
    ):
        raise RuntimeError("successful row lacks exact quality-v2 terminal evidence")
    if not success and terminal_quality is not None:
        raise RuntimeError("failed row must not carry terminal task quality")
    control_steps = result.get("control_steps")
    physics_steps = result.get("physics_steps")
    action_tape = result.get("action_tape")
    trajectory_tape = result.get("trajectory_tape")
    if (
        isinstance(control_steps, bool)
        or not isinstance(control_steps, int)
        or not 1 <= control_steps <= EXECUTION_CONTRACT["horizon_control_steps"]
        or isinstance(physics_steps, bool)
        or not isinstance(physics_steps, int)
        or physics_steps
        != control_steps * EXECUTION_CONTRACT["physics_steps_per_control"]
        or terminal.get("control_step") != control_steps
        or terminal.get("policy_step") != control_steps
        or isinstance(terminal.get("physics_step"), bool)
        or not isinstance(terminal.get("physics_step"), int)
        or not (
            EXECUTION_CONTRACT["physics_steps_per_control"] * (control_steps - 1)
            < terminal["physics_step"]
            <= physics_steps
        )
    ):
        raise RuntimeError(
            "row result tape/terminal clocks violate the frozen contract"
        )
    if (
        not isinstance(action_tape, list)
        or len(action_tape) != control_steps
        or not isinstance(trajectory_tape, list)
        or len(trajectory_tape) != control_steps + 1
    ):
        raise RuntimeError(
            "row result action/trajectory tape cardinality is incomplete"
        )
    for index, action in enumerate(action_tape):
        if (
            not isinstance(action, Mapping)
            or set(action) != {"mode", "policy_step", "values"}
            or action.get("mode") != "E7"
            or action.get("policy_step") != index
        ):
            raise RuntimeError("row result action tape identity/order is invalid")
        values = action.get("values")
        if (
            not isinstance(values, list)
            or len(values) != 7
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not -1.0 <= value <= 1.0
                for value in values
            )
        ):
            raise RuntimeError("row result action tape contains an invalid E7 action")
    if early_accepted and (
        early_executed_steps != control_steps - 1
        or terminal_early.get("unexecuted_action_payload_sha256")
        != payload_sha256(action_tape[early_executed_steps:])
    ):
        raise RuntimeError(
            "early terminal-grid receipt is not bound to the primary action suffix"
        )
    for index, digest in enumerate(trajectory_tape):
        _require_sha256(digest, f"trajectory_tape[{index}]")
    if (
        not isinstance(evidence_export, Mapping)
        or evidence_export.get("passed") is not True
    ):
        raise RuntimeError("row result evidence export gate did not pass")
    action_digest = _require_sha256(
        evidence_export.get("action_tape_sha256"),
        "evidence_export.action_tape_sha256",
    )
    trajectory_digest = _require_sha256(
        evidence_export.get("trajectory_tape_sha256"),
        "evidence_export.trajectory_tape_sha256",
    )
    if action_digest != payload_sha256(action_tape):
        raise RuntimeError(
            "evidence action tape digest differs from the result payload"
        )
    if trajectory_digest != payload_sha256(trajectory_tape):
        raise RuntimeError(
            "evidence trajectory tape digest differs from the result payload"
        )
    _require_sha256(
        replay.get("replay_observation_sha256"),
        "replay.replay_observation_sha256",
    )
    _require_sha256(
        replay.get("replay_review_sha256"),
        "replay.replay_review_sha256",
    )
    _require_sha256(
        replay.get("replay_ledger_sha256"),
        "replay.replay_ledger_sha256",
    )
    # Replay digests are provenance receipts. Numeric/byte equality with the
    # primary CUDA rollout is diagnostic only under the semantic replay gate.
    evidence_files = (
        ("tape_file", "tape_file_sha256", ".npz"),
        ("visual_file", "visual_sha256", ".gif"),
    )
    paths: list[Path] = []
    for path_name, digest_name, suffix in evidence_files:
        value = evidence_export.get(path_name)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"evidence export lacks {path_name}")
        path = Path(value)
        if not path.is_absolute() or path.suffix.lower() != suffix:
            raise RuntimeError(
                f"evidence export {path_name} must be an absolute {suffix} path"
            )
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"evidence export file is missing: {path}") from exc
        if not resolved.is_file():
            raise RuntimeError(f"evidence export path is not a file: {resolved}")
        expected_digest = _require_sha256(
            evidence_export.get(digest_name),
            f"evidence_export.{digest_name}",
        )
        if sha256_file(resolved) != expected_digest:
            raise RuntimeError(f"evidence export file digest drifted: {resolved}")
        paths.append(resolved)
    if len(set(paths)) != len(paths):
        raise RuntimeError("evidence export paths must identify distinct files")


def summarize_d32_results(
    results: Sequence[Mapping[str, Any]],
    *,
    manifest: T1XYZFrozenManifest | None = None,
) -> dict[str, Any]:
    """Return no rate or countable outcome until all 32 strict rows pass."""

    rows = tuple(results)
    valid: dict[int, Mapping[str, Any]] = {}
    invalid = 0
    for result in rows:
        index = result.get("manifest_index")
        if manifest is None:
            invalid += 1
            continue
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(manifest.rows)
            or index in valid
        ):
            invalid += 1
            continue
        try:
            validate_result_for_row(result, manifest=manifest, row=manifest.row(index))
        except (KeyError, TypeError, ValueError, RuntimeError):
            invalid += 1
        else:
            valid[index] = result
    complete = (
        manifest is not None
        and manifest.phase == "d32"
        and len(rows) == len(manifest.rows) == 32
        and invalid == 0
        and set(valid) == set(range(32))
    )
    ordered = [valid[index] for index in range(32)] if complete else []
    successes = sum(int(bool(result.get("success"))) for result in ordered)
    failure_reasons = Counter(
        str(result["termination_reason"])
        for result in ordered
        if result.get("success") is False
    )
    drop_count = int(failure_reasons.get("drop", 0))
    return {
        "schema_version": D32_SUMMARY_SCHEMA_VERSION,
        "status": "completed" if complete else "incomplete_strict_evidence",
        "task_id": TASK_ID,
        "development_only": True,
        "qualification": False,
        "requested_total": 32 if manifest is None else len(manifest.rows),
        "received_rows": len(rows),
        "valid_completed": len(valid),
        "invalid_or_engineering_failures": invalid,
        "valid_successes": successes if complete else 0,
        "valid_failures": (32 - successes) if complete else 0,
        "drop_count": drop_count if complete else 0,
        "success_rate": (successes / 32) if complete else None,
        "drop_rate": (drop_count / 32) if complete else None,
        "failure_reason_counts": (
            {
                name: int(failure_reasons.get(name, 0))
                for name in FAILURE_TERMINATION_REASONS
            }
            if complete
            else {}
        ),
        "complete_cohort": complete,
        "candidate_result_available": complete,
        "promotion_eligible": False,
        "manifest_sha256": None if manifest is None else manifest.manifest_sha256,
        "results": list(rows),
    }


__all__ = [
    "BACKEND_ID",
    "CANONICAL_REQUEST_FIELDS",
    "D32_SUMMARY_SCHEMA_VERSION",
    "EXECUTION_CONTRACT",
    "FAILURE_TERMINATION_REASONS",
    "MANIFEST_SCHEMA_VERSION",
    "QUALITY_EVALUATOR_ID",
    "QUALITY_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SOURCE_REPOSITORIES",
    "TASK_ID",
    "T1XYZFrozenManifest",
    "canonical_json_bytes",
    "load_frozen_manifest",
    "payload_sha256",
    "preflight_export_request",
    "request_identity",
    "sha256_file",
    "summarize_d32_results",
    "validate_export_files",
    "validate_repository_tuple",
    "validate_result_for_row",
]
