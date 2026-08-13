# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed identity contracts for the GPUPLAN0 T4-sphere D32 repair.

The historical T4-sphere engineering run changed only episode IDs and seeds
while reusing the row-zero ramp geometry.  This module defines the replacement
contract: one exact export and one batch-size-one execution per frozen row.
Terminal-ledger materialization and fresh action-tape replay are mandatory
gates; neither can be downgraded to a diagnostic warning.

This module validates identities and aggregates already-produced row receipts.
It does not select GPU resources, acquire runtime leases, or authorize a result
phase.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_ID = "t4_sphere"
BACKEND_ID = "mjwarp_gpu_v1"
CANDIDATE_SCHEMA_VERSION = "gpuplan0-t4-sphere-candidate-identity-v2"
D32_MANIFEST_SCHEMA_VERSION = "gpuplan0-t4-sphere-d32-export-manifest-v2"
D32_ROW_REPORT_SCHEMA_VERSION = "gpuplan0-t4-sphere-d32-row-report-v2"
D32_AGGREGATE_SCHEMA_VERSION = "gpuplan0-t4-sphere-d32-aggregate-v2"
D32_MANIFEST_SEED = 20261040
D32_EPISODES = 32
D32_SEED_SET_SHA256 = "09b5763999ad9060710e6c1dd43403841efb1764f0b256299b29c51f303fcf61"
FACTOR_NAMES = frozenset({"lateral_offset_m", "ramp_angle_deg", "surface_friction"})

EXECUTION_CONTRACT = {
    "action_mode": "E7",
    "batch_size": 1,
    "control_hz": 20,
    "dynamics": "free_dynamics",
    "evaluator_backend_id": "gpuplan0-t4-sphere-current-main-v2",
    "horizon_control_steps": 120,
    "observation_track": "state",
    "physics_hz": 500,
    "physics_steps_per_control": 25,
    "quality_source": "terminal_ledger_task_quality",
    "replay_blocking": True,
    "sensor_hz": 20,
    "task_quality_schema_version": "db0-episode-task-quality-v2",
    "terminal_ledger_blocking": True,
}


@dataclass(frozen=True)
class T4SphereCandidateIdentity:
    """Validated candidate identity, including the matching repository tuple."""

    payload: Mapping[str, Any]
    candidate_sha256: str

    @property
    def repositories(self) -> Mapping[str, Any]:
        return self.payload["repositories"]

    @property
    def task_config_sha256(self) -> str:
        return str(self.payload["task_config_sha256"])


@dataclass(frozen=True)
class T4SphereManifest:
    """Validated 32-row export manifest bound to one candidate identity."""

    path: Path
    payload: Mapping[str, Any]
    manifest_sha256: str
    rows: tuple[Mapping[str, Any], ...]

    def row(self, candidate_index: int) -> Mapping[str, Any]:
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise TypeError("row index must be an integer")
        if not 0 <= candidate_index < D32_EPISODES:
            raise IndexError(f"row index must be in [0, {D32_EPISODES})")
        row = self.rows[candidate_index]
        if row["candidate_index"] != candidate_index:
            raise RuntimeError("D32 manifest row order changed after validation")
        return row

    def export_dir(self, row: Mapping[str, Any]) -> Path:
        value = Path(str(row["export_dir"]))
        if not value.is_absolute():
            value = (self.path.parent / value).resolve()
        return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single JSON representation used by all v2 identity digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Mapping[str, Any], digest_field: str) -> str:
    body = dict(payload)
    body.pop(digest_field, None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_git_oid(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-character Git object ID")
    return value


def _repository_identity(
    repositories: Mapping[str, Any],
    name: str,
    *,
    gitlink_name: str | None = None,
) -> Mapping[str, Any]:
    value = repositories.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"candidate repositories.{name} must be an object")
    _require_git_oid(value.get("commit"), f"repositories.{name}.commit")
    _require_git_oid(value.get("tree"), f"repositories.{name}.tree")
    if gitlink_name is not None:
        _require_git_oid(
            value.get(gitlink_name),
            f"repositories.{name}.{gitlink_name}",
        )
    return value


def load_candidate_identity(path: Path) -> T4SphereCandidateIdentity:
    """Load and validate a current-main candidate freeze before any GPU work."""

    payload = _load_json_object(path, "candidate identity")
    if payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("candidate identity schema is not the T4-sphere v2 contract")
    if payload.get("task_id") != TASK_ID or payload.get("backend_id") != BACKEND_ID:
        raise ValueError("candidate task/backend identity mismatch")
    candidate_id = payload.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string")
    task_config_sha256 = _require_sha256(
        payload.get("task_config_sha256"), "task_config_sha256"
    )
    repositories = payload.get("repositories")
    if not isinstance(repositories, Mapping) or set(repositories) != {
        "dynamic_benchmark",
        "mujoco_warp",
        "rlinf",
        "se3_wam",
    }:
        raise ValueError("candidate must freeze the exact four-repository tuple")
    se3 = _repository_identity(
        repositories, "se3_wam", gitlink_name="mujoco_warp_gitlink"
    )
    mjwarp = _repository_identity(repositories, "mujoco_warp")
    rlinf = _repository_identity(repositories, "rlinf")
    dynamic = _repository_identity(
        repositories, "dynamic_benchmark", gitlink_name="rlinf_gitlink"
    )
    if se3["mujoco_warp_gitlink"] != mjwarp["commit"]:
        raise ValueError("SE3-WAM MJWarp gitlink differs from the frozen MJWarp commit")
    if dynamic["rlinf_gitlink"] != rlinf["commit"]:
        raise ValueError(
            "Dynamic Benchmark RLinf gitlink differs from the Planner commit"
        )
    if payload.get("execution") != EXECUTION_CONTRACT:
        raise ValueError(
            "candidate execution contract differs from frozen STATE/E7 B=1"
        )
    declared_sha = _require_sha256(payload.get("candidate_sha256"), "candidate_sha256")
    expected_sha = _sha256_payload(payload, "candidate_sha256")
    if declared_sha != expected_sha:
        raise ValueError("candidate_sha256 does not match the candidate identity body")
    if task_config_sha256 != payload["task_config_sha256"]:
        raise AssertionError("validated task config digest changed unexpectedly")
    return T4SphereCandidateIdentity(payload=payload, candidate_sha256=declared_sha)


def _factor_tuple(row: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    factors = row.get("factors")
    if not isinstance(factors, Mapping) or set(factors) != FACTOR_NAMES:
        raise ValueError(
            "each T4-sphere row must contain the complete geometry factors"
        )
    converted = {name: float(factors[name]) for name in sorted(FACTOR_NAMES)}
    if not all(math.isfinite(value) for value in converted.values()):
        raise ValueError("T4-sphere geometry factors must be finite")
    if converted["ramp_angle_deg"] not in {5.0, 10.0, 15.0}:
        raise ValueError("T4-sphere D32 ramp angle is outside the frozen TEST_ID cells")
    if converted["surface_friction"] != 0.8:
        raise ValueError(
            "T4-sphere D32 surface friction differs from the frozen TEST_ID cell"
        )
    if not -0.015 <= converted["lateral_offset_m"] <= 0.015:
        raise ValueError(
            "T4-sphere D32 lateral offset is outside the frozen TEST_ID cell"
        )
    return tuple(converted.items())


def _validate_manifest_row(row: Any, index: int) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"D32 manifest row {index} must be an object")
    expected_scalars = {
        "action_mode": EXECUTION_CONTRACT["action_mode"],
        "candidate_index": index,
        "object_mode": "sphere",
        "observation_track": EXECUTION_CONTRACT["observation_track"],
        "reset_mode": "default",
        "split": "test_id",
        "task_id": TASK_ID,
    }
    for name, expected in expected_scalars.items():
        if row.get(name) != expected:
            raise ValueError(f"D32 row {index} field {name} differs from {expected!r}")
    episode_id = row.get("episode_id")
    seed = row.get("seed")
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError(f"D32 row {index} episode_id is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"D32 row {index} seed is invalid")
    api_version = row.get("api_version")
    if not isinstance(api_version, str) or not api_version.strip():
        raise ValueError(f"D32 row {index} api_version is invalid")
    export_dir = row.get("export_dir")
    if not isinstance(export_dir, str) or not export_dir.strip():
        raise ValueError(f"D32 row {index} export_dir is invalid")
    _require_sha256(row.get("export_sha256"), f"rows[{index}].export_sha256")
    _require_sha256(
        row.get("request_json_sha256"),
        f"rows[{index}].request_json_sha256",
    )
    _factor_tuple(row)
    teacher_identity_value = row.get("teacher_reset_identity")
    expected_teacher_fields = {
        "effective_capture_plane_world_y_m",
        "effective_close_lead_s",
        "effective_staging_eef_position_m",
    }
    if (
        not isinstance(teacher_identity_value, Mapping)
        or set(teacher_identity_value) != expected_teacher_fields
    ):
        raise ValueError(
            f"D32 row {index} lacks the complete per-row teacher reset identity"
        )
    teacher_reset_identity(teacher_identity_value)
    return row


def load_d32_manifest(
    path: Path,
    *,
    candidate: T4SphereCandidateIdentity,
    verify_exports: bool = True,
) -> T4SphereManifest:
    """Validate all 32 frozen rows before selecting a B=1 execution row."""

    resolved = path.resolve(strict=True)
    payload = _load_json_object(resolved, "D32 export manifest")
    if payload.get("schema_version") != D32_MANIFEST_SCHEMA_VERSION:
        raise ValueError("D32 manifest schema is not the per-row v2 contract")
    expected_top_level = {
        "action_mode": EXECUTION_CONTRACT["action_mode"],
        "backend_id": BACKEND_ID,
        "candidate_sha256": candidate.candidate_sha256,
        "episode_count": D32_EPISODES,
        "manifest_seed": D32_MANIFEST_SEED,
        "observation_track": EXECUTION_CONTRACT["observation_track"],
        "row_execution_batch_size": 1,
        "seed_set_sha256": D32_SEED_SET_SHA256,
        "split": "test_id",
        "task_config_sha256": candidate.task_config_sha256,
        "task_id": TASK_ID,
    }
    for name, expected in expected_top_level.items():
        if payload.get(name) != expected:
            raise ValueError(f"D32 manifest field {name} differs from {expected!r}")
    rows_value = payload.get("rows")
    if not isinstance(rows_value, list) or len(rows_value) != D32_EPISODES:
        raise ValueError("D32 manifest must contain exactly 32 rows")
    rows = tuple(
        _validate_manifest_row(row, index) for index, row in enumerate(rows_value)
    )
    candidate_indices = [row["candidate_index"] for row in rows]
    episode_ids = [str(row["episode_id"]) for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    export_dirs = [str(row["export_dir"]) for row in rows]
    if payload.get("candidate_indices") != candidate_indices:
        raise ValueError("D32 candidate_indices drifted from the row bodies")
    if payload.get("episode_ids") != episode_ids:
        raise ValueError("D32 episode_ids drifted from the row bodies")
    if len(set(episode_ids)) != D32_EPISODES:
        raise ValueError("D32 episode IDs are not globally unique")
    if len(set(seeds)) != D32_EPISODES:
        raise ValueError("D32 seeds are not globally unique")
    if len(set(export_dirs)) != D32_EPISODES:
        raise ValueError("D32 rows do not reference 32 distinct exports")
    factor_tuples = tuple(_factor_tuple(row) for row in rows)
    if len(set(factor_tuples)) < 2:
        raise ValueError("D32 rows all reuse one geometry factor tuple")
    teacher_identities = {
        canonical_json_bytes(teacher_reset_identity(row["teacher_reset_identity"]))
        for row in rows
    }
    if len(teacher_identities) < 2:
        raise ValueError("D32 rows all reuse one per-row teacher reset identity")
    seed_set_sha = hashlib.sha256(canonical_json_bytes(sorted(seeds))).hexdigest()
    if seed_set_sha != D32_SEED_SET_SHA256:
        raise ValueError("D32 seed set differs from the preregistered 32-row identity")
    declared_sha = _require_sha256(payload.get("manifest_sha256"), "manifest_sha256")
    expected_sha = _sha256_payload(payload, "manifest_sha256")
    if declared_sha != expected_sha:
        raise ValueError("manifest_sha256 does not match the D32 manifest body")
    manifest = T4SphereManifest(
        path=resolved,
        payload=payload,
        manifest_sha256=declared_sha,
        rows=rows,
    )
    if verify_exports:
        for index, row in enumerate(rows):
            export_dir = manifest.export_dir(row)
            if not export_dir.is_dir():
                raise FileNotFoundError(export_dir)
            sums = export_dir / "SHA256SUMS"
            request_json = export_dir / "request.json"
            if not sums.is_file() or not request_json.is_file():
                raise ValueError(f"D32 row {index} export is incomplete")
            if sha256_file(sums) != row["export_sha256"]:
                raise ValueError(f"D32 row {index} SHA256SUMS digest drifted")
            if sha256_file(request_json) != row["request_json_sha256"]:
                raise ValueError(f"D32 row {index} request.json digest drifted")
    return manifest


def request_identity(request: Any) -> dict[str, Any]:
    """Materialize every ResetRequest field that is frozen by the row contract."""

    def enum_value(value: Any) -> Any:
        return value.value if hasattr(value, "value") else value

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


def validate_request_identity(
    request: Any,
    row: Mapping[str, Any],
    *,
    actual_task_config_sha256: str,
    candidate: T4SphereCandidateIdentity,
) -> dict[str, Any]:
    """Fail before reset if one exported ResetRequest field differs from its row."""

    actual = request_identity(request)
    expected = {name: row[name] for name in actual}
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"T4-sphere export request identity drift: {mismatches}")
    if actual_task_config_sha256 != candidate.task_config_sha256:
        raise RuntimeError("T4-sphere export task configuration identity drift")
    if _factor_tuple(actual) != _factor_tuple(row):
        raise AssertionError("validated request factors changed unexpectedly")
    return actual


def assert_seed_disjointness(
    d32_rows: Sequence[Mapping[str, Any]],
    other_payload: Mapping[str, Any],
    *,
    other_name: str,
) -> dict[str, Any]:
    """Prove manifest uniqueness and D32 seed non-overlap without running episodes."""

    other_rows = other_payload.get("rows")
    if not isinstance(other_rows, list) or not other_rows:
        raise ValueError(f"{other_name} must contain a non-empty rows list")
    d32_seeds = [int(row["seed"]) for row in d32_rows]
    other_seeds = [int(row["seed"]) for row in other_rows]
    if len(set(d32_seeds)) != len(d32_seeds):
        raise ValueError("D32 seeds are not unique")
    if len(set(other_seeds)) != len(other_seeds):
        raise ValueError(f"{other_name} seeds are not unique")
    overlap = sorted(set(d32_seeds).intersection(other_seeds))
    if overlap:
        raise ValueError(f"{other_name} overlaps D32 seeds: {overlap}")
    return {
        "d32_rows": len(d32_seeds),
        "d32_unique_seeds": len(set(d32_seeds)),
        "other_name": other_name,
        "other_rows": len(other_seeds),
        "other_unique_seeds": len(set(other_seeds)),
        "seed_intersection": [],
    }


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
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
    root = root.resolve(strict=True)
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if head != expected["commit"] or tree != expected["tree"]:
        raise RuntimeError(f"source checkout identity mismatch at {root}")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError(f"source checkout is dirty: {root}")
    result = {"path": str(root), "commit": head, "tree": tree}
    if gitlink is not None:
        path, field = gitlink
        entry = _git(root, "ls-tree", "HEAD", "--", path).split()
        if len(entry) < 4 or entry[0] != "160000" or entry[1] != "commit":
            raise RuntimeError(f"source checkout has no gitlink at {path}")
        if entry[2] != expected[field]:
            raise RuntimeError(f"source checkout gitlink mismatch at {path}")
        result[field] = entry[2]
    return result


def validate_repository_tuple(
    candidate: T4SphereCandidateIdentity,
    *,
    se3_root: Path,
    rlinf_root: Path,
    dynamic_root: Path,
) -> dict[str, Any]:
    """Require exact clean SE3/RLinf/Dynamic checkouts and both gitlinks."""

    repositories = candidate.repositories
    se3 = _validate_checkout(
        se3_root,
        repositories["se3_wam"],
        gitlink=("third_party/mujoco_warp", "mujoco_warp_gitlink"),
    )
    rlinf = _validate_checkout(rlinf_root, repositories["rlinf"])
    dynamic = _validate_checkout(
        dynamic_root,
        repositories["dynamic_benchmark"],
        gitlink=("third_party/RLinf", "rlinf_gitlink"),
    )
    mjwarp = repositories["mujoco_warp"]
    if se3["mujoco_warp_gitlink"] != mjwarp["commit"]:
        raise RuntimeError(
            "runtime SE3 checkout does not pin the candidate MJWarp commit"
        )
    if dynamic["rlinf_gitlink"] != rlinf["commit"]:
        raise RuntimeError(
            "runtime Dynamic checkout does not pin the candidate RLinf commit"
        )
    return {
        "dynamic_benchmark": dynamic,
        "mujoco_warp": dict(mjwarp),
        "rlinf": rlinf,
        "se3_wam": se3,
    }


def validate_scientific_contract(
    candidate: T4SphereCandidateIdentity,
) -> dict[str, Any]:
    """Bind the runtime imports to the frozen task config and physical clocks."""

    from se3_wam.benchmark.config import load_task_config, task_config_sha256
    from se3_wam.benchmark.registry import get_task_spec

    config_sha = task_config_sha256(TASK_ID)
    if config_sha != candidate.task_config_sha256:
        raise RuntimeError(
            "runtime T4-sphere task config differs from the candidate freeze"
        )
    config = load_task_config(TASK_ID)
    task = get_task_spec(TASK_ID)
    clock = dict(config["clock"])
    actual = {
        "action_mode": "E7",
        "batch_size": 1,
        "control_hz": int(clock["control_hz"]),
        "dynamics": task.driver_type.value,
        "evaluator_backend_id": EXECUTION_CONTRACT["evaluator_backend_id"],
        "horizon_control_steps": int(clock["horizon_steps"]),
        "observation_track": "state",
        "physics_hz": int(clock["physics_hz"]),
        "physics_steps_per_control": int(clock["physics_hz"])
        // int(clock["control_hz"]),
        "quality_source": "terminal_ledger_task_quality",
        "replay_blocking": True,
        "sensor_hz": int(clock["sensor_hz"]),
        "task_quality_schema_version": EXECUTION_CONTRACT[
            "task_quality_schema_version"
        ],
        "terminal_ledger_blocking": True,
    }
    if actual != EXECUTION_CONTRACT:
        raise RuntimeError("runtime T4-sphere physics/clock contract drifted")
    return {
        "config_id": config.get("config_id"),
        "execution": actual,
        "task_config_sha256": config_sha,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return jsonable(value.to_dict())
    if hasattr(value, "value"):
        return jsonable(value.value)
    if hasattr(value, "item") and callable(value.item):
        return jsonable(value.item())
    return value


def teacher_reset_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Select and validate the request-sensitive T4 teacher reset parameters."""

    if not isinstance(metadata, Mapping):
        raise ValueError("T4 teacher reset metadata must be an object")
    staging = metadata.get("effective_staging_eef_position_m")
    if (
        not isinstance(staging, Sequence)
        or isinstance(staging, (str, bytes))
        or len(staging) != 3
    ):
        raise ValueError("T4 teacher reset metadata lacks a three-vector staging pose")
    staging_values = [float(value) for value in staging]
    close_lead = float(metadata.get("effective_close_lead_s"))
    capture_plane_value = metadata.get("effective_capture_plane_world_y_m")
    capture_plane = None if capture_plane_value is None else float(capture_plane_value)
    numeric_values = [*staging_values, close_lead]
    if capture_plane is not None:
        numeric_values.append(capture_plane)
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("T4 teacher reset metadata contains non-finite values")
    return {
        "effective_capture_plane_world_y_m": capture_plane,
        "effective_close_lead_s": close_lead,
        "effective_staging_eef_position_m": staging_values,
    }


def terminal_row_payload(row: Any) -> dict[str, Any]:
    """Serialize only authoritative ledger fields; quality remains null if absent."""

    quality = getattr(row, "task_quality", None)
    return {
        "completion": float(row.completion),
        "control_step": int(row.control_step),
        "episode_id": row.episode_id,
        "events": [
            {
                "name": event.name,
                "physics_step": int(event.physics_step),
                "time_s": float(event.time_s),
            }
            for event in row.events
        ],
        "lane": int(row.lane),
        "outcome": jsonable(row.outcome),
        "physics_step": int(row.physics_step),
        "policy_step": int(row.policy_step),
        "success": bool(row.success),
        "task_id": row.task_id,
        "task_quality": None if quality is None else jsonable(quality),
        "terminated": bool(row.terminated),
        "termination_reason": row.termination_reason,
        "truncated": bool(row.truncated),
    }


def validate_terminal_row(row: Any, *, episode_id: str) -> dict[str, Any]:
    """Require the blocking ledger row to identify one physical control interval."""

    payload = terminal_row_payload(row)
    if payload["lane"] != 0 or payload["episode_id"] != episode_id:
        raise RuntimeError("terminal ledger changed the B=1 lane/episode identity")
    if payload["task_id"] != TASK_ID:
        raise RuntimeError("terminal ledger changed the task identity")
    if payload["terminated"] == payload["truncated"]:
        raise RuntimeError(
            "terminal ledger must be exactly one of terminated/truncated"
        )
    if payload["policy_step"] != payload["control_step"]:
        raise RuntimeError("terminal ledger policy/control clocks differ")
    if not (
        25 * (payload["control_step"] - 1)
        < payload["physics_step"]
        <= 25 * payload["control_step"]
    ):
        raise RuntimeError("terminal ledger lacks a physical event clock")
    previous_step = -1
    for event in payload["events"]:
        if event["physics_step"] < previous_step:
            raise RuntimeError("terminal ledger event clocks are not monotonic")
        if event["physics_step"] > payload["physics_step"]:
            raise RuntimeError("terminal ledger event occurs after termination")
        expected_time = event["physics_step"] / 500.0
        if not math.isclose(event["time_s"], expected_time, abs_tol=1.0e-12):
            raise RuntimeError(
                "terminal ledger event time lacks the 500 Hz physical clock"
            )
        previous_step = event["physics_step"]
    if not payload["events"]:
        raise RuntimeError("terminal ledger contains no physical events")
    return payload


def validate_blocking_replay(
    *,
    primary_terminal: Mapping[str, Any],
    replay_terminal: Any,
    primary_observation_fingerprints: Sequence[str],
    replay_observation_fingerprints: Sequence[str],
) -> dict[str, Any]:
    """Require full terminal and per-control observation identity on fresh replay."""

    replay_payload = validate_terminal_row(
        replay_terminal,
        episode_id=str(primary_terminal["episode_id"]),
    )
    if replay_payload != dict(primary_terminal):
        raise RuntimeError(
            "fresh replay terminal ledger differs from the primary ledger"
        )
    primary = tuple(str(value) for value in primary_observation_fingerprints)
    replay = tuple(str(value) for value in replay_observation_fingerprints)
    if primary != replay:
        raise RuntimeError("fresh replay observation fingerprints differ from primary")
    return {
        "blocking": True,
        "observation_fingerprint_count": len(primary),
        "observation_fingerprints_match": True,
        "passed": True,
        "terminal_ledger_match": True,
    }


def aggregate_d32_reports(
    *,
    candidate: T4SphereCandidateIdentity,
    manifest: T4SphereManifest,
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly once only after all 32 blocking row receipts exist."""

    if len(reports) != D32_EPISODES:
        raise RuntimeError("D32 aggregate requires complete 32/32 row receipts")
    by_index: dict[int, Mapping[str, Any]] = {}
    for report in reports:
        if report.get("schema_version") != D32_ROW_REPORT_SCHEMA_VERSION:
            raise RuntimeError("D32 row report schema mismatch")
        if report.get("status") != "completed" or report.get("phase") != "d32":
            raise RuntimeError("D32 aggregate accepts only completed D32 row reports")
        if report.get("batch_size") != 1 or report.get("sample_count") != 1:
            raise RuntimeError("D32 row report is not one exact B=1 sample")
        if (
            report.get("counts_as_d32_result") is not True
            or report.get("closed_loop_planner") is not True
            or report.get("frozen_action_replay") is not False
        ):
            raise RuntimeError("D32 row report is not a live countable Planner sample")
        if (
            report.get("task_id") != TASK_ID
            or report.get("execution") != EXECUTION_CONTRACT
        ):
            raise RuntimeError("D32 row report task/execution contract drifted")
        identity = report.get("identity")
        if not isinstance(identity, Mapping):
            raise RuntimeError("D32 row report lacks frozen identity")
        if (
            identity.get("candidate_sha256") != candidate.candidate_sha256
            or identity.get("manifest_sha256") != manifest.manifest_sha256
            or identity.get("task_config_sha256") != candidate.task_config_sha256
        ):
            raise RuntimeError("D32 row report candidate/manifest identity mismatch")
        index = identity.get("candidate_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in by_index:
            raise RuntimeError(
                "D32 row report candidate index is invalid or duplicated"
            )
        expected_row = manifest.row(index)
        if (
            identity.get("episode_id") != expected_row["episode_id"]
            or identity.get("seed") != expected_row["seed"]
        ):
            raise RuntimeError("D32 row report episode identity mismatch")
        request = report.get("request")
        request_names = (
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
        expected_request = {name: expected_row[name] for name in request_names}
        if not isinstance(request, Mapping) or dict(request) != expected_request:
            raise RuntimeError("D32 row report full ResetRequest identity mismatch")
        teacher = report.get("teacher")
        if (
            not isinstance(teacher, Mapping)
            or teacher.get("full_reset_request_bound") is not True
            or teacher.get("reset_identity") != expected_row["teacher_reset_identity"]
        ):
            raise RuntimeError("D32 row report per-row teacher identity mismatch")
        resource = report.get("resource")
        if (
            not isinstance(resource, Mapping)
            or resource.get("gpu_created") is not True
            or not isinstance(resource.get("observed_device_uuid"), str)
            or resource.get("observed_device_uuid")
            != resource.get("expected_device_uuid")
        ):
            raise RuntimeError("D32 row report lacks the exact leased GPU identity")
        for gate_name in ("terminal_ledger", "replay"):
            gate = report.get(gate_name)
            if not isinstance(gate, Mapping) or gate.get("blocking") is not True:
                raise RuntimeError(f"D32 row report {gate_name} is not blocking")
            if gate.get("passed") is not True:
                raise RuntimeError(f"D32 row report {gate_name} did not pass")
            if gate.get("exact_once_second_consumption_rejected") is not True:
                raise RuntimeError(
                    f"D32 row report {gate_name} lacks an exact-once negative witness"
                )
        terminal = report.get("terminal")
        if not isinstance(terminal, Mapping):
            raise RuntimeError(
                "D32 row report lacks an authoritative terminal ledger row"
            )
        if terminal.get("episode_id") != expected_row["episode_id"]:
            raise RuntimeError("D32 terminal row episode identity mismatch")
        by_index[index] = report
    if set(by_index) != set(range(D32_EPISODES)):
        raise RuntimeError("D32 aggregate is missing one or more frozen row indices")

    ordered = [by_index[index] for index in range(D32_EPISODES)]
    success = sum(int(bool(report["terminal"]["success"])) for report in ordered)
    reasons: dict[str, int] = {}
    for report in ordered:
        reason = str(report["terminal"]["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "schema_version": D32_AGGREGATE_SCHEMA_VERSION,
        "status": "completed",
        "task_id": TASK_ID,
        "development_only": True,
        "candidate_sha256": candidate.candidate_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "completed": D32_EPISODES,
        "total": D32_EPISODES,
        "success": success,
        "failure": D32_EPISODES - success,
        "termination_reasons": dict(sorted(reasons.items())),
        "quality": [report["terminal"].get("task_quality") for report in ordered],
        "rows": [
            {
                "candidate_index": index,
                "episode_id": report["identity"]["episode_id"],
                "success": bool(report["terminal"]["success"]),
                "termination_reason": report["terminal"]["termination_reason"],
                "task_quality": report["terminal"].get("task_quality"),
            }
            for index, report in enumerate(ordered)
        ],
    }


__all__ = [
    "BACKEND_ID",
    "CANDIDATE_SCHEMA_VERSION",
    "D32_AGGREGATE_SCHEMA_VERSION",
    "D32_EPISODES",
    "D32_MANIFEST_SCHEMA_VERSION",
    "D32_MANIFEST_SEED",
    "D32_ROW_REPORT_SCHEMA_VERSION",
    "D32_SEED_SET_SHA256",
    "EXECUTION_CONTRACT",
    "FACTOR_NAMES",
    "TASK_ID",
    "T4SphereCandidateIdentity",
    "T4SphereManifest",
    "aggregate_d32_reports",
    "assert_seed_disjointness",
    "canonical_json_bytes",
    "jsonable",
    "load_candidate_identity",
    "load_d32_manifest",
    "request_identity",
    "sha256_file",
    "teacher_reset_identity",
    "terminal_row_payload",
    "validate_blocking_replay",
    "validate_repository_tuple",
    "validate_request_identity",
    "validate_scientific_contract",
    "validate_terminal_row",
]
