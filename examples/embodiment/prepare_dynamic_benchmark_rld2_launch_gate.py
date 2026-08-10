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

"""Prepare the immutable, pre-allocation RLD2 calibration launch package.

This utility performs checkpoint inventory only: it hashes and CPU-loads frozen
policies, builds an exact-14 deterministic-mean launch candidate pool, emits one
compatibility request per unique task/checkpoint, freezes fourteen planner-
calibration jobs, and signs the complete package.  Stochastic policy expansions
from historical manifests are deliberately excluded.  The utility never creates
a simulator or performs policy inference.  Consequently the output is explicitly
a launch input, not a production RLD2 candidate release.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.embodiment.build_dynamic_benchmark_rld2_manifests import (
    EXACT_TASKS,
    LEGACY_CANDIDATE_SCHEMA,
    LEGACY_TASKS,
)

SOURCE_SPEC_SCHEMA = "rlinf-dynamic-benchmark-rld2-launch-source-spec-v0.1"
LAUNCH_CANDIDATE_SCHEMA = (
    "rlinf-dynamic-benchmark-rld2-launch-candidates-v0.2"
)
CHECKPOINT_REQUEST_SCHEMA = (
    "rlinf-dynamic-benchmark-rld2-compatibility-request-v0.1"
)
CALIBRATION_JOB_SCHEMA = "rlinf-dynamic-benchmark-rld2-calibration-job-v0.1"
LANE_PLAN_SCHEMA = "rlinf-dynamic-benchmark-rld2-lane-plan-v0.1"
PACKAGE_SCHEMA = "rlinf-dynamic-benchmark-rld2-launch-package-v0.4"
DETERMINISTIC_INFERENCE_MODE = "deterministic_mean"
PLANNER_DOMINANCE_SCHEMA = "rlinf-dynamic-benchmark-planner-dominance-v0.1"
POLICY_SCHEMA = "rlinf-dynamic-benchmark-expert-policy-v0.1"
BACKEND_ID = "mujoco311-rs140-v1-rld2-quality"
REQUIRED_ADDITIONS = {
    "t1_xyz": ("RLE0", None),
    "t1_so3": ("RLOPT-SO3", "A4"),
    "t2_se3": ("RLOPT-SE3", "D1"),
    "p0_grasp": ("RLOPT-P0G", "A3"),
}
DEFAULT_LANES = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7")
LANE_UUIDS = {
    "L0": "GPU-4a88a785-f469-753c-9189-84154bb9a117",
    "L1": "GPU-ebe372b5-0e21-3bd2-35bd-222538da2102",
    "L2": "GPU-9335f48f-5da5-d490-ccf4-ebdebab38617",
    "L3": "GPU-be9057e0-1a71-5df6-c61c-91e7cd289423",
    "L4": "GPU-14e30924-0dd6-4435-1e80-758ff81dfd6f",
    "L5": "GPU-60a4eedc-98f9-ce75-c44f-61d0c0931bfe",
    "L6": "GPU-e090848d-104b-46ea-f7cf-e47b2bd72123",
    "L7": "GPU-1b2b1213-f608-9c8e-285c-3ab06dc8e738",
}
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class LaunchGateError(ValueError):
    """Raised when an RLD2 launch input fails closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_64.fullmatch(value) is None:
        raise LaunchGateError(f"{label} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_40.fullmatch(value) is None:
        raise LaunchGateError(f"{label} must be a full lowercase Git commit")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchGateError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise LaunchGateError(f"{label} must contain a JSON object")
    _canonical_json(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_identity(root: Path, expected_commit: str, label: str) -> dict[str, str]:
    if not root.is_dir():
        raise LaunchGateError(f"{label} source root is missing: {root}")

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        head = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaunchGateError(f"cannot verify {label} Git snapshot: {error}") from error
    if head != expected_commit:
        raise LaunchGateError(
            f"{label} snapshot HEAD mismatch: expected {expected_commit}, got {head}"
        )
    if status:
        raise LaunchGateError(f"{label} snapshot is not clean")
    return {"path": str(root.resolve()), "commit": expected_commit}


def _default_checkpoint_loader(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise LaunchGateError("checkpoint inventory requires PyTorch") from error
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise LaunchGateError(f"checkpoint payload must be a mapping: {path}")
    return payload


def _quality_schema_loader(se3_source_root: Path) -> Callable[[str], dict[str, Any]]:
    source = str((se3_source_root / "src").resolve())
    sys.path.insert(0, source)
    try:
        from se3_wam.benchmark.task_quality import task_quality_schema_manifest
    except ImportError as error:
        raise LaunchGateError("cannot import canonical task-quality schemas") from error
    finally:
        sys.path.pop(0)
    return task_quality_schema_manifest


def _checkpoint_inventory(
    *,
    task: str,
    path: Path,
    expected_sha256: str,
    expected_rlinf_commit: str,
    expected_benchmark_commit: str,
    expected_seed: int | None,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    if not path.is_file():
        raise LaunchGateError(f"missing checkpoint for {task}: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise LaunchGateError(
            f"checkpoint hash mismatch for {task}: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = checkpoint_loader(path)
    if payload.get("schema_version") != POLICY_SCHEMA:
        raise LaunchGateError(f"unsupported checkpoint schema for {task}: {path}")
    config = payload.get("config")
    state_schema = payload.get("state_schema")
    if (
        not isinstance(config, Mapping)
        or not isinstance(state_schema, Mapping)
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("normalizer"), Mapping)
    ):
        raise LaunchGateError(f"incomplete checkpoint payload for {task}: {path}")
    if (
        config.get("task") != task
        or config.get("rlinf_commit") != expected_rlinf_commit
        or config.get("benchmark_commit") != expected_benchmark_commit
    ):
        raise LaunchGateError(f"checkpoint source identity mismatch for {task}: {path}")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise LaunchGateError(f"checkpoint training seed is invalid for {task}: {path}")
    if expected_seed is not None and seed != expected_seed:
        raise LaunchGateError(f"checkpoint seed mismatch for {task}: {path}")
    state_dim = state_schema.get("state_dim")
    mask_dim = state_schema.get("mask_dim")
    if (
        isinstance(state_dim, bool)
        or not isinstance(state_dim, int)
        or state_dim < 1
        or isinstance(mask_dim, bool)
        or not isinstance(mask_dim, int)
        or not 0 <= mask_dim <= state_dim
    ):
        raise LaunchGateError(f"checkpoint state schema is invalid for {task}: {path}")
    env_steps = payload.get("env_steps")
    if isinstance(env_steps, bool) or not isinstance(env_steps, int) or env_steps < 0:
        raise LaunchGateError(f"checkpoint env_steps is invalid for {task}: {path}")
    reward_config = {
        key: value
        for key, value in config.items()
        if key.startswith("reward_")
        or key in {"algorithm", "residual_scale", "allow_failed_demos"}
    }
    return {
        "task": task,
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "schema_version": POLICY_SCHEMA,
        "policy_rlinf_commit": expected_rlinf_commit,
        "policy_benchmark_commit": expected_benchmark_commit,
        "training_seed": seed,
        "env_steps": env_steps,
        "config": copy.deepcopy(dict(config)),
        "config_sha256": _payload_sha256(config),
        "state_schema": copy.deepcopy(dict(state_schema)),
        "state_schema_sha256": _payload_sha256(state_schema),
        "state_dim": state_dim,
        "mask_dim": mask_dim,
        "action_dim": 7,
        "embedded_normalizer": True,
        "reward_contract": "rlinf-dynamic-benchmark-training-reward-config-v0.1",
        "reward_sha256": _payload_sha256(reward_config),
    }


def _source_spec(path: Path) -> dict[str, Any]:
    payload = _load_json(path, "RLD2 launch source spec")
    if set(payload) != {
        "schema_version",
        "release_id",
        "old_manifest_sha256",
        "additions",
    }:
        raise LaunchGateError("launch source spec field inventory mismatch")
    if (
        payload.get("schema_version") != SOURCE_SPEC_SCHEMA
        or payload.get("release_id") != "RLD2"
    ):
        raise LaunchGateError("launch source spec identity mismatch")
    old_hashes = payload.get("old_manifest_sha256")
    if not isinstance(old_hashes, Mapping) or set(old_hashes) != set(LEGACY_TASKS):
        raise LaunchGateError("old manifest hash inventory is not exact13")
    for task, sha256 in old_hashes.items():
        _require_sha256(sha256, f"old manifest {task}")
    additions = payload.get("additions")
    if not isinstance(additions, list) or len(additions) != len(REQUIRED_ADDITIONS):
        raise LaunchGateError("addition inventory must contain the four frozen groups")
    by_task: dict[str, dict[str, Any]] = {}
    for group in additions:
        if not isinstance(group, Mapping) or set(group) != {
            "task",
            "experiment",
            "arm",
            "source_rlinf_commit",
            "source_benchmark_commit",
            "residual_scale",
            "policies",
        }:
            raise LaunchGateError("addition group field inventory mismatch")
        task = group.get("task")
        if task not in REQUIRED_ADDITIONS or task in by_task:
            raise LaunchGateError(f"invalid or duplicate addition task {task!r}")
        expected_experiment, expected_arm = REQUIRED_ADDITIONS[task]
        if group.get("experiment") != expected_experiment or group.get("arm") != expected_arm:
            raise LaunchGateError(f"frozen addition identity mismatch for {task}")
        _require_commit(group.get("source_rlinf_commit"), f"{task} source RLinf commit")
        _require_commit(
            group.get("source_benchmark_commit"), f"{task} source benchmark commit"
        )
        residual_scale = group.get("residual_scale")
        if residual_scale is not None and (
            isinstance(residual_scale, bool)
            or not isinstance(residual_scale, (int, float))
            or not 0.0 < float(residual_scale) <= 1.0
        ):
            raise LaunchGateError(f"invalid residual scale for {task}")
        policies = group.get("policies")
        if not isinstance(policies, list) or len(policies) != 5:
            raise LaunchGateError(f"{task} requires five frozen policies")
        seeds = []
        for policy in policies:
            if not isinstance(policy, Mapping) or set(policy) != {
                "seed",
                "path",
                "sha256",
            }:
                raise LaunchGateError(f"{task} policy source field inventory mismatch")
            seed = policy.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise LaunchGateError(f"{task} policy seed is invalid")
            if not isinstance(policy.get("path"), str) or not policy["path"]:
                raise LaunchGateError(f"{task} policy path is missing")
            _require_sha256(policy.get("sha256"), f"{task} policy seed {seed}")
            seeds.append(seed)
        if sorted(seeds) != [1, 2, 3, 4, 5]:
            raise LaunchGateError(f"{task} policy seeds must be exact 1..5")
        by_task[task] = copy.deepcopy(dict(group))
    if set(by_task) != set(REQUIRED_ADDITIONS):
        raise LaunchGateError("addition task inventory mismatch")
    return {**payload, "addition_by_task": by_task}


def _discover_old_manifests(
    root: Path, expected_hashes: Mapping[str, str]
) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.glob("*/candidate_manifest.json")):
        payload = _load_json(path, "old candidate manifest")
        task = payload.get("task")
        if task in result:
            raise LaunchGateError(f"duplicate old candidate manifest for {task}")
        if task not in LEGACY_TASKS:
            raise LaunchGateError(f"unexpected old candidate manifest task {task!r}")
        if payload.get("schema_version") != LEGACY_CANDIDATE_SCHEMA:
            raise LaunchGateError(f"old candidate schema mismatch for {task}")
        if _sha256(path) != expected_hashes[task]:
            raise LaunchGateError(f"old candidate manifest hash mismatch for {task}")
        _require_commit(payload.get("rlinf_commit"), f"{task} old RLinf commit")
        _require_commit(payload.get("benchmark_commit"), f"{task} old benchmark commit")
        result[task] = (path, payload)
    if set(result) != set(LEGACY_TASKS):
        raise LaunchGateError("old candidate manifest root is not exact13")
    return result


def _expansion(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("kind") == "planner":
        return {"mode": "planner", "stochastic": False, "exploration_seed_offset": 0}
    stochastic = bool(candidate.get("stochastic", False))
    offset = int(candidate.get("exploration_seed_offset", 0))
    return {
        "mode": "stochastic" if stochastic else "deterministic",
        "stochastic": stochastic,
        "exploration_seed_offset": offset,
    }


def _semantics(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    if candidate.get("kind") == "planner":
        return ("planner",)
    return (
        "policy",
        candidate["policy_sha256"],
        bool(candidate.get("stochastic", False)),
        int(candidate.get("exploration_seed_offset", 0)),
        candidate.get("residual_scale"),
    )


def _policy_provenance(
    inventory: Mapping[str, Any],
    *,
    experiment: str,
    run: str,
    arm: str | None,
    source_manifest_path: str,
    source_manifest_sha256: str,
    backend_id: str,
    evaluator_rlinf_commit: str,
) -> dict[str, Any]:
    return {
        "origin": {
            "experiment": experiment,
            "run": run,
            "arm": arm,
            "train_seed": inventory["training_seed"],
        },
        "checkpoint": {
            "id": Path(inventory["path"]).name,
            "step": inventory["env_steps"],
            "path": inventory["path"],
            "sha256": inventory["sha256"],
        },
        "source": {
            "manifest_path": source_manifest_path,
            "manifest_sha256": source_manifest_sha256,
            "rlinf_commit": inventory["policy_rlinf_commit"],
        },
        "runtime": {
            "id": backend_id,
            "evaluator_rlinf_commit": evaluator_rlinf_commit,
        },
        "benchmark": {"commit": inventory["policy_benchmark_commit"]},
        "config": {
            "path": inventory["path"] + "#embedded-config",
            "sha256": inventory["config_sha256"],
        },
        "state_schema": {
            "schema_version": inventory["state_schema"].get("schema_version"),
            "sha256": inventory["state_schema_sha256"],
            "state_dim": inventory["state_dim"],
            "mask_dim": inventory["mask_dim"],
            "embedded_normalizer": True,
        },
        "reward": {
            "contract": inventory["reward_contract"],
            "sha256": inventory["reward_sha256"],
        },
        "selection": {
            "split": "validation",
            "rule": "frozen-formal-candidate",
            "test_exposure": {"test_id": False, "test_ood": False},
        },
    }


def _planner_provenance(
    *,
    task: str,
    source_path: str,
    source_sha256: str,
    quality_schema: Mapping[str, Any],
    backend_id: str,
    evaluator_rlinf_commit: str,
    evaluator_benchmark_commit: str,
) -> dict[str, Any]:
    reward = {
        "task": task,
        "backend_id": backend_id,
        "task_config_sha256": quality_schema["task_config_sha256"],
    }
    return {
        "origin": {
            "experiment": "RLD2",
            "run": "planner-calibration-launch-gate",
            "arm": "planner",
            "train_seed": None,
        },
        "checkpoint": {"id": None, "step": None, "path": None, "sha256": None},
        "source": {
            "manifest_path": source_path,
            "manifest_sha256": source_sha256,
            "rlinf_commit": evaluator_rlinf_commit,
        },
        "runtime": {
            "id": backend_id,
            "evaluator_rlinf_commit": evaluator_rlinf_commit,
        },
        "benchmark": {"commit": evaluator_benchmark_commit},
        "config": {
            "path": f"canonical-task-config://{task}",
            "sha256": quality_schema["task_config_sha256"],
        },
        "state_schema": {
            "schema_version": None,
            "sha256": None,
            "state_dim": None,
            "mask_dim": None,
            "embedded_normalizer": None,
        },
        "reward": {
            "contract": "rlinf-dynamic-benchmark-evaluator-reward-v0.1",
            "sha256": _payload_sha256(reward),
        },
        "selection": {
            "split": "validation",
            "rule": "same-reset-planner-reference",
            "test_exposure": {"test_id": False, "test_ood": False},
        },
    }


def _metric(
    direction: str,
    resolution: float,
    *,
    action_l2: bool = False,
    control_steps: bool = False,
) -> dict[str, Any]:
    result = {
        "direction": direction,
        "max_observed_replay_drift": 0.0,
        "scientific_resolution": resolution,
    }
    if action_l2:
        result.update(numeric_floor_absolute=1.0e-6, numeric_floor_relative=1.0e-6)
    else:
        result["numeric_floor"] = 0.0 if control_steps else 1.0e-6
    return result


def _contract_template(
    task: str, quality_schema: Mapping[str, Any], backend_id: str
) -> dict[str, Any]:
    quality_metrics = {
        row["name"]: _metric(
            "max" if row["direction"] == "maximize" else "min",
            float(row["scientific_resolution"]),
        )
        for row in quality_schema["components"]
    }
    tie_break = [
        "trajectory_completion",
        *(f"task_quality.{row['name']}" for row in quality_schema["components"]),
        "completion_time_s",
        "control_steps",
        "action_l2_sum",
    ]
    return {
        "schema_version": PLANNER_DOMINANCE_SCHEMA,
        "task": task,
        "backend_id": backend_id,
        "quality_schema": copy.deepcopy(dict(quality_schema)),
        "calibration": {
            "replay_count": 3,
            "reset_episode_id": "pending-allocation",
            "reset_manifest_sha256": "0" * 64,
            "evidence_path": "pending-allocation",
            "evidence_sha256": "0" * 64,
        },
        "metrics": {
            "trajectory_completion": _metric("max", 1.0e-6),
            "task_quality": quality_metrics,
            "completion_time_s": _metric("min", 0.002),
            "control_steps": _metric("min", 1.0, control_steps=True),
            "action_l2_sum": _metric("min", 1.0e-6, action_l2=True),
        },
        "tie_break_order": tie_break,
    }


def _release_sha256sums(root: Path) -> str:
    rows = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    return "\n".join(rows) + "\n"


def _validate_sha256sums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    declared: dict[str, str] = {}
    for index, line in enumerate(lines, start=1):
        if "  " not in line:
            raise LaunchGateError(f"malformed SHA256SUMS line {index}")
        sha256, relative = line.split("  ", 1)
        _require_sha256(sha256, f"SHA256SUMS line {index}")
        if not relative or relative in declared:
            raise LaunchGateError("SHA256SUMS paths must be non-empty and unique")
        declared[relative] = sha256
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(declared) != actual:
        raise LaunchGateError("SHA256SUMS file inventory mismatch")
    for relative, expected in declared.items():
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or _sha256(path) != expected:
            raise LaunchGateError(f"SHA256SUMS mismatch for {relative}")


def _runtime_deps_identity(root: Path) -> dict[str, str]:
    root = root.resolve()
    checksum_path = root / "SHA256SUMS"
    if not root.is_dir() or not checksum_path.is_file():
        raise LaunchGateError("runtime dependency snapshot or SHA256SUMS is missing")
    _validate_sha256sums(root)
    return {
        "path": str(root),
        "sha256sums_sha256": _sha256(checksum_path),
    }


def _origin_from_path(path: str) -> tuple[str, str]:
    parts = Path(path).parts
    try:
        index = parts.index("artifacts")
        experiment = parts[index + 1]
        run = "/".join(parts[index + 2 : -1])
    except (ValueError, IndexError):
        experiment = "legacy"
        run = Path(path).parent.name
    return experiment, run


def _build_package(
    *,
    old_manifest_root: Path,
    source_spec_path: Path,
    output_root: Path,
    rlinf_source_root: Path,
    rlinf_commit: str,
    se3_source_root: Path,
    se3_commit: str,
    runtime_deps_root: Path,
    backend_id: str,
    lanes: Sequence[str],
    manifest_seed: int,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] = _default_checkpoint_loader,
    quality_schema_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise LaunchGateError(f"refusing to overwrite output root: {output_root}")
    if tuple(lanes) != DEFAULT_LANES:
        raise LaunchGateError(
            "lane inventory must exactly match L0-L7 in canonical order"
        )
    rlinf_identity = _git_identity(rlinf_source_root, rlinf_commit, "RLinf")
    se3_identity = _git_identity(se3_source_root, se3_commit, "SE3-WAM")
    runtime_deps_identity = _runtime_deps_identity(runtime_deps_root)
    spec = _source_spec(source_spec_path)
    spec_sha256 = _sha256(source_spec_path)
    old_manifests = _discover_old_manifests(
        old_manifest_root, spec["old_manifest_sha256"]
    )
    schema_loader = quality_schema_loader or _quality_schema_loader(se3_source_root)
    quality_schemas = {task: dict(schema_loader(task)) for task in EXACT_TASKS}

    checkpoint_rows: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_manifests: dict[str, dict[str, Any]] = {}
    source_provenance: dict[tuple[str, str], dict[str, Any]] = {}
    for task in LEGACY_TASKS:
        manifest_path, payload = old_manifests[task]
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise LaunchGateError(f"old candidate pool is empty for {task}")
        normalized = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or candidate.get("kind") not in {
                "planner",
                "policy",
            }:
                raise LaunchGateError(f"old candidate row is invalid for {task}")
            row = copy.deepcopy(dict(candidate))
            if row["kind"] == "policy" and bool(row.get("stochastic", False)):
                continue
            if row["kind"] == "policy":
                sha256 = _require_sha256(
                    row.get("policy_sha256"), f"{task} old policy"
                )
                key = (task, sha256)
                if key not in checkpoint_rows:
                    inventory = _checkpoint_inventory(
                        task=task,
                        path=Path(row["policy_path"]),
                        expected_sha256=sha256,
                        expected_rlinf_commit=payload["rlinf_commit"],
                        expected_benchmark_commit=payload["benchmark_commit"],
                        expected_seed=None,
                        checkpoint_loader=checkpoint_loader,
                    )
                    experiment, run = _origin_from_path(inventory["path"])
                    checkpoint_rows[key] = inventory
                    source_provenance[key] = _policy_provenance(
                        inventory,
                        experiment=experiment,
                        run=run,
                        arm=None,
                        source_manifest_path=str(manifest_path.resolve()),
                        source_manifest_sha256=spec["old_manifest_sha256"][task],
                        backend_id=backend_id,
                        evaluator_rlinf_commit=rlinf_commit,
                    )
                row["provenance"] = copy.deepcopy(source_provenance[key])
            else:
                row["provenance"] = _planner_provenance(
                    task=task,
                    source_path=str(manifest_path.resolve()),
                    source_sha256=spec["old_manifest_sha256"][task],
                    quality_schema=quality_schemas[task],
                    backend_id=backend_id,
                    evaluator_rlinf_commit=rlinf_commit,
                    evaluator_benchmark_commit=se3_commit,
                )
            row["provenance"]["expansion"] = _expansion(row)
            normalized.append(row)
        candidate_manifests[task] = {
            "schema_version": LAUNCH_CANDIDATE_SCHEMA,
            "release_id": "RLD2",
            "task": task,
            "production_release": False,
            "inference_mode": DETERMINISTIC_INFERENCE_MODE,
            "evaluator_target": {
                "evaluator_rlinf_commit": rlinf_commit,
                "evaluator_benchmark_commit": se3_commit,
                "backend_id": backend_id,
            },
            "candidates": normalized,
        }

    t1_planner_source = str(source_spec_path.resolve())
    candidate_manifests["t1_xyz"] = {
        "schema_version": LAUNCH_CANDIDATE_SCHEMA,
        "release_id": "RLD2",
        "task": "t1_xyz",
        "production_release": False,
        "inference_mode": DETERMINISTIC_INFERENCE_MODE,
        "evaluator_target": {
            "evaluator_rlinf_commit": rlinf_commit,
            "evaluator_benchmark_commit": se3_commit,
            "backend_id": backend_id,
        },
        "candidates": [
            {
                "candidate_id": "planner",
                "kind": "planner",
                "provenance": {
                    **_planner_provenance(
                        task="t1_xyz",
                        source_path=t1_planner_source,
                        source_sha256=spec_sha256,
                        quality_schema=quality_schemas["t1_xyz"],
                        backend_id=backend_id,
                        evaluator_rlinf_commit=rlinf_commit,
                        evaluator_benchmark_commit=se3_commit,
                    ),
                    "expansion": {
                        "mode": "planner",
                        "stochastic": False,
                        "exploration_seed_offset": 0,
                    },
                },
            }
        ],
    }

    for task, group in spec["addition_by_task"].items():
        incoming = []
        for policy in sorted(group["policies"], key=lambda row: row["seed"]):
            inventory = _checkpoint_inventory(
                task=task,
                path=Path(policy["path"]),
                expected_sha256=policy["sha256"],
                expected_rlinf_commit=group["source_rlinf_commit"],
                expected_benchmark_commit=group["source_benchmark_commit"],
                expected_seed=policy["seed"],
                checkpoint_loader=checkpoint_loader,
            )
            configured_residual = inventory["config"].get("residual_scale")
            if (
                group["residual_scale"] is None
                or isinstance(configured_residual, bool)
                or not isinstance(configured_residual, (int, float))
                or float(configured_residual) != float(group["residual_scale"])
            ):
                raise LaunchGateError(
                    f"frozen candidate residual scale differs from checkpoint for {task}"
                )
            key = (task, inventory["sha256"])
            existing = checkpoint_rows.get(key)
            if existing is not None and existing != inventory:
                raise LaunchGateError(f"checkpoint identity collision for {task}")
            checkpoint_rows[key] = inventory
            run = "/".join(Path(inventory["path"]).parts[-4:-1])
            source_provenance[key] = _policy_provenance(
                inventory,
                experiment=group["experiment"],
                run=run,
                arm=group["arm"],
                source_manifest_path=str(source_spec_path.resolve()),
                source_manifest_sha256=spec_sha256,
                backend_id=backend_id,
                evaluator_rlinf_commit=rlinf_commit,
            )
            prefix = (
                f"{task}-{group['experiment']}-{group['arm'] or 'none'}-"
                f"s{policy['seed']}-best"
            ).lower().replace("_", "-")
            for mode, stochastic, offset in [("deterministic", False, 0)]:
                row = {
                    "candidate_id": prefix
                    + f"-{mode}"
                    + (f"-{offset}" if stochastic else ""),
                    "kind": "policy",
                    "policy_path": inventory["path"],
                    "policy_sha256": inventory["sha256"],
                    "stochastic": stochastic,
                    "exploration_seed_offset": offset,
                }
                if group["residual_scale"] is not None:
                    row["residual_scale"] = float(group["residual_scale"])
                provenance = copy.deepcopy(source_provenance[key])
                provenance["expansion"] = _expansion(row)
                row["provenance"] = provenance
                incoming.append(row)
        destination = candidate_manifests[task]["candidates"]
        seen = {_semantics(row) for row in destination}
        for candidate in incoming:
            identity = _semantics(candidate)
            if identity not in seen:
                seen.add(identity)
                destination.append(candidate)

    for task in EXACT_TASKS:
        candidates = candidate_manifests[task]["candidates"]
        if not candidates or candidates[0]["kind"] != "planner":
            raise LaunchGateError(f"{task} planner must be candidate zero")
        if sum(row["kind"] == "planner" for row in candidates) != 1:
            raise LaunchGateError(f"{task} must contain exactly one planner")
        if any(
            row["kind"] == "policy"
            and (
                bool(row.get("stochastic", False))
                or int(row.get("exploration_seed_offset", 0)) != 0
                or row.get("provenance", {}).get("expansion", {}).get("mode")
                != "deterministic"
            )
            for row in candidates
        ):
            raise LaunchGateError(
                f"{task} launch candidates violate deterministic-mean inference"
            )
        semantics = [_semantics(row) for row in candidates]
        if len(semantics) != len(set(semantics)):
            raise LaunchGateError(f"{task} launch candidates are not de-duplicated")

    checkpoint_requests = []
    for index, ((task, sha256), inventory) in enumerate(sorted(checkpoint_rows.items())):
        checkpoint_requests.append(
            {
                "schema_version": CHECKPOINT_REQUEST_SCHEMA,
                "request_id": f"{task}-{sha256[:16]}",
                "task": task,
                "policy_path": inventory["path"],
                "policy_sha256": sha256,
                "policy_rlinf_commit": inventory["policy_rlinf_commit"],
                "policy_benchmark_commit": inventory["policy_benchmark_commit"],
                "policy_state_schema_sha256": inventory["state_schema_sha256"],
                "policy_state_dim": inventory["state_dim"],
                "policy_mask_dim": inventory["mask_dim"],
                "policy_action_dim": inventory["action_dim"],
                "manifest_seed": manifest_seed,
                "manifest_episode_index": 0,
                "split": "validation",
                "test_exposure": {"test_id": False, "test_ood": False},
                "expected_output": f"compatibility/{task}-{sha256}/probe.json",
                "lane": lanes[index % len(lanes)],
            }
        )

    calibration_jobs = []
    for index, task in enumerate(EXACT_TASKS):
        lane = lanes[index % len(lanes)]
        calibration_jobs.append(
            {
                "schema_version": CALIBRATION_JOB_SCHEMA,
                "job_id": f"planner-calibration-{task}",
                "task": task,
                "backend_id": backend_id,
                "evaluator_identity": {
                    "evaluator_rlinf_commit": rlinf_commit,
                    "evaluator_benchmark_commit": se3_commit,
                    "backend_id": backend_id,
                },
                "split": "validation",
                "test_exposure": {"test_id": False, "test_ood": False},
                "manifest_seed": manifest_seed,
                "manifest_episode_index": 0,
                "replay_count": 3,
                "image_size": 64,
                "contract_template": f"calibration/{task}/contract_template.json",
                "expected_output_root": f"calibration/{task}",
                "lane": lane,
            }
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        for task in EXACT_TASKS:
            _write_json(
                staging / "candidates" / task / "candidate_manifest.json",
                candidate_manifests[task],
            )
            _write_json(
                staging / "calibration" / task / "contract_template.json",
                _contract_template(task, quality_schemas[task], backend_id),
            )
            job = next(row for row in calibration_jobs if row["task"] == task)
            _write_json(staging / "calibration" / task / "job.json", job)
        inventory_path = staging / "compatibility" / "checkpoint_requests.jsonl"
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(
            "".join(_canonical_json(row) + "\n" for row in checkpoint_requests),
            encoding="utf-8",
        )
        expected_outputs = {
            "schema_version": "rlinf-dynamic-benchmark-rld2-expected-outputs-v0.1",
            "lane_outputs": {
                lane: {
                    "lane_result": "lane_result.json",
                    "sha256sums": "SHA256SUMS",
                    "calibration": [
                        row["expected_output_root"]
                        for row in calibration_jobs
                        if row["lane"] == lane
                    ],
                    "compatibility": [
                        row["expected_output"]
                        for row in checkpoint_requests
                        if row["lane"] == lane
                    ],
                }
                for lane in lanes
            },
            "finalization": {
                "compatibility_evidence": "final/evidence/compatibility/*.json",
                "calibration_evidence": "final/evidence/calibration/*.json",
                "production_candidate_release": "final/candidates/release_manifest.json",
                "release_audit": "final/audit/audit_report.json",
            },
        }
        _write_json(staging / "expected_outputs.json", expected_outputs)
        lane_plans = {}
        for lane in lanes:
            plan = {
                "schema_version": LANE_PLAN_SCHEMA,
                "release_id": "RLD2",
                "lane": lane,
                "expected_gpu_uuid": LANE_UUIDS[lane],
                "calibration_jobs": [
                    f"calibration/{row['task']}/job.json"
                    for row in calibration_jobs
                    if row["lane"] == lane
                ],
                "compatibility_request_ids": [
                    row["request_id"]
                    for row in checkpoint_requests
                    if row["lane"] == lane
                ],
            }
            _write_json(staging / "lanes" / f"{lane}.json", plan)
            lane_plans[lane] = {
                "calibration_job_count": len(plan["calibration_jobs"]),
                "compatibility_request_count": len(
                    plan["compatibility_request_ids"]
                ),
            }
        package = {
            "schema_version": PACKAGE_SCHEMA,
            "release_id": "RLD2",
            "status": "blocked-awaiting-allocation",
            "production_release": False,
            "candidate_inference_mode": DETERMINISTIC_INFERENCE_MODE,
            "tasks": list(EXACT_TASKS),
            "task_count": len(EXACT_TASKS),
            "candidate_count": {
                task: len(candidate_manifests[task]["candidates"])
                for task in EXACT_TASKS
            },
            "unique_checkpoint_count": len(checkpoint_requests),
            "calibration_job_count": len(calibration_jobs),
            "calibration_replay_count_per_task": 3,
            "source_spec": {
                "path": str(source_spec_path.resolve()),
                "sha256": spec_sha256,
            },
            "old_manifest_root": str(old_manifest_root.resolve()),
            "rlinf_source": rlinf_identity,
            "se3_source": se3_identity,
            "runtime_deps": runtime_deps_identity,
            "backend_id": backend_id,
            "allowed_lanes": list(DEFAULT_LANES),
            "lanes": lane_plans,
            "allocation_required_before_execution": True,
        }
        package["payload_sha256"] = _payload_sha256(package)
        _write_json(staging / "launch_package.json", package)
        (staging / "SHA256SUMS").write_text(
            _release_sha256sums(staging), encoding="utf-8"
        )
        _validate_sha256sums(staging)
        os.replace(staging, output_root)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "prepared",
        "output_root": str(output_root.resolve()),
        "task_count": len(EXACT_TASKS),
        "unique_checkpoint_count": len(checkpoint_requests),
        "calibration_job_count": len(calibration_jobs),
        "launch_package_sha256": _sha256(output_root / "launch_package.json"),
        "sha256sums_sha256": _sha256(output_root / "SHA256SUMS"),
    }


def validate_package(root: Path) -> dict[str, Any]:
    """Validate a prepared package without rewriting it or loading policies."""

    root = root.resolve()
    _validate_sha256sums(root)
    package = _load_json(root / "launch_package.json", "launch package")
    lane_summaries = package.get("lanes")
    if (
        package.get("schema_version") != PACKAGE_SCHEMA
        or tuple(package.get("tasks", [])) != EXACT_TASKS
        or package.get("task_count") != len(EXACT_TASKS)
        or package.get("status") != "blocked-awaiting-allocation"
        or package.get("production_release") is not False
        or package.get("candidate_inference_mode")
        != DETERMINISTIC_INFERENCE_MODE
        or package.get("allocation_required_before_execution") is not True
        or package.get("allowed_lanes") != list(DEFAULT_LANES)
        or not isinstance(lane_summaries, dict)
        or tuple(lane_summaries) != DEFAULT_LANES
        or "forbidden_lane" in package
    ):
        raise LaunchGateError("launch package identity or state mismatch")
    stored_payload = package.get("payload_sha256")
    unhashed = dict(package)
    unhashed.pop("payload_sha256", None)
    if stored_payload != _payload_sha256(unhashed):
        raise LaunchGateError("launch package payload hash mismatch")
    runtime_deps = package.get("runtime_deps")
    if not isinstance(runtime_deps, Mapping):
        raise LaunchGateError("runtime dependency identity is missing")
    runtime_deps_root = Path(str(runtime_deps.get("path", ""))).resolve()
    runtime_identity = _runtime_deps_identity(runtime_deps_root)
    if runtime_deps != runtime_identity:
        raise LaunchGateError("runtime dependency identity mismatch")
    manifest_paths = list(root.glob("candidates/*/candidate_manifest.json"))
    if {path.parent.name for path in manifest_paths} != set(EXACT_TASKS):
        raise LaunchGateError("launch candidate manifest set is not exact14")
    manifests = {
        path.parent.name: _load_json(path, "launch candidate manifest")
        for path in manifest_paths
    }
    for task, manifest in manifests.items():
        candidates = manifest.get("candidates")
        if (
            manifest.get("schema_version") != LAUNCH_CANDIDATE_SCHEMA
            or manifest.get("task") != task
            or manifest.get("production_release") is not False
            or manifest.get("inference_mode") != DETERMINISTIC_INFERENCE_MODE
            or not isinstance(candidates, list)
            or not candidates
            or candidates[0].get("kind") != "planner"
            or len(candidates) != package.get("candidate_count", {}).get(task)
        ):
            raise LaunchGateError(
                f"{task} launch candidate identity or count mismatch"
            )
        for candidate in candidates:
            if candidate.get("kind") == "policy" and (
                candidate.get("stochastic", False) is not False
                or candidate.get("exploration_seed_offset", 0) != 0
                or candidate.get("provenance", {}).get("expansion", {}).get("mode")
                != "deterministic"
            ):
                raise LaunchGateError(
                    f"{task} launch package contains a stochastic policy candidate"
                )
    request_lines = (root / "compatibility" / "checkpoint_requests.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    requests = [json.loads(line) for line in request_lines]
    if len(requests) != package.get("unique_checkpoint_count"):
        raise LaunchGateError("compatibility request count mismatch")
    request_ids = [row.get("request_id") for row in requests]
    if len(request_ids) != len(set(request_ids)):
        raise LaunchGateError("compatibility request IDs are duplicated")
    for row in requests:
        if row.get("schema_version") != CHECKPOINT_REQUEST_SCHEMA:
            raise LaunchGateError("compatibility request schema mismatch")
        if row.get("lane") not in package["lanes"]:
            raise LaunchGateError("compatibility request uses an undeclared lane")
    calibration_paths = list(root.glob("calibration/*/job.json"))
    if {path.parent.name for path in calibration_paths} != set(EXACT_TASKS):
        raise LaunchGateError("planner calibration job set is not exact14")
    calibration_jobs = {
        path.parent.name: _load_json(path, "planner calibration job")
        for path in calibration_paths
    }
    for task, job in calibration_jobs.items():
        if (
            job.get("schema_version") != CALIBRATION_JOB_SCHEMA
            or job.get("task") != task
            or job.get("lane") not in lane_summaries
            or job.get("replay_count") != 3
        ):
            raise LaunchGateError("planner calibration job identity mismatch")
    lane_plan_paths = list(root.glob("lanes/*.json"))
    if {path.stem for path in lane_plan_paths} != set(DEFAULT_LANES):
        raise LaunchGateError("lane plan set is not exact L0-L7")
    for lane in DEFAULT_LANES:
        plan = _load_json(root / "lanes" / f"{lane}.json", "lane plan")
        expected_calibration = [
            f"calibration/{task}/job.json"
            for task in EXACT_TASKS
            if calibration_jobs[task]["lane"] == lane
        ]
        expected_requests = [
            row["request_id"] for row in requests if row["lane"] == lane
        ]
        if (
            plan.get("schema_version") != LANE_PLAN_SCHEMA
            or plan.get("release_id") != "RLD2"
            or plan.get("lane") != lane
            or plan.get("expected_gpu_uuid") != LANE_UUIDS[lane]
            or plan.get("calibration_jobs") != expected_calibration
            or plan.get("compatibility_request_ids") != expected_requests
            or lane_summaries[lane]
            != {
                "calibration_job_count": len(expected_calibration),
                "compatibility_request_count": len(expected_requests),
            }
        ):
            raise LaunchGateError("lane plan coverage or identity mismatch")
    return {
        "status": "validated",
        "task_count": len(manifest_paths),
        "unique_checkpoint_count": len(requests),
        "launch_package_sha256": _sha256(root / "launch_package.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--old-manifest-root", type=Path)
    parser.add_argument("--source-spec", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rlinf-source-root", type=Path)
    parser.add_argument("--rlinf-commit")
    parser.add_argument("--se3-source-root", type=Path)
    parser.add_argument("--se3-commit")
    parser.add_argument("--runtime-deps-root", type=Path)
    parser.add_argument("--backend-id", default=BACKEND_ID)
    parser.add_argument("--lane", action="append", dest="lanes")
    parser.add_argument("--manifest-seed", type=int, default=20262150)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.validate_only:
        forbidden = (
            args.old_manifest_root,
            args.source_spec,
            args.rlinf_source_root,
            args.rlinf_commit,
            args.se3_source_root,
            args.se3_commit,
            args.runtime_deps_root,
            args.lanes,
        )
        if any(value is not None for value in forbidden):
            raise LaunchGateError("--validate-only accepts only --output-root")
        result = validate_package(args.output_root)
    else:
        required = {
            "old_manifest_root": args.old_manifest_root,
            "source_spec": args.source_spec,
            "rlinf_source_root": args.rlinf_source_root,
            "rlinf_commit": args.rlinf_commit,
            "se3_source_root": args.se3_source_root,
            "se3_commit": args.se3_commit,
            "runtime_deps_root": args.runtime_deps_root,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise LaunchGateError(f"build mode missing required arguments: {missing}")
        result = _build_package(
            old_manifest_root=args.old_manifest_root,
            source_spec_path=args.source_spec,
            output_root=args.output_root,
            rlinf_source_root=args.rlinf_source_root,
            rlinf_commit=_require_commit(args.rlinf_commit, "RLinf evaluator commit"),
            se3_source_root=args.se3_source_root,
            se3_commit=_require_commit(args.se3_commit, "SE3 evaluator commit"),
            runtime_deps_root=args.runtime_deps_root,
            backend_id=args.backend_id,
            lanes=tuple(args.lanes or DEFAULT_LANES),
            manifest_seed=args.manifest_seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
