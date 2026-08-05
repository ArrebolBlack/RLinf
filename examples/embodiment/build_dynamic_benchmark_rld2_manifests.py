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

"""Build immutable exact-14 RLD2 Dynamic Benchmark candidate manifests.

The builder deliberately treats the RLD1 candidate manifests as read-only
inputs.  It copies their incumbent candidates into a new directory, creates the
missing ``t1_xyz`` pool, appends the frozen RLE0/A4/D1/A3 policies, expands each
new checkpoint into deterministic and stochastic candidates, and de-duplicates
by policy content plus rollout semantics.

Every output candidate carries a fixed-shape provenance object.  Unknown facts
are represented by explicit JSON nulls.  ``--production`` upgrades those nulls
to fail-closed validation errors for scientifically required fields.  The
result also contains ``input_inventory.jsonl`` and ``INPUTS.sha256`` so a later
export can prove exactly which source manifests, spec, and policy blobs were
used.

Build::

    python examples/embodiment/build_dynamic_benchmark_rld2_manifests.py \
      --old-manifest-root /inputs/RLD1/candidates \
      --input-spec /inputs/RLD2/input_spec.json \
      --output-root /outputs/RLD2/candidates \
      --path-map /old/policies=/mounted/policies \
      --production

Validation never rewrites the release::

    python examples/embodiment/build_dynamic_benchmark_rld2_manifests.py \
      --validate-only --output-root /outputs/RLD2/candidates --production
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CANDIDATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-candidates-v0.1"
INPUT_SPEC_SCHEMA = "rlinf-dynamic-benchmark-rld2-input-spec-v0.1"
PROVENANCE_SCHEMA = "rlinf-dynamic-benchmark-candidate-provenance-v0.1"
INVENTORY_SCHEMA = "rlinf-dynamic-benchmark-rld2-input-inventory-v0.1"
RELEASE_SCHEMA = "rlinf-dynamic-benchmark-rld2-candidate-release-v0.1"

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
LEGACY_TASKS = tuple(task for task in EXACT_TASKS if task != "t1_xyz")
REQUIRED_ADDITION_SOURCES = {
    "t1_xyz": ("RLE0", None),
    "t1_so3": ("RLOPT-SO3", "A4"),
    "t2_se3": ("RLOPT-SE3", "D1"),
    "p0_grasp": ("RLOPT-P0G", "A3"),
}

PROVENANCE_KEYS = {
    "schema_version",
    "origin",
    "checkpoint",
    "source",
    "runtime",
    "benchmark",
    "config",
    "state_schema",
    "reward",
    "selection",
    "expansion",
}
PROVENANCE_NESTED_KEYS = {
    "origin": {"experiment", "run", "arm", "train_seed"},
    "checkpoint": {"id", "step", "path", "sha256"},
    "source": {"manifest_path", "manifest_sha256", "rlinf_commit"},
    "runtime": {"id", "evaluator_rlinf_commit"},
    "benchmark": {"commit"},
    "config": {"path", "sha256"},
    "state_schema": {
        "schema_version",
        "sha256",
        "state_dim",
        "mask_dim",
        "embedded_normalizer",
    },
    "reward": {"contract", "sha256"},
    "selection": {"split", "rule", "test_exposure"},
    "expansion": {"mode", "stochastic", "exploration_seed_offset"},
}
TEST_EXPOSURE_KEYS = {"test_id", "test_ood"}
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class ManifestBuildError(ValueError):
    """Raised when an RLD2 manifest input fails closed."""


@dataclass(frozen=True)
class PathMap:
    """One lexical path-prefix rewrite."""

    source: str
    target: str


@dataclass
class BuildContext:
    """Mutable caches and inventory collected during one build."""

    path_maps: tuple[PathMap, ...]
    file_hash_cache: dict[str, str]
    input_files: dict[tuple[str, str], dict[str, Any]]
    inventory_rows: list[dict[str, Any]]
    deduplicated: list[dict[str, Any]]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_64.fullmatch(value) is None:
        raise ManifestBuildError(f"{label} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_40.fullmatch(value) is None:
        raise ManifestBuildError(f"{label} must be a full lowercase Git commit")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestBuildError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestBuildError(f"{label} must contain a JSON object")
    _canonical_json(value)
    return value


def _normalize_lexical_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if normalized != "/":
        normalized = normalized.rstrip("/")
    return normalized


def _parse_path_map(value: str) -> PathMap:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path maps must use SOURCE=TARGET")
    source, target = value.split("=", 1)
    source = _normalize_lexical_path(source.strip())
    target = _normalize_lexical_path(target.strip())
    if not source or not target:
        raise argparse.ArgumentTypeError("path-map prefixes must not be empty")
    return PathMap(source=source, target=target)


def _ordered_path_maps(values: Sequence[PathMap]) -> tuple[PathMap, ...]:
    sources = [value.source for value in values]
    if len(sources) != len(set(sources)):
        raise ManifestBuildError("path-map source prefixes must be unique")
    return tuple(sorted(values, key=lambda value: (-len(value.source), value.source)))


def _map_path(value: str, path_maps: Sequence[PathMap]) -> str:
    normalized = _normalize_lexical_path(value)
    for item in path_maps:
        if normalized == item.source:
            return item.target
        prefix = item.source + "/"
        if normalized.startswith(prefix):
            return item.target + normalized[len(item.source) :]
    return normalized


def _looks_absolute(value: str) -> bool:
    return value.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", value) is not None


def _resolved_policy_path(
    raw_path: Any,
    *,
    manifest_path: Path,
    path_maps: Sequence[PathMap],
) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ManifestBuildError("policy candidate is missing policy_path")
    source = raw_path.strip()
    if not _looks_absolute(source):
        source = str((manifest_path.parent / source).resolve())
    return _map_path(source, path_maps)


def _hash_input_file(
    path_value: str,
    *,
    expected_sha256: str | None,
    role: str,
    logical_id: str,
    context: BuildContext,
) -> str:
    path = Path(path_value)
    if not path.is_file():
        raise ManifestBuildError(f"missing {role} input {logical_id}: {path_value}")
    cache_key = str(path.resolve())
    actual = context.file_hash_cache.get(cache_key)
    if actual is None:
        actual = _sha256(path)
        context.file_hash_cache[cache_key] = actual
    if expected_sha256 is not None and actual != expected_sha256:
        raise ManifestBuildError(
            f"{role} hash mismatch for {logical_id}: expected {expected_sha256}, got {actual}"
        )
    key = (role, logical_id)
    row = {
        "role": role,
        "logical_id": logical_id,
        "path": path_value,
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }
    existing = context.input_files.get(key)
    if existing is not None and existing != row:
        raise ManifestBuildError(
            f"input logical identity collision: {role}:{logical_id}"
        )
    context.input_files[key] = row
    return actual


def _expansion(mode: str, offset: int) -> dict[str, Any]:
    if mode not in {"planner", "deterministic", "stochastic"}:
        raise ManifestBuildError(f"unsupported candidate expansion mode {mode!r}")
    return {
        "mode": mode,
        "stochastic": mode == "stochastic",
        "exploration_seed_offset": offset,
    }


def _blank_provenance(expansion: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "origin": {
            "experiment": None,
            "run": None,
            "arm": None,
            "train_seed": None,
        },
        "checkpoint": {
            "id": None,
            "step": None,
            "path": None,
            "sha256": None,
        },
        "source": {
            "manifest_path": None,
            "manifest_sha256": None,
            "rlinf_commit": None,
        },
        "runtime": {"id": None, "evaluator_rlinf_commit": None},
        "benchmark": {"commit": None},
        "config": {"path": None, "sha256": None},
        "state_schema": {
            "schema_version": None,
            "sha256": None,
            "state_dim": None,
            "mask_dim": None,
            "embedded_normalizer": None,
        },
        "reward": {"contract": None, "sha256": None},
        "selection": {
            "split": None,
            "rule": None,
            "test_exposure": {"test_id": None, "test_ood": None},
        },
        "expansion": dict(expansion),
    }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _candidate_expansion(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("kind") == "planner":
        return _expansion("planner", 0)
    stochastic = candidate.get("stochastic", False)
    if not isinstance(stochastic, bool):
        raise ManifestBuildError("candidate stochastic must be boolean")
    offset = candidate.get("exploration_seed_offset", 0)
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset < 2**31
    ):
        raise ManifestBuildError(
            "candidate exploration_seed_offset must be in [0, 2**31)"
        )
    return _expansion("stochastic" if stochastic else "deterministic", offset)


def _validate_optional_string(value: Any, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ManifestBuildError(f"{label} must be a non-empty string or null")


def _validate_provenance(
    provenance: Any,
    *,
    candidate: Mapping[str, Any],
    task: str,
    benchmark_commit: str,
    production: bool,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or set(provenance) != PROVENANCE_KEYS:
        raise ManifestBuildError(
            f"{task} candidate provenance field inventory mismatch"
        )
    value = copy.deepcopy(dict(provenance))
    if value.get("schema_version") != PROVENANCE_SCHEMA:
        raise ManifestBuildError(f"{task} candidate provenance schema mismatch")
    for section, keys in PROVENANCE_NESTED_KEYS.items():
        nested = value.get(section)
        if not isinstance(nested, Mapping) or set(nested) != keys:
            raise ManifestBuildError(
                f"{task} provenance {section} field inventory mismatch"
            )
        value[section] = dict(nested)
    exposure = value["selection"].get("test_exposure")
    if not isinstance(exposure, Mapping) or set(exposure) != TEST_EXPOSURE_KEYS:
        raise ManifestBuildError(f"{task} provenance test exposure inventory mismatch")
    value["selection"]["test_exposure"] = dict(exposure)

    for section, key in (
        ("origin", "experiment"),
        ("origin", "run"),
        ("origin", "arm"),
        ("checkpoint", "id"),
        ("checkpoint", "path"),
        ("source", "manifest_path"),
        ("runtime", "id"),
        ("config", "path"),
        ("state_schema", "schema_version"),
        ("reward", "contract"),
        ("selection", "split"),
        ("selection", "rule"),
    ):
        _validate_optional_string(value[section][key], f"provenance {section}.{key}")
    train_seed = value["origin"]["train_seed"]
    if train_seed is not None and (
        isinstance(train_seed, bool) or not isinstance(train_seed, int)
    ):
        raise ManifestBuildError(
            "provenance origin.train_seed must be an integer or null"
        )
    checkpoint_step = value["checkpoint"]["step"]
    if checkpoint_step is not None and (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise ManifestBuildError(
            "provenance checkpoint.step must be a non-negative integer or null"
        )
    for key in ("state_dim", "mask_dim"):
        item = value["state_schema"][key]
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int) or item < 1
        ):
            raise ManifestBuildError(
                f"provenance state_schema.{key} must be positive or null"
            )
    normalizer = value["state_schema"]["embedded_normalizer"]
    if normalizer is not None and not isinstance(normalizer, bool):
        raise ManifestBuildError(
            "provenance embedded_normalizer must be boolean or null"
        )
    for key, item in value["selection"]["test_exposure"].items():
        if item is not None and not isinstance(item, bool):
            raise ManifestBuildError(
                f"provenance test_exposure.{key} must be boolean or null"
            )
    for section, key in (
        ("checkpoint", "sha256"),
        ("source", "manifest_sha256"),
        ("config", "sha256"),
        ("state_schema", "sha256"),
        ("reward", "sha256"),
    ):
        item = value[section][key]
        if item is not None:
            _require_sha256(item, f"provenance {section}.{key}")
    for section, key in (
        ("source", "rlinf_commit"),
        ("runtime", "evaluator_rlinf_commit"),
        ("benchmark", "commit"),
    ):
        item = value[section][key]
        if item is not None:
            _require_commit(item, f"provenance {section}.{key}")
    if value["benchmark"]["commit"] not in {None, benchmark_commit}:
        raise ManifestBuildError(
            f"{task} candidate provenance benchmark commit mismatch"
        )
    expected_expansion = _candidate_expansion(candidate)
    if value["expansion"] != expected_expansion:
        raise ManifestBuildError(f"{task} candidate provenance expansion mismatch")
    if candidate.get("kind") == "policy":
        policy_path = candidate.get("policy_path")
        policy_sha256 = candidate.get("policy_sha256")
        if value["checkpoint"]["path"] not in {None, policy_path}:
            raise ManifestBuildError(f"{task} candidate checkpoint path mismatch")
        if value["checkpoint"]["sha256"] not in {None, policy_sha256}:
            raise ManifestBuildError(f"{task} candidate checkpoint hash mismatch")

    if production:
        required_paths = [
            ("origin", "experiment"),
            ("origin", "run"),
            ("source", "manifest_path"),
            ("source", "manifest_sha256"),
            ("source", "rlinf_commit"),
            ("runtime", "id"),
            ("runtime", "evaluator_rlinf_commit"),
            ("benchmark", "commit"),
            ("reward", "contract"),
            ("reward", "sha256"),
            ("selection", "split"),
            ("selection", "rule"),
        ]
        if candidate.get("kind") == "policy":
            required_paths.extend(
                [
                    ("origin", "train_seed"),
                    ("checkpoint", "id"),
                    ("checkpoint", "path"),
                    ("checkpoint", "sha256"),
                    ("config", "path"),
                    ("config", "sha256"),
                    ("state_schema", "schema_version"),
                    ("state_schema", "sha256"),
                    ("state_schema", "state_dim"),
                    ("state_schema", "mask_dim"),
                    ("state_schema", "embedded_normalizer"),
                ]
            )
        missing = [
            f"{section}.{key}"
            for section, key in required_paths
            if value[section][key] is None
        ]
        for key, item in value["selection"]["test_exposure"].items():
            if item is None:
                missing.append(f"selection.test_exposure.{key}")
            elif item:
                raise ManifestBuildError(
                    f"production provenance forbids test exposure: {task} {key}=true"
                )
        if missing:
            raise ManifestBuildError(
                f"production provenance for {task}/{candidate.get('candidate_id')} has nulls: "
                + ", ".join(missing)
            )
    _canonical_json(value)
    return value


def _candidate_semantics(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    kind = candidate.get("kind")
    if kind == "planner":
        return ("planner",)
    if kind != "policy":
        raise ManifestBuildError(f"unsupported candidate kind {kind!r}")
    residual = candidate.get("residual_scale")
    if residual is not None:
        residual = float(residual)
        if not 0.0 < residual <= 1.0:
            raise ManifestBuildError("candidate residual_scale must be in (0, 1]")
    return (
        "policy",
        _require_sha256(candidate.get("policy_sha256"), "candidate policy_sha256"),
        bool(candidate.get("stochastic", False)),
        int(candidate.get("exploration_seed_offset", 0)),
        residual,
    )


def _candidate_semantics_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    key = _candidate_semantics(candidate)
    if key[0] == "planner":
        return {"kind": "planner"}
    return {
        "kind": "policy",
        "policy_sha256": key[1],
        "stochastic": key[2],
        "exploration_seed_offset": key[3],
        "residual_scale": key[4],
    }


def _validate_old_candidate(
    row: Any,
    *,
    task: str,
    manifest_path: Path,
    manifest_sha256: str,
    manifest_rlinf_commit: str,
    manifest_benchmark_commit: str,
    override: Mapping[str, Any] | None,
    context: BuildContext,
    production: bool,
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ManifestBuildError(f"{task} old candidate row must be a mapping")
    candidate_id = row.get("candidate_id")
    kind = row.get("kind")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ManifestBuildError(f"{task} old candidate_id is missing")
    if kind not in {"planner", "policy"}:
        raise ManifestBuildError(f"{task}/{candidate_id} candidate kind is invalid")
    candidate: dict[str, Any] = {"candidate_id": candidate_id, "kind": kind}
    if kind == "planner":
        for key in ("policy_path", "policy_sha256", "residual_scale"):
            if row.get(key) is not None:
                raise ManifestBuildError(f"{task} planner cannot declare {key}")
        if row.get("stochastic", False) or row.get("exploration_seed_offset", 0):
            raise ManifestBuildError(f"{task} planner cannot declare exploration")
    else:
        expected_sha256 = _require_sha256(
            row.get("policy_sha256"), f"{task}/{candidate_id} policy_sha256"
        )
        mapped_path = _resolved_policy_path(
            row.get("policy_path"),
            manifest_path=manifest_path,
            path_maps=context.path_maps,
        )
        _hash_input_file(
            mapped_path,
            expected_sha256=expected_sha256,
            role="policy",
            logical_id=f"{task}:{candidate_id}",
            context=context,
        )
        candidate.update(
            policy_path=mapped_path,
            policy_sha256=expected_sha256,
            stochastic=row.get("stochastic", False),
            exploration_seed_offset=row.get("exploration_seed_offset", 0),
        )
        if row.get("residual_scale") is not None:
            candidate["residual_scale"] = float(row["residual_scale"])
    expansion = _candidate_expansion(candidate)
    provenance = _blank_provenance(expansion)
    provenance["source"].update(
        manifest_path=str(manifest_path.resolve()),
        manifest_sha256=manifest_sha256,
        rlinf_commit=manifest_rlinf_commit,
    )
    provenance["benchmark"]["commit"] = manifest_benchmark_commit
    if kind == "policy":
        provenance["checkpoint"].update(
            path=candidate["policy_path"], sha256=candidate["policy_sha256"]
        )
    if isinstance(row.get("provenance"), Mapping):
        provenance = _deep_merge(provenance, row["provenance"])
    if override is not None:
        provenance = _deep_merge(provenance, override)
    provenance["expansion"] = expansion
    if kind == "policy":
        provenance["checkpoint"]["path"] = candidate["policy_path"]
        provenance["checkpoint"]["sha256"] = candidate["policy_sha256"]
    candidate["provenance"] = _validate_provenance(
        provenance,
        candidate=candidate,
        task=task,
        benchmark_commit=manifest_benchmark_commit,
        production=production,
    )
    _candidate_semantics(candidate)
    return candidate


def _discover_old_manifests(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ManifestBuildError(f"old manifest root does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("candidate_manifest.json")):
        payload = _load_json(path, "old candidate manifest")
        task = payload.get("task")
        if not isinstance(task, str) or not task:
            raise ManifestBuildError(f"old candidate manifest has no task: {path}")
        if task in result:
            raise ManifestBuildError(f"duplicate old candidate manifest for {task}")
        result[task] = path
    tasks = set(result)
    if tasks != set(LEGACY_TASKS):
        missing = sorted(set(LEGACY_TASKS) - tasks)
        extra = sorted(tasks - set(LEGACY_TASKS))
        raise ManifestBuildError(
            f"old candidate boundary must be exact 13 without t1_xyz; missing={missing}, extra={extra}"
        )
    return result


def _validate_expansion_spec(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Mapping) or set(value) != {
        "include_deterministic",
        "exploration_seed_offsets",
    }:
        raise ManifestBuildError("stochastic_expansion field inventory mismatch")
    if value.get("include_deterministic") is not True:
        raise ManifestBuildError("RLD2 additions require deterministic candidates")
    offsets = value.get("exploration_seed_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 6
        or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
        or any(not 0 <= item < 2**31 for item in offsets)
        or len(offsets) != len(set(offsets))
    ):
        raise ManifestBuildError(
            "RLD2 stochastic expansion requires six unique integer offsets"
        )
    return tuple(offsets)


def _validate_spec(payload: Mapping[str, Any], *, production: bool) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "release_id",
        "candidate_schema_version",
        "rlinf_commit",
        "benchmark_commit",
        "stochastic_expansion",
        "additions",
        "provenance_overrides",
        "planner_dominance",
    }
    if set(payload) - allowed:
        raise ManifestBuildError("RLD2 input spec has unsupported top-level fields")
    if payload.get("schema_version") != INPUT_SPEC_SCHEMA:
        raise ManifestBuildError("unsupported RLD2 input spec schema")
    if payload.get("release_id") != "RLD2":
        raise ManifestBuildError("RLD2 input spec release_id mismatch")
    if payload.get("candidate_schema_version") != CANDIDATE_SCHEMA:
        raise ManifestBuildError("RLD2 candidate schema version mismatch")
    rlinf_commit = _require_commit(
        payload.get("rlinf_commit"), "RLD2 evaluator RLinf commit"
    )
    benchmark_commit = _require_commit(
        payload.get("benchmark_commit"), "RLD2 benchmark commit"
    )
    offsets = _validate_expansion_spec(payload.get("stochastic_expansion"))
    additions = payload.get("additions")
    if not isinstance(additions, list):
        raise ManifestBuildError("RLD2 additions must be a list")
    addition_by_task: dict[str, dict[str, Any]] = {}
    for group in additions:
        if not isinstance(group, Mapping) or set(group) != {"task", "policies"}:
            raise ManifestBuildError("RLD2 addition group field inventory mismatch")
        task = group.get("task")
        if task in addition_by_task:
            raise ManifestBuildError(f"duplicate RLD2 addition group for {task}")
        if task not in REQUIRED_ADDITION_SOURCES:
            raise ManifestBuildError(f"unexpected RLD2 addition task {task!r}")
        policies = group.get("policies")
        if not isinstance(policies, list) or len(policies) != 5:
            raise ManifestBuildError(
                f"RLD2 {task} requires exactly five frozen policies"
            )
        addition_by_task[str(task)] = {"task": task, "policies": list(policies)}
    if set(addition_by_task) != set(REQUIRED_ADDITION_SOURCES):
        missing = sorted(set(REQUIRED_ADDITION_SOURCES) - set(addition_by_task))
        raise ManifestBuildError(
            f"RLD2 addition inventory is incomplete; missing={missing}"
        )

    overrides = payload.get("provenance_overrides", [])
    if not isinstance(overrides, list):
        raise ManifestBuildError("provenance_overrides must be a list")
    normalized_overrides = []
    selectors = set()
    for row in overrides:
        if not isinstance(row, Mapping) or set(row) != {
            "task",
            "candidate_id",
            "policy_sha256",
            "provenance",
        }:
            raise ManifestBuildError("provenance override field inventory mismatch")
        task = row.get("task")
        candidate_id = row.get("candidate_id")
        policy_sha256 = row.get("policy_sha256")
        if task not in EXACT_TASKS:
            raise ManifestBuildError(f"provenance override has unknown task {task!r}")
        if candidate_id is not None and (
            not isinstance(candidate_id, str) or not candidate_id.strip()
        ):
            raise ManifestBuildError("override candidate_id must be non-empty or null")
        if policy_sha256 is not None:
            _require_sha256(policy_sha256, "override policy_sha256")
        if (candidate_id is None) == (policy_sha256 is None):
            raise ManifestBuildError(
                "provenance override must select exactly one of candidate_id or policy_sha256"
            )
        if not isinstance(row.get("provenance"), Mapping):
            raise ManifestBuildError("provenance override payload must be a mapping")
        selector = (task, candidate_id, policy_sha256)
        if selector in selectors:
            raise ManifestBuildError(
                f"duplicate provenance override selector {selector}"
            )
        selectors.add(selector)
        normalized_overrides.append(dict(row))

    dominance = payload.get("planner_dominance")
    if dominance is None:
        dominance = {}
    if not isinstance(dominance, Mapping):
        raise ManifestBuildError("planner_dominance must be a task mapping")
    if set(dominance) - set(EXACT_TASKS):
        raise ManifestBuildError("planner_dominance includes an unknown task")
    for task, contract in dominance.items():
        if not isinstance(contract, Mapping) or not contract:
            raise ManifestBuildError(f"planner_dominance contract for {task} is empty")
        _canonical_json(contract)
    if production and set(dominance) != set(EXACT_TASKS):
        missing = sorted(set(EXACT_TASKS) - set(dominance))
        raise ManifestBuildError(
            f"production RLD2 requires planner_dominance for exact14: {missing}"
        )
    return {
        "rlinf_commit": rlinf_commit,
        "benchmark_commit": benchmark_commit,
        "offsets": offsets,
        "addition_by_task": addition_by_task,
        "overrides": normalized_overrides,
        "planner_dominance": dict(dominance),
    }


def _override_for_candidate(
    overrides: Sequence[Mapping[str, Any]],
    *,
    task: str,
    candidate: Mapping[str, Any],
    used: set[int],
) -> Mapping[str, Any] | None:
    matches = []
    for index, row in enumerate(overrides):
        if row["task"] != task:
            continue
        candidate_id = row["candidate_id"]
        policy_sha256 = row["policy_sha256"]
        if candidate_id is not None and candidate_id == candidate.get("candidate_id"):
            matches.append((index, row["provenance"]))
        elif policy_sha256 is not None and policy_sha256 == candidate.get(
            "policy_sha256"
        ):
            matches.append((index, row["provenance"]))
    if len(matches) > 1:
        raise ManifestBuildError(
            f"multiple provenance overrides match {task}/{candidate.get('candidate_id')}"
        )
    if not matches:
        return None
    index, provenance = matches[0]
    used.add(index)
    return provenance


def _validate_addition_policy(
    row: Any,
    *,
    task: str,
    spec_path: Path,
    benchmark_commit: str,
    context: BuildContext,
) -> dict[str, Any]:
    allowed = {"policy_path", "policy_sha256", "residual_scale", "provenance"}
    if not isinstance(row, Mapping) or set(row) != allowed:
        raise ManifestBuildError(f"{task} addition policy field inventory mismatch")
    policy_sha256 = _require_sha256(
        row.get("policy_sha256"), f"{task} addition policy hash"
    )
    policy_path = _resolved_policy_path(
        row.get("policy_path"), manifest_path=spec_path, path_maps=context.path_maps
    )
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ManifestBuildError(f"{task} addition policy provenance is missing")
    origin = provenance.get("origin")
    if not isinstance(origin, Mapping):
        raise ManifestBuildError(f"{task} addition origin provenance is missing")
    expected_experiment, expected_arm = REQUIRED_ADDITION_SOURCES[task]
    if origin.get("experiment") != expected_experiment:
        raise ManifestBuildError(
            f"{task} additions must originate from {expected_experiment}, got {origin.get('experiment')}"
        )
    if expected_arm is not None and origin.get("arm") != expected_arm:
        raise ManifestBuildError(f"{task} additions must use frozen arm {expected_arm}")
    train_seed = origin.get("train_seed")
    if isinstance(train_seed, bool) or not isinstance(train_seed, int):
        raise ManifestBuildError(f"{task} addition train_seed must be an integer")
    _hash_input_file(
        policy_path,
        expected_sha256=policy_sha256,
        role="policy",
        logical_id=f"{task}:addition-seed-{train_seed}",
        context=context,
    )
    residual = row.get("residual_scale")
    if residual is not None:
        residual = float(residual)
        if not 0.0 < residual <= 1.0:
            raise ManifestBuildError(
                f"{task} addition residual_scale must be in (0, 1]"
            )
    return {
        "policy_path": policy_path,
        "policy_sha256": policy_sha256,
        "residual_scale": residual,
        "provenance": copy.deepcopy(dict(provenance)),
        "train_seed": train_seed,
    }


def _slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "unknown"


def _addition_candidates(
    source: Mapping[str, Any],
    *,
    task: str,
    offsets: Sequence[int],
    spec_path: Path,
    benchmark_commit: str,
    context: BuildContext,
    production: bool,
) -> list[dict[str, Any]]:
    policies = [
        _validate_addition_policy(
            row,
            task=task,
            spec_path=spec_path,
            benchmark_commit=benchmark_commit,
            context=context,
        )
        for row in source["policies"]
    ]
    seeds = [item["train_seed"] for item in policies]
    if len(seeds) != len(set(seeds)):
        raise ManifestBuildError(f"{task} additions must have five unique train seeds")
    candidates = []
    for item in sorted(policies, key=lambda value: value["train_seed"]):
        origin = item["provenance"]["origin"]
        prefix = "-".join(
            (
                _slug(task),
                _slug(origin["experiment"]),
                _slug(origin.get("arm")),
                f"s{item['train_seed']}",
                "best",
            )
        )
        variants = [("deterministic", False, 0)]
        variants.extend(("stochastic", True, offset) for offset in offsets)
        for mode, stochastic, offset in variants:
            candidate = {
                "candidate_id": f"{prefix}-{mode}"
                + (f"-{offset}" if stochastic else ""),
                "kind": "policy",
                "policy_path": item["policy_path"],
                "policy_sha256": item["policy_sha256"],
                "stochastic": stochastic,
                "exploration_seed_offset": offset,
            }
            if item["residual_scale"] is not None:
                candidate["residual_scale"] = item["residual_scale"]
            provenance = _deep_merge(
                _blank_provenance(_expansion(mode, offset)), item["provenance"]
            )
            provenance["checkpoint"]["path"] = item["policy_path"]
            provenance["checkpoint"]["sha256"] = item["policy_sha256"]
            provenance["benchmark"]["commit"] = benchmark_commit
            provenance["expansion"] = _expansion(mode, offset)
            candidate["provenance"] = _validate_provenance(
                provenance,
                candidate=candidate,
                task=task,
                benchmark_commit=benchmark_commit,
                production=production,
            )
            candidates.append(candidate)
    return candidates


def _append_deduplicated(
    destination: list[dict[str, Any]],
    incoming: Sequence[dict[str, Any]],
    *,
    task: str,
    context: BuildContext,
) -> None:
    semantics_to_index = {
        _candidate_semantics(row): index for index, row in enumerate(destination)
    }
    ids = {row["candidate_id"]: _candidate_semantics(row) for row in destination}
    for candidate in incoming:
        semantics = _candidate_semantics(candidate)
        candidate_id = candidate["candidate_id"]
        existing_semantics = ids.get(candidate_id)
        if existing_semantics is not None and existing_semantics != semantics:
            raise ManifestBuildError(
                f"candidate ID collision with different semantics: {task}/{candidate_id}"
            )
        if semantics in semantics_to_index:
            kept = destination[semantics_to_index[semantics]]
            context.deduplicated.append(
                {
                    "task": task,
                    "kept_candidate_id": kept["candidate_id"],
                    "dropped_candidate_id": candidate_id,
                    "semantics": _candidate_semantics_payload(candidate),
                }
            )
            continue
        semantics_to_index[semantics] = len(destination)
        ids[candidate_id] = semantics
        destination.append(candidate)


def _inventory_candidate(
    *,
    task: str,
    index: int,
    candidate: Mapping[str, Any],
    source_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": INVENTORY_SCHEMA,
        "task": task,
        "candidate_index": index,
        "candidate_id": candidate["candidate_id"],
        "kind": candidate["kind"],
        "source_kind": source_kind,
        "semantics": _candidate_semantics_payload(candidate),
        "semantics_sha256": _payload_sha256(_candidate_semantics_payload(candidate)),
        "provenance": copy.deepcopy(candidate["provenance"]),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _snapshot_tree(paths: Mapping[str, Path]) -> dict[str, str]:
    return {task: _sha256(path) for task, path in paths.items()}


def _build_in_memory(
    *,
    old_manifests: Mapping[str, Path],
    spec: Mapping[str, Any],
    spec_path: Path,
    context: BuildContext,
    production: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    normalized = _validate_spec(spec, production=production)
    rlinf_commit = normalized["rlinf_commit"]
    benchmark_commit = normalized["benchmark_commit"]
    overrides = normalized["overrides"]
    used_overrides: set[int] = set()
    manifests: dict[str, dict[str, Any]] = {}
    source_kind_by_identity: dict[tuple[str, tuple[Any, ...]], str] = {}

    old_payloads = {
        task: _load_json(path, "old candidate manifest")
        for task, path in old_manifests.items()
    }
    for task in LEGACY_TASKS:
        path = old_manifests[task]
        payload = old_payloads[task]
        if (
            payload.get("schema_version") != CANDIDATE_SCHEMA
            or payload.get("task") != task
        ):
            raise ManifestBuildError(
                f"old candidate manifest schema/task mismatch for {task}"
            )
        old_rlinf_commit = _require_commit(
            payload.get("rlinf_commit"), f"{task} old RLinf commit"
        )
        old_benchmark_commit = _require_commit(
            payload.get("benchmark_commit"), f"{task} old benchmark commit"
        )
        if old_benchmark_commit != benchmark_commit:
            raise ManifestBuildError(f"{task} old benchmark commit differs from RLD2")
        rows = payload.get("candidates")
        if not isinstance(rows, list) or not rows:
            raise ManifestBuildError(f"old candidate manifest for {task} is empty")
        manifest_sha256 = _sha256(path)
        _hash_input_file(
            str(path.resolve()),
            expected_sha256=manifest_sha256,
            role="old-manifest",
            logical_id=task,
            context=context,
        )
        candidates: list[dict[str, Any]] = []
        for raw_row in rows:
            preliminary = dict(raw_row) if isinstance(raw_row, Mapping) else {}
            override = _override_for_candidate(
                overrides,
                task=task,
                candidate=preliminary,
                used=used_overrides,
            )
            candidate = _validate_old_candidate(
                raw_row,
                task=task,
                manifest_path=path,
                manifest_sha256=manifest_sha256,
                manifest_rlinf_commit=old_rlinf_commit,
                manifest_benchmark_commit=benchmark_commit,
                override=override,
                context=context,
                production=production,
            )
            before = len(candidates)
            _append_deduplicated(candidates, [candidate], task=task, context=context)
            if len(candidates) != before:
                source_kind_by_identity[(task, _candidate_semantics(candidate))] = (
                    "incumbent"
                )
        if not candidates or candidates[0]["kind"] != "planner":
            raise ManifestBuildError(
                f"{task} old pool must preserve planner at index zero"
            )
        manifests[task] = {
            "schema_version": CANDIDATE_SCHEMA,
            "task": task,
            "rlinf_commit": rlinf_commit,
            "benchmark_commit": benchmark_commit,
            "candidates": candidates,
        }

    planner = {"candidate_id": "planner", "kind": "planner"}
    planner_override = _override_for_candidate(
        overrides,
        task="t1_xyz",
        candidate=planner,
        used=used_overrides,
    )
    planner_provenance = _blank_provenance(_expansion("planner", 0))
    if planner_override is not None:
        planner_provenance = _deep_merge(planner_provenance, planner_override)
    planner_provenance["benchmark"]["commit"] = benchmark_commit
    planner["provenance"] = _validate_provenance(
        planner_provenance,
        candidate=planner,
        task="t1_xyz",
        benchmark_commit=benchmark_commit,
        production=production,
    )
    manifests["t1_xyz"] = {
        "schema_version": CANDIDATE_SCHEMA,
        "task": "t1_xyz",
        "rlinf_commit": rlinf_commit,
        "benchmark_commit": benchmark_commit,
        "candidates": [planner],
    }
    source_kind_by_identity[("t1_xyz", _candidate_semantics(planner))] = (
        "synthetic-planner"
    )

    for task, source in normalized["addition_by_task"].items():
        additions = _addition_candidates(
            source,
            task=task,
            offsets=normalized["offsets"],
            spec_path=spec_path,
            benchmark_commit=benchmark_commit,
            context=context,
            production=production,
        )
        candidates = manifests[task]["candidates"]
        before_keys = {_candidate_semantics(row) for row in candidates}
        _append_deduplicated(candidates, additions, task=task, context=context)
        for candidate in candidates:
            identity = (task, _candidate_semantics(candidate))
            if identity not in source_kind_by_identity:
                source_kind_by_identity[identity] = (
                    "addition" if identity[1] not in before_keys else "incumbent"
                )

    for task in EXACT_TASKS:
        if task in normalized["planner_dominance"]:
            manifests[task]["planner_dominance"] = copy.deepcopy(
                normalized["planner_dominance"][task]
            )
        candidates = manifests[task]["candidates"]
        ids = [row["candidate_id"] for row in candidates]
        if len(ids) != len(set(ids)):
            raise ManifestBuildError(f"{task} output candidate IDs are not unique")
        if sum(row["kind"] == "planner" for row in candidates) != 1:
            raise ManifestBuildError(f"{task} output must contain exactly one planner")
        if candidates[0]["kind"] != "planner":
            raise ManifestBuildError(f"{task} output planner must remain index zero")
        if len({_candidate_semantics(row) for row in candidates}) != len(candidates):
            raise ManifestBuildError(
                f"{task} output candidate semantics were not de-duplicated"
            )
        for index, candidate in enumerate(candidates):
            source_kind = source_kind_by_identity[
                (task, _candidate_semantics(candidate))
            ]
            context.inventory_rows.append(
                _inventory_candidate(
                    task=task,
                    index=index,
                    candidate=candidate,
                    source_kind=source_kind,
                )
            )

    unused = sorted(set(range(len(overrides))) - used_overrides)
    if unused:
        raise ManifestBuildError(f"unused provenance overrides at indices {unused}")
    summary = {
        "release_id": "RLD2",
        "task_count": len(manifests),
        "candidate_count": {
            task: len(manifests[task]["candidates"]) for task in EXACT_TASKS
        },
        "deduplicated_count": len(context.deduplicated),
        "production": production,
    }
    return manifests, summary


def _inputs_lines(input_files: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for row in sorted(
        input_files, key=lambda value: (value["role"], value["logical_id"])
    ):
        fields = (str(row["role"]), str(row["logical_id"]), str(row["path"]))
        if any(any(character in field for character in "\r\n\t") for field in fields):
            raise ManifestBuildError(
                "input inventory fields must not contain control separators"
            )
        label = "\t".join(fields)
        rows.append(f"{row['sha256']}  {label}")
    return "\n".join(rows) + "\n"


def _write_release(
    staging: Path,
    *,
    manifests: Mapping[str, Mapping[str, Any]],
    context: BuildContext,
    spec_sha256: str,
    spec: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    manifest_hashes = {}
    for task in EXACT_TASKS:
        path = staging / task / "candidate_manifest.json"
        _write_json(path, manifests[task])
        manifest_hashes[task] = _sha256(path)
    inventory_path = staging / "input_inventory.jsonl"
    inventory_text = "".join(
        _canonical_json(row) + "\n" for row in context.inventory_rows
    )
    inventory_path.write_text(inventory_text, encoding="utf-8")
    files = list(context.input_files.values())
    inputs_path = staging / "INPUTS.sha256"
    inputs_path.write_text(_inputs_lines(files), encoding="utf-8")
    release = {
        "schema_version": RELEASE_SCHEMA,
        "release_id": "RLD2",
        "candidate_schema_version": CANDIDATE_SCHEMA,
        "rlinf_commit": spec["rlinf_commit"],
        "benchmark_commit": spec["benchmark_commit"],
        "tasks": list(EXACT_TASKS),
        "task_manifest_sha256": manifest_hashes,
        "candidate_count": dict(summary["candidate_count"]),
        "deduplicated": context.deduplicated,
        "input_spec_sha256": spec_sha256,
        "input_inventory_sha256": _sha256(inventory_path),
        "inputs_sha256_sha256": _sha256(inputs_path),
        "production_validated": bool(summary["production"]),
    }
    release["payload_sha256"] = _payload_sha256(release)
    _write_json(staging / "release_manifest.json", release)


def build_release(
    *,
    old_manifest_root: Path,
    input_spec: Path,
    output_root: Path,
    path_maps: Sequence[PathMap] = (),
    dry_run: bool = False,
    production: bool = False,
) -> dict[str, Any]:
    """Build an exact-14 candidate release without modifying the old manifests."""

    old_manifest_root = old_manifest_root.resolve()
    input_spec = input_spec.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise ManifestBuildError(f"refusing to overwrite output root: {output_root}")
    if output_root == old_manifest_root or old_manifest_root in output_root.parents:
        raise ManifestBuildError(
            "output root must not be inside the immutable old-manifest root"
        )
    old_manifests = _discover_old_manifests(old_manifest_root)
    old_before = _snapshot_tree(old_manifests)
    spec_payload = _load_json(input_spec, "RLD2 input spec")
    spec_sha256 = _sha256(input_spec)
    context = BuildContext(
        path_maps=_ordered_path_maps(path_maps),
        file_hash_cache={},
        input_files={},
        inventory_rows=[],
        deduplicated=[],
    )
    _hash_input_file(
        str(input_spec),
        expected_sha256=spec_sha256,
        role="input-spec",
        logical_id="RLD2",
        context=context,
    )
    manifests, summary = _build_in_memory(
        old_manifests=old_manifests,
        spec=spec_payload,
        spec_path=input_spec,
        context=context,
        production=production,
    )
    old_after = _snapshot_tree(old_manifests)
    if old_before != old_after:
        raise ManifestBuildError(
            "immutable old candidate manifests changed during the build"
        )
    if dry_run:
        return {**summary, "status": "dry-run", "output_root": str(output_root)}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        _write_release(
            staging,
            manifests=manifests,
            context=context,
            spec_sha256=spec_sha256,
            spec=spec_payload,
            summary=summary,
        )
        validated = validate_release(staging, production=production)
        if validated["task_count"] != len(EXACT_TASKS):
            raise ManifestBuildError("staged RLD2 release did not validate as exact14")
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**summary, "status": "built", "output_root": str(output_root)}


def _read_inventory(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ManifestBuildError(f"cannot read input inventory: {error}") from error
    rows = []
    for index, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ManifestBuildError(
                f"invalid input inventory row {index}: {error}"
            ) from error
        if not isinstance(row, dict) or row.get("schema_version") != INVENTORY_SCHEMA:
            raise ManifestBuildError(f"input inventory row {index} schema mismatch")
        rows.append(row)
    return rows


def _parse_inputs_file(path: Path) -> list[dict[str, str]]:
    rows = []
    for index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if "  " not in line:
            raise ManifestBuildError(f"malformed INPUTS.sha256 line {index}")
        sha256, label = line.split("  ", 1)
        _require_sha256(sha256, f"INPUTS.sha256 line {index}")
        parts = label.split("\t")
        if len(parts) != 3:
            raise ManifestBuildError(f"malformed INPUTS.sha256 label at line {index}")
        role, logical_id, path_value = parts
        rows.append(
            {
                "role": role,
                "logical_id": logical_id,
                "path": path_value,
                "sha256": sha256,
            }
        )
    return rows


def validate_release(output_root: Path, *, production: bool = False) -> dict[str, Any]:
    """Validate an already-built release without rewriting any files."""

    output_root = output_root.resolve()
    release_path = output_root / "release_manifest.json"
    release = _load_json(release_path, "RLD2 release manifest")
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("release_id") != "RLD2"
    ):
        raise ManifestBuildError("RLD2 release manifest identity mismatch")
    stored_payload_sha = release.get("payload_sha256")
    unhashed = dict(release)
    unhashed.pop("payload_sha256", None)
    if stored_payload_sha != _payload_sha256(unhashed):
        raise ManifestBuildError("RLD2 release manifest payload hash mismatch")
    if tuple(release.get("tasks", [])) != EXACT_TASKS:
        raise ManifestBuildError("RLD2 release task inventory is not exact14")
    rlinf_commit = _require_commit(release.get("rlinf_commit"), "release RLinf commit")
    benchmark_commit = _require_commit(
        release.get("benchmark_commit"), "release benchmark commit"
    )
    if production and release.get("production_validated") is not True:
        raise ManifestBuildError("release was not built with production validation")

    discovered = {
        path.parent.name: path for path in output_root.rglob("candidate_manifest.json")
    }
    if set(discovered) != set(EXACT_TASKS):
        raise ManifestBuildError("candidate manifest directory is not exact14")
    expected_hashes = release.get("task_manifest_sha256")
    if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != set(
        EXACT_TASKS
    ):
        raise ManifestBuildError("release candidate-manifest hash inventory mismatch")
    expected_counts = release.get("candidate_count")
    if not isinstance(expected_counts, Mapping) or set(expected_counts) != set(
        EXACT_TASKS
    ):
        raise ManifestBuildError("release candidate-count inventory mismatch")

    expected_inventory: dict[tuple[str, int], dict[str, Any]] = {}
    inventory_path = output_root / "input_inventory.jsonl"
    if _sha256(inventory_path) != release.get("input_inventory_sha256"):
        raise ManifestBuildError("input inventory checksum mismatch")
    for row in _read_inventory(inventory_path):
        key = (row.get("task"), row.get("candidate_index"))
        if key in expected_inventory:
            raise ManifestBuildError(
                f"duplicate input inventory candidate identity {key}"
            )
        expected_inventory[key] = row

    actual_inventory_keys = set()
    for task in EXACT_TASKS:
        path = discovered[task]
        if _sha256(path) != expected_hashes[task]:
            raise ManifestBuildError(f"{task} candidate manifest hash mismatch")
        payload = _load_json(path, f"{task} candidate manifest")
        if (
            payload.get("schema_version") != CANDIDATE_SCHEMA
            or payload.get("task") != task
        ):
            raise ManifestBuildError(f"{task} candidate manifest schema/task mismatch")
        if payload.get("rlinf_commit") != rlinf_commit:
            raise ManifestBuildError(f"{task} evaluator RLinf commit mismatch")
        if payload.get("benchmark_commit") != benchmark_commit:
            raise ManifestBuildError(f"{task} benchmark commit mismatch")
        if production and not isinstance(payload.get("planner_dominance"), Mapping):
            raise ManifestBuildError(f"{task} production planner_dominance is missing")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != expected_counts[task]:
            raise ManifestBuildError(f"{task} candidate count mismatch")
        if not candidates or candidates[0].get("kind") != "planner":
            raise ManifestBuildError(f"{task} planner is not index zero")
        if sum(row.get("kind") == "planner" for row in candidates) != 1:
            raise ManifestBuildError(f"{task} must contain exactly one planner")
        ids = [row.get("candidate_id") for row in candidates]
        if any(not isinstance(item, str) or not item for item in ids) or len(
            ids
        ) != len(set(ids)):
            raise ManifestBuildError(f"{task} candidate ID inventory is invalid")
        semantic_keys = [_candidate_semantics(row) for row in candidates]
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ManifestBuildError(f"{task} candidate semantics are duplicated")
        for index, candidate in enumerate(candidates):
            provenance = _validate_provenance(
                candidate.get("provenance"),
                candidate=candidate,
                task=task,
                benchmark_commit=benchmark_commit,
                production=production,
            )
            if candidate.get("kind") == "policy":
                policy_path = candidate.get("policy_path")
                policy_sha = _require_sha256(
                    candidate.get("policy_sha256"),
                    f"{task}/{candidate.get('candidate_id')} policy",
                )
                if (
                    not isinstance(policy_path, str)
                    or _sha256(Path(policy_path)) != policy_sha
                ):
                    raise ManifestBuildError(
                        f"policy hash mismatch for {task}/{candidate.get('candidate_id')}"
                    )
            inventory = expected_inventory.get((task, index))
            if inventory is None:
                raise ManifestBuildError(
                    f"input inventory is missing {task} candidate index {index}"
                )
            if inventory.get("candidate_id") != candidate.get("candidate_id"):
                raise ManifestBuildError(
                    f"input inventory candidate ID mismatch for {task}/{index}"
                )
            if inventory.get("semantics") != _candidate_semantics_payload(candidate):
                raise ManifestBuildError(
                    f"input inventory semantics mismatch for {task}/{index}"
                )
            if inventory.get("provenance") != provenance:
                raise ManifestBuildError(
                    f"input inventory provenance mismatch for {task}/{index}"
                )
            actual_inventory_keys.add((task, index))
    if actual_inventory_keys != set(expected_inventory):
        raise ManifestBuildError(
            "input inventory contains candidates outside exact14 manifests"
        )

    inputs_path = output_root / "INPUTS.sha256"
    if _sha256(inputs_path) != release.get("inputs_sha256_sha256"):
        raise ManifestBuildError("INPUTS.sha256 checksum mismatch")
    for row in _parse_inputs_file(inputs_path):
        path = Path(row["path"])
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ManifestBuildError(
                f"input hash mismatch for {row['role']}:{row['logical_id']}"
            )
    return {
        "status": "validated",
        "release_id": "RLD2",
        "task_count": len(EXACT_TASKS),
        "candidate_count": dict(expected_counts),
        "production": production,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest-root", type=Path)
    parser.add_argument("--input-spec", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--path-map",
        action="append",
        type=_parse_path_map,
        default=[],
        metavar="SOURCE=TARGET",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--production", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.validate_only:
        if (
            args.dry_run
            or args.old_manifest_root is not None
            or args.input_spec is not None
        ):
            raise ManifestBuildError(
                "--validate-only accepts only --output-root and optional --production"
            )
        result = validate_release(args.output_root, production=args.production)
    else:
        if args.old_manifest_root is None or args.input_spec is None:
            raise ManifestBuildError(
                "build mode requires --old-manifest-root and --input-spec"
            )
        result = build_release(
            old_manifest_root=args.old_manifest_root,
            input_spec=args.input_spec,
            output_root=args.output_root,
            path_maps=args.path_map,
            dry_run=args.dry_run,
            production=args.production,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
