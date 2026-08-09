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

"""Fail-closed admission of a trainer-selected learned policy checkpoint."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

POLICY_SCHEMA = "rlinf-dynamic-benchmark-expert-policy-v0.1"
TRAINER_SUMMARY_SCHEMA = "rlinf-dynamic-benchmark-expert-summary-v0.1"
CHECKPOINT_SELECTION_SCHEMA = "rlinf-dynamic-benchmark-checkpoint-selection-v0.1"
CHECKPOINT_SELECTION_RUN_SCHEMA = (
    "rlinf-dynamic-benchmark-checkpoint-selection-run-v0.1"
)
CHECKPOINT_SELECTION_OUTCOME_SCHEMA = (
    "rlinf-dynamic-benchmark-checkpoint-selection-outcome-v0.1"
)
CHECKPOINT_SELECTION_OUTCOME_FILENAME = "checkpoint_selection_outcome.json"
SOURCE_SNAPSHOT_SCHEMA = "rld2-qa-source-snapshot-v0.1"
SOURCE_SNAPSHOT_MANIFEST_FILENAME = "source_manifest.json"

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_METRIC_NAMES = (
    "success_rate",
    "safety_failure_rate",
    "mean_completion",
    "mean_return",
    "mean_duration_steps",
    "mean_action_l2_sum",
)


@dataclass(frozen=True)
class _AuthenticatedSource:
    path: Path
    module: str
    repository_path: str
    sha256: str
    git_blob_sha1: str
    content: bytes

    def public_identity(self) -> dict[str, str]:
        return {
            "module": self.module,
            "repository_path": self.repository_path,
            "sha256": self.sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    try:
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not finite canonical JSON") from error
    return payload


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys do not match its canonical schema")


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


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


def _json_safe_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not finite canonical JSON") from error


def _canonical_artifact(path: Path, name: str, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = path.resolve(strict=True)
    if resolved.name != name:
        raise ValueError(f"{label} must be the canonical trainer artifact {name}")
    return resolved


def _verify_payload_hash(
    payload: Mapping[str, Any],
    *,
    label: str,
    canonical_sha256: Any,
) -> str:
    claimed = _require_sha256(payload.get("payload_sha256"), f"{label} payload SHA-256")
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("payload_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError(f"{label} payload SHA-256 does not recompute")
    return claimed


def _canonical_json_sha256(value: Any) -> str:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("value is not finite canonical JSON") from error
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _canonical_json_equal(left: Any, right: Any) -> bool:
    return _canonical_json_sha256(left) == _canonical_json_sha256(right)


def _selector_metrics_projection(
    metrics: Mapping[str, Any], label: str
) -> dict[str, float]:
    projection: dict[str, float] = {}
    for metric_name in _SELECTION_METRIC_NAMES:
        metric_value = metrics.get(metric_name)
        if not isinstance(metric_value, float) or not math.isfinite(metric_value):
            raise ValueError(f"{label} selector metrics must be finite floats")
        projection[metric_name] = metric_value
    return projection


def _reject_unsafe_path_string(value: str, label: str) -> None:
    windows_path = PureWindowsPath(value)
    if (
        PurePosixPath(value).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
    ):
        raise ValueError(f"{label} contains an absolute path")
    if "\x00" in value or "\\" in value:
        raise ValueError(f"{label} contains an unsafe path string")
    if "/" in value or value in {".", ".."}:
        relative = PurePosixPath(value)
        if (
            relative.as_posix() != value
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"{label} contains an unsafe path string")


def _reject_unsafe_path_strings(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string mapping key")
            _reject_unsafe_path_string(key, f"{label} mapping key")
            _reject_unsafe_path_strings(item, f"{label}[{key!r}]")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_unsafe_path_strings(item, f"{label}[{index}]")
        return
    if isinstance(value, str):
        _reject_unsafe_path_string(value, label)


def _canonical_run_root(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("trainer run root must not be a symlink")
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    return candidate.resolve(strict=True)


def _safe_posix_parts(value: Any, label: str) -> tuple[str, ...]:
    rendered = _require_string(value, label)
    if "\\" in rendered:
        raise ValueError(f"{label} must be a safe POSIX relative path")
    relative = PurePosixPath(rendered)
    if (
        relative.is_absolute()
        or relative.as_posix() != rendered
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} must be a safe POSIX relative path")
    return relative.parts


def _reject_symlink_chain(root: Path, path: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its declared root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")


def _canonical_run_file(root: Path, relative_path: str, label: str) -> Path:
    parts = _safe_posix_parts(relative_path, f"{label} path")
    path = root.joinpath(*parts)
    _reject_symlink_chain(root, path, label)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} path is not canonical")
    return path


def _absent_run_file(root: Path, relative_path: str, label: str) -> Path:
    parts = _safe_posix_parts(relative_path, f"{label} path")
    path = root.joinpath(*parts)
    _reject_symlink_chain(root, path, label)
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be absent")
    return path


def _match_expected_sha256(path: Path, expected: Any, label: str) -> str:
    expected_sha256 = _require_sha256(expected, f"expected {label} SHA-256")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def _require_image_id(value: Any, label: str) -> str:
    rendered = _require_string(value, label)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", rendered) is None:
        raise ValueError(f"{label} must be an immutable Docker image ID")
    return rendered


def _snapshot_source_rows(
    source: Mapping[str, Any], label: str
) -> tuple[Path, str, str, dict[str, Mapping[str, Any]]]:
    _require_exact_keys(
        source,
        {"root", "commit", "tree", "files", "inventory_sha256"},
        label,
    )
    root_text = _require_string(source.get("root"), f"{label} root")
    root = Path(root_text)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root must be an absolute regular directory")
    root = root.resolve(strict=True)
    if str(root) != root_text:
        raise ValueError(f"{label} root path is not canonical")
    commit = _require_commit(source.get("commit"), f"{label} commit")
    tree = _require_commit(source.get("tree"), f"{label} tree")
    rows = _require_sequence(source.get("files"), f"{label} files")
    indexed: dict[str, Mapping[str, Any]] = {}
    canonical_rows: list[dict[str, str]] = []
    previous = ""
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"{label} file row {index}")
        _require_exact_keys(
            row,
            {"path", "mode", "git_blob_sha1", "sha256"},
            f"{label} file row {index}",
        )
        path = _require_string(row.get("path"), f"{label} file path")
        _safe_posix_parts(path, f"{label} file path")
        if path <= previous or path in indexed:
            raise ValueError(f"{label} file inventory is not strictly sorted")
        previous = path
        mode = _require_string(row.get("mode"), f"{label} file mode")
        if mode not in {"100644", "100755"}:
            raise ValueError(f"{label} contains an unsupported Git file mode")
        git_blob_sha1 = _require_commit(
            row.get("git_blob_sha1"), f"{label} file Git blob"
        )
        sha256 = _require_sha256(row.get("sha256"), f"{label} file SHA-256")
        canonical_row = {
            "path": path,
            "mode": mode,
            "git_blob_sha1": git_blob_sha1,
            "sha256": sha256,
        }
        canonical_rows.append(canonical_row)
        indexed[path] = canonical_row
    if _require_sha256(
        source.get("inventory_sha256"), f"{label} inventory SHA-256"
    ) != _canonical_json_sha256(canonical_rows):
        raise ValueError(f"{label} inventory SHA-256 does not recompute")
    return root, commit, tree, indexed


def _snapshot_runtime_rows(
    dependency: Mapping[str, Any], label: str
) -> tuple[Path, str]:
    _require_exact_keys(
        dependency,
        {"root", "inventory_sha256"},
        label,
    )
    root_text = _require_string(dependency.get("root"), f"{label} root")
    root = Path(root_text)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root must be an absolute regular directory")
    root = root.resolve(strict=True)
    if str(root) != root_text:
        raise ValueError(f"{label} root path is not canonical")
    inventory_sha256 = _require_sha256(
        dependency.get("inventory_sha256"), f"{label} inventory SHA-256"
    )
    return root, inventory_sha256


def validate_source_snapshot_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str,
    expected_base_image_id: str | None = None,
    expected_sources: Mapping[str, tuple[Path, str, str | None]] | None = None,
    verify_inventory: bool = True,
) -> dict[str, Any]:
    """Validate the immutable image's exact source-tree inventory."""

    raw_path = Path(manifest_path)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError("source snapshot manifest must be a regular non-symlink file")
    manifest = raw_path.resolve(strict=True)
    if manifest.name != SOURCE_SNAPSHOT_MANIFEST_FILENAME:
        raise ValueError("source snapshot manifest filename is not canonical")
    manifest_sha256 = _match_expected_sha256(
        manifest, expected_sha256, "source snapshot manifest"
    )
    payload = _read_json(manifest, "source snapshot manifest")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "base_image_id",
            "sources",
            "runtime_dependencies",
            "payload_sha256",
        },
        "source snapshot manifest",
    )
    if payload.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA:
        raise ValueError("source snapshot manifest schema mismatch")
    _verify_payload_hash(
        payload,
        label="source snapshot manifest",
        canonical_sha256=_canonical_json_sha256,
    )
    base_image_id = _require_image_id(
        payload.get("base_image_id"), "source snapshot base image ID"
    )
    if expected_base_image_id is not None and base_image_id != _require_image_id(
        expected_base_image_id, "expected base image ID"
    ):
        raise ValueError("source snapshot base image ID mismatch")
    sources = _require_mapping(payload.get("sources"), "source snapshot sources")
    required_roles = {"policy_rlinf", "evaluator_rlinf", "benchmark"}
    _require_exact_keys(sources, required_roles, "source snapshot sources")
    if expected_sources is not None and set(expected_sources) != required_roles:
        raise ValueError("expected source snapshot roles do not match")

    for role in sorted(required_roles):
        source = _require_mapping(sources.get(role), f"source snapshot {role}")
        root, commit, tree, indexed = _snapshot_source_rows(
            source, f"source snapshot {role}"
        )
        if expected_sources is not None:
            expected_root, expected_commit, expected_tree = expected_sources[role]
            if (
                root != Path(expected_root).resolve(strict=True)
                or commit
                != _require_commit(
                    expected_commit, f"expected source snapshot {role} commit"
                )
                or (
                    expected_tree is not None
                    and tree
                    != _require_commit(
                        expected_tree, f"expected source snapshot {role} tree"
                    )
                )
            ):
                raise ValueError(f"source snapshot {role} identity mismatch")
        if not verify_inventory:
            continue
        observed_paths: set[str] = set()
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"source snapshot {role} contains a symlink")
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ValueError(
                    f"source snapshot {role} contains a non-regular artifact"
                )
            relative = candidate.relative_to(root).as_posix()
            observed_paths.add(relative)
            expected = indexed.get(relative)
            if expected is None:
                raise ValueError(f"source snapshot {role} contains an extra file")
            observed_mode = (
                "100755" if candidate.stat().st_mode & stat.S_IXUSR else "100644"
            )
            content = candidate.read_bytes()
            if (
                observed_mode != expected["mode"]
                or hashlib.sha256(content).hexdigest() != expected["sha256"]
                or _git_blob_sha1(content) != expected["git_blob_sha1"]
            ):
                raise ValueError(f"source snapshot {role} file identity mismatch")
        if observed_paths != set(indexed):
            raise ValueError(f"source snapshot {role} file inventory mismatch")
    dependencies = _require_mapping(
        payload.get("runtime_dependencies"), "source snapshot runtime dependencies"
    )
    _require_exact_keys(
        dependencies,
        {"portable", "a800_core"},
        "source snapshot runtime dependencies",
    )
    for name in ("portable", "a800_core"):
        dependency = _require_mapping(
            dependencies.get(name), f"source snapshot runtime {name}"
        )
        root, expected_inventory_sha256 = _snapshot_runtime_rows(
            dependency, f"source snapshot runtime {name}"
        )
        if not verify_inventory:
            continue
        observed_rows: list[dict[str, str]] = []
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"source snapshot runtime {name} contains a symlink")
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ValueError(
                    f"source snapshot runtime {name} contains a non-regular artifact"
                )
            relative = candidate.relative_to(root).as_posix()
            _safe_posix_parts(relative, f"source snapshot runtime {name} file path")
            observed_mode = (
                "100755" if candidate.stat().st_mode & stat.S_IXUSR else "100644"
            )
            observed_rows.append(
                {
                    "path": relative,
                    "mode": observed_mode,
                    "sha256": _sha256(candidate),
                }
            )
        observed_rows.sort(key=lambda row: row["path"])
        if _canonical_json_sha256(observed_rows) != expected_inventory_sha256:
            raise ValueError(f"source snapshot runtime {name} inventory mismatch")
    if _sha256(manifest) != manifest_sha256:
        raise RuntimeError("source snapshot manifest changed during validation")
    return copy.deepcopy(payload)


def _source_snapshot_from_environment() -> tuple[Path, str] | None:
    manifest = os.environ.get("RLD2_SOURCE_SNAPSHOT_MANIFEST")
    expected_sha256 = os.environ.get("RLD2_SOURCE_SNAPSHOT_MANIFEST_SHA256")
    if manifest is None and expected_sha256 is None:
        return None
    if not manifest or not expected_sha256:
        raise ValueError("source snapshot environment identity is incomplete")
    return Path(manifest), _require_sha256(
        expected_sha256, "source snapshot environment SHA-256"
    )


def _verify_snapshot_source_checkout(
    *,
    root: Path,
    expected_commit: str,
    sources: Mapping[str, tuple[str, str]],
    label: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[str, dict[str, _AuthenticatedSource]]:
    payload = validate_source_snapshot_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        verify_inventory=False,
    )
    snapshot_sources = _require_mapping(
        payload.get("sources"), "source snapshot sources"
    )
    resolved_root = Path(root).resolve(strict=True)
    declared_commit = _require_commit(expected_commit, f"expected {label} commit")
    matched: tuple[str, Mapping[str, Any], dict[str, Mapping[str, Any]]] | None = None
    for role, raw_source in snapshot_sources.items():
        source = _require_mapping(raw_source, f"source snapshot {role}")
        snapshot_root, commit, _, indexed = _snapshot_source_rows(
            source, f"source snapshot {role}"
        )
        if snapshot_root == resolved_root:
            matched = (str(role), source, indexed)
            if commit != declared_commit:
                raise ValueError(f"{label} snapshot commit mismatch")
            break
    if matched is None:
        raise ValueError(f"{label} root is absent from the source snapshot")
    _, _, indexed = matched
    authenticated: dict[str, _AuthenticatedSource] = {}
    for name, (module, repository_path) in sources.items():
        parts = _safe_posix_parts(repository_path, f"{label} {name} repository path")
        expected = indexed.get(repository_path)
        if expected is None:
            raise ValueError(f"{label} {name} is absent from the source snapshot")
        source_path = resolved_root.joinpath(*parts)
        _reject_symlink_chain(resolved_root, source_path, f"{label} {name} source")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        content = source_path.read_bytes()
        if (
            hashlib.sha256(content).hexdigest() != expected["sha256"]
            or _git_blob_sha1(content) != expected["git_blob_sha1"]
        ):
            raise ValueError(f"{label} {name} differs from its snapshot blob")
        authenticated[name] = _AuthenticatedSource(
            path=source_path,
            module=module,
            repository_path=repository_path,
            sha256=str(expected["sha256"]),
            git_blob_sha1=str(expected["git_blob_sha1"]),
            content=content,
        )
    return declared_commit, authenticated


def _run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    label: str,
    text: bool,
) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=text,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot inspect {label} Git source root") from error
    if result.returncode != 0:
        stderr = (
            result.stderr.strip() if text else result.stderr.decode(errors="replace")
        )
        raise ValueError(f"cannot inspect {label} Git source root: {stderr}")
    return result.stdout


def _verify_source_checkout(
    *,
    root: Path,
    expected_commit: str,
    sources: Mapping[str, tuple[str, str]],
    label: str,
) -> tuple[str, dict[str, _AuthenticatedSource]]:
    candidate = Path(root)
    if candidate.is_symlink():
        raise ValueError(f"{label} source root must not be a symlink")
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    resolved_root = candidate.resolve(strict=True)
    declared_commit = _require_commit(expected_commit, f"expected {label} commit")
    snapshot_environment = _source_snapshot_from_environment()
    if snapshot_environment is not None:
        manifest_path, manifest_sha256 = snapshot_environment
        return _verify_snapshot_source_checkout(
            root=resolved_root,
            expected_commit=declared_commit,
            sources=sources,
            label=label,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
    top_level = str(
        _run_git(
            resolved_root,
            ["rev-parse", "--show-toplevel"],
            label=label,
            text=True,
        )
    ).strip()
    if Path(top_level).resolve(strict=True) != resolved_root:
        raise ValueError(f"{label} source root must be the Git repository root")
    observed_commit = str(
        _run_git(
            resolved_root,
            ["rev-parse", "HEAD"],
            label=label,
            text=True,
        )
    ).strip()
    if observed_commit != declared_commit:
        raise ValueError(f"{label} source-root HEAD does not match expected commit")
    status = str(
        _run_git(
            resolved_root,
            ["status", "--porcelain=v1", "--untracked-files=all"],
            label=label,
            text=True,
        )
    )
    if status.strip():
        raise ValueError(f"{label} source root must be clean")

    authenticated: dict[str, _AuthenticatedSource] = {}
    for name, (module, repository_path) in sources.items():
        parts = _safe_posix_parts(repository_path, f"{label} {name} repository path")
        source_path = resolved_root.joinpath(*parts)
        _reject_symlink_chain(resolved_root, source_path, f"{label} {name} source")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        tracked = str(
            _run_git(
                resolved_root,
                ["ls-files", "--error-unmatch", repository_path],
                label=label,
                text=True,
            )
        ).strip()
        if tracked != repository_path:
            raise ValueError(f"{label} {name} source path is not canonical")
        git_blob_sha1 = str(
            _run_git(
                resolved_root,
                ["rev-parse", f"{observed_commit}:{repository_path}"],
                label=label,
                text=True,
            )
        ).strip()
        if _HEX_40.fullmatch(git_blob_sha1) is None:
            raise ValueError(f"{label} {name} source Git blob identity is invalid")
        committed_bytes = _run_git(
            resolved_root,
            ["show", f"{observed_commit}:{repository_path}"],
            label=label,
            text=False,
        )
        if _git_blob_sha1(committed_bytes) != git_blob_sha1:
            raise ValueError(f"{label} {name} committed Git blob identity mismatch")
        observed_bytes = source_path.read_bytes()
        if observed_bytes != committed_bytes:
            raise ValueError(f"{label} {name} source differs from its committed blob")
        authenticated[name] = _AuthenticatedSource(
            path=source_path,
            module=module,
            repository_path=repository_path,
            sha256=hashlib.sha256(observed_bytes).hexdigest(),
            git_blob_sha1=git_blob_sha1,
            content=observed_bytes,
        )
    return observed_commit, authenticated


def _load_authoritative_trainer(source_bytes: bytes, *, source_path: Path) -> Any:
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    module_name = f"_rlinf_policy_trainer_{source_sha256}"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(
            source_bytes,
            str(source_path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise ValueError(
            "cannot execute authoritative policy trainer source"
        ) from error
    required = (
        "_canonical_json_sha256",
        "_validate_checkpoint_selection_metrics",
        "_CheckpointSelectionLedger",
    )
    if any(not hasattr(module, name) for name in required):
        sys.modules.pop(module_name, None)
        raise ValueError("authoritative policy trainer lacks selector verifier symbols")
    return module


def _read_canonical_jsonl(path: Path, label: str) -> list[tuple[int, dict[str, Any]]]:
    try:
        raw = path.read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise ValueError(
                f"{label} must be a non-empty newline-terminated JSONL file"
            )
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{label} contains a blank line at {line_number}")
        try:
            payload = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{label} line {line_number} is malformed JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        payload = _json_safe_copy(payload, f"{label} line {line_number}")
        if line != json.dumps(payload, sort_keys=True):
            raise ValueError(
                f"{label} line {line_number} is not canonical trainer JSONL"
            )
        records.append((line_number, payload))
    return records


def _manifest_policy_identity(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": policy["path"],
        "sha256": policy["sha256"],
        "env_steps": policy["env_steps"],
        "validation_metrics_sha256": policy["validation_metrics_sha256"],
        "config_payload_sha256": policy["config_payload_sha256"],
        "state_schema_sha256": policy["state_schema_sha256"],
    }


def _validate_manifest_policy_identity(
    identity: Mapping[str, Any],
    *,
    expected_path: str,
    expected_env_steps: int,
    label: str,
) -> None:
    _require_exact_keys(
        identity,
        {
            "path",
            "sha256",
            "env_steps",
            "validation_metrics_sha256",
            "config_payload_sha256",
            "state_schema_sha256",
        },
        label,
    )
    if identity.get("path") != expected_path:
        raise ValueError(f"{label} path does not match")
    if (
        _require_int(identity.get("env_steps"), f"{label} env_steps")
        != expected_env_steps
    ):
        raise ValueError(f"{label} env_steps do not match")
    for field in (
        "sha256",
        "validation_metrics_sha256",
        "config_payload_sha256",
        "state_schema_sha256",
    ):
        _require_sha256(identity.get(field), f"{label} {field}")


def _verify_policy_artifact(
    *,
    root: Path,
    relative_path: str,
    manifest_identity: Mapping[str, Any],
    expected_env_steps: int,
    expected_validation: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    expected_infra_identity: Any,
    expected_file_sha256: str | None,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from torch import load as torch_load

    _validate_manifest_policy_identity(
        manifest_identity,
        expected_path=relative_path,
        expected_env_steps=expected_env_steps,
        label=f"{label} manifest identity",
    )
    path = _canonical_run_file(root, relative_path, label)
    file_sha256 = _sha256(path)
    if expected_file_sha256 is not None and file_sha256 != _require_sha256(
        expected_file_sha256, f"expected {label} SHA-256"
    ):
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        raw_checkpoint = torch_load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"cannot load {label} {path}") from error
    checkpoint = _require_mapping(raw_checkpoint, label)
    if checkpoint.get("schema_version") != POLICY_SCHEMA:
        raise ValueError(f"{label} schema does not match")
    _require_exact_keys(
        checkpoint,
        {
            "schema_version",
            "config",
            "model",
            "normalizer",
            "state_schema",
            "infra_identity",
            "validation",
            "env_steps",
        },
        label,
    )
    env_steps = _require_int(checkpoint.get("env_steps"), f"{label} env_steps")
    if env_steps != expected_env_steps:
        raise ValueError(f"{label} env_steps do not match")
    config = _json_safe_copy(
        _require_mapping(checkpoint.get("config"), f"{label} config"),
        f"{label} config",
    )
    if not _canonical_json_equal(config, expected_config):
        raise ValueError(f"{label} config does not match trainer config")
    state_schema = _json_safe_copy(
        _require_mapping(checkpoint.get("state_schema"), f"{label} state schema"),
        f"{label} state schema",
    )
    validation = _json_safe_copy(
        _require_mapping(checkpoint.get("validation"), f"{label} validation"),
        f"{label} validation",
    )
    if not _canonical_json_equal(validation, expected_validation):
        raise ValueError(f"{label} validation does not match selector evidence")
    if "infra_identity" not in checkpoint:
        raise ValueError(f"{label} infra identity is missing")
    infra_identity = checkpoint["infra_identity"]
    if infra_identity is not None and not isinstance(infra_identity, Mapping):
        raise ValueError(f"{label} infra identity must be a mapping or null")
    infra_identity = _json_safe_copy(infra_identity, f"{label} infra identity")
    if not _canonical_json_equal(infra_identity, expected_infra_identity):
        raise ValueError(f"{label} infra identity does not match trainer summary")

    identity = {
        "path": relative_path,
        "sha256": file_sha256,
        "schema_version": POLICY_SCHEMA,
        "env_steps": env_steps,
        "validation_metrics_sha256": _canonical_json_sha256(validation),
        "config_payload_sha256": _canonical_json_sha256(config),
        "state_schema_sha256": _canonical_json_sha256(state_schema),
        "infra_identity_sha256": _canonical_json_sha256(infra_identity),
    }
    if not _canonical_json_equal(
        manifest_identity, _manifest_policy_identity(identity)
    ):
        raise ValueError(f"{label} manifest identity does not match file contents")
    if _sha256(path) != file_sha256:
        raise RuntimeError(f"{label} changed while it was being verified")
    return identity, state_schema


def _validation_evidence(
    *,
    metrics_path: str,
    line_number: int,
    event: Mapping[str, Any],
    validation_metrics_sha256: str,
    role: str,
    checkpoint_selection_manifest_payload_sha256: str | None,
    checkpoint_snapshot_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "metrics_path": metrics_path,
        "line_number": line_number,
        "event_payload_sha256": _canonical_json_sha256(event),
        "validation_metrics_sha256": validation_metrics_sha256,
        "role": role,
        "checkpoint_selection_manifest_payload_sha256": (
            checkpoint_selection_manifest_payload_sha256
        ),
        "checkpoint_snapshot_identity": (
            None
            if checkpoint_snapshot_identity is None
            else copy.deepcopy(dict(checkpoint_snapshot_identity))
        ),
    }


def build_checkpoint_selection_outcome_payload(
    *,
    run_root: Path,
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
) -> dict[str, Any]:
    """Reopen and seal the trainer's complete checkpoint-selection outcome.

    This admits neither branch as a learned policy. It authenticates the trainer's
    authoritative selected-snapshot or no-eligible fallback decision so later
    stages can apply their own explicitly narrower admission rule.
    """

    policy_source_specs = {
        "trainer_source": (
            "examples.embodiment.train_dynamic_benchmark_expert",
            "examples/embodiment/train_dynamic_benchmark_expert.py",
        )
    }
    verifier_source_specs = {
        "verifier_source": (
            "examples.embodiment.dynamic_benchmark_checkpoint_admission",
            "examples/embodiment/dynamic_benchmark_checkpoint_admission.py",
        ),
        "builder_source": (
            "examples.embodiment.build_dynamic_benchmark_checkpoint_selection_outcome",
            "examples/embodiment/build_dynamic_benchmark_checkpoint_selection_outcome.py",
        ),
    }
    evaluator_source_specs = {
        "evaluator_source": (
            "examples.embodiment.evaluate_dynamic_benchmark_expert",
            "examples/embodiment/evaluate_dynamic_benchmark_expert.py",
        ),
    }
    policy_commit, policy_authenticated = _verify_source_checkout(
        root=policy_rlinf_source_root,
        expected_commit=expected_policy_rlinf_commit,
        sources=policy_source_specs,
        label="policy RLinf",
    )
    verifier_commit, verifier_authenticated = _verify_source_checkout(
        root=verifier_rlinf_source_root,
        expected_commit=expected_verifier_rlinf_commit,
        sources=verifier_source_specs,
        label="verifier RLinf",
    )
    evaluator_commit, evaluator_authenticated = _verify_source_checkout(
        root=evaluator_rlinf_source_root,
        expected_commit=expected_evaluator_rlinf_commit,
        sources=evaluator_source_specs,
        label="evaluator RLinf",
    )
    policy_sources = {
        name: source.public_identity() for name, source in policy_authenticated.items()
    }
    verifier_sources = {
        name: source.public_identity()
        for name, source in verifier_authenticated.items()
    }
    evaluator_sources = {
        name: source.public_identity()
        for name, source in evaluator_authenticated.items()
    }
    trainer_source = policy_authenticated["trainer_source"]
    trainer = _load_authoritative_trainer(
        trainer_source.content, source_path=trainer_source.path
    )
    raw_executing_sources = {
        "verifier_source": Path(__file__),
        "builder_source": Path(__file__).with_name(
            "build_dynamic_benchmark_checkpoint_selection_outcome.py"
        ),
    }
    executing_sources: dict[str, Path] = {}
    for name, raw_path in raw_executing_sources.items():
        if raw_path.is_symlink():
            raise ValueError(f"executing {name} must not be a symlink")
        executing_sources[name] = raw_path.resolve(strict=True)
    declared_sources = {
        **policy_authenticated,
        **verifier_authenticated,
        **evaluator_authenticated,
    }
    for name, executing_path in executing_sources.items():
        if executing_path.read_bytes() != declared_sources[name].content:
            raise ValueError(
                f"executing {name} does not match the declared clean source root"
            )

    root = _canonical_run_root(run_root)
    summary_path = _canonical_run_file(root, "summary.json", "trainer summary")
    selection_path = _canonical_run_file(
        root, "checkpoint_selection.json", "checkpoint-selection manifest"
    )
    config_path = _canonical_run_file(root, "config.json", "trainer config")
    metrics_path = _canonical_run_file(root, "metrics.jsonl", "trainer metrics")
    initial_policy_path = _canonical_run_file(
        root, "initial_policy.pt", "initial planner policy"
    )
    observed_file_hashes: dict[Path, str] = {
        summary_path: _match_expected_sha256(
            summary_path, expected_summary_sha256, "trainer summary"
        ),
        selection_path: _match_expected_sha256(
            selection_path,
            expected_checkpoint_selection_sha256,
            "checkpoint-selection manifest",
        ),
        config_path: _match_expected_sha256(
            config_path, expected_config_sha256, "trainer config"
        ),
        metrics_path: _match_expected_sha256(
            metrics_path, expected_metrics_sha256, "trainer metrics"
        ),
        initial_policy_path: _match_expected_sha256(
            initial_policy_path,
            expected_initial_policy_sha256,
            "initial planner policy",
        ),
    }

    summary = _read_json(summary_path, "trainer summary")
    manifest = _read_json(selection_path, "checkpoint-selection manifest")
    config = _read_json(config_path, "trainer config")
    if summary.get("schema_version") != TRAINER_SUMMARY_SCHEMA:
        raise ValueError("trainer summary schema does not match")
    summary_payload_sha256 = _verify_payload_hash(
        summary,
        label="trainer summary",
        canonical_sha256=trainer._canonical_json_sha256,
    )
    if summary.get("status") != "complete":
        raise ValueError("checkpoint-selection outcome requires complete training")
    summary_env_steps = _require_int(
        summary.get("env_steps"), "trainer summary env_steps", minimum=1
    )
    summary_update_steps = _require_int(
        summary.get("update_steps"), "trainer summary update_steps"
    )
    if manifest.get("schema_version") != CHECKPOINT_SELECTION_SCHEMA:
        raise ValueError("checkpoint-selection manifest schema does not match")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "selector",
            "run_identity",
            "matched_planner_baseline",
            "evaluated_snapshots",
            "selection",
            "payload_sha256",
        },
        "checkpoint-selection manifest",
    )
    selection_payload_sha256 = _verify_payload_hash(
        manifest,
        label="checkpoint-selection manifest",
        canonical_sha256=trainer._canonical_json_sha256,
    )

    summary_config = _json_safe_copy(
        _require_mapping(summary.get("config"), "trainer summary config"),
        "trainer summary config",
    )
    if not _canonical_json_equal(summary_config, config):
        raise ValueError("trainer summary config does not match config.json")
    config_payload_sha256 = trainer._canonical_json_sha256(config)
    config_file_sha256 = observed_file_hashes[config_path]
    if summary.get("config_sha256") != config_payload_sha256:
        raise ValueError("trainer summary config payload SHA-256 does not recompute")
    if summary.get("config_file_sha256") != config_file_sha256:
        raise ValueError("trainer summary config file SHA-256 does not recompute")
    if "infra_identity" not in summary:
        raise ValueError("trainer summary infra identity is missing")
    infra_identity = summary["infra_identity"]
    if infra_identity is not None and not isinstance(infra_identity, Mapping):
        raise ValueError("trainer summary infra identity must be a mapping or null")
    infra_identity = _json_safe_copy(infra_identity, "trainer summary infra identity")
    infra_identity_sha256 = trainer._canonical_json_sha256(infra_identity)

    task = _require_string(config.get("task"), "trainer task")
    algorithm = _require_string(config.get("algorithm"), "trainer algorithm")
    if algorithm != "residual_rlpd":
        raise ValueError("checkpoint selection requires residual_rlpd")
    config_policy_commit = _require_commit(
        config.get("rlinf_commit"), "trainer policy RLinf commit"
    )
    if config_policy_commit != policy_commit:
        raise ValueError("trainer config does not match policy RLinf source commit")
    benchmark_commit = _require_commit(
        config.get("benchmark_commit"), "trainer benchmark commit"
    )
    if benchmark_commit != _require_commit(
        expected_benchmark_commit, "expected benchmark commit"
    ):
        raise ValueError("trainer config does not match expected benchmark commit")
    training_seed = _require_int(config.get("seed"), "trainer seed")
    validation_manifest_seed = _require_int(
        config.get("validation_manifest_seed"), "validation manifest seed"
    )
    eval_episodes = _require_int(
        config.get("eval_episodes"), "evaluation episodes", minimum=1
    )
    eval_num_envs = _require_int(
        config.get("eval_num_envs"), "evaluation environment count", minimum=1
    )

    baseline = _require_mapping(
        manifest.get("matched_planner_baseline"), "matched planner baseline"
    )
    _require_exact_keys(
        baseline,
        {
            "source",
            "safety_failure_rate_ceiling",
            "validation_metrics",
            "validation_metrics_sha256",
            "policy",
        },
        "matched planner baseline",
    )
    baseline_metrics = _json_safe_copy(
        _require_mapping(
            baseline.get("validation_metrics"), "matched planner validation metrics"
        ),
        "matched planner validation metrics",
    )
    trainer._validate_checkpoint_selection_metrics(baseline_metrics)
    baseline_selector_metrics = _selector_metrics_projection(
        baseline_metrics, "matched planner"
    )
    baseline_selector_metrics_sha256 = trainer._canonical_json_sha256(
        baseline_selector_metrics
    )
    baseline_metrics_sha256 = trainer._canonical_json_sha256(baseline_metrics)
    if baseline.get("validation_metrics_sha256") != baseline_metrics_sha256:
        raise ValueError("matched planner validation identity does not recompute")
    safety_ceiling = baseline.get("safety_failure_rate_ceiling")
    if (
        not isinstance(safety_ceiling, float)
        or not math.isfinite(float(safety_ceiling))
        or float(safety_ceiling)
        != float(baseline_selector_metrics["safety_failure_rate"])
    ):
        raise ValueError("matched planner safety ceiling is not canonical")
    baseline_policy_manifest = _require_mapping(
        baseline.get("policy"), "matched planner policy identity"
    )
    if baseline_policy_manifest.get("path") != "initial_policy.pt":
        raise ValueError("matched planner policy is not initial_policy.pt")
    initial_policy, state_schema = _verify_policy_artifact(
        root=root,
        relative_path="initial_policy.pt",
        manifest_identity=baseline_policy_manifest,
        expected_env_steps=0,
        expected_validation=baseline_metrics,
        expected_config=config,
        expected_infra_identity=infra_identity,
        expected_file_sha256=expected_initial_policy_sha256,
        label="initial planner policy",
    )
    state_schema_sha256 = trainer._canonical_json_sha256(state_schema)
    expected_run_identity = {
        "schema_version": CHECKPOINT_SELECTION_RUN_SCHEMA,
        "task": task,
        "algorithm": algorithm,
        "rlinf_commit": policy_commit,
        "benchmark_commit": benchmark_commit,
        "seed": training_seed,
        "validation_manifest_seed": validation_manifest_seed,
        "eval_episodes": eval_episodes,
        "eval_num_envs": eval_num_envs,
        "config_sha256": config_file_sha256,
        "config_payload_sha256": config_payload_sha256,
        "state_schema_sha256": state_schema_sha256,
    }
    if manifest.get("run_identity") != expected_run_identity:
        raise ValueError("checkpoint-selection run identity does not match artifacts")

    raw_rows = manifest.get("evaluated_snapshots")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(
            "checkpoint-selection outcome requires a non-empty learned snapshot ledger"
        )
    rows: list[Mapping[str, Any]] = []
    snapshot_paths: dict[int, Path] = {}
    for index, raw_row in enumerate(raw_rows):
        row = _require_mapping(raw_row, f"checkpoint-selection row {index}")
        _require_exact_keys(
            row,
            {
                "env_steps",
                "policy",
                "validation_metrics",
                "validation_metrics_sha256",
                "eligible",
                "eligibility_reason",
                "selection_score",
                "selected",
            },
            f"checkpoint-selection row {index}",
        )
        env_steps = _require_int(
            row.get("env_steps"),
            f"checkpoint-selection row {index} env_steps",
            minimum=1,
        )
        policy_manifest = _require_mapping(
            row.get("policy"), f"checkpoint-selection row {index} policy"
        )
        expected_path = f"policy_snapshots/policy_step_{env_steps:012d}.pt"
        if policy_manifest.get("path") != expected_path:
            raise ValueError("checkpoint-selection snapshot path is not canonical")
        if not isinstance(row.get("eligible"), bool) or not isinstance(
            row.get("selected"), bool
        ):
            raise ValueError("checkpoint-selection row markers must be boolean")
        selection_score = row.get("selection_score")
        if not isinstance(selection_score, list) or len(selection_score) != 6:
            raise ValueError("checkpoint-selection score must be a six-number array")
        for value in selection_score:
            if not isinstance(value, float) or not math.isfinite(float(value)):
                raise ValueError(
                    "checkpoint-selection score must be a six-number array"
                )
        snapshot_path = _canonical_run_file(
            root, expected_path, f"checkpoint-selection snapshot {env_steps}"
        )
        snapshot_paths[env_steps] = snapshot_path
        observed_file_hashes[snapshot_path] = _sha256(snapshot_path)
        rows.append(row)
    snapshots_dir = root / "policy_snapshots"
    _reject_symlink_chain(root, snapshots_dir, "policy snapshot directory")
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(snapshots_dir)
    expected_snapshot_files = set(snapshot_paths.values())
    observed_snapshot_files: set[Path] = set()
    for child in snapshots_dir.iterdir():
        if child.is_symlink():
            raise ValueError("policy snapshot directory contains a symlink")
        if not child.is_file():
            raise ValueError("policy snapshot directory contains a non-file artifact")
        observed_snapshot_files.add(child)
    if observed_snapshot_files != expected_snapshot_files:
        raise ValueError("policy snapshot directory inventory does not match ledger")
    if int(rows[-1]["env_steps"]) != summary_env_steps:
        raise ValueError("trainer summary does not end at the final selector snapshot")

    selection = _require_mapping(
        manifest.get("selection"), "checkpoint-selection result"
    )
    _require_exact_keys(
        selection,
        {
            "status",
            "eligible_snapshot_count",
            "selected_snapshot_identity",
            "best_policy",
            "planner_fallback_policy",
        },
        "checkpoint-selection result",
    )
    selection_status = selection.get("status")
    if selection_status not in {
        "selected_eligible_snapshot",
        "planner_fallback_no_eligible",
    }:
        raise ValueError("checkpoint-selection status is not recognized")
    eligible_count = _require_int(
        selection.get("eligible_snapshot_count"),
        "checkpoint-selection eligible snapshot count",
    )
    best_path = root / "best_policy.pt"
    if selection_status == "selected_eligible_snapshot":
        if eligible_count < 1:
            raise ValueError("selected outcome requires an eligible snapshot")
        if expected_best_policy_sha256 is None:
            raise ValueError(
                "selected outcome requires expected best_policy.pt SHA-256"
            )
        best_path = _canonical_run_file(root, "best_policy.pt", "best policy")
        observed_file_hashes[best_path] = _match_expected_sha256(
            best_path, expected_best_policy_sha256, "best policy"
        )
    else:
        if expected_best_policy_sha256 is not None:
            raise ValueError(
                "planner fallback forbids an expected best_policy.pt SHA-256"
            )
        _absent_run_file(root, "best_policy.pt", "fallback best policy")

    ledger = trainer._CheckpointSelectionLedger(
        root, expected_run_identity, copy.deepcopy(manifest)
    )

    expected_summary_reference = {
        "manifest_path": "checkpoint_selection.json",
        "manifest_payload_sha256": selection_payload_sha256,
        "status": selection_status,
        "eligible_snapshot_count": eligible_count,
        "selected_snapshot_identity": selection.get("selected_snapshot_identity"),
        "planner_fallback_policy": selection.get("planner_fallback_policy"),
    }
    if not _canonical_json_equal(
        summary.get("checkpoint_selection"), expected_summary_reference
    ):
        raise ValueError(
            "trainer summary and checkpoint-selection manifest identities diverged"
        )

    final_validation = _json_safe_copy(
        _require_mapping(summary.get("final_validation"), "final validation"),
        "final validation",
    )
    trainer._validate_checkpoint_selection_metrics(final_validation)
    final_row = rows[-1]
    if not _canonical_json_equal(final_validation, final_row.get("validation_metrics")):
        raise ValueError("trainer summary final validation is not the final snapshot")
    final_validation_sha256 = trainer._canonical_json_sha256(final_validation)

    selected_rows = [row for row in rows if row.get("selected") is True]
    eligible_rows = [row for row in rows if row.get("eligible") is True]
    best_validation = summary.get("best_validation")
    best_score = summary.get("best_score")
    selected_row: Mapping[str, Any] | None
    if selection_status == "planner_fallback_no_eligible":
        if (
            eligible_count != 0
            or eligible_rows
            or selected_rows
            or selection.get("selected_snapshot_identity") is not None
            or selection.get("best_policy") is not None
            or best_validation is not None
            or best_score is not None
        ):
            raise ValueError("planner fallback contains learned-selection evidence")
        selected_row = None
    else:
        if eligible_count != len(eligible_rows):
            raise ValueError("eligible snapshot count does not match snapshot ledger")
        if len(selected_rows) != 1 or selected_rows[0].get("eligible") is not True:
            raise ValueError("selected outcome must mark exactly one eligible snapshot")
        selected_row = selected_rows[0]
        if not _canonical_json_equal(
            best_validation, selected_row.get("validation_metrics")
        ):
            raise ValueError("trainer summary best validation is not selected snapshot")
        if best_score is None or not _canonical_json_equal(
            list(_require_sequence(best_score, "best score")),
            list(
                _require_sequence(selected_row.get("selection_score"), "selected score")
            ),
        ):
            raise ValueError("trainer summary best score is not selected snapshot")

    enriched_rows: list[dict[str, Any]] = []
    snapshot_identities: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        env_steps = int(row["env_steps"])
        validation = _json_safe_copy(
            _require_mapping(
                row.get("validation_metrics"),
                f"checkpoint-selection row {index} validation",
            ),
            f"checkpoint-selection row {index} validation",
        )
        trainer._validate_checkpoint_selection_metrics(validation)
        selector_metrics = _selector_metrics_projection(
            validation, f"checkpoint-selection row {index}"
        )
        selector_metrics_sha256 = trainer._canonical_json_sha256(selector_metrics)
        policy_manifest = _require_mapping(
            row.get("policy"), f"checkpoint-selection row {index} policy"
        )
        policy, snapshot_state_schema = _verify_policy_artifact(
            root=root,
            relative_path=str(policy_manifest["path"]),
            manifest_identity=policy_manifest,
            expected_env_steps=env_steps,
            expected_validation=validation,
            expected_config=config,
            expected_infra_identity=infra_identity,
            expected_file_sha256=None,
            label=f"checkpoint-selection snapshot {env_steps}",
        )
        if trainer._canonical_json_sha256(snapshot_state_schema) != state_schema_sha256:
            raise ValueError("snapshot state schema does not match initial policy")
        snapshot_identities[env_steps] = policy
        if policy["sha256"] != observed_file_hashes[snapshot_paths[env_steps]]:
            raise RuntimeError("snapshot changed during outcome verification")
        enriched_rows.append(
            {
                "env_steps": env_steps,
                "policy": policy,
                "validation_metrics": validation,
                "validation_metrics_sha256": row["validation_metrics_sha256"],
                "selector_metrics": selector_metrics,
                "selector_metrics_sha256": selector_metrics_sha256,
                "validation_evidence": None,
                "eligible": row["eligible"],
                "eligibility_reason": row["eligibility_reason"],
                "selection_score": copy.deepcopy(row["selection_score"]),
                "selected": row["selected"],
            }
        )

    best_policy: dict[str, Any] | None = None
    if selected_row is not None:
        best_manifest = _require_mapping(
            selection.get("best_policy"), "selected best policy identity"
        )
        best_policy, best_state_schema = _verify_policy_artifact(
            root=root,
            relative_path="best_policy.pt",
            manifest_identity=best_manifest,
            expected_env_steps=int(selected_row["env_steps"]),
            expected_validation=_require_mapping(
                selected_row.get("validation_metrics"), "selected validation"
            ),
            expected_config=config,
            expected_infra_identity=infra_identity,
            expected_file_sha256=expected_best_policy_sha256,
            label="best policy",
        )
        if trainer._canonical_json_sha256(best_state_schema) != state_schema_sha256:
            raise ValueError("best policy state schema does not match initial policy")

    records = _read_canonical_jsonl(metrics_path, "trainer metrics")
    validation_records = [
        (line_number, event)
        for line_number, event in records
        if event.get("event") == "validation"
    ]
    expected_steps = [0, *[int(row["env_steps"]) for row in rows]]
    if len(validation_records) != len(expected_steps):
        raise ValueError("trainer metrics validation-event inventory is incomplete")
    validation_by_step: dict[int, tuple[int, dict[str, Any]]] = {}
    for line_number, event in validation_records:
        env_steps = _require_int(
            event.get("env_steps"),
            f"trainer metrics validation line {line_number} env_steps",
        )
        if env_steps in validation_by_step:
            raise ValueError("trainer metrics contains duplicate validation evidence")
        validation_by_step[env_steps] = (line_number, event)
    if list(validation_by_step) != expected_steps:
        raise ValueError(
            "trainer metrics validation evidence does not match selector snapshot order"
        )

    initial_selection = {
        "status": "planner_fallback_no_eligible",
        "eligible_snapshot_count": 0,
        "selected_snapshot_identity": None,
        "best_policy": None,
        "planner_fallback_policy": dict(baseline_policy_manifest),
    }
    initial_manifest_unsigned = {
        "schema_version": CHECKPOINT_SELECTION_SCHEMA,
        "selector": copy.deepcopy(manifest["selector"]),
        "run_identity": copy.deepcopy(expected_run_identity),
        "matched_planner_baseline": copy.deepcopy(dict(baseline)),
        "evaluated_snapshots": [],
        "selection": initial_selection,
    }
    initial_manifest_payload_sha256 = trainer._canonical_json_sha256(
        initial_manifest_unsigned
    )
    baseline_line, baseline_event = validation_by_step[0]
    if baseline_event.get("checkpoint_selection_role") != (
        "matched_planner_safety_ceiling"
    ):
        raise ValueError("trainer metrics planner validation role does not match")
    if baseline_event.get("validation_metrics_sha256") != baseline_metrics_sha256:
        raise ValueError("trainer metrics planner validation hash does not match")
    if baseline_event.get("checkpoint_selection_manifest_payload_sha256") != (
        initial_manifest_payload_sha256
    ):
        raise ValueError("trainer metrics initial selector receipt does not match")
    baseline_event_metrics = {
        key: value
        for key, value in baseline_event.items()
        if key
        not in {
            "event",
            "env_steps",
            "checkpoint_selection_role",
            "validation_metrics_sha256",
            "checkpoint_selection_manifest_payload_sha256",
        }
    }
    if not _canonical_json_equal(baseline_event_metrics, baseline_metrics):
        raise ValueError("trainer metrics planner validation payload does not match")
    baseline_evidence = _validation_evidence(
        metrics_path="metrics.jsonl",
        line_number=baseline_line,
        event=baseline_event,
        validation_metrics_sha256=baseline_metrics_sha256,
        role="matched_planner_safety_ceiling",
        checkpoint_selection_manifest_payload_sha256=(initial_manifest_payload_sha256),
        checkpoint_snapshot_identity=None,
    )

    evidence_inventory = [baseline_evidence]
    for row, enriched in zip(rows, enriched_rows, strict=True):
        env_steps = int(row["env_steps"])
        line_number, event = validation_by_step[env_steps]
        expected_snapshot_identity = (
            trainer._CheckpointSelectionLedger._snapshot_identity(row)
        )
        if not _canonical_json_equal(
            event.get("checkpoint_snapshot_identity"), expected_snapshot_identity
        ):
            raise ValueError(
                "trainer metrics snapshot identity does not match selector"
            )
        raw_role = event.get("validation_role")
        if raw_role is None:
            evidence_role = "scheduled"
        elif raw_role == "final":
            evidence_role = "final"
        else:
            raise ValueError("trainer metrics validation role is not recognized")
        event_metrics = {
            key: value
            for key, value in event.items()
            if key
            not in {
                "event",
                "env_steps",
                "checkpoint_snapshot_identity",
                "validation_role",
            }
        }
        if not _canonical_json_equal(event_metrics, row.get("validation_metrics")):
            raise ValueError(
                "trainer metrics snapshot validation payload does not match"
            )
        evidence = _validation_evidence(
            metrics_path="metrics.jsonl",
            line_number=line_number,
            event=event,
            validation_metrics_sha256=str(row["validation_metrics_sha256"]),
            role=evidence_role,
            checkpoint_selection_manifest_payload_sha256=None,
            checkpoint_snapshot_identity=expected_snapshot_identity,
        )
        enriched["validation_evidence"] = evidence
        evidence_inventory.append(evidence)

    try:
        ledger._verify_manifest()
    except Exception as error:
        raise ValueError("authoritative checkpoint-selection replay failed") from error

    selected_snapshot_identity = selection.get("selected_snapshot_identity")
    selected_identity_copy = (
        None
        if selected_snapshot_identity is None
        else _json_safe_copy(
            _require_mapping(selected_snapshot_identity, "selected snapshot identity"),
            "selected snapshot identity",
        )
    )
    summary_best_validation_sha256 = (
        None
        if best_validation is None
        else trainer._canonical_json_sha256(
            _json_safe_copy(
                _require_mapping(best_validation, "trainer summary best validation"),
                "trainer summary best validation",
            )
        )
    )
    summary_best_score = (
        None
        if best_score is None
        else _json_safe_copy(
            list(_require_sequence(best_score, "trainer summary best score")),
            "trainer summary best score",
        )
    )
    outcome = {
        "schema_version": CHECKPOINT_SELECTION_OUTCOME_SCHEMA,
        "source_identity": {
            "task": task,
            "algorithm": algorithm,
            "training_seed": training_seed,
            "validation_manifest_seed": validation_manifest_seed,
            "eval_episodes": eval_episodes,
            "eval_num_envs": eval_num_envs,
            "policy_rlinf_commit": policy_commit,
            "verifier_rlinf_commit": verifier_commit,
            "evaluator_rlinf_commit": evaluator_commit,
            "benchmark_commit": benchmark_commit,
            "infra_identity_sha256": infra_identity_sha256,
            "trainer_source": policy_sources["trainer_source"],
            "verifier_source": verifier_sources["verifier_source"],
            "builder_source": verifier_sources["builder_source"],
            "evaluator_source": evaluator_sources["evaluator_source"],
        },
        "run_identity": copy.deepcopy(expected_run_identity),
        "trainer_artifacts": {
            "summary": {
                "path": "summary.json",
                "sha256": observed_file_hashes[summary_path],
                "schema_version": TRAINER_SUMMARY_SCHEMA,
                "payload_sha256": summary_payload_sha256,
                "status": "complete",
                "env_steps": summary_env_steps,
                "update_steps": summary_update_steps,
                "best_validation_metrics_sha256": summary_best_validation_sha256,
                "best_selection_score": summary_best_score,
                "final_validation_metrics_sha256": final_validation_sha256,
            },
            "checkpoint_selection": {
                "path": "checkpoint_selection.json",
                "sha256": observed_file_hashes[selection_path],
                "schema_version": CHECKPOINT_SELECTION_SCHEMA,
                "payload_sha256": selection_payload_sha256,
            },
            "config": {
                "path": "config.json",
                "sha256": config_file_sha256,
                "payload_sha256": config_payload_sha256,
            },
            "metrics": {
                "path": "metrics.jsonl",
                "sha256": observed_file_hashes[metrics_path],
                "format": "jsonl",
                "validation_event_count": len(evidence_inventory),
                "validation_event_inventory_sha256": trainer._canonical_json_sha256(
                    evidence_inventory
                ),
            },
        },
        "selector": copy.deepcopy(manifest["selector"]),
        "matched_planner_baseline": {
            "source": baseline["source"],
            "safety_failure_rate_ceiling": baseline["safety_failure_rate_ceiling"],
            "validation_metrics": baseline_metrics,
            "validation_metrics_sha256": baseline_metrics_sha256,
            "selector_metrics": baseline_selector_metrics,
            "selector_metrics_sha256": baseline_selector_metrics_sha256,
            "policy": initial_policy,
            "validation_evidence": baseline_evidence,
        },
        "evaluated_snapshots": enriched_rows,
        "selection": {
            "status": selection_status,
            "eligible_snapshot_count": eligible_count,
            "selected_snapshot_identity": selected_identity_copy,
            "best_policy": best_policy,
            "planner_fallback_policy": initial_policy,
        },
    }
    _reject_unsafe_path_strings(outcome, "checkpoint-selection outcome")
    outcome["payload_sha256"] = trainer._canonical_json_sha256(outcome)

    for path, expected_sha256 in observed_file_hashes.items():
        if _sha256(path) != expected_sha256:
            raise RuntimeError("trainer artifacts changed during outcome verification")
    final_snapshot_files: set[Path] = set()
    for child in snapshots_dir.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RuntimeError(
                "policy snapshot directory changed during outcome verification"
            )
        final_snapshot_files.add(child)
    if final_snapshot_files != expected_snapshot_files:
        raise RuntimeError(
            "policy snapshot directory changed during outcome verification"
        )
    for name, executing_path in executing_sources.items():
        if executing_path.read_bytes() != declared_sources[name].content:
            raise RuntimeError("source artifacts changed during outcome verification")
    final_policy_commit, final_policy_authenticated = _verify_source_checkout(
        root=policy_rlinf_source_root,
        expected_commit=expected_policy_rlinf_commit,
        sources=policy_source_specs,
        label="policy RLinf",
    )
    final_verifier_commit, final_verifier_authenticated = _verify_source_checkout(
        root=verifier_rlinf_source_root,
        expected_commit=expected_verifier_rlinf_commit,
        sources=verifier_source_specs,
        label="verifier RLinf",
    )
    final_evaluator_commit, final_evaluator_authenticated = _verify_source_checkout(
        root=evaluator_rlinf_source_root,
        expected_commit=expected_evaluator_rlinf_commit,
        sources=evaluator_source_specs,
        label="evaluator RLinf",
    )
    if (
        final_policy_commit != policy_commit
        or final_verifier_commit != verifier_commit
        or final_evaluator_commit != evaluator_commit
        or final_policy_authenticated != policy_authenticated
        or final_verifier_authenticated != verifier_authenticated
        or final_evaluator_authenticated != evaluator_authenticated
    ):
        raise RuntimeError("source checkouts changed during outcome verification")
    return outcome


def validate_selected_learned_policy(
    *,
    policy_path: Path,
    trainer_summary_path: Path,
    checkpoint_selection_path: Path,
    checkpoint_selection_outcome_path: Path,
    policy_rlinf_source_root: Path,
    verifier_rlinf_source_root: Path,
    evaluator_rlinf_source_root: Path,
    expected_checkpoint_selection_outcome_sha256: str,
    expected_policy_sha256: str | None = None,
    expected_trainer_summary_sha256: str | None = None,
    expected_checkpoint_selection_sha256: str | None = None,
    expected_rlinf_commit: str | None = None,
    expected_benchmark_commit: str | None = None,
    expected_verifier_rlinf_commit: str | None = None,
    expected_evaluator_rlinf_commit: str | None = None,
) -> dict[str, Any]:
    """Reopen a trainer run and admit only its selected eligible snapshot.

    The trainer's own selection ledger is replayed so a re-sealed manifest cannot
    change eligibility, ordering, snapshot contents, or the selected best-policy
    copy. The returned identity is JSON-safe and can be embedded in an evaluator
    artifact without treating the planner fallback as a learned checkpoint.
    """

    policy_commit = _require_commit(
        expected_rlinf_commit, "expected policy RLinf commit"
    )
    verifier_commit = _require_commit(
        expected_verifier_rlinf_commit, "expected verifier RLinf commit"
    )
    evaluator_commit = _require_commit(
        expected_evaluator_rlinf_commit, "expected evaluator RLinf commit"
    )
    policy_source_specs = {
        "trainer_source": (
            "examples.embodiment.train_dynamic_benchmark_expert",
            "examples/embodiment/train_dynamic_benchmark_expert.py",
        )
    }
    verifier_source_specs = {
        "verifier_source": (
            "examples.embodiment.dynamic_benchmark_checkpoint_admission",
            "examples/embodiment/dynamic_benchmark_checkpoint_admission.py",
        ),
        "builder_source": (
            "examples.embodiment.build_dynamic_benchmark_checkpoint_selection_outcome",
            "examples/embodiment/build_dynamic_benchmark_checkpoint_selection_outcome.py",
        ),
    }
    evaluator_source_specs = {
        "evaluator_source": (
            "examples.embodiment.evaluate_dynamic_benchmark_expert",
            "examples/embodiment/evaluate_dynamic_benchmark_expert.py",
        ),
    }
    observed_policy_commit, policy_authenticated = _verify_source_checkout(
        root=policy_rlinf_source_root,
        expected_commit=policy_commit,
        sources=policy_source_specs,
        label="policy RLinf",
    )
    observed_verifier_commit, verifier_authenticated = _verify_source_checkout(
        root=verifier_rlinf_source_root,
        expected_commit=verifier_commit,
        sources=verifier_source_specs,
        label="verifier RLinf",
    )
    observed_evaluator_commit, evaluator_authenticated = _verify_source_checkout(
        root=evaluator_rlinf_source_root,
        expected_commit=evaluator_commit,
        sources=evaluator_source_specs,
        label="evaluator RLinf",
    )
    policy_sources = {
        name: source.public_identity() for name, source in policy_authenticated.items()
    }
    verifier_sources = {
        name: source.public_identity()
        for name, source in verifier_authenticated.items()
    }
    evaluator_sources = {
        name: source.public_identity()
        for name, source in evaluator_authenticated.items()
    }
    trainer_source = policy_authenticated["trainer_source"]
    trainer = _load_authoritative_trainer(
        trainer_source.content, source_path=trainer_source.path
    )

    summary_path = _canonical_artifact(
        trainer_summary_path, "summary.json", "trainer summary"
    )
    selection_path = _canonical_artifact(
        checkpoint_selection_path,
        "checkpoint_selection.json",
        "checkpoint-selection manifest",
    )
    output = summary_path.parent
    if selection_path.parent != output:
        raise ValueError(
            "trainer summary and checkpoint-selection manifest must be siblings"
        )
    raw_outcome_path = Path(checkpoint_selection_outcome_path)
    if raw_outcome_path.is_symlink():
        raise ValueError("checkpoint-selection outcome must not be a symlink")
    if not raw_outcome_path.is_file():
        raise FileNotFoundError(raw_outcome_path)
    outcome_path = raw_outcome_path.resolve(strict=True)
    if outcome_path != output / CHECKPOINT_SELECTION_OUTCOME_FILENAME:
        raise ValueError(
            "checkpoint-selection outcome must be the canonical trainer-run sibling"
        )
    outcome_file_sha256 = _match_expected_sha256(
        outcome_path,
        expected_checkpoint_selection_outcome_sha256,
        "checkpoint-selection outcome",
    )
    outcome = _read_json(outcome_path, "checkpoint-selection outcome")
    if outcome.get("schema_version") != CHECKPOINT_SELECTION_OUTCOME_SCHEMA:
        raise ValueError("checkpoint-selection outcome schema does not match")
    _require_exact_keys(
        outcome,
        {
            "schema_version",
            "source_identity",
            "run_identity",
            "trainer_artifacts",
            "selector",
            "matched_planner_baseline",
            "evaluated_snapshots",
            "selection",
            "payload_sha256",
        },
        "checkpoint-selection outcome",
    )
    outcome_payload_sha256 = _verify_payload_hash(
        outcome,
        label="checkpoint-selection outcome",
        canonical_sha256=_canonical_json_sha256,
    )

    # Replay the complete canonical producer from clean, commit-pinned policy and
    # evaluator source roots.  Comparing the full payload validates metrics.jsonl,
    # every validation-evidence row, selector projections, and source blobs without
    # maintaining a second weaker interpretation of the outcome schema here.
    config_candidate = output / "config.json"
    metrics_candidate = output / "metrics.jsonl"
    initial_policy_candidate = output / "initial_policy.pt"
    best_policy_candidate = output / "best_policy.pt"
    authoritative_outcome = build_checkpoint_selection_outcome_payload(
        run_root=output,
        policy_rlinf_source_root=policy_rlinf_source_root,
        verifier_rlinf_source_root=verifier_rlinf_source_root,
        evaluator_rlinf_source_root=evaluator_rlinf_source_root,
        expected_policy_rlinf_commit=observed_policy_commit,
        expected_verifier_rlinf_commit=observed_verifier_commit,
        expected_evaluator_rlinf_commit=observed_evaluator_commit,
        expected_benchmark_commit=_require_commit(
            expected_benchmark_commit, "expected benchmark commit"
        ),
        expected_summary_sha256=_sha256(summary_path),
        expected_checkpoint_selection_sha256=_sha256(selection_path),
        expected_config_sha256=_sha256(config_candidate),
        expected_metrics_sha256=_sha256(metrics_candidate),
        expected_initial_policy_sha256=_sha256(initial_policy_candidate),
        expected_best_policy_sha256=(
            _sha256(best_policy_candidate) if best_policy_candidate.is_file() else None
        ),
    )
    if outcome != authoritative_outcome:
        raise ValueError(
            "checkpoint-selection outcome differs from authoritative producer replay"
        )

    # Resolve without requiring existence so the no-eligible fallback produces an
    # explicit selection error instead of being mistaken for a missing learned file.
    resolved_policy = policy_path.resolve(strict=False)
    if resolved_policy != output / "best_policy.pt":
        raise ValueError(
            "formal learned-policy admission requires checkpoint_role=best at "
            "best_policy.pt"
        )

    summary = _read_json(summary_path, "trainer summary")
    manifest = _read_json(selection_path, "checkpoint-selection manifest")
    summary_file_sha256 = _sha256(summary_path)
    selection_file_sha256 = _sha256(selection_path)
    if expected_trainer_summary_sha256 is not None and summary_file_sha256 != (
        _require_sha256(
            expected_trainer_summary_sha256, "expected trainer summary SHA-256"
        )
    ):
        raise ValueError("trainer summary SHA-256 mismatch")
    if expected_checkpoint_selection_sha256 is not None and selection_file_sha256 != (
        _require_sha256(
            expected_checkpoint_selection_sha256,
            "expected checkpoint-selection SHA-256",
        )
    ):
        raise ValueError("checkpoint-selection manifest SHA-256 mismatch")

    if summary.get("schema_version") != TRAINER_SUMMARY_SCHEMA:
        raise ValueError("trainer summary schema does not match")
    summary_payload_sha256 = _verify_payload_hash(
        summary,
        label="trainer summary",
        canonical_sha256=trainer._canonical_json_sha256,
    )
    if summary.get("status") != "complete":
        raise ValueError("formal learned-policy admission requires complete training")
    if manifest.get("schema_version") != CHECKPOINT_SELECTION_SCHEMA:
        raise ValueError("checkpoint-selection manifest schema does not match")
    selection_payload_sha256 = _verify_payload_hash(
        manifest,
        label="checkpoint-selection manifest",
        canonical_sha256=trainer._canonical_json_sha256,
    )

    selection = _require_mapping(
        manifest.get("selection"), "checkpoint-selection result"
    )
    status = selection.get("status")
    if status == "planner_fallback_no_eligible":
        raise ValueError(
            "planner_fallback_no_eligible is not a learned policy checkpoint"
        )
    if status != "selected_eligible_snapshot":
        raise ValueError("checkpoint-selection status is not an eligible snapshot")
    eligible_count = _require_int(
        selection.get("eligible_snapshot_count"),
        "checkpoint-selection eligible snapshot count",
        minimum=1,
    )
    selected_identity = selection.get("selected_snapshot_identity")
    best_policy = selection.get("best_policy")
    if selected_identity is None:
        raise ValueError("checkpoint-selection selected snapshot is null")
    if best_policy is None:
        raise ValueError("checkpoint-selection best policy is null")
    selected_identity = _require_mapping(
        selected_identity, "checkpoint-selection selected snapshot"
    )
    best_policy = _require_mapping(best_policy, "checkpoint-selection best policy")
    selected_env_steps = _require_int(
        selected_identity.get("env_steps"), "selected snapshot env_steps", minimum=1
    )
    if (
        best_policy.get("path") != "best_policy.pt"
        or _require_int(
            best_policy.get("env_steps"), "best policy env_steps", minimum=1
        )
        != selected_env_steps
    ):
        raise ValueError("checkpoint-selection best policy identity is not canonical")

    rows = manifest.get("evaluated_snapshots")
    if not isinstance(rows, list) or not rows:
        raise ValueError("checkpoint-selection snapshot ledger is empty")
    eligible_rows = 0
    selected_rows: list[Mapping[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"checkpoint-selection row {index}")
        _require_int(
            row.get("env_steps"),
            f"checkpoint-selection row {index} env_steps",
            minimum=1,
        )
        if not isinstance(row.get("eligible"), bool):
            raise ValueError(
                f"checkpoint-selection row {index} eligible must be boolean"
            )
        if not isinstance(row.get("selected"), bool):
            raise ValueError(
                f"checkpoint-selection row {index} selected must be boolean"
            )
        eligible_rows += int(row["eligible"])
        if row["selected"]:
            selected_rows.append(row)
    if eligible_rows != eligible_count:
        raise ValueError("checkpoint-selection eligible snapshot count was tampered")
    if len(selected_rows) != 1 or selected_rows[0].get("eligible") is not True:
        raise ValueError(
            "checkpoint-selection must select exactly one eligible snapshot"
        )
    selected_row = selected_rows[0]
    selected_policy = _require_mapping(
        selected_row.get("policy"), "selected checkpoint policy"
    )
    expected_selected_identity = {
        "env_steps": selected_row["env_steps"],
        "policy_path": selected_policy.get("path"),
        "policy_sha256": selected_policy.get("sha256"),
        "validation_metrics_sha256": selected_row.get("validation_metrics_sha256"),
    }
    if dict(selected_identity) != expected_selected_identity:
        raise ValueError("checkpoint-selection selected snapshot identity was tampered")
    expected_best_policy = {**dict(selected_policy), "path": "best_policy.pt"}
    if dict(best_policy) != expected_best_policy:
        raise ValueError("checkpoint-selection best policy identity was tampered")

    if not resolved_policy.is_file():
        raise FileNotFoundError(resolved_policy)
    policy_sha256 = _sha256(resolved_policy)
    if expected_policy_sha256 is not None and policy_sha256 != _require_sha256(
        expected_policy_sha256, "expected policy SHA-256"
    ):
        raise ValueError("policy SHA-256 mismatch")
    try:
        import torch

        raw_checkpoint = torch.load(
            resolved_policy, map_location="cpu", weights_only=False
        )
    except Exception as error:
        raise ValueError(
            f"cannot load selected policy checkpoint {resolved_policy}"
        ) from error
    checkpoint = _require_mapping(raw_checkpoint, "selected policy checkpoint")
    if checkpoint.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("selected policy checkpoint schema does not match")
    checkpoint_env_steps = _require_int(
        checkpoint.get("env_steps"), "selected policy env_steps", minimum=1
    )
    if checkpoint_env_steps != selected_env_steps:
        raise ValueError("selected policy env_steps do not match checkpoint selection")
    if policy_sha256 != _require_sha256(
        best_policy.get("sha256"), "checkpoint-selection best policy SHA-256"
    ):
        raise ValueError("best_policy.pt does not match the selected eligible snapshot")
    config = _json_safe_copy(
        _require_mapping(checkpoint.get("config"), "selected policy config"),
        "selected policy config",
    )
    state_schema = _json_safe_copy(
        _require_mapping(
            checkpoint.get("state_schema"), "selected policy state schema"
        ),
        "selected policy state schema",
    )
    validation = _json_safe_copy(
        _require_mapping(checkpoint.get("validation"), "selected policy validation"),
        "selected policy validation",
    )
    if "infra_identity" not in checkpoint:
        raise ValueError("selected policy infra identity is missing")
    raw_infra_identity = checkpoint["infra_identity"]
    if raw_infra_identity is not None and not isinstance(raw_infra_identity, Mapping):
        raise ValueError("selected policy infra identity must be a mapping or null")
    infra_identity = _json_safe_copy(
        raw_infra_identity, "selected policy infra identity"
    )

    summary_config = _require_mapping(summary.get("config"), "trainer summary config")
    if "infra_identity" not in summary:
        raise ValueError("trainer summary infra identity is missing")
    if dict(summary_config) != config:
        raise ValueError("trainer summary config does not match selected policy")
    if summary["infra_identity"] != infra_identity:
        raise ValueError(
            "trainer summary infra identity does not match selected policy"
        )
    if summary.get("best_validation") != validation:
        raise ValueError(
            "trainer summary best validation does not match selected policy"
        )
    if summary.get("best_validation") != selected_row.get("validation_metrics"):
        raise ValueError("trainer summary best validation is not the selected snapshot")
    if summary.get("best_score") is None:
        raise ValueError("trainer summary best score is null")
    summary_best_score = _require_sequence(
        summary["best_score"], "trainer summary best score"
    )
    selected_score = _require_sequence(
        selected_row.get("selection_score"), "selected snapshot score"
    )
    if list(summary_best_score) != list(selected_score):
        raise ValueError(
            "trainer summary best score is not the selected snapshot score"
        )
    summary_env_steps = _require_int(
        summary.get("env_steps"), "trainer summary env_steps", minimum=1
    )
    if summary_env_steps < checkpoint_env_steps:
        raise ValueError("selected policy env_steps exceed trainer summary env_steps")

    config_payload_sha256 = trainer._canonical_json_sha256(config)
    if summary.get("config_sha256") != config_payload_sha256:
        raise ValueError("trainer summary config SHA-256 does not recompute")
    config_path = _canonical_artifact(
        output / "config.json", "config.json", "trainer config"
    )
    if _read_json(config_path, "trainer config") != config:
        raise ValueError("trainer config artifact does not match selected policy")
    config_file_sha256 = _sha256(config_path)
    if summary.get("config_file_sha256") != config_file_sha256:
        raise ValueError("trainer summary config-file SHA-256 does not recompute")

    task = _require_string(config.get("task"), "policy task")
    algorithm = _require_string(config.get("algorithm"), "policy algorithm")
    if algorithm != "residual_rlpd":
        raise ValueError("checkpoint-selected learned policy must use residual_rlpd")
    rlinf_commit = _require_commit(config.get("rlinf_commit"), "policy RLinf commit")
    benchmark_commit = _require_commit(
        config.get("benchmark_commit"), "policy benchmark commit"
    )
    if expected_rlinf_commit is not None and rlinf_commit != _require_commit(
        expected_rlinf_commit, "expected RLinf commit"
    ):
        raise ValueError("selected policy RLinf source identity mismatch")
    if expected_benchmark_commit is not None and benchmark_commit != _require_commit(
        expected_benchmark_commit, "expected benchmark commit"
    ):
        raise ValueError("selected policy benchmark source identity mismatch")
    training_seed = _require_int(config.get("seed"), "policy training seed")
    validation_manifest_seed = _require_int(
        config.get("validation_manifest_seed"),
        "checkpoint-selection validation manifest seed",
    )
    expected_run_identity = {
        "schema_version": CHECKPOINT_SELECTION_RUN_SCHEMA,
        "task": task,
        "algorithm": algorithm,
        "rlinf_commit": rlinf_commit,
        "benchmark_commit": benchmark_commit,
        "seed": training_seed,
        "validation_manifest_seed": validation_manifest_seed,
        "eval_episodes": _require_int(
            config.get("eval_episodes"), "checkpoint-selection eval_episodes", minimum=1
        ),
        "eval_num_envs": _require_int(
            config.get("eval_num_envs"), "checkpoint-selection eval_num_envs", minimum=1
        ),
        "config_sha256": config_file_sha256,
        "config_payload_sha256": config_payload_sha256,
        "state_schema_sha256": trainer._canonical_json_sha256(state_schema),
    }
    if manifest.get("run_identity") != expected_run_identity:
        raise ValueError("checkpoint-selection config/source identity does not match")

    expected_summary_reference = {
        "manifest_path": "checkpoint_selection.json",
        "manifest_payload_sha256": selection_payload_sha256,
        "status": "selected_eligible_snapshot",
        "eligible_snapshot_count": eligible_count,
        "selected_snapshot_identity": dict(selected_identity),
        "planner_fallback_policy": selection.get("planner_fallback_policy"),
    }
    if summary.get("checkpoint_selection") != expected_summary_reference:
        raise ValueError(
            "trainer summary and checkpoint-selection manifest identities diverged"
        )

    validation_metrics_sha256 = trainer._canonical_json_sha256(validation)
    if selected_row.get("validation_metrics_sha256") != validation_metrics_sha256:
        raise ValueError("selected policy validation identity does not match")
    if selected_policy.get("config_payload_sha256") != config_payload_sha256:
        raise ValueError("selected policy config identity does not match")
    if (
        selected_policy.get("state_schema_sha256")
        != expected_run_identity["state_schema_sha256"]
    ):
        raise ValueError("selected policy state-schema identity does not match")

    # Replay the trainer's complete selection verifier. This reopens the planner
    # baseline, every immutable snapshot, and best_policy.pt, then recomputes the
    # safety eligibility and lexicographic winner.
    try:
        ledger = trainer._CheckpointSelectionLedger(
            output, expected_run_identity, copy.deepcopy(manifest)
        )
        ledger._verify_manifest()
    except Exception as error:
        raise ValueError("checkpoint-selection ledger verification failed") from error

    # The portable outcome is the canonical authority that joins the complete
    # trainer ledger to the verifier/evaluator source identities.  Reopen it only
    # after the trainer artifacts themselves have passed their independent replay,
    # then require exact (not subset) joins at every boundary used downstream.
    verifier_rlinf_commit = verifier_commit
    evaluator_rlinf_commit = evaluator_commit
    if outcome.get("run_identity") != expected_run_identity:
        raise ValueError("checkpoint-selection outcome run identity mismatch")

    outcome_source = _require_mapping(
        outcome.get("source_identity"), "checkpoint-selection outcome source identity"
    )
    _require_exact_keys(
        outcome_source,
        {
            "task",
            "algorithm",
            "training_seed",
            "validation_manifest_seed",
            "eval_episodes",
            "eval_num_envs",
            "policy_rlinf_commit",
            "verifier_rlinf_commit",
            "evaluator_rlinf_commit",
            "benchmark_commit",
            "infra_identity_sha256",
            "trainer_source",
            "verifier_source",
            "builder_source",
            "evaluator_source",
        },
        "checkpoint-selection outcome source identity",
    )
    expected_source_scalars = {
        "task": task,
        "algorithm": algorithm,
        "training_seed": training_seed,
        "validation_manifest_seed": validation_manifest_seed,
        "eval_episodes": expected_run_identity["eval_episodes"],
        "eval_num_envs": expected_run_identity["eval_num_envs"],
        "policy_rlinf_commit": rlinf_commit,
        "verifier_rlinf_commit": verifier_rlinf_commit,
        "evaluator_rlinf_commit": evaluator_rlinf_commit,
        "benchmark_commit": benchmark_commit,
        "infra_identity_sha256": trainer._canonical_json_sha256(infra_identity),
    }
    for field, expected_value in expected_source_scalars.items():
        if outcome_source.get(field) != expected_value:
            raise ValueError(
                f"checkpoint-selection outcome source identity {field} mismatch"
            )
    expected_source_descriptors = {
        **policy_sources,
        **verifier_sources,
        **evaluator_sources,
    }
    for name, expected_identity in expected_source_descriptors.items():
        if outcome_source.get(name) != expected_identity:
            raise ValueError(
                f"checkpoint-selection outcome {name} does not match clean source blob"
            )

    source_contracts = {
        "trainer_source": (
            "examples.embodiment.train_dynamic_benchmark_expert",
            "examples/embodiment/train_dynamic_benchmark_expert.py",
            policy_authenticated["trainer_source"].path,
        ),
        "verifier_source": (
            "examples.embodiment.dynamic_benchmark_checkpoint_admission",
            "examples/embodiment/dynamic_benchmark_checkpoint_admission.py",
            verifier_authenticated["verifier_source"].path,
        ),
        "builder_source": (
            "examples.embodiment.build_dynamic_benchmark_checkpoint_selection_outcome",
            "examples/embodiment/build_dynamic_benchmark_checkpoint_selection_outcome.py",
            verifier_authenticated["builder_source"].path,
        ),
        "evaluator_source": (
            "examples.embodiment.evaluate_dynamic_benchmark_expert",
            "examples/embodiment/evaluate_dynamic_benchmark_expert.py",
            evaluator_authenticated["evaluator_source"].path,
        ),
    }
    for name, (module, repository_path, executing_path) in source_contracts.items():
        identity = _require_mapping(
            outcome_source.get(name), f"checkpoint-selection outcome {name}"
        )
        _require_exact_keys(
            identity,
            {"module", "repository_path", "sha256"},
            f"checkpoint-selection outcome {name}",
        )
        if (
            identity.get("module") != module
            or identity.get("repository_path") != repository_path
        ):
            raise ValueError(f"checkpoint-selection outcome {name} identity mismatch")
        _require_sha256(identity.get("sha256"), f"checkpoint-selection outcome {name}")
        if executing_path is not None:
            if executing_path.is_symlink() or not executing_path.is_file():
                raise ValueError(f"executing {name} is missing or symlinked")
            if _sha256(executing_path.resolve(strict=True)) != identity["sha256"]:
                raise ValueError(
                    f"checkpoint-selection outcome {name} does not match executing source"
                )

    trainer_artifacts = _require_mapping(
        outcome.get("trainer_artifacts"),
        "checkpoint-selection outcome trainer artifacts",
    )
    _require_exact_keys(
        trainer_artifacts,
        {"summary", "checkpoint_selection", "config", "metrics"},
        "checkpoint-selection outcome trainer artifacts",
    )
    outcome_summary = _require_mapping(
        trainer_artifacts.get("summary"), "checkpoint-selection outcome summary"
    )
    expected_outcome_summary = {
        "path": "summary.json",
        "sha256": summary_file_sha256,
        "schema_version": TRAINER_SUMMARY_SCHEMA,
        "payload_sha256": summary_payload_sha256,
        "status": "complete",
        "env_steps": summary_env_steps,
        "update_steps": _require_int(
            summary.get("update_steps"), "trainer summary update_steps"
        ),
        "best_validation_metrics_sha256": trainer._canonical_json_sha256(validation),
        "best_selection_score": _json_safe_copy(
            list(summary_best_score), "trainer summary best selection score"
        ),
        "final_validation_metrics_sha256": trainer._canonical_json_sha256(
            _json_safe_copy(
                _require_mapping(summary.get("final_validation"), "final validation"),
                "final validation",
            )
        ),
    }
    if dict(outcome_summary) != expected_outcome_summary:
        raise ValueError("checkpoint-selection outcome summary identity mismatch")
    outcome_selection_artifact = _require_mapping(
        trainer_artifacts.get("checkpoint_selection"),
        "checkpoint-selection outcome manifest artifact",
    )
    if dict(outcome_selection_artifact) != {
        "path": "checkpoint_selection.json",
        "sha256": selection_file_sha256,
        "schema_version": CHECKPOINT_SELECTION_SCHEMA,
        "payload_sha256": selection_payload_sha256,
    }:
        raise ValueError("checkpoint-selection outcome manifest identity mismatch")
    outcome_config = _require_mapping(
        trainer_artifacts.get("config"), "checkpoint-selection outcome config"
    )
    if dict(outcome_config) != {
        "path": "config.json",
        "sha256": config_file_sha256,
        "payload_sha256": config_payload_sha256,
    }:
        raise ValueError("checkpoint-selection outcome config identity mismatch")
    outcome_metrics = _require_mapping(
        trainer_artifacts.get("metrics"), "checkpoint-selection outcome metrics"
    )
    _require_exact_keys(
        outcome_metrics,
        {
            "path",
            "sha256",
            "format",
            "validation_event_count",
            "validation_event_inventory_sha256",
        },
        "checkpoint-selection outcome metrics",
    )
    if (
        outcome_metrics.get("path") != "metrics.jsonl"
        or outcome_metrics.get("format") != "jsonl"
        or outcome_metrics.get("validation_event_count") != len(rows) + 1
    ):
        raise ValueError("checkpoint-selection outcome metrics identity mismatch")
    _require_sha256(
        outcome_metrics.get("sha256"), "checkpoint-selection outcome metrics SHA-256"
    )
    _require_sha256(
        outcome_metrics.get("validation_event_inventory_sha256"),
        "checkpoint-selection outcome validation inventory SHA-256",
    )

    if outcome.get("selector") != manifest.get("selector"):
        raise ValueError("checkpoint-selection outcome selector identity mismatch")
    outcome_baseline = _require_mapping(
        outcome.get("matched_planner_baseline"),
        "checkpoint-selection outcome matched planner baseline",
    )
    baseline_manifest = _require_mapping(
        manifest.get("matched_planner_baseline"), "matched planner baseline"
    )
    baseline_policy_manifest = _require_mapping(
        baseline_manifest.get("policy"), "matched planner policy"
    )
    expected_baseline_policy = {
        **dict(baseline_policy_manifest),
        "schema_version": POLICY_SCHEMA,
        "infra_identity_sha256": trainer._canonical_json_sha256(infra_identity),
    }
    if (
        outcome_baseline.get("source") != baseline_manifest.get("source")
        or outcome_baseline.get("safety_failure_rate_ceiling")
        != baseline_manifest.get("safety_failure_rate_ceiling")
        or outcome_baseline.get("validation_metrics")
        != baseline_manifest.get("validation_metrics")
        or outcome_baseline.get("validation_metrics_sha256")
        != baseline_manifest.get("validation_metrics_sha256")
        or outcome_baseline.get("policy") != expected_baseline_policy
    ):
        raise ValueError("checkpoint-selection outcome planner baseline mismatch")

    outcome_rows_raw = outcome.get("evaluated_snapshots")
    if not isinstance(outcome_rows_raw, list) or len(outcome_rows_raw) != len(rows):
        raise ValueError("checkpoint-selection outcome snapshot inventory mismatch")
    for index, (outcome_row_raw, manifest_row) in enumerate(
        zip(outcome_rows_raw, rows, strict=True)
    ):
        outcome_row = _require_mapping(
            outcome_row_raw, f"checkpoint-selection outcome row {index}"
        )
        manifest_policy = _require_mapping(
            manifest_row.get("policy"),
            f"checkpoint-selection manifest row {index} policy",
        )
        expected_outcome_policy = {
            **dict(manifest_policy),
            "schema_version": POLICY_SCHEMA,
            "infra_identity_sha256": trainer._canonical_json_sha256(infra_identity),
        }
        for field in (
            "env_steps",
            "validation_metrics",
            "validation_metrics_sha256",
            "eligible",
            "eligibility_reason",
            "selection_score",
            "selected",
        ):
            if outcome_row.get(field) != manifest_row.get(field):
                raise ValueError(
                    f"checkpoint-selection outcome row {index} {field} mismatch"
                )
        if outcome_row.get("policy") != expected_outcome_policy:
            raise ValueError(
                f"checkpoint-selection outcome row {index} policy mismatch"
            )
        selector_metrics = _selector_metrics_projection(
            _require_mapping(
                manifest_row.get("validation_metrics"),
                f"checkpoint-selection manifest row {index} validation",
            ),
            f"checkpoint-selection manifest row {index}",
        )
        if outcome_row.get("selector_metrics") != selector_metrics or outcome_row.get(
            "selector_metrics_sha256"
        ) != trainer._canonical_json_sha256(selector_metrics):
            raise ValueError(
                f"checkpoint-selection outcome row {index} selector metrics mismatch"
            )

    outcome_decision = _require_mapping(
        outcome.get("selection"), "checkpoint-selection outcome decision"
    )
    _require_exact_keys(
        outcome_decision,
        {
            "status",
            "eligible_snapshot_count",
            "selected_snapshot_identity",
            "best_policy",
            "planner_fallback_policy",
        },
        "checkpoint-selection outcome decision",
    )
    if outcome_decision.get("status") != "selected_eligible_snapshot":
        raise ValueError(
            "checkpoint-selection outcome is not a selected eligible snapshot"
        )
    expected_outcome_best_policy = {
        **dict(best_policy),
        "schema_version": POLICY_SCHEMA,
        "infra_identity_sha256": trainer._canonical_json_sha256(infra_identity),
    }
    planner_fallback_manifest = _require_mapping(
        selection.get("planner_fallback_policy"), "planner fallback policy"
    )
    expected_outcome_planner_fallback = {
        **dict(planner_fallback_manifest),
        "schema_version": POLICY_SCHEMA,
        "infra_identity_sha256": trainer._canonical_json_sha256(infra_identity),
    }
    if (
        outcome_decision.get("eligible_snapshot_count") != eligible_count
        or outcome_decision.get("selected_snapshot_identity") != dict(selected_identity)
        or outcome_decision.get("best_policy") != expected_outcome_best_policy
        or outcome_decision.get("planner_fallback_policy")
        != expected_outcome_planner_fallback
    ):
        raise ValueError(
            "checkpoint-selection outcome selected policy identity mismatch"
        )
    _reject_unsafe_path_strings(outcome, "checkpoint-selection outcome")

    final_policy_commit, final_policy_authenticated = _verify_source_checkout(
        root=policy_rlinf_source_root,
        expected_commit=policy_commit,
        sources=policy_source_specs,
        label="policy RLinf",
    )
    final_verifier_commit, final_verifier_authenticated = _verify_source_checkout(
        root=verifier_rlinf_source_root,
        expected_commit=verifier_commit,
        sources=verifier_source_specs,
        label="verifier RLinf",
    )
    final_evaluator_commit, final_evaluator_authenticated = _verify_source_checkout(
        root=evaluator_rlinf_source_root,
        expected_commit=evaluator_commit,
        sources=evaluator_source_specs,
        label="evaluator RLinf",
    )
    if (
        final_policy_commit != observed_policy_commit
        or final_verifier_commit != observed_verifier_commit
        or final_evaluator_commit != observed_evaluator_commit
        or final_policy_authenticated != policy_authenticated
        or final_verifier_authenticated != verifier_authenticated
        or final_evaluator_authenticated != evaluator_authenticated
    ):
        raise RuntimeError("admission source roots changed during validation")

    if (
        _sha256(resolved_policy) != policy_sha256
        or _sha256(summary_path) != summary_file_sha256
        or _sha256(selection_path) != selection_file_sha256
        or _sha256(config_path) != config_file_sha256
        or _sha256(outcome_path) != outcome_file_sha256
    ):
        raise RuntimeError("trainer admission artifacts changed during validation")

    return {
        "checkpoint_role": "best",
        "policy": {
            "path": str(resolved_policy),
            "sha256": policy_sha256,
            "schema_version": POLICY_SCHEMA,
            "env_steps": checkpoint_env_steps,
            "config_sha256": config_payload_sha256,
            "state_schema_sha256": expected_run_identity["state_schema_sha256"],
            "validation_metrics_sha256": validation_metrics_sha256,
        },
        "trainer_summary": {
            "path": str(summary_path),
            "sha256": summary_file_sha256,
            "schema_version": TRAINER_SUMMARY_SCHEMA,
            "payload_sha256": summary_payload_sha256,
        },
        "checkpoint_selection": {
            "path": str(selection_path),
            "sha256": selection_file_sha256,
            "schema_version": CHECKPOINT_SELECTION_SCHEMA,
            "payload_sha256": selection_payload_sha256,
            "status": "selected_eligible_snapshot",
            "eligible_snapshot_count": eligible_count,
            "selected_snapshot_identity": copy.deepcopy(dict(selected_identity)),
        },
        "checkpoint_selection_outcome": {
            "path": str(outcome_path),
            "sha256": outcome_file_sha256,
            "schema_version": CHECKPOINT_SELECTION_OUTCOME_SCHEMA,
            "payload_sha256": outcome_payload_sha256,
        },
        "config": {
            "path": str(config_path),
            "sha256": config_file_sha256,
            "payload_sha256": config_payload_sha256,
        },
        "source_identity": {
            "task": task,
            "algorithm": algorithm,
            "training_seed": training_seed,
            "validation_manifest_seed": validation_manifest_seed,
            "rlinf_commit": rlinf_commit,
            "benchmark_commit": benchmark_commit,
        },
    }
