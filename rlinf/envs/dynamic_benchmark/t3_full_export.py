# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed loader for GPUPLAN0 T3-full row-export v2 manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .gpu_backend import reset_request_identity

SCHEMA_VERSION = "gpuplan0-t3-full-row-export-v2"
TASK_ID = "t3_full"
_ROW_REQUEST_FIELDS = (
    "api_version",
    "episode_id",
    "task_id",
    "split",
    "seed",
    "action_mode",
    "observation_track",
    "object_mode",
    "reset_mode",
    "factors",
)


@dataclass(frozen=True)
class T3FullExportRow:
    """One integrity-checked manifest row and its exact reset artifact."""

    manifest_path: Path
    manifest_file_sha256: str
    payload_sha256: str
    manifest_seed: int
    candidate_index: int
    export_dir: Path
    row: Mapping[str, Any]
    request: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "row", MappingProxyType(dict(self.row)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value


def _load_artifacts(export_dir: Path) -> Any:
    try:
        from se3_wam.benchmark.gpu_native.p0_grasp_engine import (
            load_p0_grasp_artifacts,
        )
    except ImportError as exc:
        raise RuntimeError("T3-full export loading requires the pinned SE3-WAM source") from exc
    return load_p0_grasp_artifacts(export_dir)


def _request_identity_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(_ROW_REQUEST_FIELDS) - set(row)
    if missing:
        raise ValueError(f"T3-full export row omits reset fields: {sorted(missing)}")
    return {name: row[name] for name in _ROW_REQUEST_FIELDS}


def load_t3_full_export_row(
    manifest_path: str | Path,
    *,
    candidate_index: int,
) -> T3FullExportRow:
    """Load one row only after manifest, artifact, and full reset identity agree."""

    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
    ):
        raise ValueError("candidate_index must be a non-negative integer")
    path = Path(manifest_path).resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"T3-full export manifest is not a file: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("T3-full export manifest schema mismatch")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("T3-full export manifest payload is missing")
    payload_sha256 = _require_sha256("payload_sha256", document.get("payload_sha256"))
    if _canonical_sha256(payload) != payload_sha256:
        raise ValueError("T3-full export manifest payload hash mismatch")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("observation_track") != "state"
        or payload.get("action_mode") != "E7"
    ):
        raise ValueError("T3-full export manifest changes the frozen task/STATE/E7 contract")
    manifest_seed = payload.get("manifest_seed")
    if isinstance(manifest_seed, bool) or not isinstance(manifest_seed, int) or manifest_seed < 0:
        raise ValueError("T3-full export manifest_seed must be a non-negative integer")
    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("episode_count") != len(rows):
        raise ValueError("T3-full export row count does not match episode_count")
    if payload.get("candidate_indices") != [row.get("candidate_index") for row in rows]:
        raise ValueError("T3-full export candidate index projection drifted")
    if payload.get("episode_ids") != [row.get("episode_id") for row in rows]:
        raise ValueError("T3-full export episode ID projection drifted")
    selected = [row for row in rows if row.get("candidate_index") == candidate_index]
    if len(selected) != 1 or not isinstance(selected[0], dict):
        raise ValueError("T3-full export manifest must contain exactly one selected row")
    row = selected[0]
    if row.get("task_id") != payload["task_id"] or row.get("split") != payload.get("split"):
        raise ValueError("T3-full export row changes top-level task or split")
    if row.get("observation_track") != payload["observation_track"]:
        raise ValueError("T3-full export row changes top-level observation track")
    if row.get("action_mode") != payload["action_mode"]:
        raise ValueError("T3-full export row changes top-level action mode")

    relative_export = row.get("export_dir")
    if not isinstance(relative_export, str) or not relative_export.strip():
        raise ValueError("T3-full export row lacks export_dir")
    export_dir = (path.parent / relative_export).resolve(strict=True)
    if not export_dir.is_dir() or not export_dir.is_relative_to(path.parent):
        raise ValueError("T3-full export row escapes its manifest root")
    sums_path = export_dir / "SHA256SUMS"
    request_path = export_dir / "request.json"
    for artifact in (sums_path, request_path):
        if not artifact.is_file():
            raise FileNotFoundError(f"T3-full export artifact is missing: {artifact}")
    if _sha256(sums_path) != _require_sha256(
        "row.sha256sums_sha256", row.get("sha256sums_sha256")
    ):
        raise ValueError("T3-full export SHA256SUMS identity mismatch")
    if _sha256(request_path) != _require_sha256(
        "row.request_json_sha256", row.get("request_json_sha256")
    ):
        raise ValueError("T3-full export request.json identity mismatch")

    artifacts = _load_artifacts(export_dir)
    actual_identity = reset_request_identity(artifacts.reset_request)
    expected_identity = _request_identity_from_row(row)
    if actual_identity != expected_identity:
        raise ValueError(
            "T3-full export reset identity mismatch: "
            f"expected={expected_identity}, actual={actual_identity}"
        )
    if row.get("independent_reset", {}).get("exact") is not True:
        raise ValueError("T3-full export row lacks an exact independent-reset receipt")
    state_sha256 = _require_sha256("row.state_sha256", row.get("state_sha256"))
    if row["independent_reset"].get("state_sha256") != state_sha256:
        raise ValueError("T3-full independent-reset state receipt changed")

    return T3FullExportRow(
        manifest_path=path,
        manifest_file_sha256=_sha256(path),
        payload_sha256=payload_sha256,
        manifest_seed=manifest_seed,
        candidate_index=candidate_index,
        export_dir=export_dir,
        row=row,
        request=artifacts.reset_request,
    )


__all__ = [
    "SCHEMA_VERSION",
    "T3FullExportRow",
    "load_t3_full_export_row",
]
