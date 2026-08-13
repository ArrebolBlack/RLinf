# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only contracts for the T3-full row-export v2 loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rlinf.envs.dynamic_benchmark import t3_full_export


@dataclass(frozen=True)
class _Value:
    value: str


@dataclass(frozen=True)
class _Request:
    api_version: str = "db-api-v0.1"
    episode_id: str = "d0-test_id-t3_full-00000-s1785018189"
    task_id: str = "t3_full"
    split: Any = _Value("test_id")
    seed: int = 1785018189
    action_mode: Any = _Value("E7")
    observation_track: Any = _Value("state")
    object_mode: str = "wand_loop"
    reset_mode: str = "full"
    factors: Any = None

    def __post_init__(self) -> None:
        if self.factors is None:
            object.__setattr__(
                self,
                "factors",
                {
                    "angular_speed_deg_s": -6.5,
                    "initial_phase_rad": 0.27431170755728007,
                    "speed_class": "normal",
                },
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(root: Path, request: _Request) -> Path:
    export_dir = root / f"00000-{request.episode_id}"
    export_dir.mkdir(parents=True)
    request_path = export_dir / "request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    sums_path = export_dir / "SHA256SUMS"
    sums_path.write_text("0" * 64 + "  request.json\n", encoding="ascii")
    state_sha256 = "a" * 64
    row = {
        "api_version": request.api_version,
        "episode_id": request.episode_id,
        "task_id": request.task_id,
        "split": request.split.value,
        "seed": request.seed,
        "action_mode": request.action_mode.value,
        "observation_track": request.observation_track.value,
        "object_mode": request.object_mode,
        "reset_mode": request.reset_mode,
        "factors": request.factors,
        "candidate_index": 0,
        "source_group_id": "d0-t3_full-test_id-id-00000",
        "pair_id": None,
        "pair_member_id": None,
        "export_dir": export_dir.name,
        "request_json_sha256": _sha256(request_path),
        "state_sha256": state_sha256,
        "sha256sums_sha256": _sha256(sums_path),
        "independent_reset": {"exact": True, "state_sha256": state_sha256},
    }
    payload = {
        "task_id": "t3_full",
        "split": "test_id",
        "manifest_seed": 20261040,
        "observation_track": "state",
        "action_mode": "E7",
        "dataset_manifest_version": "db0-dataset-manifest-v0.3",
        "start_index": 0,
        "episode_count": 1,
        "image_size": 64,
        "candidate_indices": [0],
        "episode_ids": [request.episode_id],
        "rows": [row],
    }
    document = {
        "schema_version": t3_full_export.SCHEMA_VERSION,
        "payload": payload,
        "payload_sha256": t3_full_export._canonical_sha256(payload),
    }
    path = root / "t3_full_export_manifest_v2.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def test_load_t3_full_export_row_requires_full_artifact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _Request()
    path = _write_manifest(tmp_path, request)
    monkeypatch.setattr(
        t3_full_export,
        "_load_artifacts",
        lambda _export_dir: SimpleNamespace(reset_request=request),
    )

    loaded = t3_full_export.load_t3_full_export_row(path, candidate_index=0)

    assert loaded.request is request
    assert loaded.manifest_seed == 20261040
    assert loaded.candidate_index == 0
    assert loaded.row["factors"] == request.factors
    assert loaded.export_dir.parent == tmp_path.resolve()


def test_load_t3_full_export_row_rejects_episode_or_factor_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _Request()
    path = _write_manifest(tmp_path, request)
    drifted = replace(
        request,
        episode_id="copied-row-zero",
        factors={**request.factors, "initial_phase_rad": 0.0},
    )
    monkeypatch.setattr(
        t3_full_export,
        "_load_artifacts",
        lambda _export_dir: SimpleNamespace(reset_request=drifted),
    )

    with pytest.raises(ValueError, match="reset identity mismatch"):
        t3_full_export.load_t3_full_export_row(path, candidate_index=0)


def test_load_t3_full_export_row_rejects_manifest_payload_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_manifest(tmp_path, _Request())
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["rows"][0]["seed"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        t3_full_export,
        "_load_artifacts",
        lambda _export_dir: pytest.fail("artifact loading must follow payload verification"),
    )

    with pytest.raises(ValueError, match="payload hash mismatch"):
        t3_full_export.load_t3_full_export_row(path, candidate_index=0)
