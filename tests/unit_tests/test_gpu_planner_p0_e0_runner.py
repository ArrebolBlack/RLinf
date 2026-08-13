# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = (
    Path(__file__).parents[2] / "examples" / "embodiment" / "run_gpu_planner_p0_e0.py"
)
_SPEC = importlib.util.spec_from_file_location("_p0_e0_runner_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _pending_bundle(root: Path) -> None:
    instructions = root / "import_instructions.md"
    instructions.write_text("owner import\n", encoding="utf-8")
    (root / "artifact.json").write_text('{"complete":true}\n', encoding="utf-8")
    checksums = _MODULE._write_checksums(root)
    _MODULE._write_json(
        root / "bundle.json",
        {
            "schema_version": _MODULE.SCHEMA_VERSION,
            "runtime": {"runtime_ledger_lease_released": False},
            "checksums": {
                "path": "SHA256SUMS",
                "sha256": _MODULE._file_sha256(checksums),
            },
            "import_instructions": {
                "path": "import_instructions.md",
                "sha256": _MODULE._file_sha256(instructions),
            },
        },
    )


def test_finalize_bundle_changes_only_released_lease_receipt(tmp_path: Path) -> None:
    _pending_bundle(tmp_path)
    before = (tmp_path / "SHA256SUMS").read_bytes()

    assert _MODULE._finalize_bundle(tmp_path) == 0

    payload = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    assert payload["runtime"]["runtime_ledger_lease_released"] is True
    assert (tmp_path / "SHA256SUMS").read_bytes() == before
    _MODULE._verify_checksums(tmp_path)


def test_checksum_verifier_rejects_an_unlisted_bundle_artifact(tmp_path: Path) -> None:
    _pending_bundle(tmp_path)
    (tmp_path / "late-artifact.txt").write_text("not covered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="complete bundle"):
        _MODULE._verify_checksums(tmp_path)


def test_runner_uses_current_task_quality_schema() -> None:
    assert _MODULE.TASK_QUALITY_SCHEMA_VERSION == "db0-episode-task-quality-v2"


def test_runner_emits_development_export_source_contract() -> None:
    entry = SimpleNamespace(
        policy_step=0,
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        observation_fingerprint_sha256="1" * 64,
        health_after={"physics_step": 25},
    )
    tape = SimpleNamespace(
        entries=(entry,),
        sha256="3" * 64,
        as_dict=lambda: {"complete": True},
    )
    terminal = {
        "episode_id": "episode-0",
        "task_id": "p0_grasp",
        "terminated": True,
        "truncated": False,
        "success": True,
        "termination_reason": "success",
        "task_quality": None,
    }
    replay = SimpleNamespace(
        passed=True,
        backend_identity_sha256="4" * 64,
        steps=(),
    )

    action, trajectory, replay_payload = _MODULE._review_contract_payloads(
        tape=tape,
        terminal=terminal,
        replay=replay,
        terminal_observation_fingerprint="2" * 64,
        online_teacher_audits=2,
        online_transport_checks=1,
        replay_teacher_audits=2,
        replay_transport_checks=1,
    )

    expected_digest = "0266f17513a1845e570adb97e304703a61ee583d3b3cf53cc5191e135c3de866"
    assert _MODULE.SOURCE_EVALUATION_PHASE == "engineering_e0"
    assert action["action_tape_sha256"] == expected_digest
    assert trajectory["action_tape_sha256"] == expected_digest
    assert trajectory["observation_fingerprints"] == ["1" * 64, "2" * 64]
    assert trajectory["results"][0]["observation_fingerprint_sha256"] == "2" * 64
    assert trajectory["terminal"] == terminal
    assert replay_payload["action_count"] == 1
    assert replay_payload["action_tape_sha256"] == expected_digest
    assert replay_payload["observation_fingerprints"] == ["1" * 64, "2" * 64]
    assert replay_payload["terminal_quality_summary_sha256"] is None
