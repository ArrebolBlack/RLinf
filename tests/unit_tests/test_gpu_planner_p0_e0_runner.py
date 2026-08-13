# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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
