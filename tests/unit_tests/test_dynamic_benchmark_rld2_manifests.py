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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.embodiment.build_dynamic_benchmark_rld2_manifests import (
    CANDIDATE_SCHEMA,
    EXACT_TASKS,
    INPUT_SPEC_SCHEMA,
    LEGACY_TASKS,
    PROVENANCE_SCHEMA,
    ManifestBuildError,
    build_release,
    validate_release,
)

OLD_RLINF_COMMIT = "1" * 40
RLD2_RLINF_COMMIT = "2" * 40
BENCHMARK_COMMIT = "3" * 40
SOURCE_RLINF_COMMIT = "4" * 40
EVALUATOR_COMMIT = "5" * 40

SOURCE_BY_TASK = {
    "t1_xyz": ("RLE0", None),
    "t1_so3": ("RLOPT-SO3", "A4"),
    "t2_se3": ("RLOPT-SE3", "D1"),
    "p0_grasp": ("RLOPT-P0G", "A3"),
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_blob(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode("utf-8"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(
    *,
    experiment: str,
    arm: str | None,
    train_seed: int,
    policy_path: Path,
    policy_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "origin": {
            "experiment": experiment,
            "run": f"{experiment.lower()}-formal",
            "arm": arm,
            "train_seed": train_seed,
        },
        "checkpoint": {
            "id": "best_policy.pt",
            "step": 100,
            "path": str(policy_path),
            "sha256": policy_sha256,
        },
        "source": {
            "manifest_path": f"evidence/{experiment.lower()}-formal.json",
            "manifest_sha256": "6" * 64,
            "rlinf_commit": SOURCE_RLINF_COMMIT,
        },
        "runtime": {
            "id": "mujoco311-rs140-v1",
            "evaluator_rlinf_commit": EVALUATOR_COMMIT,
        },
        "benchmark": {"commit": BENCHMARK_COMMIT},
        "config": {"path": "config.yaml", "sha256": "7" * 64},
        "state_schema": {
            "schema_version": "dynamic-state-v1",
            "sha256": "8" * 64,
            "state_dim": 217,
            "mask_dim": 32,
            "embedded_normalizer": True,
        },
        "reward": {"contract": "dynamic-reward-v1", "sha256": "9" * 64},
        "selection": {
            "split": "validation",
            "rule": "frozen-formal-winner",
            "test_exposure": {"test_id": False, "test_ood": False},
        },
        "expansion": {
            "mode": "deterministic",
            "stochastic": False,
            "exploration_seed_offset": 0,
        },
    }


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    old_root = tmp_path / "old"
    old_policy_root = tmp_path / "old-policies"
    snapshots = {}
    for index, task in enumerate(LEGACY_TASKS):
        policy_path = old_policy_root / task / "best_policy.pt"
        policy_sha256 = _write_blob(policy_path, f"old-policy:{task}")
        payload = {
            "schema_version": CANDIDATE_SCHEMA,
            "task": task,
            "rlinf_commit": OLD_RLINF_COMMIT,
            "benchmark_commit": BENCHMARK_COMMIT,
            "candidates": [
                {"candidate_id": "planner", "kind": "planner"},
                {
                    "candidate_id": f"{task}-incumbent-deterministic",
                    "kind": "policy",
                    "policy_path": str(policy_path),
                    "policy_sha256": policy_sha256,
                    "stochastic": False,
                    "exploration_seed_offset": 0,
                },
            ],
        }
        manifest_path = old_root / f"{index:02d}-{task}" / "candidate_manifest.json"
        _write_json(manifest_path, payload)
        snapshots[str(manifest_path)] = manifest_path.read_bytes()

    additions = []
    for task, (experiment, arm) in SOURCE_BY_TASK.items():
        policies = []
        for seed in range(1, 6):
            policy_path = tmp_path / "new-policies" / task / f"seed-{seed}.pt"
            policy_sha256 = _write_blob(policy_path, f"new-policy:{task}:{seed}")
            policies.append(
                {
                    "policy_path": str(policy_path),
                    "policy_sha256": policy_sha256,
                    "residual_scale": 0.1 if task == "t2_se3" else None,
                    "provenance": _provenance(
                        experiment=experiment,
                        arm=arm,
                        train_seed=seed,
                        policy_path=policy_path,
                        policy_sha256=policy_sha256,
                    ),
                }
            )
        additions.append({"task": task, "policies": policies})
    spec = {
        "schema_version": INPUT_SPEC_SCHEMA,
        "release_id": "RLD2",
        "candidate_schema_version": CANDIDATE_SCHEMA,
        "rlinf_commit": RLD2_RLINF_COMMIT,
        "benchmark_commit": BENCHMARK_COMMIT,
        "stochastic_expansion": {
            "include_deterministic": True,
            "exploration_seed_offsets": [1, 2, 3, 4, 5, 6],
        },
        "additions": additions,
        "provenance_overrides": [],
        "planner_dominance": {},
    }
    spec_path = tmp_path / "rld2_input_spec.json"
    _write_json(spec_path, spec)
    return old_root, spec_path, snapshots


def _read_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_outputs_exact14_and_preserves_incumbents_and_old_inputs(
    tmp_path: Path,
) -> None:
    old_root, spec_path, old_snapshots = _make_inputs(tmp_path)
    output = tmp_path / "release"

    result = build_release(
        old_manifest_root=old_root,
        input_spec=spec_path,
        output_root=output,
    )
    validation = validate_release(output)

    assert result["status"] == "built"
    assert validation["task_count"] == 14
    manifests = list(output.rglob("candidate_manifest.json"))
    assert {path.parent.name for path in manifests} == set(EXACT_TASKS)
    assert len(manifests) == 14
    xyz = json.loads((output / "t1_xyz" / "candidate_manifest.json").read_text())
    assert len(xyz["candidates"]) == 36
    assert xyz["candidates"][0]["kind"] == "planner"
    so3 = json.loads((output / "t1_so3" / "candidate_manifest.json").read_text())
    assert so3["candidates"][1]["candidate_id"] == "t1_so3-incumbent-deterministic"
    assert len(so3["candidates"]) == 37
    assert all("provenance" in candidate for candidate in so3["candidates"])
    assert (output / "input_inventory.jsonl").is_file()
    assert (output / "INPUTS.sha256").is_file()
    for path_value, before in old_snapshots.items():
        assert Path(path_value).read_bytes() == before


def test_missing_t1_xyz_addition_fails_closed(tmp_path: Path) -> None:
    old_root, spec_path, _ = _make_inputs(tmp_path)
    spec = _read_spec(spec_path)
    spec["additions"] = [row for row in spec["additions"] if row["task"] != "t1_xyz"]
    _write_json(spec_path, spec)

    with pytest.raises(ManifestBuildError, match="missing=.*t1_xyz"):
        build_release(
            old_manifest_root=old_root,
            input_spec=spec_path,
            output_root=tmp_path / "release",
        )


def test_policy_hash_mismatch_fails_before_output_creation(tmp_path: Path) -> None:
    old_root, spec_path, _ = _make_inputs(tmp_path)
    spec = _read_spec(spec_path)
    spec["additions"][0]["policies"][0]["policy_sha256"] = "0" * 64
    _write_json(spec_path, spec)
    output = tmp_path / "release"

    with pytest.raises(ManifestBuildError, match="hash mismatch"):
        build_release(
            old_manifest_root=old_root,
            input_spec=spec_path,
            output_root=output,
        )
    assert not output.exists()


def test_duplicate_policy_and_semantics_are_deduplicated_with_first_provenance(
    tmp_path: Path,
) -> None:
    old_root, spec_path, _ = _make_inputs(tmp_path)
    spec = _read_spec(spec_path)
    xyz = next(row for row in spec["additions"] if row["task"] == "t1_xyz")
    first = xyz["policies"][0]
    second = xyz["policies"][1]
    second["policy_path"] = first["policy_path"]
    second["policy_sha256"] = first["policy_sha256"]
    second["provenance"]["checkpoint"]["path"] = first["policy_path"]
    second["provenance"]["checkpoint"]["sha256"] = first["policy_sha256"]
    _write_json(spec_path, spec)
    output = tmp_path / "release"

    result = build_release(
        old_manifest_root=old_root,
        input_spec=spec_path,
        output_root=output,
    )

    manifest = json.loads((output / "t1_xyz" / "candidate_manifest.json").read_text())
    assert len(manifest["candidates"]) == 29
    assert result["deduplicated_count"] == 7
    semantics = [
        (
            row.get("policy_sha256"),
            row.get("stochastic", False),
            row.get("exploration_seed_offset", 0),
            row.get("residual_scale"),
        )
        for row in manifest["candidates"]
        if row["kind"] == "policy"
    ]
    assert len(semantics) == len(set(semantics))


def test_dry_run_validates_inputs_without_creating_output(tmp_path: Path) -> None:
    old_root, spec_path, _ = _make_inputs(tmp_path)
    output = tmp_path / "release"

    result = build_release(
        old_manifest_root=old_root,
        input_spec=spec_path,
        output_root=output,
        dry_run=True,
    )

    assert result["status"] == "dry-run"
    assert result["task_count"] == 14
    assert not output.exists()


def test_validate_only_detects_policy_hash_drift(tmp_path: Path) -> None:
    old_root, spec_path, _ = _make_inputs(tmp_path)
    output = tmp_path / "release"
    build_release(
        old_manifest_root=old_root,
        input_spec=spec_path,
        output_root=output,
    )
    spec = _read_spec(spec_path)
    policy_path = Path(spec["additions"][0]["policies"][0]["policy_path"])
    policy_path.write_bytes(b"tampered")

    with pytest.raises(
        ManifestBuildError, match="policy hash mismatch|input hash mismatch"
    ):
        validate_release(output)


def test_production_validation_rejects_null_incumbent_provenance(
    tmp_path: Path,
) -> None:
    old_root, spec_path, _ = _make_inputs(tmp_path)
    spec = _read_spec(spec_path)
    spec["planner_dominance"] = {
        task: {"schema_version": "test-planner-dominance-v0"} for task in EXACT_TASKS
    }
    _write_json(spec_path, spec)

    with pytest.raises(ManifestBuildError, match="production provenance"):
        build_release(
            old_manifest_root=old_root,
            input_spec=spec_path,
            output_root=tmp_path / "release",
            production=True,
        )
