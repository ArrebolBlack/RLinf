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

from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
    _artifact_sha256,
)
from examples.embodiment.build_dynamic_benchmark_rld2_manifests import (
    EXACT_TASKS,
    LEGACY_CANDIDATE_SCHEMA,
    LEGACY_TASKS,
)
from examples.embodiment.prepare_dynamic_benchmark_rld2_launch_gate import (
    DEFAULT_LANES,
    LAUNCH_CANDIDATE_SCHEMA,
    PACKAGE_SCHEMA,
    REQUIRED_ADDITIONS,
    SOURCE_SPEC_SCHEMA,
    LaunchGateError,
    _build_package,
    validate_package,
)
from examples.embodiment.run_dynamic_benchmark_rld2_launch_gate import (
    _environment_config,
    copy_json,
)
from examples.embodiment.run_dynamic_benchmark_rld2_launch_gate import (
    _write_json as write_launch_result_json,
)

POLICY_COMMIT = "1" * 40
BENCHMARK_COMMIT = "2" * 40
EVALUATOR_COMMIT = "3" * 40
EVALUATOR_BENCHMARK_COMMIT = "4" * 40


def test_rld2_environment_config_explicitly_enables_task_quality() -> None:
    config = _environment_config(
        task="t1_xyz",
        split="validation",
        manifest_seed=20262150,
        image_size=64,
        policy=None,
        task_quality_schema_version="db0-episode-task-quality-v1",
        task_quality_evaluator_backend_id="mujoco311-rs140-v1-rld2-quality",
    )

    assert config["task_quality_schema_version"] == "db0-episode-task-quality-v1"
    assert (
        config["task_quality_evaluator_backend_id"]
        == "mujoco311-rs140-v1-rld2-quality"
    )


def test_task_quality_component_order_survives_json_round_trips(
    tmp_path: Path,
) -> None:
    value = {
        "components": {
            "terminal_goal_planar_error_m": {"value": 0.1},
            "maximum_rim_impulse_n_s": {"value": 0.2},
        }
    }

    copied = copy_json(value)
    output = tmp_path / "calibration_evidence.json"
    write_launch_result_json(output, copied)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert list(copied["components"]) == list(value["components"])
    assert list(persisted["components"]) == list(value["components"])
    assert hashlib.sha256(output.read_bytes()).hexdigest() == _artifact_sha256(copied)
    with pytest.raises(ValueError, match="Out of range float values"):
        copy_json({"components": {"quality": {"value": float("nan")}}})


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_blob(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _quality_schema(task: str) -> dict[str, Any]:
    components = [
        {
            "name": "quality",
            "direction": "maximize",
            "unit": "fraction",
            "scientific_resolution": 0.01,
            "reducer": "maximum",
            "source": "monitor.quality",
            "description": "Synthetic quality used by the launch-gate unit test.",
        }
    ]
    payload = {
        "schema_version": "task-quality-v0.1",
        "task_id": task,
        "task_config_sha256": hashlib.sha256(f"config:{task}".encode()).hexdigest(),
        "components": components,
    }
    return {
        **payload,
        "schema_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    old_root = tmp_path / "old"
    payloads: dict[str, dict[str, Any]] = {}
    old_hashes = {}
    for task in LEGACY_TASKS:
        policy = tmp_path / "policies" / "old" / task / "best_policy.pt"
        sha256 = _write_blob(policy, f"old:{task}")
        manifest = {
            "schema_version": LEGACY_CANDIDATE_SCHEMA,
            "task": task,
            "rlinf_commit": POLICY_COMMIT,
            "benchmark_commit": BENCHMARK_COMMIT,
            "candidates": [
                {"candidate_id": "planner", "kind": "planner"},
                {
                    "candidate_id": f"{task}-policy",
                    "kind": "policy",
                    "policy_path": str(policy),
                    "policy_sha256": sha256,
                    "stochastic": False,
                    "exploration_seed_offset": 0,
                },
            ],
        }
        path = old_root / task / "candidate_manifest.json"
        _write_json(path, manifest)
        old_hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
        payloads[str(policy.resolve())] = {
            "schema_version": "rlinf-dynamic-benchmark-expert-policy-v0.1",
            "config": {
                "task": task,
                "seed": 1,
                "rlinf_commit": POLICY_COMMIT,
                "benchmark_commit": BENCHMARK_COMMIT,
                "algorithm": "residual_rlpd",
                "residual_scale": 0.25,
            },
            "state_schema": {
                "schema_version": "state-v0.1",
                "task_id": task,
                "state_dim": 8,
                "mask_dim": 2,
            },
            "env_steps": 100,
            "model": {"weight": 1},
            "normalizer": {"mean": 0},
        }

    additions = []
    for task, (experiment, arm) in REQUIRED_ADDITIONS.items():
        policies = []
        for seed in range(1, 6):
            policy = tmp_path / "policies" / "new" / task / f"seed-{seed}.pt"
            sha256 = _write_blob(policy, f"new:{task}:{seed}")
            policies.append({"seed": seed, "path": str(policy), "sha256": sha256})
            payloads[str(policy.resolve())] = {
                "schema_version": "rlinf-dynamic-benchmark-expert-policy-v0.1",
                "config": {
                    "task": task,
                    "seed": seed,
                    "rlinf_commit": POLICY_COMMIT,
                    "benchmark_commit": BENCHMARK_COMMIT,
                    "algorithm": "residual_rlpd",
                    "residual_scale": 0.1 if task == "t2_se3" else 0.25,
                },
                "state_schema": {
                    "schema_version": "state-v0.1",
                    "task_id": task,
                    "state_dim": 8,
                    "mask_dim": 2,
                },
                "env_steps": 200,
                "model": {"weight": seed},
                "normalizer": {"mean": 0},
            }
        additions.append(
            {
                "task": task,
                "experiment": experiment,
                "arm": arm,
                "source_rlinf_commit": POLICY_COMMIT,
                "source_benchmark_commit": BENCHMARK_COMMIT,
                "residual_scale": 0.1 if task == "t2_se3" else 0.25,
                "policies": policies,
            }
        )
    source_spec = tmp_path / "source_spec.json"
    _write_json(
        source_spec,
        {
            "schema_version": SOURCE_SPEC_SCHEMA,
            "release_id": "RLD2",
            "old_manifest_sha256": old_hashes,
            "additions": additions,
        },
    )
    return old_root, source_spec, payloads


def _runtime_deps(tmp_path: Path) -> Path:
    root = tmp_path / "runtime-deps"
    dependency = root / "omegaconf" / "__init__.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text('__version__ = "2.3.1"\n', encoding="utf-8")
    sha256 = hashlib.sha256(dependency.read_bytes()).hexdigest()
    (root / "SHA256SUMS").write_text(
        f"{sha256}  omegaconf/__init__.py\n", encoding="utf-8"
    )
    return root


def test_prepare_exact14_launch_gate_and_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_root, source_spec, payloads = _inputs(tmp_path)
    runtime_deps = _runtime_deps(tmp_path)
    rlinf_root = tmp_path / "rlinf"
    se3_root = tmp_path / "se3"
    rlinf_root.mkdir()
    se3_root.mkdir()

    monkeypatch.setattr(
        "examples.embodiment.prepare_dynamic_benchmark_rld2_launch_gate._git_identity",
        lambda root, commit, label: {"path": str(root.resolve()), "commit": commit},
    )

    output = tmp_path / "launch"
    result = _build_package(
        old_manifest_root=old_root,
        source_spec_path=source_spec,
        output_root=output,
        rlinf_source_root=rlinf_root,
        rlinf_commit=EVALUATOR_COMMIT,
        se3_source_root=se3_root,
        se3_commit=EVALUATOR_BENCHMARK_COMMIT,
        runtime_deps_root=runtime_deps,
        backend_id="mujoco311-rs140-v1-rld2-quality",
        lanes=DEFAULT_LANES,
        manifest_seed=20262150,
        checkpoint_loader=lambda path: payloads[str(path.resolve())],
        quality_schema_loader=_quality_schema,
    )

    assert result["task_count"] == 14
    assert result["unique_checkpoint_count"] == len(LEGACY_TASKS) + 20
    validated = validate_package(output)
    assert validated["task_count"] == 14
    package = json.loads((output / "launch_package.json").read_text())
    assert package["schema_version"] == PACKAGE_SCHEMA
    assert package["status"] == "blocked-awaiting-allocation"
    assert package["production_release"] is False
    assert package["allowed_lanes"] == list(DEFAULT_LANES)
    assert tuple(package["lanes"]) == DEFAULT_LANES
    assert "forbidden_lane" not in package
    assert package["runtime_deps"]["path"] == str(runtime_deps.resolve())
    manifests = {
        path.parent.name: json.loads(path.read_text())
        for path in output.glob("candidates/*/candidate_manifest.json")
    }
    assert set(manifests) == set(EXACT_TASKS)
    assert all(
        manifest["schema_version"] == LAUNCH_CANDIDATE_SCHEMA
        and manifest["production_release"] is False
        for manifest in manifests.values()
    )
    assert len(manifests["t1_xyz"]["candidates"]) == 36
    assert len(manifests["t1_so3"]["candidates"]) == 37
    assert len(package["lanes"]) == 8
    assert sum(
        row["calibration_job_count"] for row in package["lanes"].values()
    ) == 14


def test_prepare_refuses_overwrite_and_non_exact_lane_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_root, source_spec, payloads = _inputs(tmp_path)
    runtime_deps = _runtime_deps(tmp_path)
    rlinf_root = tmp_path / "rlinf"
    se3_root = tmp_path / "se3"
    rlinf_root.mkdir()
    se3_root.mkdir()
    monkeypatch.setattr(
        "examples.embodiment.prepare_dynamic_benchmark_rld2_launch_gate._git_identity",
        lambda root, commit, label: {"path": str(root.resolve()), "commit": commit},
    )
    output = tmp_path / "launch"
    output.mkdir()
    kwargs = {
        "old_manifest_root": old_root,
        "source_spec_path": source_spec,
        "output_root": output,
        "rlinf_source_root": rlinf_root,
        "rlinf_commit": EVALUATOR_COMMIT,
        "se3_source_root": se3_root,
        "se3_commit": EVALUATOR_BENCHMARK_COMMIT,
        "runtime_deps_root": runtime_deps,
        "backend_id": "mujoco311-rs140-v1-rld2-quality",
        "manifest_seed": 20262150,
        "checkpoint_loader": lambda path: payloads[str(path.resolve())],
        "quality_schema_loader": _quality_schema,
    }
    with pytest.raises(LaunchGateError, match="overwrite"):
        _build_package(**kwargs, lanes=DEFAULT_LANES)
    output.rmdir()
    invalid_lane_sets = (
        DEFAULT_LANES[:-1],
        DEFAULT_LANES + ("L7",),
        tuple(reversed(DEFAULT_LANES)),
    )
    for lanes in invalid_lane_sets:
        with pytest.raises(LaunchGateError, match="exactly match L0-L7"):
            _build_package(**kwargs, lanes=lanes)


def test_package_validation_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_root, source_spec, payloads = _inputs(tmp_path)
    runtime_deps = _runtime_deps(tmp_path)
    rlinf_root = tmp_path / "rlinf"
    se3_root = tmp_path / "se3"
    rlinf_root.mkdir()
    se3_root.mkdir()
    monkeypatch.setattr(
        "examples.embodiment.prepare_dynamic_benchmark_rld2_launch_gate._git_identity",
        lambda root, commit, label: {"path": str(root.resolve()), "commit": commit},
    )
    output = tmp_path / "launch"
    _build_package(
        old_manifest_root=old_root,
        source_spec_path=source_spec,
        output_root=output,
        rlinf_source_root=rlinf_root,
        rlinf_commit=EVALUATOR_COMMIT,
        se3_source_root=se3_root,
        se3_commit=EVALUATOR_BENCHMARK_COMMIT,
        runtime_deps_root=runtime_deps,
        backend_id="mujoco311-rs140-v1-rld2-quality",
        lanes=DEFAULT_LANES,
        manifest_seed=20262150,
        checkpoint_loader=lambda path: payloads[str(path.resolve())],
        quality_schema_loader=_quality_schema,
    )
    manifest = output / "candidates" / "t1_xyz" / "candidate_manifest.json"
    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")
    with pytest.raises(LaunchGateError, match="SHA256SUMS mismatch"):
        validate_package(output)
