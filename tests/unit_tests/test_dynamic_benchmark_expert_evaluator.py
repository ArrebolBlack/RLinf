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

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_dynamic_benchmark_checkpoint_selection_outcome as outcome_fixtures
import torch

from examples.embodiment import (
    dynamic_benchmark_checkpoint_admission as checkpoint_admission,
)
from examples.embodiment import evaluate_dynamic_benchmark_expert as expert_evaluator
from examples.embodiment import train_dynamic_benchmark_expert as expert_trainer
from examples.embodiment.build_dynamic_benchmark_checkpoint_selection_outcome import (
    write_checkpoint_selection_outcome,
)
from examples.embodiment.dynamic_benchmark_evaluation_attempt import (
    attempt_schema_version,
    materialize_evaluation_attempt,
    recursive_output_checksums,
    validate_formal_quality_v2_thresholds,
)
from examples.embodiment.evaluate_dynamic_benchmark_expert import (
    EVALUATION_SCHEMA,
    FORMAL_EVALUATION_SCHEMA,
    TASK_QUALITY_BACKEND_ID,
    _aggregate_task_quality_values,
    _ArmedResetReplayEnv,
    _evaluation_schema,
    _expected_sha256,
    _latency_summary,
    _load_inference_policy,
    _model_kwargs,
    _replay_actions_on_fresh_env,
    _reset_rollout_on_fresh_env,
    _task_quality_aggregates,
    _task_quality_env_config,
    _task_quality_from_terminal_infos,
    _task_quality_identity,
    _task_summary,
    _validate_policy_payload,
    validate_selected_learned_policy,
)


def _payload(algorithm: str = "rlpd") -> dict:
    return {
        "schema_version": "rlinf-dynamic-benchmark-expert-policy-v0.1",
        "config": {
            "task": "t1_xyz",
            "algorithm": algorithm,
            "rlinf_commit": "a" * 40,
            "benchmark_commit": "b" * 40,
            "q_heads": 10,
        },
        "state_schema": {"state_dim": 173, "mask_dim": 35},
        "model": {"fixture": 1},
        "normalizer": {"fixture": 2},
    }


def _trainer_selection_artifacts(
    root: Path, *, eligible: bool = True
) -> SimpleNamespace:
    """Use the canonical producer fixture, including clean source roots/metrics."""

    run = outcome_fixtures._trainer_run(root, eligible=eligible)
    outcome_path = write_checkpoint_selection_outcome(
        **outcome_fixtures._write_kwargs(run)
    )
    return SimpleNamespace(
        policy_path=run.best_policy_path,
        initial_policy_path=run.initial_policy_path,
        summary_path=run.summary_path,
        selection_path=run.selection_path,
        metrics_path=run.metrics_path,
        outcome_path=outcome_path,
        outcome_sha256=expert_trainer._file_sha256(outcome_path),
        policy_root=run.policy_root,
        verifier_root=run.verifier_root,
        evaluator_root=run.evaluator_root,
        policy_commit=run.policy_commit,
        verifier_commit=run.verifier_commit,
        evaluator_commit=run.evaluator_commit,
    )


def _reseal_trainer_json(path: Path, payload: dict) -> None:
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = expert_trainer._canonical_json_sha256(payload)
    expert_trainer._atomic_json(path, payload)


def _outcome_admission_kwargs(artifacts: SimpleNamespace) -> dict:
    return {
        "checkpoint_selection_outcome_path": artifacts.outcome_path,
        "policy_rlinf_source_root": artifacts.policy_root,
        "verifier_rlinf_source_root": artifacts.verifier_root,
        "evaluator_rlinf_source_root": artifacts.evaluator_root,
        "expected_checkpoint_selection_outcome_sha256": artifacts.outcome_sha256,
        "expected_rlinf_commit": artifacts.policy_commit,
        "expected_benchmark_commit": "b" * 40,
        "expected_verifier_rlinf_commit": artifacts.verifier_commit,
        "expected_evaluator_rlinf_commit": artifacts.evaluator_commit,
    }


def test_evaluator_admits_real_trainer_selected_snapshot(tmp_path: Path) -> None:
    artifacts = _trainer_selection_artifacts(tmp_path / "run")

    admission = validate_selected_learned_policy(
        policy_path=artifacts.policy_path,
        trainer_summary_path=artifacts.summary_path,
        checkpoint_selection_path=artifacts.selection_path,
        **_outcome_admission_kwargs(artifacts),
        expected_policy_sha256=expert_trainer._file_sha256(artifacts.policy_path),
    )

    assert admission["checkpoint_role"] == "best"
    assert admission["policy"]["env_steps"] == 100
    assert admission["checkpoint_selection"]["status"] == ("selected_eligible_snapshot")
    assert set(admission["checkpoint_selection_outcome"]) == {
        "path",
        "sha256",
        "schema_version",
        "payload_sha256",
    }
    assert admission["checkpoint_selection_outcome"]["sha256"] == (
        artifacts.outcome_sha256
    )
    assert admission["source_identity"]["algorithm"] == "residual_rlpd"


@pytest.mark.parametrize("case", ["missing", "symlink", "wrong_sha"])
def test_evaluator_rejects_missing_symlinked_or_wrong_sha_outcome(
    tmp_path: Path, case: str
) -> None:
    artifacts = _trainer_selection_artifacts(tmp_path / case)
    kwargs = _outcome_admission_kwargs(artifacts)
    if case == "missing":
        artifacts.outcome_path.unlink()
        expected_error: type[Exception] = FileNotFoundError
    elif case == "symlink":
        target = artifacts.outcome_path.with_name("outcome-copy.json")
        target.write_bytes(artifacts.outcome_path.read_bytes())
        artifacts.outcome_path.unlink()
        try:
            artifacts.outcome_path.symlink_to(target.name)
        except OSError as error:
            pytest.skip(f"symlink creation is unavailable: {error}")
        expected_error = ValueError
    else:
        kwargs["expected_checkpoint_selection_outcome_sha256"] = "0" * 64
        expected_error = ValueError

    with pytest.raises(expected_error):
        validate_selected_learned_policy(
            policy_path=artifacts.policy_path,
            trainer_summary_path=artifacts.summary_path,
            checkpoint_selection_path=artifacts.selection_path,
            **kwargs,
        )


def test_evaluator_rejects_resealed_cross_run_and_source_drift_outcomes(
    tmp_path: Path,
) -> None:
    first = _trainer_selection_artifacts(tmp_path / "first")
    second = _trainer_selection_artifacts(tmp_path / "second")
    first.outcome_path.write_bytes(second.outcome_path.read_bytes())
    first.outcome_sha256 = expert_trainer._file_sha256(first.outcome_path)
    with pytest.raises(ValueError, match="authoritative producer replay"):
        validate_selected_learned_policy(
            policy_path=first.policy_path,
            trainer_summary_path=first.summary_path,
            checkpoint_selection_path=first.selection_path,
            **_outcome_admission_kwargs(first),
        )

    drift = _trainer_selection_artifacts(tmp_path / "source-drift")
    outcome = json.loads(drift.outcome_path.read_text(encoding="utf-8"))
    outcome["source_identity"]["evaluator_source"]["sha256"] = "f" * 64
    _reseal_trainer_json(drift.outcome_path, outcome)
    drift.outcome_sha256 = expert_trainer._file_sha256(drift.outcome_path)
    with pytest.raises(ValueError, match="authoritative producer replay"):
        validate_selected_learned_policy(
            policy_path=drift.policy_path,
            trainer_summary_path=drift.summary_path,
            checkpoint_selection_path=drift.selection_path,
            **_outcome_admission_kwargs(drift),
        )


@pytest.mark.parametrize("target", ["metrics", "baseline", "snapshot"])
def test_evaluator_rejects_resealed_outcome_evidence_drift(
    tmp_path: Path, target: str
) -> None:
    artifacts = _trainer_selection_artifacts(tmp_path / target)
    outcome = json.loads(artifacts.outcome_path.read_text(encoding="utf-8"))
    if target == "metrics":
        events = [
            json.loads(line)
            for line in artifacts.metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        events[-1]["success_rate"] = 0.125
        artifacts.metrics_path.write_bytes(
            "".join(
                json.dumps(event, sort_keys=True) + "\n" for event in events
            ).encode("utf-8")
        )
        outcome["trainer_artifacts"]["metrics"]["sha256"] = expert_trainer._file_sha256(
            artifacts.metrics_path
        )
        outcome["trainer_artifacts"]["metrics"]["validation_event_inventory_sha256"] = (
            "f" * 64
        )
    elif target == "baseline":
        outcome["matched_planner_baseline"]["validation_evidence"]["line_number"] += 1
    else:
        selector_metrics = outcome["evaluated_snapshots"][0]["selector_metrics"]
        selector_metrics["success_rate"] = 0.125
        outcome["evaluated_snapshots"][0]["selector_metrics_sha256"] = (
            expert_trainer._canonical_json_sha256(selector_metrics)
        )
    _reseal_trainer_json(artifacts.outcome_path, outcome)
    artifacts.outcome_sha256 = expert_trainer._file_sha256(artifacts.outcome_path)

    with pytest.raises(ValueError):
        validate_selected_learned_policy(
            policy_path=artifacts.policy_path,
            trainer_summary_path=artifacts.summary_path,
            checkpoint_selection_path=artifacts.selection_path,
            **_outcome_admission_kwargs(artifacts),
        )


def test_evaluator_rejects_wrong_policy_trainer_commit_semantics(
    tmp_path: Path,
) -> None:
    artifacts = _trainer_selection_artifacts(tmp_path / "wrong-trainer-commit")
    kwargs = _outcome_admission_kwargs(artifacts)
    kwargs["expected_rlinf_commit"] = artifacts.evaluator_commit

    with pytest.raises(ValueError, match="policy RLinf source-root HEAD"):
        validate_selected_learned_policy(
            policy_path=artifacts.policy_path,
            trainer_summary_path=artifacts.summary_path,
            checkpoint_selection_path=artifacts.selection_path,
            **kwargs,
        )


def test_evaluator_revalidation_rejects_policy_source_mutation(
    tmp_path: Path,
) -> None:
    artifacts = _trainer_selection_artifacts(tmp_path / "source-mutation")
    validate_selected_learned_policy(
        policy_path=artifacts.policy_path,
        trainer_summary_path=artifacts.summary_path,
        checkpoint_selection_path=artifacts.selection_path,
        **_outcome_admission_kwargs(artifacts),
    )
    trainer_source = (
        artifacts.policy_root
        / "examples"
        / "embodiment"
        / "train_dynamic_benchmark_expert.py"
    )
    with trainer_source.open("a", encoding="utf-8") as stream:
        stream.write("\n# mutation after admission preflight\n")

    with pytest.raises(ValueError, match="policy RLinf source root must be clean"):
        validate_selected_learned_policy(
            policy_path=artifacts.policy_path,
            trainer_summary_path=artifacts.summary_path,
            checkpoint_selection_path=artifacts.selection_path,
            **_outcome_admission_kwargs(artifacts),
        )


def test_trainer_loader_uses_authenticated_bytes_after_source_path_swap(
    tmp_path: Path,
) -> None:
    artifacts = _trainer_selection_artifacts(tmp_path / "authenticated-trainer-bytes")
    commit, sources = checkpoint_admission._verify_source_checkout(
        root=artifacts.policy_root,
        expected_commit=artifacts.policy_commit,
        sources={
            "trainer_source": (
                "examples.embodiment.train_dynamic_benchmark_expert",
                "examples/embodiment/train_dynamic_benchmark_expert.py",
            )
        },
        label="policy RLinf",
    )
    assert commit == artifacts.policy_commit
    authenticated = sources["trainer_source"]
    authenticated.path.write_text(
        "raise RuntimeError('untrusted swapped trainer source')\n", encoding="utf-8"
    )

    loaded = checkpoint_admission._load_authoritative_trainer(
        authenticated.content, source_path=authenticated.path
    )

    assert loaded._canonical_json_sha256({"authenticated": True})


def test_evaluator_rejects_wrong_actual_source_path(tmp_path: Path) -> None:
    declared_root = Path(expert_evaluator.__file__).resolve().parents[2]
    original_file = expert_evaluator.__file__
    wrong_source = tmp_path / "evaluate_dynamic_benchmark_expert.py"
    wrong_source.write_bytes(Path(original_file).read_bytes())
    expert_evaluator.__file__ = str(wrong_source)
    try:
        with pytest.raises(ValueError, match="executing evaluator source path"):
            expert_evaluator._actual_evaluator_source_identity(declared_root)
    finally:
        expert_evaluator.__file__ = original_file


def _source_snapshot_manifest_fixture(
    root: Path,
) -> tuple[Path, str, str, dict[str, tuple[Path, str, str]]]:
    base_image_id = "sha256:" + "d" * 64
    source_specs = {
        "policy_rlinf": ("a" * 40, "1" * 40),
        "evaluator_rlinf": ("b" * 40, "2" * 40),
        "benchmark": ("c" * 40, "3" * 40),
    }
    sources = {}
    expected_sources = {}
    for role, (commit, tree) in source_specs.items():
        source_root = root / "source" / role
        source_root.mkdir(parents=True)
        relative = f"src/{role}.py"
        source_path = source_root / relative
        source_path.parent.mkdir()
        source_path.write_text(f"ROLE = {role!r}\n", encoding="utf-8")
        row = {
            "path": relative,
            "mode": "100644",
            "git_blob_sha1": checkpoint_admission._git_blob_sha1(
                source_path.read_bytes()
            ),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
        sources[role] = {
            "root": str(source_root.resolve()),
            "commit": commit,
            "tree": tree,
            "files": [row],
            "inventory_sha256": checkpoint_admission._canonical_json_sha256([row]),
        }
        expected_sources[role] = (source_root, commit, tree)
    dependencies = {}
    for name in ("portable", "a800_core"):
        dependency_root = root / "runtime" / name
        dependency_root.mkdir(parents=True)
        dependency_file = dependency_root / "dependency.py"
        dependency_file.write_text(f"NAME = {name!r}\n", encoding="utf-8")
        rows = [
            {
                "path": "dependency.py",
                "mode": "100644",
                "sha256": hashlib.sha256(dependency_file.read_bytes()).hexdigest(),
            }
        ]
        dependencies[name] = {
            "root": str(dependency_root.resolve()),
            "inventory_sha256": checkpoint_admission._canonical_json_sha256(rows),
        }
    manifest = {
        "schema_version": checkpoint_admission.SOURCE_SNAPSHOT_SCHEMA,
        "base_image_id": base_image_id,
        "sources": sources,
        "runtime_dependencies": dependencies,
    }
    manifest["payload_sha256"] = checkpoint_admission._canonical_json_sha256(manifest)
    manifest_path = root / checkpoint_admission.SOURCE_SNAPSHOT_MANIFEST_FILENAME
    expert_trainer._atomic_json(manifest_path, manifest)
    return (
        manifest_path,
        expert_trainer._file_sha256(manifest_path),
        base_image_id,
        expected_sources,
    )


def test_source_snapshot_manifest_rejects_file_and_image_drift(tmp_path: Path) -> None:
    manifest_path, manifest_sha, base_image_id, expected_sources = (
        _source_snapshot_manifest_fixture(tmp_path / "snapshot")
    )
    checkpoint_admission.validate_source_snapshot_manifest(
        manifest_path,
        expected_sha256=manifest_sha,
        expected_base_image_id=base_image_id,
        expected_sources=expected_sources,
        verify_inventory=True,
    )
    with pytest.raises(ValueError, match="base image ID mismatch"):
        checkpoint_admission.validate_source_snapshot_manifest(
            manifest_path,
            expected_sha256=manifest_sha,
            expected_base_image_id="sha256:" + "e" * 64,
            expected_sources=expected_sources,
            verify_inventory=True,
        )
    policy_file = (
        tmp_path / "snapshot" / "source" / "policy_rlinf" / "src" / ("policy_rlinf.py")
    )
    policy_file.write_text("ROLE = 'mutated'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file identity mismatch"):
        checkpoint_admission.validate_source_snapshot_manifest(
            manifest_path,
            expected_sha256=manifest_sha,
            expected_base_image_id=base_image_id,
            expected_sources=expected_sources,
            verify_inventory=True,
        )


def test_source_snapshot_manifest_rejects_resealed_git_blob_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, base_image_id, expected_sources = (
        _source_snapshot_manifest_fixture(tmp_path / "snapshot-git-blob")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["sources"]["policy_rlinf"]["files"]
    rows[0]["git_blob_sha1"] = "f" * 40
    manifest["sources"]["policy_rlinf"]["inventory_sha256"] = (
        checkpoint_admission._canonical_json_sha256(rows)
    )
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256")
    manifest["payload_sha256"] = checkpoint_admission._canonical_json_sha256(unsigned)
    expert_trainer._atomic_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="file identity mismatch"):
        checkpoint_admission.validate_source_snapshot_manifest(
            manifest_path,
            expected_sha256=expert_trainer._file_sha256(manifest_path),
            expected_base_image_id=base_image_id,
            expected_sources=expected_sources,
            verify_inventory=True,
        )
    manifest_sha = expert_trainer._file_sha256(manifest_path)
    monkeypatch.setenv("RLD2_SOURCE_SNAPSHOT_MANIFEST", str(manifest_path))
    monkeypatch.setenv("RLD2_SOURCE_SNAPSHOT_MANIFEST_SHA256", manifest_sha)
    policy_root, policy_commit, _ = expected_sources["policy_rlinf"]
    with pytest.raises(ValueError, match="differs from its snapshot blob"):
        checkpoint_admission._verify_source_checkout(
            root=policy_root,
            expected_commit=policy_commit,
            sources={"policy": ("snapshot.policy", "src/policy_rlinf.py")},
            label="policy RLinf",
        )


def test_source_snapshot_receipt_publishes_sibling_manifest_identity(
    tmp_path: Path,
) -> None:
    manifest_path, manifest_sha, base_image_id, expected_sources = (
        _source_snapshot_manifest_fixture(tmp_path / "snapshot-receipt")
    )
    manifest = checkpoint_admission.validate_source_snapshot_manifest(
        manifest_path,
        expected_sha256=manifest_sha,
        expected_base_image_id=base_image_id,
        expected_sources=expected_sources,
        verify_inventory=True,
    )
    evaluator_source = {
        "module": expert_evaluator.EVALUATOR_MODULE,
        "repository_path": expert_evaluator.EVALUATOR_REPOSITORY_PATH,
        "sha256": "f" * 64,
    }

    receipt = expert_evaluator._source_snapshot_receipt(
        manifest_sha256=manifest_sha,
        manifest=manifest,
        base_image_id=base_image_id,
        source_snapshot_image_id="sha256:" + "e" * 64,
        evaluator_source_identity=evaluator_source,
    )

    assert receipt["source_manifest"] == {
        "path": checkpoint_admission.SOURCE_SNAPSHOT_MANIFEST_FILENAME,
        "sha256": manifest_sha,
        "schema_version": checkpoint_admission.SOURCE_SNAPSHOT_SCHEMA,
        "payload_sha256": manifest["payload_sha256"],
    }
    assert receipt["evaluator_source"] == evaluator_source
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("payload_sha256")
    assert receipt["payload_sha256"] == expert_evaluator._payload_sha256(
        unsigned_receipt
    )
    with pytest.raises(ValueError, match="must differ"):
        expert_evaluator._source_snapshot_receipt(
            manifest_sha256=manifest_sha,
            manifest=manifest,
            base_image_id=base_image_id,
            source_snapshot_image_id=base_image_id,
            evaluator_source_identity=evaluator_source,
        )


@pytest.mark.parametrize("mutation", ["mutate", "delete"])
def test_evaluator_revalidation_rejects_outcome_mutation_or_deletion(
    tmp_path: Path, mutation: str
) -> None:
    artifacts = _trainer_selection_artifacts(tmp_path / mutation)
    admission = validate_selected_learned_policy(
        policy_path=artifacts.policy_path,
        trainer_summary_path=artifacts.summary_path,
        checkpoint_selection_path=artifacts.selection_path,
        **_outcome_admission_kwargs(artifacts),
    )
    if mutation == "mutate":
        artifacts.outcome_path.write_bytes(artifacts.outcome_path.read_bytes() + b"\n")
    else:
        artifacts.outcome_path.unlink()
    kwargs = _outcome_admission_kwargs(artifacts)
    kwargs["expected_checkpoint_selection_outcome_sha256"] = admission[
        "checkpoint_selection_outcome"
    ]["sha256"]
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_selected_learned_policy(
            policy_path=artifacts.policy_path,
            trainer_summary_path=artifacts.summary_path,
            checkpoint_selection_path=artifacts.selection_path,
            **kwargs,
        )


def test_evaluator_rejects_planner_fallback_and_initial_env0(tmp_path: Path) -> None:
    fallback = _trainer_selection_artifacts(tmp_path / "fallback", eligible=False)
    with pytest.raises(ValueError, match="planner_fallback_no_eligible"):
        validate_selected_learned_policy(
            policy_path=fallback.policy_path,
            trainer_summary_path=fallback.summary_path,
            checkpoint_selection_path=fallback.selection_path,
            **_outcome_admission_kwargs(fallback),
        )

    selected = _trainer_selection_artifacts(tmp_path / "selected")
    with pytest.raises(ValueError, match="checkpoint_role=best"):
        validate_selected_learned_policy(
            policy_path=selected.initial_policy_path,
            trainer_summary_path=selected.summary_path,
            checkpoint_selection_path=selected.selection_path,
            **_outcome_admission_kwargs(selected),
        )


@pytest.mark.parametrize(
    "field, message",
    [
        ("selected_snapshot_identity", "selected snapshot is null"),
        ("best_policy", "best policy is null"),
    ],
)
def test_evaluator_rejects_null_selected_or_best_identity(
    tmp_path: Path, field: str, message: str
) -> None:
    artifacts = _trainer_selection_artifacts(tmp_path / field)
    manifest = json.loads(artifacts.selection_path.read_text(encoding="utf-8"))
    manifest["selection"][field] = None
    _reseal_trainer_json(artifacts.selection_path, manifest)

    with pytest.raises(ValueError, match=f"{message}|identities diverged"):
        validate_selected_learned_policy(
            policy_path=artifacts.policy_path,
            trainer_summary_path=artifacts.summary_path,
            checkpoint_selection_path=artifacts.selection_path,
            **_outcome_admission_kwargs(artifacts),
        )


def test_evaluator_rejects_resealed_checkpoint_selection_tampering(
    tmp_path: Path,
) -> None:
    artifacts = _trainer_selection_artifacts(tmp_path / "run")
    manifest = json.loads(artifacts.selection_path.read_text(encoding="utf-8"))
    manifest["evaluated_snapshots"][0]["eligible"] = False
    _reseal_trainer_json(artifacts.selection_path, manifest)
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_selection"]["manifest_payload_sha256"] = manifest[
        "payload_sha256"
    ]
    _reseal_trainer_json(artifacts.summary_path, summary)

    with pytest.raises(ValueError, match="eligible snapshot count|select exactly"):
        validate_selected_learned_policy(
            policy_path=artifacts.policy_path,
            trainer_summary_path=artifacts.summary_path,
            checkpoint_selection_path=artifacts.selection_path,
            **_outcome_admission_kwargs(artifacts),
        )


def _quality_summary(
    episode_id: str,
    *,
    values: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    task_id: str = "t1_xyz",
    backend_id: str = TASK_QUALITY_BACKEND_ID,
    terminal: bool = True,
) -> dict:
    from se3_wam.benchmark.task_quality import (
        EpisodeQualitySummary,
        TaskQualityComponentValue,
        get_task_quality_schema,
    )

    schema = get_task_quality_schema(task_id)
    assert len(values) == len(schema.components)
    summary = EpisodeQualitySummary(
        episode_id=episode_id,
        task_id=task_id,
        evaluator_backend_id=backend_id,
        schema_version=schema.schema_version,
        schema_sha256=schema.schema_sha256,
        physics_sample_count=10,
        terminal=terminal,
        components={
            spec.name: TaskQualityComponentValue(
                value=value,
                direction=spec.direction,
                unit=spec.unit,
                scientific_resolution=spec.scientific_resolution,
                reducer=spec.reducer,
            )
            for spec, value in zip(schema.components, values, strict=True)
        },
    )
    return summary.to_dict()


def test_task_quality_config_covers_canonical_exact_14_inventory() -> None:
    from se3_wam.benchmark.registry import RL_EXPERT_TASK_IDS

    assert len(RL_EXPERT_TASK_IDS) == 14
    for task_id in RL_EXPERT_TASK_IDS:
        identity = _task_quality_identity(task_id)

        config = _task_quality_env_config(identity)

        assert config == {
            "task_quality_schema_version": identity["task_quality_schema"][
                "schema_version"
            ],
            "task_quality_evaluator_backend_id": TASK_QUALITY_BACKEND_ID,
        }
        assert (
            identity["task_quality_schema_sha256"]
            == identity["task_quality_schema"]["schema_sha256"]
        )


def test_terminal_task_quality_is_validated_and_recorded_canonically() -> None:
    identity = _task_quality_identity("t1_xyz")
    expected = _quality_summary("episode-1")

    recorded = _task_quality_from_terminal_infos(
        {"task_quality": [expected]},
        identity=identity,
        task_id="t1_xyz",
        episode_id="episode-1",
    )

    assert recorded == expected
    assert recorded["terminal"] is True
    assert recorded["schema_sha256"] == identity["task_quality_schema_sha256"]


def test_task_quality_aggregates_all_and_successful_records() -> None:
    identity = _task_quality_identity("t1_xyz")
    records = [
        {
            "episode_id": "episode-1",
            "task_id": "t1_xyz",
            "success": True,
            "safety_failure": False,
            "termination_reason": "success",
            "trajectory_completion": 1.0,
            "completion_time_s": 1.0,
            "return": 2.0,
            "action_l2_sum": 3.0,
            "task_quality": _quality_summary("episode-1", values=(0.1, 0.2, 0.3, 0.4)),
        },
        {
            "episode_id": "episode-2",
            "task_id": "t1_xyz",
            "success": False,
            "safety_failure": True,
            "termination_reason": "collision",
            "trajectory_completion": 0.5,
            "completion_time_s": None,
            "return": 1.0,
            "action_l2_sum": 2.0,
            "task_quality": _quality_summary("episode-2", values=(0.3, 0.4, 0.5, 0.6)),
        },
    ]

    task_summary = _task_summary(
        "t1_xyz",
        records,
        task_quality_identity=identity,
        quality_v2_enabled=False,
    )
    aggregates = task_summary["t1_xyz"]["task_quality"]

    assert set(task_summary) == {"t1_xyz"}
    component_names = [
        component["name"] for component in identity["task_quality_schema"]["components"]
    ]
    first = aggregates["components"][component_names[0]]
    assert aggregates["record_count"] == 2
    assert aggregates["successful_record_count"] == 1
    assert first["all_records"] == {
        "count": 2,
        "mean": pytest.approx(0.2),
        "minimum": pytest.approx(0.1),
        "maximum": pytest.approx(0.3),
    }
    assert first["successful_records"] == {
        "count": 1,
        "mean": pytest.approx(0.1),
        "minimum": pytest.approx(0.1),
        "maximum": pytest.approx(0.1),
    }
    assert _aggregate_task_quality_values([]) == {
        "count": 0,
        "mean": None,
        "minimum": None,
        "maximum": None,
    }
    assert (
        _task_quality_aggregates(
            records,
            identity=identity,
            task_id="t1_xyz",
        )
        == aggregates
    )


def test_task_quality_fails_closed_on_missing_or_mismatched_terminal_summary() -> None:
    identity = _task_quality_identity("t1_xyz")
    with pytest.raises(ValueError, match="missing task_quality"):
        _task_quality_from_terminal_infos(
            {},
            identity=identity,
            task_id="t1_xyz",
            episode_id="episode-1",
        )
    with pytest.raises(ValueError, match="no task-quality"):
        _task_quality_from_terminal_infos(
            {"task_quality": [None]},
            identity=identity,
            task_id="t1_xyz",
            episode_id="episode-1",
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        _task_quality_from_terminal_infos(
            {"task_quality": [_quality_summary("other-episode")]},
            identity=identity,
            task_id="t1_xyz",
            episode_id="episode-1",
        )

    tampered_identity = copy.deepcopy(identity)
    tampered_identity["task_quality_schema"]["task_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schema is not canonical"):
        _task_quality_env_config(tampered_identity)


def test_model_kwargs_reconstruct_sac_and_ppo_heads() -> None:
    sac = _model_kwargs(_payload()["config"], 173)
    ppo = _model_kwargs(_payload("ppo")["config"], 173)

    assert sac["add_q_head"] is True
    assert sac["add_value_head"] is False
    assert sac["num_q_heads"] == 10
    assert ppo["add_q_head"] is False
    assert ppo["add_value_head"] is True


def test_policy_payload_validation_is_commit_and_schema_fail_closed() -> None:
    payload = _payload()

    config, schema = _validate_policy_payload(
        payload,
        rlinf_commit="a" * 40,
        benchmark_commit="b" * 40,
    )

    assert config["task"] == "t1_xyz"
    assert schema["state_dim"] == 173
    with pytest.raises(ValueError, match="RLinf commit"):
        _validate_policy_payload(
            payload,
            rlinf_commit="c" * 40,
            benchmark_commit="b" * 40,
        )
    payload["schema_version"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        _validate_policy_payload(
            payload,
            rlinf_commit="a" * 40,
            benchmark_commit="b" * 40,
        )


def test_latency_summary_reports_20hz_p95_gate() -> None:
    fast = _latency_summary([0.001, 0.002, 0.003])
    slow = _latency_summary([0.001, 0.002, 0.100])

    assert fast["p95_meets_20hz"] is True
    assert slow["p95_meets_20hz"] is False
    assert fast["sample_count"] == 3


def test_expected_sha256_requires_full_lowercase_hex() -> None:
    assert _expected_sha256("a" * 64) == "a" * 64
    with pytest.raises(ValueError, match="64 lowercase"):
        _expected_sha256("abc")


def _actor_state(state_dim: int, *, ppo: bool) -> dict[str, torch.Tensor]:
    state = {
        "backbone.0.weight": torch.randn(256, state_dim),
        "backbone.0.bias": torch.randn(256),
        "backbone.2.weight": torch.randn(256, 256),
        "backbone.2.bias": torch.randn(256),
        "backbone.4.weight": torch.randn(256, 256),
        "backbone.4.bias": torch.randn(256),
        "actor_mean.weight": torch.randn(7, 256),
        "actor_mean.bias": torch.randn(7),
    }
    if ppo:
        state["actor_logstd"] = torch.randn(1, 7)
        state["value_head.net.0.weight"] = torch.randn(256, state_dim)
    else:
        state["actor_logstd.weight"] = torch.randn(7, 256)
        state["actor_logstd.bias"] = torch.randn(7)
        state["action_scale"] = torch.tensor(1.0)
        state["action_bias"] = torch.tensor(0.0)
        for index in range(10):
            state[f"q_head.qs.{index}.net.0.weight"] = torch.randn(256, state_dim + 7)
    return state


@pytest.mark.parametrize("algorithm", ["rlpd", "ppo"])
def test_inference_policy_reconstructs_actor_without_training_stack(
    algorithm: str,
) -> None:
    config = _payload(algorithm)["config"]
    state = _actor_state(173, ppo=algorithm == "ppo")

    model = _load_inference_policy(config, 173, state, torch.device("cpu"))
    mean, log_std = model._sample_actions(torch.zeros(2, 173))

    assert mean.shape == (2, 7)
    assert log_std.shape == (2, 7)
    assert model.training is False


def test_inference_policy_rejects_unknown_or_incomplete_heads() -> None:
    config = _payload()["config"]
    state = _actor_state(173, ppo=False)
    state["mystery.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="unsupported"):
        _load_inference_policy(config, 173, state, torch.device("cpu"))

    state = _actor_state(173, ppo=False)
    del state["q_head.qs.9.net.0.weight"]
    with pytest.raises(ValueError, match="Q-head count"):
        _load_inference_policy(config, 173, state, torch.device("cpu"))


def test_expert_rollout_and_replay_use_separate_fresh_raw_envs() -> None:
    class Request:
        task_id = "t5_replan"

    request = Request()

    class RawEnv:
        horizon_steps = 120

        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False
            self.reset_calls = []

        def reset(self, value):
            self.reset_calls.append(value)
            return f"{self.name}-observation"

        def step(self, action):
            return (self.name, action)

        def save_state(self):
            return self.name.encode()

        def close(self):
            self.closed = True

    class VectorEnv:
        num_envs = 1
        task_id = "t5_replan"
        image_size = 64
        camera_observations = False
        task_quality_schema_version = "quality-v1"
        task_quality_evaluator_backend_id = "backend-v1"
        horizon_steps = 120

        def __init__(self) -> None:
            self.previous = RawEnv("previous")
            self.rollout = RawEnv("rollout")
            self.replay = RawEnv("replay")
            self.envs = [self.previous]
            self._raw_observations = ["stale"]
            self._requests = [object()]
            self._needs_reset = np.asarray([True])
            self._last_obs = None
            self._is_start = False
            self.make_calls = []
            self.arm_calls = []
            self.reset_metric_calls = []

        def _make_mujoco_env(self, task_id, **kwargs):
            self.make_calls.append((task_id, kwargs))
            return self.rollout if len(self.make_calls) == 1 else self.replay

        def _arm_hidden_t5_event(self, raw_env, value):
            self.arm_calls.append((raw_env, value))

        def _encode(self, observation, value):
            assert observation == "rollout-observation"
            assert value is request
            return np.asarray([1.0, 2.0], dtype=np.float32)

        def _reset_metrics(self, indices):
            self.reset_metric_calls.append(indices.copy())

    vector_env = VectorEnv()
    observation = _reset_rollout_on_fresh_env(
        vector_env=vector_env,
        request=request,
    )

    assert observation == "rollout-observation"
    assert vector_env.previous.closed
    assert vector_env.envs == [vector_env.rollout]
    assert vector_env.rollout.reset_calls == [request]
    assert not vector_env._needs_reset[0]
    assert vector_env._last_obs["states"].tolist() == [[1.0, 2.0]]

    def replay_fn(proxy, **kwargs):
        assert isinstance(proxy, _ArmedResetReplayEnv)
        assert proxy.reset(request) == "replay-observation"
        assert proxy.step("action") == ("replay", "action")
        assert proxy.save_state() == b"replay"
        assert kwargs["request"] is request
        return {"passed": True}

    result = _replay_actions_on_fresh_env(
        vector_env=vector_env,
        task_id="t5_replan",
        request=request,
        expected_observations=("expected",),
        actions=("action",),
        expected_outcomes=("outcome",),
        expected_final_state=b"expected",
        replay_fn=replay_fn,
    )

    assert result == {"passed": True}
    assert vector_env.make_calls == [
        (
            "t5_replan",
            {
                "image_size": 64,
                "camera_observations": False,
                "task_quality_schema_version": "quality-v1",
                "task_quality_evaluator_backend_id": "backend-v1",
            },
        ),
        (
            "t5_replan",
            {
                "image_size": 64,
                "camera_observations": False,
                "task_quality_schema_version": "quality-v1",
                "task_quality_evaluator_backend_id": "backend-v1",
            },
        ),
    ]
    assert vector_env.replay.closed
    assert vector_env.arm_calls == [
        (vector_env.rollout, request),
        (vector_env.replay, request),
    ]


@pytest.mark.parametrize(
    "replay_values, expected_passed",
    [
        ((0.1, 0.2, 0.3, 0.4), True),
        ((0.3, 0.4, 0.5, 0.6), False),
    ],
)
def test_expert_replay_binds_terminal_task_quality_exactly(
    replay_values: tuple[float, ...], expected_passed: bool
) -> None:
    identity = _task_quality_identity("t1_xyz")
    expected = _quality_summary("episode-1", values=(0.1, 0.2, 0.3, 0.4))

    class Request:
        episode_id = "episode-1"
        task_id = "t1_xyz"

    request = Request()

    class RawEnv:
        def reset(self, value):
            assert value is request
            return "observation"

        def step(self, action):
            assert action == "action"
            return SimpleNamespace(
                terminated=True,
                truncated=False,
                task_quality=_quality_summary(
                    "episode-1",
                    values=replay_values,
                ),
            )

        def save_state(self):
            return b"state"

        def close(self):
            pass

    class VectorEnv:
        image_size = 64
        camera_observations = False
        task_quality_schema_version = identity["task_quality_schema"]["schema_version"]
        task_quality_evaluator_backend_id = TASK_QUALITY_BACKEND_ID

        def _make_mujoco_env(self, task_id, **kwargs):
            assert task_id == "t1_xyz"
            return RawEnv()

        def _arm_hidden_t5_event(self, raw_env, value):
            pass

    def replay_fn(proxy, **kwargs):
        proxy.reset(request)
        proxy.step("action")
        return {"passed": True, "final_state_exact": True, "outcomes_exact": True}

    validation = _replay_actions_on_fresh_env(
        vector_env=VectorEnv(),
        task_id="t1_xyz",
        request=request,
        expected_observations=("observation",),
        actions=("action",),
        expected_outcomes=((True, False, False, None, 0.0),),
        expected_final_state=b"state",
        expected_task_quality=expected,
        task_quality_identity=identity,
        replay_fn=replay_fn,
    )

    assert validation["task_quality_exact"] is expected_passed
    assert validation["passed"] is expected_passed
    assert (
        validation["task_quality_summary_sha256"]
        == _quality_summary("episode-1", values=replay_values)["summary_sha256"]
    )


def test_formal_producer_tape_passes_promotion_validator(tmp_path: Path) -> None:
    from se3_wam.benchmark.config import load_task_config
    from se3_wam.benchmark.trajectory_quality import (
        evaluate_quality_v2_gate,
        trajectory_quality_v2_from_observations,
    )

    from examples.embodiment import build_dynamic_benchmark_rld2_promotion as promotion

    task_id = "t1_xyz"
    episode_id = "validation-00"
    steps = 4
    assert _evaluation_schema(formal_attempts=False) == EVALUATION_SCHEMA
    assert _evaluation_schema(formal_attempts=True) == FORMAL_EVALUATION_SCHEMA
    assert EVALUATION_SCHEMA == "rlinf-dynamic-benchmark-expert-evaluation-v0.2"
    assert FORMAL_EVALUATION_SCHEMA == "rlinf-dynamic-benchmark-expert-evaluation-v0.3"
    observations = [
        SimpleNamespace(
            privileged={
                "eef_pose_xyzw": np.asarray(
                    [0.01 * index, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float64,
                ),
                "fingerpad_closing_axis_world": np.asarray(
                    [1.0, 0.0, 0.0], dtype=np.float64
                ),
                "object_pose_wxyz": np.asarray(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                ),
                "fingerpad_contact_flags": np.asarray(
                    [1.0, 1.0] if index >= 1 else [0.0, 0.0],
                    dtype=np.float64,
                ),
            },
            events_since_last_observation=(),
        )
        for index in range(steps + 1)
    ]
    actions = np.zeros((steps, 7), dtype=np.float64)
    quality = trajectory_quality_v2_from_observations(
        observations,
        actions,
        task_id=task_id,
        task_config=load_task_config(task_id),
        sample_period_s=0.05,
        continuous_dimensions=6,
    )

    metric_specs = (
        (
            "full_episode",
            "action.action_second_difference_l2_mean_per_transition",
            "action_l2",
        ),
        ("full_episode", "action.action_max_second_difference_l2", "action_l2"),
        (
            "full_episode",
            "action.action_total_variation_l2_mean_per_transition",
            "action_l2",
        ),
        (
            "full_episode",
            "eef_motion.eef_translation_path_length_m",
            "translation_path_m",
        ),
        (
            "full_episode",
            "eef_motion.eef_rotation_path_length_rad",
            "rotation_or_orientation_rad",
        ),
        (
            "full_episode",
            "eef_motion.eef_angular_jerk_max_rad_s3",
            "angular_jerk_rad_s3",
        ),
        (
            "full_episode",
            "eef_motion.eef_linear_jerk_max_m_s3",
            "linear_jerk_m_s3",
        ),
        (
            "full_episode",
            "eef_motion.eef_angular_jerk_rms_rad_s3",
            "angular_jerk_rad_s3",
        ),
        (
            "full_episode",
            "eef_motion.eef_linear_jerk_rms_m_s3",
            "linear_jerk_m_s3",
        ),
        (
            "acquisition_window",
            "approach_axis.approach_axis_error_max_rad",
            "rotation_or_orientation_rad",
        ),
        (
            "acquisition_window",
            "jaw_axis.jaw_axis_error_max_rad",
            "rotation_or_orientation_rad",
        ),
    )

    def metric_value(phase: str, path: str) -> float:
        value = quality if phase == "full_episode" else quality["phases"][phase]
        for part in path.split("."):
            value = value[part]
        return float(value)

    checks = [
        {
            "phase": phase,
            "metric": path,
            "max": metric_value(phase, path) + 1.0,
            "direction": "minimize",
            "paired_comparison_family": family,
            "paired_nonworse_absolute_tolerance": 0.01,
            "paired_nonworse_relative_tolerance": 0.0,
            "paired_strict_improvement_absolute": 0.02,
            "paired_strict_improvement_relative": 0.0,
        }
        for phase, path, family in metric_specs
    ]
    thresholds = {
        "schema_version": "se3-wam-trajectory-quality-v2-thresholds-v0.3",
        "formal_freeze_eligible": True,
        "calibration_status": "frozen",
        "minimum_attempted_episodes": 20,
        "minimum_successful_episodes": 8,
        "calibration_wave_receipt": {
            "binding_status": "bound",
            "schema_version": "rld2-qa-planner-calibration-wave-receipt-v0.1",
            "scientific_partition": "metric_calibration",
            "transport_split": "validation",
            "task_count": 14,
            "episodes_per_task": 20,
            "total_reset_count": 280,
            "relative_path": "provenance/calibration/wave_receipt.json",
            "file_sha256": "b" * 64,
            "payload_sha256": "b" * 64,
            "sha256": "b" * 64,
        },
        "tasks": {
            task_id: {
                "checks": checks,
                "orientation_mode": "world_down_tool_axis",
                "jaw_axis_mode": "object_xy_teacher_offset_mod_pi",
                "provenance": {
                    "formal_freeze_eligible": True,
                    "attempted_episode_count": 20,
                    "successful_episode_count": 20,
                },
            }
        },
    }
    threshold_sha256 = "a" * 64
    contract = validate_formal_quality_v2_thresholds(
        thresholds,
        task_id=task_id,
        thresholds_sha256=threshold_sha256,
    )
    assert contract["formal_freeze_eligible"] is True
    wrong_phase = copy.deepcopy(thresholds)
    next(
        check
        for check in wrong_phase["tasks"][task_id]["checks"]
        if check["metric"] == "eef_motion.eef_angular_jerk_rms_rad_s3"
    )["phase"] = "acquisition_window"
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate_formal_quality_v2_thresholds(
            wrong_phase,
            task_id=task_id,
            thresholds_sha256=threshold_sha256,
        )
    provisional = copy.deepcopy(thresholds)
    provisional["formal_freeze_eligible"] = False
    with pytest.raises(ValueError, match="not eligible for formal freeze"):
        validate_formal_quality_v2_thresholds(
            provisional,
            task_id=task_id,
            thresholds_sha256=threshold_sha256,
        )
    gate = evaluate_quality_v2_gate(quality, thresholds, task_id=task_id)
    gate["contract_sha256"] = threshold_sha256
    replay = {
        "passed": True,
        "final_state_exact": True,
        "outcomes_exact": True,
        "task_quality_exact": True,
    }
    record = {
        "episode_id": episode_id,
        "task_id": task_id,
        "success": True,
        "safety_failure": False,
        "termination_reason": "success",
        "trajectory_completion": 1.0,
        "completion_time_s": 1.0,
        "return": 1.0,
        "control_steps": steps,
        "action_l2_sum": 0.0,
        "task_quality": _quality_summary(episode_id),
        "actions": actions.tolist(),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest(),
        "quality_v2": quality,
        "quality_v2_sha256": promotion._payload_sha256(quality),
        "quality_v2_gate": gate,
        "replay_validation": replay,
        "events": ["success"],
    }
    output = tmp_path / "evaluation"
    output.mkdir()

    produced = materialize_evaluation_attempt(
        output,
        record,
        candidate_index=0,
        raw_env=SimpleNamespace(),
        observations=observations,
        states=[np.zeros(2, dtype=np.float32) for _ in range(steps + 1)],
        policy_actions=[np.zeros(7, dtype=np.float32) for _ in range(steps)],
        rewards=[0.25] * steps,
        terminated=[False, False, False, True],
        truncated=[False] * steps,
        quality_v2_thresholds_sha256=threshold_sha256,
    )
    sealed = promotion._audit_evaluation_attempt(
        produced,
        evaluation_path=output / "evaluation.json",
        task=task_id,
        quality_v2_thresholds=thresholds,
        quality_v2_thresholds_sha256=threshold_sha256,
        label="producer fixture",
    )

    assert produced["attempt_schema_version"] == attempt_schema_version()
    assert produced["actions"] == actions.tolist()
    assert np.asarray(produced["actions"]).dtype == np.float64
    assert sealed["path"] == produced["attempt_tape"]
    assert sealed["sha256"] == produced["attempt_tape_sha256"]
    assert produced["eligible"] is True
    checksum_paths = [
        line.split("  ", 1)[1]
        for line in recursive_output_checksums(output).splitlines()
    ]
    assert produced["attempt_tape"] in checksum_paths
