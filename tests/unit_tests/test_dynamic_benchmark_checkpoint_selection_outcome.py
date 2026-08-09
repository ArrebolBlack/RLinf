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
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.embodiment import train_dynamic_benchmark_expert as trainer
from examples.embodiment.build_dynamic_benchmark_checkpoint_selection_outcome import (
    main as outcome_builder_main,
)
from examples.embodiment.build_dynamic_benchmark_checkpoint_selection_outcome import (
    write_checkpoint_selection_outcome,
)
from examples.embodiment.dynamic_benchmark_checkpoint_admission import (
    CHECKPOINT_SELECTION_OUTCOME_SCHEMA,
    validate_selected_learned_policy,
)

BENCHMARK_COMMIT = "b" * 40


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_checkout(
    root: Path, sources: dict[Path, str], *, message: str
) -> tuple[Path, str]:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "user.name", "Checkpoint Outcome Test")
    _git(root, "config", "user.email", "checkpoint-outcome@example.invalid")
    for source, repository_path in sources.items():
        destination = root.joinpath(*PurePosixPath(repository_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    _git(root, "add", "--all")
    _git(root, "commit", "-q", "-m", message)
    return root, _git(root, "rev-parse", "HEAD")


def _metrics(*, safety: float, success: float = 0.75) -> dict[str, Any]:
    return {
        "episodes": 20,
        "success_rate": success,
        "safety_failure_rate": safety,
        "mean_completion": 0.8,
        "mean_return": 1.0,
        "mean_duration_steps": 20.0,
        "mean_action_l2_sum": 5.0,
        "wall_time_s": 1.25,
        "env_steps_per_second": 160.0,
        "environment_step_mean_ms": 2.5,
        "environment_reset_total_s": 0.125,
        "environment_construction_s": 0.25,
        "environment_restore_s": 0.0,
        "training_environment_checkpoint_s": 0.0,
        "training_environment_restore_s": 0.0,
        "validation_manifest_cache_hit": True,
        "validation_manifest_context_s": 0.0,
        "training_manifest_context_restore_s": 0.0,
        "policy_action_total_s": 0.5,
        "planner_action_total_s": 0.0,
        "process_planner_action_total_s": 0.0,
        "process_environment_step_total_s": 0.5,
        "planner_in_processes": False,
        "persistent_eval_workers": True,
        "borrow_training_env_for_eval": False,
        "persistent_evaluation_index": 1,
        "evaluation_rewind_mode": "manifest_context_restore",
        "action_digest_sha256": "a" * 64,
        "state_digest_sha256": "c" * 64,
        "worker_processes": 1,
        "worker_pids": [1234],
        "records": [
            {
                "manifest_episode_index": 0,
                "success": success >= 0.5,
                "safety_failure": safety > 0.0,
                "termination_reason": "success" if safety == 0.0 else "force_limit",
                "trajectory_completion": 0.8,
                "return": 1.0,
                "duration_steps": 20,
                "action_l2_sum": 5.0,
            }
        ],
    }


def _source_roots(tmp_path: Path) -> SimpleNamespace:
    embodiment = Path(trainer.__file__).resolve().parent
    policy_root, policy_commit = _source_checkout(
        tmp_path / "policy-source",
        {
            embodiment / "train_dynamic_benchmark_expert.py": (
                "examples/embodiment/train_dynamic_benchmark_expert.py"
            )
        },
        message="policy source",
    )
    policy_trainer = (
        policy_root / "examples" / "embodiment" / "train_dynamic_benchmark_expert.py"
    )
    with policy_trainer.open("a", encoding="utf-8") as stream:
        stream.write("\n# Distinct clean policy-source fixture blob.\n")
    _git(policy_root, "add", "--all")
    _git(policy_root, "commit", "-q", "-m", "distinct policy trainer blob")
    policy_commit = _git(policy_root, "rev-parse", "HEAD")
    verifier_root, verifier_commit = _source_checkout(
        tmp_path / "verifier-source",
        {
            embodiment / "dynamic_benchmark_checkpoint_admission.py": (
                "examples/embodiment/dynamic_benchmark_checkpoint_admission.py"
            ),
            embodiment / "build_dynamic_benchmark_checkpoint_selection_outcome.py": (
                "examples/embodiment/"
                "build_dynamic_benchmark_checkpoint_selection_outcome.py"
            ),
        },
        message="verifier source",
    )
    evaluator_root, evaluator_commit = _source_checkout(
        tmp_path / "evaluator-source",
        {
            embodiment / "evaluate_dynamic_benchmark_expert.py": (
                "examples/embodiment/evaluate_dynamic_benchmark_expert.py"
            ),
        },
        message="evaluator source",
    )
    evaluator_source = (
        evaluator_root
        / "examples"
        / "embodiment"
        / "evaluate_dynamic_benchmark_expert.py"
    )
    with evaluator_source.open("a", encoding="utf-8") as stream:
        stream.write("\n# Distinct clean evaluator-source fixture blob.\n")
    _git(evaluator_root, "add", "--all")
    _git(evaluator_root, "commit", "-q", "-m", "distinct evaluator source blob")
    evaluator_commit = _git(evaluator_root, "rev-parse", "HEAD")
    assert len({policy_commit, verifier_commit, evaluator_commit}) == 3
    return SimpleNamespace(
        policy_root=policy_root,
        policy_commit=policy_commit,
        verifier_root=verifier_root,
        verifier_commit=verifier_commit,
        evaluator_root=evaluator_root,
        evaluator_commit=evaluator_commit,
    )


def _trainer_run(
    tmp_path: Path,
    *,
    eligible: bool,
    env0_only: bool = False,
    absolute_validation_path: bool = False,
    unsafe_validation_key: str | None = None,
) -> SimpleNamespace:
    sources = _source_roots(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    config = trainer._config(
        trainer._parse_args(
            [
                "--task",
                "t1_xyz",
                "--algorithm",
                "residual_rlpd",
                "--rlinf-commit",
                sources.policy_commit,
                "--benchmark-commit",
                BENCHMARK_COMMIT,
                "--output",
                str(run_root),
            ]
        )
    )
    config_payload = asdict(config)
    config_path = run_root / "config.json"
    trainer._atomic_json(config_path, config_payload)
    config_file_sha256 = trainer._file_sha256(config_path)
    state_schema = {"state_dim": 3, "mask_dim": 0, "fields": ["fixture"]}
    planner_metrics = _metrics(safety=0.0, success=0.5)
    candidate_metrics = _metrics(safety=0.0 if eligible else 0.1)
    if absolute_validation_path:
        planner_metrics["diagnostic_path"] = str(run_root / "planner-diagnostic")
        candidate_metrics["diagnostic_path"] = str(run_root / "policy-diagnostic")
    if unsafe_validation_key is not None:
        planner_metrics["records"][0][unsafe_validation_key] = "planner diagnostic"
        candidate_metrics["records"][0][unsafe_validation_key] = "policy diagnostic"
    model = torch.nn.Linear(3, 7)
    normalizer = trainer.RunningNormalizer(3, 0)
    initial_policy = run_root / "initial_policy.pt"
    trainer._save_policy(
        initial_policy,
        config,
        model,
        normalizer,
        state_schema,
        planner_metrics,
        0,
    )
    run_identity = trainer._checkpoint_selection_run_identity(
        config, state_schema, config_file_sha256
    )
    ledger = trainer._CheckpointSelectionLedger.create(
        run_root, run_identity, planner_metrics, initial_policy
    )
    metrics_path = run_root / "metrics.jsonl"
    trainer._append_jsonl(
        metrics_path,
        {
            "event": "validation",
            "env_steps": 0,
            "checkpoint_selection_role": "matched_planner_safety_ceiling",
            "validation_metrics_sha256": trainer._canonical_json_sha256(
                planner_metrics
            ),
            "checkpoint_selection_manifest_payload_sha256": ledger.manifest[
                "payload_sha256"
            ],
            **planner_metrics,
        },
    )

    final_validation = planner_metrics
    final_env_steps = 0
    if not env0_only:
        final_env_steps = 100
        final_validation = candidate_metrics
        snapshot = run_root / "policy_snapshots" / "policy_step_000000000100.pt"
        snapshot.parent.mkdir()
        trainer._save_policy(
            snapshot,
            config,
            model,
            normalizer,
            state_schema,
            candidate_metrics,
            final_env_steps,
        )
        row = ledger.record_existing_snapshot(
            snapshot, candidate_metrics, final_env_steps
        )
        trainer._append_jsonl(
            metrics_path,
            {
                "event": "validation",
                "env_steps": final_env_steps,
                "checkpoint_snapshot_identity": (
                    trainer._CheckpointSelectionLedger._snapshot_identity(row)
                ),
                "validation_role": "final",
                **candidate_metrics,
            },
        )

    summary = {
        "schema_version": "rlinf-dynamic-benchmark-expert-summary-v0.1",
        "status": "complete",
        "config": config_payload,
        "infra_identity": trainer._infra_identity(config),
        "demo_source": {"fixture": True},
        "best_validation": ledger.best_metrics,
        "best_score": ledger.best_score,
        "final_validation": final_validation,
        "env_steps": final_env_steps,
        "update_steps": int(not env0_only),
        "config_sha256": trainer._canonical_json_sha256(config_payload),
        "config_file_sha256": config_file_sha256,
        "checkpoint_selection": ledger.summary_reference(),
    }
    summary["payload_sha256"] = trainer._canonical_json_sha256(summary)
    summary_path = run_root / "summary.json"
    trainer._atomic_json(summary_path, summary)
    return SimpleNamespace(
        **vars(sources),
        run_root=run_root,
        summary_path=summary_path,
        selection_path=ledger.manifest_path,
        config_path=config_path,
        metrics_path=metrics_path,
        initial_policy_path=initial_policy,
        best_policy_path=run_root / "best_policy.pt",
        eligible=eligible and not env0_only,
    )


def _write_kwargs(run: SimpleNamespace) -> dict:
    return {
        "run_root": run.run_root,
        "policy_rlinf_source_root": run.policy_root,
        "verifier_rlinf_source_root": run.verifier_root,
        "evaluator_rlinf_source_root": run.evaluator_root,
        "expected_policy_rlinf_commit": run.policy_commit,
        "expected_verifier_rlinf_commit": run.verifier_commit,
        "expected_evaluator_rlinf_commit": run.evaluator_commit,
        "expected_benchmark_commit": BENCHMARK_COMMIT,
        "expected_summary_sha256": trainer._file_sha256(run.summary_path),
        "expected_checkpoint_selection_sha256": trainer._file_sha256(
            run.selection_path
        ),
        "expected_config_sha256": trainer._file_sha256(run.config_path),
        "expected_metrics_sha256": trainer._file_sha256(run.metrics_path),
        "expected_initial_policy_sha256": trainer._file_sha256(run.initial_policy_path),
        "expected_best_policy_sha256": (
            trainer._file_sha256(run.best_policy_path) if run.eligible else None
        ),
    }


def _cli_args(run: SimpleNamespace) -> list[str]:
    kwargs = _write_kwargs(run)
    arguments = [
        "--run-root",
        str(kwargs["run_root"]),
        "--policy-rlinf-source-root",
        str(kwargs["policy_rlinf_source_root"]),
        "--verifier-rlinf-source-root",
        str(kwargs["verifier_rlinf_source_root"]),
        "--evaluator-rlinf-source-root",
        str(kwargs["evaluator_rlinf_source_root"]),
        "--expected-policy-rlinf-commit",
        kwargs["expected_policy_rlinf_commit"],
        "--expected-verifier-rlinf-commit",
        kwargs["expected_verifier_rlinf_commit"],
        "--expected-evaluator-rlinf-commit",
        kwargs["expected_evaluator_rlinf_commit"],
        "--expected-benchmark-commit",
        kwargs["expected_benchmark_commit"],
        "--expected-summary-sha256",
        kwargs["expected_summary_sha256"],
        "--expected-checkpoint-selection-sha256",
        kwargs["expected_checkpoint_selection_sha256"],
        "--expected-config-sha256",
        kwargs["expected_config_sha256"],
        "--expected-metrics-sha256",
        kwargs["expected_metrics_sha256"],
        "--expected-initial-policy-sha256",
        kwargs["expected_initial_policy_sha256"],
    ]
    if kwargs["expected_best_policy_sha256"] is not None:
        arguments.extend(
            [
                "--expected-best-policy-sha256",
                kwargs["expected_best_policy_sha256"],
            ]
        )
    return arguments


def _reseal_json(path: Path, payload: dict) -> None:
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = trainer._canonical_json_sha256(payload)
    trainer._atomic_json(path, payload)


def _load_outcome(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_outcome_builder_seals_real_selected_trainer_chain_and_is_exclusive(
    tmp_path: Path,
) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    kwargs = _write_kwargs(run)

    output = write_checkpoint_selection_outcome(**kwargs)
    outcome = _load_outcome(output)

    assert output == run.run_root / "checkpoint_selection_outcome.json"
    assert set(outcome) == {
        "schema_version",
        "source_identity",
        "run_identity",
        "trainer_artifacts",
        "selector",
        "matched_planner_baseline",
        "evaluated_snapshots",
        "selection",
        "payload_sha256",
    }
    assert outcome["schema_version"] == CHECKPOINT_SELECTION_OUTCOME_SCHEMA
    assert outcome["selection"]["status"] == "selected_eligible_snapshot"
    assert outcome["selection"]["best_policy"]["path"] == "best_policy.pt"
    assert outcome["evaluated_snapshots"][0]["env_steps"] == 100
    assert outcome["evaluated_snapshots"][0]["validation_evidence"]["role"] == ("final")
    assert outcome["source_identity"]["policy_rlinf_commit"] == run.policy_commit
    assert outcome["source_identity"]["trainer_source"]["sha256"] != (
        trainer._file_sha256(Path(trainer.__file__).resolve())
    )
    assert outcome["source_identity"]["verifier_rlinf_commit"] == (run.verifier_commit)
    assert outcome["source_identity"]["evaluator_rlinf_commit"] == (
        run.evaluator_commit
    )
    evaluator_source = (
        run.evaluator_root
        / "examples"
        / "embodiment"
        / "evaluate_dynamic_benchmark_expert.py"
    )
    assert outcome["source_identity"]["evaluator_source"]["sha256"] == (
        trainer._file_sha256(evaluator_source)
    )
    assert outcome["source_identity"]["evaluator_source"]["sha256"] != (
        trainer._file_sha256(
            Path(trainer.__file__)
            .resolve()
            .with_name("evaluate_dynamic_benchmark_expert.py")
        )
    )
    assert (
        len(
            {
                outcome["source_identity"]["policy_rlinf_commit"],
                outcome["source_identity"]["verifier_rlinf_commit"],
                outcome["source_identity"]["evaluator_rlinf_commit"],
            }
        )
        == 3
    )
    assert set(outcome["source_identity"]) == {
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
    }
    for source_name in (
        "trainer_source",
        "verifier_source",
        "builder_source",
        "evaluator_source",
    ):
        assert set(outcome["source_identity"][source_name]) == {
            "module",
            "repository_path",
            "sha256",
        }
    assert set(outcome["run_identity"]) == {
        "schema_version",
        "task",
        "algorithm",
        "rlinf_commit",
        "benchmark_commit",
        "seed",
        "validation_manifest_seed",
        "eval_episodes",
        "eval_num_envs",
        "config_sha256",
        "config_payload_sha256",
        "state_schema_sha256",
    }
    assert set(outcome["trainer_artifacts"]) == {
        "summary",
        "checkpoint_selection",
        "config",
        "metrics",
    }
    assert set(outcome["trainer_artifacts"]["summary"]) == {
        "path",
        "sha256",
        "schema_version",
        "payload_sha256",
        "status",
        "env_steps",
        "update_steps",
        "best_validation_metrics_sha256",
        "best_selection_score",
        "final_validation_metrics_sha256",
    }
    assert set(outcome["trainer_artifacts"]["checkpoint_selection"]) == {
        "path",
        "sha256",
        "schema_version",
        "payload_sha256",
    }
    assert set(outcome["trainer_artifacts"]["config"]) == {
        "path",
        "sha256",
        "payload_sha256",
    }
    assert set(outcome["trainer_artifacts"]["metrics"]) == {
        "path",
        "sha256",
        "format",
        "validation_event_count",
        "validation_event_inventory_sha256",
    }
    assert set(outcome["selector"]) == {
        "name",
        "safety_ceiling_tolerance",
        "eligible_order",
        "env_steps_zero_is_learned_candidate",
    }
    assert set(outcome["matched_planner_baseline"]) == {
        "source",
        "safety_failure_rate_ceiling",
        "validation_metrics",
        "validation_metrics_sha256",
        "selector_metrics",
        "selector_metrics_sha256",
        "policy",
        "validation_evidence",
    }
    selector_metric_keys = {
        "success_rate",
        "safety_failure_rate",
        "mean_completion",
        "mean_return",
        "mean_duration_steps",
        "mean_action_l2_sum",
    }
    baseline = outcome["matched_planner_baseline"]
    assert set(baseline["selector_metrics"]) == selector_metric_keys
    assert baseline["selector_metrics"] == {
        key: baseline["validation_metrics"][key] for key in selector_metric_keys
    }
    assert baseline["selector_metrics_sha256"] == trainer._canonical_json_sha256(
        baseline["selector_metrics"]
    )
    assert baseline["validation_metrics_sha256"] == trainer._canonical_json_sha256(
        baseline["validation_metrics"]
    )
    assert "records" in baseline["validation_metrics"]
    assert baseline["validation_metrics_sha256"] != baseline["selector_metrics_sha256"]
    policy_keys = {
        "path",
        "sha256",
        "schema_version",
        "env_steps",
        "validation_metrics_sha256",
        "config_payload_sha256",
        "state_schema_sha256",
        "infra_identity_sha256",
    }
    evidence_keys = {
        "metrics_path",
        "line_number",
        "event_payload_sha256",
        "validation_metrics_sha256",
        "role",
        "checkpoint_selection_manifest_payload_sha256",
        "checkpoint_snapshot_identity",
    }
    assert set(outcome["matched_planner_baseline"]["policy"]) == policy_keys
    assert (
        set(outcome["matched_planner_baseline"]["validation_evidence"]) == evidence_keys
    )
    assert set(outcome["evaluated_snapshots"][0]) == {
        "env_steps",
        "policy",
        "validation_metrics",
        "validation_metrics_sha256",
        "selector_metrics",
        "selector_metrics_sha256",
        "validation_evidence",
        "eligible",
        "eligibility_reason",
        "selection_score",
        "selected",
    }
    assert set(outcome["evaluated_snapshots"][0]["policy"]) == policy_keys
    snapshot = outcome["evaluated_snapshots"][0]
    assert set(snapshot["selector_metrics"]) == selector_metric_keys
    assert snapshot["selector_metrics"] == {
        key: snapshot["validation_metrics"][key] for key in selector_metric_keys
    }
    assert snapshot["selector_metrics_sha256"] == trainer._canonical_json_sha256(
        snapshot["selector_metrics"]
    )
    assert snapshot["validation_metrics_sha256"] == trainer._canonical_json_sha256(
        snapshot["validation_metrics"]
    )
    assert (
        snapshot["validation_evidence"]["validation_metrics_sha256"]
        == snapshot["validation_metrics_sha256"]
    )
    assert snapshot["validation_metrics_sha256"] != snapshot["selector_metrics_sha256"]
    assert (
        set(outcome["evaluated_snapshots"][0]["validation_evidence"]) == evidence_keys
    )
    assert set(outcome["selection"]) == {
        "status",
        "eligible_snapshot_count",
        "selected_snapshot_identity",
        "best_policy",
        "planner_fallback_policy",
    }
    assert set(outcome["selection"]["selected_snapshot_identity"]) == {
        "env_steps",
        "policy_path",
        "policy_sha256",
        "validation_metrics_sha256",
    }
    assert set(outcome["selection"]["best_policy"]) == policy_keys
    evidence_inventory = [
        outcome["matched_planner_baseline"]["validation_evidence"],
        *[row["validation_evidence"] for row in outcome["evaluated_snapshots"]],
    ]
    assert outcome["trainer_artifacts"]["metrics"][
        "validation_event_inventory_sha256"
    ] == trainer._canonical_json_sha256(evidence_inventory)
    unsigned = copy.deepcopy(outcome)
    claimed = unsigned.pop("payload_sha256")
    assert claimed == trainer._canonical_json_sha256(unsigned)
    for artifact in outcome["trainer_artifacts"].values():
        assert not Path(artifact["path"]).is_absolute()
        assert "\\" not in artifact["path"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_checkpoint_selection_outcome(**kwargs)


def test_real_outcome_producer_feeds_formal_learned_admission(tmp_path: Path) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    executing_evaluator = (
        Path(trainer.__file__).resolve().parent / "evaluate_dynamic_benchmark_expert.py"
    )
    evaluator_source = (
        run.evaluator_root
        / "examples"
        / "embodiment"
        / "evaluate_dynamic_benchmark_expert.py"
    )
    shutil.copyfile(executing_evaluator, evaluator_source)
    _git(run.evaluator_root, "add", "--all")
    _git(run.evaluator_root, "commit", "-q", "-m", "formal evaluator source")
    run.evaluator_commit = _git(run.evaluator_root, "rev-parse", "HEAD")

    outcome_path = write_checkpoint_selection_outcome(**_write_kwargs(run))
    admission = validate_selected_learned_policy(
        policy_path=run.best_policy_path,
        trainer_summary_path=run.summary_path,
        checkpoint_selection_path=run.selection_path,
        checkpoint_selection_outcome_path=outcome_path,
        policy_rlinf_source_root=run.policy_root,
        verifier_rlinf_source_root=run.verifier_root,
        evaluator_rlinf_source_root=run.evaluator_root,
        expected_checkpoint_selection_outcome_sha256=trainer._file_sha256(outcome_path),
        expected_policy_sha256=trainer._file_sha256(run.best_policy_path),
        expected_trainer_summary_sha256=trainer._file_sha256(run.summary_path),
        expected_checkpoint_selection_sha256=trainer._file_sha256(run.selection_path),
        expected_rlinf_commit=run.policy_commit,
        expected_benchmark_commit=BENCHMARK_COMMIT,
        expected_verifier_rlinf_commit=run.verifier_commit,
        expected_evaluator_rlinf_commit=run.evaluator_commit,
    )

    assert admission["checkpoint_role"] == "best"
    assert admission["checkpoint_selection_outcome"] == {
        "path": str(outcome_path.resolve()),
        "sha256": trainer._file_sha256(outcome_path),
        "schema_version": CHECKPOINT_SELECTION_OUTCOME_SCHEMA,
        "payload_sha256": json.loads(outcome_path.read_text(encoding="utf-8"))[
            "payload_sha256"
        ],
    }


def test_outcome_builder_seals_real_no_eligible_planner_fallback(
    tmp_path: Path,
) -> None:
    run = _trainer_run(tmp_path, eligible=False)

    outcome_builder_main(_cli_args(run))
    output = run.run_root / "checkpoint_selection_outcome.json"
    outcome = _load_outcome(output)

    assert outcome["selection"] == {
        "status": "planner_fallback_no_eligible",
        "eligible_snapshot_count": 0,
        "selected_snapshot_identity": None,
        "best_policy": None,
        "planner_fallback_policy": outcome["matched_planner_baseline"]["policy"],
    }
    assert outcome["matched_planner_baseline"]["policy"]["env_steps"] == 0
    assert outcome["evaluated_snapshots"][0]["env_steps"] == 100
    assert outcome["evaluated_snapshots"][0]["eligible"] is False
    assert (
        outcome["trainer_artifacts"]["summary"]["best_validation_metrics_sha256"]
        is None
    )
    assert outcome["trainer_artifacts"]["summary"]["best_selection_score"] is None
    assert not run.best_policy_path.exists()


@pytest.mark.parametrize("target", ["selector", "snapshot", "metrics"])
def test_outcome_builder_rejects_resealed_or_byte_tampering_before_write(
    tmp_path: Path, target: str
) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    kwargs = _write_kwargs(run)
    if target == "selector":
        manifest = json.loads(run.selection_path.read_text(encoding="utf-8"))
        manifest["evaluated_snapshots"][0]["eligibility_reason"] = (
            "safety_failure_rate_exceeds_matched_planner_ceiling"
        )
        _reseal_json(run.selection_path, manifest)
        summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
        summary["checkpoint_selection"]["manifest_payload_sha256"] = manifest[
            "payload_sha256"
        ]
        _reseal_json(run.summary_path, summary)
        kwargs["expected_checkpoint_selection_sha256"] = trainer._file_sha256(
            run.selection_path
        )
        kwargs["expected_summary_sha256"] = trainer._file_sha256(run.summary_path)
    elif target == "snapshot":
        snapshot = next((run.run_root / "policy_snapshots").glob("*.pt"))
        with snapshot.open("ab") as stream:
            stream.write(b"tamper")
    else:
        events = [
            json.loads(line)
            for line in run.metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        events[-1]["success_rate"] = 0.125
        run.metrics_path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )
        kwargs["expected_metrics_sha256"] = trainer._file_sha256(run.metrics_path)

    with pytest.raises((ValueError, RuntimeError)):
        write_checkpoint_selection_outcome(**kwargs)
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_rejects_env0_only_fallback_before_write(
    tmp_path: Path,
) -> None:
    run = _trainer_run(tmp_path, eligible=False, env0_only=True)

    with pytest.raises(ValueError, match="env_steps"):
        write_checkpoint_selection_outcome(**_write_kwargs(run))
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_rejects_best_policy_masquerade_in_fallback(
    tmp_path: Path,
) -> None:
    run = _trainer_run(tmp_path, eligible=False)
    shutil.copyfile(run.initial_policy_path, run.best_policy_path)

    with pytest.raises(ValueError, match="must be absent"):
        write_checkpoint_selection_outcome(**_write_kwargs(run))
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_requires_expected_selected_best_sha256(tmp_path: Path) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    kwargs = _write_kwargs(run)
    kwargs["expected_best_policy_sha256"] = None

    with pytest.raises(ValueError, match="requires expected best_policy"):
        write_checkpoint_selection_outcome(**kwargs)
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_rejects_nonfinite_metrics_before_write(tmp_path: Path) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    kwargs = _write_kwargs(run)
    lines = run.metrics_path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace('"success_rate": 0.75', '"success_rate": NaN')
    run.metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    kwargs["expected_metrics_sha256"] = trainer._file_sha256(run.metrics_path)

    with pytest.raises(ValueError, match="finite canonical JSON"):
        write_checkpoint_selection_outcome(**kwargs)
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_rejects_policy_source_commit_mismatch(tmp_path: Path) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    kwargs = _write_kwargs(run)
    kwargs["expected_policy_rlinf_commit"] = "a" * 40

    with pytest.raises(ValueError, match="HEAD does not match"):
        write_checkpoint_selection_outcome(**kwargs)
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_rejects_dirty_verifier_source_root(tmp_path: Path) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    verifier_source = (
        run.verifier_root
        / "examples"
        / "embodiment"
        / "dynamic_benchmark_checkpoint_admission.py"
    )
    with verifier_source.open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(ValueError, match="source root must be clean"):
        write_checkpoint_selection_outcome(**_write_kwargs(run))
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_rejects_evaluator_source_commit_mismatch(
    tmp_path: Path,
) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    kwargs = _write_kwargs(run)
    kwargs["expected_evaluator_rlinf_commit"] = "a" * 40

    with pytest.raises(ValueError, match="HEAD does not match"):
        write_checkpoint_selection_outcome(**kwargs)
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_rejects_missing_validation_evidence(tmp_path: Path) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    kwargs = _write_kwargs(run)
    lines = run.metrics_path.read_text(encoding="utf-8").splitlines()
    run.metrics_path.write_text(lines[0] + "\n", encoding="utf-8")
    kwargs["expected_metrics_sha256"] = trainer._file_sha256(run.metrics_path)

    with pytest.raises(ValueError, match="inventory is incomplete"):
        write_checkpoint_selection_outcome(**kwargs)
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_rejects_absolute_paths_in_portable_payload(
    tmp_path: Path,
) -> None:
    run = _trainer_run(
        tmp_path,
        eligible=True,
        absolute_validation_path=True,
    )

    with pytest.raises(ValueError, match="contains an absolute path"):
        write_checkpoint_selection_outcome(**_write_kwargs(run))
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


@pytest.mark.parametrize(
    "unsafe_key",
    [
        "/tmp/selector-evidence",
        r"C:\selector-evidence",
        "../selector-evidence",
    ],
    ids=["posix-absolute", "windows-absolute", "relative-traversal"],
)
def test_outcome_builder_rejects_unsafe_paths_in_mapping_keys(
    tmp_path: Path, unsafe_key: str
) -> None:
    run = _trainer_run(
        tmp_path,
        eligible=True,
        unsafe_validation_key=unsafe_key,
    )

    with pytest.raises(ValueError, match="(?:absolute|unsafe) path"):
        write_checkpoint_selection_outcome(**_write_kwargs(run))
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()


def test_outcome_builder_rejects_symlinked_trainer_artifact(tmp_path: Path) -> None:
    run = _trainer_run(tmp_path, eligible=True)
    target = tmp_path / "initial-policy-copy.pt"
    shutil.copyfile(run.initial_policy_path, target)
    run.initial_policy_path.unlink()
    try:
        os.symlink(target, run.initial_policy_path)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="symlink"):
        write_checkpoint_selection_outcome(**_write_kwargs(run))
    assert not (run.run_root / "checkpoint_selection_outcome.json").exists()
