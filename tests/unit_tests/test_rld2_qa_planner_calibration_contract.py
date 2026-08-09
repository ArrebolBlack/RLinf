#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT / "examples" / "embodiment" / "rld2_qa_planner_calibration_contract.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "rld2_qa_planner_calibration_contract",
    MODULE_PATH,
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
calibration = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(calibration)


def _selection_fixture(wave_root: Path) -> tuple[dict, dict, np.ndarray]:
    rows = [
        {
            "episode_id": f"t1_xyz-calibration-{ordinal:02d}",
            "task_id": "t1_xyz",
            "split": "validation",
            "seed": 20261350 + ordinal,
            "factors": {"ordinal": ordinal},
        }
        for ordinal in range(calibration.EPISODES_PER_TASK)
    ]
    manifest = wave_root / "reset_manifest.jsonl"
    manifest.write_text(
        "".join(calibration._canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    identities = [
        {
            "ordinal": ordinal,
            "episode_id": row["episode_id"],
            "reset_row_sha256": calibration._payload_sha256(row),
        }
        for ordinal, row in enumerate(rows)
    ]
    expected_actions = np.ascontiguousarray(
        np.asarray([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float64)
    )
    records = [
        {"success": False, "safety_failure": False}
        for _ in range(calibration.EPISODES_PER_TASK)
    ]
    records[1] = {
        "episode_id": rows[1]["episode_id"],
        "success": True,
        "safety_failure": False,
        "termination_reason": "success",
        "control_steps": len(expected_actions),
        "actions": expected_actions.tolist(),
        "action_sha256": hashlib.sha256(expected_actions.tobytes()).hexdigest(),
        "action_l2_sum": float(np.square(expected_actions).sum()),
        "task_quality": {},
    }
    contract = {
        "task_id": "t1_xyz",
        "reset_manifest_relative_path": "reset_manifest.jsonl",
        "reset_identities": identities,
    }
    return contract, {"records": records}, expected_actions


def test_selected_action_tape_preserves_float64_identity(tmp_path: Path) -> None:
    contract, evaluation, expected_actions = _selection_fixture(tmp_path)

    ordinal, record, _, actions = calibration._selected_planner_calibration_record(
        tmp_path,
        contract,
        evaluation,
    )

    assert ordinal == 1
    assert record["episode_id"] == "t1_xyz-calibration-01"
    assert actions.dtype == np.dtype(np.float64)
    assert np.array_equal(actions, expected_actions)
    assert hashlib.sha256(actions.tobytes()).hexdigest() == record["action_sha256"]


def test_selected_action_tape_rejects_float32_rehash(tmp_path: Path) -> None:
    contract, evaluation, expected_actions = _selection_fixture(tmp_path)
    mismatched = copy.deepcopy(evaluation)
    float32_actions = np.ascontiguousarray(expected_actions, dtype=np.float32)
    float32_sha256 = hashlib.sha256(float32_actions.tobytes()).hexdigest()
    assert float32_sha256 != evaluation["records"][1]["action_sha256"]
    mismatched["records"][1]["action_sha256"] = float32_sha256

    with pytest.raises(calibration.ContractError, match="action tape hash mismatch"):
        calibration._selected_planner_calibration_record(
            tmp_path,
            contract,
            mismatched,
        )


def _quality_schema() -> dict:
    schema = {
        "schema_version": "db0-episode-task-quality-v2",
        "task_id": "t1_xyz",
        "task_config_sha256": "a" * 64,
        "components": [
            {
                "name": "goal_error",
                "direction": "minimize",
                "unit": "m",
                "scientific_resolution": 0.001,
                "reducer": "minimum",
                "source": "measured_state",
                "description": "Synthetic measured goal error for contract interop.",
            }
        ],
    }
    schema["schema_sha256"] = calibration._payload_sha256(schema)
    return schema


def _task_quality(schema: dict, episode_id: str, value: float) -> dict:
    metadata = schema["components"][0]
    summary = {
        "schema_version": schema["schema_version"],
        "episode_id": episode_id,
        "task_id": "t1_xyz",
        "evaluator_backend_id": calibration.TASK_QUALITY_BACKEND_ID,
        "schema_sha256": schema["schema_sha256"],
        "physics_sample_count": 4,
        "terminal": True,
        "components": {
            metadata["name"]: {
                "value": value,
                "direction": metadata["direction"],
                "unit": metadata["unit"],
                "scientific_resolution": metadata["scientific_resolution"],
                "reducer": metadata["reducer"],
            }
        },
    }
    summary["summary_sha256"] = calibration._payload_sha256(summary)
    return summary


def test_three_replay_evidence_and_contract_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
        build_calibration_evidence,
    )

    contract, evaluation, expected_actions = _selection_fixture(tmp_path)
    schema = _quality_schema()
    contract["quality_identity"] = {"task_quality_schema": schema}
    wave = {
        "manifest_seed": calibration.DEFAULT_MANIFEST_SEED,
        "source_identity": {
            "evaluator_rlinf_commit": "1" * 40,
            "benchmark_commit": "2" * 40,
        },
        "planner_dominance_calibration": calibration._planner_calibration_policy(),
    }
    selected_ordinal, selected_record, selected_reset, _ = (
        calibration._selected_planner_calibration_record(
            tmp_path,
            contract,
            evaluation,
        )
    )
    assert selected_ordinal == 1
    reset_request_sha256 = calibration._payload_sha256(selected_reset)
    replays = []
    for replay_index, value in enumerate((0.010, 0.011, 0.009)):
        replays.append(
            {
                "replay_index": replay_index,
                "environment_instance_id": f"fresh-env-{replay_index}",
                "episode_id": selected_record["episode_id"],
                "reset_request_sha256": reset_request_sha256,
                "action_sha256": selected_record["action_sha256"],
                "success": True,
                "safety_failure": False,
                "finite_and_bounded": True,
                "termination_reason": selected_record["termination_reason"],
                "trajectory_completion": 1.0,
                "completion_time_s": 0.5 + replay_index * 0.002,
                "control_steps": selected_record["control_steps"],
                "action_l2_sum": float(np.square(expected_actions).sum()),
                "task_quality": _task_quality(
                    schema,
                    selected_record["episode_id"],
                    value,
                ),
            }
        )
    attempt_root = tmp_path / "tasks/t1_xyz/attempts/attempt-000001"
    calibration_root = attempt_root / "planner_calibration"
    calibration_root.mkdir(parents=True)
    reset_manifest_bytes = calibration._manifest_bytes((selected_reset,))
    evaluator_identity = calibration._calibration_evaluator_identity(wave)
    evidence_reference = (
        "tasks/t1_xyz/attempts/attempt-000001/"
        "planner_calibration/calibration_evidence.json"
    )
    calibration_input = {
        "task": "t1_xyz",
        "backend_id": calibration.TASK_QUALITY_BACKEND_ID,
        "evaluator_identity": evaluator_identity,
        "split": calibration.TRANSPORT_SPLIT,
        "test_exposure": {"test_id": False, "test_ood": False},
        "reset_manifest_sha256": hashlib.sha256(reset_manifest_bytes).hexdigest(),
        "replays": replays,
    }
    evidence, planner_contract = build_calibration_evidence(
        calibration_input,
        contract_template=calibration._planner_contract_template(contract),
        evidence_reference=evidence_reference,
    )
    calibration._write_exact_bytes(
        calibration_root / "selected_reset_manifest.jsonl",
        reset_manifest_bytes,
    )
    action_buffer = io.BytesIO()
    np.save(action_buffer, expected_actions, allow_pickle=False)
    calibration._write_exact_bytes(
        calibration_root / "planner_actions.npy", action_buffer.getvalue()
    )
    calibration._write_exact_bytes(
        calibration_root / "calibration_input.json",
        calibration._artifact_json_bytes(calibration_input),
    )
    calibration._write_exact_bytes(
        calibration_root / "calibration_evidence.json",
        calibration._artifact_json_bytes(evidence),
    )
    calibration._write_exact_bytes(
        calibration_root / "planner_dominance_contract.json",
        calibration._artifact_json_bytes(planner_contract),
    )
    calibration._write_exact_sha256sums(
        calibration_root / "SHA256SUMS",
        tuple(
            (calibration._sha256(calibration_root / filename), filename)
            for filename, _, _ in calibration.PLANNER_CALIBRATION_ARTIFACTS
        ),
    )

    binding = calibration._load_planner_calibration(
        tmp_path,
        wave,
        contract,
        attempt_root,
        evaluation,
    )

    assert binding["schema_version"] == calibration.PLANNER_CALIBRATION_BINDING_SCHEMA
    assert binding["selected_reset_ordinal"] == 1
    assert binding["replay_count"] == 3
    assert binding["environment_instance_ids"] == [
        "fresh-env-0",
        "fresh-env-1",
        "fresh-env-2",
    ]
    assert binding["calibration_evidence_payload_sha256"] == evidence["payload_sha256"]
    assert binding["planner_dominance_contract_payload_sha256"] == (
        calibration._payload_sha256(planner_contract)
    )
    assert planner_contract["metrics"]["task_quality"]["goal_error"][
        "max_observed_replay_drift"
    ] == pytest.approx(0.002)


def _install_fake_replay_runtime(
    monkeypatch: pytest.MonkeyPatch,
    selected_reset: dict,
    *,
    duplicate_environment_id: bool,
) -> list:
    instances = []

    class FakeDynamicBenchmarkEnv:
        def __init__(self, **kwargs: object) -> None:
            self.instance_index = len(instances)
            self.kwargs = kwargs
            self.closed = False
            self._manifest_rows = [
                SimpleNamespace(
                    request=SimpleNamespace(episode_id=f"unused-{ordinal}"),
                    payload={"episode_id": f"unused-{ordinal}"},
                )
                for ordinal in range(calibration.EPISODES_PER_TASK)
            ]
            self._manifest_rows[1] = SimpleNamespace(
                request=SimpleNamespace(episode_id=selected_reset["episode_id"]),
                payload=selected_reset,
            )
            instances.append(self)

        def close(self) -> None:
            self.closed = True

    def manifest_record(row: SimpleNamespace) -> dict:
        return row.payload

    def replay_planner_actions(**kwargs: object) -> dict:
        env = kwargs["env"]
        assert isinstance(env, FakeDynamicBenchmarkEnv)
        assert env.closed is False
        environment_id = (
            "duplicate-environment"
            if duplicate_environment_id
            else f"fresh-environment-{env.instance_index}"
        )
        return {
            "replay_index": kwargs["replay_index"],
            "environment_instance_id": environment_id,
            "episode_id": selected_reset["episode_id"],
            "reset_request_sha256": kwargs["reset_request_sha256"],
            "action_sha256": kwargs["action_sha256"],
            "success": True,
            "safety_failure": False,
            "finite_and_bounded": True,
            "termination_reason": "success",
            "trajectory_completion": 1.0,
            "completion_time_s": 0.5,
            "control_steps": 2,
            "action_l2_sum": 0.91,
            "task_quality": {},
        }

    module_attributes = {
        "se3_wam.benchmark.evaluation": {"manifest_record": manifest_record},
        "examples.embodiment.run_dynamic_benchmark_rld2_launch_gate": {
            "_replay_planner_actions": replay_planner_actions
        },
        "rlinf.envs.dynamic_benchmark.dynamic_benchmark_env": {
            "DynamicBenchmarkEnv": FakeDynamicBenchmarkEnv
        },
    }
    for module_name in (
        "se3_wam",
        "se3_wam.benchmark",
        "rlinf",
        "rlinf.envs",
        "rlinf.envs.dynamic_benchmark",
    ):
        package = ModuleType(module_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, package)
    for module_name, attributes in module_attributes.items():
        module = ModuleType(module_name)
        for name, value in attributes.items():
            setattr(module, name, value)
        monkeypatch.setitem(sys.modules, module_name, module)
    return instances


def _fresh_replay_fixture(
    tmp_path: Path,
) -> tuple[dict, dict, dict, dict, np.ndarray]:
    contract, evaluation, actions = _selection_fixture(tmp_path)
    contract["quality_identity"] = {
        "task_quality_schema_version": "db0-episode-task-quality-v2"
    }
    _, selected_record, selected_reset, _ = (
        calibration._selected_planner_calibration_record(
            tmp_path,
            contract,
            evaluation,
        )
    )
    wave = {
        "manifest_seed": calibration.DEFAULT_MANIFEST_SEED,
        "planner_dominance_calibration": calibration._planner_calibration_policy(),
    }
    return wave, contract, selected_record, selected_reset, actions


def test_three_replays_construct_and_close_distinct_fresh_environments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wave, contract, selected_record, selected_reset, actions = _fresh_replay_fixture(
        tmp_path
    )
    instances = _install_fake_replay_runtime(
        monkeypatch,
        selected_reset,
        duplicate_environment_id=False,
    )

    replays = calibration._fresh_environment_replays(
        wave,
        contract,
        selected_ordinal=1,
        selected_record=selected_record,
        selected_reset=selected_reset,
        actions=actions,
    )

    assert len(instances) == 3
    assert len({id(instance) for instance in instances}) == 3
    assert all(instance.closed for instance in instances)
    assert [row["environment_instance_id"] for row in replays] == [
        "fresh-environment-0",
        "fresh-environment-1",
        "fresh-environment-2",
    ]
    assert all(
        instance.kwargs["cfg"]["manifest_size"] == calibration.EPISODES_PER_TASK
        for instance in instances
    )


def test_three_replays_reject_duplicate_environment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wave, contract, selected_record, selected_reset, actions = _fresh_replay_fixture(
        tmp_path
    )
    instances = _install_fake_replay_runtime(
        monkeypatch,
        selected_reset,
        duplicate_environment_id=True,
    )

    with pytest.raises(calibration.ContractError, match="not uniquely identified"):
        calibration._fresh_environment_replays(
            wave,
            contract,
            selected_ordinal=1,
            selected_record=selected_record,
            selected_reset=selected_reset,
            actions=actions,
        )
    assert len(instances) == 3
    assert all(instance.closed for instance in instances)
