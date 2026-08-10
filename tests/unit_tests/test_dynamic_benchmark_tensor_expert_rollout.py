# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
# Distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from examples.embodiment.evaluate_dynamic_benchmark_tensor_expert import (
    EPISODE_LEDGER_SCHEMA,
    SEQUENCE_SCHEMA,
    PinnedSequence,
    TensorEvaluationError,
    file_sha256,
    manifest_requests_sha256,
    publish_result_bundle,
    summarize_episodes,
    verify_result_bundle,
)
from examples.embodiment.rollout_dynamic_benchmark_tensor_expert import (
    ROLLOUT_UNION_SCHEMA,
    SHARD_PLAN_SCHEMA,
    SHARD_RESULT_SCHEMA,
    ShardPlan,
    merge_shard_bundles,
    merge_shard_payloads,
)


def _export_identity() -> dict:
    return {
        "request_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "model_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "frozen_request": {"episode_id": "frozen"},
    }


def _request(index: int) -> dict:
    return {
        "episode_id": f"episode-{index}",
        "task_id": "p0_grasp",
        "split": "test_id",
        "seed": 700 + index,
        "action_mode": "E7",
        "observation_track": "state",
        "object_mode": "cube",
        "reset_mode": "tabletop",
        "factors": {"index": index},
        "api_version": "db-api-v0.1",
    }


def _sequence_payload(count: int = 4) -> dict:
    requests = [_request(index) for index in range(count)]
    return {
        "schema_version": SEQUENCE_SCHEMA,
        "task_id": "p0_grasp",
        "split": "test_id",
        "manifest_seed": 20260809,
        "manifest_sha256": manifest_requests_sha256(requests),
        "api_version": "gpu-native-api-v0.2",
        "task_quality": {
            "schema_version": "db0-episode-task-quality-v1",
            "evaluator_backend_id": "mjwarp-quality-v1",
        },
        "active_export_identity": _export_identity(),
        "requests": requests,
    }


def _plan_payload(sequence_sha256: str = "b" * 64) -> dict:
    return {
        "schema_version": SHARD_PLAN_SCHEMA,
        "sequence_sha256": sequence_sha256,
        "policy_sha256": "a" * 64,
        "episode_count": 4,
        "shards": [
            {
                "shard_id": "shard-0",
                "num_envs": 2,
                "start_ordinal": 0,
                "episode_count": 2,
            },
            {
                "shard_id": "shard-1",
                "num_envs": 2,
                "start_ordinal": 2,
                "episode_count": 2,
            },
        ],
    }


def _episode(ordinal: int) -> dict:
    return {
        "schema_version": EPISODE_LEDGER_SCHEMA,
        "ordinal": ordinal,
        "lane": ordinal % 2,
        "episode_id": f"episode-{ordinal}-cycle00000000",
        "task_id": "p0_grasp",
        "split": "test_id",
        "seed": 700 + ordinal,
        "return": float(ordinal),
        "success": ordinal % 2 == 0,
        "terminated": ordinal % 2 == 0,
        "truncated": ordinal % 2 != 0,
        "safety": None,
        "safety_available": False,
        "safety_failure": None,
        "completion": 1.0 if ordinal % 2 == 0 else 0.5,
        "task_quality": {"summary_sha256": "f" * 64} if ordinal % 2 == 0 else None,
        "episode_cost": 0.25 + ordinal,
        "episode_cost_kind": "normalized_action_l2_sum",
        "control_steps": 2,
        "terminal_reason": "success" if ordinal % 2 == 0 else "timeout",
        "terminal_physics_step": 50,
        "terminal_control_step": 2,
        "terminal_policy_step": 2,
    }


def _source_identity() -> dict:
    sources = {
        "rlinf": {
            "root": "/rlinf-a",
            "module": "rlinf",
            "loaded_file": "/rlinf-a/rlinf/__init__.py",
            "commit": "6" * 40,
            "tree": "7" * 40,
            "tracked_worktree_clean": True,
        },
        "se3_wam": {
            "root": "/se3-a",
            "module": "se3_wam",
            "loaded_file": "/se3-a/se3_wam/__init__.py",
            "commit": "8" * 40,
            "tree": "9" * 40,
            "tracked_worktree_clean": True,
        },
    }
    return {"start": sources, "end": copy.deepcopy(sources)}


def _runtime_identity() -> dict:
    runtime = {
        "path": "/runtime-a/runtime.json",
        "sha256": "c" * 64,
        "payload": {
            "versions": {
                "mujoco": "3.11.0",
                "mujoco-warp": "3.11.0",
                "warp-lang": "1.0",
            }
        },
    }
    return {"start": runtime, "end": copy.deepcopy(runtime)}


def _process_identity(pid: int, parent_pid: int) -> dict:
    return {
        "pid": pid,
        "parent_pid": parent_pid,
        "boot_id": "boot-fixture",
        "start_ticks": pid * 100,
        "identity_source": "linux_procfs",
    }


def _shard_payload(
    shard_id: str,
    start: int,
    *,
    plan_sha256: str = "d" * 64,
    sequence_sha256: str = "b" * 64,
) -> tuple[dict, list[dict]]:
    episodes = [_episode(start), _episode(start + 1)]
    device_identity = {"expected_uuid": f"GPU-{shard_id}"}
    sequence = _sequence_payload()
    stable_identity = {
        "task_id": "p0_grasp",
        "api_version": "gpu-native-api-v0.2",
        "source_commit": "8" * 40,
        "source_tree": "9" * 40,
        "runtime_versions": {
            "mujoco": "3.11.0",
            "mujoco-warp": "3.11.0",
            "warp-lang": "1.0",
        },
        "reset_manifest": {
            "origin": "caller",
            "split": "test_id",
            "seed": 20260809,
            "size": 4,
            "sha256": sequence["manifest_sha256"],
        },
        "task_quality": {
            "schema_version": "db0-episode-task-quality-v1",
            "evaluator_backend_id": "mjwarp-quality-v1",
        },
        "device_identity": device_identity,
    }
    metrics = summarize_episodes(
        episodes,
        allocated_steps=4,
        valid_steps=4,
        rollout_seconds=2.0,
    )
    result = {
        "schema_version": SHARD_RESULT_SCHEMA,
        "worker_schema_version": "rlinf-dynamic-benchmark-tensor-evaluation-v0.1",
        "status": "complete",
        "mode": "rollout_shard",
        "claim_scope": {
            "science_gate_state": "not_run",
            "production_qualified": False,
            "statement": "fixture",
        },
        "ownership": {
            "start_ordinal": start,
            "episode_count": 2,
            "stop_ordinal_exclusive": start + 2,
        },
        "policy_identity": {"sha256": "a" * 64},
        "sequence_identity": {
            "path": f"/sequence-{shard_id}.json",
            "sha256": sequence_sha256,
            "manifest_sha256": sequence["manifest_sha256"],
            "task_id": "p0_grasp",
            "split": "test_id",
            "manifest_seed": 20260809,
            "api_version": "gpu-native-api-v0.2",
            "request_count": 4,
        },
        "shard_plan_identity": {
            "path": f"/plan-{shard_id}.json",
            "sha256": plan_sha256,
            "sequence_sha256": sequence_sha256,
            "policy_sha256": "a" * 64,
            "shard_id": shard_id,
            "shard_count": 2,
        },
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
        "backend_identity": {
            "stable_start": stable_identity,
            "stable_end": copy.deepcopy(stable_identity),
            "active_export_start": {
                "export_dir": f"/export-{shard_id}",
                **_export_identity(),
            },
            "active_export_end": {
                "export_dir": f"/export-{shard_id}",
                **_export_identity(),
            },
            "portable_active_export": _export_identity(),
            "device_start": device_identity,
            "device_end": copy.deepcopy(device_identity),
        },
        "process_identity": {
            "parent_start": _process_identity(10 + start, 1),
            "parent_end": _process_identity(10 + start, 1),
            "child_start": _process_identity(20 + start, 10 + start),
            "child_end": _process_identity(20 + start, 10 + start),
        },
        "data_plane": {
            "step_api": "step_device",
            "device_ledger": "preallocated_per_cohort",
            "hot_path_host_materializations": 0,
            "terminal_control_plane_materializations": 1,
            "transport_checks": 2,
            "expected_transport_checks": 2,
            "last_transport_receipt": {"verified": True},
            "active_cohort_end": {
                "episode_ids": [
                    f"episode-{start}-cycle00000000",
                    f"episode-{start + 1}-cycle00000000",
                ],
                "manifest_ordinals": [start, start + 1],
                "generation": 1,
            },
        },
        "metrics": metrics,
    }
    return result, episodes


def test_shard_plan_rejects_overlap_gap_and_partial_cohort() -> None:
    valid = _plan_payload()

    plan = ShardPlan.from_payload(valid)

    assert plan.shard("shard-1").start_ordinal == 2

    overlap = copy.deepcopy(valid)
    overlap["shards"][1]["start_ordinal"] = 0
    with pytest.raises(ValueError, match="overlap"):
        ShardPlan.from_payload(overlap)

    gap = copy.deepcopy(valid)
    gap["shards"][1]["start_ordinal"] = 4
    with pytest.raises(ValueError, match="gap"):
        ShardPlan.from_payload(gap)

    partial = copy.deepcopy(valid)
    partial["shards"][0]["episode_count"] = 1
    with pytest.raises(ValueError, match="complete device cohorts"):
        ShardPlan.from_payload(partial)


def test_merge_requires_every_shard_once_and_exact_episode_union() -> None:
    sequence = PinnedSequence.from_payload(_sequence_payload())
    plan = ShardPlan.from_payload(_plan_payload(), sequence=sequence)
    shard0 = _shard_payload("shard-0", 0)
    shard1 = _shard_payload("shard-1", 2)

    result, episodes = merge_shard_payloads(
        plan=plan,
        plan_sha256="d" * 64,
        sequence=sequence,
        sequence_sha256="b" * 64,
        shard_payloads=[shard0, shard1],
    )

    assert result["schema_version"] == ROLLOUT_UNION_SCHEMA
    assert result["shard_plan_identity"]["union_exact"] is True
    assert [row["ordinal"] for row in episodes] == [0, 1, 2, 3]
    assert result["metrics"]["throughput"]["denominator"] == (
        "aggregate_device_rollout_seconds"
    )

    with pytest.raises(ValueError, match="duplicate shard"):
        merge_shard_payloads(
            plan=plan,
            plan_sha256="d" * 64,
            sequence=sequence,
            sequence_sha256="b" * 64,
            shard_payloads=[shard0, shard0, shard1],
        )
    with pytest.raises(ValueError, match="not exact"):
        merge_shard_payloads(
            plan=plan,
            plan_sha256="d" * 64,
            sequence=sequence,
            sequence_sha256="b" * 64,
            shard_payloads=[shard0],
        )

    duplicate_episode = copy.deepcopy(shard1)
    duplicate_episode[1][0]["episode_id"] = "episode-0"
    with pytest.raises(TensorEvaluationError, match="pinned sequence"):
        merge_shard_payloads(
            plan=plan,
            plan_sha256="d" * 64,
            sequence=sequence,
            sequence_sha256="b" * 64,
            shard_payloads=[shard0, duplicate_episode],
        )

    export_drift = copy.deepcopy(shard1)
    for name in ("active_export_start", "active_export_end"):
        export_drift[0]["backend_identity"][name]["model_sha256"] = "e" * 64
    with pytest.raises(TensorEvaluationError, match="active export identity drifted"):
        merge_shard_payloads(
            plan=plan,
            plan_sha256="d" * 64,
            sequence=sequence,
            sequence_sha256="b" * 64,
            shard_payloads=[shard0, export_drift],
        )

    host_materialization = copy.deepcopy(shard1)
    host_materialization[0]["data_plane"]["hot_path_host_materializations"] = 1
    with pytest.raises(TensorEvaluationError, match="data-plane execution"):
        merge_shard_payloads(
            plan=plan,
            plan_sha256="d" * 64,
            sequence=sequence,
            sequence_sha256="b" * 64,
            shard_payloads=[shard0, host_materialization],
        )


def test_offline_bundle_merge_is_atomic_and_sha_verified(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence.json"
    sequence_path.write_text(
        json.dumps(_sequence_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sequence_sha256 = file_sha256(sequence_path)
    plan_payload = _plan_payload(sequence_sha256)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan_sha256 = file_sha256(plan_path)
    shard0_result, shard0_episodes = _shard_payload(
        "shard-0", 0, plan_sha256=plan_sha256, sequence_sha256=sequence_sha256
    )
    shard1_result, shard1_episodes = _shard_payload(
        "shard-1", 2, plan_sha256=plan_sha256, sequence_sha256=sequence_sha256
    )
    shard0 = tmp_path / "shard-0"
    shard1 = tmp_path / "shard-1"
    publish_result_bundle(
        shard0,
        result_name="shard_result.json",
        result=shard0_result,
        episodes=shard0_episodes,
    )
    publish_result_bundle(
        shard1,
        result_name="shard_result.json",
        result=shard1_result,
        episodes=shard1_episodes,
    )
    output = tmp_path / "union"

    merge_shard_bundles(
        output=output,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha256,
        sequence_path=sequence_path,
        expected_sequence_sha256=sequence_sha256,
        shard_paths=[shard0, shard1],
    )
    result, episodes = verify_result_bundle(output, result_name="rollout_union.json")

    assert result["schema_version"] == ROLLOUT_UNION_SCHEMA
    assert len(episodes) == 4
    with pytest.raises(FileExistsError):
        merge_shard_bundles(
            output=output,
            plan_path=plan_path,
            expected_plan_sha256=plan_sha256,
            sequence_path=sequence_path,
            expected_sequence_sha256=sequence_sha256,
            shard_paths=[shard0, shard1],
        )

    with (shard1 / "episodes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    fresh_output = tmp_path / "union-tampered"
    with pytest.raises(TensorEvaluationError, match="artifact SHA-256"):
        merge_shard_bundles(
            output=fresh_output,
            plan_path=plan_path,
            expected_plan_sha256=plan_sha256,
            sequence_path=sequence_path,
            expected_sequence_sha256=sequence_sha256,
            shard_paths=[shard0, shard1],
        )
