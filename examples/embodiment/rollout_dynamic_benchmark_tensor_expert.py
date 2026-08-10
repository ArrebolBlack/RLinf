#!/usr/bin/env python3
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

"""Run or merge caller-pinned tensor-expert benchmark rollout shards.

``run`` executes one shard in a fresh CUDA worker and atomically publishes only
that shard.  ``merge`` is offline and rejects a duplicate shard, duplicate episode,
missing episode, unexpected episode, or any policy/sequence/plan identity drift.
Physical GPUs and filesystem paths may differ between shards; portable source,
runtime, policy, reset, and active-export identities may not.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Keep the documented ``python examples/embodiment/<script>.py`` entrypoint
    # usable without relying on an ambient PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.embodiment.evaluate_dynamic_benchmark_tensor_expert import (
    EVALUATION_SCHEMA,
    WORKER_SPEC_SCHEMA,
    PinnedSequence,
    ProcessIdentity,
    TensorEvaluationError,
    assert_strict_finite,
    collect_process_identity,
    file_sha256,
    launch_fresh_worker,
    load_pinned_json,
    portable_export_identity,
    publish_result_bundle,
    require_sha256,
    runtime_request_payload,
    summarize_episodes,
    validate_backend_runtime_identity,
    verify_file_pin,
    verify_result_bundle,
)

SHARD_PLAN_SCHEMA = "rlinf-dynamic-benchmark-tensor-shard-plan-v0.1"
SHARD_RESULT_SCHEMA = "rlinf-dynamic-benchmark-tensor-shard-result-v0.1"
ROLLOUT_UNION_SCHEMA = "rlinf-dynamic-benchmark-tensor-rollout-union-v0.1"


@dataclass(frozen=True)
class ShardOwnership:
    """One contiguous, full-cohort episode range owned by exactly one shard."""

    shard_id: str
    num_envs: int
    start_ordinal: int
    episode_count: int

    @property
    def stop_ordinal_exclusive(self) -> int:
        """Return the first ordinal not owned by this shard."""

        return self.start_ordinal + self.episode_count


@dataclass(frozen=True)
class ShardPlan:
    """Exact union ownership for a pinned policy and validation sequence."""

    sequence_sha256: str
    policy_sha256: str
    episode_count: int
    shards: tuple[ShardOwnership, ...]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        sequence: PinnedSequence | None = None,
    ) -> ShardPlan:
        """Validate non-overlap, full-cohort alignment, and exact union coverage."""

        expected = {
            "schema_version",
            "sequence_sha256",
            "policy_sha256",
            "episode_count",
            "shards",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("shard plan keys do not match the released schema")
        if payload["schema_version"] != SHARD_PLAN_SCHEMA:
            raise ValueError("unsupported shard plan schema")
        sequence_sha256 = require_sha256(
            "shard plan sequence_sha256", payload["sequence_sha256"]
        )
        policy_sha256 = require_sha256(
            "shard plan policy_sha256", payload["policy_sha256"]
        )
        episode_count = payload["episode_count"]
        if (
            isinstance(episode_count, bool)
            or not isinstance(episode_count, int)
            or episode_count < 1
        ):
            raise ValueError("shard plan episode_count must be a positive integer")
        raw_shards = payload["shards"]
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError("shard plan requires at least one shard")
        shards: list[ShardOwnership] = []
        shard_ids: set[str] = set()
        for raw in raw_shards:
            if not isinstance(raw, Mapping) or set(raw) != {
                "shard_id",
                "num_envs",
                "start_ordinal",
                "episode_count",
            }:
                raise ValueError("shard ownership keys do not match the schema")
            shard_id = raw["shard_id"]
            if (
                not isinstance(shard_id, str)
                or not shard_id
                or shard_id.strip() != shard_id
                or shard_id in shard_ids
            ):
                raise ValueError("shard ids must be unique non-empty trimmed strings")
            shard_ids.add(shard_id)
            values = (raw["num_envs"], raw["start_ordinal"], raw["episode_count"])
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ):
                raise ValueError("shard ownership dimensions must be integers")
            ownership = ShardOwnership(
                shard_id=shard_id,
                num_envs=raw["num_envs"],
                start_ordinal=raw["start_ordinal"],
                episode_count=raw["episode_count"],
            )
            if (
                ownership.num_envs < 1
                or ownership.start_ordinal < 0
                or ownership.episode_count < 1
                or ownership.start_ordinal % ownership.num_envs
                or ownership.episode_count % ownership.num_envs
            ):
                raise ValueError(
                    "each shard must own positive, aligned, complete device cohorts"
                )
            shards.append(ownership)
        ordered = sorted(shards, key=lambda shard: shard.start_ordinal)
        if shards != ordered:
            raise ValueError("shards must be listed in ascending ordinal order")
        expected_start = 0
        for shard in shards:
            if shard.start_ordinal != expected_start:
                relation = "overlap" if shard.start_ordinal < expected_start else "gap"
                raise ValueError(f"shard ownership has an ordinal {relation}")
            expected_start = shard.stop_ordinal_exclusive
        if expected_start != episode_count:
            raise ValueError("shard ownership union is not exact")
        if sequence is not None and episode_count != len(sequence.requests):
            raise ValueError("shard plan episode count differs from the sequence")
        return cls(
            sequence_sha256=sequence_sha256,
            policy_sha256=policy_sha256,
            episode_count=episode_count,
            shards=tuple(shards),
        )

    def shard(self, shard_id: str) -> ShardOwnership:
        """Return one uniquely named shard."""

        matches = [shard for shard in self.shards if shard.shard_id == shard_id]
        if len(matches) != 1:
            raise ValueError(f"shard plan does not contain {shard_id!r}")
        return matches[0]


def _portable_source_identity(result: Mapping[str, Any]) -> Mapping[str, Any]:
    source = result.get("source_identity")
    if not isinstance(source, Mapping) or source.get("start") != source.get("end"):
        raise TensorEvaluationError("shard source start/end identity is invalid")
    if set(source) != {"start", "end"} or not isinstance(source["start"], Mapping):
        raise TensorEvaluationError("shard source receipt fields are not exact")
    portable: dict[str, Any] = {}
    for name, identity in source["start"].items():
        if not isinstance(identity, Mapping):
            raise ValueError("shard source identity row must be a mapping")
        portable[name] = {
            "module": identity.get("module"),
            "commit": identity.get("commit"),
            "tree": identity.get("tree"),
            "tracked_worktree_clean": identity.get("tracked_worktree_clean"),
        }
    assert_strict_finite(portable)
    return portable


def _portable_runtime_identity(result: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = result.get("runtime_identity")
    if not isinstance(runtime, Mapping) or runtime.get("start") != runtime.get("end"):
        raise TensorEvaluationError("shard runtime start/end identity is invalid")
    if set(runtime) != {"start", "end"}:
        raise TensorEvaluationError("shard runtime receipt fields are not exact")
    start = runtime["start"]
    if not isinstance(start, Mapping) or set(start) != {"path", "sha256", "payload"}:
        raise ValueError("shard runtime identity must be a mapping")
    require_sha256("shard runtime manifest SHA-256", start.get("sha256"))
    if not isinstance(start.get("payload"), Mapping):
        raise ValueError("shard runtime payload must be a mapping")
    portable = {"sha256": start.get("sha256"), "payload": start.get("payload")}
    assert_strict_finite(portable)
    return portable


def _validate_process_receipt(result: Mapping[str, Any]) -> None:
    process = result.get("process_identity")
    if not isinstance(process, Mapping) or set(process) != {
        "parent_start",
        "parent_end",
        "child_start",
        "child_end",
    }:
        raise TensorEvaluationError("shard process receipt fields are not exact")
    parent_start = ProcessIdentity(**process["parent_start"])
    parent_end = ProcessIdentity(**process["parent_end"])
    child_start = ProcessIdentity(**process["child_start"])
    child_end = ProcessIdentity(**process["child_end"])
    direct_parent_mismatch = (
        child_start.identity_source != "windows_kernel_times"
        and child_start.parent_pid != parent_start.pid
    )
    if (
        parent_start != parent_end
        or child_start != child_end
        or child_start.pid == parent_start.pid
        or child_start.boot_id != parent_start.boot_id
        or direct_parent_mismatch
    ):
        raise TensorEvaluationError("shard process PID/boot/start receipt is invalid")


def _validate_backend_receipt(
    result: Mapping[str, Any], sequence: PinnedSequence
) -> None:
    _portable_source_identity(result)
    _portable_runtime_identity(result)
    backend = result.get("backend_identity")
    expected_fields = {
        "stable_start",
        "stable_end",
        "active_export_start",
        "active_export_end",
        "portable_active_export",
        "device_start",
        "device_end",
    }
    if not isinstance(backend, Mapping) or set(backend) != expected_fields:
        raise TensorEvaluationError("shard backend identity fields are not exact")
    for start_name, end_name in (
        ("stable_start", "stable_end"),
        ("active_export_start", "active_export_end"),
        ("device_start", "device_end"),
    ):
        if backend.get(start_name) != backend.get(end_name):
            raise TensorEvaluationError(
                f"shard backend {start_name}/{end_name} identity drifted"
            )
    expected_export = dict(sequence.active_export_identity)
    for name in (
        "active_export_start",
        "active_export_end",
        "portable_active_export",
    ):
        if portable_export_identity(backend.get(name)) != expected_export:
            raise TensorEvaluationError(f"shard {name} active export identity drifted")
    stable = backend["stable_start"]
    device = backend["device_start"]
    if (
        not isinstance(stable, Mapping)
        or not isinstance(device, Mapping)
        or stable.get("device_identity") != device
    ):
        raise TensorEvaluationError("shard stable/device identity linkage drifted")
    source = result["source_identity"]["start"]
    runtime = result["runtime_identity"]["start"]
    validate_backend_runtime_identity(
        stable,
        runtime_payload=runtime["payload"],
        source_snapshot=source,
        sequence=sequence,
    )


def _validate_data_plane_receipt(
    result: Mapping[str, Any],
    *,
    ownership: ShardOwnership,
    sequence: PinnedSequence,
    allocated_steps: int,
) -> None:
    """Validate the shard's device-only execution and boundary materializations."""

    data_plane = result.get("data_plane")
    expected_fields = {
        "step_api",
        "device_ledger",
        "hot_path_host_materializations",
        "terminal_control_plane_materializations",
        "transport_checks",
        "expected_transport_checks",
        "last_transport_receipt",
        "active_cohort_end",
    }
    if not isinstance(data_plane, Mapping) or set(data_plane) != expected_fields:
        raise TensorEvaluationError("shard data-plane receipt fields are not exact")
    cohort_count = ownership.episode_count // ownership.num_envs
    expected_transport_checks = allocated_steps // ownership.num_envs
    if (
        allocated_steps % ownership.num_envs
        or data_plane.get("step_api") != "step_device"
        or data_plane.get("device_ledger") != "preallocated_per_cohort"
        or data_plane.get("hot_path_host_materializations") != 0
        or data_plane.get("terminal_control_plane_materializations") != cohort_count
        or data_plane.get("transport_checks") != expected_transport_checks
        or data_plane.get("expected_transport_checks") != expected_transport_checks
        or not isinstance(data_plane.get("last_transport_receipt"), Mapping)
    ):
        raise TensorEvaluationError("shard data-plane execution receipt drifted")
    active = data_plane.get("active_cohort_end")
    final_ordinals = list(
        range(
            ownership.stop_ordinal_exclusive - ownership.num_envs,
            ownership.stop_ordinal_exclusive,
        )
    )
    final_episode_ids = [
        runtime_request_payload(sequence, ordinal)["episode_id"]
        for ordinal in final_ordinals
    ]
    if not isinstance(active, Mapping) or (
        active.get("manifest_ordinals") != final_ordinals
        or active.get("episode_ids") != final_episode_ids
    ):
        raise TensorEvaluationError("shard active cohort identity drifted")


def merge_shard_payloads(
    *,
    plan: ShardPlan,
    plan_sha256: str,
    sequence: PinnedSequence,
    sequence_sha256: str,
    shard_payloads: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    """Merge verified shard payloads with exact union and identity checks."""

    require_sha256("plan_sha256", plan_sha256)
    if sequence_sha256 != plan.sequence_sha256:
        raise TensorEvaluationError("plan and sequence SHA-256 differ")
    if plan.episode_count != len(sequence.requests):
        raise TensorEvaluationError("plan episode count differs from pinned sequence")
    expected_by_id = {shard.shard_id: shard for shard in plan.shards}
    observed_by_id: dict[
        str, tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ] = {}
    reference_source: Mapping[str, Any] | None = None
    reference_runtime: Mapping[str, Any] | None = None
    for result, episodes in shard_payloads:
        if (
            result.get("schema_version") != SHARD_RESULT_SCHEMA
            or result.get("worker_schema_version") != EVALUATION_SCHEMA
        ):
            raise ValueError("shard result schema mismatch")
        if result.get("status") != "complete" or result.get("mode") != "rollout_shard":
            raise TensorEvaluationError("shard result is not a complete rollout shard")
        claim = result.get("claim_scope")
        if (
            not isinstance(claim, Mapping)
            or claim.get("production_qualified") is not False
            or claim.get("science_gate_state") != "not_run"
        ):
            raise TensorEvaluationError("shard claim scope is not validation-only")
        plan_identity = result.get("shard_plan_identity")
        expected_plan_fields = {
            "path",
            "sha256",
            "sequence_sha256",
            "policy_sha256",
            "shard_id",
            "shard_count",
        }
        if not isinstance(plan_identity, Mapping) or (
            set(plan_identity) != expected_plan_fields
            or not isinstance(plan_identity.get("path"), str)
            or not plan_identity.get("path")
            or plan_identity.get("sha256") != plan_sha256
            or plan_identity.get("sequence_sha256") != sequence_sha256
            or plan_identity.get("policy_sha256") != plan.policy_sha256
            or plan_identity.get("shard_count") != len(plan.shards)
        ):
            raise TensorEvaluationError("shard plan identity drifted")
        _validate_process_receipt(result)
        _validate_backend_receipt(result, sequence)
        shard_id = plan_identity.get("shard_id")
        if not isinstance(shard_id, str) or shard_id not in expected_by_id:
            raise ValueError("shard result has an unexpected shard id")
        if shard_id in observed_by_id:
            raise ValueError(f"duplicate shard result {shard_id!r}")
        policy = result.get("policy_identity")
        sequence_identity = result.get("sequence_identity")
        backend = result.get("backend_identity")
        expected_sequence_identity = {
            "path",
            "sha256",
            "manifest_sha256",
            "task_id",
            "split",
            "manifest_seed",
            "api_version",
            "request_count",
        }
        if (
            not isinstance(policy, Mapping)
            or policy.get("sha256") != plan.policy_sha256
            or not isinstance(sequence_identity, Mapping)
            or set(sequence_identity) != expected_sequence_identity
            or not isinstance(sequence_identity.get("path"), str)
            or not sequence_identity.get("path")
            or sequence_identity.get("sha256") != sequence_sha256
            or sequence_identity.get("manifest_sha256") != sequence.manifest_sha256
            or sequence_identity.get("task_id") != sequence.task_id
            or sequence_identity.get("split") != sequence.split
            or sequence_identity.get("manifest_seed") != sequence.manifest_seed
            or sequence_identity.get("api_version") != sequence.api_version
            or sequence_identity.get("request_count") != len(sequence.requests)
            or not isinstance(backend, Mapping)
        ):
            raise TensorEvaluationError("shard policy/sequence/export identity drifted")
        source = _portable_source_identity(result)
        runtime = _portable_runtime_identity(result)
        if reference_source is None:
            reference_source = source
            reference_runtime = runtime
        elif source != reference_source or runtime != reference_runtime:
            raise TensorEvaluationError(
                "portable source/runtime identity differs across shards"
            )
        observed_by_id[shard_id] = (result, episodes)
    if set(observed_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(observed_by_id))
        extra = sorted(set(observed_by_id) - set(expected_by_id))
        raise ValueError(
            f"shard result set is not exact: missing={missing}, extra={extra}"
        )

    merged: list[Mapping[str, Any]] = []
    seen_ordinals: set[int] = set()
    seen_episode_ids: set[str] = set()
    allocated_steps = 0
    valid_steps = 0
    aggregate_rollout_seconds = 0.0
    shard_summaries: dict[str, Any] = {}
    shard_receipts = []
    for expected in plan.shards:
        result, episodes = observed_by_id[expected.shard_id]
        expected_ordinals = list(
            range(expected.start_ordinal, expected.stop_ordinal_exclusive)
        )
        observed_ordinals = [row.get("ordinal") for row in episodes]
        if observed_ordinals != expected_ordinals:
            raise ValueError(
                f"shard {expected.shard_id!r} ownership/order is not exact"
            )
        ownership = result.get("ownership")
        if ownership != {
            "start_ordinal": expected.start_ordinal,
            "episode_count": expected.episode_count,
            "stop_ordinal_exclusive": expected.stop_ordinal_exclusive,
        }:
            raise TensorEvaluationError("shard ownership receipt differs from its plan")
        for row in episodes:
            ordinal = row["ordinal"]
            episode_id = row["episode_id"]
            request = runtime_request_payload(sequence, ordinal)
            if episode_id != request["episode_id"]:
                raise TensorEvaluationError(
                    "episode id differs from pinned sequence ordinal"
                )
            if ordinal in seen_ordinals or episode_id in seen_episode_ids:
                raise ValueError("duplicate episode ownership across shards")
            seen_ordinals.add(ordinal)
            seen_episode_ids.add(episode_id)
            merged.append(row)
        metrics = result.get("metrics")
        throughput = metrics.get("throughput") if isinstance(metrics, Mapping) else None
        if not isinstance(throughput, Mapping):
            raise ValueError("shard result lacks throughput metrics")
        shard_allocated = throughput.get("allocated_env_steps")
        shard_valid = throughput.get("valid_env_steps")
        shard_seconds = throughput.get("rollout_seconds")
        if (
            isinstance(shard_allocated, bool)
            or not isinstance(shard_allocated, int)
            or shard_allocated < 1
            or isinstance(shard_valid, bool)
            or not isinstance(shard_valid, int)
            or not 1 <= shard_valid <= shard_allocated
            or isinstance(shard_seconds, bool)
            or not isinstance(shard_seconds, (int, float))
            or not math.isfinite(float(shard_seconds))
            or float(shard_seconds) <= 0.0
        ):
            raise ValueError("shard throughput metrics are invalid")
        allocated_steps += shard_allocated
        valid_steps += shard_valid
        aggregate_rollout_seconds += float(shard_seconds)
        expected_metrics = summarize_episodes(
            episodes,
            allocated_steps=shard_allocated,
            valid_steps=shard_valid,
            rollout_seconds=float(shard_seconds),
        )
        if metrics != expected_metrics:
            raise TensorEvaluationError("shard metrics differ from its episode ledger")
        _validate_data_plane_receipt(
            result,
            ownership=expected,
            sequence=sequence,
            allocated_steps=shard_allocated,
        )
        shard_summaries[expected.shard_id] = metrics
        shard_receipts.append(
            {
                "shard_id": expected.shard_id,
                "process_identity": result.get("process_identity"),
                "backend_identity": result.get("backend_identity"),
            }
        )
    expected_ordinals = set(range(plan.episode_count))
    expected_episode_ids = {
        runtime_request_payload(sequence, ordinal)["episode_id"]
        for ordinal in range(len(sequence.requests))
    }
    if seen_ordinals != expected_ordinals or seen_episode_ids != expected_episode_ids:
        missing_ordinals = sorted(expected_ordinals - seen_ordinals)
        extra_ordinals = sorted(seen_ordinals - expected_ordinals)
        missing_ids = sorted(expected_episode_ids - seen_episode_ids)
        extra_ids = sorted(seen_episode_ids - expected_episode_ids)
        raise ValueError(
            "offline episode union is not exact: "
            f"missing_ordinals={missing_ordinals}, extra_ordinals={extra_ordinals}, "
            f"missing_ids={missing_ids}, extra_ids={extra_ids}"
        )
    merged.sort(key=lambda row: row["ordinal"])
    metrics = dict(
        summarize_episodes(
            merged,
            allocated_steps=allocated_steps,
            valid_steps=valid_steps,
            rollout_seconds=aggregate_rollout_seconds,
        )
    )
    metrics["throughput"]["denominator"] = "aggregate_device_rollout_seconds"
    metrics["throughput"]["cross_shard_wall_rate_available"] = False
    result = {
        "schema_version": ROLLOUT_UNION_SCHEMA,
        "status": "complete",
        "mode": "offline_union_exact_merge",
        "claim_scope": {
            "science_gate_state": "not_run",
            "production_qualified": False,
            "statement": "Benchmark rollout evidence only; no production-quality claim.",
        },
        "policy_sha256": plan.policy_sha256,
        "sequence_identity": {
            "sha256": sequence_sha256,
            "manifest_sha256": sequence.manifest_sha256,
            "episode_count": len(sequence.requests),
            "task_id": sequence.task_id,
            "split": sequence.split,
        },
        "shard_plan_identity": {
            "sha256": plan_sha256,
            "shard_count": len(plan.shards),
            "union_exact": True,
        },
        "portable_source_identity": reference_source,
        "portable_runtime_identity": reference_runtime,
        "portable_active_export": dict(sequence.active_export_identity),
        "metrics": metrics,
        "shard_metrics": shard_summaries,
        "shard_receipts": shard_receipts,
    }
    assert_strict_finite(result)
    return result, tuple(merged)


def merge_shard_bundles(
    *,
    output: Path,
    plan_path: Path,
    expected_plan_sha256: str,
    sequence_path: Path,
    expected_sequence_sha256: str,
    shard_paths: Sequence[Path],
) -> Mapping[str, str]:
    """Verify, union, and atomically publish a complete set of shard bundles."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    sequence_payload = load_pinned_json(
        sequence_path,
        expected_sequence_sha256,
        name="validation sequence",
    )
    sequence = PinnedSequence.from_payload(sequence_payload)
    plan_payload = load_pinned_json(
        plan_path,
        expected_plan_sha256,
        name="shard plan",
    )
    plan = ShardPlan.from_payload(plan_payload, sequence=sequence)
    if plan.sequence_sha256 != expected_sequence_sha256:
        raise TensorEvaluationError("shard plan does not pin the supplied sequence")
    payloads = [
        verify_result_bundle(path, result_name="shard_result.json")
        for path in shard_paths
    ]
    result, episodes = merge_shard_payloads(
        plan=plan,
        plan_sha256=expected_plan_sha256,
        sequence=sequence,
        sequence_sha256=expected_sequence_sha256,
        shard_payloads=payloads,
    )
    if file_sha256(plan_path) != expected_plan_sha256:
        raise TensorEvaluationError("shard plan changed during offline merge")
    if file_sha256(sequence_path) != expected_sequence_sha256:
        raise TensorEvaluationError("validation sequence changed during offline merge")
    return publish_result_bundle(
        output,
        result_name="rollout_union.json",
        result=result,
        episodes=episodes,
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--expected-sequence-sha256", required=True)
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--expected-shard-plan-sha256", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--expected-runtime-manifest-sha256", required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-timeout-s", type=float)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one caller-owned shard")
    _add_run_arguments(run)
    merge = subparsers.add_parser("merge", help="offline union-exact shard merge")
    merge.add_argument("--sequence", type=Path, required=True)
    merge.add_argument("--expected-sequence-sha256", required=True)
    merge.add_argument("--shard-plan", type=Path, required=True)
    merge.add_argument("--expected-shard-plan-sha256", required=True)
    merge.add_argument("--shard-result", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sequence_payload = load_pinned_json(
        args.sequence,
        args.expected_sequence_sha256,
        name="validation sequence",
    )
    sequence = PinnedSequence.from_payload(sequence_payload)
    plan_payload = load_pinned_json(
        args.shard_plan,
        args.expected_shard_plan_sha256,
        name="shard plan",
    )
    plan = ShardPlan.from_payload(plan_payload, sequence=sequence)
    if plan.sequence_sha256 != args.expected_sequence_sha256:
        raise TensorEvaluationError("shard plan sequence pin differs from CLI")
    if plan.policy_sha256 != args.expected_policy_sha256:
        raise TensorEvaluationError("shard plan policy pin differs from CLI")
    verify_file_pin(args.policy, args.expected_policy_sha256, name="policy")
    load_pinned_json(
        args.source_manifest,
        args.expected_source_manifest_sha256,
        name="source manifest",
    )
    load_pinned_json(
        args.runtime_manifest,
        args.expected_runtime_manifest_sha256,
        name="runtime manifest",
    )
    ownership = plan.shard(args.shard_id)
    parent_start = collect_process_identity()
    spec = {
        "schema_version": WORKER_SPEC_SCHEMA,
        "mode": "rollout_shard",
        "policy_path": str(args.policy.resolve(strict=True)),
        "policy_sha256": args.expected_policy_sha256,
        "sequence_path": str(args.sequence.resolve(strict=True)),
        "sequence_sha256": args.expected_sequence_sha256,
        "source_manifest_path": str(args.source_manifest.resolve(strict=True)),
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "runtime_manifest_path": str(args.runtime_manifest.resolve(strict=True)),
        "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
        "export_dir": str(args.export_dir.resolve(strict=True)),
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "device_ordinal": args.device_ordinal,
        "image_size": args.image_size,
        "num_envs": ownership.num_envs,
        "start_ordinal": ownership.start_ordinal,
        "episode_count": ownership.episode_count,
        "parent_process_start": asdict(parent_start),
    }
    evaluator_script = Path(__file__).with_name(
        "evaluate_dynamic_benchmark_tensor_expert.py"
    )
    result = dict(
        launch_fresh_worker(
            spec,
            script_path=evaluator_script,
            timeout_s=args.worker_timeout_s,
        )
    )
    if (
        result.get("schema_version") != EVALUATION_SCHEMA
        or result.get("status") != "complete"
        or result.get("mode") != "rollout_shard"
    ):
        raise TensorEvaluationError("fresh rollout worker returned an invalid result")
    parent_end = collect_process_identity()
    if parent_end != parent_start:
        raise TensorEvaluationError("rollout parent boot/start identity drifted")
    result["worker_schema_version"] = result["schema_version"]
    result["schema_version"] = SHARD_RESULT_SCHEMA
    result["process_identity"]["parent_end"] = asdict(parent_end)
    result["shard_plan_identity"] = {
        "path": str(args.shard_plan.resolve(strict=True)),
        "sha256": args.expected_shard_plan_sha256,
        "sequence_sha256": plan.sequence_sha256,
        "policy_sha256": plan.policy_sha256,
        "shard_id": ownership.shard_id,
        "shard_count": len(plan.shards),
    }
    episodes = result.pop("episodes")
    if file_sha256(args.shard_plan) != args.expected_shard_plan_sha256:
        raise TensorEvaluationError("shard plan changed during rollout")
    publish_result_bundle(
        args.output,
        result_name="shard_result.json",
        result=result,
        episodes=episodes,
    )
    return 0


def _merge(args: argparse.Namespace) -> int:
    merge_shard_bundles(
        output=args.output,
        plan_path=args.shard_plan,
        expected_plan_sha256=args.expected_shard_plan_sha256,
        sequence_path=args.sequence,
        expected_sequence_sha256=args.expected_sequence_sha256,
        shard_paths=args.shard_result,
    )
    return 0


def main() -> int:
    """Run one shard or merge the exact offline union."""

    args = _parser().parse_args()
    if args.command == "run":
        return _run(args)
    return _merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
