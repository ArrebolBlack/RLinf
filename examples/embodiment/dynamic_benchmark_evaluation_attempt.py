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

"""Canonical optimal-attempt adapters for formal benchmark evaluations.

The optimal exporter owns the attempt schema, tape writer, score, eligibility,
and T5 issued/applied timing semantics.  Evaluation producers deliberately
resolve those helpers lazily: the optimal exporter imports the expert evaluator
for policy loading, so a top-level import here would create a module cycle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _optimal() -> Any:
    """Return the canonical optimal exporter without creating an import cycle."""

    from examples.embodiment import (
        export_dynamic_benchmark_optimal_trajectories as optimal,
    )

    return optimal


def _quality_v4() -> Any:
    """Return the parallel Qv4 artifact implementation lazily."""

    from examples.embodiment import dynamic_benchmark_quality_v4

    return dynamic_benchmark_quality_v4


def attempt_schema_version() -> str:
    """Return the single canonical production attempt schema."""

    return str(_optimal().ATTEMPT_SCHEMA)


def validate_formal_quality_v2_thresholds(
    payload: Mapping[str, Any],
    *,
    task_id: str,
    thresholds_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless thresholds are the formal frozen v0.3 contract."""

    return _optimal()._quality_v2_dominance_contract(
        payload,
        task=task_id,
        thresholds_sha256=thresholds_sha256,
        require_formal_freeze=True,
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _quality_source_arrays(observations: Sequence[Any]) -> dict[str, np.ndarray]:
    """Serialize the replay sources consumed by the canonical Qv3 auditor."""

    return {
        "eef_pose_xyzw": np.stack(
            [
                np.asarray(observation.privileged["eef_pose_xyzw"], dtype=np.float64)
                for observation in observations
            ]
        ),
        "fingerpad_closing_axis_world": np.stack(
            [
                np.asarray(
                    observation.privileged["fingerpad_closing_axis_world"],
                    dtype=np.float64,
                )
                for observation in observations
            ]
        ),
        "object_pose_wxyz": np.stack(
            [
                np.asarray(observation.privileged["object_pose_wxyz"], dtype=np.float64)
                for observation in observations
            ]
        ),
        "fingerpad_contact_flags": np.stack(
            [
                np.asarray(
                    observation.privileged["fingerpad_contact_flags"],
                    dtype=np.float64,
                )
                for observation in observations
            ]
        ),
    }


def materialize_evaluation_attempt(
    output: Path,
    record: Mapping[str, Any],
    *,
    candidate_index: int,
    raw_env: Any,
    observations: Sequence[Any],
    states: Sequence[np.ndarray],
    policy_actions: Sequence[np.ndarray],
    rewards: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    quality_v2_thresholds_sha256: str,
) -> dict[str, Any]:
    """Attach one evaluation-relative canonical v0.3 attempt tape and receipt."""

    optimal = _optimal()
    attempt_schema = str(optimal.ATTEMPT_SCHEMA)
    result = dict(record)
    task_id = str(result.get("task_id", ""))
    episode_id = str(result.get("episode_id", ""))
    if not task_id or not episode_id:
        raise ValueError("formal evaluation attempt identity is missing")
    episode_path = Path(episode_id)
    if (
        episode_path.is_absolute()
        or len(episode_path.parts) != 1
        or episode_id in {".", ".."}
    ):
        raise ValueError("formal evaluation episode identity is not path-safe")
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
    ):
        raise ValueError("formal evaluation candidate index must be non-negative")
    gate = result.get("quality_v2_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("formal evaluation attempt is missing its Qv3 gate")
    if gate.get("contract_sha256") != quality_v2_thresholds_sha256:
        raise ValueError(
            "formal evaluation Qv3 gate is not bound to its contract SHA-256"
        )

    action_array = np.asarray(result.get("actions"), dtype=np.float64)
    state_array = np.stack(states).astype(np.float32)
    policy_action_array = np.stack(policy_actions).astype(np.float32)
    reward_array = np.asarray(rewards, dtype=np.float32)
    terminated_array = np.asarray(terminated, dtype=np.bool_)
    truncated_array = np.asarray(truncated, dtype=np.bool_)
    control_steps = int(result.get("control_steps", -1))
    if (
        control_steps < 1
        or action_array.shape != (control_steps, 7)
        or state_array.ndim != 2
        or state_array.shape[0] != control_steps + 1
        or policy_action_array.shape != (control_steps, 7)
        or reward_array.shape != (control_steps,)
        or terminated_array.shape != (control_steps,)
        or truncated_array.shape != (control_steps,)
    ):
        raise ValueError("formal evaluation attempt arrays do not align")
    if (
        np.any(terminated_array[:-1])
        or np.any(truncated_array[:-1])
        or bool(terminated_array[-1]) == bool(truncated_array[-1])
    ):
        raise ValueError("formal evaluation attempt has an invalid done tape")

    finite_and_bounded = bool(
        np.all(np.isfinite(state_array))
        and np.all(np.isfinite(policy_action_array))
        and np.all(np.isfinite(action_array))
        and np.all(np.isfinite(reward_array))
        and np.all(np.abs(policy_action_array) <= 1.0)
        and np.all(np.abs(action_array) <= 1.0)
    )
    if task_id == "t5_replan":
        causal_fields, causal_arrays = optimal._t5_replan_causal_evidence(
            raw_env,
            control_steps=control_steps,
        )
    else:
        causal_fields = {
            "issued_equals_applied": True,
            "t5_replan_causal_timing_passed": None,
            "impact_end_to_first_qualifying_applied_correction_s": None,
        }
        causal_arrays = {}

    arrays = {
        "states": state_array,
        "policy_actions": policy_action_array,
        "actions": action_array,
        "rewards": reward_array,
        "terminated": terminated_array,
        "truncated": truncated_array,
        **_quality_source_arrays(observations),
        **causal_arrays,
    }
    relative, tape_sha256 = optimal._write_attempt_tape(
        output,
        episode_id=episode_id,
        candidate_index=candidate_index,
        arrays=arrays,
    )
    replay_validation = result.get("replay_validation")
    if not isinstance(replay_validation, Mapping):
        raise ValueError("formal evaluation attempt replay receipt is missing")
    result.update(
        attempt_schema_version=attempt_schema,
        attempt_tape=relative,
        attempt_tape_sha256=tape_sha256,
        finite_and_bounded=finite_and_bounded,
        state_sha256=_array_sha256(state_array),
        policy_action_sha256=_array_sha256(policy_action_array),
        action_sha256=_array_sha256(action_array),
        reward_sha256=_array_sha256(reward_array),
        replay_validation_sha256=optimal._payload_sha256(replay_validation),
        quality_v2_events_by_observation=[
            [str(event.name) for event in observation.events_since_last_observation]
            for observation in observations
        ],
        **causal_fields,
    )
    audit_view = {**result, "schema_version": attempt_schema}
    result["quality_score"] = list(optimal._quality_score(audit_view))
    result["eligible"] = optimal._eligible(audit_view)
    return result


def materialize_quality_v4_evaluation_attempt(
    output: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and publish one lightweight Qv4 attempt in its new namespace."""

    quality_v4 = _quality_v4()
    output_root = Path(output)
    attempt = quality_v4.build_quality_v4_attempt(source)
    path = quality_v4.write_quality_v4_attempt(output_root, attempt)
    lightweight_path = quality_v4.write_quality_v4_lightweight_source(
        output_root,
        source=source,
        recorded_attempt=attempt,
    )
    lightweight_audit = quality_v4.audit_quality_v4_lightweight_source(lightweight_path)
    return {
        "attempt": attempt,
        "attempt_path": path.relative_to(output_root).as_posix(),
        "lightweight_source_path": lightweight_path.relative_to(output_root).as_posix(),
        "lightweight_source_audit": lightweight_audit,
    }


def load_quality_v4_rollout_reference(
    root: Path,
    *,
    task_id: str,
    episode_id: str,
    expected_state_schema_sha256: str,
) -> dict[str, Any]:
    """Load one pre-registered reset-bound Qv4 path/orientation contract."""

    from se3_wam.benchmark.contracts import stable_sha256

    path = Path(root) / task_id / f"{episode_id}.json"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Qv4 rollout reference must be a JSON object")
    if payload.get("task_id") != task_id or payload.get("episode_id") != episode_id:
        raise ValueError("Qv4 rollout reference identity mismatch")
    reference_unsigned = dict(payload)
    reference_sha256 = reference_unsigned.pop("reference_sha256", None)
    if (
        not isinstance(reference_sha256, str)
        or len(reference_sha256) != 64
        or stable_sha256(reference_unsigned) != reference_sha256
    ):
        raise ValueError("Qv4 rollout reference SHA-256 mismatch")
    field_contract = payload.get("field_contract")
    if not isinstance(field_contract, Mapping) or (
        field_contract.get("state_schema_sha256") != expected_state_schema_sha256
    ):
        raise ValueError("Qv4 rollout reference state-schema identity mismatch")
    return dict(payload)


def build_quality_v4_fresh_replay_attempt(
    *,
    record: Mapping[str, Any],
    raw_env: Any,
    observations: list[Any],
    actions: np.ndarray,
    rewards: list[float],
    outcomes: list[tuple[Any, ...]],
    physics_samples: list[Any],
    events: tuple[Any, ...],
    thresholds: Mapping[str, Any],
    reference_contract: Mapping[str, Any],
    base_replay_validation: Mapping[str, Any],
    replay_capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild original/replay Qv4 tapes without publishing an artifact."""

    from se3_wam.benchmark.trajectory_quality_v4 import (
        compare_replayed_observations_v4,
    )

    quality_v4 = _quality_v4()
    task_id = str(record["task_id"])
    task_thresholds = thresholds.get("tasks", {}).get(task_id)
    if not isinstance(task_thresholds, Mapping) or not isinstance(
        task_thresholds.get("vision_tolerance"), Mapping
    ):
        raise ValueError("Qv4 thresholds have no task-specific vision tolerance")
    vision_tolerance = task_thresholds["vision_tolerance"]
    replayed_observations = replay_capture.get("observations")
    if not isinstance(replayed_observations, tuple):
        raise ValueError("Qv4 fresh replay did not capture observations")
    observation_comparison = compare_replayed_observations_v4(
        observations,
        replayed_observations,
        vision_tolerance=vision_tolerance,
    )
    placeholder_replay = {
        "vision_tolerance_sha256": vision_tolerance["tolerance_sha256"]
    }
    original_source = quality_v4.build_quality_v4_rollout_source(
        record=record,
        raw_env=raw_env,
        observations=observations,
        issued_actions=actions,
        rewards=rewards,
        outcomes=outcomes,
        physics_samples=physics_samples,
        events=events,
        thresholds=thresholds,
        reference_contract=reference_contract,
        replay_validation=placeholder_replay,
    )
    replayed_rewards = replay_capture.get("rewards")
    replayed_outcomes = replay_capture.get("outcomes")
    replayed_physics = replay_capture.get("physics_samples")
    replayed_events = replay_capture.get("events")
    replayed_raw_env = replay_capture.get("raw_env")
    if (
        not all(
            isinstance(value, tuple)
            for value in (
                replayed_rewards,
                replayed_outcomes,
                replayed_physics,
                replayed_events,
            )
        )
        or replayed_raw_env is None
    ):
        raise ValueError("Qv4 fresh replay source capture is incomplete")
    replayed_record = dict(record)
    replayed_record["return"] = float(sum(replayed_rewards))
    replayed_source = quality_v4.build_quality_v4_rollout_source(
        record=replayed_record,
        raw_env=replayed_raw_env,
        observations=replayed_observations,
        issued_actions=actions,
        rewards=replayed_rewards,
        outcomes=replayed_outcomes,
        physics_samples=replayed_physics,
        events=replayed_events,
        thresholds=thresholds,
        reference_contract=reference_contract,
        replay_validation=placeholder_replay,
    )
    finalized_source, attempt = quality_v4.finalize_quality_v4_fresh_replay(
        original_source=original_source,
        replayed_source=replayed_source,
        base_replay_validation=base_replay_validation,
        observation_comparison=observation_comparison,
    )
    return {
        "source": finalized_source,
        "attempt": attempt,
        "fresh_replay_observation_comparison": observation_comparison,
    }


def materialize_quality_v4_fresh_replay_attempt(
    *,
    output: Path,
    record: Mapping[str, Any],
    raw_env: Any,
    observations: list[Any],
    actions: np.ndarray,
    rewards: list[float],
    outcomes: list[tuple[Any, ...]],
    physics_samples: list[Any],
    events: tuple[Any, ...],
    thresholds: Mapping[str, Any],
    reference_contract: Mapping[str, Any],
    base_replay_validation: Mapping[str, Any],
    replay_capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild Qv4 tapes and publish one lightweight evaluation candidate."""

    built = build_quality_v4_fresh_replay_attempt(
        record=record,
        raw_env=raw_env,
        observations=observations,
        actions=actions,
        rewards=rewards,
        outcomes=outcomes,
        physics_samples=physics_samples,
        events=events,
        thresholds=thresholds,
        reference_contract=reference_contract,
        base_replay_validation=base_replay_validation,
        replay_capture=replay_capture,
    )
    finalized_source = built["source"]
    attempt = built["attempt"]
    observation_comparison = built["fresh_replay_observation_comparison"]
    materialized = materialize_quality_v4_evaluation_attempt(output, finalized_source)
    if materialized["attempt"] != attempt:
        raise RuntimeError("Qv4 materialized attempt differs from fresh-replay gate")
    layer1 = attempt["layer1_gate"]
    layer2 = attempt["layer2_gate"]
    return {
        "schema_version": attempt["schema_version"],
        "attempt_path": materialized["attempt_path"],
        "attempt_sha256": attempt["attempt_sha256"],
        "source_sha256": attempt["source_sha256"],
        "lightweight_source_path": materialized["lightweight_source_path"],
        "lightweight_source_audit": materialized["lightweight_source_audit"],
        "layer1_passed": layer1["passed"],
        "layer1_reason_codes": layer1["reason_codes"],
        "layer2_passed": layer2["passed"],
        "layer2_reason_codes": layer2["reason_codes"],
        "eligible": attempt["eligible"],
        "formal_thresholds_frozen": attempt["formal_thresholds_frozen"],
        "thresholds_sha256": attempt["thresholds_sha256"],
        "orientation_contract_sha256": attempt["orientation_contract_sha256"],
        "field_contract_sha256": attempt["field_contract_sha256"],
        "summary_sha256": attempt["summary"]["summary_sha256"],
        "fresh_replay_observation_comparison": observation_comparison,
    }


def materialize_quality_v4_winner_export(
    output: Path,
    *,
    source: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the selected winner's full HDF5 and independently re-gate it."""

    quality_v4 = _quality_v4()
    output_root = Path(output)
    episode_id = attempt.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("Qv4 winner attempt has no episode identity")
    export_path = (
        output_root
        / quality_v4.QUALITY_V4_FULL_EXPORT_SUBDIRECTORY
        / f"{episode_id}.h5"
    )
    quality_v4.write_quality_v4_full_export(
        export_path,
        source=source,
        recorded_attempt=attempt,
    )
    full_export_gate = quality_v4.audit_quality_v4_full_export(export_path)
    gate_path = quality_v4.write_quality_v4_full_export_gate(
        output_root, full_export_gate
    )
    return {
        "full_export_path": export_path.relative_to(output_root).as_posix(),
        "full_export_gate_path": gate_path.relative_to(output_root).as_posix(),
        "full_export_gate": full_export_gate,
        "dataset_quality_v4_validation": quality_v4.dataset_quality_v4_validation(
            full_export_gate
        ),
    }


def recursive_output_checksums(
    root: Path,
    *,
    extra_entries: Sequence[tuple[str, str]] = (),
) -> str:
    """Return SHA256SUMS content covering every evaluation-owned artifact."""

    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    rows = [
        f"{_array_file_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in paths
    ]
    rows.extend(f"{sha256}  {label}\n" for sha256, label in extra_entries)
    return "".join(rows)


def _array_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
