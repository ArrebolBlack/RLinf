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
