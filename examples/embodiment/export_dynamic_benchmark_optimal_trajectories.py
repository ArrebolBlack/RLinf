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

"""Export audited best-known Dynamic Benchmark trajectories from a frozen pool.

``optimal`` in this entrypoint means best-known under the supplied immutable
candidate manifest, reset manifest, escalation budget, and lexicographic score.
It does not claim a globally optimal continuous-control solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.embodiment.evaluate_dynamic_benchmark_expert import (
    _device,
    _load_inference_policy,
    _manifest_row,
    _sha256,
    _validate_policy_payload,
)
from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    _compose_residual_actions,
    _planner_actions,
    _policy_action,
)

CANDIDATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-candidates-v0.1"
EXPORT_SCHEMA = "rlinf-dynamic-benchmark-optimal-export-v0.1"
ATTEMPT_SCHEMA = "rlinf-dynamic-benchmark-optimal-attempt-v0.1"
STATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-export-state-v0.1"
PROGRESS_SCHEMA = "rlinf-dynamic-benchmark-optimal-progress-v0.1"
RENDER_PARITY_SKIP_SCHEMA = "rlinf-dynamic-benchmark-render-parity-skip-v0.1"
SELECTION_CONTRACT = (
    "success,safety,trajectory_completion,return,-control_steps,-action_l2_sum"
)


@dataclass(frozen=True)
class CandidateSpec:
    """One immutable planner or policy candidate."""

    candidate_id: str
    kind: str
    policy_path: Path | None = None
    policy_sha256: str | None = None
    stochastic: bool = False
    exploration_seed_offset: int = 0
    residual_scale: float | None = None


@dataclass
class LoadedCandidate:
    """Candidate plus the reconstructed model and normalizer, when applicable."""

    spec: CandidateSpec
    index: int
    config: dict[str, Any] | None = None
    state_schema: dict[str, Any] | None = None
    model: Any | None = None
    normalizer: RunningNormalizer | None = None


def _candidate_identity(spec: CandidateSpec) -> dict[str, Any]:
    """Return a canonical-JSON-safe candidate identity."""

    return {
        "candidate_id": spec.candidate_id,
        "kind": spec.kind,
        "policy_path": None if spec.policy_path is None else str(spec.policy_path),
        "policy_sha256": spec.policy_sha256,
        "stochastic": spec.stochastic,
        "exploration_seed_offset": spec.exploration_seed_offset,
        "residual_scale": spec.residual_scale,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--manifest-seed", type=int, required=True)
    parser.add_argument("--accepted-episodes", type=int, default=100)
    parser.add_argument("--max-resets", type=int, default=200)
    parser.add_argument("--initial-k", type=int, default=8)
    parser.add_argument("--max-k", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted export at its last committed reset boundary",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def _full_commit(name: str, value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


def _expected_sha256(value: str, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return normalized


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("payload_sha256", None)
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rewrite_last_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Atomically replace the last committed row of a JSONL file."""

    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} has no rows to rewrite")
    lines[-1] = json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
    body = "\n".join(lines) + "\n"
    temporary = path.with_suffix(path.suffix + ".drop.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _quality_score(record: Mapping[str, Any]) -> tuple[float, ...]:
    """Return the frozen quality score, excluding deterministic identity tie-break."""

    return (
        float(bool(record["success"])),
        float(not bool(record["safety_failure"])),
        float(record["trajectory_completion"]),
        float(record["return"]),
        -float(record["control_steps"]),
        -float(record["action_l2_sum"]),
    )


def _eligible(record: Mapping[str, Any]) -> bool:
    replay = record.get("replay_validation")
    return bool(
        record.get("success")
        and not record.get("safety_failure")
        and record.get("finite_and_bounded")
        and isinstance(replay, Mapping)
        and replay.get("passed")
    )


def _select_winner(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select one eligible winner with a stable candidate-index tie-break."""

    eligible = [record for record in records if _eligible(record)]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (_quality_score(record), -int(record["candidate_index"])),
    )


def _render_parity_failure_reason(error: BaseException | str) -> str | None:
    message = str(error)
    if "parity failed" in message:
        return "render_parity_failed"
    if "canonical replay contract" in message:
        return "canonical_replay_contract_failed"
    return None


def _render_parity_skip(
    winner: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    """Bind a failed render replay to the independently selected light attempt."""

    reason = _render_parity_failure_reason(error)
    if reason is None:
        raise ValueError("render-parity skip requires a recognized replay failure")
    return {
        "schema_version": RENDER_PARITY_SKIP_SCHEMA,
        "reason": reason,
        "error_type": type(error).__name__,
        "error": str(error),
        "candidate_id": winner["candidate_id"],
        "candidate_index": int(winner["candidate_index"]),
        "attempt_tape": winner["attempt_tape"],
        "attempt_tape_sha256": winner["attempt_tape_sha256"],
        "action_sha256": winner["action_sha256"],
    }


def _budget_sequence(initial_k: int, max_k: int) -> tuple[int, ...]:
    if initial_k < 1 or max_k < initial_k:
        raise ValueError("candidate budgets require 1 <= initial_k <= max_k")
    values = []
    budget = initial_k
    while budget < max_k:
        values.append(budget)
        budget = min(max_k, budget * 2)
    values.append(max_k)
    return tuple(values)


def _validate_candidate_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    rlinf_commit: str,
    benchmark_commit: str,
    max_k: int,
) -> tuple[str, tuple[CandidateSpec, ...]]:
    """Validate and resolve the frozen candidate pool without loading policies."""

    if payload.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError("unsupported optimal-trajectory candidate schema")
    if payload.get("rlinf_commit") != rlinf_commit:
        raise ValueError("candidate manifest RLinf commit mismatch")
    if payload.get("benchmark_commit") != benchmark_commit:
        raise ValueError("candidate manifest benchmark commit mismatch")
    task = payload.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError("candidate manifest task identity is missing")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or len(rows) < max_k:
        raise ValueError(f"candidate manifest must contain at least max_k={max_k} candidates")
    allowed = {
        "candidate_id",
        "kind",
        "policy_path",
        "policy_sha256",
        "stochastic",
        "exploration_seed_offset",
        "residual_scale",
    }
    specs = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("candidate row is not a supported mapping")
        candidate_id = row.get("candidate_id")
        kind = row.get("kind")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if kind not in {"planner", "policy"}:
            raise ValueError(f"candidate {candidate_id!r} has unsupported kind")
        raw_stochastic = row.get("stochastic", False)
        if not isinstance(raw_stochastic, bool):
            raise ValueError("candidate stochastic must be boolean")
        stochastic = raw_stochastic
        raw_seed_offset = row.get("exploration_seed_offset", 0)
        if isinstance(raw_seed_offset, bool) or not isinstance(raw_seed_offset, int):
            raise ValueError("candidate exploration_seed_offset must be an integer")
        seed_offset = raw_seed_offset
        if not 0 <= seed_offset < 2**31:
            raise ValueError("candidate exploration_seed_offset must be in [0, 2**31)")
        residual_scale = row.get("residual_scale")
        if residual_scale is not None:
            residual_scale = float(residual_scale)
            if not 0.0 < residual_scale <= 1.0:
                raise ValueError("candidate residual_scale must be in (0, 1]")
        policy_path = None
        policy_sha256 = None
        if kind == "planner":
            if stochastic or seed_offset or residual_scale is not None:
                raise ValueError("planner candidate cannot declare policy exploration fields")
            if row.get("policy_path") is not None or row.get("policy_sha256") is not None:
                raise ValueError("planner candidate cannot declare a policy file")
        else:
            raw_path = row.get("policy_path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("policy candidate is missing policy_path")
            policy_path = Path(raw_path)
            if not policy_path.is_absolute():
                policy_path = (manifest_path.parent / policy_path).resolve()
            policy_sha256 = _expected_sha256(
                str(row.get("policy_sha256", "")),
                f"candidate {candidate_id} policy_sha256",
            )
        specs.append(
            CandidateSpec(
                candidate_id=candidate_id,
                kind=kind,
                policy_path=policy_path,
                policy_sha256=policy_sha256,
                stochastic=stochastic,
                exploration_seed_offset=seed_offset,
                residual_scale=residual_scale,
            )
        )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    if sum(spec.kind == "planner" for spec in specs) != 1:
        raise ValueError("candidate manifest must contain exactly one planner")
    if specs[0].kind != "planner":
        raise ValueError("the frozen candidate pool must put its planner at index zero")
    return task, tuple(specs)


def _load_candidates(
    specs: tuple[CandidateSpec, ...],
    *,
    task: str,
    rlinf_commit: str,
    benchmark_commit: str,
    device: torch.device,
) -> tuple[LoadedCandidate, ...]:
    loaded = []
    for index, spec in enumerate(specs):
        candidate = LoadedCandidate(spec=spec, index=index)
        if spec.kind == "policy":
            assert spec.policy_path is not None and spec.policy_sha256 is not None
            if not spec.policy_path.is_file():
                raise FileNotFoundError(spec.policy_path)
            if _sha256(spec.policy_path) != spec.policy_sha256:
                raise ValueError(f"candidate {spec.candidate_id!r} policy SHA-256 mismatch")
            payload = torch.load(spec.policy_path, map_location="cpu", weights_only=False)
            config, state_schema = _validate_policy_payload(
                payload,
                rlinf_commit=rlinf_commit,
                benchmark_commit=benchmark_commit,
            )
            if config["task"] != task:
                raise ValueError(f"candidate {spec.candidate_id!r} task mismatch")
            if spec.residual_scale is not None and config["algorithm"] != "residual_rlpd":
                raise ValueError("residual_scale override requires a residual-RLPD policy")
            state_dim = int(state_schema["state_dim"])
            model = _load_inference_policy(config, state_dim, payload["model"], device)
            normalizer = RunningNormalizer(state_dim, int(state_schema["mask_dim"]))
            normalizer.load_state_dict(payload["normalizer"])
            candidate.config = config
            candidate.state_schema = state_schema
            candidate.model = model
            candidate.normalizer = normalizer
        loaded.append(candidate)
    return tuple(loaded)


def _candidate_seed(episode_id: str, candidate: LoadedCandidate) -> int:
    material = (
        f"{episode_id}\0{candidate.spec.candidate_id}\0"
        f"{candidate.spec.exploration_seed_offset}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    from se3_wam.benchmark.contracts import canonical_json

    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(dict(payload)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _file_boundary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def _progress_payload(
    *,
    export_state_sha256: str,
    started_unix_s: float,
    next_reset_index: int,
    accepted_count: int,
    candidate_attempt_count: int,
    budget_histogram: Mapping[str, int],
    attempts_path: Path,
    reset_results_path: Path,
    winners_path: Path,
    resume_count: int,
    recovery_events: list[str],
) -> dict[str, Any]:
    payload = {
        "schema_version": PROGRESS_SCHEMA,
        "export_state_sha256": export_state_sha256,
        "started_unix_s": started_unix_s,
        "next_reset_index": next_reset_index,
        "accepted_count": accepted_count,
        "candidate_attempt_count": candidate_attempt_count,
        "budget_histogram": dict(budget_histogram),
        "resume_count": resume_count,
        "recovery_events": list(recovery_events),
        "file_boundaries": {
            "attempts.jsonl": _file_boundary(attempts_path),
            "reset_results.jsonl": _file_boundary(reset_results_path),
            "winner_manifest.jsonl": _file_boundary(winners_path),
        },
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def _recover_partial_output(
    *,
    output: Path,
    progress: Mapping[str, Any],
    reset_rows: list[Any],
    task: str,
    split: str,
) -> str | None:
    """Preserve and remove only data after the last committed reset boundary."""

    boundaries = progress.get("file_boundaries")
    if not isinstance(boundaries, dict):
        raise ValueError("resume progress is missing file boundaries")
    recovery = output.parent / f"{output.name}.recovery-{time.time_ns()}"
    recovery_created = False

    def ensure_recovery() -> None:
        nonlocal recovery_created
        if not recovery_created:
            recovery.mkdir(parents=False)
            recovery_created = True

    for name in ("attempts.jsonl", "reset_results.jsonl", "winner_manifest.jsonl"):
        path = output / name
        boundary = boundaries.get(name)
        if not isinstance(boundary, dict):
            raise ValueError(f"resume progress has no boundary for {name}")
        size = int(boundary.get("size", -1))
        expected = str(boundary.get("sha256", ""))
        data = path.read_bytes()
        if size < 0 or len(data) < size:
            raise ValueError(f"{name} is shorter than its committed boundary")
        prefix = data[:size]
        actual = hashlib.sha256(prefix).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} committed prefix checksum mismatch")
        if len(data) > size:
            ensure_recovery()
            shutil.copy2(path, recovery / name)
            temporary = path.with_suffix(path.suffix + ".resume.tmp")
            with temporary.open("wb") as stream:
                stream.write(prefix)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)

    next_reset_index = int(progress.get("next_reset_index", -1))
    if not 0 <= next_reset_index <= len(reset_rows):
        raise ValueError("resume progress reset index is outside the manifest")
    dirty_paths = [output / ".staging"]
    if next_reset_index < len(reset_rows):
        episode_id = reset_rows[next_reset_index].request.episode_id
        dirty_paths.extend(
            (
                output / "lightweight" / episode_id,
                output / "episodes" / task / split / episode_id,
            )
        )
    for path in dirty_paths:
        if not path.exists():
            continue
        if path.name == ".staging" and not any(path.iterdir()):
            path.rmdir()
            continue
        ensure_recovery()
        destination = recovery / path.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    return recovery.name if recovery_created else None


def _write_attempt_tape(
    output: Path,
    *,
    episode_id: str,
    candidate_index: int,
    arrays: Mapping[str, np.ndarray],
) -> tuple[str, str]:
    relative = Path("lightweight") / episode_id / f"candidate-{candidate_index:02d}.npz"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return relative.as_posix(), _sha256(path)


def _make_teacher(task: str, request: Any) -> Any:
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    teacher, _ = make_privileged_teacher(task, request=request)
    if hasattr(teacher, "reset"):
        teacher.reset()
    return teacher


class _ArmedResetReplayEnv:
    """Expose the canonical raw-env replay API while rearming hidden T5 events."""

    def __init__(self, vector_env: Any, raw_env: Any) -> None:
        self._vector_env = vector_env
        self._raw_env = raw_env

    def reset(self, request: Any) -> Any:
        observation = self._raw_env.reset(request)
        self._vector_env._arm_hidden_t5_event(self._raw_env, request)
        return observation

    def step(self, action: Any) -> Any:
        return self._raw_env.step(action)

    def save_state(self) -> bytes:
        return self._raw_env.save_state()


def _restore_candidate_start(env: Any, state: Mapping[str, Any]) -> None:
    """Restore wrapper bookkeeping, then rebuild the canonical request reset."""

    env.load_checkpoint_state(state)
    request = env._requests[0]
    if request is None:
        raise RuntimeError("candidate restore lost its reset request")
    raw_env = env.envs[0]
    observation = raw_env.reset(request)
    env._arm_hidden_t5_event(raw_env, request)
    env._raw_observations[0] = observation
    encoded = np.asarray(env._encode(observation, request), dtype=np.float32)
    env._last_obs = {"states": torch.as_tensor(encoded[None, :], dtype=torch.float32)}


def _rollout(
    *,
    env: Any,
    candidate: LoadedCandidate,
    device: torch.device,
    capture_trace: bool,
    trace_metadata: Mapping[str, Any] | None = None,
    replay_actions_array: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], Any | None]:
    from se3_wam.benchmark.api import StepResult
    from se3_wam.benchmark.dataset import EpisodeTrace
    from se3_wam.benchmark.evaluation import replay_actions
    from se3_wam.benchmark.metrics import (
        completion_time_from_events,
        hierarchical_task_completion,
        validate_stage_event_order,
    )

    request = env._requests[0]
    observation = env._raw_observations[0]
    obs = env._last_obs
    if request is None or observation is None or obs is None:
        raise RuntimeError("optimal-trajectory environment is not initialized")
    row = _manifest_row(env, request.episode_id)
    raw_env = env.envs[0]
    task = env._get_task_spec(request.task_id)
    teacher = None
    residual = False
    residual_scale = None
    if candidate.spec.kind == "planner":
        teacher = _make_teacher(request.task_id, request)
    else:
        assert candidate.config is not None
        assert candidate.model is not None and candidate.normalizer is not None
        residual = candidate.config["algorithm"] == "residual_rlpd"
        if residual:
            teacher = _make_teacher(request.task_id, request)
            residual_scale = (
                candidate.spec.residual_scale
                if candidate.spec.residual_scale is not None
                else float(candidate.config.get("residual_scale", 0.25))
            )
    seed = _candidate_seed(request.episode_id, candidate)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    observations = [observation]
    states = [np.asarray(obs["states"][0], dtype=np.float32)]
    actions = []
    policy_actions = []
    rewards = []
    terminated_rows = []
    truncated_rows = []
    outcomes = []
    step_results = []
    result_info: dict[str, Any] | None = None
    terminated_value = False
    truncated_value = False
    while not (terminated_value or truncated_value):
        if replay_actions_array is not None:
            action_index = len(actions)
            if action_index >= replay_actions_array.shape[0]:
                raise RuntimeError("replayed action sequence is shorter than the rollout")
            env_actions = torch.as_tensor(
                replay_actions_array[action_index], dtype=torch.float32
            ).unsqueeze(0)
            policy_action = env_actions.clone()
        elif candidate.spec.kind == "planner":
            env_actions = _planner_actions(env, [teacher])
            policy_action = env_actions.clone()
        else:
            with torch.inference_mode():
                policy_action, _ = _policy_action(
                    candidate.model,
                    candidate.normalizer,
                    obs["states"],
                    device,
                    stochastic=candidate.spec.stochastic,
                )
            policy_action = policy_action.cpu()
            env_actions = policy_action
            if residual:
                assert teacher is not None and residual_scale is not None
                env_actions = _compose_residual_actions(
                    _planner_actions(env, [teacher]),
                    policy_action,
                    residual_scale,
                )
        values = np.clip(np.asarray(env_actions[0], dtype=np.float64), -1.0, 1.0)
        action = env._ActionCommand(
            mode=request.action_mode,
            values=values,
            policy_step=observation.policy_step,
        )
        next_obs, reward, terminated, truncated, infos = env.step(
            env_actions,
            auto_reset=False,
        )
        next_observation = env._raw_observations[0]
        if next_observation is None:
            raise RuntimeError("optimal-trajectory rollout lost its raw observation")
        terminated_value = bool(terminated[0])
        truncated_value = bool(truncated[0])
        reason = infos["termination_reason"][0]
        active_progress = float(infos["reward_inputs"]["active_stage_progress"][0])
        observations.append(next_observation)
        states.append(np.asarray(next_obs["states"][0], dtype=np.float32))
        actions.append(action)
        policy_actions.append(np.asarray(policy_action[0], dtype=np.float32))
        rewards.append(float(reward[0]))
        terminated_rows.append(terminated_value)
        truncated_rows.append(truncated_value)
        outcomes.append(
            (
                terminated_value,
                truncated_value,
                bool(infos["success"][0]),
                reason,
                active_progress,
            )
        )
        step_results.append(
            StepResult(
                observation=next_observation,
                terminated=terminated_value,
                truncated=truncated_value,
                success=bool(infos["success"][0]),
                termination_reason=reason,
                active_stage_progress=active_progress,
            )
        )
        result_info = {
            "success": bool(infos["success"][0]),
            "termination_reason": reason,
            "active_stage_progress": active_progress,
        }
        observation = next_observation
        obs = next_obs
        if len(actions) > int(env.horizon_steps):
            raise RuntimeError("optimal-trajectory rollout exceeded the environment horizon")
    if result_info is None:
        raise RuntimeError("optimal-trajectory candidate produced no action")

    events = tuple(raw_env._ledger.events)
    final_state = raw_env.save_state()
    completed = validate_stage_event_order(task, events)
    completion = hierarchical_task_completion(
        task,
        completed,
        result_info["active_stage_progress"],
    )
    completion_time = (
        completion_time_from_events(
            events,
            start_event=task.task_start_event,
            success_event=task.success_stages[-1],
        )
        if result_info["success"]
        else None
    )
    replay_validation = replay_actions(
        _ArmedResetReplayEnv(env, raw_env),
        request=request,
        expected_observations=tuple(observations),
        actions=tuple(actions),
        expected_outcomes=tuple(outcomes),
        expected_final_state=final_state,
    )
    action_array = np.stack([action.values for action in actions]).astype(np.float64)
    policy_action_array = np.stack(policy_actions).astype(np.float32)
    state_array = np.stack(states).astype(np.float32)
    reward_array = np.asarray(rewards, dtype=np.float32)
    finite_and_bounded = bool(
        np.all(np.isfinite(state_array))
        and np.all(np.isfinite(action_array))
        and np.all(np.isfinite(policy_action_array))
        and np.all(np.isfinite(reward_array))
        and np.all(np.abs(action_array) <= 1.0)
        and np.all(np.abs(policy_action_array) <= 1.0)
    )
    safety_failures = set(env.reward_schema["safety_failures"])
    record = {
        "schema_version": ATTEMPT_SCHEMA,
        "episode_id": request.episode_id,
        "task_id": request.task_id,
        "seed": request.seed,
        "factors": dict(request.factors),
        "source_group_id": row.source_group_id,
        "pair_id": row.pair_id,
        "pair_member_id": row.pair_member_id,
        "candidate_manifest_index": row.candidate_index,
        "candidate_id": candidate.spec.candidate_id,
        "candidate_index": candidate.index,
        "candidate_kind": candidate.spec.kind,
        "stochastic": candidate.spec.stochastic,
        "exploration_seed": seed,
        "residual_scale": residual_scale,
        "success": result_info["success"],
        "safety_failure": result_info["termination_reason"] in safety_failures,
        "termination_reason": result_info["termination_reason"],
        "trajectory_completion": completion,
        "completion_time_s": completion_time,
        "return": float(reward_array.sum(dtype=np.float64)),
        "control_steps": len(actions),
        "action_l2_sum": float(np.square(action_array).sum()),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(action_array).tobytes()
        ).hexdigest(),
        "policy_action_sha256": hashlib.sha256(
            np.ascontiguousarray(policy_action_array).tobytes()
        ).hexdigest(),
        "state_sha256": hashlib.sha256(
            np.ascontiguousarray(state_array).tobytes()
        ).hexdigest(),
        "reward_sha256": hashlib.sha256(
            np.ascontiguousarray(reward_array).tobytes()
        ).hexdigest(),
        "finite_and_bounded": finite_and_bounded,
        "replay_validation": replay_validation,
        "replay_validation_sha256": _payload_sha256(replay_validation),
        "events": [event.name for event in events],
    }
    arrays = {
        "states": state_array,
        "policy_actions": policy_action_array,
        "actions": action_array,
        "rewards": reward_array,
        "terminated": np.asarray(terminated_rows, dtype=np.bool_),
        "truncated": np.asarray(truncated_rows, dtype=np.bool_),
    }
    trace = None
    if capture_trace:
        trace = EpisodeTrace(
            request=request,
            observations=tuple(observations),
            actions=tuple(actions),
            step_results=tuple(step_results),
            teacher_phases=tuple(
                f"best_known/{candidate.spec.candidate_id}" for _ in actions
            ),
            events=events,
            teacher_preparation={
                "method": "rlinf_best_known_candidate_selection",
                "candidate": _candidate_identity(candidate.spec),
                "candidate_index": candidate.index,
                "selection_contract": SELECTION_CONTRACT,
                **dict(trace_metadata or {}),
            },
            replay_validation=replay_validation,
            action_timing={
                "policy_rate_hz": 20.0,
                "control_steps": len(actions),
                "candidate_id": candidate.spec.candidate_id,
            },
        )
    return record, arrays, trace


def _make_env(
    *,
    task: str,
    split: str,
    manifest_seed: int,
    manifest_size: int,
    image_size: int,
    camera_observations: bool,
) -> Any:
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    return DynamicBenchmarkEnv(
        cfg={
            "task_id": task,
            "split": split,
            "manifest_seed": manifest_seed,
            "manifest_size": manifest_size,
            "image_size": image_size,
            "camera_observations": camera_observations,
            "auto_reset": False,
            "ignore_terminations": False,
            "group_size": 1,
        },
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def _root_checksums(root: Path) -> int:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and ".staging" not in path.parts
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths),
        encoding="utf-8",
    )
    return len(paths)


def main() -> None:
    from se3_wam.benchmark.contracts import canonical_json
    from se3_wam.benchmark.dataset import write_episode_atomic
    from se3_wam.benchmark.evaluation import manifest_record

    args = _parser().parse_args()
    if args.shard_count < 1 or args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    sharded = args.shard_count > 1
    shard_output = (
        args.output / f"shard-{args.shard_index:02d}" if sharded else args.output
    )
    if shard_output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.resume and not shard_output.is_dir():
        raise FileNotFoundError("--resume requires an existing export directory")
    if args.accepted_episodes < 1 or args.max_resets < args.accepted_episodes:
        raise ValueError("max_resets must be at least accepted_episodes > 0")
    if args.image_size < 64:
        raise ValueError("image_size must be at least 64")
    budgets = _budget_sequence(args.initial_k, args.max_k)
    evaluator_commit = _full_commit("evaluator_commit", args.evaluator_commit)
    rlinf_commit = _full_commit("rlinf_commit", args.rlinf_commit)
    benchmark_commit = _full_commit("benchmark_commit", args.benchmark_commit)
    candidate_manifest_sha256 = _expected_sha256(
        args.expected_candidate_manifest_sha256,
        "expected candidate manifest SHA-256",
    )
    if _sha256(args.candidate_manifest) != candidate_manifest_sha256:
        raise ValueError("candidate manifest SHA-256 mismatch")
    candidate_payload = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    task, specs = _validate_candidate_manifest(
        candidate_payload,
        manifest_path=args.candidate_manifest.resolve(),
        rlinf_commit=rlinf_commit,
        benchmark_commit=benchmark_commit,
        max_k=args.max_k,
    )
    device = _device(args.device)
    candidates = _load_candidates(
        specs,
        task=task,
        rlinf_commit=rlinf_commit,
        benchmark_commit=benchmark_commit,
        device=device,
    )
    reference_schemas = {
        canonical_json(candidate.state_schema)
        for candidate in candidates
        if candidate.state_schema is not None
    }
    if len(reference_schemas) > 1:
        raise ValueError("candidate policies disagree on state schema")

    manifest_size = args.max_resets + args.max_resets % 2
    light_env = _make_env(
        task=task,
        split=args.split,
        manifest_seed=args.manifest_seed,
        manifest_size=manifest_size,
        image_size=64,
        camera_observations=False,
    )
    render_env = _make_env(
        task=task,
        split=args.split,
        manifest_seed=args.manifest_seed,
        manifest_size=manifest_size,
        image_size=args.image_size,
        camera_observations=True,
    )
    try:
        light_manifest = [manifest_record(row) for row in light_env._manifest_rows]
        render_manifest = [manifest_record(row) for row in render_env._manifest_rows]
        if canonical_json(light_manifest) != canonical_json(render_manifest):
            raise RuntimeError("lightweight and render manifests disagree")
        if reference_schemas and canonical_json(light_env.state_schema) not in reference_schemas:
            raise ValueError("export environment state schema does not match policies")
        rows = list(light_env._manifest_rows[: args.max_resets])
        if sharded:
            step = (len(rows) + args.shard_count - 1) // args.shard_count
            start = args.shard_index * step
            shard_rows = rows[start : start + step]
            run_output = shard_output
            for _ in range(start):
                light_env.reset(options={"env_idx": [0]})
                render_env.reset(options={"env_idx": [0]})
        else:
            start = 0
            shard_rows = rows
            run_output = args.output
        reset_manifest_text = "".join(
            canonical_json(manifest_record(row)) + "\n" for row in rows
        )
        reset_manifest_sha256 = hashlib.sha256(
            reset_manifest_text.encode("utf-8")
        ).hexdigest()
        source_identity = {
            "evaluator_rlinf_commit": evaluator_commit,
            "policy_rlinf_commit": rlinf_commit,
            "benchmark_commit": benchmark_commit,
        }
        export_state = {
            "schema_version": STATE_SCHEMA,
            "task": task,
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "manifest_size": manifest_size,
            "max_resets": args.max_resets,
            "accepted_target": args.accepted_episodes,
            "initial_k": args.initial_k,
            "max_k": args.max_k,
            "budget_sequence": list(budgets),
            "image_size": args.image_size,
            "device": str(device),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "reset_manifest_sha256": reset_manifest_sha256,
            "source_identity": source_identity,
            "state_schema": json.loads(json.dumps(light_env.state_schema, allow_nan=False)),
            "candidates": [_candidate_identity(candidate.spec) for candidate in candidates],
        }
        export_state["payload_sha256"] = _payload_sha256(export_state)
        attempts_path = run_output / "attempts.jsonl"
        winners_path = run_output / "winner_manifest.jsonl"
        reset_results_path = run_output / "reset_results.jsonl"
        reset_manifest_path = run_output / "reset_manifest.jsonl"
        export_state_path = run_output / "export_state.json"
        progress_path = run_output / "progress.json"
        if args.resume:
            if (run_output / "dataset_card.json").exists() or (
                run_output / "SHA256SUMS"
            ).exists():
                raise ValueError("refusing to resume a sealed export")
            if _sha256(run_output / "candidate_manifest.json") != candidate_manifest_sha256:
                raise ValueError("resume candidate-manifest copy checksum mismatch")
            if reset_manifest_path.read_text(encoding="utf-8") != reset_manifest_text:
                raise ValueError("resume reset manifest does not match the requested run")
            stored_state = json.loads(export_state_path.read_text(encoding="utf-8"))
            if _payload_sha256(stored_state) != stored_state.get("payload_sha256"):
                raise ValueError("resume export-state payload checksum mismatch")
            if stored_state != export_state:
                raise ValueError("resume arguments or resolved candidate identities changed")
            export_state_sha256 = _sha256(export_state_path)
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("schema_version") != PROGRESS_SCHEMA or (
                _payload_sha256(progress) != progress.get("payload_sha256")
            ):
                raise ValueError("resume progress schema or payload checksum mismatch")
            if progress.get("export_state_sha256") != export_state_sha256:
                raise ValueError("resume progress references a different export state")
            recovery_event = _recover_partial_output(
                output=run_output,
                progress=progress,
                reset_rows=shard_rows,
                task=task,
                split=args.split,
            )
            started = float(progress["started_unix_s"])
            accepted = int(progress["accepted_count"])
            attempted_resets = int(progress["next_reset_index"])
            attempt_count = int(progress["candidate_attempt_count"])
            budget_histogram = {
                str(key): int(value) for key, value in progress["budget_histogram"].items()
            }
            if set(budget_histogram) != {str(budget) for budget in budgets}:
                raise ValueError("resume budget histogram keys changed")
            resume_count = int(progress["resume_count"]) + 1
            recovery_events = list(progress["recovery_events"])
            if recovery_event is not None:
                recovery_events.append(recovery_event)
            expected_line_counts = {
                attempts_path: attempt_count,
                reset_results_path: attempted_resets,
                winners_path: accepted,
            }
            for path, expected_count in expected_line_counts.items():
                with path.open("r", encoding="utf-8") as stream:
                    actual_count = sum(1 for line in stream if line.strip())
                if actual_count != expected_count:
                    raise ValueError(f"resume committed line count mismatch for {path.name}")
            _atomic_json(
                progress_path,
                _progress_payload(
                    export_state_sha256=export_state_sha256,
                    started_unix_s=started,
                    next_reset_index=attempted_resets,
                    accepted_count=accepted,
                    candidate_attempt_count=attempt_count,
                    budget_histogram=budget_histogram,
                    attempts_path=attempts_path,
                    reset_results_path=reset_results_path,
                    winners_path=winners_path,
                    resume_count=resume_count,
                    recovery_events=recovery_events,
                ),
            )
        else:
            run_output.mkdir(parents=True)
            shutil.copyfile(args.candidate_manifest, run_output / "candidate_manifest.json")
            reset_manifest_path.write_text(reset_manifest_text, encoding="utf-8")
            for path in (attempts_path, winners_path, reset_results_path):
                path.write_text("", encoding="utf-8")
            _atomic_json(export_state_path, export_state)
            export_state_sha256 = _sha256(export_state_path)
            started = time.time()
            accepted = 0
            attempted_resets = 0
            attempt_count = 0
            budget_histogram = {str(budget): 0 for budget in budgets}
            resume_count = 0
            recovery_events: list[str] = []
            _atomic_json(
                progress_path,
                _progress_payload(
                    export_state_sha256=export_state_sha256,
                    started_unix_s=started,
                    next_reset_index=0,
                    accepted_count=0,
                    candidate_attempt_count=0,
                    budget_histogram=budget_histogram,
                    attempts_path=attempts_path,
                    reset_results_path=reset_results_path,
                    winners_path=winners_path,
                    resume_count=0,
                    recovery_events=[],
                ),
            )
        for local_index, row in enumerate(shard_rows):
            reset_index = start + local_index
            if accepted >= args.accepted_episodes:
                break
            if local_index < attempted_resets:
                if local_index + 1 < len(shard_rows):
                    light_env.reset(options={"env_idx": [0]})
                    render_env.reset(options={"env_idx": [0]})
                continue
            light_request = light_env._requests[0]
            render_request = render_env._requests[0]
            if (
                light_request is None
                or render_request is None
                or light_request.episode_id != row.request.episode_id
                or render_request.episode_id != row.request.episode_id
            ):
                raise RuntimeError("rollout order diverged from the frozen reset manifest")
            initial_state = light_env.checkpoint_state()
            reset_attempts: list[dict[str, Any]] = []
            winner = None
            budget_used = budgets[-1]
            for budget in budgets:
                for candidate in candidates[len(reset_attempts) : budget]:
                    _restore_candidate_start(light_env, initial_state)
                    record, arrays, _ = _rollout(
                        env=light_env,
                        candidate=candidate,
                        device=device,
                        capture_trace=False,
                    )
                    relative, tape_sha256 = _write_attempt_tape(
                        run_output,
                        episode_id=record["episode_id"],
                        candidate_index=candidate.index,
                        arrays=arrays,
                    )
                    record["attempt_tape"] = relative
                    record["attempt_tape_sha256"] = tape_sha256
                    record["quality_score"] = list(_quality_score(record))
                    record["eligible"] = _eligible(record)
                    _append_jsonl(attempts_path, record)
                    reset_attempts.append(record)
                    attempt_count += 1
                winner = _select_winner(reset_attempts)
                budget_used = budget
                if winner is not None:
                    break
            attempted_resets += 1
            budget_histogram[str(budget_used)] += 1
            reset_result = {
                "reset_index": reset_index,
                "episode_id": row.request.episode_id,
                "source_group_id": row.source_group_id,
                "candidate_count": len(reset_attempts),
                "budget_used": budget_used,
                "accepted": winner is not None,
                "winner_candidate_id": None if winner is None else winner["candidate_id"],
                "winner_candidate_index": None if winner is None else winner["candidate_index"],
            }
            _append_jsonl(reset_results_path, reset_result)
            if winner is not None:
                candidate = candidates[int(winner["candidate_index"])]
                tape_path = run_output / winner["attempt_tape"]
                replay_actions_array = np.load(tape_path)["actions"]
                try:
                    render_record, _, trace = _rollout(
                        env=render_env,
                        candidate=candidate,
                        device=device,
                        capture_trace=True,
                        replay_actions_array=replay_actions_array,
                        trace_metadata={
                            "candidate_manifest_sha256": candidate_manifest_sha256,
                            "budget_used": budget_used,
                            "winner_quality_score": list(_quality_score(winner)),
                            "lightweight_action_sha256": winner["action_sha256"],
                            "source_identity": source_identity,
                        },
                    )
                    if trace is None:
                        raise RuntimeError("winner render did not return an episode trace")
                    for key in (
                        "episode_id",
                        "success",
                        "safety_failure",
                        "termination_reason",
                        "trajectory_completion",
                        "completion_time_s",
                        "return",
                        "control_steps",
                        "action_l2_sum",
                        "action_sha256",
                    ):
                        if render_record[key] != winner[key]:
                            raise RuntimeError(f"winner render parity failed for {key}")
                    episode_record = write_episode_atomic(run_output, trace)
                    winner_row = {
                        **episode_record,
                        "candidate_id": candidate.spec.candidate_id,
                        "candidate_index": candidate.index,
                        "candidate_count": len(reset_attempts),
                        "budget_used": budget_used,
                        "selection_contract": SELECTION_CONTRACT,
                        "quality_score": list(_quality_score(winner)),
                        "lightweight_attempt_tape": winner["attempt_tape"],
                        "lightweight_attempt_tape_sha256": winner["attempt_tape_sha256"],
                    }
                    _append_jsonl(winners_path, winner_row)
                    accepted += 1
                    print(
                        json.dumps(
                            {
                                "accepted": accepted,
                                "episode_id": winner["episode_id"],
                                "candidate_id": winner["candidate_id"],
                                "budget_used": budget_used,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except (RuntimeError, ValueError) as exc:
                    if _render_parity_failure_reason(exc) is None:
                        raise
                    render_parity_skip = _render_parity_skip(winner, exc)
                    _rewrite_last_jsonl(
                        reset_results_path,
                        {
                            **reset_result,
                            "accepted": False,
                            "winner_candidate_id": None,
                            "winner_candidate_index": None,
                            "render_parity_skip": render_parity_skip,
                        },
                    )
                    winner = None
                    recovery_events.append(
                        f"render_parity_skip:reset:{reset_index}:"
                        f"{row.request.episode_id}:{str(exc)}"
                    )
            _atomic_json(
                progress_path,
                _progress_payload(
                    export_state_sha256=export_state_sha256,
                    started_unix_s=started,
                    next_reset_index=local_index + 1,
                    accepted_count=accepted,
                    candidate_attempt_count=attempt_count,
                    budget_histogram=budget_histogram,
                    attempts_path=attempts_path,
                    reset_results_path=reset_results_path,
                    winners_path=winners_path,
                    resume_count=resume_count,
                    recovery_events=recovery_events,
                ),
            )
            if local_index + 1 < len(shard_rows):
                light_env.reset(options={"env_idx": [0]})
                render_env.reset(options={"env_idx": [0]})

        if _sha256(args.candidate_manifest) != candidate_manifest_sha256:
            raise RuntimeError("candidate manifest changed during export")
        for candidate in candidates:
            if candidate.spec.kind != "policy":
                continue
            assert candidate.spec.policy_path is not None
            assert candidate.spec.policy_sha256 is not None
            if _sha256(candidate.spec.policy_path) != candidate.spec.policy_sha256:
                raise RuntimeError(
                    f"candidate policy changed during export: {candidate.spec.candidate_id}"
                )
        if sharded:
            _atomic_json(
                run_output / "shard_complete.json",
                {
                    "schema_version": "rlinf-dynamic-benchmark-optimal-shard-v0.1",
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "accepted_count": accepted,
                    "attempted_reset_count": attempted_resets,
                    "candidate_attempt_count": attempt_count,
                    "budget_histogram": dict(budget_histogram),
                },
            )
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "accepted": accepted,
                        "attempted_resets": attempted_resets,
                        "candidate_attempts": attempt_count,
                        "budget_histogram": budget_histogram,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        status = "complete" if accepted == args.accepted_episodes else "incomplete"
        card = {
            "schema_version": EXPORT_SCHEMA,
            "status": status,
            "training_eligible": False,
            "training_eligibility_reason": "independent audit has not yet passed",
            "optimality_claim": "best-known under the frozen candidate/reset/budget contract",
            "task": task,
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "accepted_target": args.accepted_episodes,
            "accepted_count": accepted,
            "attempted_reset_count": attempted_resets,
            "candidate_attempt_count": attempt_count,
            "initial_k": args.initial_k,
            "max_k": args.max_k,
            "budget_sequence": list(budgets),
            "budget_histogram": budget_histogram,
            "selection_contract": SELECTION_CONTRACT,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "reset_manifest_sha256": _sha256(reset_manifest_path),
            "export_state_sha256": export_state_sha256,
            "progress_sha256": _sha256(progress_path),
            "resume_count": resume_count,
            "recovery_events": recovery_events,
            "source_identity": source_identity,
            "image_size": args.image_size,
            "device": str(device),
            "started_unix_s": started,
            "finished_unix_s": time.time(),
        }
        card["payload_sha256"] = _payload_sha256(card)
        _atomic_json(args.output / "dataset_card.json", card)
        checksum_count = _root_checksums(args.output)
        print(
            json.dumps(
                {
                    "status": status,
                    "accepted": accepted,
                    "attempted_resets": attempted_resets,
                    "candidate_attempts": attempt_count,
                    "checksum_entries": checksum_count,
                    "dataset_card_payload_sha256": card["payload_sha256"],
                },
                sort_keys=True,
            )
        )
        if status != "complete":
            raise RuntimeError(
                f"accepted {accepted}/{args.accepted_episodes} winners within {attempted_resets} resets"
            )
    finally:
        light_env.close()
        render_env.close()


if __name__ == "__main__":
    main()
