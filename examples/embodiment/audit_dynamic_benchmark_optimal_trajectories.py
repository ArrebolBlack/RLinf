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

"""Independently audit a best-known Dynamic Benchmark trajectory export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
import numpy as np

CANDIDATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-candidates-v0.1"
EXPORT_SCHEMA = "rlinf-dynamic-benchmark-optimal-export-v0.1"
ATTEMPT_SCHEMA = "rlinf-dynamic-benchmark-optimal-attempt-v0.1"
AUDIT_SCHEMA = "rlinf-dynamic-benchmark-optimal-audit-v0.1"
STATE_SCHEMA = "rlinf-dynamic-benchmark-optimal-export-state-v0.1"
PROGRESS_SCHEMA = "rlinf-dynamic-benchmark-optimal-progress-v0.1"
RENDER_PARITY_SKIP_SCHEMA = "rlinf-dynamic-benchmark-render-parity-skip-v0.1"
SELECTION_CONTRACT = (
    "success,safety,trajectory_completion,return,-control_steps,-action_l2_sum"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-dataset-card-sha256", required=True)
    parser.add_argument("--expected-checksums-sha256", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--auditor-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha256(value: str, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return normalized


def _full_commit(name: str, value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} is not a mapping")
            rows.append(value)
    return rows


def _safe_dataset_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe dataset-relative path {relative!r}")
    target = root.joinpath(*pure.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"dataset-relative path escapes the root: {relative!r}")
    return target


def _verify_root_checksums(root: Path, expected_sha256: str) -> int:
    checksum_path = root / "SHA256SUMS"
    if _sha256(checksum_path) != expected_sha256:
        raise ValueError("root SHA256SUMS identity mismatch")
    declared: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"SHA256SUMS:{line_number} is malformed")
        digest = _expected_sha256(parts[0], f"SHA256SUMS:{line_number} digest")
        relative = parts[1]
        if relative in declared:
            raise ValueError(f"SHA256SUMS repeats {relative!r}")
        path = _safe_dataset_path(root, relative)
        if not path.is_file() or path.name == "SHA256SUMS":
            raise ValueError(f"SHA256SUMS target is missing or forbidden: {relative!r}")
        if _sha256(path) != digest:
            raise ValueError(f"dataset file checksum mismatch: {relative!r}")
        declared[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and ".staging" not in path.parts
    }
    if set(declared) != actual:
        missing = sorted(actual - set(declared))
        extra = sorted(set(declared) - actual)
        raise ValueError(f"root checksum inventory mismatch: missing={missing}, extra={extra}")
    return len(declared)


def _candidate_rows(
    payload: Mapping[str, Any],
    *,
    card: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if payload.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError("candidate manifest schema mismatch")
    if payload.get("task") != card.get("task"):
        raise ValueError("candidate manifest task mismatch")
    source = card.get("source_identity")
    if not isinstance(source, dict):
        raise ValueError("dataset card source identity is missing")
    if payload.get("rlinf_commit") != source.get("policy_rlinf_commit"):
        raise ValueError("candidate manifest RLinf source mismatch")
    if payload.get("benchmark_commit") != source.get("benchmark_commit"):
        raise ValueError("candidate manifest benchmark source mismatch")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or len(rows) < int(card["max_k"]):
        raise ValueError("candidate manifest is shorter than max_k")
    ids = []
    planner_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"candidate {index} is not a mapping")
        candidate_id = row.get("candidate_id")
        kind = row.get("kind")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"candidate {index} has no stable ID")
        if kind not in {"planner", "policy"}:
            raise ValueError(f"candidate {candidate_id!r} has invalid kind")
        if not isinstance(row.get("stochastic", False), bool):
            raise ValueError(f"candidate {candidate_id!r} stochastic flag is not boolean")
        seed_offset = row.get("exploration_seed_offset", 0)
        if (
            isinstance(seed_offset, bool)
            or not isinstance(seed_offset, int)
            or not 0 <= seed_offset < 2**31
        ):
            raise ValueError(f"candidate {candidate_id!r} exploration seed is invalid")
        planner_count += int(kind == "planner")
        if kind == "planner":
            if row.get("policy_path") is not None or row.get("policy_sha256") is not None:
                raise ValueError("planner candidate unexpectedly names a policy file")
        else:
            if not isinstance(row.get("policy_path"), str) or not row["policy_path"]:
                raise ValueError(f"policy candidate {candidate_id!r} has no path")
            _expected_sha256(
                str(row.get("policy_sha256", "")),
                f"candidate {candidate_id} policy SHA-256",
            )
        ids.append(candidate_id)
    if len(ids) != len(set(ids)) or planner_count != 1:
        raise ValueError("candidate IDs must be unique and contain exactly one planner")
    if rows[0].get("kind") != "planner":
        raise ValueError("the frozen candidate pool must put its planner at index zero")
    return rows


def _quality_score(record: Mapping[str, Any]) -> tuple[float, ...]:
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


def _selected(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [record for record in records if _eligible(record)]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (_quality_score(record), -int(record["candidate_index"])),
    )


def _render_parity_failure_reason(error: str) -> str | None:
    if "parity failed" in error:
        return "render_parity_failed"
    if "canonical replay contract" in error:
        return "canonical_replay_contract_failed"
    return None


def _render_parity_skip_events(recovery_events: Any) -> dict[int, dict[str, str]]:
    """Parse the frozen v0.1 string event used by already exported shards."""

    if not isinstance(recovery_events, list) or not all(
        isinstance(event, str) for event in recovery_events
    ):
        raise ValueError("recovery events must be a list of strings")
    skips: dict[int, dict[str, str]] = {}
    for event in recovery_events:
        if not event.startswith("render_parity_skip:"):
            continue
        parts = event.split(":", maxsplit=4)
        if len(parts) != 5 or parts[:2] != ["render_parity_skip", "reset"]:
            raise ValueError("render-parity recovery event is malformed")
        try:
            reset_index = int(parts[2])
        except ValueError as error:
            raise ValueError("render-parity recovery reset index is malformed") from error
        episode_id, message = parts[3], parts[4]
        reason = _render_parity_failure_reason(message)
        if reset_index < 0 or not episode_id or reason is None:
            raise ValueError("render-parity recovery event is invalid")
        if reset_index in skips:
            raise ValueError("multiple render-parity recovery events name one reset")
        skips[reset_index] = {
            "episode_id": episode_id,
            "error": message,
            "reason": reason,
        }
    return skips


def _audit_render_parity_skip(
    result: Mapping[str, Any],
    selected: Mapping[str, Any],
    event: Mapping[str, str] | None,
) -> str:
    """Validate a rejected publication whose light winner failed render replay."""

    if event is None:
        raise ValueError("render-parity skip has no matching recovery event")
    if event.get("episode_id") != result.get("episode_id"):
        raise ValueError("render-parity recovery event episode identity mismatch")
    skip = result.get("render_parity_skip")
    if skip is None:
        return "legacy-v0.1"
    if not isinstance(skip, dict):
        raise ValueError("render-parity skip evidence is not a mapping")
    expected_keys = {
        "schema_version",
        "reason",
        "error_type",
        "error",
        "candidate_id",
        "candidate_index",
        "attempt_tape",
        "attempt_tape_sha256",
        "action_sha256",
    }
    if set(skip) != expected_keys:
        raise ValueError("render-parity skip evidence inventory mismatch")
    if skip.get("schema_version") != RENDER_PARITY_SKIP_SCHEMA:
        raise ValueError("render-parity skip evidence schema mismatch")
    if skip.get("error_type") not in {"RuntimeError", "ValueError"}:
        raise ValueError("render-parity skip error type is invalid")
    message = skip.get("error")
    if not isinstance(message, str):
        raise ValueError("render-parity skip error is missing")
    reason = _render_parity_failure_reason(message)
    if reason is None or skip.get("reason") != reason:
        raise ValueError("render-parity skip reason does not recompute")
    if event.get("error") != message or event.get("reason") != reason:
        raise ValueError("render-parity skip disagrees with its recovery event")
    selected_values = {
        "candidate_id": selected["candidate_id"],
        "candidate_index": int(selected["candidate_index"]),
        "attempt_tape": selected["attempt_tape"],
        "attempt_tape_sha256": selected["attempt_tape_sha256"],
        "action_sha256": selected["action_sha256"],
    }
    if any(skip.get(key) != value for key, value in selected_values.items()):
        raise ValueError("render-parity skip is not bound to the selected attempt")
    return "structured-v0.1"


def _audit_attempt_tape(
    root: Path,
    record: Mapping[str, Any],
    *,
    expected_task: str,
) -> None:
    if record.get("schema_version") != ATTEMPT_SCHEMA:
        raise ValueError("attempt schema mismatch")
    if record.get("task_id") != expected_task:
        raise ValueError("attempt task mismatch")
    relative = record.get("attempt_tape")
    if not isinstance(relative, str):
        raise ValueError("attempt tape path is missing")
    path = _safe_dataset_path(root, relative)
    if _sha256(path) != record.get("attempt_tape_sha256"):
        raise ValueError("attempt tape checksum mismatch")
    with np.load(path, allow_pickle=False) as tape:
        if set(tape.files) != {
            "states",
            "policy_actions",
            "actions",
            "rewards",
            "terminated",
            "truncated",
        }:
            raise ValueError("attempt tape array inventory mismatch")
        states = np.asarray(tape["states"])
        policy_actions = np.asarray(tape["policy_actions"])
        actions = np.asarray(tape["actions"])
        rewards = np.asarray(tape["rewards"])
        terminated = np.asarray(tape["terminated"])
        truncated = np.asarray(tape["truncated"])
    steps = int(record["control_steps"])
    if (
        steps < 1
        or states.ndim != 2
        or states.shape[0] != steps + 1
        or actions.shape != (steps, 7)
        or policy_actions.shape != (steps, 7)
        or rewards.shape != (steps,)
        or terminated.shape != (steps,)
        or truncated.shape != (steps,)
    ):
        raise ValueError("attempt tape array shapes do not align")
    if terminated.dtype != np.bool_ or truncated.dtype != np.bool_:
        raise ValueError("attempt termination tapes must be boolean")
    if np.any(terminated[:-1]) or np.any(truncated[:-1]):
        raise ValueError("attempt has a pre-terminal done flag")
    if bool(terminated[-1]) == bool(truncated[-1]):
        raise ValueError("attempt must terminate or truncate exactly once at its final step")
    finite = bool(
        np.all(np.isfinite(states))
        and np.all(np.isfinite(policy_actions))
        and np.all(np.isfinite(actions))
        and np.all(np.isfinite(rewards))
        and np.all(np.abs(policy_actions) <= 1.0)
        and np.all(np.abs(actions) <= 1.0)
    )
    if finite != bool(record["finite_and_bounded"]):
        raise ValueError("attempt finite/bounded status does not recompute")
    hashes = {
        "state_sha256": hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest(),
        "policy_action_sha256": hashlib.sha256(
            np.ascontiguousarray(policy_actions).tobytes()
        ).hexdigest(),
        "action_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        "reward_sha256": hashlib.sha256(np.ascontiguousarray(rewards).tobytes()).hexdigest(),
    }
    if any(record.get(key) != value for key, value in hashes.items()):
        raise ValueError("attempt array content checksum does not recompute")
    if not math.isclose(
        float(record["return"]),
        float(rewards.sum(dtype=np.float64)),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("attempt return does not recompute")
    if not math.isclose(
        float(record["action_l2_sum"]),
        float(np.square(actions).sum(dtype=np.float64)),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("attempt action norm does not recompute")
    replay = record.get("replay_validation")
    if not isinstance(replay, dict):
        raise ValueError("attempt replay validation is missing")
    if _payload_sha256(replay) != record.get("replay_validation_sha256"):
        raise ValueError("attempt replay-validation checksum does not recompute")
    if list(_quality_score(record)) != record.get("quality_score"):
        raise ValueError("attempt quality score does not recompute")
    if _eligible(record) != bool(record.get("eligible")):
        raise ValueError("attempt eligibility does not recompute")


def _audit_winner_episode(
    root: Path,
    winner: Mapping[str, Any],
    attempt: Mapping[str, Any],
    reset: Mapping[str, Any],
    card: Mapping[str, Any],
) -> None:
    from se3_wam.benchmark.dataset import audit_episode

    relative = winner.get("relative_episode_dir")
    if not isinstance(relative, str):
        raise ValueError("winner episode directory is missing")
    episode_dir = _safe_dataset_path(root, relative)
    audit = audit_episode(episode_dir)
    if audit.get("episode_id") != attempt["episode_id"] or not audit.get("success"):
        raise ValueError("winner episode audit identity or success mismatch")
    if not audit.get("eligible_for_behavior_cloning"):
        raise ValueError("winner episode is not behavior-cloning eligible")
    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    if metadata != {key: winner[key] for key in metadata}:
        raise ValueError("winner manifest does not reproduce the episode record")
    request = metadata.get("request")
    if not isinstance(request, dict):
        raise ValueError("winner episode reset request is missing")
    for key in (
        "episode_id",
        "task_id",
        "split",
        "seed",
        "action_mode",
        "observation_track",
        "object_mode",
        "reset_mode",
        "factors",
    ):
        if request.get(key) != reset.get(key):
            raise ValueError(f"winner episode reset identity mismatch for {key}")
    teacher = metadata.get("teacher_preparation")
    if not isinstance(teacher, dict) or teacher.get("method") != (
        "rlinf_best_known_candidate_selection"
    ):
        raise ValueError("winner episode selection provenance is missing")
    candidate = teacher.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("winner candidate provenance is missing")
    if (
        candidate.get("candidate_id") != attempt["candidate_id"]
        or int(teacher.get("candidate_index", -1)) != int(attempt["candidate_index"])
        or int(teacher.get("budget_used", -1)) != int(winner["budget_used"])
        or teacher.get("candidate_manifest_sha256") != card["candidate_manifest_sha256"]
        or teacher.get("selection_contract") != SELECTION_CONTRACT
        or teacher.get("winner_quality_score") != list(_quality_score(attempt))
        or teacher.get("lightweight_action_sha256") != attempt["action_sha256"]
        or teacher.get("source_identity") != card["source_identity"]
    ):
        raise ValueError("winner episode selection provenance does not recompute")
    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        action_values = np.asarray(handle["actions/values"])
    rendered_action_sha256 = hashlib.sha256(
        np.ascontiguousarray(action_values).tobytes()
    ).hexdigest()
    if rendered_action_sha256 != attempt["action_sha256"]:
        raise ValueError("winner RGB-D trajectory differs from its selected lightweight action tape")


def _audit_export_state_and_progress(
    root: Path,
    *,
    card: Mapping[str, Any],
    candidate_rows: list[dict[str, Any]],
    reset_rows: list[dict[str, Any]],
) -> None:
    state_path = root / "export_state.json"
    progress_path = root / "progress.json"
    if _sha256(state_path) != card.get("export_state_sha256"):
        raise ValueError("export-state file identity mismatch")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA or (
        _payload_sha256(state) != state.get("payload_sha256")
    ):
        raise ValueError("export-state schema or payload checksum mismatch")
    expected_state_values = {
        "task": card["task"],
        "split": card["split"],
        "manifest_seed": card["manifest_seed"],
        "accepted_target": card["accepted_target"],
        "initial_k": card["initial_k"],
        "max_k": card["max_k"],
        "budget_sequence": card["budget_sequence"],
        "image_size": card["image_size"],
        "device": card["device"],
        "candidate_manifest_sha256": card["candidate_manifest_sha256"],
        "reset_manifest_sha256": card["reset_manifest_sha256"],
        "source_identity": card["source_identity"],
    }
    if any(state.get(key) != value for key, value in expected_state_values.items()):
        raise ValueError("export-state identity does not match the dataset card")
    if int(state.get("max_resets", -1)) != len(reset_rows):
        raise ValueError("export-state max_resets does not match reset manifest")
    state_candidates = state.get("candidates")
    if not isinstance(state_candidates, list) or len(state_candidates) != len(candidate_rows):
        raise ValueError("export-state candidate inventory mismatch")
    for index, (state_row, manifest_row) in enumerate(
        zip(state_candidates, candidate_rows, strict=True)
    ):
        if not isinstance(state_row, dict):
            raise ValueError(f"export-state candidate {index} is not a mapping")
        for key in (
            "candidate_id",
            "kind",
            "policy_sha256",
            "stochastic",
            "exploration_seed_offset",
            "residual_scale",
        ):
            default = False if key == "stochastic" else 0 if key == "exploration_seed_offset" else None
            if state_row.get(key) != manifest_row.get(key, default):
                raise ValueError(f"export-state candidate {index} differs for {key}")
        policy_path = state_row.get("policy_path")
        if manifest_row["kind"] == "policy" and not isinstance(policy_path, str):
            raise ValueError(f"export-state candidate {index} has no resolved policy path")
        if manifest_row["kind"] == "planner" and policy_path is not None:
            raise ValueError("export-state planner unexpectedly has a policy path")

    if _sha256(progress_path) != card.get("progress_sha256"):
        raise ValueError("progress file identity mismatch")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("schema_version") != PROGRESS_SCHEMA or (
        _payload_sha256(progress) != progress.get("payload_sha256")
    ):
        raise ValueError("progress schema or payload checksum mismatch")
    if progress.get("export_state_sha256") != card.get("export_state_sha256"):
        raise ValueError("progress references a different export state")
    progress_values = {
        "started_unix_s": card["started_unix_s"],
        "next_reset_index": card["attempted_reset_count"],
        "accepted_count": card["accepted_count"],
        "candidate_attempt_count": card["candidate_attempt_count"],
        "budget_histogram": card["budget_histogram"],
        "resume_count": card["resume_count"],
        "recovery_events": card["recovery_events"],
    }
    if any(progress.get(key) != value for key, value in progress_values.items()):
        raise ValueError("progress counters do not match the dataset card")
    boundaries = progress.get("file_boundaries")
    if not isinstance(boundaries, dict):
        raise ValueError("progress file boundaries are missing")
    for name in ("attempts.jsonl", "reset_results.jsonl", "winner_manifest.jsonl"):
        path = root / name
        boundary = boundaries.get(name)
        if not isinstance(boundary, dict) or boundary != {
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }:
            raise ValueError(f"progress boundary does not match final {name}")


def _audit_dataset(
    *,
    root: Path,
    expected_card_sha256: str,
    expected_checksums_sha256: str,
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    card_path = root / "dataset_card.json"
    if _sha256(card_path) != expected_card_sha256:
        raise ValueError("dataset-card file identity mismatch")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("schema_version") != EXPORT_SCHEMA:
        raise ValueError("dataset-card schema mismatch")
    if _payload_sha256(card) != card.get("payload_sha256"):
        raise ValueError("dataset-card payload checksum mismatch")
    if card.get("status") != "complete" or card.get("training_eligible") is not False:
        raise ValueError("only complete, unaudited exports can enter independent audit")
    checksum_entries = _verify_root_checksums(root, expected_checksums_sha256)
    candidate_path = root / "candidate_manifest.json"
    if _sha256(candidate_path) != expected_candidate_sha256:
        raise ValueError("candidate-manifest file identity mismatch")
    if card.get("candidate_manifest_sha256") != expected_candidate_sha256:
        raise ValueError("dataset card candidate-manifest identity mismatch")
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = _candidate_rows(candidate_payload, card=card)

    reset_rows = _jsonl(root / "reset_manifest.jsonl")
    attempt_rows = _jsonl(root / "attempts.jsonl")
    reset_results = _jsonl(root / "reset_results.jsonl")
    winner_rows = _jsonl(root / "winner_manifest.jsonl")
    _audit_export_state_and_progress(
        root,
        card=card,
        candidate_rows=candidates,
        reset_rows=reset_rows,
    )
    if _sha256(root / "reset_manifest.jsonl") != card.get("reset_manifest_sha256"):
        raise ValueError("reset-manifest identity mismatch")
    attempted_reset_count = int(card["attempted_reset_count"])
    if len(reset_rows) < attempted_reset_count or len(reset_results) != attempted_reset_count:
        raise ValueError("reset manifest/results count mismatch")
    if len(attempt_rows) != int(card["candidate_attempt_count"]):
        raise ValueError("attempt count does not match the dataset card")
    if len(winner_rows) != int(card["accepted_count"]):
        raise ValueError("winner count does not match the dataset card")
    if int(card["accepted_count"]) != int(card["accepted_target"]):
        raise ValueError("completed export did not meet its accepted target")
    if card.get("selection_contract") != SELECTION_CONTRACT:
        raise ValueError("dataset-card selection contract mismatch")
    budgets = [int(value) for value in card.get("budget_sequence", [])]
    if not budgets or budgets[0] != int(card["initial_k"]) or budgets[-1] != int(card["max_k"]):
        raise ValueError("dataset-card budget sequence is malformed")
    if any(right != min(int(card["max_k"]), left * 2) for left, right in zip(budgets, budgets[1:])):
        raise ValueError("dataset-card budget escalation is malformed")

    reset_by_episode: dict[str, dict[str, Any]] = {}
    for row in reset_rows:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in reset_by_episode:
            raise ValueError("reset episode IDs must be non-empty and unique")
        if row.get("task_id") != card["task"] or row.get("split") != card["split"]:
            raise ValueError("reset task/split identity mismatch")
        reset_by_episode[episode_id] = row
    attempts_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    referenced_tapes: set[str] = set()
    for record in attempt_rows:
        episode_id = record.get("episode_id")
        if episode_id not in reset_by_episode:
            raise ValueError("attempt references an unknown reset")
        reset = reset_by_episode[episode_id]
        candidate_index = int(record.get("candidate_index", -1))
        if not 0 <= candidate_index < len(candidates):
            raise ValueError("attempt candidate index is outside the frozen pool")
        candidate = candidates[candidate_index]
        if (
            record.get("candidate_id") != candidate["candidate_id"]
            or record.get("candidate_kind") != candidate["kind"]
        ):
            raise ValueError("attempt candidate identity mismatch")
        for key in (
            "episode_id",
            "task_id",
            "seed",
            "factors",
            "source_group_id",
            "pair_id",
            "pair_member_id",
        ):
            if record.get(key) != reset.get(key):
                raise ValueError(f"attempt reset identity mismatch for {key}")
        if record.get("candidate_manifest_index") != reset.get("candidate_index"):
            raise ValueError("attempt reset candidate-index provenance mismatch")
        _audit_attempt_tape(root, record, expected_task=str(card["task"]))
        tape = str(record["attempt_tape"])
        if tape in referenced_tapes:
            raise ValueError("multiple attempts reference the same lightweight tape")
        referenced_tapes.add(tape)
        attempts_by_episode[str(episode_id)].append(record)
    lightweight_root = root / "lightweight"
    actual_lightweight_files = (
        {
            path.relative_to(root).as_posix()
            for path in lightweight_root.rglob("*")
            if path.is_file()
        }
        if lightweight_root.exists()
        else set()
    )
    if actual_lightweight_files != referenced_tapes:
        raise ValueError("lightweight tape inventory contains missing or unreferenced files")

    winner_by_episode: dict[str, dict[str, Any]] = {}
    for winner in winner_rows:
        request = winner.get("request")
        episode_id = request.get("episode_id") if isinstance(request, dict) else None
        if not isinstance(episode_id, str) or episode_id in winner_by_episode:
            raise ValueError("winner episode IDs must be non-empty and unique")
        winner_by_episode[episode_id] = winner

    accepted = 0
    render_parity_skips: Counter[str] = Counter()
    skip_events = _render_parity_skip_events(card.get("recovery_events"))
    consumed_skip_events: set[int] = set()
    budget_histogram: Counter[str] = Counter()
    for reset_index, result in enumerate(reset_results):
        reset = reset_rows[reset_index]
        episode_id = reset["episode_id"]
        if result.get("reset_index") != reset_index or result.get("episode_id") != episode_id:
            raise ValueError("reset-result order or identity mismatch")
        for key in ("source_group_id",):
            if result.get(key) != reset.get(key):
                raise ValueError(f"reset-result identity mismatch for {key}")
        records = attempts_by_episode.get(episode_id, [])
        candidate_count = int(result["candidate_count"])
        budget_used = int(result["budget_used"])
        if candidate_count != budget_used or budget_used not in budgets:
            raise ValueError("reset-result candidate count/budget mismatch")
        if len(records) != candidate_count:
            raise ValueError("attempt rows do not match reset-result candidate count")
        if [int(row["candidate_index"]) for row in records] != list(range(candidate_count)):
            raise ValueError("attempted candidates are not the frozen pool prefix")
        budget_position = budgets.index(budget_used)
        for previous_budget in budgets[:budget_position]:
            if _selected(records[:previous_budget]) is not None:
                raise ValueError("candidate search escalated after already finding an eligible winner")
        selected = _selected(records)
        budget_histogram[str(budget_used)] += 1
        winner = winner_by_episode.get(episode_id)
        if selected is None:
            if bool(result.get("accepted")) or result.get("render_parity_skip") is not None:
                raise ValueError("reset-result acceptance does not recompute")
            if reset_index in skip_events:
                raise ValueError("render-parity recovery event has no selected attempt")
            if budget_used != budgets[-1] or winner is not None:
                raise ValueError("rejected reset did not exhaust max_k or unexpectedly has a winner")
            if result.get("winner_candidate_id") is not None or result.get(
                "winner_candidate_index"
            ) is not None:
                raise ValueError("rejected reset names a winner")
            continue
        if not bool(result.get("accepted")):
            if winner is not None:
                raise ValueError("render-parity skipped reset unexpectedly has a winner")
            if result.get("winner_candidate_id") is not None or result.get(
                "winner_candidate_index"
            ) is not None:
                raise ValueError("render-parity skipped reset names a winner")
            protocol = _audit_render_parity_skip(
                result,
                selected,
                skip_events.get(reset_index),
            )
            render_parity_skips[protocol] += 1
            consumed_skip_events.add(reset_index)
            continue
        if result.get("render_parity_skip") is not None or reset_index in skip_events:
            raise ValueError("accepted reset unexpectedly carries render-parity skip evidence")
        if (
            result.get("winner_candidate_id") != selected["candidate_id"]
            or int(result.get("winner_candidate_index", -1)) != int(selected["candidate_index"])
            or winner is None
            or winner.get("candidate_id") != selected["candidate_id"]
            or int(winner.get("candidate_index", -1)) != int(selected["candidate_index"])
            or int(winner.get("candidate_count", -1)) != candidate_count
            or int(winner.get("budget_used", -1)) != budget_used
            or winner.get("selection_contract") != SELECTION_CONTRACT
            or winner.get("quality_score") != list(_quality_score(selected))
            or winner.get("lightweight_attempt_tape") != selected["attempt_tape"]
            or winner.get("lightweight_attempt_tape_sha256")
            != selected["attempt_tape_sha256"]
        ):
            raise ValueError("published winner does not match independently selected attempt")
        _audit_winner_episode(root, winner, selected, reset, card)
        accepted += 1
    if consumed_skip_events != set(skip_events):
        raise ValueError("render-parity recovery event inventory mismatch")
    if accepted != len(winner_rows) or set(winner_by_episode) != {
        row["episode_id"] for row in reset_results if row["accepted"]
    }:
        raise ValueError("winner/reset-result inventory mismatch")
    if dict(budget_histogram) != {
        str(key): int(value) for key, value in card["budget_histogram"].items()
    }:
        raise ValueError("budget histogram does not recompute")
    actual_episode_dirs = {
        path.parent.relative_to(root).as_posix()
        for path in (root / "episodes").rglob("episode.json")
    }
    declared_episode_dirs = {
        str(winner["relative_episode_dir"]) for winner in winner_rows
    }
    if actual_episode_dirs != declared_episode_dirs:
        raise ValueError("winner episode directory inventory mismatch")
    staging_root = root / ".staging"
    if staging_root.exists() and any(staging_root.iterdir()):
        raise ValueError("dataset contains unpublished staging content")
    return {
        "dataset_card_payload_sha256": card["payload_sha256"],
        "task": card["task"],
        "split": card["split"],
        "source_identity": card["source_identity"],
        "accepted_count": accepted,
        "attempted_reset_count": len(reset_results),
        "candidate_attempt_count": len(attempt_rows),
        "candidate_pool_size": len(candidates),
        "checksum_entry_count": checksum_entries,
        "budget_histogram": dict(budget_histogram),
        "render_parity_skip_count": sum(render_parity_skips.values()),
        "render_parity_skip_protocols": dict(render_parity_skips),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    auditor_commit = _full_commit("auditor_commit", args.auditor_commit)
    expected_card = _expected_sha256(
        args.expected_dataset_card_sha256,
        "expected dataset-card SHA-256",
    )
    expected_checksums = _expected_sha256(
        args.expected_checksums_sha256,
        "expected SHA256SUMS SHA-256",
    )
    expected_candidate = _expected_sha256(
        args.expected_candidate_manifest_sha256,
        "expected candidate-manifest SHA-256",
    )
    started = time.time()
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_card_sha256": expected_card,
        "checksums_sha256": expected_checksums,
        "candidate_manifest_sha256": expected_candidate,
        "auditor_commit": auditor_commit,
        "started_unix_s": started,
    }
    try:
        summary = _audit_dataset(
            root=args.dataset_root.resolve(),
            expected_card_sha256=expected_card,
            expected_checksums_sha256=expected_checksums,
            expected_candidate_sha256=expected_candidate,
        )
        report.update(
            status="passed",
            training_eligible=True,
            training_eligibility_reason="independent audit passed",
            summary=summary,
        )
    except Exception as error:
        report.update(
            status="failed",
            training_eligible=False,
            training_eligibility_reason="independent audit failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        report["finished_unix_s"] = time.time()
        report["payload_sha256"] = _payload_sha256(report)
        _atomic_json(args.output, report)
        raise
    report["finished_unix_s"] = time.time()
    report["payload_sha256"] = _payload_sha256(report)
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
