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

"""Merge sharded best-known trajectory exports into one sealed dataset root.

Each shard produced by ``export_dynamic_benchmark_optimal_trajectories.py
--shard-index N --shard-count M`` writes ``shard-NN/`` under the same parent
root. This entrypoint validates that every shard finished its slice, merges
``attempts.jsonl`` / ``reset_results.jsonl`` / ``winner_manifest.jsonl`` in
global reset order, keeps the first ``--accepted-episodes`` winners, copies the
corresponding episodes, lightweight tapes, and independently gated Qv4 full
exports, then seals a dataset card and ``SHA256SUMS`` exactly like the
single-process exporter. An opt-in fixed-reset
mode keeps the complete reset workload and requires its winner count to equal
the declared target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from examples.embodiment.dynamic_benchmark_calibration_projection import (
    CALIBRATION_BINDING_PROJECTION_SCHEMA,
)
from examples.embodiment.export_dynamic_benchmark_optimal_trajectories import (
    EXPORT_SCHEMA,
    FIRST_ELIGIBLE_SEARCH_MODE,
    LEGACY_SELECTION_MODE,
    PLANNER_PARETO_SELECTION_MODE,
    PROGRESS_SCHEMA,
    QUALITY_V2_THRESHOLDS_SCHEMA,
    _atomic_json,
    _expected_sha256,
    _file_boundary,
    _full_commit,
    _payload_sha256,
    _quality_v2_calibration_receipt_binding,
    _root_checksums,
    _selection_contract,
    _validate_quality_v2_calibration_receipt_artifact,
)

QUALITY_V4_THRESHOLDS_FILENAME = "quality_v4_thresholds.json"
QUALITY_V4_OWNER_REVIEW_RECEIPT_RELATIVE_PATH = (
    "provenance/quality_v4/owner_review_receipt.json"
)
QUALITY_V4_OWNER_REVIEW_RECEIPT_SCHEMA = (
    "se3wam-quality-v4-owner-review-receipt-v0.1"
)
QUALITY_V4_ATTEMPT_SCHEMA = "rlinf-dynamic-benchmark-quality-v4-attempt-v0.1"
QUALITY_V4_FULL_EXPORT_GATE_SCHEMA = (
    "rlinf-dynamic-benchmark-quality-v4-full-export-gate-v0.1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-episodes", type=int, default=100)
    parser.add_argument(
        "--require-max-resets",
        action="store_true",
        help=(
            "seal every reset declared by export_state.max_resets and require "
            "the complete workload to contain exactly --accepted-episodes winners"
        ),
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a mapping")
            rows.append(value)
    return rows


_SHARD_RECEIPT_KEYS = {
    "schema_version",
    "shard_index",
    "shard_count",
    "accepted_count",
    "attempted_reset_count",
    "candidate_attempt_count",
    "candidate_search_mode",
    "selection_mode",
    "budget_histogram",
}


def _load_shard_receipts(root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    """Load an exact, gap-free shard directory and completion inventory."""

    shard_like = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("shard-")
    )
    malformed = [
        path.name
        for path in shard_like
        if re.fullmatch(r"shard-\d{2}", path.name) is None
    ]
    if malformed:
        raise ValueError(f"malformed shard directories: {malformed}")
    if not shard_like:
        raise ValueError(f"no shard-* directories under {root}")

    receipts: list[dict[str, Any]] = []
    for shard in shard_like:
        complete = shard / "shard_complete.json"
        if not complete.is_file():
            raise ValueError(f"{shard} is missing shard_complete.json")
        receipt = json.loads(complete.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or set(receipt) != _SHARD_RECEIPT_KEYS:
            raise ValueError(f"{shard} has an invalid shard completion receipt")
        receipts.append(receipt)

    raw_shard_counts = [receipt.get("shard_count") for receipt in receipts]
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_shard_counts
    ) or any(not 1 <= value <= 99 for value in raw_shard_counts):
        raise ValueError("shard_count must be an integer in [1, 99]")
    shard_counts = set(raw_shard_counts)
    if len(shard_counts) != 1:
        raise ValueError("shard completion receipts disagree on shard_count")
    shard_count = shard_counts.pop()
    expected_names = [f"shard-{index:02d}" for index in range(shard_count)]
    actual_names = [path.name for path in shard_like]
    if actual_names != expected_names:
        raise ValueError(
            "shard directory inventory is not exact and gap-free: "
            f"expected={expected_names}, actual={actual_names}"
        )
    for index, receipt in enumerate(receipts):
        if (
            receipt.get("schema_version")
            != "rlinf-dynamic-benchmark-optimal-shard-v0.1"
            or receipt.get("shard_index") != index
            or receipt.get("shard_count") != shard_count
        ):
            raise ValueError(f"shard-{index:02d} completion identity mismatch")
    return shard_like, receipts


def _expected_shard_indices(
    *, max_resets: int, shard_count: int, shard_index: int
) -> list[int]:
    """Return the exporter's exact contiguous reset slice for one shard."""

    if max_resets < 1 or shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid reset-shard dimensions")
    step = (max_resets + shard_count - 1) // shard_count
    start = shard_index * step
    return list(range(start, min(start + step, max_resets)))


def _validate_shard_records(
    *,
    shard: Path,
    receipt: dict[str, Any],
    export_state: dict[str, Any],
    reset_manifest: list[dict[str, Any]],
    results: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    winners: list[dict[str, Any]],
) -> None:
    """Prove exact reset and full-pool coverage for one completed shard."""

    shard_index = receipt["shard_index"]
    shard_count = receipt["shard_count"]
    max_resets = export_state.get("max_resets")
    candidate_pool_size = export_state.get("candidate_pool_size")
    if (
        isinstance(max_resets, bool)
        or not isinstance(max_resets, int)
        or max_resets < 1
        or isinstance(candidate_pool_size, bool)
        or not isinstance(candidate_pool_size, int)
        or candidate_pool_size < 1
    ):
        raise ValueError(
            f"{shard} export state has invalid reset or candidate dimensions"
        )
    if len(reset_manifest) != max_resets:
        raise ValueError(f"{shard} reset manifest length differs from max_resets")
    expected_indices = _expected_shard_indices(
        max_resets=max_resets,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    actual_indices = [row.get("reset_index") for row in results]
    if actual_indices != expected_indices:
        raise ValueError(
            f"{shard} reset coverage has a duplicate, gap, or order mismatch: "
            f"expected={expected_indices}, actual={actual_indices}"
        )
    if (
        receipt.get("attempted_reset_count") != len(results)
        or receipt.get("candidate_attempt_count") != len(attempts)
        or receipt.get("accepted_count") != len(winners)
        or receipt.get("candidate_search_mode")
        != export_state.get("candidate_search_mode")
        or receipt.get("selection_mode") != export_state.get("selection_mode")
    ):
        raise ValueError(f"{shard} completion receipt differs from its sealed rows")
    selection_mode = export_state.get("selection_mode")
    if export_state.get("candidate_search_mode") != "full-pool" or selection_mode not in {
        LEGACY_SELECTION_MODE,
        PLANNER_PARETO_SELECTION_MODE,
    }:
        raise ValueError(f"{shard} is not a supported full-pool export")

    attempts_by_episode: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        episode_id = attempt.get("episode_id")
        if not isinstance(episode_id, str):
            raise ValueError(f"{shard} attempt has no episode identity")
        attempts_by_episode.setdefault(episode_id, []).append(attempt)
    winner_ids = [_winner_episode_id(row) for row in winners]
    if len(winner_ids) != len(set(winner_ids)):
        raise ValueError(f"{shard} contains duplicate winner episodes")
    expected_winner_ids: list[str] = []
    for reset_index, result in zip(expected_indices, results, strict=True):
        expected_episode = reset_manifest[reset_index].get("episode_id")
        if (
            not isinstance(expected_episode, str)
            or result.get("episode_id") != expected_episode
        ):
            raise ValueError(
                f"{shard} reset result differs from the frozen reset manifest"
            )
        if (
            result.get("candidate_count") != candidate_pool_size
            or result.get("budget_used") != candidate_pool_size
            or result.get("candidate_search_mode") != "full-pool"
            or result.get("selection_mode") != selection_mode
        ):
            raise ValueError(f"{shard} reset did not evaluate the exact full pool")
        episode_attempts = attempts_by_episode.pop(expected_episode, [])
        candidate_indices = [row.get("candidate_index") for row in episode_attempts]
        if candidate_indices != list(range(candidate_pool_size)):
            raise ValueError(
                f"{shard} candidate coverage has a duplicate, gap, or order mismatch"
            )
        if result.get("accepted") is True:
            expected_winner_ids.append(expected_episode)
    if attempts_by_episode:
        raise ValueError(f"{shard} contains attempts outside its reset slice")
    if winner_ids != expected_winner_ids:
        raise ValueError(
            f"{shard} winner inventory differs from accepted reset results"
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            stream.flush()
    temporary.replace(path)


def _identity_sha256(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _episode_number(episode_id: str) -> int:
    match = re.search(r"-(\d{5,})-s\d+$", episode_id)
    if match is None:
        raise ValueError(f"cannot parse episode id {episode_id!r}")
    return int(match.group(1))


def _winner_episode_id(row: dict[str, Any]) -> str:
    request = row.get("request")
    if not isinstance(request, dict) or not isinstance(request.get("episode_id"), str):
        raise KeyError("winner row is missing request.episode_id")
    return request["episode_id"]


def _select_merged_workload(
    *,
    results: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    accepted_episodes: int,
    require_max_resets: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Select the sealed reset/winner workload after exact shard validation."""

    reset_index_by_episode = {
        row["episode_id"]: int(row["reset_index"]) for row in results
    }
    ordered_winners = sorted(
        winners,
        key=lambda row: reset_index_by_episode[_winner_episode_id(row)],
    )
    if require_max_resets:
        if len(ordered_winners) != accepted_episodes:
            raise RuntimeError(
                "fixed reset workload produced "
                f"{len(ordered_winners)}/{accepted_episodes} winners"
            )
        return ordered_winners, list(results), int(results[-1]["reset_index"])

    kept_winners = ordered_winners[:accepted_episodes]
    if len(kept_winners) < accepted_episodes:
        raise RuntimeError(
            f"only {len(kept_winners)}/{accepted_episodes} winners across shards"
        )
    max_reset = reset_index_by_episode[_winner_episode_id(kept_winners[-1])]
    kept_results = [
        row for row in results if int(row["reset_index"]) <= max_reset
    ]
    return kept_winners, kept_results, max_reset


def _kept_recovery_events(events: list[str], *, max_reset: int) -> list[str]:
    """Keep source recovery provenance that can affect the merged prefix."""

    kept = []
    for event in events:
        if not isinstance(event, str):
            raise ValueError("shard recovery events must be strings")
        if not event.startswith("render_parity_skip:"):
            kept.append(event)
            continue
        parts = event.split(":", maxsplit=4)
        if len(parts) != 5 or parts[:2] != ["render_parity_skip", "reset"]:
            raise ValueError("shard render-parity recovery event is malformed")
        try:
            reset_index = int(parts[2])
        except ValueError as error:
            raise ValueError("shard render-parity reset index is malformed") from error
        if reset_index < 0:
            raise ValueError("shard render-parity reset index is negative")
        if reset_index <= max_reset:
            kept.append(event)
    return kept


def _tree_inventory(root: Path) -> dict[str, str]:
    """Return an exact relative-path/SHA-256 inventory for a portable tree."""

    if not root.exists():
        return {}
    if not root.is_dir():
        raise ValueError(f"expected directory, got {root}")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _validate_quality_v4_shard_contract(
    shard: Path,
    export_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen one shard's frozen Qv4 threshold and owner-review receipt."""

    from examples.embodiment.dynamic_benchmark_quality_v4 import (
        validate_quality_v4_thresholds,
    )

    raw_identity = export_state.get("quality_v4_threshold_identity")
    if not isinstance(raw_identity, Mapping) or set(raw_identity) != {
        "schema_version",
        "file_sha256",
        "payload_sha256",
        "owner_review_receipt",
    }:
        raise ValueError(f"{shard} has no exact Qv4 threshold identity")
    threshold_file_sha256 = _expected_sha256(
        raw_identity.get("file_sha256"),
        f"{shard} Qv4 threshold file SHA-256",
    )
    threshold_payload_sha256 = _expected_sha256(
        raw_identity.get("payload_sha256"),
        f"{shard} Qv4 threshold payload SHA-256",
    )
    threshold_path = shard / QUALITY_V4_THRESHOLDS_FILENAME
    if threshold_path.is_symlink() or not threshold_path.is_file():
        raise ValueError(f"{shard} has no dataset-local Qv4 threshold file")
    threshold_bytes = threshold_path.read_bytes()
    if hashlib.sha256(threshold_bytes).hexdigest() != threshold_file_sha256:
        raise ValueError(f"{shard} Qv4 threshold file SHA-256 mismatch")
    try:
        thresholds = json.loads(threshold_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{shard} Qv4 threshold is not UTF-8 JSON") from error
    if not isinstance(thresholds, Mapping):
        raise ValueError(f"{shard} Qv4 threshold must be a mapping")
    validation = validate_quality_v4_thresholds(
        thresholds,
        expected_thresholds_sha256=threshold_payload_sha256,
        require_formal_freeze=True,
    )
    if (
        validation.get("schema_version") != raw_identity.get("schema_version")
        or validation.get("thresholds_sha256") != threshold_payload_sha256
        or validation.get("formal_freeze_eligible") is not True
    ):
        raise ValueError(f"{shard} Qv4 frozen threshold identity mismatch")

    raw_receipt_identity = raw_identity.get("owner_review_receipt")
    if not isinstance(raw_receipt_identity, Mapping) or set(
        raw_receipt_identity
    ) != {"relative_path", "file_sha256", "payload_sha256"}:
        raise ValueError(f"{shard} has no exact Qv4 owner-review receipt identity")
    if (
        raw_receipt_identity.get("relative_path")
        != QUALITY_V4_OWNER_REVIEW_RECEIPT_RELATIVE_PATH
    ):
        raise ValueError(f"{shard} Qv4 owner-review receipt path mismatch")
    receipt_file_sha256 = _expected_sha256(
        raw_receipt_identity.get("file_sha256"),
        f"{shard} Qv4 owner-review receipt file SHA-256",
    )
    receipt_payload_sha256 = _expected_sha256(
        raw_receipt_identity.get("payload_sha256"),
        f"{shard} Qv4 owner-review receipt payload SHA-256",
    )
    receipt_path = shard / QUALITY_V4_OWNER_REVIEW_RECEIPT_RELATIVE_PATH
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(f"{shard} has no Qv4 owner-review receipt")
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_file_sha256:
        raise ValueError(f"{shard} Qv4 owner-review receipt file SHA-256 mismatch")
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{shard} Qv4 owner-review receipt is not UTF-8 JSON") from error
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version",
        "threshold_schema_version",
        "threshold_file_sha256",
        "threshold_payload_sha256",
        "owner_review",
        "payload_sha256",
    }:
        raise ValueError(f"{shard} Qv4 owner-review receipt inventory mismatch")
    if (
        receipt.get("schema_version") != QUALITY_V4_OWNER_REVIEW_RECEIPT_SCHEMA
        or receipt.get("threshold_schema_version") != raw_identity.get("schema_version")
        or receipt.get("threshold_file_sha256") != threshold_file_sha256
        or receipt.get("threshold_payload_sha256") != threshold_payload_sha256
        or receipt.get("owner_review") != thresholds.get("owner_review")
        or receipt.get("payload_sha256") != receipt_payload_sha256
        or _payload_sha256(receipt) != receipt_payload_sha256
    ):
        raise ValueError(f"{shard} Qv4 owner-review receipt binding mismatch")
    return {
        "schema_version": str(raw_identity["schema_version"]),
        "file_sha256": threshold_file_sha256,
        "payload_sha256": threshold_payload_sha256,
        "owner_review_receipt": {
            "relative_path": QUALITY_V4_OWNER_REVIEW_RECEIPT_RELATIVE_PATH,
            "file_sha256": receipt_file_sha256,
            "payload_sha256": receipt_payload_sha256,
        },
    }


def _quality_v4_expected_files(
    reset_episode_ids: list[str],
    winner_episode_ids: list[str],
) -> set[str]:
    return {
        *(f"attempts/{episode_id}.json" for episode_id in reset_episode_ids),
        *(f"lightweight_sources/{episode_id}.h5" for episode_id in reset_episode_ids),
        *(f"full_exports/{episode_id}.h5" for episode_id in winner_episode_ids),
        *(f"full_exports/{episode_id}.gate.json" for episode_id in winner_episode_ids),
    }


def _validate_quality_v4_artifacts(
    root: Path,
    *,
    task: str,
    threshold_identity: Mapping[str, Any],
    reset_episode_ids: list[str],
    winner_episode_ids: list[str],
) -> None:
    """Validate the exact Qv4 source/full-export/gate inventory for one root."""

    from examples.embodiment.dynamic_benchmark_quality_v4 import (
        QUALITY_V4_ARTIFACT_SUBDIRECTORY,
        audit_quality_v4_lightweight_source,
    )

    directory = root / QUALITY_V4_ARTIFACT_SUBDIRECTORY
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{root} is missing the required Qv4 directory")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError(f"{root} Qv4 artifacts cannot contain symlinks")
    expected_files = _quality_v4_expected_files(
        reset_episode_ids,
        winner_episode_ids,
    )
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            f"{root} Qv4 artifact inventory mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )

    attempts: dict[str, Mapping[str, Any]] = {}
    for episode_id in reset_episode_ids:
        attempt_path = directory / "attempts" / f"{episode_id}.json"
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        if not isinstance(attempt, Mapping):
            raise ValueError(f"{root} Qv4 attempt {episode_id} is not a mapping")
        attempt_sha256 = _expected_sha256(
            attempt.get("attempt_sha256"),
            f"{root} Qv4 attempt {episode_id} SHA-256",
        )
        if (
            attempt.get("schema_version") != QUALITY_V4_ATTEMPT_SCHEMA
            or attempt.get("episode_id") != episode_id
            or attempt.get("task_id") != task
            or attempt.get("thresholds_sha256")
            != threshold_identity.get("payload_sha256")
            or _identity_sha256(attempt, "attempt_sha256") != attempt_sha256
        ):
            raise ValueError(f"{root} Qv4 attempt {episode_id} identity drift")
        lightweight = audit_quality_v4_lightweight_source(
            directory / "lightweight_sources" / f"{episode_id}.h5"
        )
        if (
            lightweight.get("episode_id") != episode_id
            or lightweight.get("task_id") != task
            or lightweight.get("attempt_sha256") != attempt_sha256
            or lightweight.get("source_sha256") != attempt.get("source_sha256")
        ):
            raise ValueError(f"{root} Qv4 lightweight source {episode_id} identity drift")
        attempts[episode_id] = attempt

    for episode_id in winner_episode_ids:
        attempt = attempts.get(episode_id)
        if attempt is None:
            raise ValueError(f"{root} Qv4 winner {episode_id} has no source attempt")
        export_path = directory / "full_exports" / f"{episode_id}.h5"
        gate_path = directory / "full_exports" / f"{episode_id}.gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not isinstance(gate, Mapping):
            raise ValueError(f"{root} Qv4 winner gate {episode_id} is not a mapping")
        gate_sha256 = _expected_sha256(
            gate.get("gate_sha256"),
            f"{root} Qv4 winner gate {episode_id} SHA-256",
        )
        if (
            gate.get("schema_version") != QUALITY_V4_FULL_EXPORT_GATE_SCHEMA
            or gate.get("episode_id") != episode_id
            or gate.get("task_id") != task
            or gate.get("attempt_sha256") != attempt.get("attempt_sha256")
            or gate.get("thresholds_sha256")
            != threshold_identity.get("payload_sha256")
            or gate.get("export_file_sha256")
            != hashlib.sha256(export_path.read_bytes()).hexdigest()
            or gate.get("formal_thresholds_frozen") is not True
            or gate.get("owner_review_complete") is not True
            or gate.get("passed") is not True
            or gate.get("eligible_for_behavior_cloning") is not True
            or _identity_sha256(gate, "gate_sha256") != gate_sha256
        ):
            raise ValueError(f"{root} Qv4 winner gate {episode_id} identity drift")


def _copy_quality_v4_artifacts(
    *,
    shard_dirs: list[Path],
    output: Path,
    reset_episode_ids: list[str],
    winner_episode_ids: list[str],
) -> None:
    """Copy the kept Qv4 prefix while rejecting every cross-shard collision."""

    from examples.embodiment.dynamic_benchmark_quality_v4 import (
        QUALITY_V4_ARTIFACT_SUBDIRECTORY,
    )

    for relative in sorted(
        _quality_v4_expected_files(reset_episode_ids, winner_episode_ids)
    ):
        sources = [
            shard / QUALITY_V4_ARTIFACT_SUBDIRECTORY / relative
            for shard in shard_dirs
            if (shard / QUALITY_V4_ARTIFACT_SUBDIRECTORY / relative).is_file()
        ]
        if len(sources) != 1:
            raise ValueError(
                f"Qv4 artifact {relative!r} has "
                f"{len(sources)} sources; expected exactly one"
            )
        destination = output / QUALITY_V4_ARTIFACT_SUBDIRECTORY / relative
        if destination.exists():
            raise FileExistsError(f"Qv4 artifact collision at {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[0], destination)


def _validate_quality_v2_shard_contract(
    shard: Path,
    export_state: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Reopen one shard's frozen Qv3 threshold and calibration receipt."""

    raw_identity = export_state.get("quality_v2_threshold_identity")
    if not isinstance(raw_identity, Mapping) or set(raw_identity) != {
        "schema_version",
        "sha256",
    }:
        raise ValueError(
            f"{shard} export state has no exact quality-v2 threshold identity"
        )
    schema_version = raw_identity.get("schema_version")
    if schema_version != QUALITY_V2_THRESHOLDS_SCHEMA:
        raise ValueError(f"{shard} export state has a non-Qv3 threshold schema")
    threshold_sha256 = _expected_sha256(
        raw_identity.get("sha256"),
        f"{shard} quality-v2 threshold SHA-256",
    )
    identity = {
        "schema_version": schema_version,
        "sha256": threshold_sha256,
    }

    threshold_path = shard / "quality_v2_thresholds.json"
    if threshold_path.is_symlink() or not threshold_path.is_file():
        raise ValueError(f"{shard} has no dataset-local quality-v2 threshold file")
    threshold_bytes = threshold_path.read_bytes()
    if hashlib.sha256(threshold_bytes).hexdigest() != threshold_sha256:
        raise ValueError(f"{shard} quality-v2 threshold identity SHA-256 mismatch")
    try:
        threshold_payload = json.loads(threshold_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{shard} quality-v2 threshold is not UTF-8 JSON") from error
    if not isinstance(threshold_payload, Mapping):
        raise ValueError(f"{shard} quality-v2 threshold must be a mapping")
    if (
        threshold_payload.get("schema_version") != schema_version
        or threshold_payload.get("formal_freeze_eligible") is not True
    ):
        raise ValueError(f"{shard} quality-v2 threshold contract identity mismatch")

    receipt_binding = _quality_v2_calibration_receipt_binding(threshold_payload)
    source_identity = export_state.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError(f"{shard} export state has no source identity")
    benchmark_key = (
        "evaluator_benchmark_commit"
        if "evaluator_benchmark_commit" in source_identity
        else "benchmark_commit"
    )
    expected_benchmark_commit = _full_commit(
        f"{shard} authenticated benchmark commit",
        source_identity.get(benchmark_key),
    )
    receipt_path = shard / receipt_binding["relative_path"]
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(
            f"{shard} quality-v2 calibration receipt is missing or symlinked"
        )
    raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_receipt_sha256 = (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        if isinstance(raw_receipt, Mapping)
        and raw_receipt.get("schema_version")
        == CALIBRATION_BINDING_PROJECTION_SCHEMA
        else receipt_binding["file_sha256"]
    )
    receipt = _validate_quality_v2_calibration_receipt_artifact(
        threshold_payload,
        receipt_path,
        expected_sha256=expected_receipt_sha256,
        expected_benchmark_commit=expected_benchmark_commit,
    )
    receipt_identity = {
        "relative_path": receipt.relative_path,
        "file_sha256": receipt.sha256,
        "payload_sha256": receipt.sha256,
    }
    return identity, receipt_identity


def _validate_quality_v4_shard_contract(
    shard: Path,
    export_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen one shard's frozen Qv4 threshold and owner-review receipt."""

    from examples.embodiment.dynamic_benchmark_quality_v4 import (
        validate_quality_v4_thresholds,
    )

    raw_identity = export_state.get("quality_v4_threshold_identity")
    if not isinstance(raw_identity, Mapping) or set(raw_identity) != {
        "schema_version",
        "file_sha256",
        "payload_sha256",
        "owner_review_receipt",
    }:
        raise ValueError(f"{shard} has no exact Qv4 threshold identity")
    threshold_file_sha256 = _expected_sha256(
        raw_identity.get("file_sha256"),
        f"{shard} Qv4 threshold file SHA-256",
    )
    threshold_payload_sha256 = _expected_sha256(
        raw_identity.get("payload_sha256"),
        f"{shard} Qv4 threshold payload SHA-256",
    )
    threshold_path = shard / QUALITY_V4_THRESHOLDS_FILENAME
    if threshold_path.is_symlink() or not threshold_path.is_file():
        raise ValueError(f"{shard} has no dataset-local Qv4 threshold file")
    threshold_bytes = threshold_path.read_bytes()
    if hashlib.sha256(threshold_bytes).hexdigest() != threshold_file_sha256:
        raise ValueError(f"{shard} Qv4 threshold file SHA-256 mismatch")
    try:
        thresholds = json.loads(threshold_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{shard} Qv4 threshold is not UTF-8 JSON") from error
    if not isinstance(thresholds, Mapping):
        raise ValueError(f"{shard} Qv4 threshold must be a mapping")
    validation = validate_quality_v4_thresholds(
        thresholds,
        expected_thresholds_sha256=threshold_payload_sha256,
        require_formal_freeze=True,
    )
    if (
        validation.get("schema_version") != raw_identity.get("schema_version")
        or validation.get("thresholds_sha256") != threshold_payload_sha256
        or validation.get("formal_freeze_eligible") is not True
    ):
        raise ValueError(f"{shard} Qv4 frozen threshold identity mismatch")

    raw_receipt_identity = raw_identity.get("owner_review_receipt")
    if not isinstance(raw_receipt_identity, Mapping) or set(
        raw_receipt_identity
    ) != {"relative_path", "file_sha256", "payload_sha256"}:
        raise ValueError(f"{shard} has no exact Qv4 owner-review receipt identity")
    if (
        raw_receipt_identity.get("relative_path")
        != QUALITY_V4_OWNER_REVIEW_RECEIPT_RELATIVE_PATH
    ):
        raise ValueError(f"{shard} Qv4 owner-review receipt path mismatch")
    receipt_file_sha256 = _expected_sha256(
        raw_receipt_identity.get("file_sha256"),
        f"{shard} Qv4 owner-review receipt file SHA-256",
    )
    receipt_payload_sha256 = _expected_sha256(
        raw_receipt_identity.get("payload_sha256"),
        f"{shard} Qv4 owner-review receipt payload SHA-256",
    )
    receipt_path = shard / QUALITY_V4_OWNER_REVIEW_RECEIPT_RELATIVE_PATH
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(f"{shard} has no Qv4 owner-review receipt")
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_file_sha256:
        raise ValueError(f"{shard} Qv4 owner-review receipt file SHA-256 mismatch")
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{shard} Qv4 owner-review receipt is not UTF-8 JSON") from error
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version",
        "threshold_schema_version",
        "threshold_file_sha256",
        "threshold_payload_sha256",
        "owner_review",
        "payload_sha256",
    }:
        raise ValueError(f"{shard} Qv4 owner-review receipt inventory mismatch")
    if (
        receipt.get("schema_version") != QUALITY_V4_OWNER_REVIEW_RECEIPT_SCHEMA
        or receipt.get("threshold_schema_version") != raw_identity.get("schema_version")
        or receipt.get("threshold_file_sha256") != threshold_file_sha256
        or receipt.get("threshold_payload_sha256") != threshold_payload_sha256
        or receipt.get("owner_review") != thresholds.get("owner_review")
        or receipt.get("payload_sha256") != receipt_payload_sha256
        or _payload_sha256(receipt) != receipt_payload_sha256
    ):
        raise ValueError(f"{shard} Qv4 owner-review receipt binding mismatch")
    return {
        "schema_version": str(raw_identity["schema_version"]),
        "file_sha256": threshold_file_sha256,
        "payload_sha256": threshold_payload_sha256,
        "owner_review_receipt": {
            "relative_path": QUALITY_V4_OWNER_REVIEW_RECEIPT_RELATIVE_PATH,
            "file_sha256": receipt_file_sha256,
            "payload_sha256": receipt_payload_sha256,
        },
    }


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    shard_dirs, shard_receipts = _load_shard_receipts(root)

    all_results: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    all_winners: list[dict[str, Any]] = []
    all_recovery_events: list[str] = []
    resume_count = 0
    started_unix_s: float | None = None
    reference_export_state: dict[str, Any] | None = None
    reference_candidate_sha256: str | None = None
    reference_reset_sha256: str | None = None
    reference_provenance: dict[str, str] | None = None
    reference_quality_v2_threshold_identity: dict[str, str] | None = None
    reference_quality_v2_receipt_binding: dict[str, str] | None = None
    reference_quality_v4_threshold_identity: dict[str, Any] | None = None
    reset_manifest: list[dict[str, Any]] | None = None
    for shard, receipt in zip(shard_dirs, shard_receipts, strict=True):
        shard_results = _read_jsonl(shard / "reset_results.jsonl")
        shard_attempts = _read_jsonl(shard / "attempts.jsonl")
        shard_winners = _read_jsonl(shard / "winner_manifest.jsonl")
        progress = json.loads((shard / "progress.json").read_text(encoding="utf-8"))
        recovery_events = progress.get("recovery_events")
        if not isinstance(recovery_events, list):
            raise ValueError(f"{shard} progress has no recovery-event list")
        all_recovery_events.extend(recovery_events)
        resume_count += int(progress.get("resume_count", 0))
        if (
            started_unix_s is None
            or progress.get("started_unix_s", float("inf")) < started_unix_s
        ):
            started_unix_s = progress.get("started_unix_s")
        shard_export_state = json.loads(
            (shard / "export_state.json").read_text(encoding="utf-8")
        )
        (
            shard_quality_v2_threshold_identity,
            shard_quality_v2_receipt_binding,
        ) = _validate_quality_v2_shard_contract(shard, shard_export_state)
        shard_quality_v4_threshold_identity = _validate_quality_v4_shard_contract(
            shard,
            shard_export_state,
        )
        shard_candidate_sha256 = hashlib.sha256(
            (shard / "candidate_manifest.json").read_bytes()
        ).hexdigest()
        shard_reset_sha256 = hashlib.sha256(
            (shard / "reset_manifest.jsonl").read_bytes()
        ).hexdigest()
        shard_reset_manifest = _read_jsonl(shard / "reset_manifest.jsonl")
        shard_provenance = _tree_inventory(shard / "provenance")
        if reference_export_state is None:
            reference_export_state = shard_export_state
            reference_candidate_sha256 = shard_candidate_sha256
            reference_reset_sha256 = shard_reset_sha256
            reference_provenance = shard_provenance
            reference_quality_v2_threshold_identity = (
                shard_quality_v2_threshold_identity
            )
            reference_quality_v2_receipt_binding = shard_quality_v2_receipt_binding
            reference_quality_v4_threshold_identity = (
                shard_quality_v4_threshold_identity
            )
            reset_manifest = shard_reset_manifest
        elif (
            shard_quality_v2_threshold_identity
            != reference_quality_v2_threshold_identity
            or shard_quality_v2_receipt_binding != reference_quality_v2_receipt_binding
        ):
            raise ValueError(
                f"{shard} has a different quality-v2 threshold or receipt identity"
            )
        elif (
            shard_quality_v4_threshold_identity
            != reference_quality_v4_threshold_identity
        ):
            raise ValueError(f"{shard} has a different Qv4 threshold or receipt identity")
        elif (
            shard_export_state != reference_export_state
            or shard_candidate_sha256 != reference_candidate_sha256
            or shard_reset_sha256 != reference_reset_sha256
            or shard_provenance != reference_provenance
        ):
            raise ValueError(f"{shard} has a different frozen export contract")
        assert reset_manifest is not None
        _validate_shard_records(
            shard=shard,
            receipt=receipt,
            export_state=shard_export_state,
            reset_manifest=reset_manifest,
            results=shard_results,
            attempts=shard_attempts,
            winners=shard_winners,
        )
        _validate_quality_v4_artifacts(
            shard,
            task=str(shard_export_state["task"]),
            threshold_identity=shard_quality_v4_threshold_identity,
            reset_episode_ids=[str(row["episode_id"]) for row in shard_results],
            winner_episode_ids=[_winner_episode_id(row) for row in shard_winners],
        )
        all_results.extend(shard_results)
        all_attempts.extend(shard_attempts)
        all_winners.extend(shard_winners)
    if started_unix_s is None:
        raise ValueError("no shard start time found")
    assert reference_export_state is not None
    assert reference_quality_v2_threshold_identity is not None
    assert reference_quality_v2_receipt_binding is not None
    assert reference_quality_v4_threshold_identity is not None
    expected_global_indices = list(range(int(reference_export_state["max_resets"])))
    if [row["reset_index"] for row in all_results] != expected_global_indices:
        raise ValueError("merged reset coverage is not exact, ordered, and gap-free")

    all_results.sort(key=lambda row: int(row["reset_index"]))
    reset_index_by_episode: dict[str, int] = {
        row["episode_id"]: int(row["reset_index"]) for row in all_results
    }
    kept_winners, kept_results, max_reset = _select_merged_workload(
        results=all_results,
        winners=all_winners,
        accepted_episodes=args.accepted_episodes,
        require_max_resets=args.require_max_resets,
    )
    recovery_events = _kept_recovery_events(all_recovery_events, max_reset=max_reset)
    kept_episodes = {row["episode_id"] for row in kept_results}
    kept_attempts = [row for row in all_attempts if row["episode_id"] in kept_episodes]
    kept_attempts.sort(
        key=lambda row: (_episode_number(row["episode_id"]), row["candidate_index"])
    )
    kept_winners.sort(key=lambda row: reset_index_by_episode[_winner_episode_id(row)])

    budget_histogram: dict[str, int] = {}
    for row in kept_results:
        key = str(row["budget_used"])
        budget_histogram[key] = budget_histogram.get(key, 0) + 1

    reference = shard_dirs[0]
    export_state = reference_export_state
    task = export_state["task"]
    split = export_state["split"]
    image_size = int(export_state["image_size"])
    device = str(export_state["device"])
    candidate_manifest_sha256 = str(export_state["candidate_manifest_sha256"])
    source_identity = dict(export_state["source_identity"])
    initial_k = int(export_state["initial_k"])
    max_k = int(export_state["max_k"])

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    shutil.copyfile(
        reference / "candidate_manifest.json", output / "candidate_manifest.json"
    )
    shutil.copyfile(
        reference / "quality_v2_thresholds.json",
        output / "quality_v2_thresholds.json",
    )
    shutil.copyfile(
        reference / QUALITY_V4_THRESHOLDS_FILENAME,
        output / QUALITY_V4_THRESHOLDS_FILENAME,
    )
    shutil.copyfile(reference / "export_state.json", output / "export_state.json")
    shutil.copyfile(reference / "reset_manifest.jsonl", output / "reset_manifest.jsonl")
    if (reference / "provenance").is_dir():
        shutil.copytree(reference / "provenance", output / "provenance")
    (
        merged_quality_v2_threshold_identity,
        merged_quality_v2_receipt_binding,
    ) = _validate_quality_v2_shard_contract(
        output,
        export_state,
    )
    if (
        merged_quality_v2_threshold_identity != reference_quality_v2_threshold_identity
        or merged_quality_v2_receipt_binding != reference_quality_v2_receipt_binding
    ):
        raise RuntimeError("merged quality-v2 threshold or receipt identity changed")
    merged_quality_v4_threshold_identity = _validate_quality_v4_shard_contract(
        output,
        export_state,
    )
    if merged_quality_v4_threshold_identity != reference_quality_v4_threshold_identity:
        raise RuntimeError("merged Qv4 threshold or receipt identity changed")

    _write_jsonl(output / "attempts.jsonl", kept_attempts)
    _write_jsonl(output / "reset_results.jsonl", kept_results)
    _write_jsonl(output / "winner_manifest.jsonl", kept_winners)

    for winner in kept_winners:
        episode_id = _winner_episode_id(winner)
        relative = winner.get("relative_episode_dir")
        target_rel = (
            relative
            if isinstance(relative, str)
            else (f"episodes/{task}/{split}/{episode_id}")
        )
        for shard in shard_dirs:
            source = (
                shard / target_rel
                if isinstance(relative, str)
                else (shard / "episodes" / task / split / episode_id)
            )
            if source.exists():
                shutil.copytree(source, output / target_rel)
                break
        else:
            raise FileNotFoundError(
                f"winner episode {episode_id} not found in any shard"
            )
    for episode_id in kept_episodes:
        for shard in shard_dirs:
            source = shard / "lightweight" / episode_id
            if source.exists():
                shutil.copytree(source, output / "lightweight" / episode_id)
                break
        else:
            raise FileNotFoundError(
                f"lightweight tape {episode_id} not found in any shard"
            )
    kept_reset_episode_ids = [str(row["episode_id"]) for row in kept_results]
    kept_winner_episode_ids = [_winner_episode_id(row) for row in kept_winners]
    _copy_quality_v4_artifacts(
        shard_dirs=shard_dirs,
        output=output,
        reset_episode_ids=kept_reset_episode_ids,
        winner_episode_ids=kept_winner_episode_ids,
    )
    _validate_quality_v4_artifacts(
        output,
        task=str(task),
        threshold_identity=reference_quality_v4_threshold_identity,
        reset_episode_ids=kept_reset_episode_ids,
        winner_episode_ids=kept_winner_episode_ids,
    )

    attempts_path = output / "attempts.jsonl"
    results_path = output / "reset_results.jsonl"
    winners_path = output / "winner_manifest.jsonl"
    reset_manifest_path = output / "reset_manifest.jsonl"
    export_state_path = output / "export_state.json"
    progress_path = output / "progress.json"
    progress = {
        "schema_version": PROGRESS_SCHEMA,
        "export_state_sha256": hashlib.sha256(
            export_state_path.read_bytes()
        ).hexdigest(),
        "started_unix_s": started_unix_s,
        "next_reset_index": max_reset + 1,
        "accepted_count": len(kept_winners),
        "candidate_attempt_count": len(kept_attempts),
        "budget_histogram": budget_histogram,
        "resume_count": resume_count,
        "recovery_events": recovery_events,
        "file_boundaries": {
            "attempts.jsonl": _file_boundary(attempts_path),
            "reset_results.jsonl": _file_boundary(results_path),
            "winner_manifest.jsonl": _file_boundary(winners_path),
        },
    }
    progress["payload_sha256"] = _payload_sha256(progress)
    _atomic_json(progress_path, progress)

    card = {
        "schema_version": EXPORT_SCHEMA,
        "status": "complete",
        "training_eligible": False,
        "training_eligibility_reason": "independent audit has not yet passed",
        "optimality_claim": "best-known under the frozen candidate/reset/budget contract",
        "task": task,
        "split": split,
        "manifest_seed": export_state["manifest_seed"],
        "accepted_target": args.accepted_episodes,
        "accepted_count": len(kept_winners),
        "attempted_reset_count": len(kept_results),
        "candidate_attempt_count": len(kept_attempts),
        "candidate_search_mode": export_state.get(
            "candidate_search_mode", FIRST_ELIGIBLE_SEARCH_MODE
        ),
        "candidate_pool_size": export_state.get("candidate_pool_size"),
        "initial_k": initial_k,
        "max_k": max_k,
        "budget_sequence": list(export_state["budget_sequence"]),
        "budget_histogram": budget_histogram,
        "selection_mode": export_state.get("selection_mode", LEGACY_SELECTION_MODE),
        "selection_contract": export_state.get(
            "selection_contract",
            _selection_contract(
                export_state.get("selection_mode", LEGACY_SELECTION_MODE)
            ),
        ),
        "planner_dominance": export_state.get("planner_dominance"),
        "candidate_schema_version": export_state.get("candidate_schema_version"),
        "evaluator_identity": export_state.get("evaluator_identity"),
        "compatibility_evidence": export_state.get("compatibility_evidence"),
        "calibration_evidence": export_state.get("calibration_evidence"),
        "candidate_release_manifest_sha256": export_state.get(
            "candidate_release_manifest_sha256"
        ),
        "candidate_release_provenance": export_state.get(
            "candidate_release_provenance"
        ),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "reset_manifest_sha256": hashlib.sha256(
            reset_manifest_path.read_bytes()
        ).hexdigest(),
        "export_state_sha256": progress["export_state_sha256"],
        "progress_sha256": hashlib.sha256(progress_path.read_bytes()).hexdigest(),
        "resume_count": resume_count,
        "recovery_events": recovery_events,
        "source_identity": source_identity,
        "quality_v2_threshold_identity": reference_quality_v2_threshold_identity,
        "quality_v4_threshold_identity": reference_quality_v4_threshold_identity,
        "image_size": image_size,
        "device": device,
        "started_unix_s": started_unix_s,
        "finished_unix_s": time.time(),
    }
    card["payload_sha256"] = _payload_sha256(card)
    _atomic_json(output / "dataset_card.json", card)
    checksum_count = _root_checksums(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "accepted": len(kept_winners),
                "attempted_resets": len(kept_results),
                "candidate_attempts": len(kept_attempts),
                "recovery_events": len(recovery_events),
                "checksum_entries": checksum_count,
                "dataset_card_payload_sha256": card["payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
