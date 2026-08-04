#!/usr/bin/env python3
"""Merge sharded best-known trajectory exports into one sealed dataset root.

Each shard produced by ``export_dynamic_benchmark_optimal_trajectories.py
--shard-index N --shard-count M`` writes ``shard-NN/`` under the same parent
root. This entrypoint validates that every shard finished its slice, merges
``attempts.jsonl`` / ``reset_results.jsonl`` / ``winner_manifest.jsonl`` in
global reset order, keeps the first ``--accepted-episodes`` winners, copies the
corresponding episodes and lightweight tapes, then seals a dataset card and
``SHA256SUMS`` exactly like the single-process exporter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from examples.embodiment.export_dynamic_benchmark_optimal_trajectories import (
    EXPORT_SCHEMA,
    PROGRESS_SCHEMA,
    SELECTION_CONTRACT,
    _atomic_json,
    _file_boundary,
    _payload_sha256,
    _root_checksums,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-episodes", type=int, default=100)
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
    temporary.replace(path)


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


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    shard_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and re.fullmatch(r"shard-\d{2}", path.name)
    )
    if not shard_dirs:
        raise ValueError(f"no shard-* directories under {root}")

    all_results: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    all_winners: list[dict[str, Any]] = []
    started_unix_s: float | None = None
    for shard in shard_dirs:
        complete = shard / "shard_complete.json"
        if not complete.is_file():
            raise ValueError(f"{shard} is missing shard_complete.json")
        all_results.extend(_read_jsonl(shard / "reset_results.jsonl"))
        all_attempts.extend(_read_jsonl(shard / "attempts.jsonl"))
        all_winners.extend(_read_jsonl(shard / "winner_manifest.jsonl"))
        progress = json.loads((shard / "progress.json").read_text(encoding="utf-8"))
        if started_unix_s is None or progress.get("started_unix_s", float("inf")) < started_unix_s:
            started_unix_s = progress.get("started_unix_s")
    if started_unix_s is None:
        raise ValueError("no shard start time found")

    all_results.sort(key=lambda row: int(row["reset_index"]))
    reset_index_by_episode: dict[str, int] = {
        row["episode_id"]: int(row["reset_index"]) for row in all_results
    }
    all_winners.sort(key=lambda row: reset_index_by_episode[_winner_episode_id(row)])
    kept_winners = all_winners[: args.accepted_episodes]
    if len(kept_winners) < args.accepted_episodes:
        raise RuntimeError(
            f"only {len(kept_winners)}/{args.accepted_episodes} winners across shards"
        )
    max_reset = reset_index_by_episode[_winner_episode_id(kept_winners[-1])]
    kept_results = [row for row in all_results if int(row["reset_index"]) <= max_reset]
    kept_episodes = {row["episode_id"] for row in kept_results}
    kept_attempts = [row for row in all_attempts if row["episode_id"] in kept_episodes]
    kept_attempts.sort(key=lambda row: (_episode_number(row["episode_id"]), row["candidate_index"]))
    kept_winners.sort(key=lambda row: reset_index_by_episode[_winner_episode_id(row)])

    budget_histogram: dict[str, int] = {}
    for row in kept_results:
        key = str(row["budget_used"])
        budget_histogram[key] = budget_histogram.get(key, 0) + 1

    reference = shard_dirs[0]
    export_state = json.loads((reference / "export_state.json").read_text(encoding="utf-8"))
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
    shutil.copyfile(reference / "candidate_manifest.json", output / "candidate_manifest.json")
    shutil.copyfile(reference / "export_state.json", output / "export_state.json")
    shutil.copyfile(reference / "reset_manifest.jsonl", output / "reset_manifest.jsonl")

    _write_jsonl(output / "attempts.jsonl", kept_attempts)
    _write_jsonl(output / "reset_results.jsonl", kept_results)
    _write_jsonl(output / "winner_manifest.jsonl", kept_winners)

    for winner in kept_winners:
        episode_id = _winner_episode_id(winner)
        relative = winner.get("relative_episode_dir")
        target_rel = relative if isinstance(relative, str) else (
            f"episodes/{task}/{split}/{episode_id}"
        )
        for shard in shard_dirs:
            source = shard / target_rel if isinstance(relative, str) else (
                shard / "episodes" / task / split / episode_id
            )
            if source.exists():
                shutil.copytree(source, output / target_rel)
                break
        else:
            raise FileNotFoundError(f"winner episode {episode_id} not found in any shard")
    for episode_id in kept_episodes:
        for shard in shard_dirs:
            source = shard / "lightweight" / episode_id
            if source.exists():
                shutil.copytree(source, output / "lightweight" / episode_id)
                break
        else:
            raise FileNotFoundError(f"lightweight tape {episode_id} not found in any shard")

    attempts_path = output / "attempts.jsonl"
    results_path = output / "reset_results.jsonl"
    winners_path = output / "winner_manifest.jsonl"
    reset_manifest_path = output / "reset_manifest.jsonl"
    export_state_path = output / "export_state.json"
    progress_path = output / "progress.json"
    progress = {
        "schema_version": PROGRESS_SCHEMA,
        "export_state_sha256": hashlib.sha256(export_state_path.read_bytes()).hexdigest(),
        "started_unix_s": started_unix_s,
        "next_reset_index": max_reset + 1,
        "accepted_count": len(kept_winners),
        "candidate_attempt_count": len(kept_attempts),
        "budget_histogram": budget_histogram,
        "resume_count": 0,
        "recovery_events": [],
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
        "initial_k": initial_k,
        "max_k": max_k,
        "budget_sequence": list(export_state["budget_sequence"]),
        "budget_histogram": budget_histogram,
        "selection_contract": SELECTION_CONTRACT,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "reset_manifest_sha256": hashlib.sha256(reset_manifest_path.read_bytes()).hexdigest(),
        "export_state_sha256": progress["export_state_sha256"],
        "progress_sha256": hashlib.sha256(progress_path.read_bytes()).hexdigest(),
        "resume_count": 0,
        "recovery_events": [],
        "source_identity": source_identity,
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
                "checksum_entries": checksum_count,
                "dataset_card_payload_sha256": card["payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
