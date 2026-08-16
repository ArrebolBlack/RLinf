#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Materialize one accepted GPU Planner row as a canonical exploratory episode.

This command performs no simulation.  It consumes the raw observation tape
written by the strict A100 row runner, reconstructs the canonical typed trace,
and atomically publishes one ``db0-dataset-v0.6`` episode.  It deliberately does
not manufacture Qv4 evidence: the resulting episode is an exploratory canary
artifact and remains ineligible for a formal RLD3 release until the independent
Qv4 audit is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

RAW_TAPE_SCHEMA = "gpu-planner-t1-xyz-raw-tape-v1"
RECEIPT_SCHEMA = "gpu-planner-t1-xyz-exploratory-episode-v1"
CAMERAS = ("agentview", "robot0_eye_in_hand")
GROUPS = ("rgb", "depth_m", "segmentation", "proprio", "privileged")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def _load_strict_contract() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/envs/dynamic_benchmark/t1_xyz_strict_evidence.py"
    )
    name = "_t1_xyz_production_strict_evidence_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load strict evidence contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _scalar_text(value: np.ndarray) -> str:
    if value.shape != ():
        raise ValueError("raw tape scalar text field is not scalar")
    return str(value.item())


def _text_rows(value: np.ndarray, *, count: int, name: str) -> list[str]:
    if value.shape != (count,) or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"raw tape {name} shape or dtype is invalid")
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in value.tolist()
    ]


def _group_keys(files: tuple[str, ...], group: str) -> tuple[str, ...]:
    prefix = f"{group}/"
    keys = tuple(
        sorted(name[len(prefix) :] for name in files if name.startswith(prefix))
    )
    if not keys:
        raise ValueError(f"raw tape lacks {group} arrays")
    return keys


def _load_trace(
    *,
    manifest: Any,
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    tape_path: Path,
) -> Any:
    from se3_wam.benchmark.api import ActionCommand, ObservationBundle, StepResult
    from se3_wam.benchmark.contracts import EventRecord
    from se3_wam.benchmark.dataset import EpisodeTrace
    from se3_wam.benchmark.task_quality import EpisodeQualitySummary

    strict = _load_strict_contract()
    request, _ = strict.preflight_export_request(manifest, row)
    with np.load(tape_path, allow_pickle=False) as tape:
        files = tuple(tape.files)
        if _scalar_text(tape["schema_version"]) != RAW_TAPE_SCHEMA:
            raise ValueError("raw tape schema mismatch")
        action_values = np.asarray(tape["action_values"], dtype=np.float64)
        if action_values.ndim != 2 or action_values.shape[1:] != (7,):
            raise ValueError("raw tape action array is not Nx7")
        action_count = int(action_values.shape[0])
        observation_count = action_count + 1
        episode_ids = _text_rows(
            tape["episode_id"], count=observation_count, name="episode_id"
        )
        task_ids = _text_rows(tape["task_id"], count=observation_count, name="task_id")
        if set(episode_ids) != {request.episode_id} or set(task_ids) != {
            request.task_id
        }:
            raise ValueError("raw tape request identity drifted")
        group_keys = {group: _group_keys(files, group) for group in GROUPS}
        if group_keys["rgb"] != CAMERAS or group_keys["depth_m"] != CAMERAS:
            raise ValueError("raw tape camera inventory drifted")
        if group_keys["segmentation"] != CAMERAS:
            raise ValueError("raw tape segmentation camera inventory drifted")
        event_rows = _text_rows(
            tape["events_since_last_observation_json"],
            count=observation_count,
            name="events_since_last_observation_json",
        )
        fingerprints = _text_rows(
            tape["raw_observation_fingerprint_sha256"],
            count=observation_count,
            name="raw_observation_fingerprint_sha256",
        )
        observations = []
        for index in range(observation_count):
            events = tuple(
                EventRecord(**value) for value in json.loads(event_rows[index])
            )
            groups = {
                group: {key: np.asarray(tape[f"{group}/{key}"][index]) for key in keys}
                for group, keys in group_keys.items()
            }
            observation = ObservationBundle(
                episode_id=episode_ids[index],
                task_id=task_ids[index],
                physics_step=int(tape["physics_step"][index]),
                control_step=int(tape["control_step"][index]),
                policy_step=int(tape["policy_step"][index]),
                time_s=float(tape["time_s"][index]),
                rgb=groups["rgb"],
                depth_m=groups["depth_m"],
                segmentation=groups["segmentation"],
                proprio=groups["proprio"],
                privileged=groups["privileged"],
                events_since_last_observation=events,
            )
            if observation.fingerprint_sha256 != fingerprints[index]:
                raise ValueError(f"raw observation {index} fingerprint mismatch")
            observations.append(observation)
        policy_steps = np.asarray(tape["action_policy_step"], dtype=np.int64)
        if policy_steps.shape != (action_count,):
            raise ValueError("raw tape action policy-step array is invalid")
        actions = tuple(
            ActionCommand(
                mode=request.action_mode,
                values=action_values[index],
                policy_step=int(policy_steps[index]),
            )
            for index in range(action_count)
        )
        terminated = np.asarray(tape["step_result_terminated"], dtype=np.bool_)
        truncated = np.asarray(tape["step_result_truncated"], dtype=np.bool_)
        success = np.asarray(tape["step_result_success"], dtype=np.bool_)
        progress = np.asarray(
            tape["step_result_active_stage_progress"], dtype=np.float64
        )
        reasons = _text_rows(
            tape["step_result_termination_reason"],
            count=action_count,
            name="step_result_termination_reason",
        )
        result_fingerprints = _text_rows(
            tape["step_result_observation_fingerprint_sha256"],
            count=action_count,
            name="step_result_observation_fingerprint_sha256",
        )
        if any(
            array.shape != (action_count,)
            for array in (terminated, truncated, success, progress)
        ):
            raise ValueError("raw tape step-result arrays are misaligned")
        terminal_rows = result.get("terminal_ledger")
        if not isinstance(terminal_rows, list) or len(terminal_rows) != 1:
            raise ValueError("accepted result lacks one terminal ledger row")
        terminal_quality = terminal_rows[0].get("task_quality")
        step_results = []
        for index in range(action_count):
            if result_fingerprints[index] != observations[index + 1].fingerprint_sha256:
                raise ValueError(
                    f"step result {index} observation fingerprint mismatch"
                )
            fields: dict[str, Any] = {
                "observation": observations[index + 1],
                "terminated": bool(terminated[index]),
                "truncated": bool(truncated[index]),
                "success": bool(success[index]),
                "termination_reason": reasons[index] or None,
                "active_stage_progress": float(progress[index]),
            }
            if fields["success"]:
                if not isinstance(terminal_quality, Mapping):
                    raise ValueError("successful row lacks terminal task-quality")
                fields["task_quality"] = EpisodeQualitySummary.from_dict(
                    terminal_quality
                )
            step_results.append(StepResult(**fields))

    terminal_events = tuple(
        EventRecord(**value) for value in terminal_rows[0].get("events", [])
    )
    preparation = result.get("teacher_preparation")
    replay = result.get("replay")
    if not isinstance(preparation, Mapping) or not isinstance(replay, Mapping):
        raise ValueError("accepted result lacks Planner preparation or replay evidence")
    profile_id = preparation.get("planner_profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("accepted result lacks Planner profile identity")
    return EpisodeTrace(
        request=request,
        observations=tuple(observations),
        actions=actions,
        step_results=tuple(step_results),
        teacher_phases=tuple(f"online_planner/{profile_id}" for _ in actions),
        events=terminal_events,
        teacher_preparation=dict(preparation),
        replay_validation=dict(replay),
        quality_v4_validation=None,
    )


def _verify_episode_against_tape(
    *, episode_dir: Path, tape_path: Path, episode_id: str
) -> dict[str, Any]:
    metadata_path = episode_dir / "episode.json"
    trajectory_path = episode_dir / "trajectory.h5"
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if record.get("request", {}).get("episode_id") != episode_id:
        raise ValueError("published episode identity mismatch")
    with (
        np.load(tape_path, allow_pickle=False) as tape,
        h5py.File(trajectory_path, "r") as handle,
    ):
        for source, target in (
            ("physics_step", "observations/physics_step"),
            ("control_step", "observations/control_step"),
            ("policy_step", "observations/policy_step"),
            ("time_s", "observations/time_s"),
            ("action_values", "actions/values"),
            ("step_result_terminated", "step_results/terminated"),
            ("step_result_truncated", "step_results/truncated"),
            ("step_result_success", "step_results/success"),
            ("step_result_active_stage_progress", "step_results/active_stage_progress"),
        ):
            if not np.array_equal(np.asarray(tape[source]), np.asarray(handle[target])):
                raise ValueError(
                    f"published episode array differs from raw tape: {target}"
                )
        visual: dict[str, Any] = {}
        for group in GROUPS:
            for key in _group_keys(tuple(tape.files), group):
                source = np.asarray(tape[f"{group}/{key}"])
                target = np.asarray(handle[f"observations/{group}/{key}"])
                if not np.array_equal(source, target):
                    raise ValueError(
                        f"published episode array differs from raw tape: {group}/{key}"
                    )
                if group == "rgb":
                    if source.dtype != np.uint8 or source.shape[1:3] != (224, 224):
                        raise ValueError(f"published {key} RGB is not uint8 224x224")
                    if int(source.max()) == int(source.min()):
                        raise ValueError(f"published {key} RGB is degenerate")
                    visual[key] = {
                        "shape": list(source.shape),
                        "dtype": str(source.dtype),
                        "minimum": int(source.min()),
                        "maximum": int(source.max()),
                        "sha256": hashlib.sha256(
                            np.ascontiguousarray(source).tobytes()
                        ).hexdigest(),
                    }
    return {
        "record": record,
        "trajectory_sha256": _sha256(trajectory_path),
        "episode_json_sha256": _sha256(metadata_path),
        "visual": visual,
    }


def main() -> None:
    args = _parser().parse_args()
    strict = _load_strict_contract()
    manifest = strict.load_frozen_manifest(
        args.manifest, expected_phase="e0", verify_exports=True
    )
    row = manifest.row(args.row_index)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    strict.validate_result_for_row(result, manifest=manifest, row=row)
    evidence_export = result["evidence_export"]
    if _sha256(args.tape) != evidence_export.get("tape_file_sha256"):
        raise ValueError("raw tape digest differs from accepted result")
    alignment = evidence_export.get("raw_visual_alignment")
    if not isinstance(alignment, Mapping) or set(alignment) != set(CAMERAS):
        raise ValueError("accepted result lacks raw visual alignment evidence")
    for camera in CAMERAS:
        row_alignment = alignment[camera]
        if (
            not isinstance(row_alignment, Mapping)
            or row_alignment.get("review_frames_equal_raw_gpu_rgb") is not True
            or row_alignment.get("resolution") != [224, 224]
        ):
            raise ValueError(f"accepted result {camera} visual alignment failed")

    trace = _load_trace(
        manifest=manifest,
        row=row,
        result=result,
        tape_path=args.tape,
    )
    from se3_wam.benchmark.dataset import write_episode_atomic

    episode_dir = (
        args.dataset_root
        / "episodes"
        / trace.request.task_id
        / trace.request.split.value
        / trace.request.episode_id
    )
    if not episode_dir.exists():
        record = write_episode_atomic(args.dataset_root, trace)
    else:
        record = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    verification = _verify_episode_against_tape(
        episode_dir=episode_dir,
        tape_path=args.tape,
        episode_id=trace.request.episode_id,
    )
    if record != verification["record"]:
        raise ValueError("published episode record changed during verification")
    if record.get("metrics", {}).get("success") is not True:
        raise ValueError("published exploratory episode is not successful")
    if record.get("quality_v4_validation") is not None:
        raise ValueError("exploratory materializer must not manufacture Qv4 evidence")

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "accepted_exploratory_episode_materialized",
        "task_id": trace.request.task_id,
        "episode_id": trace.request.episode_id,
        "reset_seed": trace.request.seed,
        "source_identity_sha256": manifest.source_identity_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "result_sha256": _sha256(args.result),
        "raw_tape_sha256": _sha256(args.tape),
        "relative_episode_dir": episode_dir.relative_to(args.dataset_root).as_posix(),
        "trajectory_sha256": verification["trajectory_sha256"],
        "episode_json_sha256": verification["episode_json_sha256"],
        "record_sha256": record["record_sha256"],
        "visual": verification["visual"],
        "backend_id": result["backend_id"],
        "gpu_uuid": result["provenance"]["physical_device_uuid"],
        "cpu_physics_or_env_fallback": result["cpu_physics_or_env_fallback"],
        "semantic_fresh_replay_passed": result["replay"]["passed"],
        "one_shot_acquisition_passed": result["one_shot_acquisition_gate"]["passed"],
        "terminal_exact_once_passed": result["terminal_ledger_gate"]["passed"],
        "task_quality_schema_version": result["quality"]["schema_version"],
        "exploratory_canary": True,
        "formal_qualification": False,
        "formal_rld3_release": False,
        "quality_v4_complete": False,
        "training_use_scope": "exact14_exploratory_canary_after_all_tasks_reach_50",
    }
    if args.receipt.exists():
        existing = json.loads(args.receipt.read_text(encoding="utf-8"))
        if existing != receipt:
            raise ValueError("existing exploratory episode receipt drifted")
    else:
        _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
