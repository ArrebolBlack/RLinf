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

"""Bounded closed-loop GPU Planner evidence runner for one frozen export.

This entry point is intentionally an evaluator, not a training job. It calls
the per-reset teacher factory on every lane, consumes the current observation
returned by mjwarp_gpu_v1 after every control interval, and writes an
append-only-friendly JSON report containing provenance, terminal rows, visual
component fingerprints, and the complete action tape identity. The separate
open-loop replay helper is never called here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="t4_slider")
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="one homogeneous frozen export (E0/diagnostic mode)",
    )
    parser.add_argument(
        "--export-manifest",
        type=Path,
        help="frozen lane-local export manifest for a D32 cohort",
    )
    parser.add_argument("--cohort-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--horizon-control-steps", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument(
        "--observation-track",
        choices=("hybrid", "visual"),
        default="hybrid",
    )
    parser.add_argument(
        "--evaluator-backend-id",
        default="gpup-plan-t4-slider-v1",
    )
    parser.add_argument(
        "--schema-version",
        default="db0-episode-task-quality-v1",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _load_cohort_manifest(
    path: Path,
    *,
    task_id: str,
    cohort_index: int,
    requested_num_envs: int | None,
) -> tuple[tuple[Path, ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Resolve and validate one frozen D32/MC64 cohort without resampling rows."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if isinstance(cohort_index, bool) or cohort_index < 0:
        raise ValueError("cohort-index must be a non-negative integer")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("export manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    supported_schemas = {
        "gpup-plan-t4-slider-d32-export-manifest-v1",
        "gpup-plan-t4-slider-mc64-export-manifest-v1",
    }
    if schema_version not in supported_schemas:
        raise ValueError("export manifest schema is not a frozen t4 D32/MC64 schema")
    if payload.get("task_id") != task_id:
        raise ValueError("export manifest task_id does not match the runner task")
    if payload.get("split") != "test_id":
        raise ValueError("t4 GPU Planner requires the registered test_id split")
    declared_payload_sha = payload.get("payload_sha256")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("payload_sha256", None)
    expected_payload_sha = hashlib.sha256(
        _canonical_json(unsigned_payload).encode("utf-8")
    ).hexdigest()
    if declared_payload_sha != expected_payload_sha:
        raise ValueError("export manifest payload_sha256 does not match its frozen content")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("export manifest rows must be a non-empty list")
    cohort_size = payload.get("cohort_size")
    cohort_count = payload.get("cohort_count")
    episode_count = payload.get("episode_count")
    if (
        isinstance(cohort_size, bool)
        or not isinstance(cohort_size, int)
        or cohort_size < 1
        or isinstance(cohort_count, bool)
        or not isinstance(cohort_count, int)
        or cohort_count < 1
        or isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count != len(rows)
        or cohort_count * cohort_size != episode_count
    ):
        raise ValueError("export manifest cohort cardinality is inconsistent")
    if requested_num_envs is not None and requested_num_envs != cohort_size:
        raise ValueError("--num-envs must equal the frozen export manifest cohort_size")
    if cohort_index >= cohort_count:
        raise ValueError("cohort-index is outside the frozen export manifest")
    candidate_indices = payload.get("candidate_indices")
    episode_ids = payload.get("episode_ids")
    if candidate_indices != [row.get("candidate_index") for row in rows]:
        raise ValueError("export manifest candidate index list drifted from rows")
    if episode_ids != [row.get("episode_id") for row in rows]:
        raise ValueError("export manifest episode ID list drifted from rows")
    if schema_version == "gpup-plan-t4-slider-d32-export-manifest-v1":
        expected_candidate_indices = list(range(1, episode_count + 1))
    else:
        disjoint_from = payload.get("disjoint_from")
        expected_candidate_indices = list(range(33, 97))
        if (
            episode_count != 64
            or cohort_size != 32
            or cohort_count != 2
            or not isinstance(disjoint_from, Mapping)
            or disjoint_from.get("manifest_sha256")
            != "af3639533b665a0c2e3475ddf16d28c85017f0c718012816b31b539e3530c2b4"
            or disjoint_from.get("candidate_indices") != list(range(1, 33))
        ):
            raise ValueError("MC64 manifest is not the frozen D32-disjoint 33..96 surface")
    if candidate_indices != expected_candidate_indices:
        raise ValueError("frozen t4 candidate indices drifted from the phase surface")
    selected = rows[cohort_index * cohort_size : (cohort_index + 1) * cohort_size]
    if len(selected) != cohort_size:
        raise ValueError("frozen export manifest cohort is incomplete")
    export_dirs: list[Path] = []
    identities: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for row in selected:
        if not isinstance(row, Mapping):
            raise ValueError("export manifest row must be an object")
        episode_id = row.get("episode_id")
        export_dir = row.get("export_dir")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("export manifest row episode_id is invalid")
        if episode_id in seen_episode_ids:
            raise ValueError("export manifest cohort repeats an episode_id")
        seen_episode_ids.add(episode_id)
        if not isinstance(export_dir, str) or not export_dir.strip():
            raise ValueError("export manifest row export_dir is invalid")
        resolved = Path(export_dir)
        if not resolved.is_absolute():
            resolved = (path.parent / resolved).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        expected_export_sha = row.get("export_sha256")
        expected_request_sha = row.get("request_json_sha256")
        if not isinstance(expected_export_sha, str) or not isinstance(
            expected_request_sha, str
        ):
            raise ValueError("export manifest row is missing export integrity digests")
        if _sha256(resolved / "SHA256SUMS") != expected_export_sha:
            raise ValueError(f"export SHA256SUMS drifted for episode {episode_id}")
        if _sha256(resolved / "request.json") != expected_request_sha:
            raise ValueError(f"request.json drifted for episode {episode_id}")
        export_dirs.append(resolved)
        identities.append(dict(row))
    context = {
        "schema_version": payload.get("schema_version"),
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "payload_sha256": declared_payload_sha,
        "manifest_seed": payload.get("manifest_seed"),
        "split": payload.get("split"),
        "episode_count": episode_count,
        "cohort_size": cohort_size,
        "cohort_count": cohort_count,
        "cohort_index": cohort_index,
        "episode_ids": [row["episode_id"] for row in identities],
    }
    return tuple(export_dirs), tuple(identities), context


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(
            {field.name: getattr(value, field.name) for field in fields(value)}
        )
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _terminal_row(row: Any) -> dict[str, Any]:
    quality = getattr(row, "task_quality", None)
    return {
        "lane": int(row.lane),
        "episode_id": row.episode_id,
        "task_id": row.task_id,
        "outcome": row.outcome.value,
        "terminated": bool(row.terminated),
        "truncated": bool(row.truncated),
        "success": bool(row.success),
        "termination_reason": row.termination_reason,
        "physics_step": int(row.physics_step),
        "control_step": int(row.control_step),
        "policy_step": int(row.policy_step),
        "completion": float(row.completion),
        "events": [
            {
                "name": event.name,
                "physics_step": int(event.physics_step),
                "time_s": float(event.time_s),
            }
            for event in row.events
        ],
        "task_quality": None
        if quality is None
        else _jsonable(quality.to_dict() if hasattr(quality, "to_dict") else quality),
    }


def _visual_fingerprint(observation: Any) -> dict[str, Any]:
    return {
        "episode_id": observation.episode_id,
        "physics_step": int(observation.physics_step),
        "control_step": int(observation.control_step),
        "policy_step": int(observation.policy_step),
        "component_sha256": dict(observation.component_sha256),
        "rgb_shapes": {
            name: list(value.shape) for name, value in observation.rgb.items()
        },
        "depth_shapes": {
            name: list(value.shape) for name, value in observation.depth_m.items()
        },
        "segmentation_shapes": {
            name: list(value.shape)
            for name, value in observation.segmentation.items()
        },
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_envs is not None and args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.horizon_control_steps < 1 or args.horizon_control_steps > 120:
        raise ValueError("--horizon-control-steps must lie in [1, 120]")
    from rlinf.envs.dynamic_benchmark.gpu_backend import GpuNativePlannerAdapter

    if args.export_manifest is not None and args.export_dir is not None:
        raise ValueError("provide exactly one of --export-dir or --export-manifest")
    manifest_context = None
    row_identities: tuple[dict[str, Any], ...] = ()
    if args.export_manifest is not None:
        export_dirs, row_identities, manifest_context = _load_cohort_manifest(
            args.export_manifest,
            task_id=args.task,
            cohort_index=args.cohort_index,
            requested_num_envs=args.num_envs,
        )
        num_envs = len(export_dirs)
        adapter = GpuNativePlannerAdapter(
            task_id=args.task,
            num_envs=num_envs,
            export_dirs=tuple(str(path) for path in export_dirs),
            device_ordinal=args.device_ordinal,
            image_size=args.image_size,
            observation_track=args.observation_track,
            evaluator_backend_id=args.evaluator_backend_id,
            schema_version=args.schema_version,
        )
    else:
        if args.export_dir is None:
            raise ValueError("one of --export-dir or --export-manifest is required")
        num_envs = args.num_envs if args.num_envs is not None else 1
        adapter = GpuNativePlannerAdapter(
            task_id=args.task,
            num_envs=num_envs,
            export_dir=str(args.export_dir),
            device_ordinal=args.device_ordinal,
            image_size=args.image_size,
            observation_track=args.observation_track,
            evaluator_backend_id=args.evaluator_backend_id,
            schema_version=args.schema_version,
        )
    try:
        requests = (
            adapter.frozen_requests
            if manifest_context is not None
            else tuple(adapter.next_request() for _ in range(num_envs))
        )
        if manifest_context is not None:
            if tuple(request.episode_id for request in requests) != tuple(
                manifest_context["episode_ids"]
            ):
                raise RuntimeError("GPU Planner export request IDs drifted from the frozen manifest")
            for request, row in zip(requests, row_identities, strict=True):
                if (
                    request.task_id != row.get("task_id")
                    or request.seed != row.get("seed")
                    or request.split.value != row.get("split")
                    or request.factors != row.get("factors")
                    or request.action_mode.value != row.get("action_mode")
                    or request.observation_track.value != row.get("observation_track")
                ):
                    raise RuntimeError(
                        f"GPU Planner export request identity drifted for {request.episode_id}"
                    )
        adapter.reset(requests)
        for _ in range(args.horizon_control_steps):
            adapter.step()
            if not adapter.active_mask.any():
                break
        if adapter.active_mask.any():
            raise RuntimeError(
                "GPU Planner did not produce a terminal row within the frozen horizon"
            )
        observations = adapter.observations
        assert observations is not None
        return {
            "status": "ok",
            "task_id": adapter.task_id,
            "backend_id": adapter.backend_id,
            "observation_track": adapter.observation_track.value,
            "closed_loop_planner": True,
            "manifest": manifest_context,
            "manifest_rows": list(row_identities),
            "replay": adapter.replay,
            "provenance": _jsonable(adapter.provenance),
            "teacher_metadata": _jsonable(adapter.teacher_metadata),
            "action_tape_sha256": adapter.action_tape_sha256,
            "action_steps_per_lane": [len(tape) for tape in adapter.action_tapes],
            "action_tapes": _jsonable(adapter.action_tapes),
            "terminal_rows": [_terminal_row(row) for row in adapter.terminal_rows],
            "visual_fingerprints": [
                _visual_fingerprint(observation) for observation in observations
            ],
        }
    finally:
        adapter.close()


def main() -> None:
    args = _parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = _run(args)
    except Exception as exc:
        report = {
            "status": "error",
            "closed_loop_planner": True,
            "replay": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
