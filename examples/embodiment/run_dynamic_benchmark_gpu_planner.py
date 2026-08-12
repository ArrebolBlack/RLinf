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
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="t4_slider")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
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
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.horizon_control_steps < 1 or args.horizon_control_steps > 120:
        raise ValueError("--horizon-control-steps must lie in [1, 120]")
    from rlinf.envs.dynamic_benchmark.gpu_backend import GpuNativePlannerAdapter

    adapter = GpuNativePlannerAdapter(
        task_id=args.task,
        num_envs=args.num_envs,
        export_dir=str(args.export_dir),
        device_ordinal=args.device_ordinal,
        image_size=args.image_size,
        observation_track=args.observation_track,
        evaluator_backend_id=args.evaluator_backend_id,
        schema_version=args.schema_version,
    )
    try:
        requests = tuple(adapter.next_request() for _ in range(args.num_envs))
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
