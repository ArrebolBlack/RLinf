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

"""Run the bounded t1_xyz GPU Planner E0 smoke.

The online path is deliberately small and explicit:

    current host STATE observation -> ConveyorScriptedTeacher -> E7
    -> the same ``mjwarp_gpu_v1`` CUDA environment

The replay path constructs a second CUDA backend and replays the complete
action tape.  This file is an E0 engineering smoke, not a D32/MC64/H1024
result runner; it refuses more than one episode until Queue and Owner freeze a
manifest and matched reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

TASK_ID = "t1_xyz"
BACKEND_ID = "mjwarp_gpu_v1"
BASE_SE3_SOURCE_COMMIT = "31b07c52660c5fe0451d9d97f0494fe3d849ec10"
BASE_SE3_SOURCE_TREE = "b38a7fff7ef00c847dabef9daee8b67e5bc99169"
SE3_SOURCE_COMMIT = "c1af4de449b05043bd01dcf1c323399bfd22e7ef"
SE3_SOURCE_TREE = "f0effcd10b5bfa1b48b052255a9067cb7c1b7b7b"
QUALITY_SCHEMA_VERSION = "db0-episode-task-quality-v2"
QUALITY_EVALUATOR_ID = "gpu-planner-t1-xyz-e0-v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument(
        "--manifest-row",
        type=Path,
        help="Common-runner manifest row; its export_dir is used when --export-dir is omitted.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tape-output",
        type=Path,
        help="Optional full observation/action tape path; defaults beside --output.",
    )
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-se3-source-commit", default=SE3_SOURCE_COMMIT)
    parser.add_argument("--expected-se3-source-tree", default=SE3_SOURCE_TREE)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--observation-track",
        choices=("hybrid",),
        default="hybrid",
        help="Capture the GPU scene and wrist cameras alongside the Planner state.",
    )
    parser.add_argument(
        "--visual-gif",
        type=Path,
        help="Optional GPU-rendered side-by-side scene/wrist GIF; valid only with --observation-track hybrid.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="E0 is intentionally limited to one natural-termination episode.",
    )
    parser.add_argument(
        "--replay-policy",
        choices=("audit", "strict"),
        default="audit",
        help="Record fresh replay divergence as audit evidence by default; strict is diagnostic-only.",
    )
    parser.add_argument(
        "--planner-lookahead-s",
        type=float,
        help="Optional single-variable Planner override for the exploratory run.",
    )
    parser.add_argument(
        "--planner-contact-lookahead-s",
        type=float,
        help="Optional contact-phase Planner lookahead override for the exploratory run.",
    )
    parser.add_argument(
        "--planner-adaptive-lookahead-speed-gain-s-per-mps",
        type=float,
        help="Optional speed-adaptive Planner lookahead gain for the exploratory run.",
    )
    parser.add_argument(
        "--planner-lift-ramp-steps",
        type=int,
        help="Optional number of Planner lift-ramp control steps for the exploratory run.",
    )
    parser.add_argument(
        "--planner-lift-contact-loss-retry-steps",
        type=int,
        help="Optional Planner lift-contact-loss retry window for the exploratory run.",
    )
    return parser


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _observation_digest(observation: Any) -> str:
    """Hash the complete public observation, including both rendered cameras."""

    payload: dict[str, Any] = {
        "identity": {
            "episode_id": observation.episode_id,
            "task_id": observation.task_id,
            "physics_step": int(observation.physics_step),
            "control_step": int(observation.control_step),
            "policy_step": int(observation.policy_step),
            "time_s": float(observation.time_s),
        },
        "rgb": {name: _array_digest(value) for name, value in observation.rgb.items()},
        "depth_m": {name: _array_digest(value) for name, value in observation.depth_m.items()},
        "segmentation": {
            name: _array_digest(value) for name, value in observation.segmentation.items()
        },
        "proprio": {name: _array_digest(value) for name, value in observation.proprio.items()},
        "privileged": {
            name: _array_digest(value) for name, value in observation.privileged.items()
        },
        "events": [
            {
                "name": event.name,
                "physics_step": int(event.physics_step),
                "time_s": float(event.time_s),
            }
            for event in observation.events_since_last_observation
        ],
    }
    return _json_sha256(payload)


def _ledger_payload(ledger: Any) -> list[dict[str, Any]]:
    if ledger is None:
        return []
    rows = []
    for row in ledger.rows:
        rows.append(
            {
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
                "task_quality": (
                    None if row.task_quality is None else row.task_quality.to_dict()
                ),
            }
        )
    return rows


def _provenance_payload(provenance: Any) -> dict[str, Any]:
    names = (
        "backend_id",
        "implementation_version",
        "device_platform",
        "device_name",
        "device_ordinal",
        "git_commit",
        "git_tree",
        "physical_device_uuid",
        "physical_device_pci_bus_id",
        "physical_device_identity_source",
        "precision",
    )
    return {
        name: getattr(provenance, name)
        for name in names
        if getattr(provenance, name, None) is not None
    } | {"runtime_versions": dict(getattr(provenance, "runtime_versions", {}))}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_command(request: Any, values: Sequence[float], policy_step: int) -> Any:
    from se3_wam.benchmark.api import ActionCommand

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise ValueError("Planner E7 action must be a finite 7-vector")
    return ActionCommand(
        mode=request.action_mode,
        values=np.clip(array, -1.0, 1.0),
        policy_step=int(policy_step),
    )


def _replay(
    *,
    backend: Any,
    request: Any,
    observations: tuple[Any, ...],
    commands: tuple[Any, ...],
    outcomes: tuple[tuple[bool, bool, bool, str | None], ...],
    terminal_ledger: Any,
) -> dict[str, Any]:
    replay_backend = backend.new_replay_backend()
    try:
        replay_observation = replay_backend.reset((request,))[0]
        observation_digests = [_observation_digest(replay_observation)]
        expected_observation_digests = [_observation_digest(value) for value in observations]
        replay_outcomes: list[tuple[bool, bool, bool, str | None]] = []
        for command in commands:
            replay_command = _make_command(
                request,
                command.values,
                int(replay_backend.policy_steps()[0]),
            )
            result = replay_backend.step((replay_command,))[0]
            observation_digests.append(_observation_digest(result.observation))
            replay_outcomes.append(
                (
                    bool(result.terminated),
                    bool(result.truncated),
                    bool(result.success),
                    result.termination_reason,
                )
            )
        replay_ledger = _ledger_payload(replay_backend.last_terminal_ledger)
        expected_ledger = _ledger_payload(terminal_ledger)
        return {
            "passed": bool(
                observation_digests == expected_observation_digests
                and tuple(replay_outcomes) == outcomes
                and replay_ledger == expected_ledger
            ),
            "observation_tape_exact": observation_digests == expected_observation_digests,
            "outcomes_exact": tuple(replay_outcomes) == outcomes,
            "terminal_ledger_exact": replay_ledger == expected_ledger,
            "replay_observation_sha256": _json_sha256(observation_digests),
            "replay_ledger_sha256": _json_sha256(replay_ledger),
        }
    finally:
        replay_backend.close()


def _write_visual_gif(path: Path, observations: Sequence[Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("--visual-gif requires imageio in the GPU runtime") from exc
    frames = []
    for observation in observations:
        left = np.asarray(observation.rgb["agentview"], dtype=np.uint8)
        right = np.asarray(observation.rgb["robot0_eye_in_hand"], dtype=np.uint8)
        if left.shape[0] != right.shape[0]:
            raise RuntimeError("GPU scene and wrist frames have different heights")
        frames.append(np.concatenate((left, right), axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), extension=".gif", duration=0.05, loop=0)


def _write_tape_npz(
    path: Path,
    observations: Sequence[Any],
    commands: Sequence[Any],
) -> None:
    """Persist the complete host audit tape without replacing online planning."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if not observations or len(observations) != len(commands) + 1:
        raise ValueError("trajectory tape must contain one more observation than actions")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "episode_id": np.asarray([value.episode_id for value in observations]),
        "task_id": np.asarray([value.task_id for value in observations]),
        "physics_step": np.asarray([value.physics_step for value in observations], dtype=np.int64),
        "control_step": np.asarray([value.control_step for value in observations], dtype=np.int64),
        "policy_step": np.asarray([value.policy_step for value in observations], dtype=np.int64),
        "time_s": np.asarray([value.time_s for value in observations], dtype=np.float64),
        "action_values": np.stack(
            [np.asarray(value.values, dtype=np.float64) for value in commands]
        ),
        "action_policy_step": np.asarray(
            [value.policy_step for value in commands], dtype=np.int64
        ),
        "observation_digest": np.asarray(
            [_observation_digest(value) for value in observations]
        ),
        "event_names_json": np.asarray(
            [
                json.dumps(
                    [
                        {
                            "name": event.name,
                            "physics_step": int(event.physics_step),
                            "time_s": float(event.time_s),
                        }
                        for event in value.events_since_last_observation
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for value in observations
            ]
        ),
    }
    for group_name in ("rgb", "depth_m", "segmentation", "proprio", "privileged"):
        keys = sorted(getattr(observations[0], group_name))
        for key in keys:
            payload[f"{group_name}/{key}"] = np.stack(
                [np.asarray(getattr(value, group_name)[key]) for value in observations]
            )
    np.savez_compressed(path, **payload)


def main() -> None:
    args = _parser().parse_args()
    if args.episodes != 1:
        raise ValueError("GPU Planner E0 is limited to exactly one episode")
    if args.observation_track != "hybrid":
        raise ValueError("GPU Planner E0 requires hybrid GPU scene/wrist observations")
    if args.image_size < 64:
        raise ValueError("--image-size must be at least 64")
    if args.export_dir is None:
        if args.manifest_row is None:
            raise ValueError("one of --export-dir or --manifest-row is required")
        row = json.loads(args.manifest_row.read_text(encoding="utf-8"))
        export_dir = row.get("export_dir")
        if not isinstance(export_dir, str) or not export_dir:
            raise ValueError("manifest row must contain a non-empty export_dir")
        args.export_dir = Path(export_dir)
    if args.planner_lookahead_s is not None and (
        not np.isfinite(args.planner_lookahead_s) or args.planner_lookahead_s <= 0
    ):
        raise ValueError("--planner-lookahead-s must be finite and positive")
    if args.planner_contact_lookahead_s is not None and (
        not np.isfinite(args.planner_contact_lookahead_s)
        or args.planner_contact_lookahead_s < 0
    ):
        raise ValueError("--planner-contact-lookahead-s must be finite and nonnegative")
    if args.planner_adaptive_lookahead_speed_gain_s_per_mps is not None and (
        not np.isfinite(args.planner_adaptive_lookahead_speed_gain_s_per_mps)
        or args.planner_adaptive_lookahead_speed_gain_s_per_mps < 0
    ):
        raise ValueError(
            "--planner-adaptive-lookahead-speed-gain-s-per-mps must be finite and nonnegative"
        )
    if args.planner_lift_ramp_steps is not None and args.planner_lift_ramp_steps < 0:
        raise ValueError("--planner-lift-ramp-steps must be nonnegative")
    if (
        args.planner_lift_contact_loss_retry_steps is not None
        and args.planner_lift_contact_loss_retry_steps < 1
    ):
        raise ValueError("--planner-lift-contact-loss-retry-steps must be positive")
    if (
        args.expected_se3_source_commit != SE3_SOURCE_COMMIT
        or args.expected_se3_source_tree != SE3_SOURCE_TREE
    ):
        raise ValueError("E0 is pinned to the committed SE3-WAM candidate source commit/tree")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    tape_output = args.tape_output or args.output.with_name(f"{args.output.stem}.tape.npz")
    if tape_output.exists():
        raise FileExistsError(f"refusing to overwrite {tape_output}")
    if tape_output.suffix.lower() != ".npz":
        raise ValueError("--tape-output must use the .npz suffix")
    visual_gif = args.visual_gif or args.output.with_name(
        f"{args.output.stem}.scene-wrist.gif"
    )

    from se3_wam.benchmark.config import load_task_config
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    from rlinf.envs.dynamic_benchmark.gpu_backend import GpuNativeBackendEnv

    task_config = load_task_config(TASK_ID)
    horizon = int(task_config["clock"]["horizon_steps"])
    backend = GpuNativeBackendEnv(
        task_id=TASK_ID,
        num_envs=1,
        export_dir=str(args.export_dir),
        device_ordinal=args.device_ordinal,
        image_size=args.image_size,
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_se3_source_commit=args.expected_se3_source_commit,
        expected_se3_source_tree=args.expected_se3_source_tree,
        task_quality_schema_version=QUALITY_SCHEMA_VERSION,
        task_quality_evaluator_backend_id=QUALITY_EVALUATOR_ID,
        observation_track=args.observation_track,
    )
    try:
        if backend.backend_id != BACKEND_ID:
            raise RuntimeError(f"unexpected backend id: {backend.backend_id!r}")
        request = backend.next_request()
        observations = list(backend.reset((request,)))
        teacher, preparation = make_privileged_teacher(TASK_ID, request=request)
        planner_overrides: list[dict[str, Any]] = []
        if args.planner_lookahead_s is not None:
            if not hasattr(teacher, "lookahead_s"):
                raise RuntimeError("t1_xyz teacher does not expose lookahead_s")
            teacher.lookahead_s = float(args.planner_lookahead_s)
            planner_overrides.append(
                {
                    "variable": "lookahead_s",
                    "value": float(args.planner_lookahead_s),
                }
            )
        if args.planner_contact_lookahead_s is not None:
            if not hasattr(teacher, "contact_lookahead_s"):
                raise RuntimeError("t1_xyz teacher does not expose contact_lookahead_s")
            teacher.contact_lookahead_s = float(args.planner_contact_lookahead_s)
            planner_overrides.append(
                {
                    "variable": "contact_lookahead_s",
                    "value": float(args.planner_contact_lookahead_s),
                }
            )
        if args.planner_adaptive_lookahead_speed_gain_s_per_mps is not None:
            if not hasattr(teacher, "adaptive_lookahead_speed_gain_s_per_mps"):
                raise RuntimeError(
                    "t1_xyz teacher does not expose adaptive_lookahead_speed_gain_s_per_mps"
                )
            teacher.adaptive_lookahead_speed_gain_s_per_mps = float(
                args.planner_adaptive_lookahead_speed_gain_s_per_mps
            )
            planner_overrides.append(
                {
                    "variable": "adaptive_lookahead_speed_gain_s_per_mps",
                    "value": float(args.planner_adaptive_lookahead_speed_gain_s_per_mps),
                }
            )
        if args.planner_lift_ramp_steps is not None:
            if not hasattr(teacher, "lift_ramp_steps"):
                raise RuntimeError("t1_xyz teacher does not expose lift_ramp_steps")
            teacher.lift_ramp_steps = int(args.planner_lift_ramp_steps)
            planner_overrides.append(
                {
                    "variable": "lift_ramp_steps",
                    "value": int(args.planner_lift_ramp_steps),
                }
            )
        if args.planner_lift_contact_loss_retry_steps is not None:
            if not hasattr(teacher, "lift_contact_loss_retry_steps"):
                raise RuntimeError(
                    "t1_xyz teacher does not expose lift_contact_loss_retry_steps"
                )
            teacher.lift_contact_loss_retry_steps = int(
                args.planner_lift_contact_loss_retry_steps
            )
            planner_overrides.append(
                {
                    "variable": "lift_contact_loss_retry_steps",
                    "value": int(args.planner_lift_contact_loss_retry_steps),
                }
            )
        if planner_overrides:
            preparation = dict(preparation)
            preparation["planner_overrides"] = planner_overrides
            if len(planner_overrides) == 1:
                preparation["planner_override"] = planner_overrides[0]
        if hasattr(teacher, "reset"):
            teacher.reset()
        commands: list[Any] = []
        outcomes: list[tuple[bool, bool, bool, str | None]] = []
        latencies_s: list[float] = []
        observation = observations[0]
        result = None
        for _ in range(horizon):
            started = time.perf_counter()
            planner_action = teacher.act(observation)
            latencies_s.append(time.perf_counter() - started)
            command = _make_command(request, planner_action.values, observation.policy_step)
            result = backend.step((command,))[0]
            commands.append(command)
            observations.append(result.observation)
            outcomes.append(
                (
                    bool(result.terminated),
                    bool(result.truncated),
                    bool(result.success),
                    result.termination_reason,
                )
            )
            observation = result.observation
            if result.terminated or result.truncated:
                break
        if result is None or not (result.terminated or result.truncated):
            raise RuntimeError("E0 Planner did not reach natural termination within horizon")

        terminal_ledger = backend.last_terminal_ledger
        if terminal_ledger is None:
            raise RuntimeError("E0 Planner did not produce an exact-once terminal ledger")
        try:
            replay = _replay(
                backend=backend,
                request=request,
                observations=tuple(observations),
                commands=tuple(commands),
                outcomes=tuple(outcomes),
                terminal_ledger=terminal_ledger,
            )
        except Exception as exc:
            replay = {
                "passed": False,
                "observation_tape_exact": False,
                "outcomes_exact": False,
                "terminal_ledger_exact": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if args.replay_policy == "strict":
                raise
        if not replay["passed"] and args.replay_policy == "strict":
            raise RuntimeError(f"GPU Planner E0 replay failed: {json.dumps(replay, sort_keys=True)}")
        _write_tape_npz(tape_output, observations, commands)
        visual_error = None
        try:
            _write_visual_gif(visual_gif, observations)
        except RuntimeError as exc:
            # GIF encoding is an optional presentation artifact.  Keep the
            # GPU-rendered observation tape and online terminal result even
            # when the shared runtime does not provide an imageio encoder.
            visual_error = {"error_type": type(exc).__name__, "error": str(exc)}

        ledger_payload = _ledger_payload(terminal_ledger)
        action_tape = [np.asarray(command.values, dtype=np.float64).tolist() for command in commands]
        trajectory_tape = [_observation_digest(value) for value in observations]
        online_success = bool(result.success)
        if online_success and replay["passed"]:
            status = "passed"
        elif online_success:
            status = "completed_replay_audit_mismatch"
        else:
            status = "completed_online_failure"
        payload = {
            "schema_version": "gpu-planner-t1-xyz-e0-v2",
            "status": status,
            "task_id": TASK_ID,
            "job_phase": "e0",
            "online_planner": True,
            "planner_observation_source": "current_observation_each_control_step",
            "planner_compute": "cpu_allowed",
            "planner_override": (
                None
                if not planner_overrides
                else planner_overrides[0]
                if len(planner_overrides) == 1
                else {"variables": planner_overrides}
            ),
            "planner_overrides": planner_overrides or None,
            "frozen_action_replay": False,
            "cpu_physics_or_env_fallback": False,
            "quality": {
                "schema_version": QUALITY_SCHEMA_VERSION,
                "evaluator_backend_id": QUALITY_EVALUATOR_ID,
            },
            "backend_id": backend.backend_id,
            "provenance": _provenance_payload(backend.provenance),
            "source_pin": {
                "se3_commit": args.expected_se3_source_commit,
                "se3_tree": args.expected_se3_source_tree,
                "base_se3_commit": BASE_SE3_SOURCE_COMMIT,
                "base_se3_tree": BASE_SE3_SOURCE_TREE,
            },
            "export_dir": str(args.export_dir),
            "episode_id": request.episode_id,
            "control_steps": len(commands),
            "physics_steps": int(observations[-1].physics_step),
            "success": online_success,
            "online_success": online_success,
            "termination_reason": result.termination_reason,
            "action_tape": action_tape,
            "action_sha256": _json_sha256(action_tape),
            "trajectory_tape": trajectory_tape,
            "trajectory_sha256": _json_sha256(trajectory_tape),
            "tape_file": str(tape_output),
            "tape_sha256": _file_sha256(tape_output),
            "terminal_ledger": ledger_payload,
            "terminal_ledger_sha256": _json_sha256(ledger_payload),
            "replay": {"policy": args.replay_policy, **replay},
            "teacher_preparation": preparation,
            "planner_latency_s": {
                "count": len(latencies_s),
                "max": max(latencies_s),
                "mean": float(np.mean(latencies_s)),
            },
            "visual": {
                "observation_track": args.observation_track,
                "scene_camera": "agentview",
                "wrist_camera": "robot0_eye_in_hand",
                "gpu_rendered": True,
                "gif": str(visual_gif) if visual_error is None else None,
                "gif_sha256": _file_sha256(visual_gif) if visual_error is None else None,
                "gif_error": visual_error,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "control_steps": len(commands), "replay": replay}, sort_keys=True))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
