# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Strict current-observation Planner seam for GPU-native T1-SO3 E0.

The online Planner may use CPU/NumPy, but every decision consumes the current
STATE observation materialized by ``mjwarp_gpu_v1``. Physics, terminal
evaluation, task quality, and review rendering remain on the CUDA environment.
Fresh replay consumes only a finalized natural-termination tape and never calls
the Planner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from rlinf.envs.dynamic_benchmark.p0_grasp_planner import (
    E7_ACTION_WIDTH,
    GPU_NATIVE_BACKEND_ID,
    PHYSICS_STEPS_PER_CONTROL,
    ActionTrajectoryTape,
    P0GraspPlannerError,
    PlannerTapeEntry,
    PlannerTapeIdentity,
    ReplayReceipt,
    ReplayStepReceipt,
    ReviewFrameRecord,
    causal_observation_fingerprint,
    observation_components,
)
from rlinf.envs.dynamic_benchmark.p0_grasp_planner import (
    CurrentStatePlannerAdapter as _CapturePlannerAdapter,
)
from rlinf.envs.dynamic_benchmark.p0_grasp_planner import (
    replay_action_trajectory as _replay_action_trajectory,
)

T1_SO3_PLANNER_TAPE_SCHEMA_VERSION = "se3-wam-t1-so3-planner-tape-v2"
T1So3PlannerError = P0GraspPlannerError


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _finite_vector(name: str, value: Any, width: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (width,) or not np.all(np.isfinite(vector)):
        raise T1So3PlannerError(f"{name} must be one finite {width}-vector")
    return vector


def _unit_quaternion(name: str, value: Any) -> np.ndarray:
    quaternion = _finite_vector(name, value, 4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2.0e-5):
        raise T1So3PlannerError(f"{name} must be unit length")
    return quaternion / norm


def _rotation_from_wxyz(quaternion: Any) -> np.ndarray:
    w, x, y, z = _unit_quaternion("object quaternion", quaternion)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _rotation_from_xyzw(quaternion: Any) -> np.ndarray:
    x, y, z, w = _unit_quaternion("EEF quaternion", quaternion)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _yaw_from_wxyz(quaternion: Any) -> float:
    w, x, y, z = _unit_quaternion("object quaternion", quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _phase_lookahead_s(planner: Any, phase: str | None) -> float:
    attribute = (
        "lookahead_s"
        if phase in {"track", "descend"}
        else "contact_lookahead_s"
        if phase in {"close", "settle"}
        else None
    )
    if attribute is None:
        return 0.0
    value = float(getattr(planner, attribute, 0.0))
    if not math.isfinite(value) or value < 0.0:
        raise T1So3PlannerError(f"Planner {attribute} must be finite and non-negative")
    return value


def planner_step_diagnostics(observation: Any, planner: Any) -> Mapping[str, Any]:
    """Record T1-SO3 stage, rotation, modulo-pi axis, and Planner audit state."""

    privileged = getattr(observation, "privileged", None)
    if not isinstance(privileged, Mapping):
        raise T1So3PlannerError(
            "T1-SO3 diagnostics require the current privileged audit mapping"
        )
    required = (
        "object_pose_wxyz",
        "eef_pose_xyzw",
        "fingerpad_closing_axis_world",
        "object_twist_world",
    )
    missing = tuple(name for name in required if name not in privileged)
    if missing:
        raise T1So3PlannerError(
            f"T1-SO3 diagnostics are missing privileged fields: {missing}"
        )
    object_pose = _finite_vector("object_pose_wxyz", privileged[required[0]], 7)
    eef_pose = _finite_vector("eef_pose_xyzw", privileged[required[1]], 7)
    closing_axis = _finite_vector(
        "fingerpad_closing_axis_world",
        privileged[required[2]],
        3,
    )
    object_twist = _finite_vector("object_twist_world", privileged[required[3]], 6)
    closing_norm = float(np.linalg.norm(closing_axis))
    if not math.isclose(closing_norm, 1.0, rel_tol=0.0, abs_tol=2.0e-5):
        raise T1So3PlannerError("fingerpad closing axis must be unit length")

    phase_value = _enum_value(getattr(planner, "phase", None))
    phase = None if phase_value is None else str(phase_value)
    orientation_mode = _enum_value(getattr(planner, "orientation_mode", None))
    object_yaw = _yaw_from_wxyz(object_pose[3:])
    future_yaw = None
    if orientation_mode == "track_object":
        future_yaw = object_yaw + _phase_lookahead_s(planner, phase) * float(
            object_twist[5]
        )
    closing_yaw = math.atan2(float(closing_axis[1]), float(closing_axis[0]))
    teacher_offset = getattr(planner, "grasp_axis_offset_rad", None)
    target_yaw = None
    axis_error = None
    if teacher_offset is not None:
        teacher_offset = float(teacher_offset)
        if not math.isfinite(teacher_offset):
            raise T1So3PlannerError("Planner grasp_axis_offset_rad must be finite")
        target_yaw = object_yaw + teacher_offset
        directed_error = target_yaw - closing_yaw
        axis_error = 0.5 * math.atan2(
            math.sin(2.0 * directed_error),
            math.cos(2.0 * directed_error),
        )
    planner_audit = getattr(planner, "planner_audit_snapshot", None)
    planner_audit_payload = planner_audit() if callable(planner_audit) else {}
    if not isinstance(planner_audit_payload, Mapping):
        raise T1So3PlannerError("Planner audit snapshot must be a mapping")

    object_rotation = _rotation_from_wxyz(object_pose[3:])
    grasp_rotation = _rotation_from_xyzw(eef_pose[3:])
    return MappingProxyType(
        {
            "phase": phase,
            "orientation_mode": orientation_mode,
            "planner_audit": dict(planner_audit_payload),
            "object_rotation_wxyz": tuple(float(value) for value in object_pose[3:]),
            "eef_rotation_xyzw": tuple(float(value) for value in eef_pose[3:]),
            "object_rotation_world": tuple(
                float(value) for value in object_rotation.reshape(-1)
            ),
            "grasp_frame_rotation_world": tuple(
                float(value) for value in grasp_rotation.reshape(-1)
            ),
            "object_yaw_rad": float(object_yaw),
            "future_yaw_rad": None if future_yaw is None else float(future_yaw),
            "target_rotation_yaw_rad": (
                None if target_yaw is None else float(target_yaw)
            ),
            "closing_axis_yaw_rad": float(closing_yaw),
            "modulo_pi_axis_error_rad": (
                None if axis_error is None else float(axis_error)
            ),
            "object_yaw_rate_rad_s": float(object_twist[5]),
            "events": tuple(
                str(getattr(event, "name", event))
                for event in getattr(
                    observation,
                    "events_since_last_observation",
                    (),
                )
            ),
        }
    )


class CurrentStatePlannerAdapter(_CapturePlannerAdapter):
    """Bind the shared strict B=1 capture seam to frozen T1-SO3 identity."""

    def __init__(self, backend: Any, planner: Any) -> None:
        super().__init__(
            backend,
            planner,
            task_id="t1_so3",
            diagnostics=planner_step_diagnostics,
        )


def replay_action_trajectory(
    backend: Any,
    tape: ActionTrajectoryTape,
) -> ReplayReceipt:
    """Strictly replay a finalized T1-SO3 tape on one fresh CUDA backend."""

    if not isinstance(tape, ActionTrajectoryTape) or tape.identity.task_id != "t1_so3":
        raise T1So3PlannerError("T1-SO3 replay requires a t1_so3 action tape")
    return _replay_action_trajectory(backend, tape)


__all__ = [
    "ActionTrajectoryTape",
    "CurrentStatePlannerAdapter",
    "E7_ACTION_WIDTH",
    "GPU_NATIVE_BACKEND_ID",
    "PHYSICS_STEPS_PER_CONTROL",
    "PlannerTapeEntry",
    "PlannerTapeIdentity",
    "ReplayReceipt",
    "ReplayStepReceipt",
    "ReviewFrameRecord",
    "T1_SO3_PLANNER_TAPE_SCHEMA_VERSION",
    "T1So3PlannerError",
    "causal_observation_fingerprint",
    "observation_components",
    "planner_step_diagnostics",
    "replay_action_trajectory",
]
