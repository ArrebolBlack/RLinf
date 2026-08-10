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

"""Current-state geometric helpers for the RLOPT shared infrastructure.

Every function is deterministic and consumes only the current (or a previous,
already observed) state.  No future state, hidden event time, or offline label
is ever used.  All quaternion conventions are explicit: world poses use wxyz
([x, y, z, qw, qx, qy, qz]) and the end-effector pose uses xyzw
([x, y, z, qx, qy, qz, qw]) to match the frozen Dynamic Benchmark schema.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_array(value: Any, *, context: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains NaN or Inf")
    return array


def _normalized_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat_wxyz))
    if norm <= 0.0:
        raise ValueError("quaternion has zero norm")
    return quat_wxyz / norm


def quat_wxyz_to_matrix(quat_wxyz: Any) -> np.ndarray:
    """Return the 3x3 rotation matrix for a wxyz quaternion."""

    q = _as_array(quat_wxyz, context="quat_wxyz_to_matrix")
    if q.shape != (4,):
        raise ValueError(f"quaternion must be length 4, got {q.shape}")
    w, x, y, z = _normalized_quat(q)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_wxyz(quat_xyzw: Any) -> np.ndarray:
    """Reorder [qx, qy, qz, qw] to [qw, qx, qy, qz]."""

    q = _as_array(quat_xyzw, context="quat_xyzw_to_wxyz")
    if q.shape != (4,):
        raise ValueError(f"quaternion must be length 4, got {q.shape}")
    return _normalized_quat(q[[3, 0, 1, 2]])


def pose_xyzw_to_wxyz(pose_xyzw: Any) -> np.ndarray:
    """Reorder an xyzw pose to [x, y, z, qw, qx, qy, qz]."""

    pose = _as_array(pose_xyzw, context="pose_xyzw_to_wxyz")
    if pose.shape != (7,):
        raise ValueError(f"pose must be length 7, got {pose.shape}")
    return np.concatenate([pose[:3], quat_xyzw_to_wxyz(pose[3:])])


def pose_wxyz_to_xyzw(pose_wxyz: Any) -> np.ndarray:
    """Reorder a wxyz pose to [x, y, z, qx, qy, qz, qw]."""

    pose = _as_array(pose_wxyz, context="pose_wxyz_to_xyzw")
    if pose.shape != (7,):
        raise ValueError(f"pose must be length 7, got {pose.shape}")
    quat = _normalized_quat(pose[3:])
    return np.concatenate([pose[:3], quat[[1, 2, 3, 0]]])


def invert_pose_wxyz(pose_wxyz: Any) -> np.ndarray:
    """Invert a rigid transform given as [x, y, z, qw, qx, qy, qz]."""

    pose = _as_array(pose_wxyz, context="invert_pose_wxyz")
    if pose.shape != (7,):
        raise ValueError(f"pose must be length 7, got {pose.shape}")
    quat = _normalized_quat(pose[3:])
    matrix = quat_wxyz_to_matrix(quat)
    inverse_position = -matrix.T @ pose[:3]
    inverse_quat = np.asarray([quat[0], -quat[1], -quat[2], -quat[3]])
    return np.concatenate([inverse_position, inverse_quat])


def compose_pose_wxyz(pose_a_wxyz: Any, pose_b_wxyz: Any) -> np.ndarray:
    """Compose A then B: T_AB = T_A * T_B."""

    a = _as_array(pose_a_wxyz, context="compose_pose_wxyz.a")
    b = _as_array(pose_b_wxyz, context="compose_pose_wxyz.b")
    if a.shape != (7,) or b.shape != (7,):
        raise ValueError("pose composition requires two length-7 poses")
    position = a[:3] + quat_wxyz_to_matrix(a[3:]) @ b[:3]
    qa = _normalized_quat(a[3:])
    qb = _normalized_quat(b[3:])
    qw = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3]
    qx = qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2]
    qy = qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1]
    qz = qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0]
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    if quat[0] < 0.0:
        quat = -quat
    return np.concatenate([position, quat])


def object_in_eef_pose_wxyz(
    object_pose_wxyz: Any,
    eef_pose_xyzw: Any,
) -> np.ndarray:
    """Object pose expressed in the end-effector frame (wxyz)."""

    object_pose = _as_array(object_pose_wxyz, context="object_in_eef.object")
    eef_pose = _as_array(eef_pose_xyzw, context="object_in_eef.eef")
    if object_pose.shape != (7,) or eef_pose.shape != (7,):
        raise ValueError("object_in_eef requires two length-7 poses")
    return compose_pose_wxyz(invert_pose_wxyz(pose_xyzw_to_wxyz(eef_pose)), object_pose)


def quaternion_geodesic_wxyz(quat_a_wxyz: Any, quat_b_wxyz: Any) -> float:
    """SO(3) geodesic distance in radians between two wxyz quaternions."""

    a = _as_array(quat_a_wxyz, context="geodesic.a")
    b = _as_array(quat_b_wxyz, context="geodesic.b")
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("geodesic requires two length-4 quaternions")
    a = _normalized_quat(a)
    b = _normalized_quat(b)
    dot = float(np.abs(np.dot(a, b)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def closing_axis_object_alignment_rad(
    object_pose_wxyz: Any,
    fingerpad_closing_axis_world: Any,
    object_axis_local: Any = (1.0, 0.0, 0.0),
) -> float:
    """Return the closing/object-axis angle modulo finger exchange."""

    object_pose = _as_array(object_pose_wxyz, context="alignment.object")
    closing_axis = _as_array(
        fingerpad_closing_axis_world, context="alignment.closing_axis"
    ).reshape(-1)
    local_axis = _as_array(object_axis_local, context="alignment.object_axis").reshape(
        -1
    )
    if object_pose.shape != (7,):
        raise ValueError("alignment requires a length-7 object pose")
    if closing_axis.shape != (3,) or local_axis.shape != (3,):
        raise ValueError("alignment axes must be length 3")
    closing_norm = float(np.linalg.norm(closing_axis))
    local_norm = float(np.linalg.norm(local_axis))
    if closing_norm <= 0.0 or local_norm <= 0.0:
        raise ValueError("alignment axes must have non-zero norm")
    closing_axis = closing_axis / closing_norm
    local_axis = local_axis / local_norm
    object_axis_world = quat_wxyz_to_matrix(object_pose[3:]) @ local_axis
    cosine = float(np.clip(np.abs(np.dot(closing_axis, object_axis_world)), -1.0, 1.0))
    return float(np.arccos(cosine))


def quaternion_yaw_wxyz(quat_wxyz: Any) -> float:
    """Return the extrinsic-z yaw angle of a wxyz quaternion."""

    w, x, y, z = _normalized_quat(
        _as_array(quat_wxyz, context="quaternion_yaw_wxyz")
    )
    return float(
        np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )


def rotation_vector_world(
    prev_pose_wxyz: Any,
    cur_pose_wxyz: Any,
) -> np.ndarray:
    """World-frame rotation vector R_prev^T * R_cur -> axis-angle (radians)."""

    prev = _as_array(prev_pose_wxyz, context="rotation_vector.prev")
    cur = _as_array(cur_pose_wxyz, context="rotation_vector.cur")
    if prev.shape != (7,) or cur.shape != (7,):
        raise ValueError("rotation vector requires two length-7 poses")
    r_prev = quat_wxyz_to_matrix(prev[3:])
    r_cur = quat_wxyz_to_matrix(cur[3:])
    relative = r_prev.T @ r_cur
    cos_angle = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    world_axis = r_prev @ (axis / axis_norm)
    return world_axis * angle


def eef_velocity_world(
    prev_pose_xyzw: Any,
    cur_pose_xyzw: Any,
    dt_s: float,
) -> np.ndarray:
    """Finite-difference end-effector twist in the world frame.

    Returns a length-6 vector [linear_xyz, angular_xyz] in SI units.  Only the
    previous and current observations are used, so the result is causal.
    """

    prev = pose_xyzw_to_wxyz(prev_pose_xyzw)
    cur = pose_xyzw_to_wxyz(cur_pose_xyzw)
    dt = float(dt_s)
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt_s must be finite and positive")
    linear = (cur[:3] - prev[:3]) / dt
    angular = rotation_vector_world(prev, cur) / dt
    return np.concatenate([linear, angular])


__all__ = [
    "closing_axis_object_alignment_rad",
    "compose_pose_wxyz",
    "eef_velocity_world",
    "invert_pose_wxyz",
    "object_in_eef_pose_wxyz",
    "pose_xyzw_to_wxyz",
    "pose_wxyz_to_xyzw",
    "quat_wxyz_to_matrix",
    "quat_xyzw_to_wxyz",
    "quaternion_geodesic_wxyz",
    "quaternion_yaw_wxyz",
    "rotation_vector_world",
]
