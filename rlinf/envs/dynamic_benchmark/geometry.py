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

"""Current-state geometric helpers for Dynamic Benchmark expert arms.

Every function here is deterministic and uses only the current observation.
No future state, hidden event time, or offline label is ever consumed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_array(value: Any, *, context: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains NaN or Inf")
    return array


def quat_wxyz_to_matrix(quat_wxyz: Any) -> np.ndarray:
    """Return the 3x3 rotation matrix for a wxyz quaternion."""

    q = _as_array(quat_wxyz, context="quat_wxyz_to_matrix")
    if q.shape != (4,):
        raise ValueError(f"quaternion must be length 4, got {q.shape}")
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_wxyz_to_xyzw(pose_wxyz: Any) -> np.ndarray:
    """Reorder a [x,y,z,qw,qx,qy,qz] pose to [x,y,z,qx,qy,qz,qw]."""

    pose = _as_array(pose_wxyz, context="pose_wxyz_to_xyzw")
    if pose.shape != (7,):
        raise ValueError(f"pose must be length 7, got {pose.shape}")
    quat = pose[3:] / max(float(np.linalg.norm(pose[3:])), 1e-12)
    return np.concatenate([pose[:3], quat[[1, 2, 3, 0]]])


def pose_xyzw_to_wxyz(pose_xyzw: Any) -> np.ndarray:
    """Reorder a [x,y,z,qx,qy,qz,qw] pose to [x,y,z,qw,qx,qy,qz]."""

    pose = _as_array(pose_xyzw, context="pose_xyzw_to_wxyz")
    if pose.shape != (7,):
        raise ValueError(f"pose must be length 7, got {pose.shape}")
    quat = pose[3:] / max(float(np.linalg.norm(pose[3:])), 1e-12)
    return np.concatenate([pose[:3], quat[[3, 0, 1, 2]]])


def invert_pose_wxyz(pose_wxyz: Any) -> np.ndarray:
    """Invert a rigid transform given as [x,y,z,qw,qx,qy,qz]."""

    pose = _as_array(pose_wxyz, context="invert_pose_wxyz")
    if pose.shape != (7,):
        raise ValueError(f"pose must be length 7, got {pose.shape}")
    position = pose[:3]
    quat = pose[3:] / max(float(np.linalg.norm(pose[3:])), 1e-12)
    matrix = quat_wxyz_to_matrix(quat)
    inverse_position = -matrix.T @ position
    inverse_quat = np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)
    return np.concatenate([inverse_position, inverse_quat])


def compose_pose_wxyz(pose_a_wxyz: Any, pose_b_wxyz: Any) -> np.ndarray:
    """Compose A then B: T_AB = T_A * T_B."""

    a = _as_array(pose_a_wxyz, context="compose_pose_wxyz.a")
    b = _as_array(pose_b_wxyz, context="compose_pose_wxyz.b")
    if a.shape != (7,) or b.shape != (7,):
        raise ValueError("pose composition requires two length-7 poses")
    position = a[:3] + quat_wxyz_to_matrix(a[3:]) @ b[:3]
    qa = a[3:] / max(float(np.linalg.norm(a[3:])), 1e-12)
    qb = b[3:] / max(float(np.linalg.norm(b[3:])), 1e-12)
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
    eef_wxyz = pose_xyzw_to_wxyz(eef_pose)
    return compose_pose_wxyz(invert_pose_wxyz(eef_wxyz), object_pose)


def quaternion_geodesic_wxyz(quat_a_wxyz: Any, quat_b_wxyz: Any) -> float:
    """SO(3) geodesic distance in radians between two wxyz quaternions."""

    a = _as_array(quat_a_wxyz, context="geodesic.a")
    b = _as_array(quat_b_wxyz, context="geodesic.b")
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("geodesic requires two length-4 quaternions")
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.abs(np.dot(a, b)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def closing_axis_object_alignment_rad(
    object_pose_wxyz: Any,
    fingerpad_closing_axis_world: Any,
    object_axis_local: Any = (1.0, 0.0, 0.0),
) -> float:
    """Smallest angle (radians, [0, pi/2]) between the closing axis and a
    designated object axis, evaluated modulo pi (finger exchange equivalence).

    The default object axis is the local bar axis (+x) of the asymmetric T
    target used by t1_xyz/t1_so3.
    """

    object_pose = _as_array(object_pose_wxyz, context="alignment.object")
    axis_world = _as_array(
        fingerpad_closing_axis_world, context="alignment.closing_axis"
    )
    if object_pose.shape != (7,):
        raise ValueError("alignment requires a length-7 object pose")
    axis_world = np.asarray(axis_world, dtype=np.float64).reshape(-1)
    if axis_world.shape != (3,):
        raise ValueError("closing axis must be length 3")
    closing = axis_world / max(float(np.linalg.norm(axis_world)), 1e-12)
    local_axis = np.asarray(object_axis_local, dtype=np.float64).reshape(3)
    local_axis = local_axis / max(float(np.linalg.norm(local_axis)), 1e-12)
    object_axis_world = quat_wxyz_to_matrix(object_pose[3:]) @ local_axis
    object_axis_world = object_axis_world / max(
        float(np.linalg.norm(object_axis_world)), 1e-12
    )
    cosine = float(np.clip(np.abs(np.dot(closing, object_axis_world)), -1.0, 1.0))
    return float(np.arccos(cosine))


def quaternion_yaw_wxyz(quat_wxyz: Any) -> float:
    """Extrinsic z (yaw) angle in radians from a wxyz quaternion."""

    q = _as_array(quat_wxyz, context="quaternion_yaw")
    if q.shape != (4,):
        raise ValueError("quaternion must be length 4")
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


__all__ = [
    "closing_axis_object_alignment_rad",
    "compose_pose_wxyz",
    "invert_pose_wxyz",
    "object_in_eef_pose_wxyz",
    "pose_xyzw_to_wxyz",
    "pose_wxyz_to_xyzw",
    "quat_wxyz_to_matrix",
    "quaternion_geodesic_wxyz",
    "quaternion_yaw_wxyz",
]
