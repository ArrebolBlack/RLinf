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

"""Device-resident public-observation rollout storage for Direct PPO."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import (
    GpuNativeVisualPolicyObservation,
)

_CAMERAS = ("agentview", "robot0_eye_in_hand")


@dataclass(frozen=True)
class DirectPPORollout:
    observations: GpuNativeVisualPolicyObservation
    action: torch.Tensor
    raw_action: torch.Tensor
    old_log_prob: torch.Tensor
    value: torch.Tensor
    next_value: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    valid: torch.Tensor
    horizon: int
    num_envs: int


class DirectPPORolloutBuffer:
    """Preallocated CUDA storage with no privileged-observation field."""

    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        image_size: int,
        device: torch.device,
    ) -> None:
        if min(capacity, num_envs, image_size) < 1:
            raise ValueError("rollout buffer dimensions must be positive")
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.image_size = int(image_size)
        self.device = torch.device(device)
        prefix = (capacity, num_envs)
        image = (image_size, image_size)
        self.proprio = torch.empty((*prefix, 32), dtype=torch.float32, device=device)
        self.rgb = {
            camera: torch.empty((*prefix, *image, 3), dtype=torch.float32, device=device)
            for camera in _CAMERAS
        }
        self.depth_m = {
            camera: torch.empty((*prefix, *image), dtype=torch.float32, device=device)
            for camera in _CAMERAS
        }
        self.segmentation = {
            camera: torch.empty((*prefix, *image), dtype=torch.int32, device=device)
            for camera in _CAMERAS
        }
        self.action = torch.empty((*prefix, 7), dtype=torch.float32, device=device)
        self.raw_action = torch.empty((*prefix, 7), dtype=torch.float32, device=device)
        self.old_log_prob = torch.empty(prefix, dtype=torch.float32, device=device)
        self.value = torch.empty(prefix, dtype=torch.float32, device=device)
        self.next_value = torch.empty(prefix, dtype=torch.float32, device=device)
        self.reward = torch.empty(prefix, dtype=torch.float32, device=device)
        self.terminated = torch.empty(prefix, dtype=torch.bool, device=device)
        self.truncated = torch.empty(prefix, dtype=torch.bool, device=device)
        self.valid = torch.empty(prefix, dtype=torch.bool, device=device)
        self.cursor = 0
        self.pending = False

    def reset(self) -> None:
        if self.pending:
            raise RuntimeError("cannot reset a pending Direct PPO transition")
        self.cursor = 0

    def begin_step(
        self,
        *,
        observation: GpuNativeVisualPolicyObservation,
        action: torch.Tensor,
        raw_action: torch.Tensor,
        old_log_prob: torch.Tensor,
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if self.pending or self.cursor >= self.capacity:
            raise RuntimeError("Direct PPO rollout buffer overflow")
        if not isinstance(observation, GpuNativeVisualPolicyObservation):
            raise TypeError("rollout buffer accepts only public visual observations")
        row = self.cursor
        self.proprio[row].copy_(observation.proprio)
        for camera in _CAMERAS:
            self.rgb[camera][row].copy_(observation.rgb[camera])
            self.depth_m[camera][row].copy_(observation.depth_m[camera])
            self.segmentation[camera][row].copy_(observation.segmentation[camera])
        values: Mapping[str, tuple[torch.Tensor, torch.Tensor]] = {
            "action": (action, self.action[row]),
            "raw_action": (raw_action, self.raw_action[row]),
            "old_log_prob": (old_log_prob, self.old_log_prob[row]),
            "value": (value, self.value[row]),
            "valid": (valid.bool(), self.valid[row]),
        }
        for name, (source, destination) in values.items():
            if tuple(source.shape) != tuple(destination.shape) or source.device != self.device:
                raise ValueError(f"rollout {name} has the wrong CUDA schema")
            destination.copy_(source)
        self.pending = True

    def commit_step(
        self,
        *,
        next_value: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        if not self.pending:
            raise RuntimeError("Direct PPO rollout commit has no pending transition")
        row = self.cursor
        values: Mapping[str, tuple[torch.Tensor, torch.Tensor]] = {
            "next_value": (next_value, self.next_value[row]),
            "reward": (reward, self.reward[row]),
            "terminated": (terminated.bool(), self.terminated[row]),
            "truncated": (truncated.bool(), self.truncated[row]),
        }
        for name, (source, destination) in values.items():
            if tuple(source.shape) != tuple(destination.shape) or source.device != self.device:
                raise ValueError(f"rollout {name} has the wrong CUDA schema")
            destination.copy_(source)
        self.pending = False
        self.cursor += 1

    def abort_step(self) -> None:
        if not self.pending:
            raise RuntimeError("Direct PPO rollout abort has no pending transition")
        self.pending = False

    def view(self) -> DirectPPORollout:
        if self.pending or self.cursor < 1:
            raise RuntimeError("Direct PPO rollout buffer is empty")
        horizon = self.cursor
        observations = GpuNativeVisualPolicyObservation(
            proprio=self.proprio[:horizon],
            rgb=MappingProxyType(
                {camera: self.rgb[camera][:horizon] for camera in _CAMERAS}
            ),
            depth_m=MappingProxyType(
                {camera: self.depth_m[camera][:horizon] for camera in _CAMERAS}
            ),
            segmentation=MappingProxyType(
                {camera: self.segmentation[camera][:horizon] for camera in _CAMERAS}
            ),
        )
        return DirectPPORollout(
            observations=observations,
            action=self.action[:horizon],
            raw_action=self.raw_action[:horizon],
            old_log_prob=self.old_log_prob[:horizon],
            value=self.value[:horizon],
            next_value=self.next_value[:horizon],
            reward=self.reward[:horizon],
            terminated=self.terminated[:horizon],
            truncated=self.truncated[:horizon],
            valid=self.valid[:horizon],
            horizon=horizon,
            num_envs=self.num_envs,
        )


def select_rollout_observations(
    rollout: DirectPPORollout,
    indices: torch.Tensor,
) -> GpuNativeVisualPolicyObservation:
    """Gather flattened valid rows for one PPO minibatch on device."""

    total = rollout.horizon * rollout.num_envs
    proprio = rollout.observations.proprio.reshape(total, 32).index_select(0, indices)
    rgb = {
        camera: value.reshape(total, *value.shape[2:]).index_select(0, indices)
        for camera, value in rollout.observations.rgb.items()
    }
    depth_m = {
        camera: value.reshape(total, *value.shape[2:]).index_select(0, indices)
        for camera, value in rollout.observations.depth_m.items()
    }
    segmentation = {
        camera: value.reshape(total, *value.shape[2:]).index_select(0, indices)
        for camera, value in rollout.observations.segmentation.items()
    }
    return GpuNativeVisualPolicyObservation(
        proprio=proprio,
        rgb=rgb,
        depth_m=depth_m,
        segmentation=segmentation,
    )


__all__ = [
    "DirectPPORollout",
    "DirectPPORolloutBuffer",
    "select_rollout_observations",
]
