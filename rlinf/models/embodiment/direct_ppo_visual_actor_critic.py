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

"""Compact visual actor-critic for GPUENV0 Direct PPO."""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch import nn
from torch.distributions import Normal

from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import (
    GpuNativeVisualPolicyObservation,
)

PUBLIC_CAMERAS = ("agentview", "robot0_eye_in_hand")
POLICY_INFORMATION_BOUNDARY = "proprio_plus_public_rgb_depth_segmentation"


class DirectPPOVisualActorCritic(nn.Module):
    """Shared two-camera encoder plus proprio actor/value heads.

    Each camera contributes RGB, normalized metric depth, a target mask, and an
    other-geometry mask.  The accepted input type contains no privileged state.
    """

    def __init__(self, *, hidden_size: int = 256, image_size: int = 64) -> None:
        super().__init__()
        if hidden_size < 32 or image_size < 32:
            raise ValueError("visual PPO model requires hidden_size/image_size >= 32")
        self.hidden_size = int(hidden_size)
        self.image_size = int(image_size)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 * len(PUBLIC_CAMERAS) + 64, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, 7)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((7,), -0.5))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _validate_observation(
        self, observation: GpuNativeVisualPolicyObservation
    ) -> int:
        if not isinstance(observation, GpuNativeVisualPolicyObservation):
            raise TypeError("Direct PPO policy accepts only the public visual dataclass")
        if observation.information_boundary != POLICY_INFORMATION_BOUNDARY:
            raise ValueError("Direct PPO policy information boundary drifted")
        proprio = observation.proprio
        if (
            not isinstance(proprio, torch.Tensor)
            or proprio.dtype != torch.float32
            or proprio.ndim != 2
            or proprio.shape[1] != 32
        ):
            raise ValueError("Direct PPO proprioception must be float32 [B,32]")
        batch_size = int(proprio.shape[0])
        for camera in PUBLIC_CAMERAS:
            expected_image = (batch_size, self.image_size, self.image_size, 3)
            expected_plane = (batch_size, self.image_size, self.image_size)
            rgb = observation.rgb[camera]
            depth = observation.depth_m[camera]
            segmentation = observation.segmentation[camera]
            if (
                not isinstance(rgb, torch.Tensor)
                or rgb.dtype != torch.float32
                or tuple(rgb.shape) != expected_image
                or rgb.device != proprio.device
            ):
                raise ValueError(f"{camera} RGB has the wrong policy schema")
            if (
                not isinstance(depth, torch.Tensor)
                or depth.dtype != torch.float32
                or tuple(depth.shape) != expected_plane
                or depth.device != proprio.device
            ):
                raise ValueError(f"{camera} depth has the wrong policy schema")
            if (
                not isinstance(segmentation, torch.Tensor)
                or segmentation.dtype != torch.int32
                or tuple(segmentation.shape) != expected_plane
                or segmentation.device != proprio.device
            ):
                raise ValueError(f"{camera} segmentation has the wrong policy schema")
        return batch_size

    @staticmethod
    def _camera_tensor(
        observation: GpuNativeVisualPolicyObservation,
        camera: str,
        selection: slice,
    ) -> torch.Tensor:
        rgb = observation.rgb[camera][selection].permute(0, 3, 1, 2)
        depth = observation.depth_m[camera][selection]
        depth = torch.log1p(torch.clamp(depth, 0.0, 10.0)) / math.log(11.0)
        segmentation = observation.segmentation[camera][selection]
        target = ((segmentation == 85) | (segmentation == 86)).to(torch.float32)
        other = ((segmentation != 0) & (segmentation != 85) & (segmentation != 86)).to(
            torch.float32
        )
        return torch.cat(
            (rgb, depth[:, None], target[:, None], other[:, None]), dim=1
        ).contiguous()

    @staticmethod
    def _batches(batch_size: int, microbatch_size: int | None) -> Iterator[slice]:
        effective = batch_size if microbatch_size is None else int(microbatch_size)
        if effective < 1:
            raise ValueError("encoder microbatch size must be positive")
        for start in range(0, batch_size, effective):
            yield slice(start, min(start + effective, batch_size))

    def encode(
        self,
        observation: GpuNativeVisualPolicyObservation,
        *,
        encoder_batch_size: int | None = None,
    ) -> torch.Tensor:
        batch_size = self._validate_observation(observation)
        rows = []
        for selection in self._batches(batch_size, encoder_batch_size):
            camera_features = [
                self.image_encoder(self._camera_tensor(observation, camera, selection))
                for camera in PUBLIC_CAMERAS
            ]
            proprio = self.proprio_encoder(observation.proprio[selection])
            rows.append(self.fusion(torch.cat((*camera_features, proprio), dim=-1)))
        return torch.cat(rows, dim=0)

    def distribution_and_value(
        self,
        observation: GpuNativeVisualPolicyObservation,
        *,
        encoder_batch_size: int | None = None,
    ) -> tuple[Normal, torch.Tensor]:
        hidden = self.encode(observation, encoder_batch_size=encoder_batch_size)
        mean = self.actor(hidden)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), self.critic(hidden).squeeze(-1)


__all__ = [
    "DirectPPOVisualActorCritic",
    "POLICY_INFORMATION_BOUNDARY",
    "PUBLIC_CAMERAS",
]
