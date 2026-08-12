# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from types import MappingProxyType

import pytest

torch = pytest.importorskip("torch")

from rlinf.data.direct_ppo_rollout_buffer import DirectPPORolloutBuffer
from rlinf.envs.dynamic_benchmark.direct_ppo_reward import (
    CONTRACT_SCHEMA_VERSION,
    DirectPPOReward,
    DirectPPORewardViolation,
    REWARD_COMPONENT_NAMES,
)
from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import (
    GpuNativePrivilegedRewardState,
    GpuNativeVisualPolicyObservation,
)
from rlinf.models.embodiment.direct_ppo_visual_actor_critic import (
    DirectPPOVisualActorCritic,
)


def _contract() -> dict:
    weights = {
        "terminal_success": 10.0,
        "approach_delta": 1.0,
        "grasp_delta": 0.5,
        "lift_delta": 2.0,
        "bilateral_contact_once": 0.5,
        "stable_lift_once": 1.0,
        "time": -0.002,
        "action_magnitude": -0.002,
        "action_jitter": -0.001,
        "terminal_failure": -4.0,
        "invalid_state": -10.0,
        "timeout": -1.0,
        "overflow": -10.0,
    }
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "reward": {
            "normalization": "none",
            "total_clip": [-12.0, 12.0],
            "parameters": {
                "approach_distance_scale_m": 0.3,
                "approach_delta_clip": 0.2,
                "grasp_delta_clip": 0.5,
                "lift_scale_m": 0.08,
                "lift_delta_clip": 0.2,
                "stable_lift_potential": 0.75,
                "stable_lift_max_linear_speed_m_s": 0.1,
            },
            "components": {
                name: {"weight": weights[name]} for name in REWARD_COMPONENT_NAMES
            },
        },
        "reward_hacking_checks": {
            "limits": {
                "high_return_failure": 8.0,
                "contact_without_lift_max_potential": 0.25,
                "contact_without_lift_min_return": 1.0,
                "jitter_mean_normalized": 0.35,
                "jitter_positive_shaping": 1.5,
                "minimum_success_control_steps": 4,
            }
        },
    }


def _reward_state(
    *,
    eef_x: float = 0.3,
    object_z: float = 0.0,
    contacts: tuple[float, float] = (0.0, 0.0),
    post_hold: float = 0.0,
) -> GpuNativePrivilegedRewardState:
    return GpuNativePrivilegedRewardState(
        eef_position_m=torch.tensor([[eef_x, 0.0, object_z]], dtype=torch.float32),
        object_position_m=torch.tensor([[0.0, 0.0, object_z]], dtype=torch.float32),
        object_linear_velocity_m_s=torch.zeros((1, 3), dtype=torch.float32),
        fingerpad_contact_flags=torch.tensor([contacts], dtype=torch.float32),
        post_hold_contact_valid=torch.tensor([post_hold], dtype=torch.float32),
        layout_sha256="unit-test-layout",
    )


def _signals(*, terminal: bool = False, success: bool = False, reason: int = 0):
    return {
        "terminated": torch.tensor([terminal]),
        "truncated": torch.tensor([False]),
        "success": torch.tensor([success]),
        "terminal_reason": torch.tensor([reason], dtype=torch.int32),
        "valid": torch.tensor([True]),
    }


def test_reward_potential_is_symmetric_and_terminal_summary_is_exact_once() -> None:
    reward = DirectPPOReward(_contract(), num_envs=1, device=torch.device("cpu"))
    reward.reset(_reward_state())
    action = torch.zeros((1, 7), dtype=torch.float32)
    closer = reward.step(state=_reward_state(eef_x=0.24), action=action, **_signals())
    farther = reward.step(state=_reward_state(), action=action, **_signals())
    assert closer.components["approach_delta"].item() == pytest.approx(0.2)
    assert farther.components["approach_delta"].item() == pytest.approx(-0.2)
    reward.step(
        state=_reward_state(),
        action=action,
        **_signals(terminal=True, reason=3),
    )
    summary = reward.consume_episode_summaries((0,))[0]
    assert summary["clipped_return"] == pytest.approx(summary["return"])
    assert "unclipped_return" in summary
    assert summary["components"]["terminal_failure"] == pytest.approx(-4.0)
    with pytest.raises(DirectPPORewardViolation, match="more than once"):
        reward.consume_episode_summaries((0,))


def test_reward_rejects_repeat_terminal_nan_and_early_success() -> None:
    action = torch.zeros((1, 7), dtype=torch.float32)
    reward = DirectPPOReward(_contract(), num_envs=1, device=torch.device("cpu"))
    reward.reset(_reward_state())
    reward.step(
        state=_reward_state(), action=action, **_signals(terminal=True, reason=3)
    )
    with pytest.raises(DirectPPORewardViolation, match="more than once"):
        reward.step(
            state=_reward_state(), action=action, **_signals(terminal=True, reason=3)
        )

    reward = DirectPPOReward(_contract(), num_envs=1, device=torch.device("cpu"))
    reward.reset(_reward_state())
    bad_action = action.clone()
    bad_action[0, 0] = float("nan")
    with pytest.raises(DirectPPORewardViolation, match="NaN or Inf"):
        reward.step(state=_reward_state(), action=bad_action, **_signals())

    reward = DirectPPOReward(_contract(), num_envs=1, device=torch.device("cpu"))
    reward.reset(_reward_state())
    with pytest.raises(DirectPPORewardViolation, match="stable-lift"):
        reward.step(
            state=_reward_state(
                object_z=0.08, contacts=(1.0, 1.0), post_hold=1.0
            ),
            action=action,
            **_signals(terminal=True, success=True, reason=1),
        )


def _public_observation(batch: int = 2, image_size: int = 32):
    cameras = ("agentview", "robot0_eye_in_hand")
    rgb = {
        camera: torch.rand((batch, image_size, image_size, 3), dtype=torch.float32)
        for camera in cameras
    }
    depth = {
        camera: torch.rand((batch, image_size, image_size), dtype=torch.float32)
        for camera in cameras
    }
    segmentation = {
        camera: torch.zeros((batch, image_size, image_size), dtype=torch.int32)
        for camera in cameras
    }
    segmentation["agentview"][:, 5:9, 5:9] = 85
    return GpuNativeVisualPolicyObservation(
        proprio=torch.rand((batch, 32), dtype=torch.float32),
        rgb=MappingProxyType(rgb),
        depth_m=MappingProxyType(depth),
        segmentation=MappingProxyType(segmentation),
    )


def test_visual_policy_type_excludes_privileged_state_and_backpropagates_images() -> None:
    observation = _public_observation()
    assert set(observation.__dataclass_fields__) == {
        "proprio",
        "rgb",
        "depth_m",
        "segmentation",
        "information_boundary",
    }
    model = DirectPPOVisualActorCritic(hidden_size=64, image_size=32)
    distribution, value = model.distribution_and_value(
        observation, encoder_batch_size=1
    )
    loss = distribution.mean.square().mean() + value.square().mean()
    loss.backward()
    visual_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.image_encoder.parameters()
        if parameter.grad is not None
    )
    assert visual_gradient > 0.0


def test_rollout_buffer_copies_public_observation_before_render_reuse() -> None:
    observation = _public_observation(batch=1)
    buffer = DirectPPORolloutBuffer(
        capacity=1, num_envs=1, image_size=32, device=torch.device("cpu")
    )
    original = observation.rgb["agentview"].clone()
    action = torch.zeros((1, 7), dtype=torch.float32)
    buffer.begin_step(
        observation=observation,
        action=action,
        raw_action=action,
        old_log_prob=torch.zeros(1),
        value=torch.zeros(1),
        valid=torch.ones(1, dtype=torch.bool),
    )
    observation.rgb["agentview"].zero_()
    buffer.commit_step(
        next_value=torch.zeros(1),
        reward=torch.zeros(1),
        terminated=torch.zeros(1, dtype=torch.bool),
        truncated=torch.zeros(1, dtype=torch.bool),
    )
    assert torch.equal(buffer.view().observations.rgb["agentview"][0], original)

