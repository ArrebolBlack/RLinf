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

from __future__ import annotations

import torch

from examples.embodiment.train_dynamic_benchmark_ppo import (
    _compute_gae,
    _ppo_policy_loss,
)


def test_gae_bootstraps_truncation_but_stops_cross_episode_recursion() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.4]])
    next_values = torch.tensor([[0.4], [0.3]])
    terminated = torch.tensor([[False], [False]])
    truncated = torch.tensor([[True], [False]])

    advantages, returns = _compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.8,
    )

    assert torch.allclose(advantages[0], torch.tensor([0.86]))
    assert torch.allclose(advantages[1], torch.tensor([1.87]))
    assert torch.allclose(returns, advantages + values)


def test_ppo_policy_loss_clips_harmful_large_ratio() -> None:
    old_log_prob = torch.zeros(2)
    new_log_prob = torch.log(torch.tensor([2.0, 0.5]))
    advantages = torch.tensor([1.0, -1.0])

    loss, approx_kl, clip_fraction = _ppo_policy_loss(
        new_log_prob,
        old_log_prob,
        advantages,
        clip_coef=0.2,
    )

    assert torch.allclose(loss, torch.tensor(-0.2))
    assert approx_kl > 0
    assert torch.allclose(clip_fraction, torch.tensor(1.0))
