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

from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    TransitionReplay,
    _score,
)


def test_running_normalizer_standardizes_values_but_preserves_mask() -> None:
    normalizer = RunningNormalizer(dimension=4, mask_dim=2)
    normalizer.update(
        torch.tensor(
            [
                [1.0, 10.0, 0.0, 1.0],
                [3.0, 14.0, 1.0, 0.0],
            ]
        )
    )

    normalized = normalizer.normalize(
        torch.tensor([[2.0, 12.0, 1.0, 0.0]]),
        torch.device("cpu"),
    )

    assert torch.allclose(normalized[:, :2], torch.zeros(1, 2))
    assert torch.equal(normalized[:, 2:], torch.tensor([[1.0, 0.0]]))


def test_transition_replay_round_trip_preserves_data_cursor_and_sampling_rng() -> None:
    replay = TransitionReplay(capacity=4, state_dim=2, seed=17)
    states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    replay.add(
        states,
        torch.arange(42, dtype=torch.float32).reshape(6, 7),
        torch.arange(6, dtype=torch.float32),
        states + 1.0,
        torch.tensor([False, False, True, False, False, True]),
        torch.zeros(6, dtype=torch.bool),
    )
    checkpoint = replay.state_dict()
    restored = TransitionReplay(capacity=4, state_dim=2, seed=999)
    restored.load_state_dict(checkpoint)

    assert restored.size == replay.size == 4
    assert restored.cursor == replay.cursor == 2
    original_sample = replay.sample(16)
    restored_sample = restored.sample(16)
    for name in TransitionReplay.FIELDS:
        assert torch.equal(original_sample[name], restored_sample[name])


def test_policy_score_is_success_then_safety_lexicographic() -> None:
    baseline = {
        "success_rate": 0.5,
        "safety_failure_rate": 0.0,
        "mean_completion": 0.9,
        "mean_return": 10.0,
        "mean_duration_steps": 20.0,
        "mean_action_l2_sum": 5.0,
    }
    safer_but_less_complete = dict(
        baseline,
        safety_failure_rate=0.0,
        mean_completion=0.1,
    )
    unsafe = dict(baseline, safety_failure_rate=0.1, mean_completion=1.0)
    more_success = dict(baseline, success_rate=0.6, safety_failure_rate=1.0)

    assert _score(safer_but_less_complete) > _score(unsafe)
    assert _score(more_success) > _score(baseline)
