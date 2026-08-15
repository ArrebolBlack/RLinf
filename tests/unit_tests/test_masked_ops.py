# Copyright 2026 The RLinf Authors.
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

from rlinf.algorithms.masked_ops import masked_mean, masked_mean_ratio


def test_masked_mean_reduces_only_selected_values() -> None:
    values = torch.tensor([1.0, 3.0, 9.0])
    mask = torch.tensor([True, True, False])

    assert torch.equal(masked_mean(values, mask), torch.tensor(2.0))


def test_masked_mean_returns_zero_for_an_empty_mask() -> None:
    values = torch.tensor([1.0, 3.0])
    mask = torch.tensor([False, False])

    assert torch.equal(masked_mean(values, mask), torch.tensor(0.0))


def test_masked_mean_ratio_applies_inverse_sampling_ratio() -> None:
    values = torch.tensor([2.0, 8.0])
    mask = torch.tensor([True, False])
    ratios = torch.tensor([0.5, 0.25])

    assert torch.equal(masked_mean_ratio(values, mask, ratios), torch.tensor(2.0))
