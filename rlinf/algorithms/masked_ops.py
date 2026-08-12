# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Small tensor reductions shared by native RL losses.

Keeping these operations in the algorithms package lets standalone learners use
GAE and PPO losses without importing the Ray-backed worker scheduler.
"""

from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None, axis=None):
    if mask is None:
        return values.mean(axis=axis)
    if (~mask).all():
        return (values * mask).sum(axis=axis)
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_mean_ratio(
    values: torch.Tensor, mask: torch.Tensor, loss_mask_ratio: torch.Tensor
):
    return (values / loss_mask_ratio * mask).mean()
