# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

from rlinf.runners.dynamic_benchmark_direct_ppo_runner import (  # noqa: E402
    DirectPPORunConfig,
)


def _run_config(**overrides: Any) -> DirectPPORunConfig:
    values = {
        "name": "test",
        "seed": 1,
        "num_envs": 1,
        "cohorts": 1,
        "rollout_horizon": 1,
        "minibatch_size": 1,
        "ppo_epochs": 1,
        "encoder_batch_size": 1,
        "manifest_name": "train",
    }
    values.update(overrides)
    return DirectPPORunConfig(**values)


def test_direct_ppo_policy_rgb_defaults_to_224_and_rejects_lower_resolution() -> None:
    assert _run_config().image_size == 224
    with pytest.raises(ValueError, match="image_size >= 224"):
        _run_config(image_size=64)
