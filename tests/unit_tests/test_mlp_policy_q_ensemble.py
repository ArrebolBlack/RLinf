from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy import get_model
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


def test_mlp_policy_builds_requested_q_ensemble() -> None:
    model = MLPPolicy(
        obs_dim=11,
        action_dim=7,
        num_action_chunks=1,
        add_value_head=False,
        add_q_head=True,
        num_q_heads=10,
    )

    q_values = model.sac_q_forward(
        {"states": torch.randn(4, 11)},
        torch.randn(4, 7),
    )

    assert model.q_head.num_q_heads == 10
    assert q_values.shape == (4, 10)


def test_mlp_policy_factory_forwards_num_q_heads() -> None:
    cfg = OmegaConf.create(
        {
            "model_type": "mlp_policy",
            "obs_dim": 11,
            "action_dim": 7,
            "num_action_chunks": 1,
            "add_value_head": False,
            "add_q_head": True,
            "num_q_heads": 6,
        }
    )

    model = get_model(cfg, torch_dtype=torch.float32)

    assert model.q_head.num_q_heads == 6


def test_mlp_policy_rejects_single_q_head() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        MLPPolicy(
            obs_dim=11,
            action_dim=7,
            num_action_chunks=1,
            add_value_head=False,
            add_q_head=True,
            num_q_heads=1,
        )
