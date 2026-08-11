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

import pytest

from rlinf.envs.dynamic_benchmark.reward_registry import (
    COMPONENT_IMPL,
    RewardRegistry,
)


def _inputs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "action_l2": 1.0,
        "action_delta_l2": 0.4,
        "completion": 0.95,
        "stage_progress": 0.6,
        "geodesic_error_rad": 0.25,
        "relative_translation_error_m": 0.02,
        "relative_rotation_error_rad": 0.1,
        "relative_velocity_norm_m_s": 0.3,
        "distance_m": 0.03,
        "time_to_goal_s": 0.15,
    }
    values.update(overrides)
    return values


def test_default_registry_is_empty_and_adds_nothing() -> None:
    registry = RewardRegistry.from_config({})
    assert registry.is_empty
    total, values, recorded = registry.step(_inputs())
    assert total == 0.0
    assert values == {}
    assert recorded["previous_stage_progress"] == 0.0
    assert registry.identity_sha256() == RewardRegistry.from_config({}).identity_sha256()


def test_each_component_is_recomputable_below_1e_7() -> None:
    config = {
        "r_ori_geodesic": {"weight": 2.0, "scale_rad": 1.0},
        "r_rel_pose": {"weight": 3.0, "scale_pos_m": 0.1, "scale_rot_rad": 1.0},
        "r_effort": {"weight": 0.005},
        "r_action_rate": {"weight": 0.25, "scale": 1.0},
        "r_completion_shaping": {"weight": 1.0, "near_threshold": 0.9},
        "r_vel_align": {"weight": 0.5, "speed_scale_m_s": 1.0},
        "r_timing": {
            "weight": 1.0,
            "dist_threshold_m": 0.05,
            "target_ttc_s": 0.1,
            "ttc_scale_s": 0.1,
        },
        "r_stage": {"weight": 1.0},
    }
    registry = RewardRegistry.from_config(config)
    assert registry.enabled_names == tuple(sorted(COMPONENT_IMPL))

    first_inputs = _inputs()
    total, values, recorded = registry.step(first_inputs)
    assert total != 0.0
    assert set(values) == set(COMPONENT_IMPL)
    recomputed_total, recomputed_values = RewardRegistry.recompute(
        registry.components, recorded
    )
    assert abs(total - recomputed_total) < 1e-7
    assert abs(total - sum(values.values())) < 1e-12
    for name in values:
        assert abs(values[name] - recomputed_values[name]) < 1e-7

    # A second step must reflect the previous stage progress stored in the
    # registry (r_stage potential difference).
    second_inputs = _inputs(stage_progress=0.8)
    second_total, second_values, _ = registry.step(second_inputs)
    assert second_values["r_stage"] == pytest.approx(0.8 - 0.6)
    assert second_total - total == pytest.approx(
        second_values["r_stage"] - values["r_stage"]
    )


def test_weight_zero_disables_component_and_state_is_checkpointed() -> None:
    registry = RewardRegistry.from_config(
        {"r_effort": {"weight": 0.0}, "r_stage": {"weight": 2.0}}
    )
    assert registry.enabled_names == ("r_stage",)
    total, values, _ = registry.step(_inputs(stage_progress=0.4))
    assert values == {"r_stage": pytest.approx(0.8)}
    assert total == pytest.approx(0.8)

    state = registry.state_dict()
    restored = RewardRegistry.from_config({})
    restored.load_state_dict(state)
    assert restored._previous_stage_progress == pytest.approx(0.4)
    with pytest.raises(ValueError, match="checkpoint schema"):
        restored.load_state_dict({"schema_version": "legacy"})


def test_missing_or_non_finite_inputs_yield_zero_for_that_component() -> None:
    registry = RewardRegistry.from_config(
        {
            "r_ori_geodesic": {"weight": 1.0},
            "r_timing": {"weight": 1.0},
            "r_stage": {"weight": 1.0},
        }
    )
    total, values, _ = registry.step(_inputs(geodesic_error_rad=None, distance_m=None))
    assert values["r_ori_geodesic"] == 0.0
    assert values["r_timing"] == 0.0
    assert values["r_stage"] != 0.0
    assert total == pytest.approx(values["r_stage"])


@pytest.mark.parametrize(
    "config, match",
    [
        ({"unknown_component": {"weight": 1.0}}, "unknown reward component"),
        ({"r_effort": {}}, "requires a 'weight'"),
        ({"r_effort": {"weight": -1.0}}, "weight must be non-negative"),
        ({"r_effort": {"weight": 1.0, "bogus": 1.0}}, "unknown parameters"),
        ({"r_effort": {"weight": float("nan")}}, "must be finite"),
        ({"r_completion_shaping": {"weight": 1.0, "near_threshold": 2.0}}, "in \\[0, 1\\]"),
        ({"r_timing": {"weight": 1.0, "ttc_scale_s": 0.0}}, "must be positive"),
    ],
)
def test_invalid_component_configs_fail_closed(
    config: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        RewardRegistry.from_config(config)


def test_identity_changes_with_weights_and_parameters() -> None:
    base = RewardRegistry.from_config({"r_effort": {"weight": 0.005}})
    heavier = RewardRegistry.from_config({"r_effort": {"weight": 0.01}})
    timed = RewardRegistry.from_config(
        {"r_timing": {"weight": 1.0, "target_ttc_s": 0.2}}
    )
    assert base.identity_sha256() != heavier.identity_sha256()
    assert base.identity_sha256() != timed.identity_sha256()


def test_action_rate_uses_only_the_previous_action_and_is_recomputable() -> None:
    registry = RewardRegistry.from_config(
        {"r_action_rate": {"weight": 2.0, "scale": 0.5}}
    )
    total, values, recorded = registry.step(_inputs(action_delta_l2=0.25))

    assert values["r_action_rate"] == pytest.approx(-1.0)
    recomputed_total, recomputed_values = RewardRegistry.recompute(
        registry.components, recorded
    )
    assert recomputed_total == pytest.approx(total)
    assert recomputed_values["r_action_rate"] == pytest.approx(-1.0)
    missing_total, missing_values, _ = registry.step(_inputs(action_delta_l2=None))
    assert missing_values["r_action_rate"] == 0.0
    assert missing_total == 0.0
