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

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.dynamic_benchmark.feature_registry import (
    FEATURE_SPECS,
    FeatureRegistry,
)


def _observation(privileged: dict[str, object] | None = None) -> SimpleNamespace:
    values = {
        "object_pose_wxyz": np.asarray([0.5, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
        "eef_pose_xyzw": np.asarray([0.4, 0.2, 0.25, 0.0, 0.0, 0.0, 1.0]),
        "object_twist_world": np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "goal_pose_wxyz": np.asarray([0.5, 0.2, 0.35, 1.0, 0.0, 0.0, 0.0]),
        "belt_surface_velocity_geom": np.asarray([0.05, 0.0, 0.0]),
    }
    if privileged is not None:
        values.update(privileged)
    return SimpleNamespace(
        task_id="t1_so3",
        policy_step=4,
        proprio={"robot0_proprio_state": np.asarray([0.0, 0.0])},
        privileged=values,
    )


def _extra(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "ee_velocity": np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "time_to_goal_s": 0.5,
        "stage_progress": 0.75,
        "action_history": [
            np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
    }
    values.update(overrides)
    return values


def test_default_registry_is_empty() -> None:
    registry = FeatureRegistry.from_config({})
    assert registry.is_empty
    values, masks = registry.encode(observation=_observation(), extra=_extra())
    assert values.shape == (0,)
    assert masks.shape == (0,)


def test_all_features_have_fixed_shapes_and_masks() -> None:
    registry = FeatureRegistry.from_config(dict.fromkeys(FEATURE_SPECS, True))
    values, masks = registry.encode(observation=_observation(), extra=_extra())
    expected_size = sum(spec.size for spec in registry.features.values())
    assert values.shape == (expected_size,)
    assert masks.shape == (len(registry.features),)
    assert np.all(masks == 1.0)
    assert registry.value_dim == expected_size
    assert registry.mask_dim == len(registry.features)

    # relative_pose must be the object expressed in the EE frame.  Features are
    # emitted in sorted name order, so locate the segment by name.
    offset = sum(
        spec.size
        for name, spec in registry.features.items()
        if name < "relative_pose"
    )
    relative = values[offset : offset + 7]
    assert relative[:3] == pytest.approx([0.1, 0.0, 0.05])


def test_missing_prerequisites_are_masked_and_zeroed() -> None:
    registry = FeatureRegistry.from_config(
        {"relative_pose": True, "geodesic_error": True, "object_vel": True}
    )
    observation = _observation()
    del observation.privileged["goal_pose_wxyz"]
    values, masks = registry.encode(observation=observation, extra=_extra())
    # Sorted order: geodesic_error, object_vel, relative_pose.
    assert masks.tolist() == [0.0, 1.0, 1.0]
    assert values[0:1] == pytest.approx([0.0])
    assert values[7:14] != pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_action_history_pads_and_is_fixed_length() -> None:
    registry = FeatureRegistry.from_config({"action_history": {"k": 3}})
    empty = FeatureRegistry.from_config({"action_history": {"k": 3}})
    values, masks = empty.encode(
        observation=_observation(), extra=_extra(action_history=[])
    )
    assert masks.tolist() == [0.0]
    assert values.shape == (21,)
    assert np.all(values == 0.0)

    values, masks = registry.encode(observation=_observation(), extra=_extra())
    assert masks.tolist() == [1.0]
    # Oldest action first, padded from the front.
    assert values[:7] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert values[7:14] == pytest.approx([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert values[14:21] == pytest.approx([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_no_future_leakage_through_observation_fields() -> None:
    registry = FeatureRegistry.from_config(dict.fromkeys(FEATURE_SPECS, True))
    current = _observation()
    future = _observation(
        {
            "future_object_pose_wxyz": np.asarray([9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0]),
            "future_object_twist_world": np.ones(6),
            "hidden_event_time_s": np.asarray([3.0]),
            "next_goal_pose_wxyz": np.ones(7),
        }
    )
    values, masks = registry.encode(observation=current, extra=_extra())
    future_values, future_masks = registry.encode(observation=future, extra=_extra())
    assert np.array_equal(values, future_values)
    assert np.array_equal(masks, future_masks)


@pytest.mark.parametrize(
    "config, match",
    [
        ({"not_a_feature": True}, "unknown state feature"),
        ({"relative_pose": False}, "omit it instead"),
        ({"action_history": {"k": 0}}, "must lie in \\[1, 8\\]"),
        ({"action_history": {"k": 9}}, "must lie in \\[1, 8\\]"),
        ({"relative_pose": {"bogus": 1}}, "unknown parameters"),
    ],
)
def test_invalid_feature_configs_fail_closed(
    config: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        FeatureRegistry.from_config(config)


def test_identity_changes_with_feature_set_and_k() -> None:
    base = FeatureRegistry.from_config({"relative_pose": True})
    extended = FeatureRegistry.from_config(
        {"relative_pose": True, "goal_error": True}
    )
    history = FeatureRegistry.from_config({"action_history": {"k": 4}})
    assert base.identity_sha256() != extended.identity_sha256()
    assert base.identity_sha256() != history.identity_sha256()
    assert base.to_dict()["value_dim"] == 7
    assert history.to_dict()["features"]["action_history"]["shape"] == [28]
