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

from examples.embodiment.evaluate_dynamic_benchmark_expert import (
    _expected_sha256,
    _latency_summary,
    _model_kwargs,
    _validate_policy_payload,
)


def _payload(algorithm: str = "rlpd") -> dict:
    return {
        "schema_version": "rlinf-dynamic-benchmark-expert-policy-v0.1",
        "config": {
            "task": "t1_xyz",
            "algorithm": algorithm,
            "rlinf_commit": "a" * 40,
            "benchmark_commit": "b" * 40,
            "q_heads": 10,
        },
        "state_schema": {"state_dim": 173, "mask_dim": 35},
        "model": {"fixture": 1},
        "normalizer": {"fixture": 2},
    }


def test_model_kwargs_reconstruct_sac_and_ppo_heads() -> None:
    sac = _model_kwargs(_payload()["config"], 173)
    ppo = _model_kwargs(_payload("ppo")["config"], 173)

    assert sac["add_q_head"] is True
    assert sac["add_value_head"] is False
    assert sac["num_q_heads"] == 10
    assert ppo["add_q_head"] is False
    assert ppo["add_value_head"] is True


def test_policy_payload_validation_is_commit_and_schema_fail_closed() -> None:
    payload = _payload()

    config, schema = _validate_policy_payload(
        payload,
        rlinf_commit="a" * 40,
        benchmark_commit="b" * 40,
    )

    assert config["task"] == "t1_xyz"
    assert schema["state_dim"] == 173
    with pytest.raises(ValueError, match="RLinf commit"):
        _validate_policy_payload(
            payload,
            rlinf_commit="c" * 40,
            benchmark_commit="b" * 40,
        )
    payload["schema_version"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        _validate_policy_payload(
            payload,
            rlinf_commit="a" * 40,
            benchmark_commit="b" * 40,
        )


def test_latency_summary_reports_20hz_p95_gate() -> None:
    fast = _latency_summary([0.001, 0.002, 0.003])
    slow = _latency_summary([0.001, 0.002, 0.100])

    assert fast["p95_meets_20hz"] is True
    assert slow["p95_meets_20hz"] is False
    assert fast["sample_count"] == 3


def test_expected_sha256_requires_full_lowercase_hex() -> None:
    assert _expected_sha256("a" * 64) == "a" * 64
    with pytest.raises(ValueError, match="64 lowercase"):
        _expected_sha256("abc")
