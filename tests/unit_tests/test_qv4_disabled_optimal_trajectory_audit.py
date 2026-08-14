from __future__ import annotations

import pytest

from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _audit_winner_release_eligibility,
)


def _disabled_metadata() -> dict[str, object]:
    return {
        "quality_v4_validation": None,
        "metrics": {
            "eligible_for_behavior_cloning": False,
            "success": True,
            "label_valid": True,
            "termination_reason": "success",
            "trajectory_completion": 1.0,
        },
    }


def test_qv4_disabled_contract_accepts_successful_non_bc_winner() -> None:
    _audit_winner_release_eligibility(
        {"eligible_for_behavior_cloning": False},
        _disabled_metadata(),
        qv4_disabled_nonblocking=True,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success", False),
        ("label_valid", False),
        ("termination_reason", "time_limit"),
        ("trajectory_completion", 0.99),
        ("eligible_for_behavior_cloning", True),
    ],
)
def test_qv4_disabled_contract_rejects_invalid_release_gate(
    field: str, value: object
) -> None:
    metadata = _disabled_metadata()
    metadata["metrics"][field] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="release eligibility mismatch"):
        _audit_winner_release_eligibility(
            {"eligible_for_behavior_cloning": False},
            metadata,
            qv4_disabled_nonblocking=True,
        )


def test_qv4_disabled_contract_rejects_exported_qv4() -> None:
    metadata = _disabled_metadata()
    metadata["quality_v4_validation"] = {"passed": True}
    with pytest.raises(ValueError, match="unexpectedly contains Qv4 validation"):
        _audit_winner_release_eligibility(
            {"eligible_for_behavior_cloning": False},
            metadata,
            qv4_disabled_nonblocking=True,
        )


def test_default_contract_still_requires_bc_eligibility() -> None:
    with pytest.raises(ValueError, match="not behavior-cloning eligible"):
        _audit_winner_release_eligibility(
            {"eligible_for_behavior_cloning": False},
            _disabled_metadata(),
            qv4_disabled_nonblocking=False,
        )
