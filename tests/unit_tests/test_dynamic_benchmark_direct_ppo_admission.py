# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json

import pytest

from examples.embodiment.run_dynamic_benchmark_direct_ppo import (
    _compatible_nvml_pid,
    _parser,
    _wait_for_expected_nvml_processes,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_nvml_process_count_defaults_to_exclusive_single_process() -> None:
    assert _parser().get_default("expected_nvml_process_count") == 1


def test_nvml_admission_waits_for_complete_sorted_process_set(tmp_path) -> None:
    observations = iter(([31], [47, 31]))
    clock = _Clock()
    heartbeat = tmp_path / "heartbeat.json"

    pids = _wait_for_expected_nvml_processes(
        query_pids=lambda: list(next(observations)),
        expected_count=2,
        timeout_s=1.0,
        heartbeat=heartbeat,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert pids == [31, 47]
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["phase"] == "cuda_admission"
    assert payload["expected_nvml_process_count"] == 2
    assert payload["observed_nvml_pids"] == [31]


def test_nvml_admission_fails_closed_on_excess_or_timeout() -> None:
    with pytest.raises(RuntimeError, match="more NVML compute processes"):
        _wait_for_expected_nvml_processes(
            query_pids=lambda: [11, 12, 13],
            expected_count=2,
            timeout_s=1.0,
            heartbeat=None,
        )

    clock = _Clock()
    with pytest.raises(TimeoutError, match="expected=2, observed=1"):
        _wait_for_expected_nvml_processes(
            query_pids=lambda: [11],
            expected_count=2,
            timeout_s=0.5,
            heartbeat=None,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def test_compatible_nvml_pid_supports_direct_and_namespace_identity() -> None:
    assert _compatible_nvml_pid([31, 47], control_pid=31, namespace_pids=()) == (
        31,
        "direct_pid_match",
    )
    assert _compatible_nvml_pid([31], control_pid=1, namespace_pids=()) == (
        31,
        "exclusive_uuid_transition_pid_namespace_hidden",
    )
    assert _compatible_nvml_pid([31, 47], control_pid=1, namespace_pids=(31, 1)) == (
        31,
        "pid_namespace_mapping",
    )
    assert _compatible_nvml_pid(
        [31, 47], control_pid=1, namespace_pids=(31, 47, 1)
    ) == (31, "compound_nvml_pid_set")
