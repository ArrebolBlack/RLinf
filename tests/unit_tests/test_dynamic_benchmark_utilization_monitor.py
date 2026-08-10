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

from pathlib import Path

import pytest

from examples.embodiment.monitor_dynamic_benchmark_utilization import (
    _combined_anomalies,
    _distribution,
    _log_anomalies,
    _trainer_output,
)


def test_distribution_reports_interpolated_percentiles() -> None:
    result = _distribution([0.0, 10.0, 20.0, 30.0])

    assert result == {
        "samples": 4,
        "mean": 15.0,
        "p50": 15.0,
        "p90": pytest.approx(27.0),
        "max": 30.0,
    }
    assert _distribution([])["mean"] is None


def test_trainer_output_requires_explicit_output() -> None:
    output = _trainer_output(["python", "trainer.py", "--output", "run/trainer"])

    assert output == Path("run/trainer").resolve()
    with pytest.raises(ValueError, match="--output"):
        _trainer_output(["python", "trainer.py"])


def test_log_anomalies_classifies_known_failure_modes(tmp_path: Path) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text(
        "CUDA out of memory\nNo space left on device\nWorker exited unexpectedly\n",
        encoding="utf-8",
    )

    assert _log_anomalies([stderr, tmp_path / "missing.log"]) == [
        "io",
        "oom",
        "worker_crash",
    ]


def test_combined_anomalies_adds_watchdog_hang_without_type_error(
    tmp_path: Path,
) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text("CUDA out of memory\n", encoding="utf-8")

    assert _combined_anomalies([stderr], watchdog_triggered=True) == ["hang", "oom"]
    assert _combined_anomalies([stderr], watchdog_triggered=False) == ["oom"]
