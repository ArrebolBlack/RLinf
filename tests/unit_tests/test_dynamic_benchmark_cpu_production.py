# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.embodiment.run_dynamic_benchmark_cpu_production import (
    DEFAULT_WORKERS_PER_JOB,
    SCHEMA_VERSION,
    ProductionJob,
    _discover_numa_cpus,
    _load_jobs,
    _merge_command,
    _normalized_export_args,
    _parse_cpu_list,
    _run_node_queue,
    _select_physical_first_cpus,
    _terminate_processes,
)


def test_parse_cpu_list_supports_ranges() -> None:
    assert _parse_cpu_list("0-2,8,10-11") == [0, 1, 2, 8, 10, 11]


def test_discovers_only_affinity_visible_numa_cpus(tmp_path: Path) -> None:
    node = tmp_path / "node"
    (node / "node0").mkdir(parents=True)
    (node / "node1").mkdir()
    (node / "node0" / "cpulist").write_text("0-3\n")
    (node / "node1" / "cpulist").write_text("4-7\n")
    assert _discover_numa_cpus(tmp_path, {1, 2, 6}) == {0: [1, 2], 1: [6]}


def test_selects_one_thread_per_core_before_siblings(tmp_path: Path) -> None:
    for cpu, core in ((0, 0), (1, 0), (2, 1), (3, 1)):
        topology = tmp_path / "cpu" / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text("0")
        (topology / "core_id").write_text(str(core))
    assert _select_physical_first_cpus([0, 1, 2, 3], 2, tmp_path) == [0, 2]
    assert _select_physical_first_cpus([0, 1, 2, 3], 4, tmp_path) == [0, 2, 1, 3]


def test_production_args_add_scientific_selection_defaults() -> None:
    result = _normalized_export_args(
        ("--accepted-episodes", "10", "--max-resets", "20")
    )
    assert result[-4:] == (
        "--candidate-search-mode",
        "full-pool",
        "--selection-mode",
        "planner-pareto",
    )


def test_production_args_preserve_explicit_planner_only_selection() -> None:
    explicit = (
        "--accepted-episodes",
        "10",
        "--max-resets",
        "20",
        "--initial-k",
        "1",
        "--max-k",
        "1",
        "--candidate-search-mode",
        "full-pool",
        "--selection-mode",
        "legacy-lexicographic",
    )
    assert _normalized_export_args(explicit) == explicit


@pytest.mark.parametrize("forbidden", ["--output", "--shard-count", "--resume"])
def test_production_args_reject_supervisor_owned_flags(forbidden: str) -> None:
    with pytest.raises(ValueError, match="supervisor owns"):
        _normalized_export_args(
            ("--accepted-episodes", "10", "--max-resets", "20", forbidden, "x")
        )


def test_loads_campaign_jobs_without_source_pins(tmp_path: Path) -> None:
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "jobs": [
                    {
                        "job_id": "t1",
                        "output": str(tmp_path / "out"),
                        "export_args": [
                            "--accepted-episodes",
                            "10",
                            "--max-resets",
                            "20",
                        ],
                    }
                ],
            }
        )
    )
    args = SimpleNamespace(
        output=None,
        campaign_manifest=manifest,
        export_args=[],
    )
    jobs = _load_jobs(args)
    assert len(jobs) == 1
    assert jobs[0].job_id == "t1"
    assert "--rlinf-commit" not in jobs[0].export_args
    assert DEFAULT_WORKERS_PER_JOB == 16


def test_merge_defaults_to_accepted_prefix_and_fixed_reset_is_opt_in(
    tmp_path: Path,
) -> None:
    base = _merge_command(
        tmp_path / "merge.py",
        tmp_path / "shards",
        tmp_path / "output",
        "100",
        fixed_reset_workload=False,
    )
    assert "--require-max-resets" not in base
    fixed = _merge_command(
        tmp_path / "merge.py",
        tmp_path / "shards",
        tmp_path / "output",
        "100",
        fixed_reset_workload=True,
    )
    assert fixed[-1] == "--require-max-resets"


def test_export_worker_command_is_numa_memory_bound(tmp_path: Path) -> None:
    # Command construction is covered as a stable source-level contract because
    # the Windows unit runner cannot execute numactl.
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/run_dynamic_benchmark_cpu_production.py"
    ).read_text(encoding="utf-8")
    assert 'f"--membind={numa_node}"' in source
    assert 'f"--cpunodebind={numa_node}"' in source
    assert '"taskset",\n                "-c",\n                str(cpu)' in source


def test_one_node_queue_never_overlaps_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_run(job: ProductionJob, **_: object) -> dict[str, object]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {"job_id": job.job_id}

    monkeypatch.setattr(
        "examples.embodiment.run_dynamic_benchmark_cpu_production._run_job",
        fake_run,
    )
    jobs = [
        (ProductionJob(str(index), Path(str(index)), ()), 0, [index])
        for index in range(3)
    ]
    result = _run_node_queue(jobs)
    assert [row["job_id"] for row in result] == ["0", "1", "2"]
    assert peak == 1


def test_cleanup_terminates_live_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        pid = 123
        waited = False

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.waited = True
            return 0

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "examples.embodiment.run_dynamic_benchmark_cpu_production.os.killpg",
        lambda pid, value: signals.append((pid, value)),
        raising=False,
    )
    process = Process()
    _terminate_processes([process])
    assert signals[0][0] == process.pid
    assert process.waited is True
