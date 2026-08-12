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
from pathlib import Path

import pytest

from examples.embodiment.run_dynamic_benchmark_residual_wave import (
    SCHEMA_VERSION,
    _command,
    _load_jobs,
    _validate_profile,
    _wait_fail_fast,
)


def _manifest(tmp_path: Path) -> Path:
    demo = tmp_path / "demo.pt"
    demo.write_bytes(b"demo")
    rows = []
    for index in range(32):
        rows.append(
            {
                "task": "t1_xyz",
                "seed": index + 1,
                "demo_seed": 1,
                "demo_replay": str(demo),
                "output": str(tmp_path / f"job-{index}"),
                "gpu": index // 4,
                "cpu_affinity": [2 * index, 2 * index + 1],
                "numa_node": 0 if index < 16 else 1,
            }
        )
    path = tmp_path / "wave.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "jobs": rows}))
    return path


def test_j32w2_manifest_and_command_bind_demo_identity(tmp_path: Path) -> None:
    jobs = _load_jobs(_manifest(tmp_path))
    command = _command(
        jobs[0],
        config=tmp_path / "profile.yaml",
        rlinf_commit="a" * 40,
        benchmark_commit="b" * 40,
    )
    assert len(jobs) == 32
    assert command[command.index("--demo-seed") + 1] == "1"
    assert command[command.index("--demo-rlinf-commit") + 1] == "a" * 40
    assert command[command.index("--demo-replay-in") + 1].endswith("demo.pt")
    assert command[:6] == [
        "numactl",
        "--cpunodebind=0",
        "--membind=0",
        "taskset",
        "-c",
        "0,1",
    ]


def test_j32w2_rejects_cpu_overlap(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["jobs"][1]["cpu_affinity"] = payload["jobs"][0]["cpu_affinity"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="CPU affinity overlaps"):
        _load_jobs(path)


def test_j32w2_rejects_gpu_cross_numa_jobs(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["jobs"][0]["numa_node"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="each GPU to stay on one NUMA"):
        _load_jobs(path)


def test_j32w2_profile_is_fail_closed_against_topology_drift(tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "algorithm: residual_rlpd",
                "num_envs: 2",
                "eval_num_envs: 2",
                "env_worker_processes: 2",
                "eval_worker_processes: 2",
                "updates_per_vector_step: 1",
                "sampler_learner_overlap: false",
            ]
        ),
        encoding="utf-8",
    )
    _validate_profile(profile)
    profile.write_text(profile.read_text().replace("num_envs: 2", "num_envs: 4", 1))
    with pytest.raises(ValueError, match="profile topology drifted"):
        _validate_profile(profile)


def test_j32w2_wave_cancels_live_jobs_on_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self, statuses: list[int | None]) -> None:
            self.statuses = statuses

        def poll(self) -> int | None:
            return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    processes = [Process([None, None]), Process([7])]
    stopped: list[object] = []
    monkeypatch.setattr(
        "examples.embodiment.run_dynamic_benchmark_residual_wave._stop",
        lambda values: stopped.extend(values),
    )
    monkeypatch.setattr(
        "examples.embodiment.run_dynamic_benchmark_residual_wave.time.sleep",
        lambda _: None,
    )
    with pytest.raises(RuntimeError, match="job 1 failed"):
        _wait_fail_fast(processes)
    assert stopped == processes
