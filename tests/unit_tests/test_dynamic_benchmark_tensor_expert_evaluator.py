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

import hashlib
import json
import os
import sys
import textwrap
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.embodiment.evaluate_dynamic_benchmark_tensor_expert import (
    EPISODE_LEDGER_SCHEMA,
    POLICY_SCHEMAS,
    SEQUENCE_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    WORKER_SPEC_SCHEMA,
    PinnedSequence,
    TensorEvaluationError,
    _build_backend,
    _rename_directory_with_claim,
    assert_strict_finite,
    collect_process_identity,
    launch_fresh_worker,
    launch_identity_probe,
    manifest_requests_sha256,
    normalize_terminal_rows,
    portable_export_identity,
    process_identity_matches,
    publish_result_bundle,
    run_device_cohort,
    runtime_request_payload,
    validate_backend_runtime_identity,
    verify_result_bundle,
)


def test_policy_schema_accepts_planner_tuned_offpolicy_v03() -> None:
    assert "rlinf-gpuenv0-tensor-offpolicy-smoke-v0.3" in POLICY_SCHEMAS


def _export_identity() -> dict:
    return {
        "request_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "model_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "frozen_request": {"episode_id": "frozen"},
    }


def _request(index: int, *, split: str = "validation") -> dict:
    return {
        "episode_id": f"p0-validation-{index:04d}",
        "task_id": "p0_grasp",
        "split": split,
        "seed": 100 + index,
        "action_mode": "E7",
        "observation_track": "state",
        "object_mode": "cube",
        "reset_mode": "tabletop",
        "factors": {"x": 0.1 * index},
        "api_version": "db-api-v0.1",
    }


def _sequence_payload(count: int = 2) -> dict:
    requests = [_request(index) for index in range(count)]
    return {
        "schema_version": SEQUENCE_SCHEMA,
        "task_id": "p0_grasp",
        "split": "validation",
        "manifest_seed": 20260809,
        "manifest_sha256": manifest_requests_sha256(requests),
        "api_version": "gpu-native-api-v0.2",
        "task_quality": {
            "schema_version": "db0-episode-task-quality-v1",
            "evaluator_backend_id": "mjwarp-quality-v1",
        },
        "active_export_identity": _export_identity(),
        "requests": requests,
    }


def test_pinned_sequence_requires_exact_manifest_and_unique_episodes() -> None:
    payload = _sequence_payload()

    sequence = PinnedSequence.from_payload(payload)

    assert sequence.task_id == "p0_grasp"
    assert len(sequence.requests) == 2
    assert sequence.active_export_identity == _export_identity()

    tampered = json.loads(json.dumps(payload))
    tampered["requests"][0]["seed"] += 1
    with pytest.raises(ValueError, match="manifest SHA-256"):
        PinnedSequence.from_payload(tampered)

    duplicate = json.loads(json.dumps(payload))
    duplicate["requests"][1]["episode_id"] = duplicate["requests"][0]["episode_id"]
    duplicate["manifest_sha256"] = manifest_requests_sha256(duplicate["requests"])
    with pytest.raises(ValueError, match="duplicate"):
        PinnedSequence.from_payload(duplicate)


def test_portable_export_identity_allows_only_relocatable_path_difference() -> None:
    left = {"export_dir": "/machine-a/export", **_export_identity()}
    right = {"export_dir": "/machine-b/export", **_export_identity()}

    assert portable_export_identity(left) == portable_export_identity(right)

    right["model_sha256"] = "a" * 64
    assert portable_export_identity(left) != portable_export_identity(right)
    with pytest.raises(ValueError, match="fields drifted"):
        portable_export_identity({**left, "untracked": "value"})


def test_runtime_episode_ids_and_backend_api_runtime_pins_are_exact() -> None:
    sequence = PinnedSequence.from_payload(_sequence_payload())
    runtime_row = runtime_request_payload(sequence, 1)
    stable = {
        "task_id": "p0_grasp",
        "api_version": "gpu-native-api-v0.2",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "runtime_versions": {
            "mujoco": "3.11.0",
            "mujoco-warp": "3.11.0",
            "warp-lang": "1.16.0",
        },
        "reset_manifest": {
            "origin": "caller",
            "split": "validation",
            "seed": 20260809,
            "size": 2,
            "sha256": sequence.manifest_sha256,
        },
        "task_quality": {
            "schema_version": "db0-episode-task-quality-v1",
            "evaluator_backend_id": "mjwarp-quality-v1",
        },
    }
    sources = {
        "se3_wam": {"commit": "a" * 40, "tree": "b" * 40},
    }
    runtime = {
        "versions": {
            "mujoco": "3.11.0",
            "mujoco-warp": "3.11.0",
            "warp-lang": "1.16.0",
        }
    }

    validate_backend_runtime_identity(
        stable,
        runtime_payload=runtime,
        source_snapshot=sources,
        sequence=sequence,
    )

    assert runtime_row["episode_id"].endswith("-cycle00000000")
    drifted = dict(stable)
    drifted["runtime_versions"] = {
        **stable["runtime_versions"],
        "warp-lang": "drift",
    }
    with pytest.raises(TensorEvaluationError, match="version mismatch"):
        validate_backend_runtime_identity(
            drifted,
            runtime_payload=runtime,
            source_snapshot=sources,
            sequence=sequence,
        )


def test_backend_construction_threads_caller_pinned_se3_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "sources": [
            {
                "name": "rlinf",
                "root": "/source/rlinf",
                "commit": "1" * 40,
                "tree": "2" * 40,
                "module": "rlinf",
            },
            {
                "name": "se3_wam",
                "root": "/source/se3",
                "commit": "a" * 40,
                "tree": "b" * 40,
                "module": "se3_wam",
            },
        ],
    }
    source_path = tmp_path / "source-manifest.json"
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    captured: dict = {}

    class _Backend:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    backend_module = types.ModuleType("rlinf.envs.dynamic_benchmark.gpu_tensor_backend")
    backend_module.GpuNativeTensorBackendEnv = _Backend
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.dynamic_benchmark.gpu_tensor_backend",
        backend_module,
    )
    sequence = PinnedSequence.from_payload(_sequence_payload())
    spec = {
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": source_sha256,
        "num_envs": 2,
        "export_dir": "/export",
        "expected_gpu_uuid": "GPU-803b6f88-a884-134a-d92d-cdc532e22e14",
        "device_ordinal": 0,
        "image_size": 64,
    }

    _build_backend(spec, sequence, (object(), object()))

    assert captured["expected_se3_source_commit"] == "a" * 40
    assert captured["expected_se3_source_tree"] == "b" * 40
    assert captured["manifest_sha256"] == sequence.manifest_sha256


@dataclass(frozen=True)
class _FakeQuality:
    episode_id: str
    task_id: str = "p0_grasp"
    terminal: bool = True

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "terminal": self.terminal,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> _FakeQuality:
        return cls(**payload)


@dataclass(frozen=True)
class _FakeTerminal:
    lane: int
    episode_id: str
    task_id: str
    terminated: bool
    truncated: bool
    success: bool
    termination_reason: str
    completion: float
    task_quality: _FakeQuality | None
    physics_step: int = 25
    control_step: int = 1
    policy_step: int = 1


def test_terminal_rows_require_typed_quality_only_for_success() -> None:
    requests = [_request(0), _request(1)]
    terminals = [
        _FakeTerminal(
            lane=0,
            episode_id=requests[0]["episode_id"],
            task_id="p0_grasp",
            terminated=True,
            truncated=False,
            success=True,
            termination_reason="success",
            completion=1.0,
            task_quality=_FakeQuality(requests[0]["episode_id"]),
        ),
        _FakeTerminal(
            lane=1,
            episode_id=requests[1]["episode_id"],
            task_id="p0_grasp",
            terminated=False,
            truncated=True,
            success=False,
            termination_reason="timeout",
            completion=0.25,
            task_quality=None,
        ),
    ]
    numeric = [
        {"lane": 0, "return": 1.5, "episode_cost": 0.2, "valid_steps": 1, "done": True},
        {
            "lane": 1,
            "return": -0.5,
            "episode_cost": 0.3,
            "valid_steps": 1,
            "done": True,
        },
    ]

    rows = normalize_terminal_rows(
        terminals,
        expected_rows=requests,
        numeric_rows=numeric,
        quality_type=_FakeQuality,
        quality_from_dict=_FakeQuality.from_dict,
    )

    assert rows[0]["schema_version"] == EPISODE_LEDGER_SCHEMA
    assert rows[0]["task_quality"]["episode_id"] == requests[0]["episode_id"]
    assert rows[1]["task_quality"] is None
    assert rows[0]["safety"] is None
    assert rows[0]["safety_available"] is False
    assert rows[0]["safety_failure"] is None

    terminals[1] = _FakeTerminal(
        **{
            **terminals[1].__dict__,
            "task_quality": _FakeQuality(requests[1]["episode_id"]),
        }
    )
    with pytest.raises(TensorEvaluationError, match="strictly None"):
        normalize_terminal_rows(
            terminals,
            expected_rows=requests,
            numeric_rows=numeric,
            quality_type=_FakeQuality,
            quality_from_dict=_FakeQuality.from_dict,
        )


def test_device_cohort_uses_step_device_and_one_terminal_materialization() -> None:
    requests = [_request(0), _request(1)]
    manifest_digest = manifest_requests_sha256(requests)

    class Policy:
        observation_dim = 3
        algorithm = "fixture"
        checkpoint_schema = "fixture"

        def __init__(self) -> None:
            self.calls = 0

        def act(self, observation):
            self.calls += 1
            return ("action", observation)

    class Ledger:
        def __init__(self) -> None:
            self.records = 0
            self.materializations = 0

        def record(self, action, step) -> None:
            assert action[0] == "action"
            assert step.done is True
            self.records += 1

        def materialize_once(self):
            self.materializations += 1
            assert self.materializations == 1
            return tuple(
                {
                    "lane": lane,
                    "return": 2.0 + lane,
                    "episode_cost": 0.5,
                    "valid_steps": 2,
                    "done": True,
                }
                for lane in range(2)
            )

    class Environment:
        num_envs = 2
        device = "fake:0"
        cohort_horizon_steps = 2
        manifest_sha256 = manifest_digest

        def __init__(self) -> None:
            self.step_device_calls = 0
            self.terminal_materializations = 0

        def reset(self):
            return SimpleNamespace(
                observation=SimpleNamespace(shape=(2, 3)),
                episode_ids=tuple(row["episode_id"] for row in requests),
                seeds=tuple(row["seed"] for row in requests),
                manifest_ordinals=(0, 1),
                manifest_sha256=manifest_digest,
            )

        def step_device(self, action):
            self.step_device_calls += 1
            return SimpleNamespace(
                observation=SimpleNamespace(shape=(2, 3)),
                episode_ids=tuple(row["episode_id"] for row in requests),
                done=True,
            )

        def materialize_terminal_ledger_once(self, indices, episode_ids):
            self.terminal_materializations += 1
            assert indices == (0, 1)
            assert episode_ids == tuple(row["episode_id"] for row in requests)
            return (
                _FakeTerminal(
                    lane=0,
                    episode_id=requests[0]["episode_id"],
                    task_id="p0_grasp",
                    terminated=True,
                    truncated=False,
                    success=True,
                    termination_reason="success",
                    completion=1.0,
                    task_quality=_FakeQuality(requests[0]["episode_id"]),
                ),
                _FakeTerminal(
                    lane=1,
                    episode_id=requests[1]["episode_id"],
                    task_id="p0_grasp",
                    terminated=False,
                    truncated=True,
                    success=False,
                    termination_reason="timeout",
                    completion=0.5,
                    task_quality=None,
                ),
            )

    env = Environment()
    policy = Policy()
    ledger = Ledger()
    clock_values = iter((10.0, 12.0))
    synchronizations = []

    result = run_device_cohort(
        env=env,
        policy=policy,
        expected_rows=requests,
        expected_ordinals=(0, 1),
        ledger_factory=lambda _size, _device: ledger,
        synchronize=synchronizations.append,
        quality_type=_FakeQuality,
        quality_from_dict=_FakeQuality.from_dict,
        clock=lambda: next(clock_values),
    )

    assert env.step_device_calls == env.cohort_horizon_steps
    assert env.terminal_materializations == 1
    assert ledger.records == env.cohort_horizon_steps
    assert ledger.materializations == 1
    assert policy.calls == env.cohort_horizon_steps
    assert synchronizations == ["fake:0", "fake:0"]
    assert result.rollout_seconds == 2.0
    assert [row["ordinal"] for row in result.episodes] == [0, 1]


def test_process_identity_parses_comm_parentheses_and_start_ticks(
    tmp_path: Path,
) -> None:
    pid = 321
    process_dir = tmp_path / str(pid)
    process_dir.mkdir()
    boot_dir = tmp_path / "sys" / "kernel" / "random"
    boot_dir.mkdir(parents=True)
    remainder = ["S", "123", *("0" for _ in range(17)), "998877"]
    (process_dir / "stat").write_text(
        f"{pid} (worker ) name) {' '.join(remainder)}\n", encoding="ascii"
    )
    (boot_dir / "boot_id").write_text("BOOT-FIXTURE\n", encoding="ascii")

    identity = collect_process_identity(pid, proc_root=tmp_path)

    assert identity.pid == pid
    assert identity.parent_pid == 123
    assert identity.start_ticks == 998877
    assert identity.boot_id == "boot-fixture"


def test_identity_probe_is_a_fresh_same_boot_subprocess() -> None:
    script = (
        Path(__file__).parents[2]
        / "examples"
        / "embodiment"
        / "evaluate_dynamic_benchmark_tensor_expert.py"
    )

    parent = collect_process_identity()
    child = launch_identity_probe(script)

    assert child["pid"] != parent.pid
    if os.name != "nt":
        assert child["parent_pid"] == parent.pid
    assert child["boot_id"] == parent.boot_id
    assert child["start_ticks"] != parent.start_ticks


def test_process_reobservation_and_fresh_worker_receipt(tmp_path: Path) -> None:
    parent = collect_process_identity()
    assert process_identity_matches(
        asdict(parent), collect_process_identity(parent.pid)
    )
    repository = Path(__file__).parents[2]
    worker = tmp_path / "host_worker.py"
    worker.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import json
            import sys
            from dataclasses import asdict
            from pathlib import Path

            sys.path.insert(0, {str(repository)!r})
            from examples.embodiment.evaluate_dynamic_benchmark_tensor_expert import (
                atomic_json,
                collect_process_identity,
                file_sha256,
            )

            parser = argparse.ArgumentParser()
            parser.add_argument("--worker-spec", type=Path, required=True)
            parser.add_argument("--expected-worker-spec-sha256", required=True)
            parser.add_argument("--worker-result", type=Path, required=True)
            parser.add_argument("--worker-failure", type=Path, required=True)
            args = parser.parse_args()
            if file_sha256(args.worker_spec) != args.expected_worker_spec_sha256:
                raise RuntimeError("worker spec hash drifted")
            spec = json.loads(args.worker_spec.read_text(encoding="utf-8"))
            child = collect_process_identity()
            atomic_json(
                args.worker_result,
                {{
                    "schema_version": "host-fixture",
                    "status": "complete",
                    "mode": spec["mode"],
                    "process_identity": {{
                        "parent_start": spec["parent_process_start"],
                        "child_start": asdict(child),
                        "child_end": asdict(collect_process_identity()),
                    }},
                }},
            )
            """
        ),
        encoding="utf-8",
    )
    spec = {
        "schema_version": WORKER_SPEC_SCHEMA,
        "mode": "validation",
        "policy_path": str(tmp_path / "policy.pt"),
        "policy_sha256": "a" * 64,
        "sequence_path": str(tmp_path / "sequence.json"),
        "sequence_sha256": "b" * 64,
        "source_manifest_path": str(tmp_path / "source.json"),
        "source_manifest_sha256": "c" * 64,
        "runtime_manifest_path": str(tmp_path / "runtime.json"),
        "runtime_manifest_sha256": "d" * 64,
        "export_dir": str(tmp_path / "export"),
        "expected_gpu_uuid": "GPU-fixture",
        "device_ordinal": 0,
        "image_size": 64,
        "num_envs": 2,
        "start_ordinal": 0,
        "episode_count": 2,
        "parent_process_start": asdict(parent),
    }

    result = launch_fresh_worker(spec, script_path=worker, timeout_s=30)

    assert result["schema_version"] == "host-fixture"
    assert result["process_identity"]["parent_start"] == asdict(parent)
    if os.name != "nt":
        assert result["process_identity"]["child_start"]["parent_pid"] == parent.pid
    assert result["worker_stdio"]["stdout_bytes"] == 0
    assert result["worker_stdio"]["stderr_bytes"] == 0


def test_result_bundle_is_atomic_strict_and_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    episode = {
        "schema_version": EPISODE_LEDGER_SCHEMA,
        "ordinal": 0,
        "episode_id": "episode-0",
        "return": 1.0,
        "success": True,
        "completion": 1.0,
        "task_quality": {"summary_sha256": "a" * 64},
        "episode_cost": 0.5,
    }

    digests = publish_result_bundle(
        output,
        result_name="evaluation.json",
        result={"schema_version": "fixture", "status": "complete"},
        episodes=[episode],
    )
    result, episodes = verify_result_bundle(output, result_name="evaluation.json")

    assert set(digests) == {"evaluation.json", "episodes.jsonl"}
    assert result["status"] == "complete"
    assert episodes == [episode]
    with pytest.raises(FileExistsError):
        publish_result_bundle(
            output,
            result_name="evaluation.json",
            result={"schema_version": "fixture"},
            episodes=[],
        )
    with pytest.raises(ValueError, match="non-finite"):
        assert_strict_finite({"bad": float("nan")})


def test_claim_fallback_is_atomic_no_clobber_and_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "stage"
    source.mkdir()
    (source / "payload").write_text("complete\n", encoding="utf-8")
    destination = tmp_path / "published"

    _rename_directory_with_claim(source, destination)

    assert not source.exists()
    assert (destination / "payload").read_text(encoding="utf-8") == "complete\n"
    assert not (tmp_path / ".published.publish.lock").exists()

    second_source = tmp_path / "second-stage"
    second_source.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        _rename_directory_with_claim(second_source, destination)
    assert second_source.is_dir()

    blocked_source = tmp_path / "blocked-stage"
    blocked_source.mkdir()
    blocked_destination = tmp_path / "blocked"
    claim = tmp_path / ".blocked.publish.lock"
    claim.write_text("interrupted publisher\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already in progress"):
        _rename_directory_with_claim(blocked_source, blocked_destination)
    assert blocked_source.is_dir()
    assert claim.read_text(encoding="utf-8") == "interrupted publisher\n"
