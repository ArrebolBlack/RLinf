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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _audit_attempt_tape,
    _audit_render_parity_skip,
    _render_parity_skip_events,
)
from examples.embodiment.audit_dynamic_benchmark_optimal_trajectories import (
    _payload_sha256 as _audit_payload_sha256,
)
from examples.embodiment.export_dynamic_benchmark_optimal_trajectories import (
    CANDIDATE_SCHEMA,
    _ArmedResetReplayEnv,
    _budget_sequence,
    _candidate_identity,
    _eligible,
    _file_boundary,
    _progress_payload,
    _quality_score,
    _recover_partial_output,
    _render_parity_skip,
    _restore_candidate_start,
    _select_winner,
    _task_quality_from_infos,
    _validate_candidate_manifest,
    _write_attempt_tape,
)
from examples.embodiment.merge_optimal_export_shards import _kept_recovery_events


def _record(candidate_index: int, *, value: float = 1.0) -> dict:
    replay = {"passed": True, "outcomes_exact": True}
    record = {
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "replay_validation": replay,
        "trajectory_completion": 1.0,
        "return": value,
        "control_steps": 4,
        "action_l2_sum": 2.0,
        "candidate_index": candidate_index,
    }
    record["quality_score"] = list(_quality_score(record))
    record["eligible"] = _eligible(record)
    return record


def test_budget_sequence_doubles_to_frozen_maximum() -> None:
    assert _budget_sequence(8, 32) == (8, 16, 32)
    assert _budget_sequence(12, 32) == (12, 24, 32)
    with pytest.raises(ValueError, match="candidate budgets"):
        _budget_sequence(16, 8)


def test_missing_vector_task_quality_remains_an_absent_summary() -> None:
    assert _task_quality_from_infos({"task_quality": [None]}) is None
    with pytest.raises(ValueError, match="non-empty mapping"):
        _task_quality_from_infos({"task_quality": [{}]})


def test_candidate_manifest_is_commit_bound_and_resolves_relative_paths(
    tmp_path: Path,
) -> None:
    candidates = [{"candidate_id": "planner", "kind": "planner"}]
    candidates.extend(
        {
            "candidate_id": f"policy-{index}",
            "kind": "policy",
            "policy_path": f"policies/{index}.pt",
            "policy_sha256": f"{index:x}" * 64,
            "stochastic": True,
            "exploration_seed_offset": index,
        }
        for index in range(1, 8)
    )
    payload = {
        "schema_version": CANDIDATE_SCHEMA,
        "task": "t2_trans",
        "rlinf_commit": "a" * 40,
        "benchmark_commit": "b" * 40,
        "candidates": candidates,
    }

    task, specs = _validate_candidate_manifest(
        payload,
        manifest_path=tmp_path / "candidates.json",
        rlinf_commit="a" * 40,
        benchmark_commit="b" * 40,
        max_k=8,
    )

    assert task == "t2_trans"
    assert specs[1].policy_path == (tmp_path / "policies/1.pt").resolve()
    assert _candidate_identity(specs[1])["policy_path"] == str(
        (tmp_path / "policies/1.pt").resolve()
    )
    with pytest.raises(ValueError, match="benchmark commit"):
        _validate_candidate_manifest(
            payload,
            manifest_path=tmp_path / "candidates.json",
            rlinf_commit="a" * 40,
            benchmark_commit="c" * 40,
            max_k=8,
        )
    payload["candidates"] = [*payload["candidates"][1:], payload["candidates"][0]]
    with pytest.raises(ValueError, match="planner at index zero"):
        _validate_candidate_manifest(
            payload,
            manifest_path=tmp_path / "candidates.json",
            rlinf_commit="a" * 40,
            benchmark_commit="b" * 40,
            max_k=8,
        )


def test_winner_selection_is_quality_first_then_stable_candidate_index() -> None:
    first = _record(0)
    same_quality_later = _record(1)
    better_return = _record(2, value=2.0)

    assert _select_winner([same_quality_later, first]) is first
    assert _select_winner([first, better_return]) is better_return
    better_return["safety_failure"] = True
    better_return["eligible"] = _eligible(better_return)
    assert _select_winner([first, better_return]) is first


def test_render_parity_skip_binds_the_selected_attempt_and_recovery_event() -> None:
    selected = {
        **_record(3),
        "candidate_id": "policy-3",
        "attempt_tape": "lightweight/episode-1/candidate-03.npz",
        "attempt_tape_sha256": "a" * 64,
        "action_sha256": "b" * 64,
    }
    error = RuntimeError("winner render parity failed for return")
    result = {
        "reset_index": 7,
        "episode_id": "episode-1",
        "accepted": False,
        "winner_candidate_id": None,
        "winner_candidate_index": None,
        "render_parity_skip": _render_parity_skip(selected, error),
    }
    events = _render_parity_skip_events(
        ["render_parity_skip:reset:7:episode-1:winner render parity failed for return"]
    )

    assert _audit_render_parity_skip(result, selected, events[7]) == "structured-v0.1"
    result["render_parity_skip"]["candidate_id"] = "policy-tampered"
    with pytest.raises(ValueError, match="selected attempt"):
        _audit_render_parity_skip(result, selected, events[7])


def test_legacy_render_parity_skip_requires_one_recognized_recovery_event() -> None:
    selected = {
        **_record(0),
        "candidate_id": "planner",
        "attempt_tape": "lightweight/episode-1/candidate-00.npz",
        "attempt_tape_sha256": "a" * 64,
        "action_sha256": "b" * 64,
    }
    result = {"reset_index": 1, "episode_id": "episode-1", "accepted": False}
    events = _render_parity_skip_events(
        [
            "resume.recovery-1",
            "render_parity_skip:reset:1:episode-1:canonical replay contract mismatch",
        ]
    )

    assert _audit_render_parity_skip(result, selected, events[1]) == "legacy-v0.1"
    with pytest.raises(ValueError, match="no matching recovery event"):
        _audit_render_parity_skip(result, selected, None)
    with pytest.raises(ValueError, match="invalid"):
        _render_parity_skip_events(
            ["render_parity_skip:reset:1:episode-1:unrecognized failure"]
        )


def test_shard_merge_keeps_only_render_skip_events_in_the_sealed_prefix() -> None:
    events = [
        "shard-00.recovery-1",
        "render_parity_skip:reset:3:episode-3:winner render parity failed for return",
        "render_parity_skip:reset:9:episode-9:winner render parity failed for return",
    ]

    assert _kept_recovery_events(events, max_reset=5) == events[:2]
    with pytest.raises(ValueError, match="malformed"):
        _kept_recovery_events(["render_parity_skip:bad"], max_reset=5)


def test_attempt_tape_round_trip_recomputes_shapes_hashes_and_score(tmp_path: Path) -> None:
    steps = 3
    arrays = {
        "states": np.arange((steps + 1) * 5, dtype=np.float32).reshape(steps + 1, 5),
        "policy_actions": np.full((steps, 7), 0.25, dtype=np.float32),
        "actions": np.full((steps, 7), 0.5, dtype=np.float64),
        "rewards": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "terminated": np.asarray([False, False, True], dtype=np.bool_),
        "truncated": np.zeros(steps, dtype=np.bool_),
    }
    relative, tape_sha256 = _write_attempt_tape(
        tmp_path,
        episode_id="episode-1",
        candidate_index=0,
        arrays=arrays,
    )
    replay = {"passed": True, "outcomes_exact": True}
    record = {
        "schema_version": "rlinf-dynamic-benchmark-optimal-attempt-v0.1",
        "task_id": "t2_trans",
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "trajectory_completion": 1.0,
        "completion_time_s": steps * 0.002,
        "return": float(arrays["rewards"].sum(dtype=np.float64)),
        "control_steps": steps,
        "action_l2_sum": float(np.square(arrays["actions"]).sum()),
        "candidate_index": 0,
        "attempt_tape": relative,
        "attempt_tape_sha256": tape_sha256,
        "state_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["states"]).tobytes()
        ).hexdigest(),
        "policy_action_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["policy_actions"]).tobytes()
        ).hexdigest(),
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["actions"]).tobytes()
        ).hexdigest(),
        "reward_sha256": hashlib.sha256(
            np.ascontiguousarray(arrays["rewards"]).tobytes()
        ).hexdigest(),
        "replay_validation": replay,
        "replay_validation_sha256": _audit_payload_sha256(replay),
    }
    record["quality_score"] = list(_quality_score(record))
    record["eligible"] = _eligible(record)

    _audit_attempt_tape(tmp_path, record, expected_task="t2_trans")

    record["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content checksum"):
        _audit_attempt_tape(tmp_path, record, expected_task="t2_trans")


def test_resume_preserves_dirty_tail_and_restores_committed_boundary(tmp_path: Path) -> None:
    output = tmp_path / "export"
    output.mkdir()
    paths = {
        name: output / name
        for name in ("attempts.jsonl", "reset_results.jsonl", "winner_manifest.jsonl")
    }
    committed = {
        "attempts.jsonl": b'{"attempt":0}\n',
        "reset_results.jsonl": b'{"reset":0}\n',
        "winner_manifest.jsonl": b'{"winner":0}\n',
    }
    for name, path in paths.items():
        path.write_bytes(committed[name])
    progress = _progress_payload(
        export_state_sha256="a" * 64,
        started_unix_s=1.0,
        next_reset_index=1,
        accepted_count=1,
        candidate_attempt_count=1,
        budget_histogram={"8": 1},
        attempts_path=paths["attempts.jsonl"],
        reset_results_path=paths["reset_results.jsonl"],
        winners_path=paths["winner_manifest.jsonl"],
        resume_count=0,
        recovery_events=[],
    )
    for path in paths.values():
        with path.open("ab") as stream:
            stream.write(b'{"dirty":true}\n')
    dirty_episode = "episode-1"
    lightweight = output / "lightweight" / dirty_episode
    published = output / "episodes" / "t2_trans" / "train" / dirty_episode
    staging = output / ".staging" / "partial"
    for directory in (lightweight, published, staging):
        directory.mkdir(parents=True)
        (directory / "evidence.bin").write_bytes(b"dirty")
    rows = [
        SimpleNamespace(request=SimpleNamespace(episode_id="episode-0")),
        SimpleNamespace(request=SimpleNamespace(episode_id=dirty_episode)),
    ]

    recovery_name = _recover_partial_output(
        output=output,
        progress=progress,
        reset_rows=rows,
        task="t2_trans",
        split="train",
    )

    assert recovery_name is not None
    recovery = output.parent / recovery_name
    assert recovery.is_dir()
    for name, path in paths.items():
        assert path.read_bytes() == committed[name]
        assert _file_boundary(path) == progress["file_boundaries"][name]
        assert (recovery / name).is_file()
    assert not lightweight.exists()
    assert not published.exists()
    assert not (output / ".staging").exists()
    assert (recovery / "lightweight" / dirty_episode / "evidence.bin").is_file()
    assert (
        recovery
        / "episodes"
        / "t2_trans"
        / "train"
        / dirty_episode
        / "evidence.bin"
    ).is_file()


def test_candidate_restore_uses_canonical_request_reset_and_rearms_hidden_event() -> None:
    request = SimpleNamespace(episode_id="episode-1")

    class RawEnv:
        def __init__(self) -> None:
            self.reset_requests = []

        def reset(self, value):
            self.reset_requests.append(value)
            return "canonical-observation"

    class VectorEnv:
        def __init__(self) -> None:
            self.envs = [RawEnv()]
            self._requests = [None]
            self._raw_observations = [None]
            self._last_obs = None
            self.armed = []

        def load_checkpoint_state(self, state):
            assert state == {"checkpoint": True}
            self._requests[0] = request
            self._raw_observations[0] = "loaded-state-observation"

        def _arm_hidden_t5_event(self, raw_env, value):
            self.armed.append((raw_env, value))

        def _encode(self, observation, value):
            assert observation == "canonical-observation"
            assert value is request
            return np.asarray([1.0, 2.0], dtype=np.float32)

    env = VectorEnv()
    _restore_candidate_start(env, {"checkpoint": True})

    assert env.envs[0].reset_requests == [request]
    assert env.armed == [(env.envs[0], request)]
    assert env._raw_observations == ["canonical-observation"]
    np.testing.assert_array_equal(
        env._last_obs["states"].numpy(),
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    )


def test_replay_proxy_rearms_after_canonical_reset() -> None:
    request = object()

    class RawEnv:
        def reset(self, value):
            assert value is request
            return "observation"

        def step(self, action):
            return ("step", action)

        def save_state(self):
            return b"state"

    class VectorEnv:
        def __init__(self) -> None:
            self.armed = []

        def _arm_hidden_t5_event(self, raw_env, value):
            self.armed.append((raw_env, value))

    raw_env = RawEnv()
    vector_env = VectorEnv()
    proxy = _ArmedResetReplayEnv(vector_env, raw_env)

    assert proxy.reset(request) == "observation"
    assert vector_env.armed == [(raw_env, request)]
    assert proxy.step("action") == ("step", "action")
    assert proxy.save_state() == b"state"
