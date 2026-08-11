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
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from se3_wam.benchmark.trajectory_quality_v4 import (
    QUALITY_V4_PHYSICS_REDUCER_SCHEMA,
    PhysicsRateEEFReducer,
)

from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import (
    DynamicBenchmarkEnv,
    _DynamicBenchmarkProcessHandler,
    _observation_payload_sha256,
    _pack_process_observation,
)
from rlinf.envs.dynamic_benchmark.process_vector import OrderedProcessVector


class _DeterministicHandler:
    def __init__(self, payload: dict[str, int], indices: tuple[int, ...]) -> None:
        self._offset = int(payload["offset"])
        self._indices = indices
        self._states = dict.fromkeys(indices, 0)

    def ready_metadata(self) -> dict[str, Any]:
        return {"indices": self._indices, "horizons": dict.fromkeys(self._indices, 100)}

    def handle(
        self,
        command: str,
        items: list[tuple[int, Any]],
    ) -> list[tuple[int, Any]]:
        results = []
        for index, payload in items:
            if command == "reset":
                self._states[index] = self._offset + int(payload)
            elif command == "step":
                self._states[index] += int(payload)
            elif command == "save":
                pass
            elif command == "restore":
                self._states[index] = int(payload)
            else:
                raise ValueError(f"unknown command {command}")
            results.append((index, self._states[index]))
        return results

    def close(self) -> None:
        self._states.clear()


def _make_deterministic_handler(
    payload: dict[str, int],
    indices: tuple[int, ...],
) -> _DeterministicHandler:
    return _DeterministicHandler(payload, indices)


class _ReducerHandler:
    """Exercise the real Qv4 reducer snapshot across spawned worker boundaries."""

    def __init__(self, payload: dict[str, int], indices: tuple[int, ...]) -> None:
        del payload
        self._indices = indices
        self._reducers = {index: PhysicsRateEEFReducer() for index in indices}

    def ready_metadata(self) -> dict[str, Any]:
        return {"indices": self._indices}

    @staticmethod
    def _pose(position_x: float) -> np.ndarray:
        return np.asarray([position_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def handle(
        self, command: str, items: list[tuple[int, Any]]
    ) -> list[tuple[int, Any]]:
        results = []
        for index, payload in items:
            if command == "reset":
                reducer = PhysicsRateEEFReducer()
                reducer.update(time_s=0.0, eef_pose_xyzw=self._pose(0.0))
                self._reducers[index] = reducer
                result = reducer.summary()
            elif command == "step":
                time_s, position_x = payload
                self._reducers[index].update(
                    time_s=float(time_s),
                    eef_pose_xyzw=self._pose(float(position_x)),
                )
                result = self._reducers[index].summary()
            elif command == "save":
                result = self._reducers[index].snapshot()
            elif command == "restore":
                self._reducers[index] = PhysicsRateEEFReducer.from_snapshot(payload)
                result = self._reducers[index].summary()
            else:
                raise ValueError(f"unknown reducer command {command}")
            results.append((index, result))
        return results

    def close(self) -> None:
        self._reducers.clear()


def _make_reducer_handler(
    payload: dict[str, int], indices: tuple[int, ...]
) -> _ReducerHandler:
    return _ReducerHandler(payload, indices)


class _SlowHandler(_DeterministicHandler):
    def handle(
        self,
        command: str,
        items: list[tuple[int, Any]],
    ) -> list[tuple[int, Any]]:
        time.sleep(0.5)
        return super().handle(command, items)


def _make_slow_handler(
    payload: dict[str, int],
    indices: tuple[int, ...],
) -> _SlowHandler:
    return _SlowHandler(payload, indices)


def _make_partially_failing_handler(
    payload: dict[str, int],
    indices: tuple[int, ...],
) -> _DeterministicHandler:
    if 1 in indices:
        raise RuntimeError("intentional partial startup failure")
    return _DeterministicHandler(payload, indices)


def _parent_process_with_workers(connection: Any, start_method: str) -> None:
    processes = OrderedProcessVector(
        num_envs=4,
        num_workers=2,
        handler_factory=_make_deterministic_handler,
        handler_payload={"offset": 0},
        start_method=start_method,
        timeout_s=5.0,
    )
    connection.send(processes.worker_pids)
    connection.close()
    while True:
        time.sleep(1.0)


def _pid_exists(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


def _start_method() -> str:
    return (
        "spawn"
        if "spawn" in mp.get_all_start_methods()
        else mp.get_all_start_methods()[0]
    )


def _failure_start_methods() -> tuple[str, ...]:
    available = mp.get_all_start_methods()
    selected = tuple(
        method for method in ("spawn", "forkserver") if method in available
    )
    return selected or (available[0],)


def _digest(rows: list[tuple[int, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_process_observation_transport_removes_mapping_proxies() -> None:
    event = SimpleNamespace(
        name="contact",
        physics_step=3,
        time_s=0.01,
        details=MappingProxyType({"force": 1.5}),
    )
    observation = SimpleNamespace(
        episode_id="episode-1",
        task_id="t4_sphere",
        physics_step=3,
        control_step=1,
        policy_step=1,
        time_s=0.01,
        rgb=MappingProxyType({}),
        depth_m=MappingProxyType({}),
        segmentation=MappingProxyType({}),
        proprio=MappingProxyType({"joint": [1.0]}),
        privileged=MappingProxyType({"object": [2.0]}),
        events_since_last_observation=(event,),
        api_version="db-api-v0.1",
    )

    packed = _pack_process_observation(observation)
    transport = pickle.loads(pickle.dumps(packed))
    env = object.__new__(DynamicBenchmarkEnv)
    env._EventRecord = lambda **kwargs: SimpleNamespace(**kwargs)
    env._ObservationBundle = lambda **kwargs: SimpleNamespace(**kwargs)
    restored = env._unpack_process_observation(transport)

    assert restored.episode_id == observation.episode_id
    assert restored.proprio == {"joint": [1.0]}
    assert restored.events_since_last_observation[0].details == {"force": 1.5}

    original_sha256 = _observation_payload_sha256(packed)
    assert _observation_payload_sha256(transport) == original_sha256
    transport["privileged"]["object"][0] = 3.0
    assert _observation_payload_sha256(transport) != original_sha256


def test_process_restore_installs_checkpoint_observation_in_worker_cache() -> None:
    def observation(*, value: float, with_event: bool) -> SimpleNamespace:
        events = (
            (
                SimpleNamespace(
                    name="contact",
                    physics_step=7,
                    time_s=0.07,
                    details={"force": 2.5},
                ),
            )
            if with_event
            else ()
        )
        return SimpleNamespace(
            episode_id="episode-restore",
            task_id="t4_sphere",
            physics_step=7,
            control_step=3,
            policy_step=3,
            time_s=0.07,
            rgb={},
            depth_m={},
            segmentation={},
            proprio={"joint": np.asarray([1.0], dtype=np.float32)},
            privileged={"object": np.asarray([value], dtype=np.float32)},
            events_since_last_observation=events,
            api_version="db-api-v0.1",
        )

    authoritative = observation(value=2.0, with_event=True)
    reencoded = observation(value=999.0, with_event=False)

    class FakeCanonicalEnv:
        def reset(self, request) -> None:
            self.request = request

        def load_state(self, env_state):
            assert env_state == b"env-state"
            return reencoded

        def save_state(self):
            return b"saved-state"

    handler = object.__new__(_DynamicBenchmarkProcessHandler)
    handler._envs = {0: FakeCanonicalEnv()}
    handler._observations = {}
    handler._ObservationBundle = lambda **kwargs: SimpleNamespace(**kwargs)
    handler._EventRecord = lambda **kwargs: SimpleNamespace(**kwargs)
    handler._arm_hidden_t5_event = lambda env, request: None
    handler._reset_residual_planner = lambda index, request: None
    request = SimpleNamespace(episode_id="episode-restore", task_id="t4_sphere")

    restored = handler.handle(
        "restore",
        [(0, (request, b"env-state", _pack_process_observation(authoritative)))],
    )
    cached = handler._observations[0]
    assert np.array_equal(
        cached.privileged["object"], np.asarray([2.0], dtype=np.float32)
    )
    assert cached.events_since_last_observation[0].name == "contact"
    assert restored[0][1]["events_since_last_observation"][0]["name"] == "contact"

    saved = handler.handle("save", [(0, None)])[0][1]
    assert saved["env_state"] == b"saved-state"
    assert np.array_equal(
        saved["observation"]["privileged"]["object"],
        np.asarray([2.0], dtype=np.float32),
    )


def test_process_step_transports_terminal_quality_v4_physics_summary() -> None:
    summary = {
        "schema_version": QUALITY_V4_PHYSICS_REDUCER_SCHEMA,
        "sample_count": 4,
    }
    step_result = SimpleNamespace(
        observation=SimpleNamespace(
            episode_id="episode-qv4",
            task_id="t4_can",
            physics_step=3,
            control_step=1,
            policy_step=1,
            time_s=0.006,
            rgb={},
            depth_m={},
            segmentation={},
            proprio={},
            privileged={},
            events_since_last_observation=(),
            api_version="db-api-v0.1",
        ),
        terminated=True,
        truncated=False,
        success=True,
        termination_reason="success",
        active_stage_progress=1.0,
        task_quality=None,
        trajectory_quality_v4_physics=summary,
    )
    env = SimpleNamespace(_ledger=SimpleNamespace(events=()))

    payload = _DynamicBenchmarkProcessHandler._pack_step_result(env, step_result)

    assert payload["trajectory_quality_v4_physics"] == summary
    summary["sample_count"] = 999
    assert payload["trajectory_quality_v4_physics"]["sample_count"] == 4


def test_process_residual_composition_matches_frozen_float32_contract() -> None:
    planner = np.asarray([0.9, -0.9, 0.0, 0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    residual = np.asarray([1.0, -1.0, 0.4, -0.4, 0.0, 0.8, -0.8], dtype=np.float64)

    observed = _DynamicBenchmarkProcessHandler._compose_residual_action(
        planner, residual, 0.25
    )

    assert observed.dtype == np.float32
    assert np.array_equal(
        observed,
        np.asarray([1.0, -1.0, 0.1, 0.0, -0.1, 0.4, -0.4], dtype=np.float32),
    )


def test_process_vector_matches_serial_and_restores_reset_order() -> None:
    resets = [(3, 30), (0, 0), (2, 20), (1, 10)]
    serial = sorted((index, 7 + value) for index, value in resets)

    with OrderedProcessVector(
        num_envs=4,
        num_workers=2,
        handler_factory=_make_deterministic_handler,
        handler_payload={"offset": 7},
        start_method=_start_method(),
    ) as processes:
        observed = processes.run("reset", resets)
        stepped = processes.run("step", [(3, 4), (1, 2), (0, 1), (2, 3)])

    assert observed == serial
    assert _digest(observed) == _digest(serial)
    assert stepped == [(0, 8), (1, 19), (2, 30), (3, 41)]


def test_process_vector_checkpoint_resume_is_exact() -> None:
    common = {
        "num_envs": 4,
        "num_workers": 2,
        "handler_factory": _make_deterministic_handler,
        "handler_payload": {"offset": 5},
        "start_method": _start_method(),
    }
    with OrderedProcessVector(**common) as continuous:
        continuous.run("reset", [(index, index * 10) for index in range(4)])
        continuous.run("step", [(index, index + 1) for index in range(4)])
        checkpoint = continuous.run("save", [(index, None) for index in range(4)])
        expected = continuous.run("step", [(index, 11 - index) for index in range(4)])

    with OrderedProcessVector(**common) as resumed:
        restored = resumed.run("restore", checkpoint)
        observed = resumed.run("step", [(index, 11 - index) for index in range(4)])

    assert restored == checkpoint
    assert observed == expected


def test_process_vector_quality_v4_reducer_checkpoint_resume_is_exact() -> None:
    common = {
        "num_envs": 4,
        "num_workers": 2,
        "handler_factory": _make_reducer_handler,
        "handler_payload": {},
        "start_method": _start_method(),
    }
    all_indices = [(index, None) for index in range(4)]
    first = [(index, (0.002, 0.001 + index * 1.0e-5)) for index in range(4)]
    second = [(index, (0.004, 0.002 + index * 1.0e-5)) for index in range(4)]
    third = [(index, (0.006, 0.004 + index * 1.0e-5)) for index in range(4)]
    fourth = [(index, (0.008, 0.005 + index * 1.0e-5)) for index in range(4)]
    with OrderedProcessVector(**common) as continuous:
        continuous.run("reset", all_indices)
        continuous.run("step", first)
        continuous.run("step", second)
        checkpoint = continuous.run("save", all_indices)
        continuous.run("step", third)
        expected = continuous.run("step", fourth)

    with OrderedProcessVector(**common) as resumed:
        resumed.run("restore", checkpoint)
        resumed.run("step", third)
        observed = resumed.run("step", fourth)

    assert observed == expected
    assert all(
        summary["schema_version"] == QUALITY_V4_PHYSICS_REDUCER_SCHEMA
        and summary["sample_count"] == 5
        for _, summary in observed
    )


@pytest.mark.parametrize("start_method", _failure_start_methods())
def test_process_vector_worker_failure_terminates_every_shard(
    start_method: str,
) -> None:
    processes = OrderedProcessVector(
        num_envs=4,
        num_workers=2,
        handler_factory=_make_deterministic_handler,
        handler_payload={"offset": 0},
        start_method=start_method,
        timeout_s=5.0,
    )
    pids = processes.worker_pids

    with pytest.raises(RuntimeError, match="exited|closed"):
        processes.crash_worker_for_test()

    assert processes.closed
    assert processes.alive_pids == ()
    assert len(pids) == 2
    time.sleep(0.05)
    assert all(not process.is_alive() for process in processes._processes)


@pytest.mark.parametrize("start_method", _failure_start_methods())
def test_process_vector_timeout_terminates_every_shard(
    start_method: str,
) -> None:
    processes = OrderedProcessVector(
        num_envs=4,
        num_workers=2,
        handler_factory=_make_slow_handler,
        handler_payload={"offset": 0},
        start_method=start_method,
        timeout_s=5.0,
    )
    processes.timeout_s = 0.05

    with pytest.raises(RuntimeError, match="timed out"):
        processes.run("step", [(index, 1) for index in range(4)])

    assert processes.closed
    assert processes.alive_pids == ()


@pytest.mark.parametrize("start_method", _failure_start_methods())
def test_process_vector_partial_startup_terminates_started_shards(
    start_method: str,
) -> None:
    prior_children = {child.pid for child in mp.active_children()}

    with pytest.raises(RuntimeError, match="partial startup failure"):
        OrderedProcessVector(
            num_envs=4,
            num_workers=2,
            handler_factory=_make_partially_failing_handler,
            handler_payload={"offset": 0},
            start_method=start_method,
            timeout_s=5.0,
        )

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        remaining = {
            child.pid
            for child in mp.active_children()
            if child.pid not in prior_children
        }
        if not remaining:
            break
        time.sleep(0.05)
    assert not remaining


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death contract")
@pytest.mark.parametrize("worker_start_method", _failure_start_methods())
def test_process_vector_parent_sigkill_leaves_no_workers(
    worker_start_method: str,
) -> None:
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    parent = context.Process(
        target=_parent_process_with_workers,
        args=(send_connection, worker_start_method),
    )
    parent.start()
    send_connection.close()
    worker_pids: tuple[int, ...] = ()
    try:
        assert receive_connection.poll(10.0)
        worker_pids = tuple(int(pid) for pid in receive_connection.recv())
        assert len(worker_pids) == 2
        assert all(_pid_exists(pid) for pid in worker_pids)
        os.kill(int(parent.pid), signal.SIGKILL)
        parent.join(timeout=5.0)
        assert not parent.is_alive()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(
            _pid_exists(pid) for pid in worker_pids
        ):
            time.sleep(0.05)
        assert all(not _pid_exists(pid) for pid in worker_pids)
    finally:
        receive_connection.close()
        if parent.is_alive():
            parent.kill()
            parent.join(timeout=5.0)
