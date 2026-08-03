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
import pickle
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

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


def _start_method() -> str:
    return (
        "spawn"
        if "spawn" in mp.get_all_start_methods()
        else mp.get_all_start_methods()[0]
    )


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
            SimpleNamespace(
                name="contact",
                physics_step=7,
                time_s=0.07,
                details={"force": 2.5},
            ),
        ) if with_event else ()
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
    assert np.array_equal(cached.privileged["object"], np.asarray([2.0], dtype=np.float32))
    assert cached.events_since_last_observation[0].name == "contact"
    assert restored[0][1]["events_since_last_observation"][0]["name"] == "contact"

    saved = handler.handle("save", [(0, None)])[0][1]
    assert saved["env_state"] == b"saved-state"
    assert np.array_equal(
        saved["observation"]["privileged"]["object"],
        np.asarray([2.0], dtype=np.float32),
    )


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


def test_process_vector_worker_failure_terminates_every_shard() -> None:
    processes = OrderedProcessVector(
        num_envs=4,
        num_workers=2,
        handler_factory=_make_deterministic_handler,
        handler_payload={"offset": 0},
        start_method=_start_method(),
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
