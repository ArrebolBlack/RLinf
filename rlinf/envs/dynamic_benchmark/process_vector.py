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

"""Small ordered multiprocessing primitive for Dynamic Benchmark environments."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from typing import Any, Protocol


class ProcessVectorHandler(Protocol):
    """Protocol implemented inside each persistent environment subprocess."""

    def ready_metadata(self) -> dict[str, Any]:
        """Return immutable worker metadata after local environments initialize."""

    def handle(
        self, command: str, items: list[tuple[int, Any]]
    ) -> list[tuple[int, Any]]:
        """Execute one batched command for the worker's environment indices."""

    def close(self) -> None:
        """Release worker-local environments."""


def _arm_linux_parent_death_signal() -> None:
    """Ask Linux to terminate a worker if its immediate owner disappears."""

    if sys.platform != "linux":
        return
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != parent_pid:
        os._exit(98)


def _worker_main(
    connection: Any,
    handler_factory: Callable[[Any, tuple[int, ...]], ProcessVectorHandler],
    handler_payload: Any,
    indices: tuple[int, ...],
) -> None:
    """Own one deterministic shard until shutdown or an unrecoverable failure."""

    handler: ProcessVectorHandler | None = None
    try:
        _arm_linux_parent_death_signal()
        handler = handler_factory(handler_payload, indices)
        connection.send(
            {
                "kind": "ready",
                "pid": os.getpid(),
                "indices": indices,
                "metadata": handler.ready_metadata(),
            }
        )
        while True:
            message = connection.recv()
            command = str(message["command"])
            if command == "__close__":
                handler.close()
                handler = None
                connection.send({"kind": "closed", "pid": os.getpid()})
                return
            if command == "__crash_for_test__":
                os._exit(97)
            items = list(message["items"])
            result = handler.handle(command, items)
            connection.send(
                {
                    "kind": "result",
                    "command": command,
                    "items": result,
                }
            )
    except BaseException as error:
        try:
            connection.send(
                {
                    "kind": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        except BaseException:
            pass
        if handler is not None:
            try:
                handler.close()
            except BaseException:
                pass
        raise
    finally:
        connection.close()


class OrderedProcessVector:
    """Persistent process shards with deterministic env-index result ordering.

    The caller remains the sole owner of scheduling and manifest order.  Each
    subprocess owns a fixed set of environment indices, and every operation is
    dispatched to all affected shards before any result is collected.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        num_workers: int,
        handler_factory: Callable[[Any, tuple[int, ...]], ProcessVectorHandler],
        handler_payload: Any,
        start_method: str = "spawn",
        timeout_s: float = 120.0,
    ) -> None:
        if num_envs < 1:
            raise ValueError("ordered process vector requires at least one environment")
        if num_workers < 1:
            raise ValueError("ordered process vector requires at least one worker")
        if timeout_s <= 0.0:
            raise ValueError("ordered process vector timeout must be positive")
        available = mp.get_all_start_methods()
        if start_method not in available:
            raise ValueError(
                f"multiprocessing start method {start_method!r} is unavailable; available={available}"
            )
        self.num_envs = int(num_envs)
        self.num_workers = min(int(num_workers), self.num_envs)
        self.start_method = start_method
        self.timeout_s = float(timeout_s)
        self._closed = False
        self._connections: list[Any] = []
        self._processes: list[Any] = []
        self._indices_by_worker: list[tuple[int, ...]] = []
        self._worker_for_index: dict[int, int] = {}
        self.ready_metadata: list[dict[str, Any]] = []

        shards: list[list[int]] = [[] for _ in range(self.num_workers)]
        for index in range(self.num_envs):
            shards[index % self.num_workers].append(index)
        context = mp.get_context(start_method)
        try:
            for worker_index, shard in enumerate(shards):
                parent_connection, child_connection = context.Pipe(duplex=True)
                indices = tuple(shard)
                process = context.Process(
                    target=_worker_main,
                    args=(
                        child_connection,
                        handler_factory,
                        handler_payload,
                        indices,
                    ),
                    name=f"dynamic-benchmark-env-{worker_index}",
                    daemon=True,
                )
                process.start()
                child_connection.close()
                self._connections.append(parent_connection)
                self._processes.append(process)
                self._indices_by_worker.append(indices)
                for index in indices:
                    self._worker_for_index[index] = worker_index
            ready = self._receive_all(range(self.num_workers), expected_kind="ready")
            self.ready_metadata = [message["metadata"] for message in ready]
        except BaseException:
            self._abort()
            raise

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """Return child PIDs in stable worker-shard order."""

        return tuple(
            int(process.pid) for process in self._processes if process.pid is not None
        )

    @property
    def alive_pids(self) -> tuple[int, ...]:
        """Return currently live child PIDs."""

        return tuple(
            int(process.pid)
            for process in self._processes
            if process.pid is not None and process.is_alive()
        )

    @property
    def closed(self) -> bool:
        """Whether the process group has been closed or aborted."""

        return self._closed

    def run(
        self, command: str, items: Iterable[tuple[int, Any]]
    ) -> list[tuple[int, Any]]:
        """Run a command concurrently and restore results to env-index order."""

        if self._closed:
            raise RuntimeError("ordered process vector is closed")
        grouped: list[list[tuple[int, Any]]] = [[] for _ in range(self.num_workers)]
        seen: set[int] = set()
        for raw_index, payload in items:
            index = int(raw_index)
            if index not in self._worker_for_index:
                raise IndexError(f"environment index {index} is out of range")
            if index in seen:
                raise ValueError(f"environment index {index} was dispatched twice")
            seen.add(index)
            grouped[self._worker_for_index[index]].append((index, payload))
        active_workers = [index for index, batch in enumerate(grouped) if batch]
        try:
            for worker_index in active_workers:
                self._connections[worker_index].send(
                    {"command": command, "items": grouped[worker_index]}
                )
            messages = self._receive_all(active_workers, expected_kind="result")
            results: list[tuple[int, Any]] = []
            for worker_index, message in zip(active_workers, messages, strict=True):
                if message.get("command") != command:
                    raise RuntimeError(
                        "process vector worker replied to the wrong command"
                    )
                worker_results = list(message["items"])
                expected = {index for index, _ in grouped[worker_index]}
                observed = {int(index) for index, _ in worker_results}
                if observed != expected or len(worker_results) != len(expected):
                    raise RuntimeError(
                        "process vector worker returned the wrong environment indices"
                    )
                results.extend(
                    (int(index), payload) for index, payload in worker_results
                )
            return sorted(results, key=lambda item: item[0])
        except BaseException:
            self._abort()
            raise

    def crash_worker_for_test(self, worker_index: int = 0) -> None:
        """Crash one child and assert the fail-closed cleanup path in tests."""

        if self._closed:
            raise RuntimeError("ordered process vector is closed")
        if not 0 <= worker_index < self.num_workers:
            raise IndexError("worker index is out of range")
        try:
            self._connections[worker_index].send(
                {"command": "__crash_for_test__", "items": []}
            )
            self._receive_all([worker_index], expected_kind="result")
        except BaseException:
            self._abort()
            raise

    def _receive_all(
        self,
        worker_indices: Iterable[int],
        *,
        expected_kind: str,
    ) -> list[dict[str, Any]]:
        indices = list(worker_indices)
        deadline = time.monotonic() + self.timeout_s
        messages = []
        for worker_index in indices:
            process = self._processes[worker_index]
            connection = self._connections[worker_index]
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not connection.poll(remaining):
                exit_code = process.exitcode
                raise RuntimeError(
                    "process vector worker timed out or exited before replying: "
                    f"worker={worker_index}, pid={process.pid}, exitcode={exit_code}"
                )
            try:
                message = connection.recv()
            except (EOFError, OSError) as error:
                raise RuntimeError(
                    "process vector worker connection closed unexpectedly: "
                    f"worker={worker_index}, pid={process.pid}, exitcode={process.exitcode}"
                ) from error
            if message.get("kind") == "error":
                raise RuntimeError(
                    "process vector worker failed: "
                    f"worker={worker_index}, pid={process.pid}, "
                    f"{message.get('error_type')}: {message.get('error')}\n"
                    f"{message.get('traceback', '')}"
                )
            if message.get("kind") != expected_kind:
                raise RuntimeError(
                    "process vector worker returned an unexpected response: "
                    f"expected={expected_kind}, observed={message.get('kind')}"
                )
            messages.append(message)
        return messages

    def _abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass

    def close(self) -> None:
        """Gracefully close all workers; fall back to bounded termination."""

        if self._closed:
            return
        try:
            active = []
            for worker_index, process in enumerate(self._processes):
                if process.is_alive():
                    self._connections[worker_index].send(
                        {"command": "__close__", "items": []}
                    )
                    active.append(worker_index)
            if active:
                self._receive_all(active, expected_kind="closed")
            self._closed = True
            for process in self._processes:
                process.join(timeout=5.0)
            if any(process.is_alive() for process in self._processes):
                self._closed = False
                self._abort()
            for connection in self._connections:
                connection.close()
        except BaseException:
            self._abort()
            raise

    def __enter__(self) -> OrderedProcessVector:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            try:
                self._abort()
            except BaseException:
                pass
