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

"""Preallocated CUDA replay storage with device-local sampling.

The ring accepts fixed-size transition batches, including post-terminal padding,
and stores an explicit device validity mask.  Sampling uses a CUDA generator and
CUDA multinomial indices, so neither insertion nor sampling materializes tensor
data on the host.  Host copies are confined to the checkpoint boundary methods.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


class DeviceReplayContractError(ValueError):
    """Raised when a replay tensor violates the CUDA data-plane contract."""


@dataclass(frozen=True)
class DeviceReplayBatch:
    """One sampled transition batch whose fields all remain on one GPU."""

    observation: Any
    action: Any
    reward: Any
    next_observation: Any
    terminated: Any
    truncated: Any

    @property
    def done(self) -> Any:
        return self.terminated | self.truncated

    @property
    def rows(self) -> int:
        return int(self.observation.shape[0])


@dataclass(frozen=True)
class DeviceImitationBatch:
    """One sampled observation/action label batch on a single GPU."""

    observation: Any
    action: Any

    @property
    def rows(self) -> int:
        return int(self.observation.shape[0])


class DeviceReplayBuffer:
    """Own a fixed-capacity CUDA transition ring and CUDA sampling RNG."""

    FIELDS = (
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
    )
    SCHEMA_VERSION = "rlinf-device-replay-v0.1"

    def __init__(
        self,
        *,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: Any,
        seed: int,
        observation_dtype: Any,
        action_dtype: Any,
        reward_dtype: Any,
        torch_module: Any | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name, shape in (
            ("observation_shape", observation_shape),
            ("action_shape", action_shape),
        ):
            if not isinstance(shape, tuple) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in shape
            ):
                raise ValueError(f"{name} must contain positive integer dimensions")
        torch = (
            importlib.import_module("torch") if torch_module is None else torch_module
        )
        if str(device).lower().split(":", 1)[0] != "cuda":
            raise DeviceReplayContractError("device replay storage requires CUDA")

        self._torch = torch
        self._capacity = capacity
        self._observation_shape = observation_shape
        self._action_shape = action_shape
        self._device = device
        self._observation_dtype = observation_dtype
        self._action_dtype = action_dtype
        self._reward_dtype = reward_dtype
        self._cursor = 0
        self._size = 0
        self._inserted_rows = 0
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._observation = torch.empty(
            (capacity, *observation_shape), dtype=observation_dtype, device=device
        )
        self._action = torch.empty(
            (capacity, *action_shape), dtype=action_dtype, device=device
        )
        self._reward = torch.empty((capacity,), dtype=reward_dtype, device=device)
        self._next_observation = torch.empty(
            (capacity, *observation_shape), dtype=observation_dtype, device=device
        )
        self._terminated = torch.empty((capacity,), dtype=torch.bool, device=device)
        self._truncated = torch.empty((capacity,), dtype=torch.bool, device=device)
        self._valid = torch.zeros((capacity,), dtype=torch.bool, device=device)
        self._sampling_weight = torch.zeros(
            (capacity,), dtype=torch.float32, device=device
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def inserted_rows(self) -> int:
        return self._inserted_rows

    @property
    def device(self) -> Any:
        return self._device

    def _require_tensor(
        self,
        value: Any,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> None:
        if tuple(value.shape) != shape:
            raise DeviceReplayContractError(f"{name} must have shape {shape}")
        if value.dtype != dtype:
            raise DeviceReplayContractError(f"{name} has the wrong dtype")
        if value.device != self._device:
            raise DeviceReplayContractError(f"{name} is not on {self._device}")
        if not value.is_contiguous():
            raise DeviceReplayContractError(f"{name} must be contiguous")

    def add_batch(
        self,
        *,
        observation: Any,
        action: Any,
        reward: Any,
        next_observation: Any,
        terminated: Any,
        truncated: Any,
        valid: Any,
    ) -> None:
        """Insert a fixed batch without reading tensor values on the host."""

        rows = int(observation.shape[0])
        if rows < 1:
            raise DeviceReplayContractError("replay insertion batch must not be empty")
        specifications = (
            (
                "observation",
                observation,
                (rows, *self._observation_shape),
                self._observation_dtype,
            ),
            ("action", action, (rows, *self._action_shape), self._action_dtype),
            ("reward", reward, (rows,), self._reward_dtype),
            (
                "next_observation",
                next_observation,
                (rows, *self._observation_shape),
                self._observation_dtype,
            ),
            ("terminated", terminated, (rows,), self._torch.bool),
            ("truncated", truncated, (rows,), self._torch.bool),
            ("valid", valid, (rows,), self._torch.bool),
        )
        for name, value, shape, dtype in specifications:
            self._require_tensor(value, name=name, shape=shape, dtype=dtype)

        payload = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "terminated": terminated,
            "truncated": truncated,
            "valid": valid,
        }
        retained_rows = min(rows, self._capacity)
        skipped_rows = rows - retained_rows
        write_cursor = (self._cursor + skipped_rows) % self._capacity
        first_rows = min(retained_rows, self._capacity - write_cursor)
        for name, value in payload.items():
            retained = value[skipped_rows:]
            storage = getattr(self, f"_{name}")
            storage[write_cursor : write_cursor + first_rows].copy_(
                retained[:first_rows]
            )
            if first_rows < retained_rows:
                storage[: retained_rows - first_rows].copy_(retained[first_rows:])
        retained_valid = valid[skipped_rows:].to(dtype=self._torch.float32)
        self._sampling_weight[write_cursor : write_cursor + first_rows].copy_(
            retained_valid[:first_rows]
        )
        if first_rows < retained_rows:
            self._sampling_weight[: retained_rows - first_rows].copy_(
                retained_valid[first_rows:]
            )
        self._cursor = (self._cursor + rows) % self._capacity
        self._size = min(self._capacity, self._size + rows)
        self._inserted_rows += rows

    def sample(self, batch_size: int) -> DeviceReplayBatch:
        """Sample valid rows with replacement using only device-side operations."""

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if self._size < 1:
            raise RuntimeError("cannot sample an empty replay")
        indices = self._torch.multinomial(
            self._sampling_weight[: self._size],
            batch_size,
            replacement=True,
            generator=self._generator,
        )
        return DeviceReplayBatch(
            observation=self._observation.index_select(0, indices),
            action=self._action.index_select(0, indices),
            reward=self._reward.index_select(0, indices),
            next_observation=self._next_observation.index_select(0, indices),
            terminated=self._terminated.index_select(0, indices),
            truncated=self._truncated.index_select(0, indices),
        )

    def state_dict(self) -> dict[str, Any]:
        """Copy replay contents and CUDA RNG state to the host at a checkpoint boundary."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self._capacity,
            "observation_shape": self._observation_shape,
            "action_shape": self._action_shape,
            "cursor": self._cursor,
            "size": self._size,
            "inserted_rows": self._inserted_rows,
            "generator_state": self._generator.get_state().cpu().clone(),
            "data": {
                name: getattr(self, f"_{name}")[: self._size].detach().cpu().clone()
                for name in (*self.FIELDS, "valid")
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore an exact checkpoint, rejecting capacity or schema drift."""

        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("replay checkpoint schema mismatch")
        expected_identity = {
            "capacity": self._capacity,
            "observation_shape": self._observation_shape,
            "action_shape": self._action_shape,
        }
        observed_identity = {
            "capacity": int(state["capacity"]),
            "observation_shape": tuple(state["observation_shape"]),
            "action_shape": tuple(state["action_shape"]),
        }
        if observed_identity != expected_identity:
            raise ValueError(
                "replay checkpoint identity does not match configured ring"
            )
        size = int(state["size"])
        cursor = int(state["cursor"])
        inserted_rows = int(state["inserted_rows"])
        if not 0 <= size <= self._capacity:
            raise ValueError("replay checkpoint size is invalid")
        if not 0 <= cursor < self._capacity:
            raise ValueError("replay checkpoint cursor is invalid")
        if inserted_rows < size:
            raise ValueError("replay checkpoint inserted-row counter is invalid")
        data = state.get("data")
        if not isinstance(data, dict) or set(data) != {*self.FIELDS, "valid"}:
            raise ValueError("replay checkpoint data fields drifted")
        for name in (*self.FIELDS, "valid"):
            target = getattr(self, f"_{name}")[:size]
            source = data[name].to(device=self._device, dtype=target.dtype)
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"replay checkpoint field {name} has the wrong shape")
            target.copy_(source)
        if size < self._capacity:
            self._valid[size:].fill_(False)
            self._sampling_weight[size:].fill_(0.0)
        self._sampling_weight[:size].copy_(
            self._valid[:size].to(dtype=self._torch.float32)
        )
        self._cursor = cursor
        self._size = size
        self._inserted_rows = inserted_rows
        self._generator.set_state(state["generator_state"])


class DeviceImitationReplayBuffer:
    """Own CUDA-only observation/action labels without transition semantics."""

    FIELDS = ("observation", "action")
    SCHEMA_VERSION = "rlinf-device-imitation-replay-v0.1"

    def __init__(
        self,
        *,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: Any,
        seed: int,
        observation_dtype: Any,
        action_dtype: Any,
        torch_module: Any | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name, shape in (
            ("observation_shape", observation_shape),
            ("action_shape", action_shape),
        ):
            if not isinstance(shape, tuple) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in shape
            ):
                raise ValueError(f"{name} must contain positive integer dimensions")
        torch = (
            importlib.import_module("torch") if torch_module is None else torch_module
        )
        if str(device).lower().split(":", 1)[0] != "cuda":
            raise DeviceReplayContractError("device imitation storage requires CUDA")

        self._torch = torch
        self._capacity = capacity
        self._observation_shape = observation_shape
        self._action_shape = action_shape
        self._device = device
        self._observation_dtype = observation_dtype
        self._action_dtype = action_dtype
        self._cursor = 0
        self._size = 0
        self._inserted_rows = 0
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._observation = torch.empty(
            (capacity, *observation_shape), dtype=observation_dtype, device=device
        )
        self._action = torch.empty(
            (capacity, *action_shape), dtype=action_dtype, device=device
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def inserted_rows(self) -> int:
        return self._inserted_rows

    @property
    def device(self) -> Any:
        return self._device

    def _require_tensor(
        self,
        value: Any,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> None:
        if tuple(value.shape) != shape:
            raise DeviceReplayContractError(f"{name} must have shape {shape}")
        if value.dtype != dtype:
            raise DeviceReplayContractError(f"{name} has the wrong dtype")
        if value.device != self._device:
            raise DeviceReplayContractError(f"{name} is not on {self._device}")
        if not value.is_contiguous():
            raise DeviceReplayContractError(f"{name} must be contiguous")

    def add_batch(self, *, observation: Any, action: Any) -> None:
        """Insert already-selected label rows without any host tensor read."""

        rows = int(observation.shape[0])
        if rows < 1:
            raise DeviceReplayContractError(
                "imitation insertion batch must not be empty"
            )
        self._require_tensor(
            observation,
            name="observation",
            shape=(rows, *self._observation_shape),
            dtype=self._observation_dtype,
        )
        self._require_tensor(
            action,
            name="action",
            shape=(rows, *self._action_shape),
            dtype=self._action_dtype,
        )
        retained_rows = min(rows, self._capacity)
        skipped_rows = rows - retained_rows
        write_cursor = (self._cursor + skipped_rows) % self._capacity
        first_rows = min(retained_rows, self._capacity - write_cursor)
        for storage, value in (
            (self._observation, observation),
            (self._action, action),
        ):
            retained = value[skipped_rows:]
            storage[write_cursor : write_cursor + first_rows].copy_(
                retained[:first_rows]
            )
            if first_rows < retained_rows:
                storage[: retained_rows - first_rows].copy_(retained[first_rows:])
        self._cursor = (self._cursor + rows) % self._capacity
        self._size = min(self._capacity, self._size + rows)
        self._inserted_rows += rows

    def sample(self, batch_size: int) -> DeviceImitationBatch:
        """Sample labels with replacement using only the CUDA RNG."""

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if self._size < 1:
            raise RuntimeError("cannot sample an empty imitation replay")
        indices = self._torch.randint(
            self._size,
            (batch_size,),
            device=self._device,
            generator=self._generator,
        )
        return DeviceImitationBatch(
            observation=self._observation.index_select(0, indices),
            action=self._action.index_select(0, indices),
        )

    def state_dict(self) -> dict[str, Any]:
        """Copy labels and CUDA RNG state to the host at a checkpoint boundary."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self._capacity,
            "observation_shape": self._observation_shape,
            "action_shape": self._action_shape,
            "cursor": self._cursor,
            "size": self._size,
            "inserted_rows": self._inserted_rows,
            "generator_state": self._generator.get_state().cpu().clone(),
            "data": {
                name: getattr(self, f"_{name}")[: self._size].detach().cpu().clone()
                for name in self.FIELDS
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact labels, rejecting capacity or schema drift."""

        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("imitation replay checkpoint schema mismatch")
        expected_identity = {
            "capacity": self._capacity,
            "observation_shape": self._observation_shape,
            "action_shape": self._action_shape,
        }
        observed_identity = {
            "capacity": int(state["capacity"]),
            "observation_shape": tuple(state["observation_shape"]),
            "action_shape": tuple(state["action_shape"]),
        }
        if observed_identity != expected_identity:
            raise ValueError(
                "imitation replay checkpoint identity does not match configured ring"
            )
        size = int(state["size"])
        cursor = int(state["cursor"])
        inserted_rows = int(state["inserted_rows"])
        if not 0 <= size <= self._capacity:
            raise ValueError("imitation replay checkpoint size is invalid")
        if not 0 <= cursor < self._capacity:
            raise ValueError("imitation replay checkpoint cursor is invalid")
        if inserted_rows < size:
            raise ValueError("imitation replay inserted-row counter is invalid")
        data = state.get("data")
        if not isinstance(data, dict) or set(data) != set(self.FIELDS):
            raise ValueError("imitation replay checkpoint data fields drifted")
        for name in self.FIELDS:
            target = getattr(self, f"_{name}")[:size]
            source = data[name].to(device=self._device, dtype=target.dtype)
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"imitation replay checkpoint field {name} has the wrong shape"
                )
            target.copy_(source)
        self._cursor = cursor
        self._size = size
        self._inserted_rows = inserted_rows
        self._generator.set_state(state["generator_state"])


__all__ = [
    "DeviceImitationBatch",
    "DeviceImitationReplayBuffer",
    "DeviceReplayBatch",
    "DeviceReplayBuffer",
    "DeviceReplayContractError",
]
