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

"""Preallocated device-only transition storage for fixed-reset RL cohorts.

The buffer is deliberately algorithm-neutral.  It stores the environment data
plane plus caller-defined policy fields, while a device-local ``alive`` mask
marks the first terminal transition valid and masks all post-terminal steps
until the entire cohort is reset.  No tensor is materialized on the host in
``append`` or ``view``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


class DeviceTransitionContractError(ValueError):
    """Raised when a tensor would violate the device-only storage contract."""


@dataclass(frozen=True)
class DeviceFieldSpec:
    """Shape and dtype for one algorithm-owned per-transition tensor."""

    shape: tuple[int, ...]
    dtype: Any

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in self.shape):
            raise ValueError("device field dimensions must be positive integers")
        if self.dtype is None:
            raise ValueError("device field dtype must not be None")


@dataclass(frozen=True)
class DeviceTransitionView:
    """A zero-copy leading slice of one device transition buffer."""

    observation: Any
    action: Any
    reward: Any
    next_observation: Any
    terminated: Any
    truncated: Any
    success: Any
    event_mask: Any
    terminal_reason: Any
    physics_step: Any
    valid: Any
    extras: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))

    @property
    def done(self) -> Any:
        """Return a device-local terminal mask."""

        return self.terminated | self.truncated

    @property
    def horizon(self) -> int:
        return int(self.observation.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.observation.shape[1])


class DeviceTransitionBuffer:
    """Own a fixed-horizon CUDA buffer for one full-reset environment cohort."""

    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: Any,
        observation_dtype: Any,
        action_dtype: Any,
        reward_dtype: Any,
        event_mask_dtype: Any,
        terminal_reason_dtype: Any,
        physics_step_dtype: Any,
        extra_fields: Mapping[str, DeviceFieldSpec] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        for name, value in (("capacity", capacity), ("num_envs", num_envs)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, shape in (
            ("observation_shape", observation_shape),
            ("action_shape", action_shape),
        ):
            if not isinstance(shape, tuple) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in shape
            ):
                raise ValueError(f"{name} must contain positive integer dimensions")
        torch = importlib.import_module("torch") if torch_module is None else torch_module
        if str(device).lower().split(":", 1)[0] != "cuda":
            raise DeviceTransitionContractError("device transition storage requires CUDA")
        specs = {} if extra_fields is None else dict(extra_fields)
        reserved = {
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
            "success",
            "event_mask",
            "terminal_reason",
            "physics_step",
            "valid",
        }
        overlap = sorted(reserved & set(specs))
        if overlap:
            raise ValueError(f"extra field names are reserved: {overlap}")
        if any(not isinstance(name, str) or not name for name in specs):
            raise ValueError("extra field names must be non-empty strings")
        if any(not isinstance(spec, DeviceFieldSpec) for spec in specs.values()):
            raise TypeError("extra_fields values must be DeviceFieldSpec")

        prefix = (capacity, num_envs)
        self._torch = torch
        self._capacity = capacity
        self._num_envs = num_envs
        self._device = device
        self._observation_shape = observation_shape
        self._action_shape = action_shape
        self._observation_dtype = observation_dtype
        self._action_dtype = action_dtype
        self._reward_dtype = reward_dtype
        self._event_mask_dtype = event_mask_dtype
        self._terminal_reason_dtype = terminal_reason_dtype
        self._physics_step_dtype = physics_step_dtype
        self._specs = MappingProxyType(specs)
        self._observation = torch.empty(
            (*prefix, *observation_shape), dtype=observation_dtype, device=device
        )
        self._action = torch.empty((*prefix, *action_shape), dtype=action_dtype, device=device)
        self._reward = torch.empty(prefix, dtype=reward_dtype, device=device)
        self._next_observation = torch.empty(
            (*prefix, *observation_shape), dtype=observation_dtype, device=device
        )
        self._terminated = torch.empty(prefix, dtype=torch.bool, device=device)
        self._truncated = torch.empty(prefix, dtype=torch.bool, device=device)
        self._success = torch.empty(prefix, dtype=torch.bool, device=device)
        self._event_mask = torch.empty(prefix, dtype=event_mask_dtype, device=device)
        self._terminal_reason = torch.empty(
            prefix,
            dtype=terminal_reason_dtype,
            device=device,
        )
        self._physics_step = torch.empty(prefix, dtype=physics_step_dtype, device=device)
        self._valid = torch.empty(prefix, dtype=torch.bool, device=device)
        self._extras = {
            name: torch.empty((*prefix, *spec.shape), dtype=spec.dtype, device=device)
            for name, spec in specs.items()
        }
        self._alive = torch.ones((num_envs,), dtype=torch.bool, device=device)
        self._cursor = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def full(self) -> bool:
        return self._cursor == self._capacity

    def _require_tensor(
        self,
        value: Any,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> None:
        if tuple(value.shape) != shape:
            raise DeviceTransitionContractError(f"{name} must have shape {shape}")
        if value.dtype != dtype:
            raise DeviceTransitionContractError(f"{name} has the wrong dtype")
        if value.device != self._device:
            raise DeviceTransitionContractError(f"{name} is not on {self._device}")
        if not value.is_contiguous():
            raise DeviceTransitionContractError(f"{name} must be contiguous")

    def reset_cohort(self) -> None:
        """Start a new cohort without reallocating storage or reading the GPU."""

        self._cursor = 0
        self._alive.fill_(True)

    def append(
        self,
        *,
        observation: Any,
        action: Any,
        reward: Any,
        next_observation: Any,
        terminated: Any,
        truncated: Any,
        success: Any,
        event_mask: Any,
        terminal_reason: Any,
        physics_step: Any,
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        """Copy one vector transition into preallocated storage on the same GPU."""

        if self.full:
            raise DeviceTransitionContractError("device transition buffer is full")
        vector_observation_shape = (self._num_envs, *self._observation_shape)
        vector_action_shape = (self._num_envs, *self._action_shape)
        vector_shape = (self._num_envs,)
        self._require_tensor(
            observation,
            name="observation",
            shape=vector_observation_shape,
            dtype=self._observation_dtype,
        )
        self._require_tensor(
            action,
            name="action",
            shape=vector_action_shape,
            dtype=self._action_dtype,
        )
        self._require_tensor(
            reward,
            name="reward",
            shape=vector_shape,
            dtype=self._reward_dtype,
        )
        self._require_tensor(
            next_observation,
            name="next_observation",
            shape=vector_observation_shape,
            dtype=self._observation_dtype,
        )
        for name, value in (
            ("terminated", terminated),
            ("truncated", truncated),
            ("success", success),
        ):
            if tuple(value.shape) != vector_shape or value.device != self._device:
                raise DeviceTransitionContractError(
                    f"{name} must have shape {vector_shape} on {self._device}"
                )
            if not value.is_contiguous():
                raise DeviceTransitionContractError(f"{name} must be contiguous")
        for name, value, dtype in (
            ("event_mask", event_mask, self._event_mask_dtype),
            ("terminal_reason", terminal_reason, self._terminal_reason_dtype),
            ("physics_step", physics_step, self._physics_step_dtype),
        ):
            self._require_tensor(
                value,
                name=name,
                shape=vector_shape,
                dtype=dtype,
            )
        supplied_extras = {} if extras is None else dict(extras)
        if set(supplied_extras) != set(self._specs):
            raise DeviceTransitionContractError("extra field set does not match buffer schema")
        for name, spec in self._specs.items():
            self._require_tensor(
                supplied_extras[name],
                name=name,
                shape=(self._num_envs, *spec.shape),
                dtype=spec.dtype,
            )

        row = self._cursor
        self._observation[row].copy_(observation)
        self._action[row].copy_(action)
        self._reward[row].copy_(reward)
        self._next_observation[row].copy_(next_observation)
        self._valid[row].copy_(self._alive)
        self._terminated[row].copy_(terminated != 0)
        self._truncated[row].copy_(truncated != 0)
        self._success[row].copy_(success != 0)
        self._event_mask[row].copy_(event_mask)
        self._terminal_reason[row].copy_(terminal_reason)
        self._physics_step[row].copy_(physics_step)
        for name, value in supplied_extras.items():
            self._extras[name][row].copy_(value)
        self._alive.logical_and_(~(self._terminated[row] | self._truncated[row]))
        self._cursor += 1

    def view(self) -> DeviceTransitionView:
        """Return a zero-copy view of the populated prefix."""

        stop = self._cursor
        return DeviceTransitionView(
            observation=self._observation[:stop],
            action=self._action[:stop],
            reward=self._reward[:stop],
            next_observation=self._next_observation[:stop],
            terminated=self._terminated[:stop],
            truncated=self._truncated[:stop],
            success=self._success[:stop],
            event_mask=self._event_mask[:stop],
            terminal_reason=self._terminal_reason[:stop],
            physics_step=self._physics_step[:stop],
            valid=self._valid[:stop],
            extras={name: value[:stop] for name, value in self._extras.items()},
        )


__all__ = [
    "DeviceFieldSpec",
    "DeviceTransitionBuffer",
    "DeviceTransitionContractError",
    "DeviceTransitionView",
]
