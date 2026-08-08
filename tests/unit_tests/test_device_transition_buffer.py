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

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parents[2] / "rlinf" / "data" / "device_transition_buffer.py"
_SPEC = importlib.util.spec_from_file_location("_device_transition_buffer_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
DeviceFieldSpec = _MODULE.DeviceFieldSpec
DeviceTransitionBuffer = _MODULE.DeviceTransitionBuffer
DeviceTransitionContractError = _MODULE.DeviceTransitionContractError


class _Tensor:
    def __init__(self, values: Any, *, dtype: Any, device: str = "cuda:0") -> None:
        self.values = np.asarray(values)
        self.dtype = dtype
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def is_contiguous(self) -> bool:
        return True

    def copy_(self, other: Any) -> _Tensor:
        source = other.values if isinstance(other, _Tensor) else other
        self.values[...] = source
        return self

    def fill_(self, value: Any) -> _Tensor:
        self.values.fill(value)
        return self

    def logical_and_(self, other: _Tensor) -> _Tensor:
        self.values[...] = np.logical_and(self.values, other.values)
        return self

    def __getitem__(self, index: Any) -> _Tensor:
        return _Tensor(self.values[index], dtype=self.dtype, device=self.device)

    def __ne__(self, other: Any) -> _Tensor:
        return _Tensor(self.values != other, dtype="bool", device=self.device)

    def __or__(self, other: _Tensor) -> _Tensor:
        return _Tensor(self.values | other.values, dtype="bool", device=self.device)

    def __invert__(self) -> _Tensor:
        return _Tensor(~self.values, dtype="bool", device=self.device)


class _Torch:
    bool = "bool"
    float32 = "float32"

    @staticmethod
    def empty(shape: tuple[int, ...], *, dtype: Any, device: str) -> _Tensor:
        numpy_dtype = np.bool_ if dtype == "bool" else np.float32
        return _Tensor(np.empty(shape, dtype=numpy_dtype), dtype=dtype, device=device)

    @staticmethod
    def ones(shape: tuple[int, ...], *, dtype: Any, device: str) -> _Tensor:
        numpy_dtype = np.bool_ if dtype == "bool" else np.float32
        return _Tensor(np.ones(shape, dtype=numpy_dtype), dtype=dtype, device=device)


def _tensor(values: Any, *, dtype: Any = "float32") -> _Tensor:
    numpy_dtype = np.bool_ if dtype == "bool" else np.float32
    return _Tensor(np.asarray(values, dtype=numpy_dtype), dtype=dtype)


def _buffer() -> Any:
    return DeviceTransitionBuffer(
        capacity=3,
        num_envs=2,
        observation_shape=(2,),
        action_shape=(1,),
        device="cuda:0",
        observation_dtype="float32",
        action_dtype="float32",
        reward_dtype="float32",
        extra_fields={"value": DeviceFieldSpec(shape=(), dtype="float32")},
        torch_module=_Torch,
    )


def _append(buffer: Any, *, terminated: tuple[int, int]) -> None:
    buffer.append(
        observation=_tensor([[1, 2], [3, 4]]),
        action=_tensor([[0], [1]]),
        reward=_tensor([1, 2]),
        next_observation=_tensor([[2, 3], [4, 5]]),
        terminated=_tensor(terminated),
        truncated=_tensor([0, 0]),
        success=_tensor(terminated),
        extras={"value": _tensor([0.5, 0.6])},
    )


def test_post_terminal_rows_are_masked_until_full_cohort_reset() -> None:
    buffer = _buffer()
    _append(buffer, terminated=(1, 0))
    _append(buffer, terminated=(0, 0))
    view = buffer.view()

    np.testing.assert_array_equal(view.valid.values, [[True, True], [False, True]])
    np.testing.assert_array_equal(view.done.values, [[True, False], [False, False]])
    assert view.horizon == 2
    assert view.num_envs == 2

    buffer.reset_cohort()
    _append(buffer, terminated=(0, 0))
    np.testing.assert_array_equal(buffer.view().valid.values, [[True, True]])


def test_buffer_rejects_cross_device_and_schema_drift() -> None:
    buffer = _buffer()
    with pytest.raises(DeviceTransitionContractError, match="extra field set"):
        buffer.append(
            observation=_tensor([[1, 2], [3, 4]]),
            action=_tensor([[0], [1]]),
            reward=_tensor([1, 2]),
            next_observation=_tensor([[2, 3], [4, 5]]),
            terminated=_tensor([0, 0]),
            truncated=_tensor([0, 0]),
            success=_tensor([0, 0]),
        )

    wrong_device = _tensor([[1, 2], [3, 4]])
    wrong_device.device = "cuda:1"
    with pytest.raises(DeviceTransitionContractError, match="observation is not"):
        buffer.append(
            observation=wrong_device,
            action=_tensor([[0], [1]]),
            reward=_tensor([1, 2]),
            next_observation=_tensor([[2, 3], [4, 5]]),
            terminated=_tensor([0, 0]),
            truncated=_tensor([0, 0]),
            success=_tensor([0, 0]),
            extras={"value": _tensor([0.5, 0.6])},
        )
