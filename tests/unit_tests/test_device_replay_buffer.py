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

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parents[2] / "rlinf" / "data" / "device_replay_buffer.py"
_SPEC = importlib.util.spec_from_file_location(
    "_device_replay_buffer_under_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
DeviceReplayBuffer = _MODULE.DeviceReplayBuffer
DeviceImitationReplayBuffer = _MODULE.DeviceImitationReplayBuffer
DeviceReplayContractError = _MODULE.DeviceReplayContractError


class _Generator:
    def __init__(self, *, device: str) -> None:
        self.device = device
        self._rng = np.random.default_rng()

    def manual_seed(self, seed: int) -> _Generator:
        self._rng = np.random.default_rng(seed)
        return self

    def get_state(self) -> _Tensor:
        return _Tensor(
            np.array([self._rng.bit_generator.state], dtype=object),
            dtype="object",
            device="cpu",
        )

    def set_state(self, state: _Tensor) -> None:
        self._rng.bit_generator.state = state.values[0]


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

    def to(self, *, device: str | None = None, dtype: Any | None = None) -> _Tensor:
        selected_dtype = self.dtype if dtype is None else dtype
        numpy_dtype = self.values.dtype if dtype is None else _Torch._numpy_dtype(dtype)
        return _Tensor(
            self.values.astype(numpy_dtype, copy=True),
            dtype=selected_dtype,
            device=self.device if device is None else device,
        )

    def index_select(self, dimension: int, indices: _Tensor) -> _Tensor:
        assert dimension == 0
        return _Tensor(
            self.values[indices.values],
            dtype=self.dtype,
            device=self.device,
        )

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return _Tensor(self.values.copy(), dtype=self.dtype, device="cpu")

    def clone(self) -> _Tensor:
        return _Tensor(self.values.copy(), dtype=self.dtype, device=self.device)

    def __getitem__(self, index: Any) -> _Tensor:
        return _Tensor(self.values[index], dtype=self.dtype, device=self.device)

    def __or__(self, other: _Tensor) -> _Tensor:
        return _Tensor(self.values | other.values, dtype="bool", device=self.device)


class _Torch:
    bool = "bool"
    float32 = "float32"
    int64 = "int64"
    Generator = _Generator

    @staticmethod
    def _numpy_dtype(dtype: Any) -> Any:
        return {
            "bool": np.bool_,
            "float32": np.float32,
            "int64": np.int64,
            "object": object,
        }[dtype]

    @staticmethod
    def empty(shape: tuple[int, ...], *, dtype: Any, device: str) -> _Tensor:
        return _Tensor(
            np.empty(shape, dtype=_Torch._numpy_dtype(dtype)),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def zeros(shape: tuple[int, ...], *, dtype: Any, device: str) -> _Tensor:
        return _Tensor(
            np.zeros(shape, dtype=_Torch._numpy_dtype(dtype)),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def multinomial(
        weights: _Tensor,
        rows: int,
        *,
        replacement: bool,
        generator: _Generator,
    ) -> _Tensor:
        probabilities = weights.values.astype(np.float64)
        if probabilities.sum() <= 0:
            raise RuntimeError("invalid multinomial distribution")
        probabilities /= probabilities.sum()
        indices = generator._rng.choice(
            len(probabilities), size=rows, replace=replacement, p=probabilities
        )
        return _Tensor(indices.astype(np.int64), dtype="int64", device=weights.device)

    @staticmethod
    def randint(
        high: int,
        shape: tuple[int, ...],
        *,
        device: str,
        generator: _Generator,
    ) -> _Tensor:
        indices = generator._rng.integers(0, high, size=shape, dtype=np.int64)
        return _Tensor(indices, dtype="int64", device=device)


def _tensor(values: Any, *, dtype: Any = "float32", device: str = "cuda:0") -> _Tensor:
    return _Tensor(
        np.asarray(values, dtype=_Torch._numpy_dtype(dtype)),
        dtype=dtype,
        device=device,
    )


def _buffer(*, seed: int = 17) -> Any:
    return DeviceReplayBuffer(
        capacity=5,
        observation_shape=(1,),
        action_shape=(1,),
        device="cuda:0",
        seed=seed,
        observation_dtype="float32",
        action_dtype="float32",
        reward_dtype="float32",
        torch_module=_Torch,
    )


def _add(buffer: Any, values: list[float], valid: list[bool]) -> None:
    rows = len(values)
    column = np.asarray(values, dtype=np.float32).reshape(rows, 1)
    buffer.add_batch(
        observation=_tensor(column),
        action=_tensor(column + 100.0),
        reward=_tensor(values),
        next_observation=_tensor(column + 1.0),
        terminated=_tensor([False] * rows, dtype="bool"),
        truncated=_tensor([False] * rows, dtype="bool"),
        valid=_tensor(valid, dtype="bool"),
    )


def _imitation_buffer(*, seed: int = 23) -> Any:
    return DeviceImitationReplayBuffer(
        capacity=5,
        observation_shape=(1,),
        action_shape=(1,),
        device="cuda:0",
        seed=seed,
        observation_dtype="float32",
        action_dtype="float32",
        torch_module=_Torch,
    )


def test_sampling_stays_on_device_and_excludes_padding() -> None:
    buffer = _buffer()
    _add(buffer, [1.0, 2.0, 3.0], [True, False, True])

    batch = buffer.sample(256)

    assert batch.rows == 256
    assert batch.observation.device == "cuda:0"
    assert set(batch.observation.values[:, 0]) <= {1.0, 3.0}
    np.testing.assert_array_equal(batch.done.values, np.zeros(256, dtype=np.bool_))


def test_wrapping_and_checkpoint_restore_preserve_ring_and_cuda_rng() -> None:
    buffer = _buffer()
    _add(buffer, [1.0, 2.0, 3.0], [True, False, True])
    _add(buffer, [10.0, 11.0, 12.0, 13.0], [True, False, True, True])
    assert buffer.size == 5
    assert buffer.cursor == 2
    assert buffer.inserted_rows == 7

    state = buffer.state_dict()
    restored = _buffer(seed=999)
    restored.load_state_dict(state)

    first = buffer.sample(64).observation.values
    second = restored.sample(64).observation.values
    np.testing.assert_array_equal(first, second)
    assert set(first[:, 0]) <= {3.0, 10.0, 12.0, 13.0}


def test_contract_rejects_cross_device_and_checkpoint_identity_drift() -> None:
    buffer = _buffer()
    with pytest.raises(DeviceReplayContractError, match="observation is not"):
        buffer.add_batch(
            observation=_tensor([[1.0]], device="cuda:1"),
            action=_tensor([[1.0]]),
            reward=_tensor([1.0]),
            next_observation=_tensor([[2.0]]),
            terminated=_tensor([False], dtype="bool"),
            truncated=_tensor([False], dtype="bool"),
            valid=_tensor([True], dtype="bool"),
        )
    _add(buffer, [1.0], [True])
    state = buffer.state_dict()
    state["capacity"] = 6
    with pytest.raises(ValueError, match="identity"):
        _buffer().load_state_dict(state)


def test_imitation_replay_has_only_labels_and_restores_cuda_rng() -> None:
    buffer = _imitation_buffer()
    buffer.add_batch(
        observation=_tensor([[1.0], [2.0], [3.0]]),
        action=_tensor([[101.0], [102.0], [103.0]]),
    )
    state = buffer.state_dict()
    assert set(state["data"]) == {"observation", "action"}
    assert "reward" not in state["data"]
    restored = _imitation_buffer(seed=999)
    restored.load_state_dict(state)

    first = buffer.sample(64)
    second = restored.sample(64)
    np.testing.assert_array_equal(first.observation.values, second.observation.values)
    np.testing.assert_array_equal(first.action.values, second.action.values)
    assert first.rows == 64
    assert set(first.observation.values[:, 0]) <= {1.0, 2.0, 3.0}

    with pytest.raises(DeviceReplayContractError, match="observation is not"):
        buffer.add_batch(
            observation=_tensor([[4.0]], device="cuda:1"),
            action=_tensor([[104.0]]),
        )


def test_replay_hot_methods_forbid_host_materialization_and_cpu_sampling_rng() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    hot_nodes = []
    for class_name in ("DeviceReplayBuffer", "DeviceImitationReplayBuffer"):
        methods = {
            node.name: node
            for node in classes[class_name].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        hot_nodes.extend((methods["add_batch"], methods["sample"]))
    calls = [
        call
        for method in hot_nodes
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
    ]
    forbidden = {"cpu", "item", "numpy", "tolist"}
    assert not [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr in forbidden
    ]
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "torch.Generator(device=device)" in source
    assert "self._torch.multinomial(" in source
    assert "self._torch.randint(" in source


def test_public_replay_entrypoint_does_not_initialize_storage_stack() -> None:
    code = (
        "import sys; import rlinf.data as data; "
        "assert data.DeviceReplayBuffer.__name__ == 'DeviceReplayBuffer'; "
        "assert 'rlinf.data.storage' not in sys.modules"
    )

    subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
