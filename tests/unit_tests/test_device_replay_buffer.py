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


def test_replay_hot_methods_forbid_host_materialization_and_cpu_sampling_rng() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    hot_nodes = [methods["add_batch"], methods["sample"]]
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


def test_tensor_offpolicy_rollout_is_two_phase_and_device_only() -> None:
    script = (
        Path(__file__).parents[2]
        / "examples"
        / "embodiment"
        / "train_dynamic_benchmark_tensor_offpolicy_smoke.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    rollout = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_rollout_cohort"
    )
    loop = next(
        node
        for node in ast.walk(rollout)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_step"
    )
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)]

    def named_calls(name: str) -> list[ast.Call]:
        return [
            call
            for call in calls
            if isinstance(call.func, ast.Attribute) and call.func.attr == name
        ]

    begin = named_calls("begin_step")
    environment_step = [
        call
        for call in named_calls("step")
        if isinstance(call.func.value, ast.Name) and call.func.value.id == "env"
    ]
    commit = named_calls("commit_step")
    assert len(begin) == len(environment_step) == len(commit) == 1
    assert begin[0].lineno < environment_step[0].lineno < commit[0].lineno
    forbidden = {"clone", "cpu", "numpy", "item", "tolist"}
    assert not [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr in forbidden
    ]


def test_tensor_offpolicy_checkpoint_covers_full_sac_rlpd_resume_state() -> None:
    script = (
        Path(__file__).parents[2]
        / "examples"
        / "embodiment"
        / "train_dynamic_benchmark_tensor_offpolicy_smoke.py"
    )
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    manifest_restore = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "load_manifest_state_dict"
    )
    initial_reset = min(
        call.lineno
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "reset"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "env"
    )
    assert manifest_restore.lineno < initial_reset
    required_checkpoint_fragments = (
        'parser.add_argument("--resume-from", type=Path)',
        '"actor": actor.state_dict()',
        '"critic": critic.state_dict()',
        '"target_critic": target_critic.state_dict()',
        '"actor_optimizer": actor_optimizer.state_dict()',
        '"critic_optimizer": critic_optimizer.state_dict()',
        '"alpha_optimizer": alpha_optimizer.state_dict()',
        '"log_alpha": log_alpha.detach()',
        '"online_replay": online.state_dict()',
        '"demo_replay": demos.state_dict()',
        '"actor_bc_pretrain": actor_bc_pretrain',
        '"rng_state": _capture_rng_state()',
        '"manifest_cursor": dict(env.manifest_state_dict())',
        'checkpoint_payload["training_state_sha256"] = _checkpoint_state_sha256(',
        '_restore_rng_state(restored["rng_state"])',
    )
    assert all(fragment in source for fragment in required_checkpoint_fragments)
    assert '"zero_action": "zero_action_device_cohort_v1"' in source
    assert '"privileged_teacher": "current_gpu_state_privileged_teacher_v2"' in source
    assert 'CHECKPOINT_SCHEMA = "rlinf-gpuenv0-tensor-offpolicy-smoke-v0.5"' in source
    assert 'DEMO_QUALITY_SCHEMA = "rlinf-gpuenv0-demo-quality-v0.3"' in source
    assert '"demo_quality": demo_quality' in source
    assert '"rlpd_demo_quality_qualified": bool(' in source


def test_tensor_offpolicy_teacher_control_plane_is_separate_and_accounted() -> None:
    script = (
        Path(__file__).parents[2]
        / "examples"
        / "embodiment"
        / "train_dynamic_benchmark_tensor_offpolicy_smoke.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    methods = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    online = methods["_rollout_cohort"]
    teacher = methods["_rollout_privileged_teacher_cohort"]
    online_source = ast.unparse(online)
    teacher_source = ast.unparse(teacher)

    def host_transfers(method: ast.AST) -> list[ast.Call]:
        return [
            call
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "to"
            and any(
                keyword.arg == "device"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "cpu"
                for keyword in call.keywords
            )
        ]

    assert "materialize_teacher_observations" not in online_source
    assert host_transfers(online) == []
    assert "materialize_teacher_observations" in teacher_source
    assert len(host_transfers(teacher)) == 1
    assert "torch.as_tensor" in teacher_source
    assert "terminal_mask_host_materializations" in teacher_source


def test_success_only_demo_selection_stays_on_the_rollout_device() -> None:
    script = (
        Path(__file__).parents[2]
        / "examples"
        / "embodiment"
        / "train_dynamic_benchmark_tensor_offpolicy_smoke.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    methods = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    selector = methods["_successful_demo_lane_mask"]
    insertion = methods["_add_rollout_to_replay"]
    calls = [
        call
        for method in (selector, insertion)
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
    ]
    forbidden = {"cpu", "item", "numpy", "tolist"}
    assert not [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr in forbidden
    ]
    selector_source = ast.unparse(selector)
    insertion_source = ast.unparse(insertion)
    assert "rollout.success & rollout.done & rollout.valid" in selector_source
    assert "selector = rollout.valid & lane_mask.unsqueeze(0)" in insertion_source
    assert "rollout.observation[selector]" in insertion_source
    assert "replay.add_batch" in insertion_source
