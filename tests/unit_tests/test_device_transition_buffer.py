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

_MODULE_PATH = (
    Path(__file__).parents[2] / "rlinf" / "data" / "device_transition_buffer.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_device_transition_buffer_under_test", _MODULE_PATH
)
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
    int32 = "int32"
    int64 = "int64"

    @staticmethod
    def _numpy_dtype(dtype: Any) -> Any:
        return {
            "bool": np.bool_,
            "float32": np.float32,
            "int32": np.int32,
            "int64": np.int64,
        }[dtype]

    @staticmethod
    def empty(shape: tuple[int, ...], *, dtype: Any, device: str) -> _Tensor:
        return _Tensor(
            np.empty(shape, dtype=_Torch._numpy_dtype(dtype)),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def ones(shape: tuple[int, ...], *, dtype: Any, device: str) -> _Tensor:
        return _Tensor(
            np.ones(shape, dtype=_Torch._numpy_dtype(dtype)), dtype=dtype, device=device
        )


def _tensor(values: Any, *, dtype: Any = "float32") -> _Tensor:
    return _Tensor(np.asarray(values, dtype=_Torch._numpy_dtype(dtype)), dtype=dtype)


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
        terminal_signal_dtype="int32",
        event_mask_dtype="int32",
        terminal_reason_dtype="int32",
        physics_step_dtype="int64",
        extra_fields={
            "raw_action": DeviceFieldSpec(shape=(1,), dtype="float32"),
            "log_prob": DeviceFieldSpec(shape=(), dtype="float32"),
            "value": DeviceFieldSpec(shape=(), dtype="float32"),
            "next_value": DeviceFieldSpec(
                shape=(),
                dtype="float32",
                phase="commit",
            ),
        },
        torch_module=_Torch,
    )


def _begin_extras() -> dict[str, _Tensor]:
    return {
        "raw_action": _tensor([[0.1], [0.2]]),
        "log_prob": _tensor([-0.1, -0.2]),
        "value": _tensor([0.5, 0.6]),
    }


def _transition(buffer: Any, *, terminated: tuple[int, int]) -> None:
    buffer.begin_step(
        observation=_tensor([[1, 2], [3, 4]]),
        action=_tensor([[0], [1]]),
        extras=_begin_extras(),
    )
    buffer.commit_step(
        reward=_tensor([1, 2]),
        next_observation=_tensor([[2, 3], [4, 5]]),
        terminated=_tensor(terminated, dtype="int32"),
        truncated=_tensor([0, 0], dtype="int32"),
        success=_tensor(terminated, dtype="int32"),
        event_mask=_tensor([1, 2], dtype="int32"),
        terminal_reason=_tensor(terminated, dtype="int32"),
        physics_step=_tensor([25, 25], dtype="int64"),
        extras={"next_value": _tensor([0.7, 0.8])},
    )


def test_post_terminal_rows_are_masked_until_full_cohort_reset() -> None:
    buffer = _buffer()
    _transition(buffer, terminated=(1, 0))
    _transition(buffer, terminated=(0, 0))
    view = buffer.view()

    np.testing.assert_array_equal(view.valid.values, [[True, True], [False, True]])
    np.testing.assert_array_equal(view.done.values, [[True, False], [False, False]])
    assert view.horizon == 2
    assert view.num_envs == 2

    buffer.reset_cohort()
    _transition(buffer, terminated=(0, 0))
    np.testing.assert_array_equal(buffer.view().valid.values, [[True, True]])


def test_buffer_rejects_cross_device_and_schema_drift() -> None:
    buffer = _buffer()
    with pytest.raises(DeviceTransitionContractError, match="begin extra field set"):
        buffer.begin_step(
            observation=_tensor([[1, 2], [3, 4]]),
            action=_tensor([[0], [1]]),
        )

    wrong_device = _tensor([[1, 2], [3, 4]])
    wrong_device.device = "cuda:1"
    with pytest.raises(DeviceTransitionContractError, match="observation is not"):
        buffer.begin_step(
            observation=wrong_device,
            action=_tensor([[0], [1]]),
            extras=_begin_extras(),
        )


def test_two_phase_write_preserves_pre_step_observation_under_aliasing() -> None:
    buffer = _buffer()
    engine_observation = _tensor([[1, 2], [3, 4]])
    policy_fields = _begin_extras()
    buffer.begin_step(
        observation=engine_observation,
        action=_tensor([[0], [1]]),
        extras=policy_fields,
    )

    # Model the engine reusing the exact same device storage for s[t + 1].
    engine_observation.values[...] = [[11, 12], [13, 14]]
    policy_fields["raw_action"].values[...] = [[9.1], [9.2]]
    policy_fields["log_prob"].values[...] = [-9.1, -9.2]
    policy_fields["value"].values[...] = [9.5, 9.6]
    buffer.commit_step(
        reward=_tensor([1, 2]),
        next_observation=engine_observation,
        terminated=_tensor([0, 0], dtype="int32"),
        truncated=_tensor([0, 0], dtype="int32"),
        success=_tensor([0, 0], dtype="int32"),
        event_mask=_tensor([1, 2], dtype="int32"),
        terminal_reason=_tensor([0, 0], dtype="int32"),
        physics_step=_tensor([25, 25], dtype="int64"),
        extras={"next_value": _tensor([0.7, 0.8])},
    )

    view = buffer.view()
    np.testing.assert_array_equal(view.observation.values[0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(
        view.next_observation.values[0],
        [[11, 12], [13, 14]],
    )
    np.testing.assert_allclose(view.extras["raw_action"].values[0], [[0.1], [0.2]])
    np.testing.assert_allclose(view.extras["log_prob"].values[0], [-0.1, -0.2])
    np.testing.assert_allclose(view.extras["value"].values[0], [0.5, 0.6])


def test_pending_transition_is_fail_closed_and_abort_is_reusable() -> None:
    buffer = _buffer()
    buffer.begin_step(
        observation=_tensor([[1, 2], [3, 4]]),
        action=_tensor([[0], [1]]),
        extras=_begin_extras(),
    )
    assert buffer.pending is True
    assert buffer.cursor == 0

    with pytest.raises(DeviceTransitionContractError, match="already pending"):
        buffer.begin_step(
            observation=_tensor([[1, 2], [3, 4]]),
            action=_tensor([[0], [1]]),
            extras=_begin_extras(),
        )
    with pytest.raises(DeviceTransitionContractError, match="cannot reset"):
        buffer.reset_cohort()
    with pytest.raises(DeviceTransitionContractError, match="cannot view"):
        buffer.view()

    buffer.abort_step()
    assert buffer.pending is False
    assert buffer.cursor == 0
    _transition(buffer, terminated=(0, 0))
    assert buffer.cursor == 1


def test_commit_requires_pending_row_and_validates_boolean_dtype() -> None:
    buffer = _buffer()
    commit = {
        "reward": _tensor([1, 2]),
        "next_observation": _tensor([[2, 3], [4, 5]]),
        "terminated": _tensor([0, 0], dtype="int32"),
        "truncated": _tensor([0, 0], dtype="int32"),
        "success": _tensor([0, 0], dtype="int32"),
        "event_mask": _tensor([1, 2], dtype="int32"),
        "terminal_reason": _tensor([0, 0], dtype="int32"),
        "physics_step": _tensor([25, 25], dtype="int64"),
        "extras": {"next_value": _tensor([0.7, 0.8])},
    }
    with pytest.raises(DeviceTransitionContractError, match="no pending"):
        buffer.commit_step(**commit)

    buffer.begin_step(
        observation=_tensor([[1, 2], [3, 4]]),
        action=_tensor([[0], [1]]),
        extras=_begin_extras(),
    )
    commit["terminated"] = _tensor([0, 0])
    with pytest.raises(DeviceTransitionContractError, match="wrong dtype"):
        buffer.commit_step(**commit)
    assert buffer.pending is True
    buffer.abort_step()


def test_field_phase_is_validated() -> None:
    with pytest.raises(ValueError, match="phase"):
        DeviceFieldSpec(shape=(), dtype="float32", phase="later")


def test_tensor_ppo_rollout_uses_two_phase_order_without_host_materialization() -> None:
    script = (
        Path(__file__).parents[2]
        / "examples"
        / "embodiment"
        / "train_dynamic_benchmark_tensor_ppo_smoke.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    rollout_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_step"
        and "cohort_horizon_steps" in ast.unparse(node.iter)
    )
    calls = [node for node in ast.walk(rollout_loop) if isinstance(node, ast.Call)]

    def method_calls(name: str) -> list[ast.Call]:
        return [
            call
            for call in calls
            if isinstance(call.func, ast.Attribute) and call.func.attr == name
        ]

    begin = method_calls("begin_step")
    environment_step = [
        call
        for call in method_calls("step")
        if isinstance(call.func.value, ast.Name) and call.func.value.id == "env"
    ]
    commit = method_calls("commit_step")
    assert len(begin) == len(environment_step) == len(commit) == 1
    assert begin[0].lineno < environment_step[0].lineno < commit[0].lineno

    forbidden = {"clone", "cpu", "numpy", "item", "tolist"}
    assert not [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr in forbidden
    ]


def test_tensor_ppo_checkpoint_captures_exact_cross_process_resume_state() -> None:
    script = (
        Path(__file__).parents[2]
        / "examples"
        / "embodiment"
        / "train_dynamic_benchmark_tensor_ppo_smoke.py"
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
    assert 'parser.add_argument("--resume-from", type=Path)' in source
    assert '"schema_version": "rlinf-gpuenv0-tensor-ppo-smoke-v0.2"' in source
    assert '"rng_state": _capture_rng_state()' in source
    assert '"manifest_cursor": dict(env.manifest_state_dict())' in source
    assert '"parameter_sha256": parameter_sha256_end' in source
    assert '_restore_rng_state(restored["rng_state"])' in source
    assert "_atomic_torch_save(checkpoint_path, checkpoint_payload)" in source
