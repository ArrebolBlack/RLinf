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
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "embodiment"
    / "train_dynamic_benchmark_tensor_offpolicy_smoke.py"
)


def _load_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    distributions = types.ModuleType("torch.distributions")

    class Module:
        pass

    class Normal:
        pass

    nn.Module = Module
    torch.nn = nn
    distributions.Normal = Normal
    for name, module in (
        ("torch", torch),
        ("torch.nn", nn),
        ("torch.nn.functional", functional),
        ("torch.distributions", distributions),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    stubs = {
        "rlinf": types.ModuleType("rlinf"),
        "rlinf.data": types.ModuleType("rlinf.data"),
        "rlinf.data.device_replay_buffer": types.ModuleType(
            "rlinf.data.device_replay_buffer"
        ),
        "rlinf.data.device_transition_buffer": types.ModuleType(
            "rlinf.data.device_transition_buffer"
        ),
        "rlinf.envs": types.ModuleType("rlinf.envs"),
        "rlinf.envs.dynamic_benchmark": types.ModuleType(
            "rlinf.envs.dynamic_benchmark"
        ),
        "rlinf.envs.dynamic_benchmark.gpu_tensor_backend": types.ModuleType(
            "rlinf.envs.dynamic_benchmark.gpu_tensor_backend"
        ),
    }
    replay = stubs["rlinf.data.device_replay_buffer"]
    replay.DeviceReplayBatch = type("DeviceReplayBatch", (), {})
    replay.DeviceReplayBuffer = type("DeviceReplayBuffer", (), {})
    stubs["rlinf.data.device_transition_buffer"].DeviceTransitionBuffer = type(
        "DeviceTransitionBuffer", (), {}
    )
    stubs[
        "rlinf.envs.dynamic_benchmark.gpu_tensor_backend"
    ].GpuNativeTensorBackendEnv = type("GpuNativeTensorBackendEnv", (), {})
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "_tensor_offpolicy_smoke_under_test", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _arguments(module: Any, *extra: str) -> Any:
    return module._parser().parse_args(
        [
            "--algorithm",
            "rlpd",
            "--export-dir",
            "export",
            "--expected-gpu-uuid",
            "GPU-00000000-0000-0000-0000-000000000000",
            "--output",
            "output",
            "--se3-source",
            "se3",
            "--se3-commit",
            "a" * 40,
            "--se3-tree",
            "b" * 40,
            "--rlinf-source",
            "rlinf",
            "--rlinf-commit",
            "c" * 40,
            "--rlinf-tree",
            "d" * 40,
            "--runtime-manifest",
            "runtime.json",
            "--runtime-manifest-sha256",
            "e" * 64,
            "--expected-cpuset",
            "0-3",
            *extra,
        ]
    )


def test_privileged_teacher_config_requires_positive_predeclared_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    config = module._config(
        _arguments(
            module,
            "--demo-policy",
            "privileged_teacher",
            "--demo-cohorts",
            "4",
            "--minimum-demo-success-rate",
            "0.75",
        )
    )
    assert config.demo_policy == "privileged_teacher"
    assert config.demo_cohorts == 4
    assert config.minimum_demo_success_rate == pytest.approx(0.75)

    with pytest.raises(ValueError, match="positive quality gate"):
        module._config(
            _arguments(
                module,
                "--demo-policy",
                "privileged_teacher",
                "--minimum-demo-success-rate",
                "0",
            )
        )


def test_zero_demo_cannot_be_mislabeled_as_quality_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    with pytest.raises(ValueError, match="cannot carry a quality gate"):
        module._config(_arguments(module, "--minimum-demo-success-rate", "0.5"))


def test_teacher_action_validation_binds_mode_clock_shape_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    mode = object()
    observation = SimpleNamespace(policy_step=7)
    command = SimpleNamespace(mode=mode, policy_step=7, values=np.zeros(7))
    action = module._validated_teacher_action(
        command,
        observation,
        action_mode=mode,
    )
    assert action.dtype == np.float32
    assert action.shape == (7,)

    with pytest.raises(RuntimeError, match="clock"):
        module._validated_teacher_action(
            SimpleNamespace(mode=mode, policy_step=8, values=np.zeros(7)),
            observation,
            action_mode=mode,
        )
    with pytest.raises(RuntimeError, match="invalid normalized"):
        module._validated_teacher_action(
            SimpleNamespace(mode=mode, policy_step=7, values=np.full(7, 1.01)),
            observation,
            action_mode=mode,
        )


def test_privileged_teacher_rollout_keeps_device_observations_in_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)

    class FakeTensor:
        def __init__(self, values: Any) -> None:
            self.values = np.asarray(values)

        def contiguous(self) -> FakeTensor:
            return self

        def detach(self) -> FakeTensor:
            return self

        def to(self, *, device: str) -> FakeTensor:
            assert device == "cpu"
            return self

        def numpy(self) -> np.ndarray:
            return self.values

    module.torch.float32 = np.float32
    module.torch.zeros = lambda shape, **_kwargs: FakeTensor(np.zeros(shape))
    module.torch.as_tensor = lambda values, **_kwargs: FakeTensor(values)
    module.torch.inference_mode = nullcontext
    module.torch.cuda = SimpleNamespace(synchronize=lambda _device: None)

    action_mode = SimpleNamespace(E7=object())

    class Teacher:
        def reset(self) -> None:
            pass

        def act(self, observation: Any) -> Any:
            return SimpleNamespace(
                mode=action_mode.E7,
                policy_step=observation.policy_step,
                values=np.zeros(7),
            )

    contracts = types.ModuleType("se3_wam.benchmark.contracts")
    contracts.ActionMode = action_mode
    teacher_factory = types.ModuleType("se3_wam.benchmark.teacher_factory")
    teacher_factory.make_privileged_teacher = lambda _task: (
        Teacher(),
        {"teacher_type": "fixture"},
    )
    for name, stub in (
        ("se3_wam", types.ModuleType("se3_wam")),
        ("se3_wam.benchmark", types.ModuleType("se3_wam.benchmark")),
        ("se3_wam.benchmark.contracts", contracts),
        ("se3_wam.benchmark.teacher_factory", teacher_factory),
    ):
        monkeypatch.setitem(sys.modules, name, stub)

    device_observations = (object(), object(), object())

    class Env:
        num_envs = 2
        cohort_horizon_steps = 2
        task_id = "p0_grasp"
        device = "cuda:0"
        teacher_audit_materializations = 0

        def __init__(self) -> None:
            self.step_index = 0
            self.audit_lanes: list[tuple[int, ...]] = []

        def materialize_teacher_observations(
            self, lanes: tuple[int, ...]
        ) -> tuple[Any, ...]:
            self.teacher_audit_materializations += 1
            self.audit_lanes.append(lanes)
            return tuple(
                SimpleNamespace(
                    episode_id=f"episode-{lane}",
                    task_id=self.task_id,
                    control_step=self.step_index,
                    policy_step=self.step_index,
                )
                for lane in lanes
            )

        def step(self, _action: FakeTensor) -> Any:
            self.step_index += 1
            done = [True, False] if self.step_index == 1 else [True, True]
            return SimpleNamespace(
                observation=device_observations[self.step_index],
                reward=object(),
                terminated=object(),
                truncated=object(),
                success=object(),
                event_mask=object(),
                terminal_reason=object(),
                physics_step=object(),
                done=FakeTensor(done),
            )

    class Buffer:
        pending = False

        def __init__(self) -> None:
            self.observations: list[Any] = []

        def reset_cohort(self) -> None:
            self.observations.clear()

        def begin_step(self, *, observation: Any, action: FakeTensor) -> None:
            assert isinstance(action, FakeTensor)
            self.observations.append(observation)
            self.pending = True

        def commit_step(self, **_kwargs: Any) -> None:
            self.pending = False

        def abort_step(self) -> None:
            self.pending = False

        def view(self) -> str:
            return "rollout"

    env = Env()
    buffer = Buffer()
    reset = SimpleNamespace(
        observation=device_observations[0],
        episode_ids=("episode-0", "episode-1"),
    )
    rollout, elapsed, evidence = module._rollout_privileged_teacher_cohort(
        env=env,
        buffer=buffer,
        reset=reset,
    )
    assert rollout == "rollout"
    assert elapsed > 0.0
    assert buffer.observations == list(device_observations[:2])
    assert env.audit_lanes == [(0, 1), (1,)]
    assert evidence["observation_audit_calls"] == 2
    assert evidence["observation_audit_lanes"] == 3
    assert evidence["terminal_mask_host_materializations"] == 2
    assert evidence["host_to_device_action_transfers"] == 2
