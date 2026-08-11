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
import importlib.util
import json
import math
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
    replay.DeviceImitationReplayBuffer = type("DeviceImitationReplayBuffer", (), {})
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
    assert config.demo_success_only_replay is False
    assert config.minimum_qualified_demo_episodes == 0
    assert config.demo_teacher_overrides == {}
    assert config.demo_teacher_overrides_sha256 is None

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


def test_success_only_demo_bank_requires_24_successes_and_explicit_count_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    config = module._config(
        _arguments(
            module,
            "--demo-policy",
            "privileged_teacher",
            "--demo-success-only-replay",
            "--minimum-qualified-demo-episodes",
            "24",
        )
    )
    assert config.demo_success_only_replay is True
    assert config.minimum_qualified_demo_episodes == 24
    assert config.minimum_demo_success_rate == 0.0

    with pytest.raises(ValueError, match="at least 24"):
        module._config(
            _arguments(
                module,
                "--demo-policy",
                "privileged_teacher",
                "--demo-success-only-replay",
                "--minimum-qualified-demo-episodes",
                "23",
            )
        )
    with pytest.raises(ValueError, match="exceeds configured demo attempts"):
        module._config(
            _arguments(
                module,
                "--demo-policy",
                "privileged_teacher",
                "--demo-success-only-replay",
                "--minimum-qualified-demo-episodes",
                "65",
            )
        )
    with pytest.raises(ValueError, match="episode-count gate"):
        module._config(
            _arguments(
                module,
                "--demo-policy",
                "privileged_teacher",
                "--demo-success-only-replay",
                "--minimum-qualified-demo-episodes",
                "24",
                "--minimum-demo-success-rate",
                "0.5",
            )
        )
    with pytest.raises(ValueError, match="requires privileged_teacher"):
        module._config(
            _arguments(
                module,
                "--demo-success-only-replay",
                "--minimum-qualified-demo-episodes",
                "24",
            )
        )
    with pytest.raises(ValueError, match="requires success-only replay"):
        module._config(
            _arguments(
                module,
                "--demo-policy",
                "privileged_teacher",
                "--minimum-demo-success-rate",
                "0.75",
                "--minimum-qualified-demo-episodes",
                "24",
            )
        )
    sac_args = _arguments(
        module,
        "--demo-success-only-replay",
        "--minimum-qualified-demo-episodes",
        "24",
    )
    sac_args.algorithm = "sac"
    with pytest.raises(ValueError, match="require RLPD"):
        module._config(sac_args)


def test_actor_bc_controls_default_off_and_require_success_only_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    default = module._config(_arguments(module))
    assert default.actor_bc_pretrain_updates == 0
    assert default.actor_bc_weight == 0.0

    enabled = module._config(
        _arguments(
            module,
            "--demo-policy",
            "privileged_teacher",
            "--demo-success-only-replay",
            "--minimum-qualified-demo-episodes",
            "24",
            "--actor-bc-pretrain-updates",
            "4000",
            "--actor-bc-weight",
            "100",
        )
    )
    assert enabled.actor_bc_pretrain_updates == 4000
    assert enabled.actor_bc_weight == pytest.approx(100.0)

    with pytest.raises(ValueError, match="must be non-negative"):
        module._config(_arguments(module, "--actor-bc-pretrain-updates", "-1"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        module._config(_arguments(module, "--actor-bc-weight", "nan"))
    with pytest.raises(ValueError, match="success-only replay"):
        module._config(_arguments(module, "--actor-bc-weight", "1"))


def test_dagger_controls_default_off_and_require_separate_correction_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    default = module._config(_arguments(module))
    assert default.dagger_cohorts == 0
    assert default.dagger_bc_updates_per_cohort == 0
    assert default.dagger_correction_ratio == pytest.approx(0.5)
    assert default.actor_sac_weight == pytest.approx(1.0)

    enabled = module._config(
        _arguments(
            module,
            "--demo-policy",
            "privileged_teacher",
            "--demo-success-only-replay",
            "--minimum-qualified-demo-episodes",
            "24",
            "--actor-bc-pretrain-updates",
            "4000",
            "--actor-bc-weight",
            "100",
            "--actor-sac-weight",
            "0",
            "--dagger-cohorts",
            "4",
            "--dagger-bc-updates-per-cohort",
            "1000",
            "--dagger-correction-ratio",
            "0.75",
        )
    )
    assert enabled.dagger_cohorts == 4
    assert enabled.dagger_bc_updates_per_cohort == 1000
    assert enabled.dagger_correction_ratio == pytest.approx(0.75)
    assert enabled.actor_sac_weight == 0.0

    with pytest.raises(ValueError, match="enabled together"):
        module._config(_arguments(module, "--dagger-cohorts", "1"))
    with pytest.raises(ValueError, match="success-only replay and actor BC"):
        module._config(
            _arguments(
                module,
                "--dagger-cohorts",
                "1",
                "--dagger-bc-updates-per-cohort",
                "1",
            )
        )
    with pytest.raises(ValueError, match="strictly between zero and one"):
        module._config(_arguments(module, "--dagger-correction-ratio", "1"))


def test_actor_bc_pretrain_updates_actor_and_reports_boundary_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)

    class FakeLoss:
        def __init__(self, value: float) -> None:
            self.value = value
            self.backward_calls = 0

        def backward(self) -> None:
            self.backward_calls += 1

        def detach(self) -> FakeLoss:
            return self

        def clone(self) -> FakeLoss:
            return FakeLoss(self.value)

        def __add__(self, other: FakeLoss) -> FakeLoss:
            return FakeLoss(self.value + other.value)

        def __truediv__(self, divisor: int) -> FakeLoss:
            return FakeLoss(self.value / divisor)

        def __float__(self) -> float:
            return self.value

    class Actor:
        def __init__(self) -> None:
            self.version = 0
            self.training = False

        def state_dict(self) -> dict[str, int]:
            return {"version": self.version}

        def train(self) -> None:
            self.training = True

        def sample(
            self, _observation: object, *, stochastic: bool
        ) -> tuple[object, object]:
            assert stochastic is False
            return object(), object()

        def parameters(self) -> tuple[Any, ...]:
            return ()

    class Optimizer:
        def __init__(self, actor: Actor) -> None:
            self.actor = actor
            self.zero_grad_calls = 0
            self.step_calls = 0

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            self.zero_grad_calls += 1

        def step(self) -> None:
            self.actor.version += 1
            self.step_calls += 1

    class Replay:
        size = 9
        device = "cuda:0"

        def __init__(self) -> None:
            self.sample_calls = 0

        def sample(self, batch_size: int) -> SimpleNamespace:
            assert batch_size == 4
            self.sample_calls += 1
            return SimpleNamespace(observation=object(), action=object())

    losses = iter((FakeLoss(0.3), FakeLoss(0.2), FakeLoss(0.1)))
    monkeypatch.setattr(
        module.F,
        "mse_loss",
        lambda _prediction, _target: next(losses),
        raising=False,
    )
    module.torch.Tensor = type("Tensor", (), {})
    module.nn.utils = SimpleNamespace(clip_grad_norm_=lambda _parameters, _limit: None)
    module.torch.cuda = SimpleNamespace(synchronize=lambda device: device == "cuda:0")
    actor = Actor()
    optimizer = Optimizer(actor)
    replay = Replay()
    evidence = module._actor_bc_pretrain(
        config=SimpleNamespace(actor_bc_pretrain_updates=3, batch_size=4),
        actor=actor,
        actor_optimizer=optimizer,
        demos=replay,
    )

    assert actor.training is True
    assert replay.sample_calls == optimizer.zero_grad_calls == optimizer.step_calls == 3
    assert evidence["enabled"] is True
    assert evidence["completed_updates"] == 3
    assert evidence["first_loss"] == pytest.approx(0.3)
    assert evidence["last_loss"] == pytest.approx(0.1)
    assert evidence["mean_loss"] == pytest.approx(0.2)
    assert evidence["parameters_changed"] is True
    assert evidence["actor_sha256_before"] != evidence["actor_sha256_after"]


def test_success_only_replay_filters_complete_success_lanes_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    module.torch.bool = "bool"

    class Tensor:
        def __init__(
            self,
            values: Any,
            *,
            dtype: str = "float32",
            device: str = "cuda:0",
        ) -> None:
            self.values = np.asarray(values)
            self.dtype = dtype
            self.device = device

        @property
        def shape(self) -> tuple[int, ...]:
            return self.values.shape

        def __and__(self, other: Tensor) -> Tensor:
            return Tensor(
                self.values & other.values,
                dtype="bool",
                device=self.device,
            )

        def __getitem__(self, index: Any) -> Tensor:
            selected = index.values if isinstance(index, Tensor) else index
            return Tensor(self.values[selected], dtype=self.dtype, device=self.device)

        def any(self, *, dim: int) -> Tensor:
            return Tensor(
                self.values.any(axis=dim),
                dtype="bool",
                device=self.device,
            )

        def contiguous(self) -> Tensor:
            return self

        def reshape(self, *shape: int) -> Tensor:
            return Tensor(
                self.values.reshape(*shape),
                dtype=self.dtype,
                device=self.device,
            )

        def unsqueeze(self, dim: int) -> Tensor:
            return Tensor(
                np.expand_dims(self.values, axis=dim),
                dtype=self.dtype,
                device=self.device,
            )

    valid = Tensor(
        [[True, True, True], [True, True, False], [True, False, False]],
        dtype="bool",
    )
    done = Tensor(
        [[False, False, True], [False, True, False], [True, False, False]],
        dtype="bool",
    )
    success = Tensor(
        [[False, False, True], [False, False, False], [True, False, False]],
        dtype="bool",
    )
    values = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
    rollout = SimpleNamespace(
        observation=Tensor(values),
        action=Tensor(np.repeat(values, 7, axis=2)),
        reward=Tensor(values[..., 0]),
        next_observation=Tensor(values + 1),
        terminated=done,
        truncated=Tensor(np.zeros((3, 3), dtype=np.bool_), dtype="bool"),
        success=success,
        done=done,
        valid=valid,
    )

    class Replay:
        def __init__(self) -> None:
            self.batch: dict[str, Tensor] | None = None

        def add_batch(self, **batch: Tensor) -> None:
            self.batch = batch

    replay = Replay()
    lane_mask = module._successful_demo_lane_mask(rollout)
    inserted = module._add_rollout_to_replay(
        replay,
        rollout,
        lane_mask=lane_mask,
    )

    assert inserted == 4
    assert replay.batch is not None
    np.testing.assert_array_equal(
        replay.batch["observation"].values[:, 0],
        np.array([0.0, 2.0, 3.0, 6.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        replay.batch["valid"].values,
        np.ones(4, dtype=np.bool_),
    )
    assert 1.0 not in replay.batch["observation"].values
    assert 4.0 not in replay.batch["observation"].values

    legacy_replay = Replay()
    legacy_inserted = module._add_rollout_to_replay(legacy_replay, rollout)
    assert legacy_inserted == 9
    assert legacy_replay.batch is not None
    assert legacy_replay.batch["valid"].values.sum() == 6
    assert 1.0 in legacy_replay.batch["observation"].values


def test_teacher_overrides_are_strict_planner_only_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(monkeypatch)
    path = tmp_path / "teacher-overrides.json"
    payload = {
        "close_axis_alignment_tolerance_rad": 0.1,
        "close_retry_steps": 20,
        "lift_action_z_max": 0.4,
        "post_hold_settle_steps": 2,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = module._config(
        _arguments(
            module,
            "--demo-policy",
            "privileged_teacher",
            "--minimum-demo-success-rate",
            "0.75",
            "--demo-teacher-overrides",
            str(path),
        )
    )
    assert config.demo_teacher_overrides == payload
    assert (
        config.demo_teacher_overrides_sha256
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )

    unknown = tmp_path / "unknown.json"
    unknown.write_text('{"scientific_resolution": 1.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported demo teacher override"):
        module._config(
            _arguments(
                module,
                "--demo-policy",
                "privileged_teacher",
                "--minimum-demo-success-rate",
                "0.75",
                "--demo-teacher-overrides",
                str(unknown),
            )
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"lookahead_s": 0.3, "lookahead_s": 0.4}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate demo teacher override key"):
        module._config(
            _arguments(
                module,
                "--demo-policy",
                "privileged_teacher",
                "--minimum-demo-success-rate",
                "0.75",
                "--demo-teacher-overrides",
                str(duplicate),
            )
        )

    out_of_range = tmp_path / "out-of-range.json"
    out_of_range.write_text(
        json.dumps({"close_axis_alignment_tolerance_rad": math.pi}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="close_axis_alignment_tolerance_rad"):
        module._config(
            _arguments(
                module,
                "--demo-policy",
                "privileged_teacher",
                "--minimum-demo-success-rate",
                "0.75",
                "--demo-teacher-overrides",
                str(out_of_range),
            )
        )


def test_empty_teacher_override_file_still_requires_privileged_teacher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(monkeypatch)
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="require privileged_teacher"):
        module._config(_arguments(module, "--demo-teacher-overrides", str(path)))


def test_teacher_override_application_requires_existing_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    teacher = SimpleNamespace(lookahead_s=0.3)
    module._apply_demo_teacher_overrides(teacher, {"lookahead_s": 0.4})
    assert teacher.lookahead_s == pytest.approx(0.4)
    with pytest.raises(RuntimeError, match="does not expose"):
        module._apply_demo_teacher_overrides(teacher, {"lift_action_z_max": 0.4})


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


def test_dagger_rollout_executes_actor_but_records_teacher_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)

    class FakeTensor:
        def __init__(self, values: Any) -> None:
            self.values = np.asarray(values, dtype=np.float32)

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
                values=np.full(7, 0.75),
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

    observations = tuple(FakeTensor([[float(index)]]) for index in range(3))

    class Actor:
        def eval(self) -> None:
            pass

        def sample(
            self, _observation: FakeTensor, *, stochastic: bool
        ) -> tuple[FakeTensor, object]:
            assert stochastic is False
            return FakeTensor(np.full((1, 7), 0.25)), object()

    class Env:
        num_envs = 1
        cohort_horizon_steps = 2
        task_id = "p0_grasp"
        device = "cuda:0"
        teacher_audit_materializations = 0

        def __init__(self) -> None:
            self.step_index = 0
            self.executed_actions: list[np.ndarray] = []

        def materialize_teacher_observations(
            self, lanes: tuple[int, ...]
        ) -> tuple[Any, ...]:
            assert lanes == (0,)
            self.teacher_audit_materializations += 1
            return (
                SimpleNamespace(
                    episode_id="episode-0",
                    task_id=self.task_id,
                    control_step=self.step_index,
                    policy_step=self.step_index,
                ),
            )

        def step(self, action: FakeTensor) -> Any:
            self.executed_actions.append(action.values.copy())
            self.step_index += 1
            return SimpleNamespace(
                observation=observations[self.step_index],
                reward=object(),
                terminated=object(),
                truncated=object(),
                success=object(),
                event_mask=object(),
                terminal_reason=object(),
                physics_step=object(),
                done=FakeTensor([self.step_index == self.cohort_horizon_steps]),
            )

    class Buffer:
        pending = False

        def __init__(self, name: str) -> None:
            self.name = name
            self.actions: list[np.ndarray] = []

        def reset_cohort(self) -> None:
            self.actions.clear()

        def begin_step(self, *, observation: Any, action: FakeTensor) -> None:
            assert observation in observations
            self.actions.append(action.values.copy())
            self.pending = True

        def commit_step(self, **_kwargs: Any) -> None:
            self.pending = False

        def abort_step(self) -> None:
            self.pending = False

        def view(self) -> str:
            return self.name

    env = Env()
    online = Buffer("online")
    corrections = Buffer("corrections")
    reset = SimpleNamespace(
        observation=observations[0],
        episode_ids=("episode-0",),
    )
    online_view, correction_view, elapsed, evidence = (
        module._rollout_dagger_correction_cohort(
            env=env,
            actor=Actor(),
            online_buffer=online,
            correction_buffer=corrections,
            reset=reset,
        )
    )

    assert online_view == "online"
    assert correction_view == "corrections"
    assert elapsed > 0.0
    for executed, stored_online, stored_correction in zip(
        env.executed_actions, online.actions, corrections.actions, strict=True
    ):
        np.testing.assert_allclose(executed, 0.25)
        np.testing.assert_allclose(stored_online, 0.25)
        np.testing.assert_allclose(stored_correction, 0.75)
    assert evidence["executed_action_source"] == "deterministic_gpu_actor"
    assert evidence["labels_are_successful_demonstrations"] is False
    assert evidence["host_to_device_label_transfers"] == 2
