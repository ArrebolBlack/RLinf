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

"""Causal CPU-Planner control plane for registered GPU-native task backends.

The backend owns MuJoCo/Warp physics, terminal state, and device tensors.  This
module only coordinates the explicitly permitted control-plane seam:

``current GPU observation audit -> CPU Planner -> fresh E7 CUDA action -> GPU step``

The Planner is called once for every active lane at every control boundary.  It
never receives the reset tensor or a previous action.  A small host copy of the
device ``done`` mask is used only to decide which lanes still need a Planner;
the terminal ledger remains device-authoritative and is materialized exactly
once per terminal lane.

This is an execution primitive, not a result runner.  It does not choose
manifests, alter task quality, or write experiment artifacts.  The returned
control tape is an in-memory audit of the causal E7 commands and is not a
substitute for the full per-step tape/media bundle required by GPUPLAN0.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Planner(Protocol):
    """Minimal causal Planner interface used by the GPU control plane."""

    def act(self, observation: Any) -> Any:
        """Return one fresh E7 command for the supplied current observation."""


@dataclass(frozen=True)
class PlannerControlRecord:
    """One causal Planner decision bound to a GPU observation clock."""

    lane: int
    episode_id: str
    physics_step: int
    control_step: int
    policy_step: int
    action: tuple[float, ...]


@dataclass(frozen=True)
class GpuPlannerRollout:
    """In-memory result of one full-reset GPU Planner cohort."""

    reset: Any
    control_tape: tuple[PlannerControlRecord, ...]
    terminal_rows: tuple[Any, ...]
    control_steps: int
    device_identity: Any


ActionTensorFactory = Callable[[np.ndarray, Any], Any]
DoneReader = Callable[[Any], np.ndarray]


def _default_action_tensor_factory(values: np.ndarray, device: Any) -> Any:
    """Create a contiguous float32 CUDA tensor from newly issued E7 values."""

    torch = importlib.import_module("torch")
    action = torch.as_tensor(values, dtype=torch.float32, device=device)
    if not action.is_contiguous():
        action = action.contiguous()
    return action


def _default_done_reader(done: Any) -> np.ndarray:
    """Read only the small device done mask into the host control plane."""

    value = done.detach() if hasattr(done, "detach") else done
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    values = np.asarray(value)
    if values.ndim != 1:
        raise RuntimeError(f"GPU done mask must be one-dimensional, got {values.shape}")
    if values.dtype != np.bool_:
        if not np.all(np.isin(values, (0, 1))):
            raise RuntimeError("GPU done mask must contain only boolean values")
        values = values != 0
    return np.ascontiguousarray(values, dtype=np.bool_)


def _validate_planners(
    planners: Sequence[Planner], num_envs: int
) -> tuple[Planner, ...]:
    """Validate one stateful Planner instance per GPU lane."""

    planners = tuple(planners)
    if len(planners) != num_envs:
        raise ValueError(
            f"GPU Planner requires one Planner per lane ({num_envs}), got {len(planners)}"
        )
    if any(not callable(getattr(planner, "act", None)) for planner in planners):
        raise TypeError(
            "every GPU Planner must expose a callable act(observation) method"
        )
    return planners


def _validate_gpu_backend(backend: Any) -> None:
    """Reject CPU environments and non-MjWarp backends before reset."""

    provenance = getattr(backend, "provenance", None)
    if getattr(provenance, "backend_id", None) != "mjwarp_gpu_v1":
        raise RuntimeError("GPU Planner requires backend=mjwarp_gpu_v1")
    device = getattr(backend, "device", None)
    if getattr(device, "type", None) != "cuda" and not str(device).startswith("cuda:"):
        raise RuntimeError("GPU Planner requires a CUDA device-resident backend")
    task_id = getattr(backend, "task_id", None)
    if task_id not in {"p0_grasp", "t4_sphere"}:
        raise RuntimeError(
            "GPU Planner control plane requires a registered task, got "
            f"{task_id!r}"
        )


def _validate_command(command: Any, observation: Any) -> np.ndarray:
    """Require the exact SE3-WAM ActionCommand E7 contract at each decision."""

    from se3_wam.benchmark.api import ActionCommand
    from se3_wam.benchmark.contracts import ActionMode

    if type(command) is not ActionCommand:
        raise TypeError("GPU Planner must return the exact se3_wam ActionCommand type")
    if command.mode is not ActionMode.E7:
        raise ValueError("GPU Planner control plane accepts E7 commands only")
    if command.policy_step != observation.policy_step:
        raise ValueError(
            "GPU Planner command policy_step differs from the current observation"
        )
    values = np.asarray(command.values, dtype=np.float32)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("GPU Planner must return a finite normalized E7 7-vector")
    return values


def _validate_observation(
    observation: Any, episode_id: str, task_id: str, step: int
) -> None:
    """Bind a materialized observation to the active lane and control boundary."""

    if observation.episode_id != episode_id or observation.task_id != task_id:
        raise RuntimeError("GPU Planner observation changed episode or task identity")
    if observation.control_step != step or observation.policy_step != step:
        raise RuntimeError(
            "GPU Planner observation is not from the current control boundary"
        )


def run_gpu_planner_cohort(
    backend: Any,
    planners: Sequence[Planner],
    *,
    action_tensor_factory: ActionTensorFactory = _default_action_tensor_factory,
    done_reader: DoneReader = _default_done_reader,
    max_steps: int | None = None,
) -> GpuPlannerRollout:
    """Run one full-reset cohort with a causal CPU Planner on GPU physics.

    Args:
        backend: A ``GpuNativeTensorBackendEnv``-compatible GPU backend.
        planners: One stateful ``act(observation)`` Planner per lane.
        action_tensor_factory: Converts the freshly issued host E7 matrix to a
            contiguous float32 tensor on ``backend.device``.  The default uses
            PyTorch and performs no action replay.
        done_reader: Reads the device done mask for host-side lane bookkeeping.
        max_steps: Optional positive upper bound, no greater than the fixed GPU
            cohort horizon.  Leaving it unset uses that horizon.

    Returns:
        An in-memory causal control tape, typed terminal rows, and final device
        identity attestation.  The caller owns backend closure and artifact
        serialization.

    Raises:
        RuntimeError: If the backend exposes a stale/misaligned observation or
            leaves a lane active at the requested bound.
        ValueError: If a Planner emits a non-E7 command or the requested bound
            is outside the fixed cohort horizon.
    """

    _validate_gpu_backend(backend)
    num_envs = backend.num_envs
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("GPU Planner backend must expose a positive integer num_envs")
    planners = _validate_planners(planners, num_envs)
    horizon = backend.cohort_horizon_steps
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("GPU Planner backend must expose a positive cohort horizon")
    if max_steps is None:
        max_steps = horizon
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if max_steps > horizon:
        raise ValueError("max_steps cannot exceed the fixed GPU cohort horizon")

    reset = backend.reset()
    episode_ids = tuple(reset.episode_ids)
    if len(episode_ids) != num_envs:
        raise RuntimeError("GPU reset returned the wrong number of episode identities")

    active = tuple(range(num_envs))
    terminal_rows_by_lane: dict[int, Any] = {}
    control_tape: list[PlannerControlRecord] = []
    neutral_action = np.zeros((7,), dtype=np.float32)

    for control_step in range(max_steps):
        observations = backend.materialize_teacher_observations(active)
        if len(observations) != len(active):
            raise RuntimeError("GPU current-observation audit lost active lanes")

        action_values = np.zeros((num_envs, 7), dtype=np.float32)
        for lane, observation in zip(active, observations, strict=True):
            _validate_observation(
                observation,
                episode_ids[lane],
                backend.task_id,
                control_step,
            )
            command = planners[lane].act(observation)
            values = _validate_command(command, observation)
            action_values[lane] = values
            control_tape.append(
                PlannerControlRecord(
                    lane=lane,
                    episode_id=episode_ids[lane],
                    physics_step=observation.physics_step,
                    control_step=observation.control_step,
                    policy_step=observation.policy_step,
                    action=tuple(float(value) for value in values),
                )
            )

        for lane in set(range(num_envs)).difference(active):
            action_values[lane] = neutral_action

        action = action_tensor_factory(action_values, backend.device)
        step_result = backend.step_device(action)
        done = np.asarray(done_reader(step_result.done), dtype=np.bool_)
        if done.shape != (num_envs,):
            raise RuntimeError(
                f"GPU done reader returned {done.shape}, expected {(num_envs,)}"
            )
        newly_terminal = tuple(lane for lane in active if bool(done[lane]))
        if newly_terminal:
            rows = backend.materialize_terminal_ledger_once(
                newly_terminal,
                tuple(episode_ids[lane] for lane in newly_terminal),
            )
            if len(rows) != len(newly_terminal):
                raise RuntimeError("GPU terminal ledger lost newly terminal lanes")
            for lane, row in zip(newly_terminal, rows, strict=True):
                if row.lane != lane or row.episode_id != episode_ids[lane]:
                    raise RuntimeError(
                        "GPU terminal ledger changed lane or episode identity"
                    )
                if lane in terminal_rows_by_lane:
                    raise RuntimeError(
                        "GPU terminal ledger lane was materialized twice"
                    )
                terminal_rows_by_lane[lane] = row
            active = tuple(lane for lane in active if lane not in newly_terminal)

        if not active:
            break

    if active:
        raise RuntimeError(
            "GPU Planner cohort remained active at max_steps; no partial result is returned"
        )
    device_identity = backend.attest_end()
    terminal_rows = tuple(terminal_rows_by_lane[lane] for lane in range(num_envs))
    return GpuPlannerRollout(
        reset=reset,
        control_tape=tuple(control_tape),
        terminal_rows=terminal_rows,
        control_steps=max(record.control_step for record in control_tape) + 1
        if control_tape
        else 0,
        device_identity=device_identity,
    )


__all__ = [
    "GpuPlannerRollout",
    "Planner",
    "PlannerControlRecord",
    "run_gpu_planner_cohort",
]
