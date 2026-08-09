#!/usr/bin/env python3
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

"""Evaluate a frozen Dynamic Benchmark expert on deterministic test resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from examples.embodiment.dynamic_benchmark_checkpoint_admission import (
    SOURCE_SNAPSHOT_MANIFEST_FILENAME,
    SOURCE_SNAPSHOT_SCHEMA,
    validate_selected_learned_policy,
    validate_source_snapshot_manifest,
)
from examples.embodiment.dynamic_benchmark_evaluation_attempt import (
    materialize_evaluation_attempt,
    recursive_output_checksums,
    validate_formal_quality_v2_thresholds,
)
from examples.embodiment.train_dynamic_benchmark_expert import (
    RunningNormalizer,
    _compose_residual_actions,
    _planner_actions,
    _policy_action,
)

POLICY_SCHEMA = "rlinf-dynamic-benchmark-expert-policy-v0.1"
EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-expert-evaluation-v0.2"
FORMAL_EVALUATION_SCHEMA = "rlinf-dynamic-benchmark-expert-evaluation-v0.3"
SOURCE_SNAPSHOT_RECEIPT_SCHEMA = "rld2-qa-source-snapshot-receipt-v0.1"
TASK_QUALITY_BACKEND_ID = "mujoco311-rs140-v1-rld2-quality"
SUPPORTED_ALGORITHMS = frozenset({"bc", "sac", "rlpd", "residual_rlpd", "ppo"})
EVALUATOR_REPOSITORY_PATH = "examples/embodiment/evaluate_dynamic_benchmark_expert.py"
EVALUATOR_MODULE = "examples.embodiment.evaluate_dynamic_benchmark_expert"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--trainer-run-root", type=Path)
    parser.add_argument("--trainer-summary", type=Path)
    parser.add_argument("--checkpoint-selection", type=Path)
    parser.add_argument("--checkpoint-selection-outcome", type=Path)
    parser.add_argument("--expected-checkpoint-selection-outcome-sha256")
    parser.add_argument("--policy-rlinf-source-root", type=Path)
    parser.add_argument("--evaluator-rlinf-source-root", type=Path)
    parser.add_argument("--benchmark-source-root", type=Path)
    parser.add_argument("--source-snapshot-manifest", type=Path)
    parser.add_argument("--expected-source-snapshot-manifest-sha256")
    parser.add_argument("--base-image-id")
    parser.add_argument("--source-snapshot-image-id")
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--rlinf-commit", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--quality-v2-thresholds", type=Path)
    parser.add_argument("--expected-quality-v2-thresholds-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test_id", "test_ood"),
        required=True,
    )
    parser.add_argument("--manifest-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actual_evaluator_source_identity(
    evaluator_rlinf_source_root: Path,
) -> dict[str, str]:
    raw_actual = Path(__file__)
    if raw_actual.is_symlink() or not raw_actual.is_file():
        raise ValueError(
            "executing evaluator source must be a regular non-symlink file"
        )
    actual = raw_actual.resolve(strict=True)
    raw_root = Path(evaluator_rlinf_source_root)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("declared evaluator source root must be a regular directory")
    expected = raw_root.resolve(strict=True) / EVALUATOR_REPOSITORY_PATH
    if actual != expected:
        raise ValueError(
            "executing evaluator source path does not match the declared source root"
        )
    source_bytes = actual.read_bytes()
    return {
        "module": EVALUATOR_MODULE,
        "repository_path": EVALUATOR_REPOSITORY_PATH,
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    from se3_wam.benchmark.contracts import canonical_json

    return hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def _task_quality_identity(task_id: str) -> dict[str, Any]:
    """Return the locally canonical task-quality schema and backend identity."""

    from se3_wam.benchmark.task_quality import (
        task_quality_schema_manifest,
        validate_task_quality_schema_manifest,
    )

    schema = task_quality_schema_manifest(task_id)
    validate_task_quality_schema_manifest(schema)
    schema_sha256 = schema.get("schema_sha256")
    if not isinstance(schema_sha256, str):
        raise ValueError("task-quality schema manifest is missing its SHA-256")
    unsigned_schema = dict(schema)
    unsigned_schema.pop("schema_sha256", None)
    if _payload_sha256(unsigned_schema) != schema_sha256:
        raise ValueError("task-quality schema SHA-256 does not recompute")
    return {
        "evaluator_backend_id": TASK_QUALITY_BACKEND_ID,
        "task_quality_schema": dict(schema),
        "task_quality_schema_sha256": schema_sha256,
    }


def _task_quality_schema_from_identity(
    identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate an evaluator identity and return its canonical schema manifest."""

    from se3_wam.benchmark.task_quality import validate_task_quality_schema_manifest

    if set(identity) != {
        "evaluator_backend_id",
        "task_quality_schema",
        "task_quality_schema_sha256",
    }:
        raise ValueError("task-quality identity field inventory mismatch")
    schema = identity.get("task_quality_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("task-quality identity is missing its schema manifest")
    try:
        validate_task_quality_schema_manifest(schema)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("task-quality identity schema is not canonical") from error
    if identity.get("task_quality_schema_sha256") != schema.get("schema_sha256"):
        raise ValueError("task-quality identity/schema SHA-256 mismatch")
    backend_id = identity.get("evaluator_backend_id")
    if (
        not isinstance(backend_id, str)
        or not backend_id
        or backend_id.strip() != backend_id
    ):
        raise ValueError("task-quality evaluator backend identity is missing")
    return schema


def _task_quality_env_config(identity: Mapping[str, Any]) -> dict[str, str]:
    """Build the DynamicBenchmarkEnv opt-in fields from a canonical identity."""

    schema = _task_quality_schema_from_identity(identity)
    schema_version = schema.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("task-quality schema version is missing")
    return {
        "task_quality_schema_version": schema_version,
        "task_quality_evaluator_backend_id": str(identity["evaluator_backend_id"]),
    }


def _validate_task_quality_summary(
    raw: Any,
    *,
    identity: Mapping[str, Any],
    task_id: str,
    episode_id: str,
) -> dict[str, Any]:
    """Validate one canonical terminal task-quality summary against evaluation identity."""

    from se3_wam.benchmark.task_quality import EpisodeQualitySummary

    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("terminal task-quality summary must be a non-empty mapping")
    try:
        canonical = json.loads(json.dumps(dict(raw), allow_nan=False))
        summary = EpisodeQualitySummary.from_dict(canonical)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("terminal task-quality summary is not canonical") from error

    schema = _task_quality_schema_from_identity(identity)
    expected = {
        "task_id": task_id,
        "episode_id": episode_id,
        "evaluator_backend_id": identity.get("evaluator_backend_id"),
        "schema_version": schema.get("schema_version"),
        "schema_sha256": identity.get("task_quality_schema_sha256"),
    }
    actual = {
        "task_id": summary.task_id,
        "episode_id": summary.episode_id,
        "evaluator_backend_id": summary.evaluator_backend_id,
        "schema_version": summary.schema_version,
        "schema_sha256": summary.schema_sha256,
    }
    if actual != expected:
        raise ValueError("terminal task-quality summary identity mismatch")
    if not summary.terminal:
        raise ValueError("task-quality summary must be terminal")
    normalized = summary.to_dict()
    if canonical != normalized:
        raise ValueError("terminal task-quality summary is not in canonical form")
    return normalized


def _task_quality_from_terminal_infos(
    infos: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    task_id: str,
    episode_id: str,
) -> dict[str, Any]:
    """Read and validate ``infos['task_quality'][0]`` at episode termination."""

    if "task_quality" not in infos:
        raise ValueError("terminal infos are missing task_quality")
    rows = infos["task_quality"]
    if not isinstance(rows, (list, tuple)) or len(rows) != 1:
        raise ValueError("terminal task-quality vector must contain exactly one row")
    if rows[0] is None:
        raise ValueError("terminal infos contain no task-quality summary")
    return _validate_task_quality_summary(
        rows[0],
        identity=identity,
        task_id=task_id,
        episode_id=episode_id,
    )


def _aggregate_task_quality_values(values: list[float]) -> dict[str, Any]:
    """Return finite scalar aggregates, using nulls for an empty subset."""

    if not values:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("task-quality aggregate values must be a finite vector")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _task_quality_aggregates(
    records: list[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    """Aggregate every canonical component over all and successful episodes."""

    if not records:
        raise ValueError("task-quality aggregation requires at least one record")
    schema = _task_quality_schema_from_identity(identity)
    components = schema.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("task-quality identity has no component schema")
    names = [component.get("name") for component in components]
    if any(not isinstance(name, str) or not name for name in names) or len(
        names
    ) != len(set(names)):
        raise ValueError("task-quality component schema is invalid")

    all_values = {name: [] for name in names}
    successful_values = {name: [] for name in names}
    successful_count = 0
    for index, record in enumerate(records):
        if record.get("task_id") != task_id:
            raise ValueError(f"task-quality record {index} task identity mismatch")
        episode_id = record.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"task-quality record {index} episode identity is missing")
        if not isinstance(record.get("success"), bool):
            raise ValueError(
                f"task-quality record {index} success flag must be boolean"
            )
        summary = _validate_task_quality_summary(
            record.get("task_quality"),
            identity=identity,
            task_id=task_id,
            episode_id=episode_id,
        )
        succeeded = bool(record["success"])
        successful_count += int(succeeded)
        raw_components = summary["components"]
        for name in names:
            value = float(raw_components[name]["value"])
            all_values[name].append(value)
            if succeeded:
                successful_values[name].append(value)

    return {
        "record_count": len(records),
        "successful_record_count": successful_count,
        "components": {
            name: {
                "all_records": _aggregate_task_quality_values(all_values[name]),
                "successful_records": _aggregate_task_quality_values(
                    successful_values[name]
                ),
            }
            for name in names
        },
    }


def _task_summary(
    task_id: str,
    records: list[Mapping[str, Any]],
    *,
    task_quality_identity: Mapping[str, Any],
    quality_v2_enabled: bool,
) -> dict[str, Any]:
    """Summarize policy records, including canonical task-quality components."""

    from se3_wam.benchmark.evaluation import summarize_task_records

    summary = summarize_task_records([task_id], records)
    summary[task_id].update(
        safety_failure_rate=float(np.mean([row["safety_failure"] for row in records])),
        mean_return=float(np.mean([row["return"] for row in records])),
        mean_action_l2_sum=float(np.mean([row["action_l2_sum"] for row in records])),
        task_quality=_task_quality_aggregates(
            records,
            identity=task_quality_identity,
            task_id=task_id,
        ),
    )
    if quality_v2_enabled:
        gate_passed = [bool(row["quality_v2_gate"]["passed"]) for row in records]
        successful_rows = [row for row in records if row["success"]]
        successful_gate_passed = [
            bool(row["quality_v2_gate"]["passed"]) for row in successful_rows
        ]
        summary[task_id].update(
            quality_v2_gate_pass_rate=float(np.mean(gate_passed)),
            successful_quality_v2_gate_pass_rate=(
                float(np.mean(successful_gate_passed))
                if successful_gate_passed
                else 0.0
            ),
            training_eligible_rate=float(
                np.mean(
                    [
                        row["success"]
                        and not row["safety_failure"]
                        and row["quality_v2_gate"]["passed"]
                        for row in records
                    ]
                )
            ),
        )
    return summary


def _full_commit(name: str, value: str) -> str:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


def _expected_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(
            "expected policy SHA-256 must be 64 lowercase hexadecimal characters"
        )
    return normalized


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _model_kwargs(config: Mapping[str, Any], state_dim: int) -> dict[str, Any]:
    algorithm = str(config.get("algorithm", ""))
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported policy algorithm {algorithm!r}")
    if state_dim < 1:
        raise ValueError("policy state dimension must be positive")
    if algorithm == "ppo":
        return {
            "obs_dim": state_dim,
            "action_dim": 7,
            "num_action_chunks": 1,
            "add_value_head": True,
            "add_q_head": False,
            "num_q_heads": 2,
        }
    q_heads = int(config.get("q_heads", 2))
    if q_heads < 2:
        raise ValueError("SAC-family policy must declare at least two Q heads")
    return {
        "obs_dim": state_dim,
        "action_dim": 7,
        "num_action_chunks": 1,
        "add_value_head": False,
        "add_q_head": True,
        "q_head_type": "default",
        "critic_obs_dim": state_dim,
        "num_q_heads": q_heads,
    }


class _InferenceMLPPolicy(nn.Module):
    """Actor-only reconstruction of the training MLP for frozen inference.

    Importing ``rlinf.models`` also imports the distributed scheduler and Ray.
    Evaluation and trajectory export only call ``_sample_actions``, so keep this
    path independent of the training stack while validating the omitted head
    weights in :func:`_load_inference_policy`.
    """

    def __init__(self, state_dim: int, algorithm: str) -> None:
        super().__init__()
        self.independent_std = algorithm == "ppo"
        self.final_tanh = algorithm != "ppo"
        self.logstd_range = (-5.0, 2.0)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, 7)
        if self.independent_std:
            self.actor_logstd = nn.Parameter(torch.empty(1, 7))
        else:
            self.actor_logstd = nn.Linear(256, 7)

    def _sample_actions(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        mean = self.actor_mean(features)
        if self.independent_std:
            log_std = self.actor_logstd.expand_as(mean)
        else:
            log_std = self.actor_logstd(features)
            log_std = torch.tanh(log_std)
            low, high = self.logstd_range
            log_std = low + 0.5 * (high - low) * (log_std + 1.0)
        return mean, log_std


def _load_inference_policy(
    config: Mapping[str, Any],
    state_dim: int,
    state_dict: Mapping[str, Any],
    device: torch.device,
) -> _InferenceMLPPolicy:
    """Strictly validate a training checkpoint and load its inference actor."""

    kwargs = _model_kwargs(config, state_dim)
    algorithm = str(config["algorithm"])
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("policy model state must be a non-empty mapping")
    if any(not isinstance(key, str) for key in state_dict):
        raise ValueError("policy model state keys must be strings")
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("policy model state values must be tensors")

    model = _InferenceMLPPolicy(state_dim, algorithm)
    actor_keys = set(model.state_dict())
    available = set(state_dict)
    missing = actor_keys - available
    if missing:
        raise ValueError(f"policy actor state is missing keys: {sorted(missing)}")
    auxiliary = available - actor_keys

    if algorithm == "ppo":
        if not auxiliary or any(not key.startswith("value_head.") for key in auxiliary):
            raise ValueError(
                "PPO checkpoint must contain only actor and value-head state"
            )
    else:
        buffer_keys = {"action_scale", "action_bias"}
        q_keys = auxiliary - buffer_keys
        if auxiliary & buffer_keys != buffer_keys or not q_keys:
            raise ValueError(
                "SAC-family checkpoint is missing Q-head or action buffers"
            )
        if any(not key.startswith("q_head.qs.") for key in q_keys):
            raise ValueError("SAC-family checkpoint contains unsupported model state")
        q_indices: set[int] = set()
        for key in q_keys:
            parts = key.split(".")
            if len(parts) < 4 or not parts[2].isdigit():
                raise ValueError("SAC-family Q-head key has an invalid structure")
            q_indices.add(int(parts[2]))
        if q_indices != set(range(int(kwargs["num_q_heads"]))):
            raise ValueError("SAC-family checkpoint Q-head count does not match config")
        for key, expected in (("action_scale", 1.0), ("action_bias", 0.0)):
            value = state_dict[key]
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError(f"policy {key} buffer must be one finite scalar")
            if float(value.detach().cpu().reshape(())) != expected:
                raise ValueError(f"policy {key} buffer has unsupported bounds")

    actor_state = {key: state_dict[key] for key in actor_keys}
    try:
        model.load_state_dict(actor_state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "policy actor tensor shapes do not match its schema"
        ) from error
    model.to(device)
    model.eval()
    return model


def _validate_policy_payload(
    payload: Mapping[str, Any],
    *,
    rlinf_commit: str,
    benchmark_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if payload.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported Dynamic Benchmark policy schema")
    config = payload.get("config")
    state_schema = payload.get("state_schema")
    if not isinstance(config, dict) or not isinstance(state_schema, dict):
        raise ValueError("policy config/state schema must be mappings")
    if config.get("rlinf_commit") != rlinf_commit:
        raise ValueError("policy RLinf commit does not match the requested source")
    if config.get("benchmark_commit") != benchmark_commit:
        raise ValueError("policy benchmark commit does not match the requested source")
    if not isinstance(config.get("task"), str) or not config["task"]:
        raise ValueError("policy task identity is missing")
    state_dim = int(state_schema.get("state_dim", 0))
    mask_dim = int(state_schema.get("mask_dim", -1))
    if state_dim < 1 or not 0 <= mask_dim <= state_dim:
        raise ValueError("policy state schema dimensions are invalid")
    if not isinstance(payload.get("model"), dict) or not isinstance(
        payload.get("normalizer"), dict
    ):
        raise ValueError("policy model/normalizer state is missing")
    _model_kwargs(config, state_dim)
    return config, state_schema


def _latency_summary(latencies_s: list[float]) -> dict[str, Any]:
    values = np.asarray(latencies_s, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("decision latency samples must be a finite non-empty vector")
    return {
        "sample_count": int(values.size),
        "mean_ms": float(values.mean() * 1000.0),
        "p50_ms": float(np.percentile(values, 50) * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
        "p99_ms": float(np.percentile(values, 99) * 1000.0),
        "max_ms": float(values.max() * 1000.0),
        "p95_meets_20hz": bool(np.percentile(values, 95) <= 0.05),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _manifest_row(env: Any, episode_id: str) -> Any:
    matches = [
        row for row in env._manifest_rows if row.request.episode_id == episode_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"manifest does not uniquely contain episode {episode_id!r}")
    return matches[0]


class _ArmedResetReplayEnv:
    """Expose raw replay while restoring Dynamic Benchmark hidden reset state."""

    def __init__(self, vector_env: Any, raw_env: Any) -> None:
        self._vector_env = vector_env
        self._raw_env = raw_env
        self._terminal_task_quality: Any | None = None

    def reset(self, request: Any) -> Any:
        observation = self._raw_env.reset(request)
        self._vector_env._arm_hidden_t5_event(self._raw_env, request)
        return observation

    def step(self, action: Any) -> Any:
        result = self._raw_env.step(action)
        if bool(getattr(result, "terminated", False)) or bool(
            getattr(result, "truncated", False)
        ):
            self._terminal_task_quality = getattr(result, "task_quality", None)
        return result

    def save_state(self) -> bytes:
        return self._raw_env.save_state()

    @property
    def terminal_task_quality(self) -> Any | None:
        """Return the raw terminal summary observed during independent replay."""

        return self._terminal_task_quality


def _replay_actions_on_fresh_env(
    *,
    vector_env: Any,
    task_id: str,
    request: Any,
    expected_observations: tuple[Any, ...],
    actions: tuple[Any, ...],
    expected_outcomes: tuple[Any, ...],
    expected_final_state: bytes,
    expected_task_quality: Mapping[str, Any] | None = None,
    task_quality_identity: Mapping[str, Any] | None = None,
    replay_fn: Any | None = None,
) -> dict[str, Any]:
    """Replay one expert action tape on an independent canonical raw environment."""
    if replay_fn is None:
        from se3_wam.benchmark.evaluation import replay_actions

        replay_fn = replay_actions
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import (
        _task_quality_make_kwargs,
    )

    task_quality_kwargs = _task_quality_make_kwargs(
        getattr(vector_env, "task_quality_schema_version", None),
        getattr(vector_env, "task_quality_evaluator_backend_id", None),
    )
    raw_env = vector_env._make_mujoco_env(
        task_id,
        image_size=vector_env.image_size,
        camera_observations=vector_env.camera_observations,
        **task_quality_kwargs,
    )
    try:
        replay_env = _ArmedResetReplayEnv(vector_env, raw_env)
        validation = replay_fn(
            replay_env,
            request=request,
            expected_observations=expected_observations,
            actions=actions,
            expected_outcomes=expected_outcomes,
            expected_final_state=expected_final_state,
        )
        if (expected_task_quality is None) != (task_quality_identity is None):
            raise ValueError(
                "expected task quality and its identity must be supplied together"
            )
        if expected_task_quality is None:
            return validation
        if not isinstance(validation, Mapping):
            raise ValueError("replay validation must be a mapping")
        episode_id = getattr(request, "episode_id", None)
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("replay request episode identity is missing")
        assert task_quality_identity is not None
        recorded = _validate_task_quality_summary(
            expected_task_quality,
            identity=task_quality_identity,
            task_id=task_id,
            episode_id=episode_id,
        )
        replayed = _validate_task_quality_summary(
            replay_env.terminal_task_quality,
            identity=task_quality_identity,
            task_id=task_id,
            episode_id=episode_id,
        )
        result = dict(validation)
        result["task_quality_exact"] = replayed == recorded
        result["task_quality_summary_sha256"] = replayed["summary_sha256"]
        result["passed"] = bool(
            result.get("passed") is True and result["task_quality_exact"]
        )
        return result
    finally:
        raw_env.close()


def _reset_rollout_on_fresh_env(*, vector_env: Any, request: Any) -> Any:
    """Start one expert rollout from a newly constructed raw environment."""
    if int(vector_env.num_envs) != 1:
        raise ValueError("expert evaluation requires exactly one vector member")
    if request.task_id != vector_env.task_id:
        raise ValueError(
            "expert evaluation request task does not match the environment"
        )
    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import (
        _task_quality_make_kwargs,
    )

    task_quality_kwargs = _task_quality_make_kwargs(
        getattr(vector_env, "task_quality_schema_version", None),
        getattr(vector_env, "task_quality_evaluator_backend_id", None),
    )
    raw_env = vector_env._make_mujoco_env(
        vector_env.task_id,
        image_size=vector_env.image_size,
        camera_observations=vector_env.camera_observations,
        **task_quality_kwargs,
    )
    try:
        if int(raw_env.horizon_steps) != int(vector_env.horizon_steps):
            raise RuntimeError("fresh expert raw environment changed the horizon")
        observation = raw_env.reset(request)
        vector_env._arm_hidden_t5_event(raw_env, request)
        state = vector_env._encode(observation, request)
    except BaseException:
        raw_env.close()
        raise

    previous_env = vector_env.envs[0]
    try:
        previous_env.close()
    except BaseException:
        raw_env.close()
        raise
    vector_env.envs[0] = raw_env
    vector_env._reset_metrics(np.asarray([0], dtype=np.int64))
    vector_env._raw_observations[0] = observation
    vector_env._requests[0] = request
    vector_env._needs_reset[0] = False
    vector_env._last_obs = {"states": torch.as_tensor(state[None], dtype=torch.float32)}
    vector_env._is_start = True
    return observation


def _make_teacher(task: str, request: Any) -> Any:
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    teacher, _ = make_privileged_teacher(task, request=request)
    if hasattr(teacher, "reset"):
        teacher.reset()
    return teacher


def _episode(
    *,
    env: Any,
    model: Any,
    normalizer: RunningNormalizer,
    config: Mapping[str, Any],
    device: torch.device,
    task_quality_identity: Mapping[str, Any],
    quality_v2_thresholds: Mapping[str, object] | None = None,
    quality_v2_thresholds_sha256: str | None = None,
    attempt_output: Path | None = None,
    attempt_index: int | None = None,
) -> tuple[dict[str, Any], list[float]]:
    from se3_wam.benchmark.metrics import (
        completion_time_from_events,
        hierarchical_task_completion,
        validate_stage_event_order,
    )

    request = env._requests[0]
    observation = env._raw_observations[0]
    obs = env._last_obs
    if request is None or observation is None or obs is None:
        raise RuntimeError("evaluation environment is not initialized")
    task_id = str(config["task"])
    row = _manifest_row(env, request.episode_id)
    raw_env = env.envs[0]
    residual = str(config["algorithm"]) == "residual_rlpd"
    teacher = _make_teacher(task_id, request) if residual else None
    residual_scale = float(config.get("residual_scale", 0.25))
    observations = [observation]
    states = [np.asarray(obs["states"][0], dtype=np.float32)]
    actions = []
    policy_action_values = []
    tape_policy_action_values = []
    outcomes = []
    rewards = []
    terminated_rows = []
    truncated_rows = []
    latencies_s = []
    result_info: dict[str, Any] | None = None
    terminated_value = False
    truncated_value = False
    model.eval()
    while not (terminated_value or truncated_value):
        _sync(device)
        started = time.perf_counter()
        with torch.inference_mode():
            policy_actions, _ = _policy_action(
                model,
                normalizer,
                obs["states"],
                device,
                stochastic=False,
            )
        policy_actions = policy_actions.cpu()
        env_actions = policy_actions
        if teacher is not None:
            env_actions = _compose_residual_actions(
                _planner_actions(env, [teacher]),
                policy_actions,
                residual_scale,
            )
        _sync(device)
        latencies_s.append(time.perf_counter() - started)
        values = np.clip(
            np.asarray(env_actions[0], dtype=np.float64),
            -1.0,
            1.0,
        )
        action = env._ActionCommand(
            mode=request.action_mode,
            values=values,
            policy_step=observation.policy_step,
        )
        next_obs, reward, terminated, truncated, infos = env.step(
            env_actions,
            auto_reset=False,
        )
        next_observation = env._raw_observations[0]
        if next_observation is None:
            raise RuntimeError("evaluation environment lost its raw observation")
        terminated_value = bool(terminated[0])
        truncated_value = bool(truncated[0])
        reason = infos["termination_reason"][0]
        active_progress = float(infos["reward_inputs"]["active_stage_progress"][0])
        observations.append(next_observation)
        states.append(np.asarray(next_obs["states"][0], dtype=np.float32))
        actions.append(action)
        policy_action_values.append(np.asarray(policy_actions[0], dtype=np.float64))
        tape_policy_action_values.append(
            np.asarray(policy_actions[0], dtype=np.float32)
        )
        outcomes.append(
            (
                terminated_value,
                truncated_value,
                bool(infos["success"][0]),
                reason,
                active_progress,
            )
        )
        rewards.append(float(reward[0]))
        terminated_rows.append(terminated_value)
        truncated_rows.append(truncated_value)
        result_info = {
            "success": bool(infos["success"][0]),
            "termination_reason": reason,
            "active_stage_progress": active_progress,
            "trajectory_completion": float(infos["trajectory_completion"][0]),
        }
        if terminated_value or truncated_value:
            result_info["task_quality"] = _task_quality_from_terminal_infos(
                infos,
                identity=task_quality_identity,
                task_id=task_id,
                episode_id=request.episode_id,
            )
        observation = next_observation
        obs = next_obs
        if len(actions) > int(env.horizon_steps):
            raise RuntimeError("evaluation rollout exceeded the environment horizon")
    if result_info is None:
        raise RuntimeError("evaluation policy produced no action")

    events = tuple(raw_env._ledger.events)
    final_state = raw_env.save_state()
    task = env._get_task_spec(task_id)
    completed = validate_stage_event_order(task, events)
    completion = hierarchical_task_completion(
        task,
        completed,
        result_info["active_stage_progress"],
    )
    completion_time = (
        completion_time_from_events(
            events,
            start_event=task.task_start_event,
            success_event=task.success_stages[-1],
        )
        if result_info["success"]
        else None
    )
    replay_validation = _replay_actions_on_fresh_env(
        vector_env=env,
        task_id=task_id,
        request=request,
        expected_observations=tuple(observations),
        actions=tuple(actions),
        expected_outcomes=tuple(outcomes),
        expected_final_state=final_state,
        expected_task_quality=result_info["task_quality"],
        task_quality_identity=task_quality_identity,
    )
    if not replay_validation["passed"]:
        raise RuntimeError(
            "expert rollout replay failed: "
            f"{request.episode_id}: {json.dumps(replay_validation, sort_keys=True)}"
        )
    action_array = np.stack([action.values for action in actions])
    policy_action_array = np.stack(policy_action_values)
    from se3_wam.benchmark.trajectory_quality import (
        evaluate_quality_v2_gate,
        trajectory_quality_v2_from_observations,
    )

    quality_v2 = trajectory_quality_v2_from_observations(
        observations,
        action_array,
        task_id=task_id,
        task_config=getattr(raw_env, "task_config", None),
        sample_period_s=0.05,
        continuous_dimensions=max(1, int(action_array.shape[1]) - 1),
    )
    quality_v2_gate = (
        None
        if quality_v2_thresholds is None
        else evaluate_quality_v2_gate(
            quality_v2,
            quality_v2_thresholds,
            task_id=task_id,
        )
    )
    if quality_v2_gate is not None:
        if quality_v2_thresholds_sha256 is None:
            raise ValueError("quality-v2 gate is missing its threshold SHA-256")
        quality_v2_gate["contract_sha256"] = quality_v2_thresholds_sha256
    safety_failures = set(env.reward_schema["safety_failures"])
    record = {
        "episode_id": request.episode_id,
        "task_id": task_id,
        "seed": request.seed,
        "factors": dict(request.factors),
        "source_group_id": row.source_group_id,
        "pair_id": row.pair_id,
        "pair_member_id": row.pair_member_id,
        "candidate_index": row.candidate_index,
        "success": result_info["success"],
        "safety_failure": result_info["termination_reason"] in safety_failures,
        "termination_reason": result_info["termination_reason"],
        "trajectory_completion": completion,
        "completion_time_s": completion_time,
        "return": float(sum(rewards)),
        "control_steps": len(actions),
        "action_l2_sum": float(np.square(action_array).sum()),
        "task_quality": result_info["task_quality"],
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(action_array).tobytes()
        ).hexdigest(),
        "policy_action_sha256": hashlib.sha256(
            np.ascontiguousarray(policy_action_array).tobytes()
        ).hexdigest(),
        "actions": action_array.tolist(),
        "quality_v2": quality_v2,
        "quality_v2_sha256": _payload_sha256(quality_v2),
        "quality_v2_gate": quality_v2_gate,
        "replay_validation": replay_validation,
        "events": [event.name for event in events],
    }
    producer_arguments = (
        quality_v2_thresholds is not None,
        quality_v2_thresholds_sha256 is not None,
        attempt_output is not None,
        attempt_index is not None,
    )
    if any(producer_arguments) and not all(producer_arguments):
        raise ValueError("formal attempt producer arguments must be supplied together")
    if all(producer_arguments):
        assert attempt_output is not None
        assert attempt_index is not None
        assert quality_v2_thresholds_sha256 is not None
        record = materialize_evaluation_attempt(
            attempt_output,
            record,
            candidate_index=attempt_index,
            raw_env=raw_env,
            observations=observations,
            states=states,
            policy_actions=tape_policy_action_values,
            rewards=rewards,
            terminated=terminated_rows,
            truncated=truncated_rows,
            quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
        )
    return record, latencies_s


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _evaluation_schema(*, formal_attempts: bool) -> str:
    """Return v0.3 only for a frozen-threshold formal attempt producer."""

    return FORMAL_EVALUATION_SCHEMA if formal_attempts else EVALUATION_SCHEMA


def _immutable_image_id(value: str | None, label: str) -> str:
    if value is None or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be an immutable Docker image ID")
    return value


def _source_snapshot_receipt(
    *,
    manifest_sha256: str,
    manifest: Mapping[str, Any],
    base_image_id: str,
    source_snapshot_image_id: str,
    evaluator_source_identity: Mapping[str, str],
) -> dict[str, Any]:
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("source snapshot manifest sources are missing")
    projected_sources: dict[str, dict[str, Any]] = {}
    for role in ("policy_rlinf", "evaluator_rlinf", "benchmark"):
        source = sources.get(role)
        if not isinstance(source, Mapping):
            raise ValueError(f"source snapshot {role} identity is missing")
        projected_sources[role] = {
            name: source[name]
            for name in ("root", "commit", "tree", "inventory_sha256")
        }
    immutable_base_image_id = _immutable_image_id(base_image_id, "base image ID")
    immutable_snapshot_image_id = _immutable_image_id(
        source_snapshot_image_id, "source snapshot image ID"
    )
    if immutable_snapshot_image_id == immutable_base_image_id:
        raise ValueError("source snapshot image must differ from its base image")
    receipt: dict[str, Any] = {
        "schema_version": SOURCE_SNAPSHOT_RECEIPT_SCHEMA,
        "base_image_id": immutable_base_image_id,
        "source_snapshot_image_id": immutable_snapshot_image_id,
        "source_manifest": {
            "path": SOURCE_SNAPSHOT_MANIFEST_FILENAME,
            "sha256": manifest_sha256,
            "schema_version": SOURCE_SNAPSHOT_SCHEMA,
            "payload_sha256": manifest["payload_sha256"],
        },
        "sources": projected_sources,
        "evaluator_source": dict(evaluator_source_identity),
    }
    receipt["payload_sha256"] = _payload_sha256(receipt)
    return receipt


def main() -> None:
    from se3_wam.benchmark.contracts import canonical_json
    from se3_wam.benchmark.evaluation import manifest_record

    from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import DynamicBenchmarkEnv

    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    evaluator_commit = _full_commit("evaluator_commit", args.evaluator_commit)
    rlinf_commit = _full_commit("rlinf_commit", args.rlinf_commit)
    benchmark_commit = _full_commit("benchmark_commit", args.benchmark_commit)
    expected_policy_sha256 = _expected_sha256(args.expected_policy_sha256)
    if (args.quality_v2_thresholds is None) != (
        args.expected_quality_v2_thresholds_sha256 is None
    ):
        raise ValueError(
            "quality-v2 thresholds and expected SHA-256 must be supplied together"
        )
    quality_v2_thresholds: Mapping[str, object] | None = None
    quality_v2_thresholds_sha256: str | None = None
    if args.quality_v2_thresholds is not None:
        if not args.quality_v2_thresholds.is_file():
            raise FileNotFoundError(args.quality_v2_thresholds)
        quality_v2_thresholds_sha256 = _sha256(args.quality_v2_thresholds)
        expected_thresholds_sha256 = _expected_sha256(
            str(args.expected_quality_v2_thresholds_sha256)
        )
        if quality_v2_thresholds_sha256 != expected_thresholds_sha256:
            raise ValueError(
                "quality-v2 threshold SHA-256 does not match the expected identity"
            )
        loaded_thresholds = json.loads(
            args.quality_v2_thresholds.read_text(encoding="utf-8")
        )
        if not isinstance(loaded_thresholds, Mapping):
            raise ValueError("quality-v2 threshold contract must be a mapping")
        quality_v2_thresholds = dict(loaded_thresholds)
    if not args.policy.is_file():
        raise FileNotFoundError(args.policy)
    policy_sha256 = _sha256(args.policy)
    if policy_sha256 != expected_policy_sha256:
        raise ValueError("policy SHA-256 does not match the expected identity")
    admission_arguments = (
        args.trainer_run_root,
        args.trainer_summary,
        args.checkpoint_selection,
        args.checkpoint_selection_outcome,
        args.expected_checkpoint_selection_outcome_sha256,
        args.policy_rlinf_source_root,
        args.evaluator_rlinf_source_root,
    )
    if any(value is not None for value in admission_arguments) and not all(
        value is not None for value in admission_arguments
    ):
        raise ValueError(
            "trainer summary, checkpoint-selection manifest, checkpoint-selection "
            "outcome, expected outcome SHA-256, explicit trainer run root, policy "
            "source root, and evaluator source root must be supplied together"
        )
    if args.quality_v2_thresholds is not None and args.trainer_summary is None:
        raise ValueError(
            "formal learned-policy evaluation requires the trainer summary and "
            "checkpoint-selection manifest"
        )
    snapshot_arguments = (
        args.benchmark_source_root,
        args.source_snapshot_manifest,
        args.expected_source_snapshot_manifest_sha256,
        args.base_image_id,
        args.source_snapshot_image_id,
    )
    if any(value is not None for value in snapshot_arguments) and not all(
        value is not None for value in snapshot_arguments
    ):
        raise ValueError("source snapshot arguments must be supplied together")
    if args.source_snapshot_manifest is not None and args.trainer_summary is None:
        raise ValueError(
            "an immutable source snapshot is valid only with learned-policy admission"
        )
    if args.quality_v2_thresholds is not None and not all(
        value is not None for value in snapshot_arguments
    ):
        raise ValueError(
            "formal learned-policy evaluation requires an immutable source snapshot"
        )
    source_snapshot_manifest: dict[str, Any] | None = None
    source_snapshot_manifest_sha256: str | None = None
    if args.source_snapshot_manifest is not None:
        assert args.benchmark_source_root is not None
        assert args.expected_source_snapshot_manifest_sha256 is not None
        assert args.base_image_id is not None
        source_snapshot_manifest_sha256 = _expected_sha256(
            args.expected_source_snapshot_manifest_sha256
        )
        source_snapshot_manifest = validate_source_snapshot_manifest(
            args.source_snapshot_manifest,
            expected_sha256=source_snapshot_manifest_sha256,
            expected_base_image_id=args.base_image_id,
            expected_sources={
                "policy_rlinf": (
                    args.policy_rlinf_source_root,
                    rlinf_commit,
                    None,
                ),
                "evaluator_rlinf": (
                    args.evaluator_rlinf_source_root,
                    evaluator_commit,
                    None,
                ),
                "benchmark": (
                    args.benchmark_source_root,
                    benchmark_commit,
                    None,
                ),
            },
            verify_inventory=True,
        )
        snapshot_environment = {
            "RLD2_SOURCE_SNAPSHOT_MANIFEST": str(
                args.source_snapshot_manifest.resolve(strict=True)
            ),
            "RLD2_SOURCE_SNAPSHOT_MANIFEST_SHA256": (source_snapshot_manifest_sha256),
        }
        for name, value in snapshot_environment.items():
            if os.environ.get(name) not in {None, value}:
                raise ValueError(f"conflicting {name} environment identity")
            os.environ[name] = value
    learned_policy_admission: dict[str, Any] | None = None
    evaluator_source_identity: dict[str, str] | None = None
    if args.trainer_summary is not None:
        assert args.trainer_run_root is not None
        assert args.checkpoint_selection is not None
        learned_policy_admission = validate_selected_learned_policy(
            trainer_run_root=args.trainer_run_root,
            policy_path=args.policy,
            trainer_summary_path=args.trainer_summary,
            checkpoint_selection_path=args.checkpoint_selection,
            checkpoint_selection_outcome_path=args.checkpoint_selection_outcome,
            policy_rlinf_source_root=args.policy_rlinf_source_root,
            verifier_rlinf_source_root=args.evaluator_rlinf_source_root,
            evaluator_rlinf_source_root=args.evaluator_rlinf_source_root,
            expected_checkpoint_selection_outcome_sha256=(
                args.expected_checkpoint_selection_outcome_sha256
            ),
            expected_policy_sha256=policy_sha256,
            expected_rlinf_commit=rlinf_commit,
            expected_benchmark_commit=benchmark_commit,
            expected_verifier_rlinf_commit=evaluator_commit,
            expected_evaluator_rlinf_commit=evaluator_commit,
        )
        evaluator_source_identity = _actual_evaluator_source_identity(
            args.evaluator_rlinf_source_root
        )
        checkpoint_outcome = json.loads(
            args.checkpoint_selection_outcome.read_text(encoding="utf-8")
        )
        outcome_source_identity = checkpoint_outcome.get("source_identity")
        if (
            not isinstance(outcome_source_identity, Mapping)
            or outcome_source_identity.get("evaluator_source")
            != evaluator_source_identity
        ):
            raise ValueError(
                "executing evaluator source identity does not match checkpoint outcome"
            )
    payload = torch.load(args.policy, map_location="cpu", weights_only=False)
    config, state_schema = _validate_policy_payload(
        payload,
        rlinf_commit=rlinf_commit,
        benchmark_commit=benchmark_commit,
    )
    task_id = str(config["task"])
    if quality_v2_thresholds is not None:
        assert quality_v2_thresholds_sha256 is not None
        validate_formal_quality_v2_thresholds(
            quality_v2_thresholds,
            task_id=task_id,
            thresholds_sha256=quality_v2_thresholds_sha256,
        )
    task_quality_identity = _task_quality_identity(task_id)
    device = _device(args.device)
    state_dim = int(state_schema["state_dim"])
    model = _load_inference_policy(config, state_dim, payload["model"], device)
    normalizer = RunningNormalizer(state_dim, int(state_schema["mask_dim"]))
    normalizer.load_state_dict(payload["normalizer"])

    manifest_size = max(args.episodes, 2)
    if manifest_size % 2:
        manifest_size += 1
    env = DynamicBenchmarkEnv(
        cfg={
            "task_id": task_id,
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "manifest_size": manifest_size,
            "image_size": int(config.get("image_size", 64)),
            "camera_observations": False,
            "auto_reset": False,
            "ignore_terminations": False,
            "group_size": 1,
            "features": config.get("features", {}),
            "reward_components": config.get("reward_components", {}),
            "reward_lift_shaping_weight": float(
                config.get("reward_lift_shaping_weight", 0.0)
            ),
            "reward_orientation_shaping_weight": float(
                config.get("reward_orientation_shaping_weight", 0.0)
            ),
            "state_derived_features": list(config.get("state_derived_features", [])),
            **_task_quality_env_config(task_quality_identity),
        },
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        if env.state_schema != state_schema:
            raise ValueError(
                "evaluation environment state schema does not match the policy"
            )
        rows = list(env._manifest_rows[: args.episodes])
        if len(rows) != args.episodes:
            raise RuntimeError("evaluation manifest is shorter than requested")
        args.output.mkdir(parents=True)
        reset_manifest_path = args.output / "reset_manifest.jsonl"
        reset_manifest_path.write_text(
            "".join(canonical_json(manifest_record(row)) + "\n" for row in rows),
            encoding="utf-8",
        )
        started = time.time()
        records = []
        latencies_s: list[float] = []
        for episode_index, row in enumerate(rows):
            _reset_rollout_on_fresh_env(vector_env=env, request=row.request)
            record, episode_latencies = _episode(
                env=env,
                model=model,
                normalizer=normalizer,
                config=config,
                device=device,
                task_quality_identity=task_quality_identity,
                quality_v2_thresholds=quality_v2_thresholds,
                quality_v2_thresholds_sha256=quality_v2_thresholds_sha256,
                attempt_output=(
                    args.output if quality_v2_thresholds is not None else None
                ),
                attempt_index=(
                    episode_index if quality_v2_thresholds is not None else None
                ),
            )
            if record["episode_id"] != row.request.episode_id:
                raise RuntimeError(
                    "rollout order diverged from the frozen reset manifest"
                )
            records.append(record)
            latencies_s.extend(episode_latencies)
            print(
                json.dumps(
                    {
                        "episode_id": record["episode_id"],
                        "success": record["success"],
                        "safety_failure": record["safety_failure"],
                        "trajectory_completion": record["trajectory_completion"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        task_summary = _task_summary(
            task_id,
            records,
            task_quality_identity=task_quality_identity,
            quality_v2_enabled=quality_v2_thresholds is not None,
        )
        latency = _latency_summary(latencies_s)
        policy_identity = {
            "path": str(args.policy.resolve()),
            "sha256": policy_sha256,
            "schema_version": payload["schema_version"],
            "task": task_id,
            "algorithm": config["algorithm"],
            "training_seed": config["seed"],
            "training_env_steps": payload["env_steps"],
            "validation": payload["validation"],
        }
        if learned_policy_admission is not None:
            policy_identity["checkpoint_role"] = learned_policy_admission[
                "checkpoint_role"
            ]
        source_identity: dict[str, Any] = {
            "evaluator_rlinf_commit": evaluator_commit,
            "policy_rlinf_commit": rlinf_commit,
            "benchmark_commit": benchmark_commit,
        }
        if evaluator_source_identity is not None:
            source_identity["evaluator_source"] = evaluator_source_identity
        result = {
            "schema_version": _evaluation_schema(
                formal_attempts=quality_v2_thresholds is not None
            ),
            "policy_identity": policy_identity,
            "source_identity": source_identity,
            "task_quality_identity": task_quality_identity,
            "quality_v2_threshold_identity": (
                None
                if quality_v2_thresholds is None
                else {
                    "schema_version": quality_v2_thresholds.get("schema_version"),
                    "sha256": quality_v2_thresholds_sha256,
                }
            ),
            "split": args.split,
            "manifest_seed": args.manifest_seed,
            "reset_manifest_sha256": _sha256(reset_manifest_path),
            "episodes": args.episodes,
            "device": str(device),
            "state_schema": state_schema,
            "records": records,
            "task_summary": task_summary,
            "decision_latency": latency,
            "all_replays_passed": all(
                row["replay_validation"]["passed"] for row in records
            ),
            "all_successful_quality_v2_gates_passed": (
                None
                if quality_v2_thresholds is None
                else all(
                    bool(row["quality_v2_gate"]["passed"])
                    for row in records
                    if row["success"]
                )
            ),
            "started_unix_s": started,
            "finished_unix_s": time.time(),
        }
        if learned_policy_admission is not None:
            result["learned_policy_admission"] = learned_policy_admission
        if _sha256(args.policy) != policy_sha256:
            raise RuntimeError("policy file changed during evaluation")
        if learned_policy_admission is not None:
            try:
                observed_admission = validate_selected_learned_policy(
                    trainer_run_root=args.trainer_run_root,
                    policy_path=args.policy,
                    trainer_summary_path=args.trainer_summary,
                    checkpoint_selection_path=args.checkpoint_selection,
                    checkpoint_selection_outcome_path=(
                        args.checkpoint_selection_outcome
                    ),
                    policy_rlinf_source_root=args.policy_rlinf_source_root,
                    verifier_rlinf_source_root=args.evaluator_rlinf_source_root,
                    evaluator_rlinf_source_root=args.evaluator_rlinf_source_root,
                    expected_checkpoint_selection_outcome_sha256=(
                        learned_policy_admission["checkpoint_selection_outcome"][
                            "sha256"
                        ]
                    ),
                    expected_policy_sha256=policy_sha256,
                    expected_trainer_summary_sha256=learned_policy_admission[
                        "trainer_summary"
                    ]["sha256"],
                    expected_checkpoint_selection_sha256=learned_policy_admission[
                        "checkpoint_selection"
                    ]["sha256"],
                    expected_rlinf_commit=rlinf_commit,
                    expected_benchmark_commit=benchmark_commit,
                    expected_verifier_rlinf_commit=evaluator_commit,
                    expected_evaluator_rlinf_commit=evaluator_commit,
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise RuntimeError(
                    "learned-policy admission artifacts changed during evaluation"
                ) from error
            if observed_admission != learned_policy_admission:
                raise RuntimeError(
                    "learned-policy admission identity changed during evaluation"
                )
        if (
            args.quality_v2_thresholds is not None
            and _sha256(args.quality_v2_thresholds) != quality_v2_thresholds_sha256
        ):
            raise RuntimeError("quality-v2 threshold file changed during evaluation")
        source_snapshot_receipt: dict[str, Any] | None = None
        if source_snapshot_manifest is not None:
            assert source_snapshot_manifest_sha256 is not None
            assert args.source_snapshot_manifest is not None
            assert args.base_image_id is not None
            assert args.source_snapshot_image_id is not None
            assert args.benchmark_source_root is not None
            observed_snapshot_manifest = validate_source_snapshot_manifest(
                args.source_snapshot_manifest,
                expected_sha256=source_snapshot_manifest_sha256,
                expected_base_image_id=args.base_image_id,
                expected_sources={
                    "policy_rlinf": (
                        args.policy_rlinf_source_root,
                        rlinf_commit,
                        None,
                    ),
                    "evaluator_rlinf": (
                        args.evaluator_rlinf_source_root,
                        evaluator_commit,
                        None,
                    ),
                    "benchmark": (
                        args.benchmark_source_root,
                        benchmark_commit,
                        None,
                    ),
                },
                verify_inventory=True,
            )
            if observed_snapshot_manifest != source_snapshot_manifest:
                raise RuntimeError("source snapshot identity changed during evaluation")
            if evaluator_source_identity is None or (
                _actual_evaluator_source_identity(args.evaluator_rlinf_source_root)
                != evaluator_source_identity
            ):
                raise RuntimeError(
                    "executing evaluator source changed during evaluation"
                )
            published_manifest_path = args.output / SOURCE_SNAPSHOT_MANIFEST_FILENAME
            source_manifest_bytes = args.source_snapshot_manifest.read_bytes()
            if hashlib.sha256(source_manifest_bytes).hexdigest() != (
                source_snapshot_manifest_sha256
            ):
                raise RuntimeError(
                    "source snapshot manifest changed before publication"
                )
            _atomic_bytes(published_manifest_path, source_manifest_bytes)
            if _sha256(published_manifest_path) != source_snapshot_manifest_sha256:
                raise RuntimeError(
                    "published source snapshot manifest identity mismatch"
                )
            source_snapshot_receipt = _source_snapshot_receipt(
                manifest_sha256=source_snapshot_manifest_sha256,
                manifest=source_snapshot_manifest,
                base_image_id=args.base_image_id,
                source_snapshot_image_id=args.source_snapshot_image_id,
                evaluator_source_identity=evaluator_source_identity,
            )
        result["payload_sha256"] = _payload_sha256(result)
        result_path = args.output / "evaluation.json"
        _atomic_json(result_path, result)
        if source_snapshot_receipt is not None:
            _atomic_json(args.output / "source_snapshot.json", source_snapshot_receipt)
        admission_checksums = (
            ()
            if learned_policy_admission is None
            else tuple(
                (
                    learned_policy_admission[name]["sha256"],
                    learned_policy_admission[name]["path"],
                )
                for name in (
                    "trainer_summary",
                    "checkpoint_selection",
                    "checkpoint_selection_outcome",
                    "config",
                )
            )
        )
        checksums = (
            recursive_output_checksums(
                args.output,
                extra_entries=(
                    (policy_sha256, str(args.policy.resolve())),
                    *admission_checksums,
                ),
            )
            if quality_v2_thresholds is not None
            else (
                f"{_sha256(result_path)}  evaluation.json\n"
                f"{_sha256(reset_manifest_path)}  reset_manifest.jsonl\n"
                f"{policy_sha256}  {args.policy.resolve()}\n"
                + "".join(f"{sha256}  {path}\n" for sha256, path in admission_checksums)
            )
        )
        (args.output / "SHA256SUMS").write_text(
            checksums, encoding="utf-8", newline="\n"
        )
        print(
            json.dumps(
                {
                    "evaluation_sha256": _sha256(result_path),
                    "payload_sha256": result["payload_sha256"],
                    "all_replays_passed": result["all_replays_passed"],
                    "decision_latency": latency,
                    "task_summary": task_summary,
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
