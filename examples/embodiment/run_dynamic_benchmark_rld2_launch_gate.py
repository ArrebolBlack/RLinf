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

"""Execute one allocated RLD2 calibration/compatibility lane.

The launcher is intentionally not self-scheduling.  A resource owner must first
record the A100X8 allocation, stop only the allocated keepalive lanes, and invoke
the authoritative resource wrapper.  This program then consumes one signed lane
plan and refuses any non-empty output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from examples.embodiment.build_dynamic_benchmark_rld2_evidence import (
    build_calibration_evidence,
)
from examples.embodiment.evaluate_dynamic_benchmark_expert import (
    RunningNormalizer,
    _load_inference_policy,
    _policy_action,
    _validate_policy_payload,
)
from examples.embodiment.evaluate_dynamic_benchmark_planner import (
    _planner_action_values,
    _reset_rollout_on_fresh_env,
)
from examples.embodiment.prepare_dynamic_benchmark_rld2_launch_gate import (
    BACKEND_ID,
    CALIBRATION_JOB_SCHEMA,
    CHECKPOINT_REQUEST_SCHEMA,
    LANE_PLAN_SCHEMA,
    LaunchGateError,
    _git_identity,
    _load_json,
    _require_commit,
    _require_sha256,
    _sha256,
    validate_package,
)

COMPATIBILITY_PROBE_SCHEMA = (
    "rlinf-dynamic-benchmark-rld2-compatibility-probe-output-v0.1"
)
CALIBRATION_OUTPUT_SCHEMA = (
    "rlinf-dynamic-benchmark-rld2-calibration-output-v0.1"
)
LANE_RESULT_SCHEMA = "rlinf-dynamic-benchmark-rld2-lane-result-v0.1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_output_root(output_root: Path, build: Any) -> dict[str, Any]:
    if output_root.exists():
        raise LaunchGateError(f"refusing to overwrite output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        result = build(staging)
        os.replace(staging, output_root)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _tree_sha256sums(root: Path) -> str:
    rows = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    return "\n".join(rows) + "\n"


def _environment_config(
    *, task: str, split: str, manifest_seed: int, image_size: int, policy: Mapping[str, Any] | None
) -> dict[str, Any]:
    config = {
        "task_id": task,
        "split": split,
        "manifest_seed": manifest_seed,
        "manifest_size": 2,
        "image_size": image_size,
        "camera_observations": False,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
    }
    if policy is not None:
        config.update(
            features=policy.get("features", {}),
            reward_components=policy.get("reward_components", {}),
            reward_lift_shaping_weight=float(
                policy.get("reward_lift_shaping_weight", 0.0)
            ),
            reward_orientation_shaping_weight=float(
                policy.get("reward_orientation_shaping_weight", 0.0)
            ),
            state_derived_features=list(policy.get("state_derived_features", [])),
        )
    return config


def _tensor_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "request_id",
        "task",
        "policy_path",
        "policy_sha256",
        "policy_rlinf_commit",
        "policy_benchmark_commit",
        "policy_state_schema_sha256",
        "policy_state_dim",
        "policy_mask_dim",
        "policy_action_dim",
        "manifest_seed",
        "manifest_episode_index",
        "split",
        "test_exposure",
        "expected_output",
        "lane",
    }
    if set(request) != expected or request.get("schema_version") != CHECKPOINT_REQUEST_SCHEMA:
        raise LaunchGateError("compatibility request schema mismatch")
    if request.get("split") != "validation" or request.get("test_exposure") != {
        "test_id": False,
        "test_ood": False,
    }:
        raise LaunchGateError("compatibility request used a formal test split")
    if request.get("manifest_episode_index") != 0:
        raise LaunchGateError("compatibility request must use frozen manifest index zero")
    _require_sha256(request.get("policy_sha256"), "compatibility policy")
    _require_sha256(
        request.get("policy_state_schema_sha256"), "compatibility policy state schema"
    )
    _require_commit(request.get("policy_rlinf_commit"), "compatibility policy RLinf commit")
    _require_commit(
        request.get("policy_benchmark_commit"), "compatibility policy benchmark commit"
    )
    return dict(request)


def collect_compatibility_probe(
    request: Mapping[str, Any], output_root: Path, *, device_name: str
) -> dict[str, Any]:
    """Load, infer, reset, and step one checkpoint against the deployed evaluator."""

    normalized = _validate_request(request)

    def build(staging: Path) -> dict[str, Any]:
        import torch
        from se3_wam.benchmark.config import task_config_sha256
        from se3_wam.benchmark.contracts import canonical_json
        from se3_wam.benchmark.evaluation import manifest_record

        from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import (
            DynamicBenchmarkEnv,
        )

        policy_path = Path(normalized["policy_path"])
        if _sha256(policy_path) != normalized["policy_sha256"]:
            raise LaunchGateError("compatibility checkpoint hash mismatch")
        payload = torch.load(policy_path, map_location="cpu", weights_only=False)
        config, state_schema = _validate_policy_payload(
            payload,
            rlinf_commit=normalized["policy_rlinf_commit"],
            benchmark_commit=normalized["policy_benchmark_commit"],
        )
        if (
            _payload_sha256(state_schema) != normalized["policy_state_schema_sha256"]
            or int(state_schema["state_dim"]) != normalized["policy_state_dim"]
            or int(state_schema["mask_dim"]) != normalized["policy_mask_dim"]
        ):
            raise LaunchGateError("compatibility checkpoint state schema drifted")
        if device_name == "cuda":
            if not torch.cuda.is_available():
                raise LaunchGateError("CUDA compatibility probe requested without CUDA")
            device = torch.device("cuda:0")
        elif device_name == "cpu":
            device = torch.device("cpu")
        else:
            raise LaunchGateError("compatibility device must be cpu or cuda")
        model = _load_inference_policy(
            config, int(state_schema["state_dim"]), payload["model"], device
        )
        normalizer = RunningNormalizer(
            int(state_schema["state_dim"]), int(state_schema["mask_dim"])
        )
        normalizer.load_state_dict(payload["normalizer"])
        env_config = _environment_config(
            task=normalized["task"],
            split=normalized["split"],
            manifest_seed=normalized["manifest_seed"],
            image_size=int(config.get("image_size", 64)),
            policy=config,
        )
        env = DynamicBenchmarkEnv(
            cfg=env_config,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        try:
            observation, _ = env.reset()
            request_value = env._requests[0]
            if request_value is None:
                raise LaunchGateError("compatibility reset did not expose a request")
            row = env._manifest_rows[0]
            reset_record = manifest_record(row)
            if env.state_schema != state_schema:
                raise LaunchGateError("evaluator state schema is checkpoint-incompatible")
            states = observation["states"]
            if not torch.isfinite(states).all():
                raise LaunchGateError("compatibility observation is not finite")
            with torch.inference_mode():
                actions, _ = _policy_action(
                    model,
                    normalizer,
                    states,
                    device,
                    stochastic=False,
                )
            actions = actions.cpu()
            if not torch.isfinite(actions).all():
                raise LaunchGateError("compatibility action is not finite")
            _, reward, _, _, _ = env.step(actions, auto_reset=False)
            reward_value = float(reward[0])
            if not math.isfinite(reward_value):
                raise LaunchGateError("compatibility reward is not finite")
            probe = {
                "task": normalized["task"],
                "policy_sha256": normalized["policy_sha256"],
                "policy_rlinf_commit": normalized["policy_rlinf_commit"],
                "policy_state_schema_sha256": normalized[
                    "policy_state_schema_sha256"
                ],
                "policy_state_dim": normalized["policy_state_dim"],
                "policy_mask_dim": normalized["policy_mask_dim"],
                "evaluator_state_schema_sha256": _payload_sha256(env.state_schema),
                "evaluator_state_dim": int(env.state_schema["state_dim"]),
                "evaluator_mask_dim": int(env.state_schema["mask_dim"]),
                "policy_action_dim": normalized["policy_action_dim"],
                "evaluator_action_dim": int(actions.shape[-1]),
                "evaluator_task_config_sha256": task_config_sha256(
                    normalized["task"]
                ),
                "environment_instance_id": (
                    f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
                ),
                "episode_id": request_value.episode_id,
                "reset_request_sha256": hashlib.sha256(
                    canonical_json(reset_record).encode()
                ).hexdigest(),
                "observation_sha256": _tensor_sha256(states),
                "action_sha256": _tensor_sha256(actions),
                "load_success": True,
                "reset_success": True,
                "inference_success": True,
                "step_success": True,
                "finite_observation": True,
                "finite_action": True,
                "finite_reward": True,
            }
        finally:
            env.close()
        output = {
            "schema_version": COMPATIBILITY_PROBE_SCHEMA,
            "request_id": normalized["request_id"],
            "policy_benchmark_commit": normalized["policy_benchmark_commit"],
            "probe": probe,
        }
        output["payload_sha256"] = _payload_sha256(output)
        _write_json(staging / "probe.json", output)
        (staging / "SHA256SUMS").write_text(
            _tree_sha256sums(staging), encoding="utf-8"
        )
        return {
            "request_id": normalized["request_id"],
            "probe_sha256": _sha256(staging / "probe.json"),
        }

    return _atomic_output_root(output_root, build)


def _validate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "job_id",
        "task",
        "backend_id",
        "evaluator_identity",
        "split",
        "test_exposure",
        "manifest_seed",
        "manifest_episode_index",
        "replay_count",
        "image_size",
        "contract_template",
        "expected_output_root",
        "lane",
    }
    if set(job) != expected or job.get("schema_version") != CALIBRATION_JOB_SCHEMA:
        raise LaunchGateError("calibration job schema mismatch")
    if (
        job.get("backend_id") != BACKEND_ID
        or job.get("split") != "validation"
        or job.get("test_exposure") != {"test_id": False, "test_ood": False}
        or job.get("manifest_episode_index") != 0
        or job.get("replay_count") != 3
    ):
        raise LaunchGateError("calibration job changed the frozen scientific boundary")
    identity = job.get("evaluator_identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "evaluator_rlinf_commit",
        "evaluator_benchmark_commit",
        "backend_id",
    }:
        raise LaunchGateError("calibration evaluator identity is invalid")
    _require_commit(identity["evaluator_rlinf_commit"], "calibration RLinf commit")
    _require_commit(identity["evaluator_benchmark_commit"], "calibration benchmark commit")
    return dict(job)


def _planner_action_tape(env: Any, request: Any, task: str) -> np.ndarray:
    from se3_wam.benchmark.teacher_factory import make_privileged_teacher

    observation = _reset_rollout_on_fresh_env(vector_env=env, request=request)
    teacher, _ = make_privileged_teacher(task, request=request)
    if hasattr(teacher, "reset"):
        teacher.reset()
    actions = []
    terminated = truncated = False
    while not (terminated or truncated):
        values = teacher.act(observation).values
        env_actions, recorded = _planner_action_values(values)
        _, _, terminated_value, truncated_value, infos = env.step(
            env_actions, auto_reset=False
        )
        terminated = bool(terminated_value[0])
        truncated = bool(truncated_value[0])
        actions.append(recorded)
        observation = env._raw_observations[0]
        if observation is None:
            raise LaunchGateError("planner calibration lost its raw observation")
        if len(actions) > int(env.horizon_steps):
            raise LaunchGateError("planner calibration exceeded the horizon")
    if not bool(infos["success"][0]):
        raise LaunchGateError("frozen planner calibration reset did not succeed")
    return np.ascontiguousarray(np.stack(actions), dtype=np.float32)


def _replay_planner_actions(
    *,
    env: Any,
    request: Any,
    task: str,
    actions: np.ndarray,
    replay_index: int,
    reset_request_sha256: str,
    action_sha256: str,
) -> dict[str, Any]:
    import torch
    from se3_wam.benchmark.metrics import (
        completion_time_from_events,
        hierarchical_task_completion,
        validate_stage_event_order,
    )

    _reset_rollout_on_fresh_env(vector_env=env, request=request)
    infos: Mapping[str, Any] | None = None
    for index, action in enumerate(actions):
        env_actions = torch.as_tensor(action[None], dtype=torch.float32)
        _, _, terminated, truncated, result_infos = env.step(
            env_actions, auto_reset=False
        )
        infos = result_infos
        ended = bool(terminated[0]) or bool(truncated[0])
        if ended != (index == len(actions) - 1):
            raise LaunchGateError("planner calibration discrete termination drifted")
    if infos is None:
        raise LaunchGateError("planner calibration action tape is empty")
    raw_env = env.envs[0]
    events = tuple(raw_env._ledger.events)
    task_spec = env._get_task_spec(task)
    completed = validate_stage_event_order(task_spec, events)
    active_progress = float(infos["reward_inputs"]["active_stage_progress"][0])
    completion = hierarchical_task_completion(task_spec, completed, active_progress)
    success = bool(infos["success"][0])
    termination_reason = str(infos["termination_reason"][0])
    completion_time = (
        completion_time_from_events(
            events,
            start_event=task_spec.task_start_event,
            success_event=task_spec.success_stages[-1],
        )
        if success
        else None
    )
    task_quality = infos["task_quality"][0]
    safety_failure = termination_reason in set(env.reward_schema["safety_failures"])
    if (
        not success
        or safety_failure
        or completion_time is None
        or not isinstance(task_quality, Mapping)
    ):
        raise LaunchGateError("planner calibration replay was not successful and safe")
    finite = bool(
        np.isfinite(actions).all()
        and np.max(np.abs(actions)) <= 1.0
        and math.isfinite(float(completion))
        and math.isfinite(float(completion_time))
    )
    if not finite:
        raise LaunchGateError("planner calibration replay was not finite and bounded")
    return {
        "replay_index": replay_index,
        "environment_instance_id": (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        ),
        "episode_id": request.episode_id,
        "reset_request_sha256": reset_request_sha256,
        "action_sha256": action_sha256,
        "success": True,
        "safety_failure": False,
        "finite_and_bounded": True,
        "termination_reason": termination_reason,
        "trajectory_completion": float(completion),
        "completion_time_s": float(completion_time),
        "control_steps": int(len(actions)),
        "action_l2_sum": float(np.square(actions.astype(np.float64)).sum()),
        "task_quality": copy_json(task_quality),
    }


def copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip a mapping through canonical JSON to reject non-JSON values."""

    return json.loads(_canonical_json(value))


def collect_planner_calibration(
    job: Mapping[str, Any],
    contract_template: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Replay one frozen planner action tape in three fresh environments."""

    normalized = _validate_job(job)

    def build(staging: Path) -> dict[str, Any]:
        from se3_wam.benchmark.contracts import canonical_json
        from se3_wam.benchmark.evaluation import manifest_record

        from rlinf.envs.dynamic_benchmark.dynamic_benchmark_env import (
            DynamicBenchmarkEnv,
        )

        env_config = _environment_config(
            task=normalized["task"],
            split=normalized["split"],
            manifest_seed=normalized["manifest_seed"],
            image_size=normalized["image_size"],
            policy=None,
        )

        def make_env() -> Any:
            return DynamicBenchmarkEnv(
                cfg=env_config,
                num_envs=1,
                seed_offset=0,
                total_num_processes=1,
                worker_info=None,
            )

        seed_env = make_env()
        try:
            manifest_row = seed_env._manifest_rows[normalized["manifest_episode_index"]]
            request_value = manifest_row.request
            actions = _planner_action_tape(
                seed_env, request_value, normalized["task"]
            )
        finally:
            seed_env.close()
        reset_line = canonical_json(manifest_record(manifest_row)) + "\n"
        reset_manifest_sha256 = hashlib.sha256(reset_line.encode()).hexdigest()
        reset_request_sha256 = hashlib.sha256(
            canonical_json(manifest_record(manifest_row)).encode()
        ).hexdigest()
        action_sha256 = hashlib.sha256(actions.tobytes()).hexdigest()
        replays = []
        for replay_index in range(normalized["replay_count"]):
            env = make_env()
            try:
                replays.append(
                    _replay_planner_actions(
                        env=env,
                        request=request_value,
                        task=normalized["task"],
                        actions=actions,
                        replay_index=replay_index,
                        reset_request_sha256=reset_request_sha256,
                        action_sha256=action_sha256,
                    )
                )
            finally:
                env.close()
        raw_input = {
            "task": normalized["task"],
            "backend_id": normalized["backend_id"],
            "evaluator_identity": normalized["evaluator_identity"],
            "split": normalized["split"],
            "test_exposure": normalized["test_exposure"],
            "reset_manifest_sha256": reset_manifest_sha256,
            "replays": replays,
        }
        evidence_reference = str((output_root / "calibration_evidence.json").resolve())
        evidence, contract = build_calibration_evidence(
            raw_input,
            contract_template=contract_template,
            evidence_reference=evidence_reference,
        )
        (staging / "reset_manifest.jsonl").write_text(reset_line, encoding="utf-8")
        np.save(staging / "planner_actions.npy", actions, allow_pickle=False)
        _write_json(staging / "calibration_input.json", raw_input)
        _write_json(staging / "calibration_evidence.json", evidence)
        _write_json(staging / "planner_contract.json", contract)
        summary = {
            "schema_version": CALIBRATION_OUTPUT_SCHEMA,
            "task": normalized["task"],
            "job_id": normalized["job_id"],
            "replay_count": len(replays),
            "reset_manifest_sha256": reset_manifest_sha256,
            "action_sha256": action_sha256,
            "calibration_evidence_sha256": _sha256(
                staging / "calibration_evidence.json"
            ),
            "planner_contract_sha256": _sha256(staging / "planner_contract.json"),
        }
        summary["payload_sha256"] = _payload_sha256(summary)
        _write_json(staging / "calibration_output.json", summary)
        (staging / "SHA256SUMS").write_text(
            _tree_sha256sums(staging), encoding="utf-8"
        )
        return summary

    return _atomic_output_root(output_root, build)


def _checkpoint_requests(package_root: Path) -> dict[str, dict[str, Any]]:
    path = package_root / "compatibility" / "checkpoint_requests.jsonl"
    result = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise LaunchGateError(f"invalid checkpoint request row {index}") from error
        normalized = _validate_request(row)
        request_id = normalized["request_id"]
        if request_id in result:
            raise LaunchGateError(f"duplicate checkpoint request {request_id}")
        result[request_id] = normalized
    return result


def run_lane(package_root: Path, lane: str, output_root: Path) -> dict[str, Any]:
    """Run all frozen planner and compatibility jobs assigned to one lane."""

    package_root = package_root.resolve()
    validate_package(package_root)
    package = _load_json(package_root / "launch_package.json", "launch package")
    if lane not in package["lanes"]:
        raise LaunchGateError(f"lane {lane} is not authorized by the launch package")
    _git_identity(
        Path(package["rlinf_source"]["path"]),
        package["rlinf_source"]["commit"],
        "RLinf",
    )
    _git_identity(
        Path(package["se3_source"]["path"]),
        package["se3_source"]["commit"],
        "SE3-WAM",
    )
    plan = _load_json(package_root / "lanes" / f"{lane}.json", "lane plan")
    if plan.get("schema_version") != LANE_PLAN_SCHEMA or plan.get("lane") != lane:
        raise LaunchGateError("lane plan identity mismatch")
    requests = _checkpoint_requests(package_root)
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise LaunchGateError(f"lane output root must be an empty directory: {output_root}")
    else:
        output_root.mkdir(parents=True)
    marker = {
        "schema_version": "rlinf-dynamic-benchmark-rld2-lane-running-v0.1",
        "lane": lane,
        "started_unix_s": time.time(),
        "pid": os.getpid(),
    }
    _write_json(output_root / "RUNNING.json", marker)
    calibration_results = []
    compatibility_results = []
    for relative in plan["calibration_jobs"]:
        job = _load_json(package_root / relative, "calibration job")
        template = _load_json(
            package_root / job["contract_template"], "planner contract template"
        )
        child = output_root / job["expected_output_root"]
        calibration_results.append(
            collect_planner_calibration(job, template, child)
        )
    for request_id in plan["compatibility_request_ids"]:
        request = requests.get(request_id)
        if request is None or request["lane"] != lane:
            raise LaunchGateError(f"lane plan references invalid request {request_id}")
        child = output_root / Path(request["expected_output"]).parent
        compatibility_results.append(
            collect_compatibility_probe(request, child, device_name="cuda")
        )
    (output_root / "RUNNING.json").unlink()
    result = {
        "schema_version": LANE_RESULT_SCHEMA,
        "release_id": "RLD2",
        "lane": lane,
        "expected_gpu_uuid": plan["expected_gpu_uuid"],
        "calibration_job_count": len(calibration_results),
        "compatibility_probe_count": len(compatibility_results),
        "calibration_results": calibration_results,
        "compatibility_results": compatibility_results,
        "finished_unix_s": time.time(),
    }
    result["payload_sha256"] = _payload_sha256(result)
    _write_json(output_root / "lane_result.json", result)
    (output_root / "SHA256SUMS").write_text(
        _tree_sha256sums(output_root), encoding="utf-8"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lane = subparsers.add_parser("lane")
    lane.add_argument("--package-root", type=Path, required=True)
    lane.add_argument("--lane", required=True)
    lane.add_argument("--output-root", type=Path, required=True)
    compatibility = subparsers.add_parser("compatibility")
    compatibility.add_argument("--request", type=Path, required=True)
    compatibility.add_argument("--output-root", type=Path, required=True)
    compatibility.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    calibration = subparsers.add_parser("calibration")
    calibration.add_argument("--job", type=Path, required=True)
    calibration.add_argument("--contract-template", type=Path, required=True)
    calibration.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "lane":
        result = run_lane(args.package_root, args.lane, args.output_root)
    elif args.command == "compatibility":
        result = collect_compatibility_probe(
            _load_json(args.request, "compatibility request"),
            args.output_root,
            device_name=args.device,
        )
    else:
        result = collect_planner_calibration(
            _load_json(args.job, "calibration job"),
            _load_json(args.contract_template, "planner contract template"),
            args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
