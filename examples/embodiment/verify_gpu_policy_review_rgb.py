#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prove that policy RGB and review pixels share one CUDA visual observation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _load_strict_contract() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/envs/dynamic_benchmark/t1_xyz_strict_evidence.py"
    )
    name = "_t1_xyz_rgb_evidence_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load review evidence contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_STRICT = _load_strict_contract()
TASK_ID = _STRICT.TASK_ID
load_frozen_manifest = _STRICT.load_frozen_manifest
preflight_export_request = _STRICT.preflight_export_request
request_identity = _STRICT.request_identity
validate_repository_tuple = _STRICT.validate_repository_tuple


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--research-source-root", type=Path, required=True)
    parser.add_argument("--se3-source-root", type=Path, required=True)
    parser.add_argument("--mjwarp-source-root", type=Path, required=True)
    parser.add_argument("--rlinf-source-root", type=Path, required=True)
    parser.add_argument("--dynamic-source-root", type=Path, required=True)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _thaw(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parser().parse_args()
    if args.image_size < 224:
        raise ValueError("--image-size must be at least 224")
    if args.device_ordinal < 0:
        raise ValueError("--device-ordinal must be nonnegative")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    manifest = load_frozen_manifest(
        args.manifest,
        expected_phase="e0",
        verify_exports=True,
    )
    row = manifest.row(args.row_index)
    request, export_identity = preflight_export_request(manifest, row)
    if request_identity(request) != dict(row["request"]):
        raise RuntimeError("preflight ResetRequest identity drifted")
    repositories = validate_repository_tuple(
        manifest,
        research_root=args.research_source_root,
        se3_root=args.se3_source_root,
        mjwarp_root=args.mjwarp_source_root,
        rlinf_root=args.rlinf_source_root,
        dynamic_root=args.dynamic_source_root,
    )
    if Path(__file__).resolve().parents[2] != args.rlinf_source_root.resolve(
        strict=True
    ):
        raise RuntimeError("RGB verifier is not executing from the sealed RLinf root")

    from rlinf.envs.dynamic_benchmark.gpu_tensor_backend import (
        GpuNativeTensorBackendEnv,
        _manifest_payload_sha256,
        _manifest_request_payload,
    )

    se3_identity = manifest.payload["repositories"]["se3_wam"]
    reset_manifest_sha256 = _manifest_payload_sha256(
        (_manifest_request_payload(request),)
    )
    backend = GpuNativeTensorBackendEnv(
        task_id=TASK_ID,
        num_envs=1,
        export_dir=str(manifest.export_dir(row)),
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_se3_source_commit=se3_identity["commit"],
        expected_se3_source_tree=se3_identity["tree"],
        device_ordinal=args.device_ordinal,
        image_size=args.image_size,
        render_observations=True,
        manifest_requests=(request,),
        manifest_sha256=reset_manifest_sha256,
        split=getattr(request.split, "value", request.split),
        observation_track="state",
    )
    try:
        executed_requests = backend.next_requests()
        if len(executed_requests) != 1:
            raise RuntimeError("RGB B=1 must execute exactly one reset request")
        executed_request = executed_requests[0]
        expected_executed_identity = request_identity(request)
        expected_executed_identity["episode_id"] = (
            f"{request.episode_id}-cycle00000000"
        )
        expected_executed_identity["observation_track"] = "state"
        if request_identity(executed_request) != expected_executed_identity:
            raise RuntimeError("manifest cursor changed the sealed reset payload")
        reset = backend.reset()
        if reset.episode_ids != (executed_request.episode_id,):
            raise RuntimeError("CUDA reset changed the sealed episode identity")
        evidence = backend.materialize_policy_review_rgb_evidence((0,))
        receipt = _thaw(evidence.receipt)
        if (
            receipt["backend_id"] != "mjwarp_gpu_v1"
            or receipt["device_platform"] != "cuda"
            or receipt["image_height"] < 224
            or receipt["image_width"] < 224
            or receipt["cpu_renderer_fallback"] is not False
            or receipt["se3_review_materialization_matches_policy_tensor"] is not True
        ):
            raise RuntimeError("policy/review RGB source receipt did not pass")

        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("RGB evidence export requires imageio") from exc
        args.output_dir.mkdir(parents=True)
        scene_path = args.output_dir / "agentview.png"
        wrist_path = args.output_dir / "robot0_eye_in_hand.png"
        combined_path = args.output_dir / "scene-wrist.png"
        scene = np.asarray(evidence.review_rgb[0]["agentview"], dtype=np.uint8)
        wrist = np.asarray(
            evidence.review_rgb[0]["robot0_eye_in_hand"], dtype=np.uint8
        )
        iio.imwrite(scene_path, scene)
        iio.imwrite(wrist_path, wrist)
        iio.imwrite(combined_path, np.concatenate((scene, wrist), axis=1))
        payload = {
            "schema_version": "gpu-policy-review-rgb-b1-evidence-v1",
            "status": "passed_same_cuda_visual_source",
            "task_id": TASK_ID,
            "manifest_index": int(row["manifest_index"]),
            "sealed_reset_request": request_identity(request),
            "reset_request": request_identity(executed_request),
            "reset_manifest_sha256": reset_manifest_sha256,
            "source_identity_sha256": manifest.source_identity_sha256,
            "repositories": repositories,
            "export": export_identity,
            "receipt": receipt,
            "policy_rgb": {
                camera: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "data_ptr": int(tensor.data_ptr()),
                }
                for camera, tensor in evidence.policy_rgb.items()
            },
            "files": {
                path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
                for path in (scene_path, wrist_path, combined_path)
            },
        }
        receipt_path = args.output_dir / "evidence.json"
        receipt_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    finally:
        backend.close()

    print(json.dumps({"status": payload["status"], "evidence": str(receipt_path)}))


if __name__ == "__main__":
    main()
