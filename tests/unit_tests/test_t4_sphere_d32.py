# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only contracts for the T4-sphere valid-D32 repair path."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rlinf.envs.dynamic_benchmark.t4_sphere_d32 import (
    BACKEND_ID,
    CANDIDATE_SCHEMA_VERSION,
    D32_EPISODES,
    D32_MANIFEST_SCHEMA_VERSION,
    D32_MANIFEST_SEED,
    D32_ROW_REPORT_SCHEMA_VERSION,
    D32_SEED_SET_SHA256,
    EXECUTION_CONTRACT,
    TASK_ID,
    aggregate_d32_reports,
    assert_seed_disjointness,
    canonical_json_bytes,
    load_candidate_identity,
    load_d32_manifest,
    teacher_reset_identity,
    validate_blocking_replay,
    validate_request_identity,
    validate_terminal_row,
)

D32_ROWS = (
    (848433872, -0.009714751652178853, 5.0),
    (497472235, -0.01340090265266412, 10.0),
    (522737087, -0.001818634506392464, 15.0),
    (1207344428, -0.0020482472816289975, 5.0),
    (868159464, 0.01044170170331964, 10.0),
    (990513966, 0.010992853732527653, 15.0),
    (1665180122, 0.001824812526173527, 5.0),
    (965082094, -0.013641515803699152, 10.0),
    (1532949651, -0.010519087516177184, 15.0),
    (1186691700, 0.002909318888014909, 5.0),
    (1218777721, -0.010847092620979597, 10.0),
    (1613506544, -0.011399608107109691, 15.0),
    (441364504, -0.01172937008116884, 5.0),
    (395322753, 0.013394984561892618, 10.0),
    (1666590940, 0.005741136353194724, 15.0),
    (1761371836, -0.008106722173057634, 5.0),
    (442132774, -0.012816276604119735, 10.0),
    (1089833358, -0.0066116816566162695, 15.0),
    (796706585, 0.0027905955275010393, 5.0),
    (1219715148, 0.014031722629591362, 10.0),
    (677246929, 0.000957464533104805, 15.0),
    (459550639, -0.008917721294458215, 5.0),
    (357549639, 0.00860738063912279, 10.0),
    (304351216, 0.012877089357796124, 15.0),
    (960001216, -0.008711123194804864, 5.0),
    (1203041089, -0.004055427720086985, 10.0),
    (470864539, 0.009078184741578236, 15.0),
    (1446977539, 0.0013144933737002165, 5.0),
    (678577059, -0.012782385248672527, 10.0),
    (895218846, 0.006000745327212925, 15.0),
    (304453839, 0.005515149471607264, 5.0),
    (724462916, 0.005916236504654376, 10.0),
)


def _sealed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _candidate_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": "GPUPLAN0/t4-sphere-valid-d32-v2-test",
        "task_id": TASK_ID,
        "backend_id": BACKEND_ID,
        "repositories": {
            "se3_wam": {
                "commit": "1" * 40,
                "tree": "2" * 40,
                "mujoco_warp_gitlink": "3" * 40,
            },
            "mujoco_warp": {"commit": "3" * 40, "tree": "4" * 40},
            "rlinf": {"commit": "5" * 40, "tree": "6" * 40},
            "dynamic_benchmark": {
                "commit": "7" * 40,
                "tree": "8" * 40,
                "rlinf_gitlink": "5" * 40,
            },
        },
        "task_config_sha256": "9" * 64,
        "execution": dict(EXECUTION_CONTRACT),
    }
    return _sealed(payload, "candidate_sha256")


def _write_candidate(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(_candidate_payload() if payload is None else payload),
        encoding="utf-8",
    )
    return path


def _manifest_payload(candidate_sha256: str) -> dict[str, Any]:
    rows = []
    for index, (seed, lateral, angle) in enumerate(D32_ROWS):
        rows.append(
            {
                "action_mode": "E7",
                "api_version": "benchmark-api-v0.1",
                "candidate_index": index,
                "episode_id": f"d0-test_id-t4_sphere-{index:05d}-s{seed}",
                "export_dir": f"exports/row{index:04d}",
                "export_sha256": f"{index + 1:064x}",
                "factors": {
                    "lateral_offset_m": lateral,
                    "ramp_angle_deg": angle,
                    "surface_friction": 0.8,
                },
                "object_mode": "sphere",
                "observation_track": "state",
                "pair_id": None,
                "pair_member_id": None,
                "request_json_sha256": f"{index + 101:064x}",
                "reset_mode": "default",
                "seed": seed,
                "source_group_id": f"d0-t4_sphere-test_id-id-{index:05d}",
                "split": "test_id",
                "task_id": TASK_ID,
                "teacher_reset_identity": {
                    "effective_capture_plane_world_y_m": None,
                    "effective_close_lead_s": 0.11,
                    "effective_staging_eef_position_m": [
                        0.06,
                        0.04 if angle == 5.0 else 0.16,
                        0.961,
                    ],
                },
            }
        )
    payload: dict[str, Any] = {
        "schema_version": D32_MANIFEST_SCHEMA_VERSION,
        "phase_id": "GPUPLAN0/t4-sphere-valid-d32-v2",
        "task_id": TASK_ID,
        "backend_id": BACKEND_ID,
        "split": "test_id",
        "manifest_seed": D32_MANIFEST_SEED,
        "dataset_manifest_version": "db-dataset-manifest-v0.1",
        "episode_count": D32_EPISODES,
        "row_execution_batch_size": 1,
        "candidate_indices": list(range(D32_EPISODES)),
        "episode_ids": [row["episode_id"] for row in rows],
        "seed_set_sha256": D32_SEED_SET_SHA256,
        "candidate_sha256": candidate_sha256,
        "task_config_sha256": "9" * 64,
        "observation_track": "state",
        "action_mode": "E7",
        "rows": rows,
    }
    return _sealed(payload, "manifest_sha256")


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_candidate_identity_requires_matching_gitlinks(tmp_path: Path) -> None:
    candidate = load_candidate_identity(_write_candidate(tmp_path))
    assert candidate.repositories["se3_wam"]["mujoco_warp_gitlink"] == "3" * 40

    payload = _candidate_payload()
    payload["repositories"]["dynamic_benchmark"]["rlinf_gitlink"] = "a" * 40
    _sealed(payload, "candidate_sha256")
    with pytest.raises(ValueError, match="RLinf gitlink"):
        load_candidate_identity(_write_candidate(tmp_path, payload))


def test_manifest_validates_all_rows_before_selecting_b1(tmp_path: Path) -> None:
    candidate = load_candidate_identity(_write_candidate(tmp_path))
    manifest = load_d32_manifest(
        _write_manifest(tmp_path, _manifest_payload(candidate.candidate_sha256)),
        candidate=candidate,
        verify_exports=False,
    )

    assert len(manifest.rows) == 32
    assert manifest.row(0)["factors"] != manifest.row(1)["factors"]
    assert (
        manifest.row(0)["teacher_reset_identity"]
        != manifest.row(1)["teacher_reset_identity"]
    )
    assert manifest.row(31)["seed"] == 724462916


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["rows"][1].update(
                factors=payload["rows"][0]["factors"]
            ),
            "all reuse",
        ),
        (
            lambda payload: payload["rows"][7].update(
                episode_id=payload["rows"][0]["episode_id"]
            ),
            "not globally unique",
        ),
        (
            lambda payload: payload["rows"][2].update(observation_track="hybrid"),
            "observation_track",
        ),
        (lambda payload: payload["rows"][3].update(action_mode="E6"), "action_mode"),
    ],
)
def test_manifest_fails_closed_on_row_identity_drift(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    candidate = load_candidate_identity(_write_candidate(tmp_path))
    payload = _manifest_payload(candidate.candidate_sha256)
    if "all reuse" in message:
        factors = copy.deepcopy(payload["rows"][0]["factors"])
        for row in payload["rows"]:
            row["factors"] = copy.deepcopy(factors)
    elif "not globally unique" in message:
        mutation(payload)
        payload["episode_ids"] = [row["episode_id"] for row in payload["rows"]]
    else:
        mutation(payload)
    _sealed(payload, "manifest_sha256")
    with pytest.raises(ValueError, match=message):
        load_d32_manifest(
            _write_manifest(tmp_path, payload),
            candidate=candidate,
            verify_exports=False,
        )


@dataclass(frozen=True)
class _Enum:
    value: str


def _request(row: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        action_mode=_Enum(row["action_mode"]),
        api_version=row["api_version"],
        episode_id=row["episode_id"],
        factors=dict(row["factors"]),
        object_mode=row["object_mode"],
        observation_track=_Enum(row["observation_track"]),
        reset_mode=row["reset_mode"],
        seed=row["seed"],
        split=_Enum(row["split"]),
        task_id=row["task_id"],
    )


def test_request_identity_checks_every_reset_field(tmp_path: Path) -> None:
    candidate = load_candidate_identity(_write_candidate(tmp_path))
    row = _manifest_payload(candidate.candidate_sha256)["rows"][1]
    actual = validate_request_identity(
        _request(row),
        row,
        actual_task_config_sha256=candidate.task_config_sha256,
        candidate=candidate,
    )
    assert actual["factors"]["ramp_angle_deg"] == 10.0

    bad = _request(row)
    bad.factors = dict(bad.factors, ramp_angle_deg=5.0)
    with pytest.raises(RuntimeError, match="request identity drift"):
        validate_request_identity(
            bad,
            row,
            actual_task_config_sha256=candidate.task_config_sha256,
            candidate=candidate,
        )


def test_manifest_rejects_row_zero_teacher_reset_reuse(tmp_path: Path) -> None:
    candidate = load_candidate_identity(_write_candidate(tmp_path))
    payload = _manifest_payload(candidate.candidate_sha256)
    shared = copy.deepcopy(payload["rows"][0]["teacher_reset_identity"])
    for row in payload["rows"]:
        row["teacher_reset_identity"] = copy.deepcopy(shared)
    _sealed(payload, "manifest_sha256")

    with pytest.raises(ValueError, match="teacher reset identity"):
        load_d32_manifest(
            _write_manifest(tmp_path, payload),
            candidate=candidate,
            verify_exports=False,
        )


def test_teacher_reset_identity_freezes_all_request_sensitive_parameters() -> None:
    identity = teacher_reset_identity(
        {
            "effective_capture_plane_world_y_m": -0.12,
            "effective_close_lead_s": 0.15,
            "effective_staging_eef_position_m": (0.06, 0.1, 0.961),
            "unrelated": "ignored",
        }
    )
    assert identity == {
        "effective_capture_plane_world_y_m": -0.12,
        "effective_close_lead_s": 0.15,
        "effective_staging_eef_position_m": [0.06, 0.1, 0.961],
    }


@dataclass(frozen=True)
class _Event:
    name: str
    physics_step: int
    time_s: float


def _terminal(episode_id: str, *, physics_step: int = 49) -> SimpleNamespace:
    return SimpleNamespace(
        completion=0.5,
        control_step=2,
        episode_id=episode_id,
        events=(
            _Event("release_gate_opens", 0, 0.0),
            _Event("downstream_workspace_exit", physics_step, physics_step / 500.0),
        ),
        lane=0,
        outcome=_Enum("failure"),
        physics_step=physics_step,
        policy_step=2,
        success=False,
        task_id=TASK_ID,
        task_quality=None,
        terminated=True,
        termination_reason="downstream_workspace_exit",
        truncated=False,
    )


def test_terminal_ledger_and_replay_are_blocking() -> None:
    terminal = _terminal("ep-0")
    primary = validate_terminal_row(terminal, episode_id="ep-0")
    replay = validate_blocking_replay(
        primary_terminal=primary,
        replay_terminal=terminal,
        primary_observation_fingerprints=("a", "b", "c"),
        replay_observation_fingerprints=("a", "b", "c"),
    )
    assert replay == {
        "blocking": True,
        "observation_fingerprint_count": 3,
        "observation_fingerprints_match": True,
        "passed": True,
        "terminal_ledger_match": True,
    }

    with pytest.raises(RuntimeError, match="physical event clock"):
        validate_terminal_row(_terminal("ep-0", physics_step=75), episode_id="ep-0")
    with pytest.raises(RuntimeError, match="fingerprints differ"):
        validate_blocking_replay(
            primary_terminal=primary,
            replay_terminal=terminal,
            primary_observation_fingerprints=("a", "b"),
            replay_observation_fingerprints=("a", "x"),
        )


def test_historical_mc64_seed_non_overlap_is_a_read_only_check(tmp_path: Path) -> None:
    candidate = load_candidate_identity(_write_candidate(tmp_path))
    d32 = _manifest_payload(candidate.candidate_sha256)["rows"]
    other = {"rows": [{"seed": 2_000_000_000 + index} for index in range(64)]}
    receipt = assert_seed_disjointness(d32, other, other_name="blocked_mc64")
    assert receipt["d32_rows"] == 32
    assert receipt["other_rows"] == 64
    assert receipt["seed_intersection"] == []


def _row_report(manifest: Any, candidate: Any, index: int) -> dict[str, Any]:
    row = manifest.row(index)
    return {
        "schema_version": D32_ROW_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "phase": "d32",
        "task_id": TASK_ID,
        "batch_size": 1,
        "sample_count": 1,
        "counts_as_d32_result": True,
        "closed_loop_planner": True,
        "frozen_action_replay": False,
        "execution": dict(EXECUTION_CONTRACT),
        "identity": {
            "candidate_index": index,
            "candidate_sha256": candidate.candidate_sha256,
            "episode_id": row["episode_id"],
            "seed": row["seed"],
            "manifest_sha256": manifest.manifest_sha256,
            "task_config_sha256": candidate.task_config_sha256,
        },
        "request": {
            name: row[name]
            for name in (
                "action_mode",
                "api_version",
                "episode_id",
                "factors",
                "object_mode",
                "observation_track",
                "reset_mode",
                "seed",
                "split",
                "task_id",
            )
        },
        "teacher": {
            "full_reset_request_bound": True,
            "reset_identity": row["teacher_reset_identity"],
        },
        "resource": {
            "gpu_created": True,
            "expected_device_uuid": "GPU-test",
            "observed_device_uuid": "GPU-test",
        },
        "terminal_ledger": {
            "blocking": True,
            "passed": True,
            "exact_once_second_consumption_rejected": True,
        },
        "replay": {
            "blocking": True,
            "passed": True,
            "exact_once_second_consumption_rejected": True,
        },
        "terminal": {
            "episode_id": row["episode_id"],
            "success": index % 2 == 0,
            "termination_reason": "success" if index % 2 == 0 else "timeout",
            "task_quality": None,
        },
    }


def test_aggregate_requires_complete_32_of_32_blocking_receipts(tmp_path: Path) -> None:
    candidate = load_candidate_identity(_write_candidate(tmp_path))
    manifest = load_d32_manifest(
        _write_manifest(tmp_path, _manifest_payload(candidate.candidate_sha256)),
        candidate=candidate,
        verify_exports=False,
    )
    reports = [_row_report(manifest, candidate, index) for index in range(32)]
    aggregate = aggregate_d32_reports(
        candidate=candidate,
        manifest=manifest,
        reports=reports,
    )
    assert aggregate["completed"] == aggregate["total"] == 32
    assert aggregate["success"] == aggregate["failure"] == 16

    with pytest.raises(RuntimeError, match="32/32"):
        aggregate_d32_reports(
            candidate=candidate,
            manifest=manifest,
            reports=reports[:-1],
        )
    reports[0]["replay"].pop("exact_once_second_consumption_rejected")
    with pytest.raises(RuntimeError, match="exact-once negative witness"):
        aggregate_d32_reports(
            candidate=candidate,
            manifest=manifest,
            reports=reports,
        )
    reports[0]["replay"] = {"blocking": False, "passed": True}
    with pytest.raises(RuntimeError, match="replay is not blocking"):
        aggregate_d32_reports(
            candidate=candidate,
            manifest=manifest,
            reports=reports,
        )
