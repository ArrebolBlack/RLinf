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

"""Resolve the explicit per-task Dynamic Benchmark expert recipe contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "rlinf-dynamic-benchmark-task-recipe-map-v1"
EXACT_TASKS = (
    "p0_grasp",
    "t1_belt",
    "t1_xyz",
    "t1_so3",
    "t1_occ",
    "t2_trans",
    "t2_se3",
    "t3_phase",
    "t3_full",
    "t4_sphere",
    "t4_sphere_tabletop",
    "t4_slider",
    "t4_can",
    "t5_replan",
)
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "rld2_qa"
    / "cpu_expert_task_recipes_v1.yaml"
)


@dataclass(frozen=True)
class ResolvedRecipe:
    task: str
    recipe: Path
    role: str
    evidence: str | None
    current_source_matched_confirmation_required: bool


def load_recipe_map(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task recipe manifest must be a YAML mapping")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("task recipe manifest schema does not match")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(EXACT_TASKS):
        missing = sorted(set(EXACT_TASKS) - set(tasks or {}))
        extra = sorted(set(tasks or {}) - set(EXACT_TASKS))
        raise ValueError(f"task recipe map is not exact-14: missing={missing}, extra={extra}")
    specialized = payload.get("specialized_tasks")
    if not isinstance(specialized, list) or len(specialized) != len(set(specialized)):
        raise ValueError("specialized_tasks must be a unique list")
    ordinary_recipe = payload.get("ordinary_recipe")
    for task, entry in tasks.items():
        if not isinstance(entry, dict):
            raise ValueError(f"recipe entry for {task} must be a mapping")
        recipe_name = entry.get("recipe")
        role = entry.get("role")
        if not isinstance(recipe_name, str) or not isinstance(role, str):
            raise ValueError(f"recipe entry for {task} is incomplete")
        recipe_path = manifest_path.parent / recipe_name
        if not recipe_path.is_file():
            raise ValueError(f"recipe for {task} does not exist: {recipe_path}")
        is_specialized = task in specialized
        if is_specialized and recipe_name == ordinary_recipe:
            raise ValueError(f"specialized task {task} silently falls back to common recipe")
        if (role == "ordinary") == is_specialized:
            raise ValueError(f"task role disagrees with specialization for {task}")
    if set(specialized) != {
        task for task, entry in tasks.items() if entry["role"] != "ordinary"
    }:
        raise ValueError("specialized_tasks disagrees with task roles")
    return payload


def resolve_recipe(task: str, manifest_path: Path = DEFAULT_MANIFEST) -> ResolvedRecipe:
    manifest_path = manifest_path.resolve()
    payload = load_recipe_map(manifest_path)
    if task not in payload["tasks"]:
        raise ValueError(f"unknown Dynamic Benchmark task: {task}")
    entry = payload["tasks"][task]
    return ResolvedRecipe(
        task=task,
        recipe=(manifest_path.parent / entry["recipe"]).resolve(),
        role=entry["role"],
        evidence=entry.get("evidence"),
        current_source_matched_confirmation_required=bool(
            entry.get("current_source_matched_confirmation_required", False)
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=EXACT_TASKS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    resolved = resolve_recipe(args.task, args.manifest)
    if args.json:
        payload = asdict(resolved)
        payload["recipe"] = str(resolved.recipe)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(resolved.recipe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

