# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest

from examples.embodiment.evaluate_dynamic_benchmark_planner import _parser


def _arguments(split: str) -> list[str]:
    return [
        "--evaluator-commit",
        "a" * 40,
        "--benchmark-commit",
        "b" * 40,
        "--output",
        "planner-eval",
        "--task",
        "t1_xyz",
        "--split",
        split,
        "--manifest-seed",
        "20261150",
    ]


def test_planner_evaluator_accepts_validation_without_test_access() -> None:
    args = _parser().parse_args(_arguments("validation"))

    assert args.split == "validation"
    assert args.task == "t1_xyz"
    assert args.episodes == 20


def test_planner_evaluator_rejects_unknown_split() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(_arguments("train"))
