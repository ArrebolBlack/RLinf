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

"""Post-reset hidden-event setup for Dynamic Benchmark T5 tasks."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

__all__ = ["arm_hidden_t5_event", "t5_branch_for_episode"]


def t5_branch_for_episode(episode_id: str) -> str:
    """Choose a reproducible hidden T5 branch without adding it to reset state."""

    if not episode_id or episode_id.strip() != episode_id:
        raise ValueError("T5 episode_id must be a non-empty trimmed string")
    digest = hashlib.sha256(
        f"rlinf-dynamic-benchmark-t5-event-v1:{episode_id}".encode("utf-8")
    ).digest()
    return "left" if digest[0] & 1 == 0 else "right"


def arm_hidden_t5_event(
    *,
    task_id: str,
    split_name: str,
    env: Any,
    request: Any,
    load_task_config: Callable[[str], Any],
    event_tape_type: Callable[..., Any],
) -> str | None:
    """Arm the benchmark-owned T5 tape after reset and before the first step."""

    if task_id not in {"t5_commit", "t5_replan"}:
        return None
    episode_id = str(request.episode_id)
    branch = t5_branch_for_episode(episode_id)
    config = load_task_config(task_id)
    tape = event_tape_type(
        event_id=f"rlinf-{split_name}-{episode_id}-{branch}",
        branch=branch,
        trigger_gate_y_m=float(config["event"]["default_trigger_gate_y_m"]),
    )
    return str(env.arm_event_tape(tape))
