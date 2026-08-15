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

"""Extension boundary for environments distributed outside RLinf."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any

ENV_PLUGIN_GROUP = "rlinf.envs"


@dataclass(frozen=True)
class EnvPlugin:
    """Describe one externally distributed RLinf environment.

    Args:
        env_cls: Environment class constructed by RLinf workers.
        prepare_actions: Optional adapter with the same keyword arguments as
            :func:`rlinf.envs.action_utils.prepare_actions`.
        validate_config: Optional validator receiving the full RLinf config,
            model config, and the matching train/eval environment configs.
    """

    env_cls: type
    prepare_actions: Callable[..., Any] | None = None
    validate_config: Callable[[Any, Any, tuple[Any, ...]], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.env_cls, type):
            raise TypeError("env plugin env_cls must be a class")
        if self.prepare_actions is not None and not callable(self.prepare_actions):
            raise TypeError("env plugin prepare_actions must be callable")
        if self.validate_config is not None and not callable(self.validate_config):
            raise TypeError("env plugin validate_config must be callable")


def _matching_entry_points(name: str) -> tuple[Any, ...]:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=ENV_PLUGIN_GROUP, name=name))
    return tuple(
        entry_point
        for entry_point in discovered.get(ENV_PLUGIN_GROUP, ())
        if entry_point.name == name
    )


@lru_cache(maxsize=None)
def get_env_plugin(name: str) -> EnvPlugin:
    """Load one environment plugin by its entry-point name.

    Args:
        name: Value used by ``env.train.env_type`` or ``env.eval.env_type``.

    Returns:
        The uniquely registered plugin.

    Raises:
        ValueError: If the name is empty, missing, or registered more than once.
        TypeError: If the entry point does not produce :class:`EnvPlugin`.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("external environment type must be a non-empty string")
    matches = _matching_entry_points(name)
    if not matches:
        raise ValueError(
            f"environment type {name!r} is not built in or installed as a plugin"
        )
    if len(matches) != 1:
        providers = sorted(f"{item.value} ({item.dist})" for item in matches)
        raise ValueError(
            f"environment type {name!r} has multiple plugin providers: {providers}"
        )
    factory = matches[0].load()
    plugin = factory() if callable(factory) else factory
    if not isinstance(plugin, EnvPlugin):
        raise TypeError(
            f"environment plugin {name!r} must return rlinf.envs.plugins.EnvPlugin"
        )
    return plugin


def resolve_builtin_env_type(name: str, supported_type: type) -> Any | None:
    """Resolve a built-in enum value or verify that an external plugin exists."""

    try:
        return supported_type(name)
    except ValueError:
        get_env_plugin(name)
        return None


__all__ = [
    "ENV_PLUGIN_GROUP",
    "EnvPlugin",
    "get_env_plugin",
    "resolve_builtin_env_type",
]
