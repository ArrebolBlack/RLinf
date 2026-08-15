# Copyright 2026 The RLinf Authors.
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

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs import plugins as env_plugins
from rlinf.envs.plugins import EnvPlugin, get_env_plugin, resolve_builtin_env_type


class _ExternalEnv:
    pass


@dataclass
class _EntryPoint:
    name: str
    value: str
    provider: object
    dist: str = "test-distribution"

    def load(self) -> object:
        return self.provider


@pytest.fixture(autouse=True)
def _clear_plugin_cache() -> Iterator[None]:
    get_env_plugin.cache_clear()
    yield
    get_env_plugin.cache_clear()


def test_external_environment_plugin_resolves_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = EnvPlugin(env_cls=_ExternalEnv)
    entry_point = _EntryPoint("external", "tests:create_plugin", lambda: plugin)
    monkeypatch.setattr(
        env_plugins, "_matching_entry_points", lambda name: (entry_point,)
    )

    assert get_env_plugin("external") is plugin
    assert get_env_cls("external") is _ExternalEnv
    assert resolve_builtin_env_type("external", SupportedEnvType) is None


def test_builtin_environment_does_not_load_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_lookup(name: str) -> tuple[object, ...]:
        raise AssertionError(f"unexpected plugin lookup for {name}")

    monkeypatch.setattr(env_plugins, "_matching_entry_points", _unexpected_lookup)

    assert (
        resolve_builtin_env_type("maniskill", SupportedEnvType)
        is SupportedEnvType.MANISKILL
    )


def test_external_environment_requires_unique_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _EntryPoint(
        "external", "first:create", lambda: EnvPlugin(_ExternalEnv), "one"
    )
    second = _EntryPoint(
        "external", "second:create", lambda: EnvPlugin(_ExternalEnv), "two"
    )
    monkeypatch.setattr(
        env_plugins, "_matching_entry_points", lambda name: (first, second)
    )

    with pytest.raises(ValueError, match="multiple plugin providers"):
        get_env_plugin("external")


def test_environment_plugin_validates_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_point = _EntryPoint("external", "tests:invalid", lambda: object())
    monkeypatch.setattr(
        env_plugins, "_matching_entry_points", lambda name: (entry_point,)
    )

    with pytest.raises(TypeError, match="must return"):
        get_env_plugin("external")
