# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for skrl wrapper environment acceptance."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from isaaclab.envs import ManagerBasedMARLEnv

from isaaclab_rl.skrl import SkrlVecEnvWrapper


def test_skrl_auto_wrapper_uses_multi_agent_adapter_for_manager_based_marl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skrl auto mode selects its multi-agent adapter for manager-based MARL environments."""
    wrapped_env = object.__new__(ManagerBasedMARLEnv)
    wrapped_env._is_closed = True
    expected = object()
    calls: list[tuple[object, str]] = []

    def wrap_env(env: object, wrapper: str) -> object:
        calls.append((env, wrapper))
        return expected

    skrl_module = ModuleType("skrl")
    envs_module = ModuleType("skrl.envs")
    wrappers_module = ModuleType("skrl.envs.wrappers")
    torch_module = ModuleType("skrl.envs.wrappers.torch")
    torch_module.wrap_env = wrap_env
    skrl_module.envs = envs_module
    envs_module.wrappers = wrappers_module
    wrappers_module.torch = torch_module
    monkeypatch.setitem(sys.modules, "skrl", skrl_module)
    monkeypatch.setitem(sys.modules, "skrl.envs", envs_module)
    monkeypatch.setitem(sys.modules, "skrl.envs.wrappers", wrappers_module)
    monkeypatch.setitem(sys.modules, "skrl.envs.wrappers.torch", torch_module)

    result = SkrlVecEnvWrapper(wrapped_env, wrapper="auto")

    assert result is expected
    assert calls == [(wrapped_env, "isaaclab-multi-agent")]
