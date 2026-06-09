# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedEnv


@pytest.mark.parametrize("env_cls", [DirectRLEnv, DirectMARLEnv, ManagerBasedEnv])
def test_env_destructor_closes_before_shutdown(env_cls, monkeypatch):
    """Environment destructors should still close open envs during normal runtime."""

    closed = False

    def close(_self):
        nonlocal closed
        closed = True

    env = object.__new__(env_cls)
    env._is_closed = False
    monkeypatch.setattr(env_cls, "close", close)

    env.__del__()

    assert closed


@pytest.mark.parametrize("env_cls", [DirectRLEnv, DirectMARLEnv, ManagerBasedEnv])
def test_env_destructor_skips_close_when_already_closed(env_cls, monkeypatch):
    """Environment destructors should not re-enter close after normal cleanup."""

    def close(_self):
        raise AssertionError("close should not be called for an already closed env")

    env = object.__new__(env_cls)
    env._is_closed = True
    monkeypatch.setattr(env_cls, "close", close)

    env.__del__()


@pytest.mark.parametrize("env_cls", [DirectRLEnv, DirectMARLEnv, ManagerBasedEnv])
def test_env_destructor_skips_close_after_import_shutdown(env_cls, monkeypatch):
    """Environment destructors should not run cleanup after import machinery is torn down."""

    def close(_self):
        raise ImportError("sys.meta_path is None, Python is likely shutting down")

    env = object.__new__(env_cls)
    env._is_closed = False
    monkeypatch.setattr(env_cls, "close", close)
    monkeypatch.setattr("sys.meta_path", None)

    env.__del__()
