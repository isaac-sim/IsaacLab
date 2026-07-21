# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton/Warp import path preparation."""

from __future__ import annotations

import builtins
import importlib
import sys
import types

import pytest

pytestmark = pytest.mark.unit


def test_prepare_warp_imports_sanitizes_before_loading_warp_fem(monkeypatch):
    calls = []
    warp_module = types.ModuleType("warp")
    warp_module.__path__ = ["/isaac-sim/extscache/omni.warp.core-1.13.0+lx64/warp", "/pip/site-packages/warp"]
    warp_fem_module = types.ModuleType("warp.fem")

    def fake_sanitize():
        calls.append("sanitize")
        warp_module.__path__ = ["/pip/site-packages/warp", "/isaac-sim/extscache/omni.warp.core-1.13.0+lx64/warp"]

    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_module._deprioritize_prebundle_paths = fake_sanitize

    def fake_import_module(name, package=None):
        if name == "isaaclab":
            return isaaclab_module
        if name == "warp":
            return warp_module
        if name == "warp.fem":
            assert calls == ["sanitize", "sanitize"]
            assert warp_module.__path__[0] == "/pip/site-packages/warp"
            return warp_fem_module
        raise AssertionError(f"unexpected import: {name}")

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in {"isaaclab", "warp"}:
            return fake_import_module(name)
        if level == 0 and name == "warp.fem":
            return fake_import_module(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "warp", warp_module)

    from isaaclab_newton import _prepare_warp_imports

    calls.clear()
    warp_module.__path__ = ["/isaac-sim/extscache/omni.warp.core-1.13.0+lx64/warp", "/pip/site-packages/warp"]
    _prepare_warp_imports()

    assert calls == ["sanitize", "sanitize"]
