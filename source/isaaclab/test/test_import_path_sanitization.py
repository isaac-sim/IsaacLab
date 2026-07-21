# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Isaac Sim prebundle/import-path sanitization."""

from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

import isaaclab
from isaaclab.utils.backend_utils import FactoryBase

pytestmark = pytest.mark.unit


def test_deprioritize_prebundle_paths_demotes_loaded_package_paths(monkeypatch):
    good_sys_path = "/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.12/site-packages"
    bad_sys_path = "/isaac-sim/extscache/omni.warp.core-1.13.0+lx64"
    good_package_path = f"{good_sys_path}/warp"
    bad_package_path = f"{bad_sys_path}/warp"

    fake_package = types.ModuleType("_fake_warp_package")
    fake_package.__path__ = [bad_package_path, good_package_path]

    monkeypatch.setattr(sys, "path", [bad_sys_path, good_sys_path])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([bad_sys_path, good_sys_path]))
    monkeypatch.setitem(sys.modules, fake_package.__name__, fake_package)

    isaaclab._deprioritize_prebundle_paths()

    assert sys.path == [good_sys_path, bad_sys_path]
    assert os.environ["PYTHONPATH"] == os.pathsep.join([good_sys_path, bad_sys_path])
    assert fake_package.__path__ == [good_package_path, bad_package_path]


def test_factory_sanitizes_paths_before_dynamic_backend_import(monkeypatch):
    calls = []
    backend_module = types.ModuleType("_fake_backend_module")

    class DummyFactoryImpl:
        pass

    backend_module.DummyFactory = DummyFactoryImpl

    class DummyFactory(FactoryBase):
        __module__ = "isaaclab.fake.factory"

        @classmethod
        def _get_backend(cls, *args, **kwargs):
            return "dummy"

        @classmethod
        def _get_module_name(cls, backend: str) -> str:
            return backend_module.__name__

    def fake_sanitize():
        calls.append("sanitize")

    def fake_import_module(name):
        assert calls == ["sanitize"]
        return backend_module

    def lazy_getattr(name):
        assert calls == ["sanitize", "sanitize"]
        return DummyFactoryImpl

    backend_module.__getattr__ = lazy_getattr

    monkeypatch.setattr(isaaclab, "_deprioritize_prebundle_paths", fake_sanitize)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    assert DummyFactory.resolve_class() is DummyFactoryImpl
