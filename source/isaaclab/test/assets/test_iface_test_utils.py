# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys
from pathlib import Path

_ASSET_TEST_DIR = Path(__file__).parent
_IFACE_UTIL_MODULES = (
    "_articulation_iface_test_utils",
    "_rigid_object_iface_test_utils",
    "_rigid_object_collection_iface_test_utils",
)


def _run_probe(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("EXP_PATH", None)
    env.pop("LD_PRELOAD", None)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ASSET_TEST_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_iface_utilities_share_one_bootstrap_module() -> None:
    script = f"""
import importlib
import sys

sys.path.insert(0, {_ASSET_TEST_DIR.as_posix()!r})
modules = [importlib.import_module(name) for name in {_IFACE_UTIL_MODULES!r}]
boot = importlib.import_module("_iface_test_boot")
assert all(module.simulation_app is boot.simulation_app for module in modules)
"""

    result = _run_probe(script)

    assert result.returncode == 0, result.stdout + result.stderr


def test_iface_utilities_import_without_physx_package() -> None:
    script = f"""
import builtins
import importlib
import importlib.util
import sys

sys.path.insert(0, {_ASSET_TEST_DIR.as_posix()!r})
real_import = builtins.__import__
real_find_spec = importlib.util.find_spec

def blocked_import(name, *args, **kwargs):
    if name == "isaaclab_physx" or name.startswith("isaaclab_physx."):
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

def backend_free_find_spec(name, *args, **kwargs):
    if name in ("isaaclab_physx", "isaaclab_newton", "isaaclab_ovphysx", "ovphysx"):
        return None
    return real_find_spec(name, *args, **kwargs)

builtins.__import__ = blocked_import
importlib.util.find_spec = backend_free_find_spec
modules = [importlib.import_module(name) for name in {_IFACE_UTIL_MODULES!r}]
assert all(not any(backend.lower() == "physx" for backend in module.BACKENDS) for module in modules)
"""

    result = _run_probe(script)

    assert result.returncode == 0, result.stdout + result.stderr


def test_iface_utilities_use_same_mock_backend_label() -> None:
    script = f"""
import importlib
import sys

sys.path.insert(0, {_ASSET_TEST_DIR.as_posix()!r})
modules = [importlib.import_module(name) for name in {_IFACE_UTIL_MODULES!r}]
assert all(module.BACKENDS[0] == "Mock" for module in modules)
"""

    result = _run_probe(script)

    assert result.returncode == 0, result.stdout + result.stderr


def test_discoverable_backend_internal_import_error_propagates() -> None:
    script = f"""
import builtins
import importlib.util
import sys
import types

sys.path.insert(0, {_ASSET_TEST_DIR.as_posix()!r})
real_find_spec = importlib.util.find_spec
real_import = builtins.__import__

def find_spec(name, package=None):
    if name in ("isaaclab_ovphysx", "ovphysx"):
        return object()
    if name in ("isaaclab_physx", "isaaclab_newton"):
        return None
    return real_find_spec(name, package)

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "isaaclab_ovphysx.assets.articulation.articulation":
        return types.SimpleNamespace(Articulation=object)
    if name == "isaaclab_ovphysx.assets.articulation.articulation_data":
        raise ModuleNotFoundError("broken backend dependency", name="broken_backend_dependency")
    return real_import(name, globals, locals, fromlist, level)

importlib.util.find_spec = find_spec
builtins.__import__ = guarded_import

try:
    import _articulation_iface_test_utils
except ModuleNotFoundError as exc:
    if exc.name == "broken_backend_dependency":
        raise SystemExit(0)
    raise
raise SystemExit("discoverable backend internal import error was swallowed")
"""

    result = _run_probe(script)

    assert result.returncode == 0, result.stdout + result.stderr


def test_mock_factory_preserves_fixed_base_and_rejects_ordering() -> None:
    script = f"""
import sys

sys.path.insert(0, {_ASSET_TEST_DIR.as_posix()!r})
from _articulation_iface_test_utils import get_articulation

articulation, _ = get_articulation("mock", device="cpu", is_fixed_base=True)
assert articulation.is_fixed_base

try:
    get_articulation("mock", device="cpu", joint_ordering=("joint_0",))
except ValueError as exc:
    assert "does not support explicit joint or body ordering" in str(exc)
else:
    raise AssertionError("mock ordering was silently ignored")
"""

    result = _run_probe(script)

    assert result.returncode == 0, result.stdout + result.stderr
