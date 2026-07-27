# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys
from pathlib import Path

import pytest

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
import sys

sys.path.insert(0, {_ASSET_TEST_DIR.as_posix()!r})
real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "isaaclab_physx" or name.startswith("isaaclab_physx."):
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
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


@pytest.mark.parametrize("module_name", _IFACE_UTIL_MODULES)
def test_iface_utility_skips_backends_that_fail_to_import(module_name: str) -> None:
    script = f"""
import builtins
import importlib
import sys

sys.path.insert(0, {_ASSET_TEST_DIR.as_posix()!r})
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    backend_prefixes = ("isaaclab_physx.", "isaaclab_newton.", "isaaclab_ovphysx.")
    if name == "ovphysx" or name.startswith(backend_prefixes):
        raise ModuleNotFoundError("backend dependency unavailable", name="backend_dependency")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
module = importlib.import_module({module_name!r})
assert module.BACKENDS == ["Mock"]
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
