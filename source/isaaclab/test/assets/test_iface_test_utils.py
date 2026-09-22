# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks that the interface-test factories degrade gracefully when backend packages are missing."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ASSET_TEST_DIR = Path(__file__).parent


@pytest.mark.parametrize(
    "blocked_prefixes, expected_backends",
    [
        (("isaaclab_physx",), ("newton", "ovphysx")),
        (("isaaclab_physx", "isaaclab_newton", "isaaclab_ov", "ovphysx"), ()),
    ],
    ids=["without_physx", "without_any_backend"],
)
def test_iface_utils_import_without_backend_packages(blocked_prefixes, expected_backends) -> None:
    script = f"""
import builtins
import importlib
import sys

sys.path.insert(0, {_ASSET_TEST_DIR.as_posix()!r})
real_import = builtins.__import__
blocked = {blocked_prefixes!r}

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in blocked or name.startswith(tuple(prefix + "." for prefix in blocked)):
        raise ModuleNotFoundError(name, name=name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
module = importlib.import_module("_iface_test_utils")
available = [backend for backend in {expected_backends!r} if backend not in module.BACKEND_UNAVAILABLE_REASONS]
assert module.BACKENDS == available, (module.BACKENDS, module.BACKEND_UNAVAILABLE_REASONS)
"""
    env = os.environ.copy()
    env.pop("EXP_PATH", None)
    env.pop("LD_PRELOAD", None)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=_ASSET_TEST_DIR, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
