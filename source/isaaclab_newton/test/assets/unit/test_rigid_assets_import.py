# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collection guard for the kitless Newton rigid-asset suites."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ASSET_TEST_DIR = Path(__file__).resolve().parents[1]
_TARGETS = ("test_rigid_object.py", "test_rigid_object_collection.py")


@pytest.mark.parametrize("target", _TARGETS)
def test_rigid_asset_module_collects_without_kit_isaacsim_or_nucleus(target: str, tmp_path: Path) -> None:
    """Collect each real module while rejecting Kit, IsaacSim, and Nucleus access."""
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import importlib.abc
import sys
import isaaclab.utils.assets as assets


class _ForbiddenFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        frame = sys._getframe(1)
        while frame and (
            frame.f_code.co_filename.startswith("<frozen importlib")
            or frame.f_code.co_filename == __file__
        ):
            frame = frame.f_back
        imported_by_target = frame is not None and frame.f_code.co_filename.endswith(
            ("test_rigid_object.py", "test_rigid_object_collection.py")
        )
        if fullname.startswith("isaacsim") or (
            fullname.startswith("isaaclab.app") and (imported_by_target or fullname == "isaaclab.app.app_launcher")
        ):
            raise ImportError(f"forbidden kit dependency imported: {fullname}")
        return None


class _ForbiddenNucleusPath(str):
    def _guard(self):
        caller = sys._getframe(2).f_code.co_filename
        if caller.endswith(("test_rigid_object.py", "test_rigid_object_collection.py")):
            raise RuntimeError("forbidden Nucleus asset used by Newton rigid-asset test")

    def __format__(self, format_spec):
        self._guard()
        return super().__format__(format_spec)

    def __add__(self, other):
        self._guard()
        return super().__add__(other)


sys.meta_path.insert(0, _ForbiddenFinder())
assets.ISAAC_NUCLEUS_DIR = _ForbiddenNucleusPath(assets.ISAAC_NUCLEUS_DIR)
assets.ISAACLAB_NUCLEUS_DIR = _ForbiddenNucleusPath(assets.ISAACLAB_NUCLEUS_DIR)
""",
        encoding="utf-8",
    )
    env = os.environ | {"PYTHONPATH": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(_ASSET_TEST_DIR / target), "--collect-only", "-q"],
        cwd=_ASSET_TEST_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
