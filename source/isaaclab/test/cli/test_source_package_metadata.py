# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for source package dependency metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

pytestmark = pytest.mark.unit


def test_isaaclab_uses_one_standalone_usd_provider(source_checkout_root: Path):
    """Isaac Lab must install only the USD provider shared with its importer dependencies.

    ``usd-core`` and ``usd-exchange`` each install a complete ``pxr`` into the same directory, so
    a second provider silently overwrites the first and removing either one leaves ``pxr`` broken.
    Nothing detects that, because the two are separate distributions.
    """
    with (source_checkout_root / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    dependencies = pyproject["project"]["dependencies"]
    usd_providers = [
        dependency
        for dependency in dependencies
        if dependency.startswith("usd-core") or dependency.startswith("usd-exchange")
    ]

    assert usd_providers == ["usd-exchange==3.0.0"]


def test_resolved_environment_has_no_second_usd_provider(source_checkout_root: Path):
    """No dependency may pull ``usd-core`` back in behind an extra.

    ``newton[importers]``, ``mujoco[usd]`` and ``warp-lang[examples]`` all require it, so selecting
    any of them would reinstate the overlap that the direct dependencies avoid. Checking the lock
    catches that, where checking ``pyproject.toml`` alone would not.
    """
    with (source_checkout_root / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)

    locked = {package["name"] for package in lock["package"]}

    assert "usd-core" not in locked
    assert "usd-exchange" in locked


def test_standalone_importers_are_opt_in(source_checkout_root: Path):
    """Standalone URDF/MJCF importers must not constrain the base environment."""
    with (source_checkout_root / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    project = pyproject["project"]
    assert "isaacsim-asset-isolated>=6.0,<6.1" not in project["dependencies"]
    assert "tinyobjloader==2.0.0rc13" not in project["dependencies"]
    assert project["optional-dependencies"]["importers"] == [
        "isaacsim-asset-isolated>=6.0,<6.1",
        "tinyobjloader==2.0.0rc13",
    ]
