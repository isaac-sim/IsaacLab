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


def _repo_root() -> Path:
    """Find the Isaac Lab repository root from this test file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find Isaac Lab repository root.")


def test_isaaclab_uses_one_standalone_usd_provider():
    """Isaac Lab must install only the USD provider shared with its importer dependencies.

    ``usd-core`` and ``usd-exchange`` each install a complete ``pxr`` into the same directory, so
    a second provider silently overwrites the first and removing either one leaves ``pxr`` broken.
    Nothing detects that, because the two are separate distributions.
    """
    with (_repo_root() / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    dependencies = pyproject["project"]["dependencies"]
    usd_providers = [
        dependency
        for dependency in dependencies
        if dependency.startswith("usd-core") or dependency.startswith("usd-exchange")
    ]

    assert usd_providers == ["usd-exchange==2.3.0"]


def test_resolved_environment_has_no_second_usd_provider():
    """No dependency may pull ``usd-core`` back in behind an extra.

    ``newton[importers]``, ``mujoco[usd]`` and ``warp-lang[examples]`` all require it, so selecting
    any of them would reinstate the overlap that the direct dependencies avoid. Checking the lock
    catches that, where checking ``pyproject.toml`` alone would not.
    """
    with (_repo_root() / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)

    locked = {package["name"] for package in lock["package"]}

    assert "usd-core" not in locked
    assert "usd-exchange" in locked


def test_standalone_importers_ship_as_base_dependencies():
    """The standalone URDF/MJCF importers install by default, so conversion works without Isaac Sim."""
    with (_repo_root() / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    assert "isaacsim-asset-isolated>=6.0,<6.1" in pyproject["project"]["dependencies"]
    assert "importers" not in pyproject["project"]["optional-dependencies"]
