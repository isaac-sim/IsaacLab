# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for source package dependency metadata."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["torch", "torchvision", "torchaudio"])
def test_resolved_torch_stack_supports_blackwell(source_checkout_root: Path, name: str):
    """All supported platforms need CUDA 13 wheels; PyTorch 2.12's cu126 excludes Blackwell."""
    with (source_checkout_root / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)

    packages = [package for package in lock["package"] if package["name"] == name]
    assert packages
    assert all(package["version"].endswith("+cu130") for package in packages)
    assert all(package["source"]["registry"] == "https://download.pytorch.org/whl/cu130" for package in packages)


def _requirement_name(requirement: str) -> str:
    """Return the normalized distribution name of a requirement string."""
    return re.split(r"[\s\[<>=!~;@]", requirement, maxsplit=1)[0].lower()


def _root_project(source_checkout_root: Path) -> dict:
    """Load the ``[project]`` table of the root ``pyproject.toml``."""
    with (source_checkout_root / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def test_resolved_environment_has_no_second_usd_provider(source_checkout_root: Path):
    """Isaac Lab must install only the USD provider shared with its importer dependencies.

    ``usd-core`` and ``usd-exchange`` each install a complete ``pxr`` into the same directory, so
    a second provider silently overwrites the first and removing either one leaves ``pxr`` broken.
    Nothing detects that, because the two are separate distributions.

    No dependency may pull ``usd-core`` back in behind an extra either:
    ``newton[importers]``, ``mujoco[usd]`` and ``warp-lang[examples]`` all require it, so selecting
    any of them would reinstate the overlap that the direct dependencies avoid. Checking the lock
    catches that, where checking ``pyproject.toml`` alone would not.
    """
    with (source_checkout_root / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)

    direct_names = [_requirement_name(dep) for dep in _root_project(source_checkout_root)["dependencies"]]
    assert [name for name in direct_names if name in ("usd-core", "usd-exchange")] == ["usd-exchange"]

    locked = {package["name"] for package in lock["package"]}

    assert "usd-core" not in locked
    assert "usd-exchange" in locked


def test_standalone_importers_are_opt_in(source_checkout_root: Path):
    """Standalone URDF/MJCF importers must not constrain the base environment."""
    project = _root_project(source_checkout_root)
    importers = {"isaacsim-asset-isolated", "tinyobjloader"}

    assert not {_requirement_name(dep) for dep in project["dependencies"]} & importers
    assert {_requirement_name(dep) for dep in project["optional-dependencies"]["importers"]} == importers
