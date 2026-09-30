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
from packaging.markers import Marker
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit


_CUDA_ENVIRONMENTS = (
    {"sys_platform": "linux", "platform_machine": "x86_64"},
    {"sys_platform": "linux", "platform_machine": "aarch64"},
    {"sys_platform": "win32", "platform_machine": "AMD64"},
)
"""Supported CUDA platforms from ``[tool.uv].environments``."""

_MACOS_ENVIRONMENT = {"sys_platform": "darwin", "platform_machine": "arm64"}
"""Supported macOS platform from ``[tool.uv].environments``, which runs on the CPU only."""


def _applies(marker: str | None, environment: dict[str, str]) -> bool:
    """Whether a requirement or lock marker applies in ``environment``."""
    return marker is None or Marker(marker).evaluate(environment)


@pytest.mark.parametrize("name", ["torch", "torchvision", "torchaudio"])
def test_resolved_torch_stack_supports_blackwell(source_checkout_root: Path, name: str):
    """CUDA platforms need CUDA 13 wheels; PyTorch 2.12's cu126 excludes Blackwell. macOS uses PyPI's build."""
    with (source_checkout_root / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)

    packages = [package for package in lock["package"] if package["name"] == name]
    cuda = [
        package
        for package in packages
        if any(_applies(m, env) for m in package["resolution-markers"] for env in _CUDA_ENVIRONMENTS)
    ]
    assert cuda
    assert all(package["version"].endswith("+cu130") for package in cuda)
    assert all(package["source"]["registry"] == "https://download.pytorch.org/whl/cu130" for package in cuda)
    assert all(
        package["source"]["registry"] == "https://pypi.org/simple" for package in packages if package not in cuda
    )


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

    ``usd-exchange`` ships no macOS wheels, so macOS installs ``usd-core`` as its only provider instead.
    """
    with (source_checkout_root / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)

    providers = [
        Requirement(dep)
        for dep in _root_project(source_checkout_root)["dependencies"]
        if _requirement_name(dep) in ("usd-core", "usd-exchange")
    ]
    for environment in (*_CUDA_ENVIRONMENTS, _MACOS_ENVIRONMENT):
        installed = [req.name for req in providers if _applies(str(req.marker) if req.marker else None, environment)]
        assert installed == (["usd-core"] if environment is _MACOS_ENVIRONMENT else ["usd-exchange"])

    usd_core_edges = [
        dep
        for package in lock["package"]
        for deps in [package.get("dependencies", []), *package.get("optional-dependencies", {}).values()]
        for dep in deps
        if dep["name"] == "usd-core"
    ]
    assert usd_core_edges
    assert not any(_applies(dep.get("marker"), env) for dep in usd_core_edges for env in _CUDA_ENVIRONMENTS)
    assert "usd-exchange" in {package["name"] for package in lock["package"]}


def test_standalone_importers_are_opt_in(source_checkout_root: Path):
    """Standalone URDF/MJCF importers must not constrain the base environment."""
    project = _root_project(source_checkout_root)
    importers = {"isaacsim-asset-isolated", "tinyobjloader"}

    assert not {_requirement_name(dep) for dep in project["dependencies"]} & importers
    assert {_requirement_name(dep) for dep in project["optional-dependencies"]["importers"]} == importers
