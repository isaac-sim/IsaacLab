# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for wheel-builder package metadata."""

from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

_SKRL_WARP_113_MIN_VERSION = Version("2.1.0")


def _repo_root() -> Path:
    """Find the Isaac Lab repository root from this test file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find Isaac Lab repository root.")


def _rsl_rl_pin_from_pyproject() -> str:
    """Return the ``rsl-rl-lib`` pin declared by ``source/isaaclab_rl/pyproject.toml``."""
    pyproject_path = _repo_root() / "source/isaaclab_rl/pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    for dependency in data.get("project", {}).get("optional-dependencies", {}).get("rsl-rl", []):
        if dependency.startswith("rsl-rl-lib=="):
            return dependency

    raise AssertionError("Could not find rsl-rl-lib pin in source/isaaclab_rl/pyproject.toml")


def _source_rl_optional_dependencies() -> dict[str, list[str]]:
    """Return optional dependencies declared by ``source/isaaclab_rl/pyproject.toml``."""
    pyproject_path = _repo_root() / "source/isaaclab_rl/pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    return data.get("project", {}).get("optional-dependencies", {})


def _wheel_builder_rl_optional_dependencies() -> dict[str, list[str]]:
    """Return wheel-builder optional dependency groups."""
    packages_path = _repo_root() / "tools/wheel_builder/res/python_packages.toml"
    with packages_path.open("rb") as f:
        packages = tomllib.load(f)

    optional_dependencies = packages["isaaclab"]["pyproject"]["optional-dependencies"]["all"]
    return {name: deps for entry in optional_dependencies for name, deps in entry.items()}


def _requirement_for(requirements: list[str], package_name: str) -> str:
    """Return the unique requirement string for ``package_name``."""
    matches = [requirement for requirement in requirements if Requirement(requirement).name == package_name]
    assert len(matches) == 1, f"Expected exactly one {package_name} requirement, got {matches}"
    return matches[0]


def _minimum_version_for(requirement: str) -> Version | None:
    """Return the lower-bound version encoded by a requirement."""
    versions = [
        Version(spec.version)
        for spec in Requirement(requirement).specifier
        if spec.operator in {"==", "===", ">=", "~="}
    ]
    return max(versions) if versions else None


def test_wheel_builder_rsl_rl_pin_matches_source_package():
    """The bundled wheel metadata must install the RSL-RL version required by training scripts."""
    expected_pin = _rsl_rl_pin_from_pyproject()
    dependencies_by_extra = _wheel_builder_rl_optional_dependencies()

    for extra_name in ("rsl-rl", "all"):
        rsl_rl_pins = [dep for dep in dependencies_by_extra[extra_name] if dep.startswith("rsl-rl-lib==")]
        assert rsl_rl_pins == [expected_pin]


def test_isaaclab_rl_skrl_requirement_supports_warp_113():
    """The skrl extra must install a version compatible with Warp 1.13."""
    optional_dependencies = _source_rl_optional_dependencies()

    for extra_name in ("skrl", "all"):
        skrl_requirement = _requirement_for(optional_dependencies[extra_name], "skrl")
        minimum_version = _minimum_version_for(skrl_requirement)
        assert minimum_version is not None
        assert minimum_version >= _SKRL_WARP_113_MIN_VERSION


def test_wheel_builder_skrl_requirement_matches_source_package():
    """Wheel metadata must preserve the skrl version floor required by source installs."""
    source_dependencies = _source_rl_optional_dependencies()
    wheel_dependencies = _wheel_builder_rl_optional_dependencies()
    expected_requirement = _requirement_for(source_dependencies["skrl"], "skrl")

    for extra_name in ("skrl", "all"):
        assert _requirement_for(wheel_dependencies[extra_name], "skrl") == expected_requirement
