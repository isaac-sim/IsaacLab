# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for wheel-builder package metadata."""

from __future__ import annotations

from pathlib import Path

import tomllib


def _repo_root() -> Path:
    """Find the Isaac Lab repository root from this test file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find Isaac Lab repository root.")


def _load_toml(relative_path: str) -> dict:
    """Load a TOML file from the Isaac Lab repository root."""
    with (_repo_root() / relative_path).open("rb") as f:
        return tomllib.load(f)


def _single_dependency(dependencies: list[str], prefix: str, source: str) -> str:
    """Return the only dependency in a list matching the provided prefix."""
    matches = [dependency for dependency in dependencies if dependency.startswith(prefix)]
    assert len(matches) == 1, f"Expected one {prefix!r} dependency in {source}, got {matches}"
    return matches[0]


def _wheel_builder_dependencies_by_extra() -> dict[str, list[str]]:
    """Return the wheel-builder optional dependencies grouped by extra name."""
    packages = _load_toml("tools/wheel_builder/res/python_packages.toml")
    optional_dependencies = packages["isaaclab"]["pyproject"]["optional-dependencies"]["all"]
    return {name: dependencies for entry in optional_dependencies for name, dependencies in entry.items()}


def _rsl_rl_pin_from_pyproject() -> str:
    """Return the ``rsl-rl-lib`` pin declared by ``source/isaaclab_rl/pyproject.toml``."""
    dependencies = _load_toml("source/isaaclab_rl/pyproject.toml")["project"]["optional-dependencies"]["rsl-rl"]
    return _single_dependency(dependencies, "rsl-rl-lib==", "source/isaaclab_rl/pyproject.toml")


def _newton_pin_from_pyproject(relative_path: str, extra_name: str) -> str:
    """Return the ``newton[sim]`` direct URL pin declared by a package extra."""
    dependencies = _load_toml(relative_path)["project"]["optional-dependencies"][extra_name]
    return _single_dependency(
        dependencies,
        "newton[sim] @ git+https://github.com/newton-physics/newton.git@",
        f"{relative_path}[{extra_name}]",
    )


def _warp_pin_from_core_pyproject() -> str:
    """Return the core ``warp-lang`` pin declared by ``source/isaaclab/pyproject.toml``."""
    dependencies = _load_toml("source/isaaclab/pyproject.toml")["project"]["dependencies"]
    return _single_dependency(dependencies, "warp-lang==", "source/isaaclab/pyproject.toml")


def test_wheel_builder_rsl_rl_pin_matches_source_package():
    """The bundled wheel metadata must install the RSL-RL version required by training scripts."""
    expected_pin = _rsl_rl_pin_from_pyproject()
    dependencies_by_extra = _wheel_builder_dependencies_by_extra()

    for extra_name in ("rsl-rl", "all"):
        rsl_rl_pin = _single_dependency(dependencies_by_extra[extra_name], "rsl-rl-lib==", extra_name)
        assert rsl_rl_pin == expected_pin


def test_wheel_builder_newton_pin_matches_source_packages():
    """The bundled wheel metadata must install the Newton revision used by source packages."""
    expected_pin = _newton_pin_from_pyproject("source/isaaclab_newton/pyproject.toml", "all")
    source_pins = [
        _newton_pin_from_pyproject("source/isaaclab_physx/pyproject.toml", "newton"),
        _newton_pin_from_pyproject("source/isaaclab_visualizers/pyproject.toml", "newton"),
        _newton_pin_from_pyproject("source/isaaclab_visualizers/pyproject.toml", "rerun"),
        _newton_pin_from_pyproject("source/isaaclab_visualizers/pyproject.toml", "viser"),
        _newton_pin_from_pyproject("source/isaaclab_visualizers/pyproject.toml", "all"),
    ]
    assert source_pins == [expected_pin] * len(source_pins)

    wheel_newton_pin = _single_dependency(
        _wheel_builder_dependencies_by_extra()["newton"],
        "newton[sim] @ git+https://github.com/newton-physics/newton.git@",
        "tools/wheel_builder/res/python_packages.toml[newton]",
    )
    assert wheel_newton_pin == expected_pin


def test_wheel_builder_warp_pin_matches_core_package():
    """The bundled wheel metadata must keep Warp aligned with the core package pin."""
    expected_pin = _warp_pin_from_core_pyproject()
    packages = _load_toml("tools/wheel_builder/res/python_packages.toml")
    wheel_core_dependencies = packages["isaaclab"]["pyproject"]["dependencies"]["all"]
    dependencies_by_extra = _wheel_builder_dependencies_by_extra()

    wheel_core_pin = _single_dependency(
        wheel_core_dependencies,
        "warp-lang==",
        "tools/wheel_builder/res/python_packages.toml[dependencies]",
    )
    wheel_newton_pin = _single_dependency(
        dependencies_by_extra["newton"],
        "warp-lang==",
        "tools/wheel_builder/res/python_packages.toml[newton]",
    )

    assert wheel_core_pin == expected_pin
    assert wheel_newton_pin == expected_pin
