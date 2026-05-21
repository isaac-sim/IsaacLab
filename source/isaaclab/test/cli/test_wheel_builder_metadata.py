# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for wheel-builder package metadata."""

from __future__ import annotations

import ast
from pathlib import Path

import tomllib


def _repo_root() -> Path:
    """Find the Isaac Lab repository root from this test file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find Isaac Lab repository root.")


def _rsl_rl_pin_from_setup() -> str:
    """Return the ``rsl-rl-lib`` pin declared by ``source/isaaclab_rl/setup.py``."""
    setup_path = _repo_root() / "source/isaaclab_rl/setup.py"
    module = ast.parse(setup_path.read_text(encoding="utf-8"))

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "EXTRAS_REQUIRE" for target in node.targets):
            continue
        extras_require = ast.literal_eval(node.value)
        for dependency in extras_require["rsl-rl"]:
            if dependency.startswith("rsl-rl-lib=="):
                return dependency

    raise AssertionError("Could not find rsl-rl-lib pin in source/isaaclab_rl/setup.py")


def test_wheel_builder_rsl_rl_pin_matches_source_package():
    """The bundled wheel metadata must install the RSL-RL version required by training scripts."""
    expected_pin = _rsl_rl_pin_from_setup()
    packages_path = _repo_root() / "tools/wheel_builder/res/python_packages.toml"
    with packages_path.open("rb") as f:
        packages = tomllib.load(f)

    optional_dependencies = packages["isaaclab"]["pyproject"]["optional-dependencies"]["all"]
    dependencies_by_extra = {name: deps for entry in optional_dependencies for name, deps in entry.items()}

    for extra_name in ("rsl-rl", "rsl_rl", "all"):
        rsl_rl_pins = [dep for dep in dependencies_by_extra[extra_name] if dep.startswith("rsl-rl-lib==")]
        assert rsl_rl_pins == [expected_pin]
