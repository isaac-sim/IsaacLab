# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for wheel-builder package metadata generated from the root pyproject."""

from __future__ import annotations

import subprocess
import sys
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


def _root_rsl_rl_pin() -> str:
    """Return the ``rsl-rl-lib`` pin declared by the root ``pyproject.toml`` core deps."""
    with (_repo_root() / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    for dependency in data["project"]["dependencies"]:
        if dependency.startswith("rsl-rl-lib=="):
            return dependency
    raise AssertionError("Could not find rsl-rl-lib pin in the root pyproject.toml")


def _generate_wheel_pyproject(tmp_path: Path) -> dict:
    """Run ``gen_pyproject.py`` against the root pyproject and return the parsed result."""
    repo_root = _repo_root()
    output = tmp_path / "pyproject.toml"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/wheel_builder/gen_pyproject.py"),
            str(repo_root / "pyproject.toml"),
            str(output),
            "3.0.0",
        ],
        check=True,
    )
    with output.open("rb") as f:
        return tomllib.load(f)


def test_wheel_builder_drops_workspace_members(tmp_path):
    """The generated wheel metadata must not depend on the bundled ``isaaclab*`` packages."""
    generated = _generate_wheel_pyproject(tmp_path)
    dependencies = generated["project"]["dependencies"]

    assert not [dep for dep in dependencies if dep.lower().startswith("isaaclab")]


def test_wheel_builder_includes_isaacsim_extra(tmp_path):
    """The ``isaacsim`` extra must ship in the generated wheel metadata."""
    generated = _generate_wheel_pyproject(tmp_path)
    optional_dependencies = generated["project"]["optional-dependencies"]

    assert "isaacsim" in optional_dependencies
    assert any(dep.startswith("isaacsim[") for dep in optional_dependencies["isaacsim"])


def test_wheel_builder_rsl_rl_pin_matches_root_pyproject(tmp_path):
    """The bundled wheel metadata must install the RSL-RL version declared at the root."""
    expected_pin = _root_rsl_rl_pin()
    generated = _generate_wheel_pyproject(tmp_path)

    # RSL-RL is a core dependency (default training library) and also exposed as an extra.
    core_pins = [dep for dep in generated["project"]["dependencies"] if dep.startswith("rsl-rl-lib==")]
    assert core_pins == [expected_pin]

    optional_dependencies = generated["project"]["optional-dependencies"]
    # RSL-RL ships in its own ``rsl-rl`` extra and in the aggregate ``all`` extra.
    for extra_name in ("rsl-rl", "all"):
        rsl_rl_pins = [dep for dep in optional_dependencies[extra_name] if dep.startswith("rsl-rl-lib==")]
        assert rsl_rl_pins == [expected_pin]
