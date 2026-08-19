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


def _generate_uv_overrides(tmp_path: Path) -> list[str]:
    """Run ``gen_uv_overrides.py`` against the root pyproject and return its requirements."""
    repo_root = _repo_root()
    output = tmp_path / "uv-overrides.txt"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/wheel_builder/gen_uv_overrides.py"),
            str(repo_root / "pyproject.toml"),
            str(output),
        ],
        check=True,
    )
    return output.read_text(encoding="utf-8").splitlines()


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


def test_wheel_builder_requests_required_tinyobjloader_prerelease_directly(tmp_path):
    """Plain wheel installs must opt into the isolated importer's prerelease dependency."""
    generated = _generate_wheel_pyproject(tmp_path)

    assert "tinyobjloader==2.0.0rc13" in generated["project"]["dependencies"]


def test_wheel_builder_expands_all_extra_into_concrete_requirements(tmp_path):
    """``isaaclab[all]`` must ship the aggregated requirements, not a self-reference.

    At the root, ``all`` is the self-reference ``isaaclab-dev[...]``. The generator
    inlines it, so the published wheel carries the concrete third-party requirements
    for the curated OV backends, RL libraries, and visualizers.
    """
    generated = _generate_wheel_pyproject(tmp_path)
    optional_dependencies = generated["project"]["optional-dependencies"]
    all_extra = optional_dependencies["all"]

    assert not any(dep.lower().startswith("isaaclab") for dep in all_extra)
    # Sampled across what ``all`` aggregates: both OV backends, the RL libraries,
    # and the visualizers.
    for prefix in ("ovphysx", "ovrtx", "ovstage", "stable-baselines3", "skrl", "viser", "rerun-sdk"):
        assert any(dep.startswith(prefix) for dep in all_extra), f"'{prefix}' missing from the 'all' extra"
    # Isaac Sim, specialized extras, and developer tooling stay opt-in by name.
    for prefix in ("isaacsim[", "ray", "robomimic", "isaacteleop", "pytetwild", "moviepy", "leapp", "pytest"):
        assert not any(dep.startswith(prefix) for dep in all_extra), f"'{prefix}' must not be in the 'all' extra"


def test_wheel_builder_rsl_rl_pin_matches_root_pyproject(tmp_path):
    """The bundled wheel metadata must install the RSL-RL version declared at the root."""
    expected_pin = _root_rsl_rl_pin()
    generated = _generate_wheel_pyproject(tmp_path)

    # RSL-RL is a core dependency (default training library) and also exposed as an extra.
    core_pins = [dep for dep in generated["project"]["dependencies"] if dep.startswith("rsl-rl-lib==")]
    assert core_pins == [expected_pin]

    optional_dependencies = generated["project"]["optional-dependencies"]
    # RSL-RL is also exposed through its own ``rsl-rl`` extra.
    rsl_rl_pins = [dep for dep in optional_dependencies["rsl-rl"] if dep.startswith("rsl-rl-lib==")]
    assert rsl_rl_pins == [expected_pin]


def test_wheel_builder_keeps_tetrahedralization_explicit(tmp_path):
    """The generated wheel must expose PyTetWild only through its explicit extra."""
    generated = _generate_wheel_pyproject(tmp_path)
    project = generated["project"]
    optional_dependencies = project["optional-dependencies"]

    assert not any(dep.startswith("pytetwild") for dep in project["dependencies"])
    assert optional_dependencies["tetrahedralization"] == ["pytetwild[all]>=0.3.0,<0.4"]
    for name, deps in optional_dependencies.items():
        if name == "tetrahedralization":
            continue
        assert not any(dep.startswith("pytetwild") for dep in deps)


def test_wheel_builder_uv_overrides_match_root_pyproject(tmp_path):
    """The wheel resolver override file must mirror the root uv overrides exactly."""
    with (_repo_root() / "pyproject.toml").open("rb") as f:
        root = tomllib.load(f)

    generated_overrides = _generate_uv_overrides(tmp_path)
    published_overrides = (
        (_repo_root() / "tools" / "wheel_builder" / "uv-overrides.txt").read_text(encoding="utf-8").splitlines()
    )
    install_ci_overrides = (
        (_repo_root() / "source" / "isaaclab" / "test" / "install_ci" / "uv_pip" / "uv-overrides.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert generated_overrides == root["tool"]["uv"]["override-dependencies"]
    assert published_overrides == generated_overrides
    assert install_ci_overrides == generated_overrides


def test_wheel_builder_uv_overrides_relax_isaacsim_exact_pins(tmp_path):
    """The wheel resolver must relax Isaac Sim 6.0's exact pins so the extras co-resolve."""
    overrides = _generate_uv_overrides(tmp_path)

    for spec in ("typing-extensions>=4.15.0", "websockets>=14.0,<17.0.0", "coverage>=7.6.1"):
        assert spec in overrides
