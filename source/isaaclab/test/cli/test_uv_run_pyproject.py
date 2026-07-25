# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the root pyproject metadata used by the ``uv run`` workflow."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import tomllib


def _repo_root() -> Path:
    """Find the Isaac Lab repository root from this test file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find Isaac Lab repository root.")


def _root_pyproject() -> dict:
    """Load the root development ``pyproject.toml``."""
    with (_repo_root() / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _ovrtx_requirement_from_setup() -> str:
    """Return the OVRTX requirement declared by ``source/isaaclab_ov/setup.py``."""
    setup_path = _repo_root() / "source/isaaclab_ov/setup.py"
    module = ast.parse(setup_path.read_text(encoding="utf-8"))

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "EXTRAS_REQUIRE" for target in node.targets):
            continue
        extras_require = ast.literal_eval(node.value)
        for dependency in extras_require["ovrtx"]:
            if dependency.startswith("ovrtx"):
                return dependency

    raise AssertionError("Could not find the OVRTX requirement in source/isaaclab_ov/setup.py")


def test_uv_run_extra_names_match_documented_workflow():
    """Docs must only reference ``uv run --extra`` names that pyproject defines."""
    repo_root = _repo_root()
    docs = (repo_root / "docs/source/setup/installation/uv_run.rst").read_text(encoding="utf-8")
    documented_extras = set(re.findall(r"--extra\s+([A-Za-z0-9_-]+)", docs))
    optional_dependencies = _root_pyproject()["project"]["optional-dependencies"]

    assert documented_extras
    assert documented_extras <= set(optional_dependencies)


def test_uv_run_keeps_modular_extras_without_isaacsim():
    """The root dev project keeps local module extras but leaves Isaac Sim opt-in out."""
    optional_dependencies = _root_pyproject()["project"]["optional-dependencies"]

    expected_extras = {
        "contrib": ["isaaclab-contrib"],
        "mimic": ["isaaclab-mimic"],
        "newton": ["isaaclab-newton[all]", "isaaclab-physx[newton]", "isaaclab-visualizers[newton]"],
        "ov": ["isaaclab-ovphysx[ovphysx]"],
        "rl": ["isaaclab-rl[rsl-rl]"],
        "rl-all": ["isaaclab-rl[all]"],
        "rtx": ["isaaclab-ov[ovrtx]"],
        "all": [
            "isaaclab-mimic",
            "isaaclab-newton[all]",
            "isaaclab-physx[newton]",
            "isaaclab-rl[all]",
            "isaaclab-visualizers[all]",
        ],
    }

    assert optional_dependencies == expected_extras
    assert "isaacsim" not in optional_dependencies


def test_uv_run_base_dependencies_cover_newton_rsl_rl_training():
    """The documented bare ``uv run train`` command needs Newton and RSL-RL extras."""
    dependencies = _root_pyproject()["project"]["dependencies"]

    assert "isaaclab-newton[all]" in dependencies
    assert "isaaclab-physx[newton]" in dependencies
    assert "isaaclab-ppisp" in dependencies
    assert "isaaclab-rl[rsl-rl]" in dependencies


def test_uv_run_maps_ppisp_to_local_source():
    """The local PPISP peer extension must not resolve from the package registry."""
    uv_sources = _root_pyproject()["tool"]["uv"]["sources"]

    assert uv_sources["isaaclab-ppisp"]["path"] == "source/isaaclab_ppisp"
    assert uv_sources["isaaclab-ppisp"]["editable"] is True


def test_uv_run_uses_managed_python():
    """Avoid building the project venv from conda Python and its older C++ runtime."""
    tool_uv = _root_pyproject()["tool"]["uv"]

    assert tool_uv["python-preference"] == "only-managed"


def test_ci_ovrtx_installs_match_source_package_requirement():
    """CI must not bypass the OVRTX version range declared by the source package."""
    expected_requirement = _ovrtx_requirement_from_setup()

    for workflow_path in (".github/workflows/build.yaml",):
        workflow = (_repo_root() / workflow_path).read_text(encoding="utf-8")
        ovrtx_install_lines = [
            line.strip() for line in workflow.splitlines() if "extra-pip-packages:" in line and "ovrtx" in line
        ]
        assert ovrtx_install_lines
        assert all(expected_requirement in line for line in ovrtx_install_lines)
