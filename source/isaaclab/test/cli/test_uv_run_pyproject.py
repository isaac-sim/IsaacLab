# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the root pyproject metadata used by the ``uv run`` workflow."""

from __future__ import annotations

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
            "isaaclab-ppisp",
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
    assert "isaaclab-rl[rsl-rl]" in dependencies


def test_uv_run_uses_managed_python():
    """Avoid building the project venv from conda Python and its older C++ runtime."""
    tool_uv = _root_pyproject()["tool"]["uv"]

    assert tool_uv["python-preference"] == "only-managed"
