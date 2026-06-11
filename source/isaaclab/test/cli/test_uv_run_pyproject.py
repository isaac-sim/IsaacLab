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


def test_uv_run_exposes_centralized_feature_extras():
    """The root project centralizes optional third-party deps into named extras."""
    optional_dependencies = _root_pyproject()["project"]["optional-dependencies"]

    # Feature extras a user can activate with ``uv run --extra``.
    expected_extras = {
        "test",
        "sb3",
        "skrl",
        "rl-games",
        "rsl-rl",
        "newton",
        "viser",
        "rerun",
        "ov",
        "rtx",
        "mimic",
        "teleop",
        "rlinf",
        "all",
    }
    assert expected_extras <= set(optional_dependencies)

    # Concrete third-party deps live in the extras (not subpackage self-references).
    assert any(dep.startswith("skrl") for dep in optional_dependencies["skrl"])
    assert any(dep.startswith("ovphysx") for dep in optional_dependencies["ov"])
    assert any(dep.startswith("ovrtx") for dep in optional_dependencies["rtx"])


def test_uv_run_keeps_isaacsim_out_of_workspace_resolution():
    """Isaac Sim is a wheel-only extra: never a base dep, never a uv workspace extra.

    The source workspace uses a repo-local Isaac Sim, and isaacsim's exact pins
    conflict with several workspace extras under uv's strict resolver, so it is
    declared under ``[tool.isaaclab.wheel-extras]`` instead.
    """
    pyproject = _root_pyproject()
    project = pyproject["project"]
    base_dependency_names = {re.split(r"[\s<>=!~\[;]", dep, maxsplit=1)[0] for dep in project["dependencies"]}

    assert "isaacsim" not in base_dependency_names
    assert "isaacsim" not in project["optional-dependencies"]
    assert "isaacsim" in pyproject["tool"]["isaaclab"]["wheel-extras"]


def test_uv_run_base_dependencies_cover_newton_rsl_rl_training():
    """The documented bare ``uv run train`` command needs Newton and RSL-RL in core."""
    dependencies = _root_pyproject()["project"]["dependencies"]

    # Newton is the default physics engine and RSL-RL the default training library,
    # so both ship as core third-party requirements (not opt-in extras).
    assert any(dep.startswith("newton[sim]") for dep in dependencies)
    assert any(dep.startswith("rsl-rl-lib") for dep in dependencies)


def test_uv_run_uses_managed_python():
    """Avoid building the project venv from conda Python and its older C++ runtime."""
    tool_uv = _root_pyproject()["tool"]["uv"]

    assert tool_uv["python-preference"] == "only-managed"
