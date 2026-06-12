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
        "viser",
        "rerun",
        "ov",
        "mimic",
        "teleop",
        "rlinf",
        "all",
    }
    assert expected_extras <= set(optional_dependencies)

    # The Newton viewer GUI is part of the base install, so there is no ``newton``
    # extra; the OV renderer/physics wheels are a single grouped ``ov`` extra (no
    # separate ``rtx`` extra).
    assert "newton" not in optional_dependencies
    assert "rtx" not in optional_dependencies

    # Concrete third-party deps live in the extras (not subpackage self-references).
    assert any(dep.startswith("skrl") for dep in optional_dependencies["skrl"])
    assert any(dep.startswith("ovphysx") for dep in optional_dependencies["ov"])
    assert any(dep.startswith("ovrtx") for dep in optional_dependencies["ov"])


def test_version_single_source_matches_literal_pins():
    """``[tool.isaaclab.versions]`` is the single source for externally-pinned versions.

    TOML cannot interpolate, so the literal pins in ``[project.optional-dependencies]``
    and ``[tool.uv].constraint-dependencies`` must mirror the table exactly. This test
    fails if any of them drift apart.
    """
    pyproject = _root_pyproject()
    versions = pyproject["tool"]["isaaclab"]["versions"]
    optional = pyproject["project"]["optional-dependencies"]
    constraints = pyproject["tool"]["uv"]["constraint-dependencies"]

    # Isaac Sim extra mirrors the table.
    assert optional["isaacsim"] == [f"isaacsim[all,extscache]=={versions['isaacsim']}"]

    # OV collection extra mirrors the table (ovphysx exact pin, ovrtx range spec).
    assert f"ovphysx=={versions['ovphysx']}" in optional["ov"]
    assert f"ovrtx{versions['ovrtx']}" in optional["ov"]

    # uv torch-stack constraints mirror the table.
    for package in ("torch", "torchvision", "torchaudio"):
        assert f"{package}=={versions[package]}" in constraints


def test_uv_run_isaacsim_extra_is_conflict_forked():
    """Isaac Sim is an opt-in uv workspace extra, forked away from clashing extras.

    PhysX/Isaac Sim is never a base dependency, but it must be a real
    ``optional-dependencies`` extra so ``uv run --extra isaacsim`` resolves. Its
    exact pins clash with several other extras, so it is declared in
    ``[tool.uv].conflicts`` (forked resolution) rather than co-resolved with them.
    """
    pyproject = _root_pyproject()
    project = pyproject["project"]
    base_dependency_names = {re.split(r"[\s<>=!~\[;]", dep, maxsplit=1)[0] for dep in project["dependencies"]}

    # PhysX/Isaac Sim is opt-in, never installed by the bare ``uv run``.
    assert "isaacsim" not in base_dependency_names
    # ...but it is a workspace extra so ``uv run --extra isaacsim`` works.
    assert "isaacsim" in project["optional-dependencies"]
    assert any(dep.startswith("isaacsim[") for dep in project["optional-dependencies"]["isaacsim"])
    # The legacy wheel-only table is gone (isaacsim now lives in the extras).
    assert "wheel-extras" not in pyproject.get("tool", {}).get("isaaclab", {})

    # isaacsim is forked away from every extra whose pins clash with it.
    conflict_groups = [{entry["extra"] for entry in group} for group in pyproject["tool"]["uv"]["conflicts"]]
    for extra in ("teleop", "ov", "viser", "mimic", "all", "test"):
        assert {"isaacsim", extra} in conflict_groups, f"isaacsim must declare a conflict with '{extra}'"


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
