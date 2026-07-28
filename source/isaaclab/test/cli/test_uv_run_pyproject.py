# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the root pyproject metadata used by the ``uv run`` workflow."""

from __future__ import annotations

import re
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


def _root_pyproject() -> dict:
    """Load the root development ``pyproject.toml``."""
    with (_repo_root() / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def test_uv_run_extra_names_match_documented_workflow():
    """Docs must only reference ``uv run --extra`` names that pyproject defines."""
    repo_root = _repo_root()
    docs = (repo_root / "docs/source/setup/installation/index.rst").read_text(encoding="utf-8")
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
        "ovphysx",
        "ovrtx",
        "mimic",
        "teleop",
        "rlinf",
        "all",
    }
    assert expected_extras <= set(optional_dependencies)

    # The Newton viewer GUI is part of the base install, so there is no ``newton`` extra.
    assert "newton" not in optional_dependencies
    assert "ov" not in optional_dependencies
    assert "rtx" not in optional_dependencies

    # Concrete third-party deps live in the extras (not subpackage self-references).
    # OVPhysX and OVRTX are separate extras, selectable via ``ov[ovphysx]`` / ``ov[ovrtx]``.
    assert any(dep.startswith("skrl") for dep in optional_dependencies["skrl"])
    assert any(dep.startswith("ovphysx") for dep in optional_dependencies["ovphysx"])
    assert any(dep.startswith("ovrtx") for dep in optional_dependencies["ovrtx"])


def test_version_single_source_matches_literal_pins():
    """``[tool.isaaclab.versions]`` is the single source for externally-pinned versions.

    TOML cannot interpolate, so the literal pins in ``[project.dependencies]``,
    ``[project.optional-dependencies]``, and ``[tool.uv].override-dependencies`` must
    mirror the table exactly. This test fails if any of them drift apart.
    """
    pyproject = _root_pyproject()
    versions = pyproject["tool"]["isaaclab"]["versions"]
    dependencies = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]
    overrides = pyproject["tool"]["uv"]["override-dependencies"]

    assert versions["ovphysx"] == "0.5.9"
    assert "omniverseclient==2.72.3" in dependencies

    # Isaac Sim extra mirrors the table.
    assert optional["isaacsim"] == [f"isaacsim[all,extscache]=={versions['isaacsim']}"]

    # OV extras mirror the table. Table values may be an exact version ("1.2.3",
    # mirrored as ``pkg==1.2.3``) or a range spec (">=1.2.3", mirrored as ``pkg>=1.2.3``).
    def spec(package: str) -> str:
        value = versions[package]
        return f"{package}=={value}" if value[0].isdigit() else f"{package}{value}"

    assert spec("ovphysx") in optional["ovphysx"]
    assert spec("ovrtx") in optional["ovrtx"]
    assert spec("ovstage") in optional["ovphysx"]
    assert spec("ovstage") in optional["ovrtx"]

    # CI installs OVRTX through a generic pip-package input (a bare ``pip install
    # ovrtx`` ignores this ceiling). Each such install must therefore be pinned:
    # either by carrying the literal range, or by referencing the ``resolve-ov-pins``
    # action output, which reads the pin from this same table. Never a bare ``ovrtx``.
    build_workflow = (_repo_root() / ".github/workflows/build.yaml").read_text(encoding="utf-8")
    assert "ovphysx==0.4.13" not in build_workflow
    ovrtx_install_lines = [
        line.strip() for line in build_workflow.splitlines() if "extra-pip-packages:" in line and "ovrtx" in line
    ]
    assert ovrtx_install_lines
    assert all(
        f"ovrtx{versions['ovrtx']}" in line or "steps.ov_pins.outputs.ovrtx" in line for line in ovrtx_install_lines
    )

    # uv torch-stack overrides mirror the table.
    for package in ("torch", "torchvision", "torchaudio"):
        assert f"{package}=={versions[package]}" in overrides

    # Newton is pinned to a git ref (branch/tag/commit) via a uv override; warp-lang is a
    # core dependency whose table value may be an exact pin ("1.2.3" -> ``==``) or a range
    # (">=1.2.3" -> mirrored verbatim).
    assert any(dep.endswith(f"newton.git@{versions['newton']}") for dep in overrides)
    warp_value = versions["warp"]
    warp_spec = f"warp-lang=={warp_value}" if warp_value[0].isdigit() else f"warp-lang{warp_value}"
    assert warp_spec in dependencies


def test_public_ov_packages_use_public_pypi_index():
    """Public OV packages must not resolve from the NVIDIA package index."""
    pyproject = _root_pyproject()
    indexes = {index.get("name"): index for index in pyproject["tool"]["uv"]["index"]}
    sources = pyproject["tool"]["uv"]["sources"]

    assert indexes["pypi-public"] == {
        "name": "pypi-public",
        "url": "https://pypi.org/simple",
        "explicit": True,
    }
    for package in ("omniverseclient", "ovphysx", "ovstage"):
        assert sources[package] == {"index": "pypi-public"}


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
    for extra in ("teleop", "ovphysx", "viser", "mimic", "all", "test"):
        assert {"isaacsim", extra} in conflict_groups, f"isaacsim must declare a conflict with '{extra}'"


def test_uv_run_base_dependencies_cover_newton_rsl_rl_training():
    """The documented bare ``uv run isaaclab train`` command needs Newton and RSL-RL in core."""
    dependencies = _root_pyproject()["project"]["dependencies"]

    # Newton is the default physics engine and RSL-RL the default training library,
    # so both ship as core third-party requirements (not opt-in extras). The importers
    # extra carries the mesh-processing deps that authored collision approximations need.
    assert any(dep.startswith("newton[sim,importers]") for dep in dependencies)
    assert any(dep.startswith("rsl-rl-lib") for dep in dependencies)


def test_uv_run_uses_managed_python():
    """Avoid building the project venv from conda Python and its older C++ runtime."""
    tool_uv = _root_pyproject()["tool"]["uv"]

    assert tool_uv["python-preference"] == "only-managed"
