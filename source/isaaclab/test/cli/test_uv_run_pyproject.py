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
        "ov",
        "ovphysx",
        "ovrtx",
        "mimic",
        "teleop",
        "rlinf",
        "tetrahedralization",
        "all",
    }
    assert expected_extras <= set(optional_dependencies)

    # The Newton viewer GUI is part of the base install, so there is no ``newton`` extra.
    assert "newton" not in optional_dependencies
    assert "rtx" not in optional_dependencies

    # Concrete third-party deps live in the extras (not subpackage self-references).
    # ``ov`` installs both Omniverse backends; ``ovphysx`` / ``ovrtx`` select one.
    # Every Omniverse extra carries ``ovstage``, which both backends need.
    assert any(dep.startswith("skrl") for dep in optional_dependencies["skrl"])
    assert any(dep.startswith("ovphysx") for dep in optional_dependencies["ov"])
    assert any(dep.startswith("ovrtx") for dep in optional_dependencies["ov"])
    assert any(dep.startswith("ovstage") for dep in optional_dependencies["ov"])
    assert any(dep.startswith("ovphysx") for dep in optional_dependencies["ovphysx"])
    assert any(dep.startswith("ovstage") for dep in optional_dependencies["ovphysx"])
    assert any(dep.startswith("ovrtx") for dep in optional_dependencies["ovrtx"])
    assert any(dep.startswith("ovstage") for dep in optional_dependencies["ovrtx"])


def test_all_extra_aggregates_backends_rl_libraries_and_visualizers():
    """``all`` is the single flag for every backend, RL library, and visualizer.

    Nothing is forked in ``[tool.uv].conflicts``, so Isaac Sim and both OV backends fit in
    one environment alongside every RL library and visualizer. The specialized workflows stay
    opt-in by name -- they are large, narrowly used, or both.
    """
    optional = _root_pyproject()["project"]["optional-dependencies"]

    # ``all`` is a single self-reference listing the extras it aggregates.
    assert len(optional["all"]) == 1
    aggregated = set(re.fullmatch(r"isaaclab-dev\[(.+)\]", optional["all"][0]).group(1).split(","))
    assert aggregated == {"isaacsim", "ov", "sb3", "skrl", "rl-games", "rsl-rl", "viser", "rerun"}

    # ``ov`` pulls both OV backends, so naming it covers ``ovphysx`` and ``ovrtx`` too.
    reachable = aggregated | {"ovphysx", "ovrtx"}

    # Everything else is requested by name. A newly added extra lands in this diff and
    # has to be classified deliberately -- into ``all`` or into this list.
    assert set(optional) - reachable - {"all"} == {
        "rlinf",
        "mimic",
        "teleop",
        "tetrahedralization",
        "video",
        "leapp",
        "test",
    }


def test_tetrahedralization_is_explicit_extra_only():
    """TetWild and its visualization stack are installed only when requested."""
    project = _root_pyproject()["project"]
    optional = project["optional-dependencies"]

    assert not any(dep.startswith("pytetwild") for dep in project["dependencies"])
    assert optional["tetrahedralization"] == ["pytetwild[all]>=0.3.0,<0.4"]
    # No other extra pulls PyTetWild in transitively, including the aggregate ``all``.
    for name, deps in optional.items():
        if name == "tetrahedralization":
            continue
        assert not any("tetrahedralization" in dep or dep.startswith("pytetwild") for dep in deps)


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

    assert versions["ovphysx"] == "0.5.10"
    assert "omniverseclient==2.72.3" in dependencies

    # Isaac Sim extra mirrors the table, and the teleop extra repeats the same pin.
    assert optional["isaacsim"] == [f"isaacsim[all,extscache]=={versions['isaacsim']}"]
    assert f"isaacsim[all,extscache]=={versions['isaacsim']}" in optional["teleop"]

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
    assert all(spec("ovrtx") in line or "steps.ov_pins.outputs.ovrtx" in line for line in ovrtx_install_lines)

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
    for package in ("omniverseclient", "ovphysx", "ovrtx", "ovstage"):
        assert sources[package] == {"index": "pypi-public"}


def test_uv_run_declares_no_extra_conflicts():
    """No extra is forked: every combination resolves into a single environment.

    ``[tool.uv].conflicts`` used to fork ``isaacsim`` / ``teleop`` away from the OV runtimes,
    and briefly away from the standalone importers. The overrides below reconcile the last of
    those pins -- ``packaging`` for ovphysx, WebSockets for Viser, coverage for ``test``. The
    importers install beside Isaac Sim without displacing it: the two distributions share no
    files, and Kit serves ``isaacsim.asset`` from its extension roots either way.
    """
    tool_uv = _root_pyproject()["tool"]["uv"]

    assert "conflicts" not in tool_uv
    for override in ("packaging>=20,<27", "websockets>=14.0,<17.0.0", "coverage>=7.6.1"):
        assert override in tool_uv["override-dependencies"]


def test_uv_run_isaacsim_is_an_opt_in_extra():
    """Isaac Sim is never a base dependency, but it is a real workspace extra.

    PhysX/Isaac Sim must stay out of ``[project.dependencies]`` so the bare ``uv run``
    keeps working without Kit, while still being an ``optional-dependencies`` entry so
    ``uv run --extra isaacsim`` resolves.
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


def test_uv_run_teleop_extra_bundles_isaacsim():
    """``--extra teleop`` is the single flag for the XR teleoperation workflow.

    XR teleop needs the Kit XR runtime, so the extra carries Isaac Sim through the
    ``isaacsim`` extra rather than repeating its pin.
    """
    optional_dependencies = _root_pyproject()["project"]["optional-dependencies"]
    teleop = optional_dependencies["teleop"]

    # Isaac Sim is listed explicitly; test_version_single_source keeps the pin from drifting.
    assert any(dep.startswith("isaacsim[all,extscache]==") for dep in teleop)
    # record_demos.py imports isaaclab_mimic at module level; robomimic stays in ``mimic``.
    assert "isaaclab-mimic" in teleop
    assert not any(dep.startswith("robomimic") for dep in teleop)


def test_uv_run_base_dependencies_cover_newton_rsl_rl_training():
    """The documented bare ``uv run isaaclab train`` command needs Newton and RSL-RL in core."""
    dependencies = _root_pyproject()["project"]["dependencies"]

    # Newton is the default physics engine and RSL-RL the default training library,
    # so both ship as core third-party requirements (not opt-in extras).
    assert any(dep.startswith("newton[sim]") for dep in dependencies)
    assert any(dep.startswith("rsl-rl-lib") for dep in dependencies)


def test_uv_run_uses_managed_python():
    """Avoid building the project venv from conda Python and its older C++ runtime."""
    tool_uv = _root_pyproject()["tool"]["uv"]

    assert tool_uv["python-preference"] == "only-managed"
