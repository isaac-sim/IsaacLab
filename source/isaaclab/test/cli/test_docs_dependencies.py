# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the lock-backed documentation dependency workflow."""

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


def _load_toml(path: Path) -> dict:
    """Load a TOML file."""
    with path.open("rb") as file:
        return tomllib.load(file)


def test_docs_dependencies_are_locked_in_a_dedicated_group():
    """The lightweight docs environment must use the root lock and the shared Warp pin."""
    repo_root = _repo_root()
    pyproject = _load_toml(repo_root / "pyproject.toml")
    docs_dependencies = set(pyproject["dependency-groups"]["docs"])
    expected_dependencies = {
        "autodocsumm",
        "gymnasium",
        "lazy_loader>=0.4",
        "matplotlib",
        "myst-parser",
        "numpy",
        "sphinx>=7.0,<9",
        "sphinx-book-theme>=1.1",
        "sphinx-copybutton",
        "sphinx-icon",
        "sphinx-multiversion==0.2.4",
        "sphinx-paramlinks",
        "sphinx-tabs",
        "sphinx_design",
        "sphinxcontrib-bibtex==2.5.0",
        "sphinxemoji",
        "warp-lang",
    }

    assert docs_dependencies == expected_dependencies
    assert not (repo_root / "docs/requirements.txt").exists()

    lock = _load_toml(repo_root / "uv.lock")
    root_package = next(package for package in lock["package"] if package["name"] == "isaaclab-dev")
    locked_docs = {dependency["name"] for dependency in root_package["dev-dependencies"]["docs"]}
    expected_names = {
        re.split(r"[\s<>=!~\[;]", dependency, maxsplit=1)[0].replace("_", "-") for dependency in docs_dependencies
    }
    assert locked_docs == expected_names

    warp_version = next(package["version"] for package in lock["package"] if package["name"] == "warp-lang")
    assert warp_version == pyproject["tool"]["isaaclab"]["versions"]["warp"]


def test_docs_consumers_use_the_locked_group():
    """Docs builds, license collection, and contributor instructions must use the UV group."""
    repo_root = _repo_root()
    docs_workflow = (repo_root / ".github/workflows/docs.yaml").read_text(encoding="utf-8")
    license_workflow = (repo_root / ".github/workflows/license-check.yaml").read_text(encoding="utf-8")
    docs_readme = (repo_root / "docs/README.md").read_text(encoding="utf-8")

    assert docs_workflow.count("uv sync --locked --only-group docs") == 2
    assert "docs/requirements.txt" not in docs_workflow
    assert "uv sync --locked --extra all --group docs" in license_workflow
    assert "docs/requirements.txt" not in license_workflow
    assert docs_readme.count("uv sync --locked --only-group docs") == 4
    assert "requirements.txt" not in docs_readme
    assert docs_readme.count("source .venv/bin/activate") == 2
    assert docs_readme.count(r".venv\Scripts\activate.bat") == 2
