# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for source package dependency metadata."""

from __future__ import annotations

from pathlib import Path

import tomllib


def _repo_root() -> Path:
    """Find the Isaac Lab repository root from this test file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find Isaac Lab repository root.")


def test_isaaclab_standalone_usd_is_an_opt_in_extra():
    """Standalone USD must not be installed with the Isaac Sim extra."""
    with (_repo_root() / "source/isaaclab/pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    standalone_usd_dependencies = [
        dependency
        for dependency in pyproject["project"]["dependencies"]
        if dependency.startswith(("usd-core", "usd-exchange"))
    ]

    assert standalone_usd_dependencies == []
    assert pyproject["project"]["optional-dependencies"]["usd"] == ["usd-exchange>=2.2"]
