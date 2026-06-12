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


def test_isaaclab_usd_core_pin_stays_on_isaacsim_compatible_abi():
    """The kit-less USD package must stay below the next incompatible USD ABI."""
    with (_repo_root() / "source/isaaclab/pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    usd_core_dependencies = [
        dependency for dependency in pyproject["project"]["dependencies"] if dependency.startswith("usd-core")
    ]

    assert usd_core_dependencies == ["usd-core>=25.5,<26.5 ; platform_machine in 'x86_64 AMD64'"]
