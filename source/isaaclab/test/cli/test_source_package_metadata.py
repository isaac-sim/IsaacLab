# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for source package dependency metadata."""

from __future__ import annotations

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


def test_isaaclab_usd_core_pin_includes_multithreaded_collider_crash_fix():
    """The kit-less USD package must include the OpenUSD 26.05 fix for the multithreaded
    UsdPhysicsParsingUtility crash (one rigid body with many mesh colliders).

    See OpenUSD PR #4002 / commit 060715f ("[usdPhysics] fix for a multithreaded crash if
    one rigidbody has multiple colliders beneath"), first released in OpenUSD 26.05
    (usd-core 26.5). Versions < 26.5 race and can corrupt the heap during USD physics parsing.
    """
    with (_repo_root() / "source/isaaclab/pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    usd_core_dependencies = [
        dependency for dependency in pyproject["project"]["dependencies"] if dependency.startswith("usd-core")
    ]

    assert usd_core_dependencies == ["usd-core>=26.5,<27.0 ; platform_machine in 'x86_64 AMD64'"]


def test_isaaclab_standalone_usd_providers_are_platform_disjoint():
    """Standalone USD packages must not overlap on platforms where both ship ``pxr``."""
    with (_repo_root() / "source/isaaclab/pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    usd_exchange_dependencies = [
        dependency for dependency in pyproject["project"]["dependencies"] if dependency.startswith("usd-exchange")
    ]

    assert usd_exchange_dependencies == ["usd-exchange>=2.2 ; platform_machine in 'aarch64 arm64'"]
