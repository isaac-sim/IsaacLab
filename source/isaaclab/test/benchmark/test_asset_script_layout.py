# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the consolidated asset benchmark script layout."""

from pathlib import Path


def test_each_backend_retains_exactly_three_asset_scripts() -> None:
    """Each backend should retain one combined script per asset concept."""
    root = Path(__file__).resolve().parents[4]
    expected = {
        "benchmark_articulation.py",
        "benchmark_rigid_object.py",
        "benchmark_rigid_object_collection.py",
    }

    for package in ("isaaclab_physx", "isaaclab_newton", "isaaclab_ovphysx"):
        scripts = {path.name for path in (root / "source" / package / "benchmark" / "assets").glob("*.py")}
        assert scripts == expected
