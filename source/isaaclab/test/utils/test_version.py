# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for version comparison utilities."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
from packaging.version import Version

from isaaclab.utils.version import compare_versions, get_isaac_sim_version

pytestmark = pytest.mark.integration


def test_get_isaac_sim_version():
    """Test that get_isaac_sim_version returns cached Version object."""
    # Call twice to ensure caching works
    version1 = get_isaac_sim_version()
    version2 = get_isaac_sim_version()

    # Should return the same object (cached)
    assert version1 is version2

    # Should return a packaging.version.Version object
    assert isinstance(version1, Version)

    # Major version should be reasonable
    assert version1.major >= 4

    # Minor and micro should be non-negative
    assert version1.minor >= 0
    assert version1.micro >= 0


@pytest.mark.parametrize(
    "v1,v2,expected",
    [
        # Equal versions, including implicit trailing zeros
        ("1.0.0", "1.0.0", 0),
        ("1", "1.0.0.0", 0),
        ("0", "0.0.0", 0),
        # Major, minor, and patch differences in both directions (numeric, not lexical)
        ("2.0.0", "1.99.99", 1),
        ("1.0.0", "2.0.0", -1),
        ("1.10.0", "1.9.99", 1),
        ("1.4.0", "1.5.0", -1),
        ("2.5.10", "2.5.9", 1),
        ("1.0.4", "1.0.5", -1),
        # Extended versions
        ("1.2.3.4.5", "1.2.3.4", 1),
        # Large numbers
        ("999.999.999", "1000.0.0", -1),
    ],
)
def test_version_comparisons(v1, v2, expected):
    """Test version comparisons with various scenarios."""
    assert compare_versions(v1, v2) == expected
