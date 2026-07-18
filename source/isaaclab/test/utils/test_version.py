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


def test_get_isaac_sim_version_format():
    """Test that get_isaac_sim_version returns correct format."""
    isaac_version = get_isaac_sim_version()

    # Should be able to convert to string
    version_str = str(isaac_version)
    assert isinstance(version_str, str)

    # Should have proper format (e.g., "5.0.0")
    parts = version_str.split(".")
    assert len(parts) >= 3

    # Can access components
    assert hasattr(isaac_version, "major")
    assert hasattr(isaac_version, "minor")
    assert hasattr(isaac_version, "micro")


def test_version_caching_performance():
    """Test that caching improves performance for version checks."""
    # First call (will cache)
    version1 = get_isaac_sim_version()

    # Subsequent calls should be instant (from cache)
    for _ in range(100):
        version = get_isaac_sim_version()
        assert version == version1
        assert version is version1  # Should be the exact same object


def test_version_comparison_operators():
    """Test that Version objects support natural comparisons."""
    isaac_version = get_isaac_sim_version()

    # Should support comparison operators
    assert isaac_version >= Version("4.0.0")
    assert isaac_version == isaac_version

    # Test less than
    if isaac_version.major >= 5:
        assert isaac_version > Version("4.5.0")
        assert isaac_version >= Version("5.0.0")

    # Test not equal
    assert isaac_version != Version("0.0.1")


@pytest.mark.parametrize(
    "v1,v2,expected",
    [
        # Equal versions
        ("1.0.0", "1.0.0", 0),
        ("2.5.3", "2.5.3", 0),
        # Equal with different lengths (implicit zeros)
        ("1.0", "1.0.0", 0),
        ("1", "1.0.0.0", 0),
        ("2.5", "2.5.0.0", 0),
        # Major version differences
        ("2.0.0", "1.0.0", 1),
        ("1.0.0", "2.0.0", -1),
        ("2.0.0", "1.99.99", 1),
        # Minor version differences
        ("1.5.0", "1.4.0", 1),
        ("1.4.0", "1.5.0", -1),
        ("1.10.0", "1.9.99", 1),
        # Patch version differences
        ("1.0.5", "1.0.4", 1),
        ("1.0.4", "1.0.5", -1),
        ("2.5.10", "2.5.9", 1),
        # Single/double digit versions
        ("2", "1", 1),
        ("1", "2", -1),
        ("1.5", "1.4", 1),
        # Extended versions
        ("1.0.0.1", "1.0.0.0", 1),
        ("1.2.3.4.5", "1.2.3.4", 1),
        # Zero versions
        ("0.0.1", "0.0.0", 1),
        ("0.1.0", "0.0.9", 1),
        ("0", "0.0.0", 0),
        # Large numbers
        ("100.200.300", "100.200.299", 1),
        ("999.999.999", "1000.0.0", -1),
    ],
)
def test_version_comparisons(v1, v2, expected):
    """Test version comparisons with various scenarios."""
    assert compare_versions(v1, v2) == expected


def test_symmetry():
    """Test anti-symmetric property: if v1 < v2, then v2 > v1."""
    test_pairs = [("1.0.0", "2.0.0"), ("1.5.3", "1.4.9"), ("1.0.0", "1.0.0")]

    for v1, v2 in test_pairs:
        result1 = compare_versions(v1, v2)
        result2 = compare_versions(v2, v1)

        if result1 == 0:
            assert result2 == 0
        else:
            assert result1 == -result2


def test_transitivity():
    """Test transitive property: if v1 < v2 < v3, then v1 < v3."""
    v1, v2, v3 = "1.0.0", "2.0.0", "3.0.0"
    assert compare_versions(v1, v2) == -1
    assert compare_versions(v2, v3) == -1
    assert compare_versions(v1, v3) == -1
