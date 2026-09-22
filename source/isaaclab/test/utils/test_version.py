# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for version comparison utilities."""

import pytest
from packaging.version import Version

from isaaclab.utils.version import compare_versions, get_isaac_sim_version


@pytest.mark.integration
def test_get_isaac_sim_version_is_cached_version_object():
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True)
    version = get_isaac_sim_version()
    assert version is get_isaac_sim_version()
    assert isinstance(version, Version)
    assert version >= Version("4.0.0")
    assert len(str(version).split(".")) >= 3


@pytest.mark.unit
@pytest.mark.parametrize(
    ("v1", "v2", "expected"),
    [
        ("1.0.0", "1.0.0", 0),
        ("1", "1.0.0.0", 0),  # implicit zeros
        ("2.0.0", "1.99.99", 1),  # major wins over minor and patch
        ("1.10.0", "1.9.99", 1),  # numeric, not lexicographic, comparison
        ("1.0.4", "1.0.5", -1),
        ("1.2.3.4.5", "1.2.3.4", 1),
        ("999.999.999", "1000.0.0", -1),
    ],
)
def test_compare_versions(v1, v2, expected):
    assert compare_versions(v1, v2) == expected
    assert compare_versions(v2, v1) == -expected  # anti-symmetry
