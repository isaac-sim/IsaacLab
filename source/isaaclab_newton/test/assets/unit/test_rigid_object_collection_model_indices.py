# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for Newton rigid-object collection model selection."""

import pytest
from isaaclab_newton.assets import RigidObjectCollection

pytestmark = pytest.mark.unit


def test_combined_pattern_preserves_common_leaf_prefix_around_body_index() -> None:
    """The collection wildcard must not admit unrelated sibling bodies."""
    pattern = RigidObjectCollection._build_combined_pattern(
        ["/World/Env_*/Object_0", "/World/Env_*/Object_1", "/World/Env_*/Object_2"]
    )

    assert pattern == "/World/Env_*/Object_*"


def test_combined_pattern_rejects_different_path_depths() -> None:
    """Body expressions at different hierarchy depths cannot form one Newton view."""
    with pytest.raises(ValueError, match="different segment counts"):
        RigidObjectCollection._build_combined_pattern(["/World/Env_*/Object_0", "/World/Env_*/Group/Object_1"])
