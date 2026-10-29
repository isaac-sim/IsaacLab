# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for switchable bodies built from candidate imports."""

import newton
import pytest
from isaaclab_newton.cloner.newton_clone_utils import add_switchable_body


def _candidate(label: str, is_kinematic: bool) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.add_body(label=label, mass=1.0, is_kinematic=is_kinematic)
    return builder


def test_candidates_must_agree_on_kinematic_bodies():
    """A switchable body cannot switch between kinematic and dynamic candidates."""
    builder = newton.ModelBuilder()
    candidates = [_candidate("/World/obj/table", True), _candidate("/World/obj/ball", False)]
    with pytest.raises(ValueError, match="mix kinematic and dynamic"):
        add_switchable_body(builder, "/World/obj", candidates)
    assert builder.body_count == 0
