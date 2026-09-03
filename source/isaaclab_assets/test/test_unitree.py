# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for predefined Unitree robot configurations."""

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


def test_go2_calves_use_reduced_joint_limits():
    """Verify the Go2 calf actuator reflects the knee reduction."""
    base_legs = UNITREE_GO2_CFG.actuators["base_legs"]
    calves = UNITREE_GO2_CFG.actuators["calves"]

    assert base_legs.joint_names_expr == [".*_hip_joint", ".*_thigh_joint"]
    assert calves.joint_names_expr == [".*_calf_joint"]
    assert calves.actuator_effort_limit == 45.05
    assert calves.saturation_effort == 45.05
    assert calves.actuator_velocity_limit == 15.65
