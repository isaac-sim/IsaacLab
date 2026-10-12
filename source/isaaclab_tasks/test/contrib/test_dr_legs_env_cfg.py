# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DR Legs task configuration."""

import math

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.parametrize(
    ("overrides", "expected_effort_limit"),
    [
        ((), math.inf),
        (("physics=newton_kamino",), math.inf),
        (("physics=isaacsim_physx",), 3.1),
        (("physics=physx",), 3.1),
    ],
)
def test_dr_legs_driven_joint_effort_limit_matches_physics_backend(
    overrides: tuple[str, ...], expected_effort_limit: float
):
    """Disable the unstable effort constraint only for the Kamino backend."""
    env_cfg = parse_env_cfg("IsaacContrib-DrLegs-Walk", overrides=overrides)
    actual_effort_limit = env_cfg.scene.robot.actuators["driven_joints"].joint_effort_limit
    assert actual_effort_limit == expected_effort_limit


@pytest.mark.parametrize(
    ("overrides", "expected_tilt_range", "expected_armature_randomization"),
    [
        ((), None, False),
        (("physics=physx",), (-0.1, 0.1), True),
    ],
)
def test_dr_legs_reset_randomization_matches_physics_backend(
    overrides: tuple[str, ...], expected_tilt_range: tuple[float, float] | None, expected_armature_randomization: bool
):
    """Disable base tilt and armature randomization only for the Kamino backend."""
    env_cfg = parse_env_cfg("IsaacContrib-DrLegs-HoldPose", overrides=overrides)
    pose_range = env_cfg.events.reset_base.params["pose_range"]
    assert pose_range.get("roll") == expected_tilt_range
    assert pose_range.get("pitch") == expected_tilt_range
    assert (env_cfg.events.randomize_joint_params is not None) is expected_armature_randomization
