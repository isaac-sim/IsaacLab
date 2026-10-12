# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DR Legs task configuration."""

import math

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.dr_legs.agents.rsl_rl_ppo_cfg import DrLegsHoldPosePPORunnerCfg
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
    ("overrides", "expected_tilt_range"),
    [
        ((), None),
        (("physics=newton_kamino",), None),
        (("physics=isaacsim_physx",), (-0.1, 0.1)),
        (("physics=physx",), (-0.1, 0.1)),
    ],
)
def test_dr_legs_base_tilt_randomization_matches_physics_backend(
    overrides: tuple[str, ...], expected_tilt_range: tuple[float, float] | None
):
    """Disable unstable base tilt randomization only for the Kamino backend."""
    env_cfg = parse_env_cfg("IsaacContrib-DrLegs-HoldPose", overrides=overrides)
    pose_range = env_cfg.events.reset_base.params["pose_range"]
    assert pose_range.get("roll") == expected_tilt_range
    assert pose_range.get("pitch") == expected_tilt_range


@pytest.mark.parametrize(
    ("overrides", "expected_armature_randomization"),
    [
        ((), False),
        (("physics=newton_kamino",), False),
        (("physics=isaacsim_physx",), True),
        (("physics=physx",), True),
    ],
)
def test_dr_legs_joint_initialization_matches_physics_backend(
    overrides: tuple[str, ...], expected_armature_randomization: bool
):
    """Initialize closed-loop coordinates consistently for each physics backend."""
    env_cfg = parse_env_cfg("IsaacContrib-DrLegs-HoldPose", overrides=overrides)
    assert (env_cfg.events.randomize_joint_params is not None) is expected_armature_randomization
    assert env_cfg.events.reset_robot_joints.params["position_range"] == (0.0, 0.0)
    assert env_cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)


@pytest.mark.parametrize(
    ("overrides", "expected_action_scale"),
    [
        ((), 0.03),
        (("physics=newton_kamino",), 0.03),
        (("physics=isaacsim_physx",), 0.1),
        (("physics=physx",), 0.1),
    ],
)
def test_dr_legs_action_scale_matches_physics_backend(overrides: tuple[str, ...], expected_action_scale: float):
    """Use the stable joint-target scale for each physics backend."""
    env_cfg = parse_env_cfg("IsaacContrib-DrLegs-HoldPose", overrides=overrides)
    assert env_cfg.actions.joint_pos.scale == expected_action_scale


def test_dr_legs_policy_action_clipping():
    """Keep policy joint targets within the stable Kamino operating range."""
    runner_cfg = DrLegsHoldPosePPORunnerCfg()
    assert runner_cfg.clip_actions == 1.0
