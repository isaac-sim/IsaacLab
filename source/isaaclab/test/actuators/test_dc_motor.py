# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

HEADLESS = True

# if not AppLauncher.instance():
simulation_app = AppLauncher(headless=HEADLESS).app

"""Rest of imports follows"""

import torch

import pytest

from isaaclab.actuators import DCMotorCfg


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dc_motor_init_minimum(num_envs, num_joints, device):
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    effort_limit = 60.0
    saturation_effort = 100.0
    velocity_limit = 50

    actuator_cfg = DCMotorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        saturation_effort=saturation_effort,
        velocity_limit=velocity_limit,
    )
    # assume Articulation class:
    #   - finds joints (names and ids) associate with the provided joint_names_expr

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
    )

    # check device and shape
    torch.testing.assert_close(actuator.computed_effort, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.applied_effort, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(
        actuator.effort_limit,
        effort_limit * torch.ones(num_envs, num_joints, device=device),
    )
    torch.testing.assert_close(
        actuator.velocity_limit, velocity_limit * torch.ones(num_envs, num_joints, device=device)
    )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("test_point", range(20))
def test_dc_motor_clip(num_envs, num_joints, device, test_point):
    r"""Test the computation of the dc motor actuator 4 quadrant torque speed curve.
    torque_speed_pairs of interest:

    0 - fully inside torque speed curve and effort limit (quadrant 1)
    1 - greater than effort limit but under torque-speed curve (quadrant 1)
    2 - greater than effort limit and outside torque-speed curve (quadrant 1)
    3 - less than effort limit but outside torque speed curve (quadrant 1)
    4 - less than effort limit but outside torque speed curve and outside corner velocity(quadrant 4)
    5 - fully inside torque speed curve and effort limit (quadrant 4)
    6 - fully outside torque speed curve and -effort limit (quadrant 4)
    7 - fully inside torque speed curve, outside -effort limit, and inside corner velocity (quadrant 4)
    8 - fully inside torque speed curves, outside -effort limit, and outside corner velocity (quadrant 4)
    9 - less than effort limit but outside torque speed curve and inside corner velocity (quadrant 4)
    e - effort_limit
    s - saturation_effort
    v - velocity_limit
    c - corner velocity
    \ - torque-speed linear boundary between v and s
    each torque_speed_point will be tested in quadrant 3 and 4
    ===========================================================
                            Torque
                             \  (+)
                               \ |
                Q2               s                   Q1
                                 | \        2
        \                        | 1 \
          c ---------------------e-----\
            \                    |       \
              \                  |  0      \ 3
                \                |           \
    (-)-----------v -------------o-------------v --------------(+) Speed
                    \            |               \   9    4
                      \          |    5            \
                        \        |                   \
                          \ -----e---------------------c
                            \    |                      \  6
                Q3            \  |              7    Q4   \
                                \s                          \
                                 |\                       8   \
                                (-) \
    ============================================================
    """
    effort_lim = 60
    saturation_effort = 100.0
    velocity_limit = 50

    torque_speed_pairs = [
        (30.0, 10.0),  # 0
        (70.0, 10.0),  # 1
        (80.0, 40.0),  # 2
        (30.0, 40.0),  # 3
        (-20.0, 90.0),  # 4
        (-30.0, 10.0),  # 5
        (-80.0, 110.0),  # 6
        (-80.0, 50.0),  # 7
        (-120.0, 90.0),  # 8
        (-10.0, 70.0),  # 9
        (-30.0, -10.0),  # -0
        (-70.0, -10.0),  # -1
        (-80.0, -40.0),  # -2
        (-30.0, -40.0),  # -3
        (20.0, -90.0),  # -4
        (30.0, -10.0),  # -5
        (80.0, -110.0),  # -6
        (80.0, -50.0),  # -7
        (120.0, -90.0),  # -8
        (10.0, -70.0),  # -9
    ]
    expected_clipped_effort = [
        30.0,  # 0
        60.0,  # 1
        20.0,  # 2
        20.0,  # 3
        -60.0,  # 4
        -30.0,  # 5
        -60.0,  # 6
        -60.0,  # 7
        -60.0,  # 8
        -40.0,  # 9
        -30.0,  # -0
        -60.0,  # -1
        -20,  # -2
        -20,  # -3
        60.0,  # -4
        30.0,  # -5
        60.0,  # -6
        60.0,  # -7
        60.0,  # -8
        40.0,  # -9
    ]

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    actuator_cfg = DCMotorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_lim,
        velocity_limit=velocity_limit,
        saturation_effort=saturation_effort,
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
    )

    ts = torque_speed_pairs[test_point]
    torque = ts[0]
    speed = ts[1]
    actuator._joint_vel[:] = speed * torch.ones(num_envs, num_joints, device=device)
    effort = torque * torch.ones(num_envs, num_joints, device=device)
    clipped_effort = actuator._clip_effort(effort)
    torch.testing.assert_close(
        expected_clipped_effort[test_point] * torch.ones(num_envs, num_joints, device=device),
        clipped_effort,
    )
