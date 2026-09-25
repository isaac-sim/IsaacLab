# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.actuators import DCMotorCfg
from isaaclab.utils.types import ArticulationActions

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("test_point", range(20))
def test_dc_motor_clip(test_point):
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
    v - actuator_velocity_limit
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

    num_envs, num_joints, device = 2, 2, "cpu"
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    # zero gains, so the computed effort is the feed-forward torque and only the motor model clips it
    actuator_cfg = DCMotorCfg(
        joint_names_expr=joint_names,
        stiffness=0.0,
        damping=0.0,
        actuator_effort_limit=effort_lim,
        actuator_velocity_limit=velocity_limit,
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

    torque, speed = torque_speed_pairs[test_point]
    zeros = torch.zeros(num_envs, num_joints, device=device)
    control_action = ArticulationActions(
        joint_positions=zeros.clone(), joint_velocities=zeros.clone(), joint_efforts=torch.full_like(zeros, torque)
    )
    applied = actuator.compute(control_action, joint_pos=zeros, joint_vel=torch.full_like(zeros, speed))
    expected = torch.full_like(zeros, expected_clipped_effort[test_point])
    torch.testing.assert_close(actuator.applied_effort, expected)
    torch.testing.assert_close(applied.joint_efforts, expected)


def test_dc_motor_clip_with_per_joint_saturation_effort():
    """Test that a per-joint ``saturation_effort`` gives each joint its own torque-speed curve.

    Joints behind different gear reductions belong to one actuator group but do not share a stall
    torque, e.g. the Unitree Go2 calf, which sits behind an extra knee reduction.
    """
    device = "cpu"
    joint_names = ["hip", "calf"]
    actuator_cfg = DCMotorCfg(
        joint_names_expr=joint_names,
        stiffness=200.0,
        damping=10.0,
        actuator_effort_limit=100.0,
        actuator_velocity_limit=50.0,
        saturation_effort={"hip": 100.0, "calf": 190.0},
    )
    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=[0, 1],
        num_envs=1,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
    )

    # at half the no-load speed each joint delivers half of its own stall torque, and the shared
    # effort limit is high enough to clip neither
    actuator._joint_vel[:] = 25.0
    clipped_effort = actuator._clip_effort(torch.full((1, 2), 500.0, device=device))
    torch.testing.assert_close(clipped_effort, torch.tensor([[50.0, 95.0]], device=device))
