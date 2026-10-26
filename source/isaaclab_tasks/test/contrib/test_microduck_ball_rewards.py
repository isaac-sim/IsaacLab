# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck on a ball: the foot slip reward measures sliding over the ball surface, not the motion of the points."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.microduck_ball.mdp.rewards import feet_slip

RADIUS = 0.12
SPIN = 2.0
"""Angular speed of the ball about the y axis [rad/s]."""
FOOT_Y = 0.04
"""Lateral offset of the two feet from the top of the ball [m]."""
CONTACT_HEIGHT = 0.04


def _array(value) -> SimpleNamespace:
    return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))


def _env(foot_height, foot_com_velocity, foot_ang_vel):
    """Two feet above a ball at the origin that spins about the y axis, their centers of mass off the frame."""
    origins = [[0.0, FOOT_Y, foot_height], [0.0, -FOOT_Y, foot_height]]
    com = [[0.02, y, z - 0.01] for _, y, z in origins]
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=_array([origins]),
            body_com_pos_w=_array([com]),
            body_com_lin_vel_w=_array([foot_com_velocity]),
            body_com_ang_vel_w=_array([foot_ang_vel]),
        )
    )
    ball = SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(radius=RADIUS)),
        data=SimpleNamespace(
            root_pos_w=_array([[0.0, 0.0, 0.0]]),
            root_lin_vel_w=_array([[0.0, 0.0, 0.0]]),
            root_ang_vel_w=_array([[0.0, SPIN, 0.0]]),
        ),
    )
    return SimpleNamespace(scene={"robot": robot, "ball": ball}), robot


def _slip(env):
    robot_cfg = SceneEntityCfg("robot", body_names=["left", "right"])
    robot_cfg.body_ids = [0, 1]
    return feet_slip(env, CONTACT_HEIGHT, robot_cfg, SceneEntityCfg("ball"))


@pytest.mark.unit
def test_feet_slip_is_the_foot_velocity_relative_to_the_ball_surface_at_the_contact_point():
    """A still foot slides at the surface speed, a foot turning with the surface does not, a lifted foot is ignored."""
    height = RADIUS + 0.03
    still = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    env, _ = _env(height, still, still)
    # the surface point below a foot moves at SPIN * RADIUS * sqrt(1 - n_y^2), n being the direction of the foot
    n_y = FOOT_Y / math.hypot(FOOT_Y, height)
    expected = 2 * SPIN * RADIUS * math.sqrt(1 - n_y**2)
    torch.testing.assert_close(_slip(env), torch.tensor([expected]), rtol=1e-4, atol=1e-6)

    # a foot that moves with the surface, and rotates about its own center of mass, does not slide: a velocity taken at
    # the center of mass instead of the contact point would report slip here
    foot_ang_vel = [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]]
    env, robot = _env(height, still, foot_ang_vel)
    ball = env.scene["ball"]
    contact = torch.nn.functional.normalize(robot.data.body_pos_w.torch, dim=-1) * RADIUS
    surface = torch.cross(ball.data.root_ang_vel_w.torch.expand_as(contact), contact, dim=-1)
    arm = contact - robot.data.body_com_pos_w.torch
    robot.data.body_com_lin_vel_w = SimpleNamespace(
        torch=surface - torch.cross(robot.data.body_com_ang_vel_w.torch, arm, dim=-1)
    )
    torch.testing.assert_close(_slip(env), torch.zeros(1), atol=1e-6, rtol=0.0)

    # a foot above the contact height is not on the ball, whatever its speed
    env, _ = _env(RADIUS + 0.1, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], still)
    torch.testing.assert_close(_slip(env), torch.zeros(1))
