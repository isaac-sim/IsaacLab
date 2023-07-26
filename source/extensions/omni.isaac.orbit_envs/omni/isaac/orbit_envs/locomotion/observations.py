# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .velocity.locomotion_env import LocomotionEnv


def base_lin_vel(env: "LocomotionEnv"):
    """Base linear velocity in base frame."""
    return env.robot.data.root_lin_vel_b


def base_ang_vel(env: "LocomotionEnv"):
    """Base angular velocity in base frame."""
    return env.robot.data.root_ang_vel_b


def projected_gravity(env: "LocomotionEnv"):
    """Gravity projection on base frame."""
    return env.robot.data.projected_gravity_b


def velocity_commands(env: "LocomotionEnv"):
    """Desired base velocity for the robot."""
    return env._command_manager.command


def dof_pos(env: "LocomotionEnv"):
    """DOF positions for legs offset by the drive default positions."""
    return env.robot.data.dof_pos - env.robot.data.default_dof_pos


def dof_vel(env: "LocomotionEnv"):
    """DOF velocity of the legs."""
    return env.robot.data.dof_vel - env.robot.data.default_dof_vel


def actions(env: "LocomotionEnv"):
    """Last actions provided to env."""
    return env.actions


def height_scan(env: "LocomotionEnv", sensor_name: str):
    """Height scan around the robot."""
    sensor = env.sensors[sensor_name]
    # TODO Remove hardcoded nan to num value
    hit_height = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], posinf=-1.0)
    heights = env.robot.data.root_state_w[:, 2].unsqueeze(1) - 0.5 - hit_height
    return heights
