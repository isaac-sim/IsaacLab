# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.orbit.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import RayCaster

if TYPE_CHECKING:
    from omni.isaac.orbit.envs.base_env import BaseEnv

"""
Root state.
"""


def base_lin_vel(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


"""
Joint state.
"""


def joint_pos_rel(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def joint_vel_rel(env: BaseEnv, asset_cfg: SceneEntityCfg):
    """The joint velocities of the asset w.r.t. the default joint velocities."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel - asset.data.default_joint_vel


"""
Sensors.
"""


def height_scan(env: BaseEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # TODO (@dhoeller): is this sensor specific or we can generalize it?
    hit_points_z = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], posinf=-1.0)
    # compute the height scan: robot_z - ground_z - offset
    heights = asset.data.root_state_w[:, 2].unsqueeze(1) - hit_points_z - 0.5
    # return the height scan
    return heights


"""
Actions.
"""


def action(env: BaseEnv) -> torch.Tensor:
    """The last input action to the environment."""
    return env.action_manager.action


"""
Commands.
"""


def generated_commands(env: BaseEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return env.command_manager.command
