# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 observation terms.

The foot terms read a foot site: a point fixed in the foot link frame, below the ankle and under the sole.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


@lru_cache(maxsize=8)
def _offset_tensor(offset: tuple[float, float, float], device: str) -> torch.Tensor:
    return torch.tensor(offset, device=device)


def _site_offset_w(asset: Articulation, body_ids, offset: tuple[float, float, float]) -> torch.Tensor:
    """Site offset [m] of the selected bodies, rotated into the world frame."""
    quat = asset.data.body_link_quat_w.torch[:, body_ids]
    return math_utils.quat_apply(quat, _offset_tensor(tuple(offset), str(asset.device)).expand(*quat.shape[:2], 3))


def foot_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, offset: tuple[float, float, float]) -> torch.Tensor:
    """World position [m] of the foot sites, shape (num_envs, num_feet, 3).

    Args:
        env: The environment.
        asset_cfg: The robot with the foot bodies selected.
        offset: The site position [m] in the foot link frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids] + _site_offset_w(asset, asset_cfg.body_ids, offset)


def foot_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, offset: tuple[float, float, float]) -> torch.Tensor:
    """World linear velocity [m/s] of the foot sites, shape (num_envs, num_feet, 3).

    The site velocity includes the contribution of the link's angular velocity about the link origin.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids]
    ang_vel = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids]
    return lin_vel + torch.cross(ang_vel, _site_offset_w(asset, asset_cfg.body_ids, offset), dim=-1)


def foot_height(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, offset: tuple[float, float, float]) -> torch.Tensor:
    """Height [m] of the foot sites above the origin of the environment's terrain patch.

    This removes the elevation of the terrain patch but not the local surface variation inside it.
    """
    return foot_pos_w(env, asset_cfg, offset)[:, :, 2] - env.scene.env_origins[:, 2:3]


def foot_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Time [s] each selected foot has been in the air, zero while in contact."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]


def foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0) -> torch.Tensor:
    """Contact flag of each selected foot: 1 if the net contact force exceeds ``force_threshold`` [N]."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (forces.norm(dim=-1) > force_threshold).float()


def foot_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame net contact force [N] on each selected foot, flattened to (num_envs, 3 * num_feet)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].reshape(env.num_envs, -1)


def joint_pos_rel_measured(
    env: ManagerBasedEnv, action_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint positions [rad] relative to the default pose, as read through the joint-zero offsets.

    The offsets belong to the :class:`~.actions.JointZeroOffsetPositionAction` term named ``action_name``, so
    a zero action targets the pose where this observation is zero.
    """
    zero_offset = env.action_manager.get_term(action_name).joint_zero_offset
    return base_mdp.joint_pos_rel(env, asset_cfg) - zero_offset[:, asset_cfg.joint_ids]
