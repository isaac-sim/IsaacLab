# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import _randomize_prop_by_op

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        pos_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos



def randomize_joint_friction_model(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the friction parameters used in joint friction model. 
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if friction_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                friction = actuator.friction_static.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_static[env_ids[:, None], actuator_joint_ids] = friction

                friction = actuator.friction_dynamic.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_dynamic[env_ids[:, None], actuator_joint_ids] = friction