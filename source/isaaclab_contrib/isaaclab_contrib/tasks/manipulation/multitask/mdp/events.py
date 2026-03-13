# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-robot event terms for heterogeneous scenes.

**Per-asset functions** (use with ``per_robot=True``):
    Accept ``asset_cfg: SceneEntityCfg`` (auto-injected by the
    manager from :class:`RobotInfo`) and group-local ``env_ids``.

**Scatter-based functions** (self-dispatching):
    Iterate :attr:`EnvLayout.robot_infos` and map env-ids internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ===========================================================
# Per-asset event functions  (use with per_robot=True)
# ===========================================================


def reset_asset_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reset_joint_targets: bool = False,
) -> None:
    """Reset a single articulation to its default root and joint state.

    ``env_ids`` are group-local (0-based) when dispatched via
    ``per_robot=True``.  The function recovers global env indices
    for :attr:`env_origins` via :meth:`EnvLayout.local_to_global`.
    """
    layout = env.scene.layout
    group_key = layout.group_for_asset(asset_cfg.name)
    global_ids = layout.local_to_global(group_key, env_ids)

    art = env.scene[asset_cfg.name]
    default_pose = wp.to_torch(art.data.default_root_pose)[env_ids].clone()
    default_vel = wp.to_torch(art.data.default_root_vel)[env_ids].clone()
    default_pose[:, :3] += env.scene.env_origins[global_ids]
    art.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=env_ids)
    art.write_root_velocity_to_sim_index(root_velocity=default_vel, env_ids=env_ids)

    default_jpos = wp.to_torch(art.data.default_joint_pos)[env_ids].clone()
    default_jvel = wp.to_torch(art.data.default_joint_vel)[env_ids].clone()
    art.write_joint_position_to_sim_index(position=default_jpos, env_ids=env_ids)
    art.write_joint_velocity_to_sim_index(velocity=default_jvel, env_ids=env_ids)
    if reset_joint_targets:
        art.set_joint_position_target_index(target=default_jpos, env_ids=env_ids)
        art.set_joint_velocity_target_index(target=default_jvel, env_ids=env_ids)
