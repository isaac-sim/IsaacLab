# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-only direct state writes for dVRK needle pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .terminations import HandoffPhaseCfg, reset_handoff_phase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_needle_pass_to_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    phase_cfg: HandoffPhaseCfg,
    left_psm_cfg: SceneEntityCfg = SceneEntityCfg("left_psm"),
    right_psm_cfg: SceneEntityCfg = SceneEntityCfg("right_psm"),
    needle_cfg: SceneEntityCfg = SceneEntityCfg("needle"),
) -> None:
    """Perform the deterministic reset in the only permitted write order.

    Both PSMs are first written to their configured arm and jaw start states,
    with matching position targets and zero velocity.  The donor start state is
    a closed, load-qualified grasp around the needle while the receiver starts
    open.  The free needle then receives exactly one pose write and one velocity
    write.  The event does not step or settle physics and never applies an
    action.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    for asset_cfg in (left_psm_cfg, right_psm_cfg):
        psm: Articulation = env.scene[asset_cfg.name]
        joint_pos = psm.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        psm.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        psm.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        psm.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        psm.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)

    needle: RigidObject = env.scene[needle_cfg.name]
    needle_pose = needle.data.default_root_pose.torch[env_ids].clone()
    needle_pose[:, :3] += env.scene.env_origins[env_ids]
    needle.write_root_pose_to_sim_index(root_pose=needle_pose, env_ids=env_ids)
    needle.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((len(env_ids), 6), dtype=needle_pose.dtype, device=needle_pose.device),
        env_ids=env_ids,
    )
    reset_handoff_phase(env, env_ids, needle_pose[:, 2], phase_cfg)


__all__ = ["reset_needle_pass_to_default"]
