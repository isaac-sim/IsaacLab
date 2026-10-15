# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged observations of the granular terrain.

These are the MPM counterparts of the hybrid soft-contact terms of ``g1_29dof_soft``: the foot
contact state comes from the reaction wrench that the coupler feeds back from the MPM entry to
the proxy feet, so every term here matches its soft-contact counterpart in shape and units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ..mpm_env import G1MovingPatchEnv


def foot_contact(env: G1MovingPatchEnv) -> torch.Tensor:
    """Contact flag of each foot against the bed.

    Args:
        env: Environment instance.

    Returns:
        The flags as ``float``, shape ``(num_envs, foot_count)``.
    """
    return env.foot_contact


def foot_contact_force(env: G1MovingPatchEnv, force_filter_threshold: float = 1.0) -> torch.Tensor:
    """Log-compressed granular reaction force on each foot.

    Mirrors ``g1_29dof_soft.mdp.foot_contact_forces_hybrid``: the world-frame force is clipped,
    zeroed below the filter threshold and compressed, so the critic sees a bounded signal across
    the several decades that a foot load spans.

    Args:
        env: Environment instance.
        force_filter_threshold: Force magnitude below which a component is zeroed [N].

    Returns:
        The compressed forces, shape ``(num_envs, 3 * foot_count)``.
    """
    max_force = 1000.0
    forces = env.foot_contact_force.clamp(-max_force, max_force).reshape(env.num_envs, -1)
    forces = forces * (forces.abs() > force_filter_threshold).float()
    return torch.sign(forces) * torch.log1p(torch.abs(forces))


def foot_contact_force_raw(env: G1MovingPatchEnv, force_filter_threshold: float = 5.0) -> torch.Tensor:
    """Granular reaction force on each foot, filtered but not compressed.

    Args:
        env: Environment instance.
        force_filter_threshold: Force magnitude below which a component is zeroed [N].

    Returns:
        The forces [N], shape ``(num_envs, 3 * foot_count)``.
    """
    forces = env.foot_contact_force.reshape(env.num_envs, -1)
    return forces * (forces.abs() > force_filter_threshold).float()


def foot_air_time(env: G1MovingPatchEnv, filter_time: float = 0.5) -> torch.Tensor:
    """Time each foot has spent out of the bed, saturated to keep standing bounded.

    Args:
        env: Environment instance.
        filter_time: Air time above which the value is reported as zero [s].

    Returns:
        The air times [s], shape ``(num_envs, foot_count)``.
    """
    air_time = env.foot_air_time
    return torch.where(air_time > filter_time, 0.0, air_time)


def terrain_material_parameters(env: G1MovingPatchEnv) -> torch.Tensor:
    """Placeholder for the terrain material parameters of the soft-contact task.

    ``g1_29dof_soft`` reports the contact-model stiffness under the feet, which the MPM bed has
    no single counterpart for: its resistance emerges from the elasto-plastic particle state
    rather than from a per-foot parameter. The term is kept at the soft-contact shape so that the
    critic input layout is identical, and it is filled with zeros until a granular stiffness
    estimate is available.

    Args:
        env: Environment instance.

    Returns:
        Zeros, shape ``(num_envs, 1)``.
    """
    return torch.zeros(env.num_envs, 1, device=env.device)
    # return torch.zeros(env.num_envs, 4, device=env.device)


def foot_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: list[float] = [0.0, 0.0, -0.03539],
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_pose_w.torch[:, asset_cfg.body_ids, :7]
    pose[..., :2] = pose[..., :2] - env.scene.env_origins.unsqueeze(1)[:, :, :2]
    position = pose[..., :3]
    quat = pose[..., 3:7]

    offset_position = torch.zeros_like(position)
    offset_position[..., 0] += offset[0]
    offset_position[..., 1] += offset[1]
    offset_position[..., 2] += offset[2]
    body_position = math_utils.quat_apply(quat, offset_position) + position
    return body_position[..., 2].reshape(env.num_envs, -1)
