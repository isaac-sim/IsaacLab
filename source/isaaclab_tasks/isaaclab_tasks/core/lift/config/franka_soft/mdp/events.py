# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Events for the deformable lift environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def reset_deformable_over_support(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    support_offset_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    support_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> None:
    """Reset a deformable object and keep a support body underneath it.

    The deformable is displaced from its default nodal state by a sample from
    :paramref:`position_range`. The support receives the same planar displacement plus an
    independent sample from :paramref:`support_offset_range`, so it stays under the deformable
    while still varying between resets.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        position_range: Deformable displacement bounds [m] keyed by ``x``, ``y``, ``z``.
        support_offset_range: Support jitter bounds [m] keyed by ``x``, ``y``, applied on top of
            the deformable's displacement.
        asset_cfg: Scene entity of the deformable object to reset.
        support_cfg: Scene entity of the rigid support body to keep underneath.
    """
    deformable: DeformableObject = env.scene[asset_cfg.name]
    support: RigidObject = env.scene[support_cfg.name]

    # shared planar displacement, so the support tracks the deformable
    ranges = torch.tensor([position_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")], device=deformable.device)
    offset = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=deformable.device)

    nodal_state = deformable.data.default_nodal_state_w.torch[env_ids].clone()
    nodal_state[..., :3] += offset.unsqueeze(1)
    deformable.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)

    ranges = torch.tensor([support_offset_range.get(key, (0.0, 0.0)) for key in ("x", "y")], device=support.device)
    jitter = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 2), device=support.device)

    root_pose = support.data.default_root_pose.torch[env_ids].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_pose[:, :2] += offset[:, :2] + jitter
    support.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    support.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(support.data.default_root_vel.torch[env_ids]), env_ids=env_ids
    )
