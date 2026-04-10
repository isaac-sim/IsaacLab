# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for ray-cast sensors."""

from __future__ import annotations

import torch
import warp as wp

import omni.physics.tensors.impl.api as physx

from isaaclab.sim.views import BaseXformPrimView


def obtain_world_pose_from_view(
    physx_view: BaseXformPrimView | physx.ArticulationView | physx.RigidBodyView,
    env_ids: torch.Tensor,
    clone: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the world poses of the prim referenced by the prim view.

    Args:
        physx_view: The prim view to get the world poses from.
        env_ids: The environment ids of the prims to get the world poses for.
        clone: Whether to clone the returned tensors (default: False).

    Returns:
        A tuple containing the world positions and orientations of the prims.
        Orientation is in (x, y, z, w) format.

    Raises:
        NotImplementedError: If the prim view is not of the supported type.
    """
    indices = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32) if env_ids is not None else None
    pos_wp, quat_wp = physx_view.get_world_poses(indices)
    pos_w = wp.to_torch(pos_wp)
    quat_w = wp.to_torch(quat_wp)

    if clone:
        return pos_w.clone(), quat_w.clone()
    else:
        return pos_w, quat_w
