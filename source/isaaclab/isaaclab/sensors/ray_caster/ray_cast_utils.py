# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for ray-cast sensors."""

from __future__ import annotations

import torch

import omni.physics.tensors.impl.api as physx

from isaaclab.sim.views import XformPrimView
from isaaclab.utils.math import convert_quat


def obtain_world_pose_from_view(
    physx_view: XformPrimView | physx.ArticulationView | physx.RigidBodyView,
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
        Orientation is in (w, x, y, z) format.

    Raises:
        NotImplementedError: If the prim view is not of the supported type.
    """
    if isinstance(physx_view, XformPrimView):
        pos_w, quat_w = physx_view.get_world_poses(env_ids)
    elif isinstance(physx_view, physx.ArticulationView):
        pos_w, quat_w = physx_view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    elif isinstance(physx_view, physx.RigidBodyView):
        pos_w, quat_w = physx_view.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    else:
        raise NotImplementedError(f"Cannot get world poses for prim view of type '{type(physx_view)}'.")

    if clone:
        return pos_w.clone(), quat_w.clone()
    else:
        return pos_w, quat_w
