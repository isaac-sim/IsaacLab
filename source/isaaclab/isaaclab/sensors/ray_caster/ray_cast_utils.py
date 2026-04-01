# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for ray-cast sensors."""

from __future__ import annotations

import torch

from isaaclab.sim.views import BaseXformPrimView


def obtain_world_pose_from_view(
    physx_view: BaseXformPrimView,
    env_ids: torch.Tensor,
    clone: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the world poses of the prim referenced by the prim view.

    Accepts any :class:`~isaaclab.sim.views.BaseXformPrimView` subclass
    (USD/Fabric ``XformPrimView``, Newton GPU-backed ``XformPrimView``, etc.).

    Args:
        physx_view: The prim view to get the world poses from.
        env_ids: The environment ids of the prims to get the world poses for.
        clone: Whether to clone the returned tensors (default: False).

    Returns:
        A tuple containing the world positions and orientations of the prims.

    Raises:
        NotImplementedError: If the prim view is not a BaseXformPrimView subclass.
    """
    if isinstance(physx_view, BaseXformPrimView):
        pos_w, quat_w = physx_view.get_world_poses(env_ids)
    else:
        raise NotImplementedError(f"Cannot get world poses for prim view of type '{type(physx_view)}'.")

    if clone:
        return pos_w.clone(), quat_w.clone()
    else:
        return pos_w, quat_w
