# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""


from dataclasses import MISSING

from isaaclab.utils import configclass

from .multi_mesh_ray_caster import MultiMeshRayCaster
from .ray_caster_cfg import RayCasterCfg


@configclass
class MultiMeshRayCasterCfg(RayCasterCfg):
    """Configuration for the ray-cast sensor."""

    @configclass
    class RaycastTargetCfg:
        """Configuration for different ray-cast targets."""

        target_prim_expr: str = MISSING
        """The regex to specify the target prim to ray cast against."""

        is_global: bool = False
        """Whether the target prim is a global object or exists for each environment instance. Defaults to False."""

        is_shared: bool = False
        """Whether the target prim is shared across all environments. Defaults to False.
        If True, the target prim is assumed to be the same mesh in all environments. In this case, the target prim is only read once
        and the same warp mesh is used for all environments. This provides a performance boost when the target prim
        is shared across all environments.
        """

    class_type: type = MultiMeshRayCaster

    mesh_prim_paths: list[str | RaycastTargetCfg] = MISSING
    """The list of mesh primitive paths to ray cast against."""

    track_mesh_transforms: bool = False
    """Whether the meshes transformations should be tracked. Defaults to False.

    Note:
        Not tracking the mesh transformations is recommended when the meshes are static to increase performance.
    """

    merge_prim_meshes: bool = True
    """Whether to merge the meshes under each entry of :attr:`mesh_prim_paths`."""

    update_mesh_ids: bool = False
    """Whether to update the mesh ids of the ray hits in the :attr:`data` container."""
