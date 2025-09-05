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
        """Whether the target prim is assumed to be the same mesh across all environments. Defaults to False.

        If True, only the first mesh is read and then reused for all environments, rather than re-parsed.
        This provides a startup performance boost when there are many environments that all use the same asset. Note,
        if reference_meshes is False, this flag has no effect.
        """

        merge_prim_meshes: bool = True
        """Whether to merge the meshes under each entry of :attr:`mesh_prim_paths`."""

        track_mesh_transforms: bool = False
        """Whether the mesh transformations should be tracked. Defaults to False.

        Note:
            Not tracking the mesh transformations is recommended when the meshes are static to increase performance.
        """

    class_type: type = MultiMeshRayCaster

    mesh_prim_paths: list[str | RaycastTargetCfg] = MISSING
    """The list of mesh primitive paths to ray cast against. If an entry is a string, it is treated as
    a global :attr:`target_prim_expr` of a :class:`RaycastTargetCfg` with default values for the other fields for backwards compatibility."""

    update_mesh_ids: bool = False
    """Whether to update the mesh ids of the ray hits in the :attr:`data` container."""

    reference_meshes: bool = True
    """Whether to reference duplicated meshes instead of loading each one separately into memory.
    Defaults to True.

    When enabled, the raycaster parses all meshes in all environments, but reuses references
    for duplicates instead of storing multiple copies. This reduces memory footprint.
    """
