# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the ray-cast camera sensor."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from .multi_mesh_ray_caster_camera import MultiMeshRayCasterCamera
from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from .ray_caster_camera_cfg import RayCasterCameraCfg


@configclass
class MultiMeshRayCasterCameraCfg(RayCasterCameraCfg, MultiMeshRayCasterCfg):
    """Configuration for the ray-cast sensor."""

    class_type: type = MultiMeshRayCasterCamera

    mesh_prim_paths: list[str | MultiMeshRayCasterCfg.RaycastTargetCfg] = MISSING
    """The list of mesh primitive paths to ray cast against."""

    track_mesh_transforms: bool = False
    """Whether the meshes transformations should be tracked. Defaults to False.

    Note:
        Not tracking the mesh transformations is recommended when the meshes are static to increase performance.
    """
