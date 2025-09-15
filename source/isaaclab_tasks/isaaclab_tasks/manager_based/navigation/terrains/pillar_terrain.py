# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Code adapted from https://github.com/leggedrobotics/nav-suite

# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from . import pillar_terrain_cfg


def pillar_terrain(
    difficulty: float, cfg: pillar_terrain_cfg.MeshPillarTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated boxes and cylinders.

    .. image:: ../../_static/terrains/mesh_pillar_terrain.jpg
       :width: 45%
       :align: center

    The terrain has a ground with a platform in the middle. The objects are randomly placed on the
    terrain s.t. they do not overlap with the platform.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """
    from .pillar_terrain_cfg import MeshPillarTerrainCfg

    # initialize list of meshes
    meshes_list = list()
    # initialize list of object meshes
    object_center_list = list()
    # constants for the terrain
    platform_clearance = 0.1
    # compute quantities
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0))
    platform_corners = np.asarray([
        [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
        [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
    ])
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # create rough terrain
    if cfg.rough_terrain:
        cfg.rough_terrain.size = cfg.size
        rough_mesh, _ = random_uniform_terrain(difficulty, cfg.rough_terrain)
        meshes_list += rough_mesh

    for object_cfg in [cfg.box_objects, cfg.cylinder_cfg]:
        # if object type is a string, get the function: make_{object_type}
        if isinstance(object_cfg.object_type, str):
            object_func = globals().get(f"make_{object_cfg.object_type}")
        else:
            object_func = object_cfg.object_type
        if not callable(object_func):
            raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

        # Resolve the terrain configuration
        # -- common parameters
        num_objects = object_cfg.num_objects[0] + int(
            difficulty * (object_cfg.num_objects[1] - object_cfg.num_objects[0])
        )
        height = object_cfg.height[0] + difficulty * (object_cfg.height[1] - object_cfg.height[0])
        # -- object specific parameters
        # note: SIM114 requires duplicated logical blocks under a single body.
        if isinstance(object_cfg, MeshPillarTerrainCfg.BoxCfg):
            object_kwargs = {
                "length": object_cfg.length[0] + difficulty * (object_cfg.length[1] - object_cfg.length[0]),
                "width": object_cfg.width[0] + difficulty * (object_cfg.width[1] - object_cfg.width[0]),
                "max_yx_angle": object_cfg.max_yx_angle[0] + difficulty * (
                    object_cfg.max_yx_angle[1] - object_cfg.max_yx_angle[0]
                ),
                "degrees": object_cfg.degrees,
            }
        elif isinstance(object_cfg, MeshPillarTerrainCfg.CylinderCfg):  # noqa: SIM114
            object_kwargs = {
                "radius": object_cfg.radius[0] + difficulty * (object_cfg.radius[1] - object_cfg.radius[0]),
                "max_yx_angle": object_cfg.max_yx_angle[0] + difficulty * (
                    object_cfg.max_yx_angle[1] - object_cfg.max_yx_angle[0]
                ),
                "degrees": object_cfg.degrees,
            }
        else:
            raise ValueError(f"Unknown terrain configuration: {cfg}")

        # sample center for objects
        while True:
            object_centers = np.zeros((num_objects, 3))
            object_centers[:, 0] = np.random.uniform(0, cfg.size[0], num_objects)
            object_centers[:, 1] = np.random.uniform(0, cfg.size[1], num_objects)
            # filter out the centers that are on the platform
            is_within_platform_x = np.logical_and(
                object_centers[:, 0] >= platform_corners[0, 0], object_centers[:, 0] <= platform_corners[1, 0]
            )
            is_within_platform_y = np.logical_and(
                object_centers[:, 1] >= platform_corners[0, 1], object_centers[:, 1] <= platform_corners[1, 1]
            )
            masks = np.logical_and(is_within_platform_x, is_within_platform_y)
            # if there are no objects on the platform, break
            if not np.any(masks):
                break

        # generate obstacles (but keep platform clean)
        for index in range(len(object_centers)):
            # randomize the height of the object
            ob_height = height + np.random.uniform(-cfg.max_height_noise, cfg.max_height_noise)
            object_centers[index, 2] = ob_height / 2
            if ob_height > 0.0:
                object_mesh = object_func(center=object_centers[index], height=ob_height, **object_kwargs)
                meshes_list.append(object_mesh)

        object_center_list.append(object_centers)

    # generate a platform in the middle
    dim = (cfg.platform_width, cfg.platform_width, 0)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0)
    platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform)

    # elevate origin
    # origin[2] = mean_height

    return meshes_list, origin
