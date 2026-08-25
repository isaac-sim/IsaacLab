# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from __future__ import annotations

"""
This file contains functions to generate, load, and process decorative elements
(for example, trees and rocks) in the scene. Assets can be downloaded from:
https://www.turbosquid.com/Search/3D-Models/low-poly/

Most objects were downloaded from these sources:
    > Trees: https://www.turbosquid.com/3d-models/3d-low-poly-trees-1431822
    > Rocks: https://www.turbosquid.com/3d-models/low-poly-stones-3d-2288912
"""
import colorsys
import os
import random
import shutil
import zipfile
from typing import TYPE_CHECKING, Literal

import numpy as np
import trimesh

from ..utils import colors, transformations, trimesh_utils
from ..utils.math import sample

if TYPE_CHECKING:
    from ..trail_cfg import TrailBaseCfg


def generate(training: bool):
    """Helper function that loads all available decorative elements, processes them, and stores them into a single glb
    file.

    Args:
        training: If true, simplifies the decorative objects.
    """
    # Selection of decorative elements with their target triangle counts
    detail_factor = 1.0 if training else 5.0
    objects = [
        # winter and summer trees
        ("trees/evergreen/pine1.obj", 300),
        ("trees/evergreen/pine2.obj", 900),
        ("trees/evergreen/pine3.obj", 600),
        ("trees/evergreen/pine4.obj", 250),
        # summer trees
        ("trees/summer/tree1.obj", 300),
        ("trees/summer/tree2.obj", 300),
        ("trees/summer/tree3.obj", 400),
        ("trees/summer/tree4.obj", 800),
        ("trees/summer/tree5.obj", 250),
        ("trees/summer/tree6.obj", 500),
        # winter trees
        ("trees/winter/tree1.obj", 250),
        # roots
        ("roots/root1.obj", 100),
        # rocks
        ("rocks/rock1.obj", 60),
        ("rocks/rock2.obj", 50),
    ]

    scene = trimesh.Scene()
    file_path = os.path.dirname(os.path.realpath(__file__))
    decoration_elements_path = os.path.join(file_path, "decoration_elements")
    decoration_elements_zip_path = os.path.join(file_path, "decoration_elements.zip")

    # Extract local asset bundle on demand when the folder is not yet present.
    if not os.path.isdir(decoration_elements_path):
        if not os.path.isfile(decoration_elements_zip_path):
            raise FileNotFoundError(
                f"Neither '{decoration_elements_path}' nor '{decoration_elements_zip_path}' exists."
            )
        with zipfile.ZipFile(decoration_elements_zip_path, "r") as zip_ref:
            zip_ref.extractall(file_path)

        if not os.path.isdir(decoration_elements_path):
            raise FileNotFoundError(
                f"Expected extracted folder '{decoration_elements_path}' was not created from "
                f"'{decoration_elements_zip_path}'."
            )

    for id, (object_name, num_target_triangles) in enumerate(objects):
        # Load object
        object_mesh = trimesh.load(os.path.join(decoration_elements_path, object_name))
        # Simplify mesh
        object_mesh = trimesh_utils.fix_mesh(object_mesh)
        object_mesh = trimesh_utils.simplify_mesh(
            mesh=object_mesh,
            method="absolute_quadric_decimation",
            parameter=round(num_target_triangles * detail_factor),
        )
        print(
            "Generate decorative object",
            object_name,
            "with",
            object_mesh.vertices.shape[0],
            "vertices",
        )
        # Add object to the scene
        scene.add_geometry(object_mesh, geom_name=object_name)
    # Store as GLB file
    scene.export(
        os.path.join(
            file_path,
            "list_of_objects_" + ("training" if training else "deployment") + ".glb",
        )
    )
    if os.path.isdir(decoration_elements_path):
        shutil.rmtree(decoration_elements_path)


def load_object_mesh(
    list_of_objects: dict[str, list[trimesh.Trimesh]],
    cfg: TrailBaseCfg,
    object_dist: dict[Literal["evergreen", "summer", "winter", "roots", "rocks"], float],
) -> trimesh.Trimesh:
    """Load a decorative object into the scene near the path center.

    Args:
        list_of_objects: Mapping from object type to lists of decorative meshes.
        cfg: Configuration for the sub-terrain.
        object_dist: Probability distribution over object types.

    Returns:
        The selected object's mesh.
    """
    # Ignore zero-probability entries and normalize active weights.
    if not object_dist:
        raise ValueError("object_dist must contain at least one entry with probability > 0.")
    choices = list(object_dist.keys())
    weights = np.array(list(object_dist.values()), dtype=float)
    weights = weights / weights.sum()
    object_type = np.random.choice(choices, p=weights)
    object_mesh = random.choice(list_of_objects[object_type]).copy()

    # Randomize scale
    scale = sample((0.4, 1.0))  # scale along all axes
    scale_factors = np.random.uniform(0.9, 1.1, 3) * scale  # scales along individual axes
    T = transformations.scale(vec=scale_factors)
    # Randomize orientation
    if object_type == "rocks":
        T_rot = trimesh.transformations.quaternion_matrix(trimesh.transformations.random_quaternion())
        T = trimesh.transformations.concatenate_matrices(T, T_rot)
    else:
        T_yaw = transformations.yaw(angle=sample((0.0, 2.0 * np.pi)))
        T = trimesh.transformations.concatenate_matrices(T, T_yaw)
    # Randomize vertex colors
    color_rgb = object_mesh.visual.vertex_colors[:, 0:3] / 255.0
    color_hsv = np.array([colorsys.rgb_to_hsv(*rgb) for rgb in color_rgb])
    color_hsv[:, 0] += np.random.uniform(-0.05, 0.05)
    color_hsv[:, 0] = color_hsv[:, 0] % 1.0
    color_hsv[:, 1] *= np.random.uniform(0.9, 1.1)
    color_hsv[:, 2] *= np.random.uniform(0.4, 1.0)
    color_hsv[:, 1] = np.clip(color_hsv[:, 1], a_min=0.0, a_max=1.0)
    object_mesh.visual.vertex_colors = colors.hsv_to_rgb(color_hsv)
    object_mesh.apply_transform(T)
    # Cut object vertices above the configured height
    if cfg.cut_objects_above is not None:
        object_mesh = trimesh_utils.cut_above(mesh=object_mesh, height=cfg.cut_objects_above)
    # Replace mesh with a convex approximation if requested
    if cfg.convex_approx:
        object_mesh_convex = object_mesh.convex_hull
        median_rgb = np.median(object_mesh.visual.vertex_colors, axis=0)[0:3]
        object_mesh_convex.visual.vertex_colors[:, 0:3] = np.tile(median_rgb, (object_mesh_convex.vertices.shape[0], 1))
        object_mesh = object_mesh_convex
    return object_mesh
