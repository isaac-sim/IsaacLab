# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import tqdm
from typing import Any

import newton
import omni.log
import warp as wp


def replicate_environment(
    source,
    prototype_path: str,
    path_pattern: str,
    positions: np.ndarray,
    orientations: np.ndarray,
    up_axis: newton.AxisType = "Z",
    simplify_meshes: bool = True,
    spawn_offset: tuple[float] = (0.0, 0.0, 20.0),
    **usd_kwargs,
) -> tuple[newton.ModelBuilder, dict[str:Any]]:
    """
    Replicates a prototype USD environment in Newton.

    Args:
        source (str | pxr.UsdStage): The file path to the USD file, or an existing USD stage instance.
        prototype_path (str): The USD path where the prototype env is defined, e.g., "/World/envs/env_0".
        path_pattern (str): The USD path pattern for replicated envs, e.g., "/World/envs/env_{}".
        num_envs (int): Number of replicas to create.
        env_spacing (tuple[float]): Environment spacing vector.
        up_axis (AxisType): The desired up-vector (should match the USD stage).
        simplify_meshes (bool): If True, simplify the meshes to reduce the number of triangles. This is useful when
        meshes with complex geometry are used as collision meshes.
        spawn_offset (tuple[float]): The offset to apply to the spawned environments.
        **usd_kwargs: Keyword arguments to pass to the USD importer (see `newton.utils.parse_usd()`).

    Returns:
        (ModelBuilder, dict): The resulting ModelBuilder containing all replicated environments and a dictionary with USD stage information.
    """

    builder = newton.ModelBuilder(up_axis=up_axis)

    # first, load everything except the prototype env
    stage_info = newton.utils.parse_usd(
        source,
        builder,
        ignore_paths=[prototype_path],
        **usd_kwargs,
    )

    # up_axis sanity check
    stage_up_axis = stage_info.get("up_axis")
    if isinstance(stage_up_axis, str) and stage_up_axis.upper() != up_axis.upper():
        print(f"WARNING: up_axis '{up_axis}' does not match USD stage up_axis '{stage_up_axis}'")

    # load just the prototype env
    prototype_builder = newton.ModelBuilder(up_axis=up_axis)
    newton.utils.parse_usd(
        source,
        prototype_builder,
        root_path=prototype_path,
        **usd_kwargs,
    )

    # If enabled, simplify the meshes to reduce the number of triangles. This is useful when meshes with complex
    # geometry are used as collision meshes.
    if simplify_meshes:
        simplified_meshes = {}
        meshes = tqdm.tqdm(prototype_builder.shape_source, desc="Simplifying meshes")

        for i, m in enumerate(meshes):
            if m is None:
                continue
            hash_m = hash(m)
            if hash_m in simplified_meshes:
                prototype_builder.shape_source[i] = simplified_meshes[hash_m]
            else:
                simplified = newton.geometry.utils.remesh_mesh(
                    m, visualize=False, method="convex_hull", recompute_inertia=False
                )
                try:
                    simplified = newton.geometry.utils.remesh_mesh(
                        simplified, visualize=False, target_reduction=None, target_count=32, recompute_inertia=False
                    )
                except Exception as e:
                    omni.log.warn(f"Error simplifying mesh {i}: {e}")
                    simplified = m
                prototype_builder.shape_source[i] = simplified
                simplified_meshes[hash_m] = simplified

    # clone the prototype env with updated paths
    for i, (pos, ori) in enumerate(zip(positions, orientations)):
        body_start = builder.body_count
        shape_start = builder.shape_count
        joint_start = builder.joint_count
        articulation_start = builder.articulation_count

        builder.add_builder(
            prototype_builder, xform=wp.transform(np.array(pos) + np.array(spawn_offset), wp.quat_identity())
        )

        if i > 0:
            update_paths(
                builder,
                prototype_path,
                path_pattern.format(i),
                body_start=body_start,
                shape_start=shape_start,
                joint_start=joint_start,
                articulation_start=articulation_start,
            )

    return builder, stage_info


def update_paths(
    builder: newton.ModelBuilder,
    old_root: str,
    new_root: str,
    body_start: int | None = None,
    shape_start: int | None = None,
    joint_start: int | None = None,
    articulation_start: int | None = None,
) -> None:
    """Updates the paths of the builder to match the new root path.

    Args:
        builder (ModelBuilder): The builder to update.
        old_root (str): The old root path.
        new_root (str): The new root path.
        body_start (int): The start index of the bodies.
        shape_start (int): The start index of the shapes.
        joint_start (int): The start index of the joints.
        articulation_start (int): The start index of the articulations.
    """
    old_len = len(old_root)
    if body_start is not None:
        for i in range(body_start, builder.body_count):
            builder.body_key[i] = f"{new_root}{builder.body_key[i][old_len:]}"
    if shape_start is not None:
        for i in range(shape_start, builder.shape_count):
            builder.shape_key[i] = f"{new_root}{builder.shape_key[i][old_len:]}"
    if joint_start is not None:
        for i in range(joint_start, builder.joint_count):
            builder.joint_key[i] = f"{new_root}{builder.joint_key[i][old_len:]}"
    if articulation_start is not None:
        for i in range(articulation_start, builder.articulation_count):
            builder.articulation_key[i] = f"{new_root}{builder.articulation_key[i][old_len:]}"
