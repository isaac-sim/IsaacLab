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


def compute_env_offsets(
    num_envs: int, env_offset: tuple[float, float, float] = (5.0, 5.0, 0.0), up_axis: newton.AxisType = newton.Axis.Z
) -> np.ndarray:
    """Computes the positional offsets for each environment.

    The environment offsets are computed based on the number of environments and the environment spacing. Setting the
    env_offset value to (i, 0, 0) will result in a line, setting it to (i, j, 0) will result in a 2D grid and setting
    it to (i, j, k) will result in a 3D grid. Currently, there is no way to offset the position of all the environments.

    Args:
        num_envs (int): Number of environments to offset.
        env_offset (tuple[float]): The offset between each environment.
        up_axis (AxisType): The desired up-vector (should match the USD stage).

    Returns:
        np.ndarray: The positional offsets for each environment.
    """
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
        if num_dim == 1:
            for i in range(num_envs):
                env_offsets.append(i * env_offset)
        elif num_dim == 2:
            for i in range(num_envs):
                d0 = i // side_length
                d1 = i % side_length
                offset = np.zeros(3)
                offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
                offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
                env_offsets.append(offset)
        elif num_dim == 3:
            for i in range(num_envs):
                d0 = i // (side_length * side_length)
                d1 = (i // side_length) % side_length
                d2 = i % side_length
                offset = np.zeros(3)
                offset[0] = d0 * env_offset[0]
                offset[1] = d1 * env_offset[1]
                offset[2] = d2 * env_offset[2]
                env_offsets.append(offset)
        env_offsets = np.array(env_offsets)
    else:
        env_offsets = np.zeros((num_envs, 3))
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    # ensure the envs are not shifted below the ground plane
    correction[newton.Axis.from_any(up_axis)] = 0.0
    env_offsets -= correction
    return env_offsets


def replicate_environment(
    source,
    prototype_path: str,
    path_pattern: str,
    num_envs: int,
    env_spacing: tuple[float],
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
        meshes = tqdm.tqdm(prototype_builder.shape_geo_src, desc="Simplifying meshes")

        for i, m in enumerate(meshes):
            if m is None:
                continue
            hash_m = hash(m)
            if hash_m in simplified_meshes:
                prototype_builder.shape_geo_src[i] = simplified_meshes[hash_m]
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
                prototype_builder.shape_geo_src[i] = simplified
                simplified_meshes[hash_m] = simplified

    # compute the environment offsets
    env_offsets = compute_env_offsets(num_envs, env_offset=env_spacing, up_axis=up_axis)

    # clone the prototype env with updated paths
    for i in range(num_envs):
        body_start = builder.body_count
        shape_start = builder.shape_count
        joint_start = builder.joint_count
        articulation_start = builder.articulation_count

        builder.add_builder(
            prototype_builder, xform=wp.transform(env_offsets[i] + np.array(spawn_offset), wp.quat_identity())
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
