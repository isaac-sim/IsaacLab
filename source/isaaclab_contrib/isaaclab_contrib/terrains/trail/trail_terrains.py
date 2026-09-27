# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import os.path
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

# python
import numpy as np
import trimesh
from shapely import minimum_bounding_circle, minimum_bounding_radius
from shapely.geometry import MultiPoint, Polygon

from .elements import decoration_functions, trail_profiles
from .utils import colors, transformations, trimesh_utils
from .utils.math import interp, interp_dict_and_sample, sample
from .utils.numpy_arrays import get_bounding_box

if TYPE_CHECKING:
    from . import trail_cfg

# global parameters
ENGINE: str = "manifold"  # geometry library (e.g. manifold, scad, blender)
EPS: float = 0.01  # smallest mesh resolution in m


def mesh_trail_segment(
    difficulty: float,
    cfg: trail_cfg.TrailBaseCfg,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Helper function to create a mesh for a trail segment with similar objects.

    The trail is constructed as follows

         y
         ^                    trail direction
         |                start -----------> end
         |
         |                   sweeping direction
         |                 P0 ------------> P1
         |
         |     +----+---------------------------+----+
     y=0 +--   | P0 | trail with objects/curves | P1 |  --- y=0 ---
         |     +----+---------------------------+----+
         |
         +-----+----------------------------------------------------> x
              x=0

    Some notes:
        * P0 indicates the starting platform and P1 the ending platform.
        * P0 and P1 are always aligned with and centered at the y-axis, even if the trail contains curves.
        * The trail is constructed along positive x axis, i.e., from P0 to P1.
        * The trail is constructed by sweeping an outline polygon along the x-axis in the positive direction,
            i.e., from P0 to P1.

    Args:
        difficulty: The difficulty parameter, 0 for easy and 1 for hard.
        cfg: The configuration of the sub-terrain.

    Returns:
           Mesh of the terrain and its origin with respect to world-frame coordinates.
    """
    # interpolate curriculum parameters
    width_limits = interp(param0=cfg.cp0.width, param1=cfg.cp1.width, x=difficulty)
    length_limits = interp(param0=cfg.cp0.length, param1=cfg.cp1.length, x=difficulty)

    # sample segment parameters
    length_between_platform_and_object = sample(cfg.length_between_platform_and_object)
    width = sample(width_limits)

    # initialize mesh and trail path
    mesh = trimesh.Trimesh()
    object_meshes: list[tuple[trimesh.Trimesh, float]] = []
    trail_path = np.array([[0.0, 0.0, 0.0]])
    object_location = np.array([length_between_platform_and_object, 0.0, 0.0])
    trail_x_under_object = None

    # bounding box of trail
    max_trail_size: tuple[float, float] = [
        cfg.size[0] - 2.0 * cfg.platform_length - 2.0 * cfg.border_width,
        cfg.size[1] - 2.0 * cfg.border_width,
    ]

    # generate sequence of objects, starting from platform P0
    while True:
        # sample object parameters
        object_length = sample(length_limits)
        length_between_objects = max(sample(cfg.length_between_objects), EPS)
        object_values = interp_dict_and_sample(cp0=cfg.cp0, cp1=cfg.cp1, x=difficulty)

        # ensure that the object fits onto the trail
        if (
            difficulty <= 0.05
            or object_location[0] + object_length + length_between_platform_and_object > max_trail_size[0]
        ):
            object_location[0] = max_trail_size[0]  # move to the end
            trail_path = np.vstack([trail_path, object_location])
            break

        # get profile polygon and sweeping path for a single object
        object = None
        path = None
        if cfg.object_function is not None:
            object = cfg.object_function(length=object_length, width=width, params=object_values, cfg=cfg)
        if cfg.sweeping_path is not None:
            path = cfg.sweeping_path(length=object_length, width=width, params=object_values, cfg=cfg)

        # add trail object
        if object is not None:
            # Generate object by sweeping the profile along the sweeping path
            if path is not None and isinstance(object, np.ndarray):
                object_3d = trimesh.creation.sweep_polygon(Polygon(object), path=path, engine=ENGINE)
            elif isinstance(object, trimesh.Trimesh):
                object_3d = object
            else:
                raise RuntimeError(
                    "If object is a NumPy array, also a sweeping path needs to be provided. Otherwise, it is assumed"
                    " that object is already a mesh."
                )

            # apply object transformation
            if cfg.object_transformation is not None:
                T1 = cfg.object_transformation(length=object_length, width=width, params=object_values, cfg=cfg)
                T2 = transformations.translation(vec=object_location)
                T = trimesh.transformations.concatenate_matrices(T1, T2)
            else:
                T = transformations.translation(vec=object_location)
            object_3d.apply_transform(T)

            # color the object
            color_mesh(
                cfg=cfg,
                mesh=object_3d,
                segment="trail_object",
            )
            # Store object mesh and its x-anchor. Twist is applied after trail
            # roll angles are known for the full path.
            object_anchor_x = object_location[0] + 0.5 * object_length
            object_meshes.append((object_3d, object_anchor_x))

        # add local terrain underneath the object
        if cfg.trail_under_object is not None:
            local_path = cfg.trail_under_object(length=object_length, width=width, params=object_values, cfg=cfg)
            local_path += object_location  # move to current object
            object_location[2] += local_path[-1, 2] - local_path[0, 2]  # adjust height
        elif cfg.extrude_trail_objects:
            s = np.linspace(0.0, 1.0, cfg.num_segments) * object_length
            local_path = np.stack([s, s * 0.0, s * 0.0], axis=1)
            local_path += object_location
        # add vertices at start and end of object
        elif object is not None:
                local_path = get_bounding_box(object_3d.vertices)
                local_path[:, 1:3] = trail_path[-1, 1:3]
        # no object available (flat segment)
        else:
            s = np.array([0.0, object_length])
            local_path = np.stack([s, s * 0.0, s * 0.0], axis=1)
            local_path += object_location
            local_path[:, 1:3] = trail_path[-1, 1:3]
        # remember x coordinates under the object
        xs = np.array([[local_path[0, 0], local_path[-1, 0]]])
        trail_x_under_object = (
            xs if trail_x_under_object is None else np.concatenate((trail_x_under_object, xs), axis=0)
        )

        # add local path only if we do not move backwards, or if the path is not located outside of the trail
        for id in range(local_path.shape[0]):
            if local_path[id, 0] - trail_path[-1, 0] >= EPS and local_path[id, 0] <= max_trail_size[0] - EPS:
                trail_path = np.vstack([trail_path, local_path[id, :]])

        # locate the coordinates of the proceeding object
        object_location[0] += object_length + length_between_objects
        object_location[1] = local_path[-1, 1]

    # add trail shape
    trail_profile, border_ids, non_trail_ids = trail_profiles.trail_profile(
        width=width,
        thickness=cfg.thickness,
        wp=cfg.wp,
    )
    trail_profile[:, 0] -= 0.5 * width  # center trail in the middle of the y axis

    # resample the path to enforce path resolution
    trail_path = down_sample_to_enforce_min_res(trail_path=trail_path, path_res=cfg.path_res)

    # add trail roll
    roll_values = interp_dict_and_sample(cp0=cfg.roll0, cp1=cfg.roll1, x=difficulty)
    trail_roll = np.zeros(trail_path.shape[0])
    if cfg.roll_functions is not None:
        for roll_function in cfg.roll_functions:
            trail_roll += roll_function(path=trail_path, params=roll_values)

    # Apply local roll to trail objects based on interpolated trail roll.
    if object_meshes:
        x_path = trail_path[:, 0]
        y_path = trail_path[:, 1]
        z_path = trail_path[:, 2]
        x_unique, unique_ids = np.unique(x_path, return_index=True)
        y_unique = y_path[unique_ids]
        z_unique = z_path[unique_ids]
        roll_unique = trail_roll[unique_ids]

        for object_3d, object_x in object_meshes:
            roll_angle = np.interp(object_x, x_unique, roll_unique)
            pivot = np.array(
                [
                    object_x,
                    np.interp(object_x, x_unique, y_unique),
                    np.interp(object_x, x_unique, z_unique),
                ]
            )
            T_roll = trimesh.transformations.rotation_matrix(
                angle=-roll_angle,
                direction=[1.0, 0.0, 0.0],
                point=pivot,
            )
            object_3d.apply_transform(T_roll)
            mesh += object_3d

    # create trail mesh by sweeping the trail profile along the trail path
    trail_mesh = trimesh.creation.sweep_polygon(
        Polygon(trail_profile),
        path=trail_path,
        angles=trail_roll,
        engine=ENGINE,
        cap=False,
    )

    # absorb objects into trail mesh
    if cfg.extrude_trail_objects and not mesh.is_empty:
        locations_ray, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=trail_mesh.vertices,
            ray_directions=np.tile([0.0, 0.0, 1.0], (len(trail_mesh.vertices), 1)),
        )
        vertices_z = trail_mesh.vertices[:, 2].copy()
        for ray_idx, hit in zip(index_ray, locations_ray):
            vertices_z[ray_idx] = max(vertices_z[ray_idx], hit[2])
        new_vertices = trail_mesh.vertices.copy()
        new_vertices[:, 2] = vertices_z
        trail_mesh = trimesh.Trimesh(vertices=new_vertices, faces=trail_mesh.faces)
        mesh = trimesh.Trimesh()  # delete objects

    # color the trail
    color_mesh(cfg=cfg, mesh=trail_mesh, segment="trail")
    if trail_x_under_object is not None:
        for id in range(trail_x_under_object.shape[0]):
            vx = trail_mesh.vertices[:, 0]
            color_ids = np.where((vx > trail_x_under_object[id, 0] + EPS) & (vx < trail_x_under_object[id, 1] - EPS))[0]
            color_mesh(
                cfg=cfg,
                mesh=trail_mesh,
                segment="trail_under_object",
                color_ids=color_ids,
            )

    # extend edges towards terrain border and color the floor
    floor_width = cfg.floor_width if (cfg.floor_width is not None) else cfg.size[1]
    mesh += process_floor(
        mesh=trail_mesh,
        delta_y=floor_width,
        profile=trail_profile,
        path=trail_path,
        border_ids=border_ids,
        non_trail_ids=non_trail_ids,
        cfg=cfg,
    )

    # add platforms P0 and P1
    mesh, origin_P0, origin_P1 = add_platforms(
        cfg=cfg,
        trail_path=trail_path,
        trail_roll=trail_roll,
        trail_profile=trail_profile,
        border_ids=border_ids,
        non_trail_ids=non_trail_ids,
        floor_width=floor_width,
        mesh=mesh,
    )
    mesh = trimesh_utils.fix_mesh(mesh)

    # adjust center (positive x is towards goal)
    origin_P0[0] += cfg.distance_start_to_trail
    origin_P1[0] -= cfg.distance_start_to_trail

    # add terrain outline (curvatures)
    # important: we ensure that all random functions are called with the same seed
    seed = random.randint(445, 1000)
    closest_idx_P0 = np.argmin(np.linalg.norm(mesh.vertices - origin_P0, axis=1))
    closest_idx_P1 = np.argmin(np.linalg.norm(mesh.vertices - origin_P1, axis=1))
    if cfg.terrain_functions is not None:
        for terrain_function in cfg.terrain_functions:
            # check if if we should skip
            if cfg.skip_terrain_functions is not None and terrain_function.func.__name__ in cfg.skip_terrain_functions:
                continue
            # set seed
            random.seed(a=seed)
            np.random.seed(seed=seed)
            # compute vertex distortion
            delta_mesh_vertices = terrain_function(vertices=mesh.vertices, difficulty=difficulty)

            # adjust platform origins using the closest mesh-vertex displacement
            origin_P0 += delta_mesh_vertices[closest_idx_P0]
            origin_P1 += delta_mesh_vertices[closest_idx_P1]

            mesh.vertices += delta_mesh_vertices
            # set seed again
            random.seed(a=seed)
            np.random.seed(seed=seed)
            # compute path distortion
            delta_vertices = terrain_function(vertices=trail_path, difficulty=difficulty)
            trail_path += delta_vertices
            seed += random.randint(1, 100)

    # add decoration
    mesh = add_decoration(
        mesh=mesh,
        width=width,
        cfg=cfg,
        trail_path=trail_path,
        max_trail_size=max_trail_size,
    )

    # reverse mesh if we want to drive down
    if cfg.ride_direction == "uphill":
        origin = origin_P0
    elif cfg.ride_direction == "downhill":
        origin = origin_P1

        # flip the mesh
        center = mesh.centroid
        T_to_center = transformations.translation(vec=-center)
        T_flip_x = transformations.identity()
        T_flip_x[0, 0] = -1.0
        T_from_center = transformations.translation(vec=center)
        T = trimesh.transformations.concatenate_matrices(T_from_center, T_flip_x, T_to_center)
        mesh.apply_transform(T)

        # flip the origin
        origin = trimesh.transformations.transform_points(np.array([origin]), T)[0]

    else:
        raise RuntimeError("Unknown ride direction.")

    # clip the y axis of the entire trail
    mesh.vertices[:, 1] = np.clip(
        mesh.vertices[:, 1],
        a_min=-0.5 * max_trail_size[1],
        a_max=0.5 * max_trail_size[1],
    )

    # move entire trail to its starting position
    trail_translation = [cfg.border_width + cfg.platform_length, 0.5 * cfg.size[1], 0.0]
    mesh.apply_translation(trail_translation)
    origin += trail_translation

    # clean up mesh
    mesh = trimesh_utils.fix_mesh(mesh)

    # debug info
    print(
        "[Trail Library] type:",
        cfg.__class__.__name__,
        ", num vertices:",
        mesh.vertices.shape[0],
        ", difficulty:",
        difficulty,
    )

    return mesh, origin


def down_sample_to_enforce_min_res(trail_path: np.ndarray, path_res: float) -> np.ndarray:
    """This function adjusts the trail path s.t.

    any two adjacent points are at least path_res meters apart from each other.
        Args:
            trail_path: parametrization of the trail path.
            path_res: resolution of the path [m].

        Returns:
            trail_path: parametrization of trail path with new resolution
    """
    knot_id = 1
    while True:
        p1 = trail_path[knot_id - 1]
        p2 = trail_path[knot_id]
        trail_heading = p2 - p1
        distance_between_knots = np.linalg.norm(trail_heading[0:2])
        if distance_between_knots > path_res:
            num_additional_segments = int(np.ceil(distance_between_knots / path_res)) + 1
            xi = np.linspace(p1[0], p2[0], num_additional_segments)
            yi = np.linspace(p1[1], p2[1], num_additional_segments)
            zi = np.linspace(p1[2], p2[2], num_additional_segments)
            si = np.stack([xi, yi, zi], axis=1)
            trail_path = np.vstack([trail_path[0:knot_id], si[1:-1], trail_path[knot_id:]])
            knot_id += num_additional_segments - 2  # remove first and last point
        knot_id += 1
        if knot_id >= trail_path.shape[0]:
            break
    return trail_path


def add_decoration(
    mesh: trimesh.Trimesh,
    width: float,
    cfg: trail_cfg.TrailBaseCfg,
    trail_path: np.ndarray,
    max_trail_size: tuple[float, float],
) -> trimesh.Trimesh:
    """This function adds decorative objects to the trail mesh.

    Warning: this function is computationally expensive.

    Args:
        mesh: mesh of the trail terrain.
        width: the width of the trail [m].
        cfg: The configuration of the sub-terrain.
        trail_path: parametrization of the trail path.
        max_trail_size: bounding box of the trail in meters.

    Returns:
        mesh of the trail terrain with decorative objects added.
    """

    def get_bounding_box_floor(mesh: trimesh.Trimesh, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # get x percent of lower vertices
        z = mesh.vertices[:, 2]
        z_min = min(z)
        z_max = max(z)
        percentile = 0.01
        floor_ids = np.nonzero(z <= z_min + percentile * (z_max - z_min))[0]
        floor_vertices = mesh.vertices[floor_ids]
        check_points = np.tile(center, (4, 1))
        check_points[0, 0] += max(floor_vertices[:, 0])
        check_points[1, 0] += min(floor_vertices[:, 0])
        check_points[2, 1] += max(floor_vertices[:, 1])
        check_points[3, 1] += min(floor_vertices[:, 1])
        return check_points, floor_ids

    decorations = trimesh.Trimesh()
    trail_path_xy = trail_path[:, 0:2]
    half_max_y = 0.5 * max_trail_size[1]
    max_x = max_trail_size[0]
    ray_directions_down = np.tile([0.0, 0.0, -1.0], (4, 1))

    if cfg.decoration_functions is not None and cfg.rel_decorated_terrains >= np.random.rand():
        # load the list of decorative elements
        decoration_path = decoration_functions.generated_path(training=cfg.training)
        if not os.path.isfile(decoration_path):
            decoration_functions.generate(training=cfg.training)
        scene = trimesh.load(decoration_path)
        list_of_objects = defaultdict(list)
        # categorize objects
        for object_name, object in scene.geometry.items():
            object_type = object_name.split("/")[0]
            if object_type == "trees":
                object_type = object_name.split("/")[1]
            list_of_objects[object_type].append(object)

        # init
        distance_between_decorations = sample(cfg.distance_between_decorations)
        distance_between_decorations_sq = distance_between_decorations * distance_between_decorations
        min_trail_to_object = cfg.wp.max_wall_width + 0.5 * width + sample(cfg.trail_border_clearance)
        min_trail_to_object_eps = max(min_trail_to_object - EPS, 0.0)
        min_trail_to_object_eps_sq = min_trail_to_object_eps * min_trail_to_object_eps
        min_trail_to_object_sq = min_trail_to_object * min_trail_to_object
        last_trail_knot = trail_path[0]
        location_buffer = np.empty((0, 3))
        if cfg.decoration_function_weights is not None:
            weights = np.array(cfg.decoration_function_weights)
            weights = weights / weights.sum()
        else:
            weights = None
        sampled_idx = np.random.choice(len(cfg.decoration_functions), p=weights)
        decoration_function = cfg.decoration_functions[sampled_idx]
        # iterate over all knots of the trail
        for trail_knot in trail_path[1:]:
            trail_heading = trail_knot - last_trail_knot
            # check if point is sufficiently far away from previous knot.
            if np.dot(trail_heading[0:2], trail_heading[0:2]) >= distance_between_decorations_sq:
                # compute trail vector
                trail_lateral = np.array([trail_heading[1], -trail_heading[0], 0.0])
                trail_lateral_norm = np.linalg.norm(trail_lateral)
                if trail_lateral_norm <= EPS:
                    continue
                trail_lateral /= trail_lateral_norm
                for layer in range(1, cfg.num_decoration_layers + 1):
                    # compute location relative to the trail path
                    for trail_sign in [-1.0, 1.0]:
                        location = trail_knot + trail_sign * trail_lateral * (
                            min_trail_to_object + (layer - 1.0) * distance_between_decorations
                        )

                        # sample decoration
                        decoration = decoration_function(
                            list_of_objects=list_of_objects,
                            cfg=cfg,
                        )

                        # conservative correction based on bounding box assumption
                        z_min = min(decoration.vertices[:, 2])
                        check_points = decoration.vertices[
                            decoration.vertices[:, 2] <= z_min + cfg.overhanging_clearance
                        ]
                        circle = minimum_bounding_circle(MultiPoint(check_points[:, 0:2]).convex_hull)
                        location[:2] += circle.centroid.coords[0]  # center
                        location += trail_sign * trail_lateral * (minimum_bounding_radius(circle))  # radius

                        # check if object is outside the terrain patch
                        if not (0.0 <= location[0] <= max_x and abs(location[1]) <= half_max_y):
                            continue

                        # check if object is too close to other objects
                        if location_buffer.shape[0] > 0:
                            delta_locations = location_buffer[:, 0:2] - location[0:2]
                            d2_locations = np.einsum("ij,ij->i", delta_locations, delta_locations)
                            if np.any(d2_locations < distance_between_decorations_sq):
                                continue

                        # check if object is too close to the trail
                        delta_to_trail = trail_path_xy - location[0:2]
                        d2_to_trail = np.einsum("ij,ij->i", delta_to_trail, delta_to_trail)
                        if np.any(d2_to_trail < min_trail_to_object_eps_sq):
                            continue

                        # check if outline of object does not touch the trail
                        check_points_xy = check_points[:, 0:2] + location[0:2]
                        distances_to_path = (
                            trail_path_xy[:, None, :] - check_points_xy[None, :, :]
                        )  # (N, 1, 2) - (1, M, 2) -> (N, M, 2)
                        d2_to_outline = np.einsum("ijk,ijk->ij", distances_to_path, distances_to_path)
                        if np.any(d2_to_outline < min_trail_to_object_sq):
                            continue
                        # project decoration onto ground
                        check_points, floor_ids = get_bounding_box_floor(mesh=decoration, center=location)
                        check_points[:, 2] += 100.0
                        intersections, _, _ = mesh.ray.intersects_location(
                            ray_origins=check_points,
                            ray_directions=ray_directions_down,
                            multiple_hits=False,
                        )
                        if intersections.size == 0:
                            continue
                        # apply transformation
                        location[2] = min(intersections[:, 2]) - 0.0
                        decoration.vertices[floor_ids, 2] -= 0.2  # conservative correction to close any gaps
                        decoration.apply_translation(location)

                        # add decoration object
                        location_buffer = np.vstack([location_buffer, location])
                        decorations += decoration
                        last_trail_knot = trail_knot
    return mesh + decorations


def process_floor(
    mesh: trimesh.Trimesh,
    delta_y: float,
    profile: np.ndarray,
    path: np.ndarray,
    border_ids: np.ndarray,
    non_trail_ids: np.ndarray,
    cfg: trail_cfg.TrailBaseCfg,
) -> trimesh.Trimesh:
    """This function post processes the mesh.

        It does so in two steps: 1. It extends vertices of the mesh on the very left and very right towards the outside.
        2. It colors the extended parts with the floor color plate.

    Args:
        mesh: mesh to be manipulated.
        delta_y: the vertices are moved by this amount along the y axis [m].
        profile: the profile used to sweep the path.
        path: path used for sweeping the profile.
        border_ids: indices indicating the left and right side of the trail profile.
        non_trail_ids: indices indicating all vertices not belonging to the trail profile.
        cfg: The configuration of the sub-terrain.

    Returns:
        mesh with extended vertices on the left and right.
    """
    if delta_y > EPS:
        # identify edge and surface indices
        num_edges = profile.shape[0]
        left_border_ids = np.zeros(path.shape[0] * 2, dtype=int)
        right_border_ids = np.zeros(path.shape[0] * 2, dtype=int)

        num_non_trail_points = len(non_trail_ids)
        color_ids = np.zeros(path.shape[0] * num_non_trail_points, dtype=int)
        for id in range(path.shape[0]):
            left_border_ids[2 * id : 2 * id + 2] = border_ids[0:2] + id * num_edges
            right_border_ids[2 * id : 2 * id + 2] = border_ids[2:4] + id * num_edges
            color_ids[num_non_trail_points * id : num_non_trail_points * id + num_non_trail_points] = (
                non_trail_ids + id * num_edges
            )

        # move border indices to the border
        mesh.vertices[left_border_ids, 1] -= delta_y
        mesh.vertices[right_border_ids, 1] += delta_y

        # sample a random floor color
        color_mesh(
            cfg=cfg,
            mesh=mesh,
            segment="floor",
            color_ids=color_ids,
        )

    return mesh


def add_platforms(
    cfg: trail_cfg.TrailBaseCfg,
    trail_path: np.ndarray,
    trail_roll: np.ndarray,
    trail_profile: np.ndarray,
    border_ids: np.ndarray,
    non_trail_ids: np.array,
    floor_width: float,
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Add two platform meshes at the start and end of the trail.

    Args:
        cfg: The configuration of the sub-terrain.
        trail_path: parametrization of the trail path.
        trail_roll: roll angles along the trail path.
        trail_profile: profile of the trail.
        border_ids: indices indicating the left and right side of the trail profile.
        non_trail_ids: indices indicating all vertices not belonging to the trail profile.
        floor_width: width of the non trail floor.
        mesh: mesh of the trail terrain and the origin of the first platform.

    Returns:
        A tuple of (mesh_with_platforms, origin_P0, origin_P1) where origin_P0 is the world-frame
        origin of the first platform and origin_P1 is the world-frame origin of the second platform.
    """
    platform_width = cfg.platform_length if (cfg.floor_width is not None) else floor_width
    origin_P0 = np.array(
        [
            trail_path[0, 0] - 0.5 * cfg.platform_length,
            trail_path[0, 1],
            trail_path[0, 2],
        ]
    )
    origin_P1 = np.array(
        [
            trail_path[-1, 0] + 0.5 * cfg.platform_length,
            trail_path[-1, 1],
            trail_path[-1, 2],
        ]
    )
    P0 = "goal" if cfg.ride_direction == "downhill" else "start"
    P1 = "start" if cfg.ride_direction == "downhill" else "goal"
    platform_defs = [
        # bottom platform
        (
            P0,
            np.vstack(
                [
                    origin_P0 - np.array([0.5 * cfg.platform_length, 0, 0]),
                    origin_P0 + np.array([0.5 * cfg.platform_length, 0, 0]),
                ]
            ),
            trail_roll[0],
        ),
        # top platform
        (
            P1,
            np.vstack(
                [
                    origin_P1 - np.array([0.5 * cfg.platform_length, 0, 0]),
                    origin_P1 + np.array([0.5 * cfg.platform_length, 0, 0]),
                ]
            ),
            trail_roll[-1],
        ),
    ]
    for segment, platform_path, platform_angle in platform_defs:
        # generate platform
        platform_path_downsampled = down_sample_to_enforce_min_res(trail_path=platform_path, path_res=cfg.path_res)
        platform_angles = np.full(platform_path_downsampled.shape[0], platform_angle)
        platform = trimesh.creation.sweep_polygon(
            Polygon(trail_profile),
            path=platform_path_downsampled,
            angles=platform_angles,
            engine=ENGINE,
            cap=False,
        )
        # color the trail within the platform
        color_mesh(cfg=cfg, mesh=platform, segment=segment)
        # process and add platform to mesh
        mesh += process_floor(
            mesh=platform,
            delta_y=0.5 * platform_width,
            profile=trail_profile,
            path=platform_path_downsampled,
            border_ids=border_ids,
            non_trail_ids=non_trail_ids,
            cfg=cfg,
        )
    return (mesh, origin_P0, origin_P1)


def color_mesh(
    cfg: trail_cfg.TrailBaseCfg,
    mesh: trimesh.Trimesh,
    segment: Literal["trail", "trail_object", "trail_under_object", "floor", "start", "goal"],
    color_ids: np.ndarray | None = None,
):
    """Color vertices of a mesh according to the configuration.

    Args:
        cfg: The configuration of the sub-terrain.
        mesh: mesh to be colored.
        segment: the name of the segment we want to color.
        color_ids: vertex indices to be colored. If None, applies to all vertices.

    Returns:
        None. The function modifies `mesh` in-place.
    """
    col_cfg = getattr(cfg, "col_" + segment)
    if col_cfg.color_mesh:
        # if no indices are provided, select all indices
        if color_ids is None:
            color_ids = np.arange(mesh.vertices.shape[0])
        size = (len(color_ids), 3)
        color_plate = col_cfg.hsv
        uniform = col_cfg.uniform
        # use hsv color plate
        if "h" in col_cfg.hsv:
            # initialize colors (hsv)
            hsv = np.ones(size)
            # sample colors
            for id, ic in enumerate("hsv"):
                if uniform or isinstance(color_plate[ic], float):
                    hsv[:, id] = sample(color_plate[ic])
                else:
                    hsv[:, id] = np.random.uniform(*color_plate[ic], size=size[0])
            # color the vertices
            mesh.visual.vertex_colors[color_ids, 0:3] = colors.hsv_to_rgb(hsv) * 255.0
        # use rgb color plate
        elif "r" in col_cfg.hsv:
            # initialize colors (rgb)
            rgb = np.ones(size)
            # sample colors
            for id, ic in enumerate("rgb"):
                if uniform or isinstance(color_plate[ic], float):
                    rgb[:, id] = sample(color_plate[ic]) * 255
                else:
                    rgb[:, id] = np.random.uniform(*color_plate[ic], size=size[0]) * 255
            # color the vertices
            mesh.visual.vertex_colors[color_ids, 0:3] = rgb
        else:
            raise RuntimeError("Color plate needs to be either HSV or RGB.")
