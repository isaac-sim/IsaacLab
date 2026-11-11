# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clearance criteria of the lift task's start states: terminations that are true where a state is invalid.

Both check the object's surface points and the vertices of the robot's collision meshes. They are eager terms: they
keep Warp meshes, and only the prefill of the reset bank runs them, once, before training.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from isaaclab_experimental.mdp2 import Term

import isaaclab.sim as sim_utils
from isaaclab import cloner

from .point_cloud import sample_object_point_cloud

FAR = wp.constant(1.0e6)
"""Clearance [m] of a world before any of its points is checked."""


@wp.func
def _slab_clearance_of(
    point: wp.vec3f,
    slab_top: wp.array(dtype=float),
    slab_bounds: wp.array2d(dtype=float),
    slab_bounded: wp.array2d(dtype=wp.bool),
    margin: float,
) -> float:
    """Return the height of a point, in its environment's frame, above the highest slab it is over."""
    clearance = float(FAR)
    for s in range(slab_top.shape[0]):
        over = True
        for axis in range(2):
            low = slab_bounds[s, 2 * axis] - margin
            high = slab_bounds[s, 2 * axis + 1] + margin
            if slab_bounded[s, axis] and (point[axis] < low or point[axis] > high):
                over = False
        if over:
            clearance = wp.min(clearance, point[2] - slab_top[s])
    return clearance


@wp.kernel
def _object_points_slab_clearance(
    points: wp.array2d(dtype=wp.vec3f),
    object_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    slab_top: wp.array(dtype=float),
    slab_bounds: wp.array2d(dtype=float),
    slab_bounded: wp.array2d(dtype=wp.bool),
    margin: float,
    clearance: wp.array(dtype=float),
):
    world, k = wp.tid()
    point = wp.transform_point(object_pose_w[world], points[world, k]) - env_origins[world]
    wp.atomic_min(clearance, world, _slab_clearance_of(point, slab_top, slab_bounds, slab_bounded, margin))


@wp.kernel
def _robot_vertices_slab_clearance(
    vertices: wp.array(dtype=wp.vec3f),
    vertex_body: wp.array(dtype=wp.int32),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    slab_top: wp.array(dtype=float),
    slab_bounds: wp.array2d(dtype=float),
    slab_bounded: wp.array2d(dtype=wp.bool),
    margin: float,
    clearance: wp.array(dtype=float),
):
    world, k = wp.tid()
    point = wp.transform_point(body_link_pose_w[world, vertex_body[k]], vertices[k]) - env_origins[world]
    wp.atomic_min(clearance, world, _slab_clearance_of(point, slab_top, slab_bounds, slab_bounded, margin))


@wp.func
def _mesh_distance(
    mesh: wp.uint64, center: wp.vec3f, radius: float, point: wp.vec3f, min_clearance: float, max_distance: float
) -> float:
    """Return the signed distance from a point to a mesh, both in the mesh's frame, or ``FAR`` if it is far away."""
    # The distance to the bounding sphere is a lower bound, so the points it skips cannot fail the clearance.
    lower_bound = wp.length(point - center) - radius
    if lower_bound > min_clearance and lower_bound > 0.0:
        return FAR

    query = wp.mesh_query_point_sign_winding_number(mesh, point, max_distance)
    if not query.result:
        return FAR
    return query.sign * wp.length(point - wp.mesh_eval_position(mesh, query.face, query.u, query.v))


@wp.kernel
def _object_points_robot_clearance(
    points: wp.array2d(dtype=wp.vec3f),
    object_pose_w: wp.array(dtype=wp.transformf),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    meshes: wp.array(dtype=wp.uint64),
    mesh_body: wp.array(dtype=wp.int32),
    mesh_center: wp.array(dtype=wp.vec3f),
    mesh_radius: wp.array(dtype=float),
    min_clearance: float,
    max_distance: float,
    clearance: wp.array(dtype=float),
):
    world, k = wp.tid()
    point_w = wp.transform_point(object_pose_w[world], points[world, k])
    distance = float(FAR)
    for m in range(meshes.shape[0]):
        point = wp.transform_point(wp.transform_inverse(body_link_pose_w[world, mesh_body[m]]), point_w)
        distance = wp.min(
            distance, _mesh_distance(meshes[m], mesh_center[m], mesh_radius[m], point, min_clearance, max_distance)
        )
    wp.atomic_min(clearance, world, distance)


@wp.kernel
def _robot_vertices_object_clearance(
    vertices: wp.array(dtype=wp.vec3f),
    vertex_body: wp.array(dtype=wp.int32),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    object_pose_w: wp.array(dtype=wp.transformf),
    world_mesh: wp.array(dtype=wp.int32),
    meshes: wp.array(dtype=wp.uint64),
    mesh_center: wp.array(dtype=wp.vec3f),
    mesh_radius: wp.array(dtype=float),
    min_clearance: float,
    max_distance: float,
    clearance: wp.array(dtype=float),
):
    world, k = wp.tid()
    m = world_mesh[world]
    point_w = wp.transform_point(body_link_pose_w[world, vertex_body[k]], vertices[k])
    point = wp.transform_point(wp.transform_inverse(object_pose_w[world]), point_w)
    distance = _mesh_distance(meshes[m], mesh_center[m], mesh_radius[m], point, min_clearance, max_distance)
    wp.atomic_min(clearance, world, distance)


def _collision_meshes(root_prim, frame_of) -> dict:
    """Return the enabled collision meshes under a prim, merged per owner and expressed in the owner's frame.

    ``frame_of(prim)`` returns the owner key and frame prim of a collision prim, or ``None`` to leave it out.
    """
    import trimesh  # noqa: PLC0415

    from pxr import UsdPhysics  # noqa: PLC0415

    from isaaclab.utils.mesh import (  # noqa: PLC0415
        PRIMITIVE_MESH_TYPES,
        create_trimesh_from_geom_mesh,
        create_trimesh_from_geom_shape,
    )

    def collides(prim) -> bool:
        if prim.GetTypeName() not in [*PRIMITIVE_MESH_TYPES, "Mesh"] or not prim.HasAPI(UsdPhysics.CollisionAPI):
            return False
        return UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()

    meshes: dict = {}
    for prim in sim_utils.get_all_matching_child_prims(root_prim.GetPath(), collides):
        frame = frame_of(prim)
        if frame is None:
            continue
        owner, frame_prim = frame

        is_mesh = prim.GetTypeName() == "Mesh"
        mesh = create_trimesh_from_geom_mesh(prim) if is_mesh else create_trimesh_from_geom_shape(prim)
        mesh.apply_scale(sim_utils.resolve_prim_scale(prim))
        position, quat = sim_utils.resolve_prim_pose(prim, frame_prim)
        transform = trimesh.transformations.quaternion_matrix([quat[3], quat[0], quat[1], quat[2]])
        transform[:3, 3] = position
        mesh.apply_transform(transform)
        meshes.setdefault(owner, []).append(mesh)

    return {owner: trimesh.util.concatenate(parts) for owner, parts in meshes.items()}


def _robot_meshes(robot, body_ids: torch.Tensor) -> dict:
    """Return the collision mesh of each of a robot's ``body_ids`` bodies that has one, in the body's frame."""
    body_id_of = {robot.body_names[index]: index for index in body_ids.tolist()}

    def body_frame(prim):
        # Links nest in converted assets: the nearest selected body above a collider owns it.
        node = prim.GetParent()
        while node.IsValid() and node.GetName() not in body_id_of:
            node = node.GetParent()
        return (body_id_of[node.GetName()], node) if node.IsValid() else None

    return _collision_meshes(sim_utils.find_matching_prims(robot.cfg.prim_path)[0], body_frame)


def _object_meshes(target) -> tuple[list, np.ndarray]:
    """Return the collision mesh of each object that a scene spawned, and the index of each world's mesh."""
    plan = sim_utils.SimulationContext.instance().get_clone_plan()
    source_paths = cloner.path.get_asset_prototype_paths(plan)
    stage = sim_utils.get_current_stage()

    meshes, world_mesh = [], np.zeros(target.num_instances, dtype=np.int32)
    for index in cloner.path.get_asset_prototypes(plan, target.cfg.prim_path):
        env_ids, _ = cloner.query.get_asset_prototype_unique_world_index(plan.topology, index)
        if not len(env_ids):
            continue
        prim = stage.GetPrimAtPath(source_paths[index])
        mesh = _collision_meshes(prim, lambda _, prim=prim: (0, prim))[0]
        mesh.apply_scale(np.asarray(sim_utils.resolve_prim_scale(prim), dtype=np.float32))
        world_mesh[np.asarray(env_ids, dtype=np.int64)] = len(meshes)
        meshes.append(mesh)
    return meshes, world_mesh


def _warp_meshes(meshes: list, device) -> dict:
    """Return Warp meshes that support signed distance queries, with their bounding spheres."""
    centers = [mesh.vertices.mean(axis=0) for mesh in meshes]
    radii = [np.linalg.norm(mesh.vertices - center, axis=1).max() for mesh, center in zip(meshes, centers)]
    warp_meshes = [
        wp.Mesh(
            points=wp.array(mesh.vertices, dtype=wp.vec3f, device=device),
            indices=wp.array(mesh.faces.reshape(-1), dtype=wp.int32, device=device),
            support_winding_number=True,
        )
        for mesh in meshes
    ]
    return {
        "meshes": warp_meshes,  # Keeps the meshes alive.
        "ids": wp.array([mesh.id for mesh in warp_meshes], dtype=wp.uint64, device=device),
        "centers": wp.array(np.stack(centers), dtype=wp.vec3f, device=device),
        "radii": wp.array(radii, dtype=wp.float32, device=device),
    }


def _geometry(ctx, num_object_points: int) -> dict:
    """Return the object's surface points and the robot's collision vertices, built on the first call."""
    cache = ctx.cache
    if cache:
        return cache

    robot, target = ctx.entity("body_link_pose_w"), ctx.entity("object_pose_w")
    device = robot.device
    robot_meshes = _robot_meshes(robot, ctx.inputs["body_ids"])
    points = sample_object_point_cloud(target, num_object_points)
    vertex_body = np.concatenate([np.full(len(mesh.vertices), body) for body, mesh in robot_meshes.items()])

    cache["robot_meshes"] = robot_meshes
    cache["points"] = wp.from_torch(points.contiguous(), dtype=wp.vec3f)
    cache["vertices"] = wp.array(
        np.concatenate([mesh.vertices for mesh in robot_meshes.values()]), dtype=wp.vec3f, device=device
    )
    cache["vertex_body"] = wp.array(vertex_body, dtype=wp.int32, device=device)
    cache["body_link_pose_w"] = robot.data.body_link_pose_w.warp
    cache["object_pose_w"] = target.data.root_link_pose_w.warp
    cache["clearance"] = wp.zeros(ctx.num_worlds, dtype=wp.float32, device=device)
    return cache


def _mesh_clearance(ctx, *, min_clearance, num_object_points=64):
    cache = _geometry(ctx, num_object_points)
    device = cache["clearance"].device
    if "robot" not in cache:
        cache["robot"] = _warp_meshes(list(cache["robot_meshes"].values()), device)
        cache["mesh_body"] = wp.array(list(cache["robot_meshes"]), dtype=wp.int32, device=device)
        object_meshes, world_mesh = _object_meshes(ctx.entity("object_pose_w"))
        cache["object"] = _warp_meshes(object_meshes, device)
        cache["world_mesh"] = wp.array(world_mesh, dtype=wp.int32, device=device)

    robot, target, clearance = cache["robot"], cache["object"], cache["clearance"]
    # The query horizon exceeds the clearance and plausible penetrations, so that contained points get a distance.
    limits = [min_clearance, max(4.0 * min_clearance, 0.15)]
    clearance.fill_(FAR)
    wp.launch(
        _object_points_robot_clearance,
        dim=cache["points"].shape,
        inputs=[
            cache["points"],
            cache["object_pose_w"],
            cache["body_link_pose_w"],
            robot["ids"],
            cache["mesh_body"],
            robot["centers"],
            robot["radii"],
            *limits,
        ],
        outputs=[clearance],
        device=device,
    )
    wp.launch(
        _robot_vertices_object_clearance,
        dim=(ctx.num_worlds, len(cache["vertices"])),
        inputs=[
            cache["vertices"],
            cache["vertex_body"],
            cache["body_link_pose_w"],
            cache["object_pose_w"],
            cache["world_mesh"],
            target["ids"],
            target["centers"],
            target["radii"],
            *limits,
        ],
        outputs=[clearance],
        device=device,
    )
    ctx.out.copy_(wp.to_torch(clearance) < min_clearance)


mesh_clearance = Term(eager=_mesh_clearance)
"""Whether the object and the robot's ``body_ids`` bodies come closer than ``min_clearance``, or overlap.

Reset draws can place the object inside the arm, which the solver resolves ballistically when the episode starts.
Signed distances from the object's ``num_object_points`` surface points to the robot's collision meshes, and from the
robot's collision vertices to the object's collision mesh; the sign catches full containment. Inputs:
``body_link_pose_w`` of the robot, ``body_ids``, and ``object_pose_w``.
"""


def _slab_clearance(ctx, *, obstacle_slabs, min_clearance, num_object_points=64):
    cache = _geometry(ctx, num_object_points)
    device = cache["clearance"].device
    if "slabs" not in cache:
        bounds = [[*(x or (0.0, 0.0)), *(y or (0.0, 0.0))] for x, y, _ in obstacle_slabs]
        cache["slabs"] = [
            wp.from_torch(ctx.inputs["env_origins"].contiguous(), dtype=wp.vec3f),
            wp.array([top for _, _, top in obstacle_slabs], dtype=wp.float32, device=device),
            wp.array(bounds, dtype=wp.float32, device=device),
            wp.array([[x is not None, y is not None] for x, y, _ in obstacle_slabs], dtype=wp.bool, device=device),
        ]

    clearance = cache["clearance"]
    clearance.fill_(FAR)
    wp.launch(
        _object_points_slab_clearance,
        dim=cache["points"].shape,
        inputs=[cache["points"], cache["object_pose_w"], *cache["slabs"], min_clearance],
        outputs=[clearance],
        device=device,
    )
    wp.launch(
        _robot_vertices_slab_clearance,
        dim=(ctx.num_worlds, len(cache["vertices"])),
        inputs=[cache["vertices"], cache["vertex_body"], cache["body_link_pose_w"], *cache["slabs"], min_clearance],
        outputs=[clearance],
        device=device,
    )
    ctx.out.copy_(wp.to_torch(clearance) < min_clearance)


slab_clearance = Term(eager=_slab_clearance)
"""Whether the object or the robot's ``body_ids`` bodies are less than ``min_clearance`` above a slab.

Reset draws can pose the arm into the table, and long objects through its top. ``obstacle_slabs`` are horizontal
obstacles ``(x_range, y_range, top_z)`` in the environment frame [m], with ``None`` for an unbounded range, such as
``(None, None, 0.0)`` for the ground. Inputs: ``body_link_pose_w`` of the robot, ``body_ids``, ``object_pose_w``, and
``env_origins``.
"""
