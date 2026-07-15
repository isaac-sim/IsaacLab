# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact endpoint collision screening for independently mixed Franka Pour resets."""

from __future__ import annotations

import newton
import torch
import warp as wp

from isaaclab.utils import math as math_utils

_DISTAL_TABLE_BODIES = {
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
}
_COMPLETE_TABLE_BODIES = _DISTAL_TABLE_BODIES | {"panda_link1", "panda_link2"}


@wp.kernel(enable_backward=False)
def _mark_colliding_reset_worlds(
    contact_count: wp.array(dtype=wp.int32),
    contact_max: int,
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=wp.float32),
    contact_margin1: wp.array(dtype=wp.float32),
    shape_body: wp.array(dtype=wp.int32),
    shape_world: wp.array(dtype=wp.int32),
    shape_obstacle: wp.array(dtype=wp.int32),
    shape_table: wp.array(dtype=wp.int32),
    body_world: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_robot: wp.array(dtype=wp.int32),
    body_tests_table: wp.array(dtype=wp.int32),
    body_allows_source_contact: wp.array(dtype=wp.int32),
    shape_source: wp.array(dtype=wp.int32),
    shape_margin: wp.array(dtype=wp.float32),
    check_self_collision: int,
    penetration_tolerance: float,
    source_finger_penetration_tolerance: float,
    colliding_worlds: wp.array(dtype=wp.int32),
):
    contact_index = wp.tid()
    if contact_index >= contact_max or contact_index >= contact_count[0]:
        return

    shape0 = contact_shape0[contact_index]
    shape1 = contact_shape1[contact_index]
    obstacle0 = shape_obstacle[shape0] != 0
    obstacle1 = shape_obstacle[shape1] != 0
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    if not obstacle0 and not obstacle1:
        if check_self_collision == 0 or body0 < 0 or body1 < 0 or body0 == body1:
            return
        if body_robot[body0] == 0 or body_robot[body1] == 0:
            return
        world0 = body_world[body0]
        world1 = body_world[body1]
        if world0 < 0 or world0 != world1:
            return
        point0_w = wp.transform_point(body_q[body0], contact_point0[contact_index])
        point1_w = wp.transform_point(body_q[body1], contact_point1[contact_index])
        separation = wp.dot(contact_normal[contact_index], point1_w - point0_w)
        separation = separation - contact_margin0[contact_index] - contact_margin1[contact_index]
        if separation < -penetration_tolerance:
            wp.atomic_max(colliding_worlds, world0, 1)
        return
    if obstacle0 and obstacle1:
        return

    obstacle_shape = shape0
    robot_body = body1
    if not obstacle0:
        obstacle_shape = shape1
        robot_body = body0
    if robot_body < 0 or body_robot[robot_body] == 0:
        return
    if shape_table[obstacle_shape] != 0 and body_tests_table[robot_body] == 0:
        return

    world = body_world[robot_body]
    if world < 0 or shape_world[obstacle_shape] != world:
        return
    transform0 = wp.transform_identity()
    transform1 = wp.transform_identity()
    if body0 >= 0:
        transform0 = body_q[body0]
    if body1 >= 0:
        transform1 = body_q[body1]
    point0_w = wp.transform_point(transform0, contact_point0[contact_index])
    point1_w = wp.transform_point(transform1, contact_point1[contact_index])
    separation = wp.dot(contact_normal[contact_index], point1_w - point0_w)
    separation = separation - contact_margin0[contact_index] - contact_margin1[contact_index]
    allowed_penetration = penetration_tolerance
    if shape_source[obstacle_shape] != 0 and body_allows_source_contact[robot_body] != 0:
        # Newton's contact margins include both geometric effective radii and the shapes'
        # speculative collision margins. Keep the effective radii in the distance, but do not
        # treat an intentionally configured speculative margin as physical finger-cup overlap.
        separation = separation + shape_margin[shape0] + shape_margin[shape1]
        allowed_penetration = source_finger_penetration_tolerance
    if separation < -allowed_penetration:
        wp.atomic_max(colliding_worlds, world, 1)


def collision_free_reset_candidates(
    prototype_builder: newton.ModelBuilder,
    robot_q: torch.Tensor,
    source_positions: torch.Tensor,
    source_quaternions: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    source_box_half: tuple[float, float, float],
    target_vertices,
    target_indices,
    collider_margin: float,
    device: str,
    penetration_tolerance: float = 1.0e-4,
    allow_source_finger_contact: bool = False,
    source_finger_penetration_tolerance: float = 0.0,
    check_self_collision: bool = False,
    check_complete_robot_table: bool = False,
) -> torch.Tensor:
    """Return exact collision validity for explicit ``(robot, source, target)`` reset triples."""
    candidate_count = robot_q.shape[0]
    coordinate_count = prototype_builder.joint_coord_count
    if robot_q.shape != (candidate_count, coordinate_count):
        raise ValueError(f"robot_q must have shape (N, {coordinate_count}), got {tuple(robot_q.shape)}.")
    if source_positions.shape != (candidate_count, 3) or source_quaternions.shape != (candidate_count, 4):
        raise ValueError("Source reset poses must have shapes (N, 3) and (N, 4).")
    if target_positions.shape != (candidate_count, 3):
        raise ValueError("Target reset positions must have shape (N, 3).")
    if candidate_count == 0:
        return torch.empty(0, device=device, dtype=torch.bool)
    if penetration_tolerance < 0.0:
        raise ValueError("penetration_tolerance must be nonnegative.")
    if source_finger_penetration_tolerance < 0.0:
        raise ValueError("source_finger_penetration_tolerance must be nonnegative.")

    half_x, half_y, half_z = (float(value) for value in source_box_half)
    center_offset = torch.zeros_like(source_positions)
    center_offset[:, 2] = half_z
    source_centers = source_positions + math_utils.quat_apply(source_quaternions, center_offset)
    source_centers_cpu = source_centers.detach().cpu().tolist()
    source_quaternions_cpu = source_quaternions.detach().cpu().tolist()
    target_positions_cpu = target_positions.detach().cpu().tolist()
    target_mesh = newton.Mesh(target_vertices, target_indices, compute_inertia=False, is_solid=False)
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        margin=float(collider_margin),
        has_shape_collision=True,
        has_particle_collision=False,
        is_visible=False,
    )

    builder = newton.ModelBuilder(up_axis=prototype_builder.up_axis)
    for candidate, (source_center, source_quaternion, target_position) in enumerate(
        zip(source_centers_cpu, source_quaternions_cpu, target_positions_cpu, strict=True)
    ):
        builder.begin_world(label=f"reset_candidate_{candidate}")
        builder.add_builder(prototype_builder)
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*source_center), wp.quat(*source_quaternion)),
            hx=half_x,
            hy=half_y,
            hz=half_z,
            cfg=shape_cfg,
            label="ResetCandidate/Source",
        )
        builder.add_shape_mesh(
            -1,
            xform=wp.transform(wp.vec3(*target_position), wp.quat_identity()),
            mesh=target_mesh,
            cfg=shape_cfg,
            label="ResetCandidate/Target",
        )
        builder.add_shape_plane(
            -1,
            xform=wp.transform_identity(),
            width=0.0,
            length=0.0,
            cfg=shape_cfg,
            label="ResetCandidate/Table",
        )
        builder.end_world()

    model = builder.finalize(device=device)
    if model.world_count != candidate_count:
        raise RuntimeError(f"Expected {candidate_count} reset worlds, got {model.world_count}.")
    model_coordinate_count = model.joint_coord_count // candidate_count
    if model_coordinate_count != coordinate_count:
        raise RuntimeError(
            f"Expected {coordinate_count} robot coordinates per reset world, got {model_coordinate_count}."
        )

    body_names = [str(label).rsplit("/", 1)[-1] for label in model.body_label]
    body_robot = torch.as_tensor(
        ["/Robot/" in str(label) for label in model.body_label], device=device, dtype=torch.int32
    )
    table_body_names = _COMPLETE_TABLE_BODIES if check_complete_robot_table else _DISTAL_TABLE_BODIES
    body_tests_table = torch.as_tensor([name in table_body_names for name in body_names], device=device, dtype=torch.int32)
    body_allows_source_contact = torch.as_tensor(
        [allow_source_finger_contact and name in {"panda_leftfinger", "panda_rightfinger"} for name in body_names],
        device=device,
        dtype=torch.int32,
    )
    prototype_robot_count = sum("/Robot/" in str(label) for label in prototype_builder.body_label)
    if prototype_robot_count <= 0 or int(body_robot.sum()) != candidate_count * prototype_robot_count:
        raise RuntimeError("Reset validation did not import the expected explicit /Robot/ bodies.")
    prototype_table_body_count = sum(
        str(label).rsplit("/", 1)[-1] in table_body_names for label in prototype_builder.body_label
    )
    if prototype_table_body_count != len(table_body_names) or int(body_tests_table.sum()) != (
        candidate_count * prototype_table_body_count
    ):
        raise RuntimeError("Reset validation did not import the expected table-tested robot bodies.")
    if allow_source_finger_contact:
        prototype_finger_count = sum(
            str(label).rsplit("/", 1)[-1] in {"panda_leftfinger", "panda_rightfinger"}
            for label in prototype_builder.body_label
        )
        if prototype_finger_count != 2 or int(body_allows_source_contact.sum()) != 2 * candidate_count:
            raise RuntimeError("Reset validation did not import exactly two source-contact finger bodies per world.")

    shape_labels = [str(label) for label in model.shape_label]
    explicit_obstacle = torch.as_tensor(
        [
            label.endswith(("ResetCandidate/Source", "ResetCandidate/Target", "ResetCandidate/Table"))
            for label in shape_labels
        ],
        device=device,
        dtype=torch.int32,
    )
    prototype_table = torch.as_tensor(
        [
            check_complete_robot_table and ("/Table/" in label or label.endswith("/Table"))
            for label in shape_labels
        ],
        device=device,
        dtype=torch.int32,
    )
    shape_obstacle = torch.maximum(explicit_obstacle, prototype_table)
    shape_table = torch.as_tensor(
        [label.endswith("ResetCandidate/Table") for label in shape_labels],
        device=device,
        dtype=torch.int32,
    )
    shape_table = torch.maximum(shape_table, prototype_table)
    shape_source = torch.as_tensor(
        [label.endswith("ResetCandidate/Source") for label in shape_labels],
        device=device,
        dtype=torch.int32,
    )
    if (
        int(explicit_obstacle.sum()) != 3 * candidate_count
        or int(shape_source.sum()) != candidate_count
    ):
        raise RuntimeError("Reset validation did not build exactly one source, target, and table obstacle per world.")
    if check_complete_robot_table and int(prototype_table.sum()) == 0:
        raise RuntimeError("Complete reset validation requires the SeattleLab table collision shapes in the prototype.")

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        include_static_kinematic_pairs=False,
        soft_contact_max=0,
        verify_buffers=True,
    )
    contacts = pipeline.contacts()
    if contacts.rigid_contact_max < 3 * candidate_count:
        raise RuntimeError(
            f"Reset validation contact capacity {contacts.rigid_contact_max} is below {3 * candidate_count}."
        )
    state = model.state()
    wp.to_torch(model.joint_q).reshape(candidate_count, coordinate_count).copy_(robot_q)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline.collide(state, contacts)
    wp.synchronize_device(model.device)
    generated_contact_count = int(wp.to_torch(contacts.rigid_contact_count)[0])
    if generated_contact_count > contacts.rigid_contact_max:
        raise RuntimeError(
            f"Reset validation generated {generated_contact_count} contacts for capacity "
            f"{contacts.rigid_contact_max}."
        )
    colliding_worlds = wp.zeros(candidate_count, dtype=wp.int32, device=model.device)
    wp.launch(
        _mark_colliding_reset_worlds,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_max,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            model.shape_body,
            model.shape_world,
            wp.from_torch(shape_obstacle, dtype=wp.int32),
            wp.from_torch(shape_table, dtype=wp.int32),
            model.body_world,
            state.body_q,
            wp.from_torch(body_robot, dtype=wp.int32),
            wp.from_torch(body_tests_table, dtype=wp.int32),
            wp.from_torch(body_allows_source_contact, dtype=wp.int32),
            wp.from_torch(shape_source, dtype=wp.int32),
            model.shape_margin,
            int(check_self_collision),
            float(penetration_tolerance),
            float(source_finger_penetration_tolerance),
        ],
        outputs=[colliding_worlds],
        device=model.device,
    )
    wp.synchronize_device(model.device)
    result = (wp.to_torch(colliding_worlds) == 0).clone()
    return result
