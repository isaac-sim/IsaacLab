# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared PVA equations and cached specialization of optional outputs."""

from functools import cache

import warp as wp

from ...utils.warp.math import acceleration_at_offset, velocity_at_offset


@cache
def make_pva_writer(write_velocity: bool, write_acceleration: bool, write_gravity: bool):
    """Return the shared device writer for the selected output groups."""

    @wp.func
    def write(
        i: int,
        body_pose: wp.transformf,
        offset: wp.transformf,
        com: wp.vec3f,
        body_velocity: wp.spatial_vectorf,
        body_acceleration: wp.spatial_vectorf,
        gravity_w: wp.vec3f,
        pose_w: wp.array(dtype=wp.transformf),
        velocity_b: wp.array(dtype=wp.spatial_vectorf),
        acceleration_b: wp.array(dtype=wp.spatial_vectorf),
        projected_gravity_b: wp.array(dtype=wp.vec3f),
    ):
        pose = body_pose * offset
        quat = wp.transform_get_rotation(pose)
        pose_w[i] = pose
        if wp.static(write_velocity or write_acceleration):
            arm = wp.quat_rotate(wp.transform_get_rotation(body_pose), wp.transform_get_translation(offset) - com)
        if wp.static(write_velocity):
            linear = wp.quat_rotate_inv(quat, velocity_at_offset(body_velocity, arm))
            angular = wp.quat_rotate_inv(quat, wp.spatial_bottom(body_velocity))
            velocity_b[i] = wp.spatial_vectorf(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])
        if wp.static(write_acceleration):
            linear = wp.quat_rotate_inv(
                quat, acceleration_at_offset(body_acceleration, wp.spatial_bottom(body_velocity), arm)
            )
            angular = wp.quat_rotate_inv(quat, wp.spatial_bottom(body_acceleration))
            acceleration_b[i] = wp.spatial_vectorf(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])
        if wp.static(write_gravity):
            projected_gravity_b[i] = wp.quat_rotate_inv(quat, gravity_w)

    return write


@cache
def make_pva_update(data_types: frozenset[str]) -> wp.Kernel:
    """Return an update kernel for only the requested output groups."""
    write_velocity = "velocity" in data_types
    acceleration = "acceleration" in data_types
    write = make_pva_writer(write_velocity, acceleration, "projected_gravity" in data_types)
    read_velocity = write_velocity or acceleration

    @wp.kernel(module="unique")
    def update(
        env_mask: wp.array(dtype=wp.bool),
        transforms: wp.array(dtype=wp.transformf),
        velocities: wp.array(dtype=wp.spatial_vectorf),
        accelerations: wp.array(dtype=wp.spatial_vectorf),
        coms: wp.array(dtype=wp.transformf),
        offset_pos_b: wp.array(dtype=wp.vec3f),
        offset_quat_b: wp.array(dtype=wp.quatf),
        gravity_vec_w: wp.vec3f,
        pose_w: wp.array(dtype=wp.transformf),
        velocity_b: wp.array(dtype=wp.spatial_vectorf),
        acceleration_b: wp.array(dtype=wp.spatial_vectorf),
        projected_gravity_b: wp.array(dtype=wp.vec3f),
    ):
        i = wp.tid()
        if not env_mask[i]:
            return
        vel = wp.spatial_vectorf()
        acc = wp.spatial_vectorf()
        com = wp.vec3f()
        if wp.static(read_velocity):
            vel = velocities[i]
            com = wp.transform_get_translation(coms[i])
        if wp.static(acceleration):
            acc = accelerations[i]
        write(
            i,
            transforms[i],
            wp.transformf(offset_pos_b[i], offset_quat_b[i]),
            com,
            vel,
            acc,
            gravity_vec_w,
            pose_w,
            velocity_b,
            acceleration_b,
            projected_gravity_b,
        )

    return update


@wp.kernel
def pva_reset_kernel(
    env_mask: wp.array(dtype=wp.bool),
    pose_w: wp.array(dtype=wp.transformf),
    velocity_b: wp.array(dtype=wp.spatial_vectorf),
    acceleration_b: wp.array(dtype=wp.spatial_vectorf),
    projected_gravity_b: wp.array(dtype=wp.vec3f),
):
    """Reset enabled measurements without touching unselected environments."""
    i = wp.tid()
    if not env_mask[i]:
        return
    pose_w[i] = wp.transform_identity()
    if velocity_b:
        velocity_b[i] = wp.spatial_vectorf()
    if acceleration_b:
        acceleration_b[i] = wp.spatial_vectorf()
    if projected_gravity_b:
        projected_gravity_b[i] = wp.vec3f(0.0, 0.0, -1.0)
