# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton native gathers feeding the common PVA output equations."""

from functools import cache

import warp as wp

from isaaclab.sensors.pva.kernels import make_pva_writer


@cache
def make_pva_update(data_types: frozenset[str]) -> wp.Kernel:
    """Return a native gather and shared writer for the requested output groups."""
    write_velocity = "velocity" in data_types
    acceleration = "acceleration" in data_types
    gravity_enabled = "projected_gravity" in data_types
    write = make_pva_writer(write_velocity, acceleration, gravity_enabled)
    read_velocity = write_velocity or acceleration

    @wp.kernel(module="unique")
    def update(
        env_mask: wp.array(dtype=wp.bool),
        site_indices: wp.array(dtype=int),
        shape_body: wp.array(dtype=int),
        shape_transform: wp.array(dtype=wp.transformf),
        body_com: wp.array(dtype=wp.vec3f),
        gravity: wp.array(dtype=wp.vec3f),
        body_world: wp.array(dtype=wp.int32),
        body_q: wp.array(dtype=wp.transformf),
        body_qd: wp.array(dtype=wp.spatial_vectorf),
        body_qdd: wp.array(dtype=wp.spatial_vectorf),
        pose_w: wp.array(dtype=wp.transformf),
        velocity_b: wp.array(dtype=wp.spatial_vectorf),
        acceleration_b: wp.array(dtype=wp.spatial_vectorf),
        projected_gravity_b: wp.array(dtype=wp.vec3f),
    ):
        i = wp.tid()
        if not env_mask[i]:
            return
        site = site_indices[i]
        body = shape_body[site]
        vel = wp.spatial_vectorf()
        acc = wp.spatial_vectorf()
        com = wp.vec3f()
        g = wp.vec3f()
        if wp.static(read_velocity):
            vel = body_qd[body]
            com = body_com[body]
        if wp.static(acceleration):
            acc = body_qdd[body]
        if wp.static(gravity_enabled):
            g = wp.normalize(gravity[wp.max(body_world[body], 0)])
        write(
            i,
            body_q[body],
            shape_transform[site],
            com,
            vel,
            acc,
            g,
            pose_w,
            velocity_b,
            acceleration_b,
            projected_gravity_b,
        )

    return update
