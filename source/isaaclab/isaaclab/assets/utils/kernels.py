# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.func
def skew_symetric_matrix(v: wp.vec3f) -> wp.mat33f:
    return wp.mat33f(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)


@wp.func
def cast_to_com_frame(position: wp.vec3f, com_position: wp.vec3f, is_global: bool) -> wp.vec3f:
    if is_global:
        return position - com_position
    else:
        return position


@wp.kernel
def add_forces_and_torques_at_position(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    com_positions: wp.array2d(dtype=wp.vec3f),
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
    is_global: bool,
):
    tid_env, tid_body = wp.tid()
    if forces.shape[0] > 0:
        composed_forces_b[env_ids[tid_env], body_ids[tid_body]] += forces[tid_env, tid_body]
    if (positions.shape[0] > 0) and (forces.shape[0] > 0):
        composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += (
            skew_symetric_matrix(
                cast_to_com_frame(positions[tid_env, tid_body], com_positions[tid_env, tid_body], is_global)
            )
            @ forces[tid_env, tid_body]
        )
    elif (com_positions.shape[0] > 0) and (forces.shape[0] > 0):
        composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += (
            skew_symetric_matrix(-com_positions[tid_env, tid_body]) @ forces[tid_env, tid_body]
        )
    if torques.shape[0] > 0:
        composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += torques[tid_env, tid_body]
