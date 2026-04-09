# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the ray caster sensor."""

import warp as wp

ALIGNMENT_WORLD = wp.constant(0)
ALIGNMENT_YAW = wp.constant(1)
ALIGNMENT_BASE = wp.constant(2)


@wp.func
def quat_yaw_only(
    # input
    q: wp.quatf,
) -> wp.quatf:
    """Extract yaw-only quaternion by zeroing x,y components and renormalizing."""
    z = q[2]
    w = q[3]
    length = wp.sqrt(z * z + w * w)
    if length > 0.0:
        return wp.quatf(0.0, 0.0, z / length, w / length)
    else:
        return wp.quatf(0.0, 0.0, 0.0, 1.0)


@wp.kernel(enable_backward=False)
def update_ray_caster_kernel(
    # input
    transforms: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    offset_pos: wp.array(dtype=wp.vec3f),
    offset_quat: wp.array(dtype=wp.quatf),
    drift: wp.array(dtype=wp.vec3f),
    ray_cast_drift: wp.array(dtype=wp.vec3f),
    ray_starts_local: wp.array2d(dtype=wp.vec3f),
    ray_directions_local: wp.array2d(dtype=wp.vec3f),
    alignment_mode: int,
    # output
    pos_w: wp.array(dtype=wp.vec3f),
    quat_w: wp.array(dtype=wp.quatf),
    ray_starts_w: wp.array2d(dtype=wp.vec3f),
    ray_directions_w: wp.array2d(dtype=wp.vec3f),
):
    """Compute sensor world poses and transform rays into world frame.

    Combines the PhysX view transform with the sensor offset, applies drift,
    and transforms local ray starts/directions according to the alignment mode.

    Launch with dim=(num_envs, num_rays).

    Args:
        transforms: World transforms from PhysX view. Shape is (num_envs,).
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        offset_pos: Per-env position offset [m] from view to sensor. Shape is (num_envs,).
        offset_quat: Per-env quaternion offset from view to sensor. Shape is (num_envs,).
        drift: Per-env position drift [m]. Shape is (num_envs,).
        ray_cast_drift: Per-env ray cast drift [m]. Shape is (num_envs,).
        ray_starts_local: Per-env local ray start positions [m]. Shape is (num_envs, num_rays).
        ray_directions_local: Per-env local ray directions (unit vectors). Shape is (num_envs, num_rays).
        alignment_mode: 0=world, 1=yaw, 2=base.
        pos_w: Output sensor position in world frame [m]. Shape is (num_envs,).
        quat_w: Output sensor orientation in world frame. Shape is (num_envs,).
        ray_starts_w: Output world-frame ray starts [m]. Shape is (num_envs, num_rays).
        ray_directions_w: Output world-frame ray directions (unit vectors). Shape is (num_envs, num_rays).
    """
    env_id, ray_id = wp.tid()
    if not env_mask[env_id]:
        return

    t = transforms[env_id]
    view_pos = wp.transform_get_translation(t)
    view_quat = wp.transform_get_rotation(t)

    # combine_frame_transforms: q02 = q01 * q12, t02 = t01 + quat_rotate(q01, t12)
    combined_quat = view_quat * offset_quat[env_id]
    combined_pos = view_pos + wp.quat_rotate(view_quat, offset_pos[env_id])

    combined_pos = combined_pos + drift[env_id]

    if ray_id == 0:
        pos_w[env_id] = combined_pos
        quat_w[env_id] = combined_quat

    local_start = ray_starts_local[env_id, ray_id]
    local_dir = ray_directions_local[env_id, ray_id]
    rcd = ray_cast_drift[env_id]

    if alignment_mode == ALIGNMENT_WORLD:
        pos_drifted = wp.vec3f(combined_pos[0] + rcd[0], combined_pos[1] + rcd[1], combined_pos[2])
        ray_starts_w[env_id, ray_id] = local_start + pos_drifted
        ray_directions_w[env_id, ray_id] = local_dir
    elif alignment_mode == ALIGNMENT_YAW:
        yaw_q = quat_yaw_only(combined_quat)
        rot_drift = wp.quat_rotate(yaw_q, rcd)
        pos_drifted = wp.vec3f(combined_pos[0] + rot_drift[0], combined_pos[1] + rot_drift[1], combined_pos[2])
        ray_starts_w[env_id, ray_id] = wp.quat_rotate(yaw_q, local_start) + pos_drifted
        ray_directions_w[env_id, ray_id] = local_dir
    else:
        rot_drift = wp.quat_rotate(combined_quat, rcd)
        pos_drifted = wp.vec3f(combined_pos[0] + rot_drift[0], combined_pos[1] + rot_drift[1], combined_pos[2])
        ray_starts_w[env_id, ray_id] = wp.quat_rotate(combined_quat, local_start) + pos_drifted
        ray_directions_w[env_id, ray_id] = wp.quat_rotate(combined_quat, local_dir)


@wp.kernel(enable_backward=False)
def fill_vec3_inf_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    inf_val: wp.float32,
    # output
    data: wp.array2d(dtype=wp.vec3f),
):
    """Fill a 2D vec3f array with a given value for masked environments.

    Launch with dim=(num_envs, num_rays).

    Args:
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        inf_val: Value to fill with (typically inf).
        data: Array to fill. Shape is (num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    data[env, ray] = wp.vec3f(inf_val, inf_val, inf_val)


@wp.kernel(enable_backward=False)
def raycast_mesh_masked_kernel(
    # input
    mesh: wp.uint64,
    env_mask: wp.array(dtype=wp.bool),
    ray_starts: wp.array2d(dtype=wp.vec3f),
    ray_directions: wp.array2d(dtype=wp.vec3f),
    max_dist: wp.float32,
    # output
    ray_hits: wp.array2d(dtype=wp.vec3f),
):
    """Ray-cast against a single static mesh for masked environments.

    Launch with dim=(num_envs, num_rays).

    Args:
        mesh: The warp mesh id to ray-cast against.
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        ray_starts: World-frame ray start positions [m]. Shape is (num_envs, num_rays).
        ray_directions: World-frame unit ray directions. Shape is (num_envs, num_rays).
        max_dist: Maximum ray-cast distance [m].
        ray_hits: Output ray hit positions [m]. Shape is (num_envs, num_rays).
            Pre-filled with inf for missed hits.
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    hit = wp.mesh_query_ray(mesh, ray_starts[env, ray], ray_directions[env, ray], max_dist, t, u, v, sign, n, f)
    if hit:
        ray_hits[env, ray] = ray_starts[env, ray] + t * ray_directions[env, ray]


@wp.kernel(enable_backward=False)
def apply_z_drift_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    ray_cast_drift: wp.array(dtype=wp.vec3f),
    # output
    ray_hits: wp.array2d(dtype=wp.vec3f),
):
    """Apply vertical (z) drift to ray hit positions for masked environments.

    Launch with dim=(num_envs, num_rays).

    Args:
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        ray_cast_drift: Per-env drift vector [m]; only z-component is used. Shape is (num_envs,).
        ray_hits: Ray hit positions to modify in-place. Shape is (num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    hit = ray_hits[env, ray]
    ray_hits[env, ray] = wp.vec3f(hit[0], hit[1], hit[2] + ray_cast_drift[env][2])


@wp.kernel(enable_backward=False)
def fill_float2d_masked_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    val: wp.float32,
    # output
    data: wp.array2d(dtype=wp.float32),
):
    """Fill a 2D float32 array with a given value for masked environments.

    Launch with dim=(num_envs, num_rays).

    Args:
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        val: Value to fill with.
        data: Array to fill. Shape is (num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    data[env, ray] = val


@wp.kernel(enable_backward=False)
def raycast_camera_mesh_masked_kernel(
    # input
    mesh: wp.uint64,
    env_mask: wp.array(dtype=wp.bool),
    ray_starts: wp.array2d(dtype=wp.vec3f),
    ray_directions: wp.array2d(dtype=wp.vec3f),
    max_dist: wp.float32,
    return_normal: int,
    # output
    ray_hits: wp.array2d(dtype=wp.vec3f),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normal: wp.array2d(dtype=wp.vec3f),
):
    """Ray-cast against a static mesh and return distances and normals for masked environments.

    Launch with dim=(num_envs, num_rays).

    Args:
        mesh: The warp mesh id to ray-cast against.
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        ray_starts: World-frame ray start positions [m]. Shape is (num_envs, num_rays).
        ray_directions: World-frame unit ray directions. Shape is (num_envs, num_rays).
        max_dist: Maximum ray-cast distance [m].
        return_normal: Whether to write surface normals (1) or skip (0).
        ray_hits: Output ray hit positions [m]. Shape is (num_envs, num_rays).
            Pre-filled with inf for missed hits.
        ray_distance: Output ray hit distances [m]. Shape is (num_envs, num_rays).
            Pre-filled with inf for missed hits.
        ray_normal: Output surface normals at hit positions. Shape is (num_envs, num_rays).
            Written only when return_normal is 1; pre-filled with inf for missed hits.
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3f()
    f = int(0)

    hit = wp.mesh_query_ray(mesh, ray_starts[env, ray], ray_directions[env, ray], max_dist, t, u, v, sign, n, f)
    if hit:
        ray_hits[env, ray] = ray_starts[env, ray] + t * ray_directions[env, ray]
        ray_distance[env, ray] = t
        if return_normal == 1:
            ray_normal[env, ray] = n


@wp.kernel(enable_backward=False)
def compute_distance_to_image_plane_masked_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    quat_w: wp.array(dtype=wp.quatf),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_directions_w: wp.array2d(dtype=wp.vec3f),
    # output
    distance_to_image_plane: wp.array2d(dtype=wp.float32),
):
    """Compute distance-to-image-plane from ray depth and direction for masked environments.

    The distance to the image plane is the signed projection of the hit displacement
    (``ray_distance * ray_direction_w``) onto the camera forward axis (+X in world convention).
    This equals the z-component of the hit vector in the camera frame.

    Launch with dim=(num_envs, num_rays).

    Args:
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        quat_w: Camera orientation in world frame (x, y, z, w). Shape is (num_envs,).
        ray_distance: Per-ray hit distances [m]. Shape is (num_envs, num_rays).
            Contains inf for missed rays.
        ray_directions_w: World-frame unit ray directions. Shape is (num_envs, num_rays).
        distance_to_image_plane: Output distance-to-image-plane [m]. Shape is (num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return

    depth = ray_distance[env, ray]
    dir_w = ray_directions_w[env, ray]
    # displacement vector in world frame
    disp_w = wp.vec3f(depth * dir_w[0], depth * dir_w[1], depth * dir_w[2])
    # rotate into camera frame (quat_rotate_inv applies q^-1 * v * q)
    disp_cam = wp.quat_rotate_inv(quat_w[env], disp_w)
    # x-component is the forward (depth) axis of the camera in world convention
    distance_to_image_plane[env, ray] = disp_cam[0]


@wp.kernel(enable_backward=False)
def apply_depth_clipping_max_masked_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    max_dist: wp.float32,
    # output
    depth: wp.array2d(dtype=wp.float32),
):
    """Clip depth values to max_dist (replacing values above max_dist and NaN with max_dist).

    Launch with dim=(num_envs, num_rays).

    Args:
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        max_dist: Maximum depth value [m]. Values above this are set to max_dist.
        depth: Depth buffer to clip in-place. Shape is (num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    val = depth[env, ray]
    if val > max_dist or wp.isnan(val):
        depth[env, ray] = max_dist


@wp.kernel(enable_backward=False)
def apply_depth_clipping_zero_masked_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    max_dist: wp.float32,
    # output
    depth: wp.array2d(dtype=wp.float32),
):
    """Clip depth values to zero for values exceeding max_dist or NaN.

    Launch with dim=(num_envs, num_rays).

    Args:
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        max_dist: Maximum depth threshold [m]. Values above this are set to zero.
        depth: Depth buffer to clip in-place. Shape is (num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    val = depth[env, ray]
    if val > max_dist or wp.isnan(val):
        depth[env, ray] = wp.float32(0.0)
