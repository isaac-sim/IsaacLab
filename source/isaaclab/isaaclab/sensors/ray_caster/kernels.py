# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the ray caster sensor."""

import warp as wp

ALIGNMENT_WORLD = wp.constant(0)
ALIGNMENT_YAW = wp.constant(1)
ALIGNMENT_BASE = wp.constant(2)

# Upper-bound ray-cast distance [m] used by camera classes. The actual depth-clipping is applied
# as a post-process step per data type, so the kernel is always given a large budget.
CAMERA_RAYCAST_MAX_DIST: float = 1e6


@wp.func
def quat_yaw_only(
    # input
    q: wp.quatf,
) -> wp.quatf:
    """Extract the yaw-only quaternion from a general quaternion.

    Equivalent to :func:`isaaclab.utils.math.yaw_quat`: extracts the yaw angle via
    ``atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))`` and returns a pure-yaw quaternion
    ``(0, 0, sin(yaw/2), cos(yaw/2))``. This is correct for all orientations, including
    those with non-zero roll and pitch.
    """
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    half_yaw = yaw * 0.5
    return wp.quatf(0.0, 0.0, wp.sin(half_yaw), wp.cos(half_yaw))


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
            After rotation by the alignment quaternion, only the x and y components
            are applied to the ray start position; the z component of the sensor
            position is preserved.
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
        # Ray DIRECTIONS are intentionally NOT rotated in yaw mode: the sensor's ray pattern
        # (e.g. straight-down (0,0,-1) for a height scanner) stays fixed in world frame.
        # Only ray STARTS are rotated by the yaw-only quaternion so the scan footprint
        # follows the body heading without tilting when the body pitches or rolls.
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
def fill_ray_hits_distance_inf_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    fill_normals: bool,
    # output
    ray_hits: wp.array2d(dtype=wp.vec3f),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normals: wp.array2d(dtype=wp.vec3f),
):
    """Fill ray hit, distance, and optionally normal buffers with infinity for masked environments.

    Launch with dim=(num_envs, num_rays).

    Args:
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        fill_normals: Whether to fill ``ray_normals``.
        ray_hits: Ray hit positions to fill with ``wp.inf``. Shape is (num_envs, num_rays).
        ray_distance: Ray distances to fill with ``wp.inf``. Shape is (num_envs, num_rays).
        ray_normals: Ray normals to fill with ``wp.inf`` when requested. Shape is (num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    inf_vec = wp.vec3f(wp.inf, wp.inf, wp.inf)
    ray_hits[env, ray] = inf_vec
    ray_distance[env, ray] = wp.inf
    if fill_normals:
        ray_normals[env, ray] = inf_vec


@wp.kernel(enable_backward=False)
def update_frame_masked_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    frame_op: int,
    # output
    frame: wp.array(dtype=wp.int64),
):
    """Update frame counters for masked environments.

    ``frame_op`` uses 1 for increment and 2 for reset.
    """
    env = wp.tid()
    if not env_mask[env]:
        return
    if frame_op == 1:
        frame[env] = frame[env] + wp.int64(1)
    elif frame_op == 2:
        frame[env] = wp.int64(0)


@wp.kernel(enable_backward=False)
def update_camera_offsets_kernel(
    # input
    transforms: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    target_positions: wp.array(dtype=wp.vec3f),
    target_quats: wp.array(dtype=wp.quatf),
    use_env_ids: bool,
    update_position: bool,
    update_orientation: bool,
    # output
    offset_pos: wp.array(dtype=wp.vec3f),
    offset_quat: wp.array(dtype=wp.quatf),
):
    """Update camera-frame offsets from target world poses.

    Launch with ``dim=count`` where ``count`` is either the number of selected
    environments or all environments. ``target_positions`` and ``target_quats``
    are compact arrays indexed by the launch id.
    """
    src_id = wp.tid()
    env_id = src_id
    if use_env_ids:
        env_id = env_ids[src_id]

    view_transform = transforms[env_id]
    view_pos = wp.transform_get_translation(view_transform)
    view_quat = wp.transform_get_rotation(view_transform)

    if update_position:
        offset_pos[env_id] = wp.quat_rotate_inv(view_quat, target_positions[src_id] - view_pos)
    if update_orientation:
        offset_quat[env_id] = wp.quat_inverse(view_quat) * target_quats[src_id]


@wp.kernel(enable_backward=False)
def copy_float2d_to_image1_depth_clipped_masked_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    src: wp.array2d(dtype=wp.float32),
    width: int,
    clip_depth: bool,
    max_dist: wp.float32,
    fill_val: wp.float32,
    # output
    dst: wp.array4d(dtype=wp.float32),
):
    """Copy a flat float buffer to ``(N, H, W, 1)`` camera output with optional depth clipping."""
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    value = src[env, ray]
    if clip_depth and (value > max_dist or wp.isnan(value)):
        value = fill_val
    row = ray // width
    col = ray - row * width
    dst[env, row, col, 0] = value


@wp.kernel(enable_backward=False)
def copy_vec3_2d_to_image3_masked_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    src: wp.array2d(dtype=wp.vec3f),
    width: int,
    # output
    dst: wp.array4d(dtype=wp.float32),
):
    """Copy a flat per-ray vec3 buffer to ``(N, H, W, 3)`` camera output."""
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    row = ray // width
    col = ray - row * width
    value = src[env, ray]
    dst[env, row, col, 0] = value[0]
    dst[env, row, col, 1] = value[1]
    dst[env, row, col, 2] = value[2]


@wp.kernel(enable_backward=False)
def copy_int16_2d_to_image1_masked_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    src: wp.array2d(dtype=wp.int16),
    width: int,
    # output
    dst: wp.array4d(dtype=wp.int16),
):
    """Copy a flat per-ray int16 buffer to ``(N, H, W, 1)`` camera output."""
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    row = ray // width
    col = ray - row * width
    dst[env, row, col, 0] = src[env, ray]


@wp.kernel(enable_backward=False)
def copy_mesh_poses_to_table_kernel(
    # input
    positions_src: wp.array(dtype=wp.vec3f),
    orientations_src: wp.array(dtype=wp.quatf),
    meshes_per_env: int,
    mesh_offset: int,
    broadcast_single_source: bool,
    # output
    positions_dst: wp.array2d(dtype=wp.vec3f),
    orientations_dst: wp.array2d(dtype=wp.quatf),
):
    """Copy flat tracked-mesh poses into the rectangular per-env mesh table."""
    env, local_mesh = wp.tid()
    src_index = local_mesh
    if not broadcast_single_source:
        src_index = env * meshes_per_env + local_mesh
    dst_index = mesh_offset + local_mesh
    positions_dst[env, dst_index] = positions_src[src_index]
    orientations_dst[env, dst_index] = orientations_src[src_index]


@wp.kernel(enable_backward=False)
def copy_mesh_transforms_to_table_kernel(
    # input
    transforms_src: wp.array(dtype=wp.transformf),
    meshes_per_env: int,
    mesh_offset: int,
    broadcast_single_source: bool,
    # output
    positions_dst: wp.array2d(dtype=wp.vec3f),
    orientations_dst: wp.array2d(dtype=wp.quatf),
):
    """Copy flat tracked-mesh transforms into the rectangular per-env mesh table."""
    env, local_mesh = wp.tid()
    src_index = local_mesh
    if not broadcast_single_source:
        src_index = env * meshes_per_env + local_mesh
    dst_index = mesh_offset + local_mesh
    xform = transforms_src[src_index]
    positions_dst[env, dst_index] = wp.transform_get_translation(xform)
    orientations_dst[env, dst_index] = wp.transform_get_rotation(xform)


@wp.kernel(enable_backward=False)
def compute_distance_to_image_plane_to_image_masked_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    quat_w: wp.array(dtype=wp.quatf),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_directions_w: wp.array2d(dtype=wp.vec3f),
    width: int,
    clip_depth: bool,
    max_dist: wp.float32,
    fill_val: wp.float32,
    # output
    dst: wp.array4d(dtype=wp.float32),
):
    """Compute distance-to-image-plane, optionally clip it, and write camera output."""
    env, ray = wp.tid()
    if not env_mask[env]:
        return

    depth = ray_distance[env, ray]
    dir_w = ray_directions_w[env, ray]
    disp_w = wp.vec3f(depth * dir_w[0], depth * dir_w[1], depth * dir_w[2])
    disp_cam = wp.quat_rotate_inv(quat_w[env], disp_w)
    value = disp_cam[0]
    if clip_depth and (value > max_dist or wp.isnan(value)):
        value = fill_val

    row = ray // width
    col = ray - row * width
    dst[env, row, col, 0] = value
