# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.func
def _segment_body_id(
    env_id: int,
    segment: int,
    root_body_ids: wp.array(dtype=wp.int32),
    link_body_ids: wp.array2d(dtype=wp.int32),
) -> int:
    """Newton body index for a cable segment. Segment 0 is the articulation root."""
    body_id = root_body_ids[env_id]
    if segment > 0:
        body_id = link_body_ids[env_id, segment - 1]
    return body_id


@wp.kernel(enable_backward=False)
def set_segment_pose_to_sim_index(
    segment_pose: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    root_body_ids: wp.array(dtype=wp.int32),
    link_body_ids: wp.array2d(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transformf),
):
    """Write selected cable segment poses to Newton body state."""
    index, segment = wp.tid()
    env_id = env_ids[index]
    body_q[_segment_body_id(env_id, segment, root_body_ids, link_body_ids)] = segment_pose[index, segment]


@wp.kernel(enable_backward=False)
def set_segment_pose_to_sim_mask(
    segment_pose: wp.array2d(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    root_body_ids: wp.array(dtype=wp.int32),
    link_body_ids: wp.array2d(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transformf),
):
    """Write masked cable segment poses to Newton body state."""
    env_id, segment = wp.tid()
    if env_mask[env_id]:
        body_q[_segment_body_id(env_id, segment, root_body_ids, link_body_ids)] = segment_pose[env_id, segment]


@wp.kernel(enable_backward=False)
def set_segment_velocity_to_sim_index(
    segment_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    root_body_ids: wp.array(dtype=wp.int32),
    link_body_ids: wp.array2d(dtype=wp.int32),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
):
    """Write selected cable segment velocities to Newton body state."""
    index, segment = wp.tid()
    env_id = env_ids[index]
    body_qd[_segment_body_id(env_id, segment, root_body_ids, link_body_ids)] = segment_velocity[index, segment]


@wp.kernel(enable_backward=False)
def set_segment_velocity_to_sim_mask(
    segment_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    root_body_ids: wp.array(dtype=wp.int32),
    link_body_ids: wp.array2d(dtype=wp.int32),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
):
    """Write masked cable segment velocities to Newton body state."""
    env_id, segment = wp.tid()
    if env_mask[env_id]:
        body_qd[_segment_body_id(env_id, segment, root_body_ids, link_body_ids)] = segment_velocity[env_id, segment]
