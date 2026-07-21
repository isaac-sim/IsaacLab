# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def gather_root_segment_state(
    root_body_ids: wp.array(dtype=wp.int32),
    body_pose: wp.array(dtype=wp.transformf),
    body_velocity: wp.array(dtype=wp.spatial_vectorf),
    segment_pose: wp.array2d(dtype=wp.transformf),
    segment_velocity: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Gather root segment state from model-wide body arrays."""
    env_id = wp.tid()
    body_id = root_body_ids[env_id]
    segment_pose[env_id, 0] = body_pose[body_id]
    segment_velocity[env_id, 0] = body_velocity[body_id]
