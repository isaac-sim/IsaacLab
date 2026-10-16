# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel(enable_backward=False)
def copy_view_rows_to_mesh_table_kernel(
    # input
    transforms_src: wp.array(dtype=wp.transformf),
    view_rows: wp.array2d(dtype=wp.int32),
    mesh_offset: int,
    # output
    positions_dst: wp.array2d(dtype=wp.vec3f),
    orientations_dst: wp.array2d(dtype=wp.quatf),
):
    """Copy tracked-body transforms into the per-env mesh table, reading the view row of each table entry.

    Table entries without a view row (``-1``) keep their poses.
    """
    env, local_mesh = wp.tid()
    row = view_rows[env, local_mesh]
    if row < 0:
        return
    xform = transforms_src[row]
    positions_dst[env, mesh_offset + local_mesh] = wp.transform_get_translation(xform)
    orientations_dst[env, mesh_offset + local_mesh] = wp.transform_get_rotation(xform)
