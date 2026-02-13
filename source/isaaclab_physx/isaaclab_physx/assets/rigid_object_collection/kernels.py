# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def resolve_view_ids(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    num_query_envs: wp.int32,
    num_total_envs: wp.int32,
    view_ids: wp.array(dtype=wp.int32),
) -> None:
    """Resolve flat view indices from environment and body index pairs.

    This kernel computes flat PhysX view indices from (env_id, body_id) pairs. The view
    ordering is body-major: view_id = body_id * num_total_envs + env_id. The output array
    is laid out in column-major order over the (env, body) grid.

    Args:
        env_ids: Input array of environment indices. Shape is (num_query_envs,).
        body_ids: Input array of body indices. Shape is (num_query_bodies,).
        num_query_envs: Input scalar number of queried environments.
        num_total_envs: Input scalar total number of environments in the simulation.
        view_ids: Output array where flat view indices are written. Shape is
            (num_query_bodies * num_query_envs,).
    """
    i, j = wp.tid()
    view_ids[j * num_query_envs + i] = body_ids[j] * num_total_envs + env_ids[i]
