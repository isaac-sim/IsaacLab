# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import torch
import warp as wp

from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher


@wp.kernel
def resolve_view_ids(
    env_ids: wp.array(dtype=Any),
    body_ids: wp.array(dtype=Any),
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
    view_ids[j * num_query_envs + i] = wp.int32(body_ids[j]) * num_total_envs + wp.int32(env_ids[i])


_RESOLVE_VIEW_IDS_DISPATCHER = IndexKernelDispatcher(resolve_view_ids, ("env_ids", "body_ids"))


def resolve_view_ids_kernel(env_ids: wp.array | torch.Tensor, body_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the view-index kernel for the selector dtypes."""
    return _RESOLVE_VIEW_IDS_DISPATCHER.select(env_ids, body_ids)
