# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Old wrench launch signatures, loaded only by deprecated imports; remove in Isaac Lab 3.2.

The adapters retain device-side frame routing for callers of the old kernels. Wrench arithmetic
is shared with the current implementation, and internal callers use ``wrench`` directly.
"""

from functools import cache
from typing import Any

import torch
import warp as wp

from . import wrench
from .index_kernel import selector_dtype


@cache
def make_write_wrench(env_dtype, body_dtype, *, accumulate: bool = False) -> wp.Kernel:
    """Adapt the old five-destination signature without wrapping ``wp.launch``."""
    indexed = env_dtype != wp.bool

    @wp.kernel(module="unique")
    def write(
        env_ids: wp.array(dtype=env_dtype),
        body_ids: wp.array(dtype=body_dtype),
        forces: wp.array2d(dtype=wp.vec3f),
        torques: wp.array2d(dtype=wp.vec3f),
        positions: wp.array2d(dtype=wp.vec3f),
        global_force_w: wp.array2d(dtype=wp.vec3f),
        global_torque_w: wp.array2d(dtype=wp.vec3f),
        global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
        local_force_b: wp.array2d(dtype=wp.vec3f),
        local_torque_b: wp.array2d(dtype=wp.vec3f),
        is_global: bool,
    ):
        i, j = wp.tid()
        if wp.static(indexed):
            env = wp.int32(env_ids[i])
            body = wp.int32(body_ids[j])
        else:
            if not env_ids[i] or not body_ids[j]:
                return
            env = i
            body = j

        force_buffer = local_force_b
        torque_buffer = local_torque_b
        if is_global:
            torque_buffer = global_torque_w
            force_buffer = global_force_at_com_w
            if positions:
                force_buffer = global_force_w

        if wp.static(accumulate):
            wrench._add_wrench(forces, torques, positions, force_buffer, torque_buffer, i, j, env, body)
        else:
            wrench._set_wrench(forces, torques, positions, force_buffer, torque_buffer, i, j, env, body)

    return write


@wp.kernel(module="unique")
def reset_wrench_composer_index(
    env_ids: wp.array(dtype=Any),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    out_force_b: wp.array2d(dtype=wp.vec3f),
    out_torque_b: wp.array2d(dtype=wp.vec3f),
):
    env, body = wp.tid()
    wrench._reset_wrench(
        global_force_w, global_torque_w, global_force_at_com_w, local_force_b, local_torque_b,
        out_force_b, out_torque_b, wp.int32(env_ids[env]), body,
    )


def set_forces_to_dual_buffers_index_kernel(
    env_ids: wp.array | torch.Tensor, body_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Return the old indexed set signature for these selector types; removed in Isaac Lab 3.2."""
    return make_write_wrench(selector_dtype(env_ids), selector_dtype(body_ids))


def add_forces_to_dual_buffers_index_kernel(
    env_ids: wp.array | torch.Tensor, body_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Return the old indexed add signature for these selector types; removed in Isaac Lab 3.2."""
    return make_write_wrench(selector_dtype(env_ids), selector_dtype(body_ids), accumulate=True)


def reset_wrench_composer_index_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Return the old indexed reset signature for this selector type; removed in Isaac Lab 3.2."""
    return wrench.make_reset_wrench(selector_dtype(env_ids))
