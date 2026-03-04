# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

import warp as wp


@wp.kernel
def copy_from_newton_kernel(
    # in
    env_mask: wp.array(dtype=wp.bool),
    newton_transforms: wp.array(dtype=wp.transformf),  # flat (num_envs * stride,)
    stride: int,
    num_targets: int,
    # outputs
    source_transforms: wp.array(dtype=wp.transformf),  # (num_envs,)
    target_transforms: wp.array2d(dtype=wp.transformf),  # (num_envs, num_targets)
):
    """Copy frame transform data from Newton sensor into owned buffers.

    Deinterleaves the flat strided Newton output into separate source and target buffers.
    Launch with dim=(num_envs, 1 + num_targets).
    """
    env, idx = wp.tid()

    if env_mask:
        if not env_mask[env]:
            return

    t = newton_transforms[env * stride + idx]

    if idx == 0:
        source_transforms[env] = t
    else:
        target_transforms[env, idx - 1] = t


@wp.kernel
def compose_target_world_kernel(
    # in
    source_transforms: wp.array(dtype=wp.transformf),  # (num_envs,)
    target_transforms: wp.array2d(dtype=wp.transformf),  # (num_envs, num_targets)
    # outputs
    target_transforms_w: wp.array2d(dtype=wp.transformf),  # (num_envs, num_targets)
):
    """Compute target world transforms: source_world * target_relative.

    Launch with dim=(num_envs, num_targets).
    """
    env, tgt = wp.tid()
    target_transforms_w[env, tgt] = wp.transform_multiply(source_transforms[env], target_transforms[env, tgt])
