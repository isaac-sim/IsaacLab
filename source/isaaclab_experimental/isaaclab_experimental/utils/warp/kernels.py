# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels used across multiple managers."""

from __future__ import annotations

import warp as wp


@wp.kernel
def count_masked(
    mask: wp.array(dtype=wp.bool),
    out_count: wp.array(dtype=wp.int32),
):
    """Count the number of True entries in a boolean mask.

    ``out_count`` must be zeroed before launch.  Result is stored in ``out_count[0]``.
    Launched with ``dim = num_envs``.
    """
    env_id = wp.tid()
    if mask[env_id]:
        wp.atomic_add(out_count, 0, 1)


@wp.kernel
def compute_reset_scale(
    reset_count: wp.array(dtype=wp.int32),
    divisor: wp.float32,
    out_scale: wp.array(dtype=wp.float32),
):
    """Compute ``1 / (count * divisor)`` scaling factor from a reset count.

    Pass ``divisor = 1.0`` for plain ``1 / count`` (e.g. command manager).
    Pass ``divisor = max_episode_length_s`` for reward-style normalization.

    Launched with ``dim = 1``.
    """
    count = reset_count[0]
    if count > 0:
        out_scale[0] = 1.0 / (wp.float32(count) * divisor)
    else:
        out_scale[0] = 0.0
