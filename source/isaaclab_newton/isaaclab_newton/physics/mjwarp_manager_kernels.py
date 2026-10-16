# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the MuJoCo Warp manager's per-world solver reset."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def clear_last_solve_load_kernel(
    world_mask: wp.array[wp.bool],
    qfrc_bias: wp.array2d[float],
    qfrc_constraint: wp.array2d[float],
    nefc: wp.array[wp.int32],
):
    """Zero the previous solve's bias and constraint forces and its constraint count in the masked worlds."""
    world, dof = wp.tid()
    if not world_mask[world]:
        return
    qfrc_bias[world, dof] = 0.0
    qfrc_constraint[world, dof] = 0.0
    if dof == 0:
        nefc[world] = 0
