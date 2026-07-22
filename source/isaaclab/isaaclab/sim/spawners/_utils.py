# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private helpers shared by the spawner implementations."""

from __future__ import annotations


def props_expr(prim_path: str, pattern: str) -> str:
    """Join an anchor prim path with a cfg-relative target pattern.

    Implements the key convention of the fragment mapping spawner configuration
    fields (e.g. :attr:`~isaaclab.sim.spawners.RigidObjectSpawnerCfg.rigid_props`):
    an empty string selects the anchor prim itself, and any other pattern is
    grafted under the anchor prim.

    Args:
        prim_path: The absolute path of the anchor prim.
        pattern: The cfg-relative target pattern.

    Returns:
        The absolute prim path expression to pass to a fragment family writer.
    """
    if not pattern:
        return prim_path
    return f"{prim_path}/{pattern}"
