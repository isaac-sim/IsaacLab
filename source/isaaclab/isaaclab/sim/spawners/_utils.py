# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private helpers shared by the spawner implementations."""

from __future__ import annotations


def props_expr(prim_path: str, pattern: str | None) -> str:
    """Join a spawn prim path with a cfg-relative target pattern.

    Implements the target-pattern convention of the ``*_props_prim_path`` spawner
    configuration fields: ``None`` selects the spawn prim and its whole subtree,
    an empty string selects the spawn prim itself, and any other pattern is
    grafted under the spawn prim.

    Args:
        prim_path: The absolute path of the spawn prim.
        pattern: The cfg-relative target pattern, or None for the whole subtree.

    Returns:
        The absolute prim path expression to pass to a fragment family writer.
    """
    if pattern is None:
        return f"{prim_path}/**"
    if not pattern:
        return prim_path
    return f"{prim_path}/{pattern}"
