# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from pxr import Usd


def newton_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate prims for Newton physics backend.

    Newton does not require explicit physics replication like PhysX.
    This is a no-op placeholder that maintains API compatibility.

    Args:
        stage: USD stage.
        sources: Source prim paths.
        destinations: Destination templates with ``"{}"`` for env index.
        env_ids: Environment indices.
        mapping: Bool mask selecting envs per source.
        use_fabric: Unused (for API compatibility).
        device: Unused (for API compatibility).

    Returns:
        None
    """
    # Newton doesn't need explicit physics replication - USD replication is sufficient
    pass
