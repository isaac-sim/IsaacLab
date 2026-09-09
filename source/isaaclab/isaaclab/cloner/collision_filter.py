# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility entry point for PhysX clone collision isolation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


def filter_collisions(
    stage: Usd.Stage,
    physicsscene_path: str,
    collision_root_path: str,
    prim_paths: Sequence[str],
    global_paths: Sequence[str] = (),
) -> None:
    """Author compact PhysX isolation groups for standalone clone workflows.

    Managed clone workflows should set :attr:`isaaclab.cloner.CloneCfg.isolate_environments`
    and let :class:`isaaclab.physics.PhysicsManager` realize the policy at the assembly barrier.

    Args:
        stage: USD stage.
        physicsscene_path: Path to the PhysX physics scene.
        collision_root_path: Root scope for the generated groups.
        prim_paths: Per-environment prim paths.
        global_paths: Global collider paths that should collide with every environment.

    Raises:
        RuntimeError: If manager-owned collision filtering has already run.
    """
    from isaaclab.physics import PhysicsManager  # noqa: PLC0415
    from isaaclab.physics._physx_collision_filter import _author_compact_environment_isolation  # noqa: PLC0415

    if PhysicsManager._collision_filter_applied:
        raise RuntimeError(
            "cloner.filter_collisions() cannot run after PhysicsManager.apply_collision_filter() for this simulation."
        )
    warnings.warn(
        "isaaclab.cloner.filter_collisions() is deprecated; use CloneCfg(isolate_environments=True) "
        "and cloner.replicate() for manager-owned collision filtering.",
        DeprecationWarning,
        stacklevel=2,
    )
    _author_compact_environment_isolation(
        stage, physicsscene_path, collision_root_path, list(prim_paths), list(global_paths)
    )
