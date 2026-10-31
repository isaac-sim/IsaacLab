# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch checks shared by the live and replay OVXR entrypoints."""

from __future__ import annotations

from typing import Any


def require_kitless_physics(env_cfg: Any, task: str) -> None:
    """Reject a task whose physics selection needs the Kit runtime before an OVXR launch.

    A task that leaves ``sim.physics`` unset selects Isaac Sim PhysX, which needs Kit.
    The kitless OVXR session cannot run it, so fail before launch. Such a task normally
    declares no physics preset, so the error points at ``--xr`` or the task configuration
    rather than at a ``physics=`` selector.

    Args:
        env_cfg: Resolved environment configuration.
        task: Task name, for the error message.

    Raises:
        ValueError: If ``env_cfg.sim.physics`` is ``None``.
    """
    if env_cfg.sim.physics is None:
        raise ValueError(
            f"--ovxr needs a task that selects kitless physics (OvPhysxCfg or NewtonCfg), but task '{task}' leaves"
            " sim.physics unset, which runs Isaac Sim PhysX through Kit. Run this task with --xr instead, or give"
            " its configuration a kitless physics preset."
        )
