# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch checks shared by the live and replay OVXR entrypoints."""

from __future__ import annotations

from typing import Any


def require_kitless_physics(env_cfg: Any) -> None:
    """Reject a task whose physics selection needs the Kit runtime before an OVXR launch.

    A task that leaves ``sim.physics`` unset selects Isaac Sim PhysX, which needs Kit.
    The kitless OVXR session cannot run it, so fail before launch with the selector to use.

    Args:
        env_cfg: Resolved environment configuration.

    Raises:
        ValueError: If ``env_cfg.sim.physics`` is ``None``.
    """
    if env_cfg.sim.physics is None:
        raise ValueError(
            "--ovxr requires a kitless physics preset, but the task leaves sim.physics unset (Isaac Sim"
            " PhysX through Kit). Select one, for example physics=ovphysx or physics=newton_mjwarp."
        )
