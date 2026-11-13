# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch checks shared by the live and replay OVXR entrypoints."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from ..xr_cfg import remove_camera_configs


def prepare_ovxr_launch(env_cfg: Any, task: str, num_envs: int) -> Any:
    """Check an OVXR launch before runtime startup and remove the task's camera sensors.

    Args:
        env_cfg: Resolved environment configuration.
        task: Task name, for the error message.
        num_envs: Requested number of environments.

    Returns:
        The configuration without camera sensors.

    Raises:
        ValueError: If ``num_envs`` is not 1, ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` is set to a value
            other than ``1``, or the task needs Kit physics (see :func:`require_kitless_physics`).
    """
    if num_envs != 1:
        raise ValueError("--ovxr currently requires --num_envs 1")
    if os.environ.get("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1").strip() != "1":
        raise ValueError("--ovxr requires OVStage (ISAAC_LAB_OVRTX_USE_OVSTAGE=1)")
    require_kitless_physics(env_cfg, task)
    return remove_camera_configs(env_cfg)


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


def reject_ovxr_runtime_conflicts(configs: Sequence[Any], kit_sources: Sequence[str]) -> None:
    """Reject validated launch components that conflict with an OVXR-owned frame loop.

    Args:
        configs: Runtime configurations returned by :func:`~isaaclab.app.validate_launch`.
        kit_sources: Components that :func:`~isaaclab.app.validate_launch` found to require Kit.

    Raises:
        ValueError: If a component requires Kit or a local visualizer is configured.
    """
    from isaaclab.visualizers import VisualizerBackendCfg

    visualizers = [cfg for cfg in configs if isinstance(cfg, VisualizerBackendCfg)]
    if any(cfg.visualizer_type == "kit" for cfg in visualizers):
        raise ValueError("--ovxr does not support the deprecated Kit visualizer; remove the Kit --viz selection")
    if kit_sources:
        raise ValueError(f"--ovxr requires kitless runtime components; Kit is required by: {', '.join(kit_sources)}")
    if visualizers:
        raise ValueError("Local visualizers are not currently supported with --ovxr; remove the --viz selection")
