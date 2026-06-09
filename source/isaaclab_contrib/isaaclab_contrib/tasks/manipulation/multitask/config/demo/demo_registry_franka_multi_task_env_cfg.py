# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registry-based single-robot multi-task env: Franka lift, reach, and cabinet.

A single Franka Panda robot (IK arm control) trains simultaneously on three tasks.
The robot asset is global (present in every environment); only task objects
(table + cube for lift, table for reach, cabinet for cabinet) are clone-group-local.

Groups:
    * ``franka_lift``    — cube-lifting task.
    * ``franka_reach``   — 6-D end-effector pose tracking.
    * ``franka_cabinet`` — drawer-opening task.

Compare with the flat equivalent :mod:`demo_franka_multi_task_env_cfg` (~600 lines).
The registry reduces this to ~25 lines.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.registry import MultiTaskRegistry
from isaaclab_contrib.tasks.manipulation.multitask.robots import FRANKA_IK
from isaaclab_contrib.tasks.manipulation.multitask.tasks import CABINET_TASK, LIFT_TASK, REACH_TASK

from .physics_cfg import MultitaskPhysicsCfg

# ---------------------------------------------------------------------------
# Build via registry (3 task registrations, 1 shared robot)
# ---------------------------------------------------------------------------

_REGISTRY = (
    MultiTaskRegistry()
    .register(FRANKA_IK, LIFT_TASK, group_name="franka_lift")
    .register(FRANKA_IK, REACH_TASK, group_name="franka_reach")
    .register(FRANKA_IK, CABINET_TASK, group_name="franka_cabinet")
)

RegistryFrankaMultiTaskEnvCfg = _REGISTRY.build_env_cfg(
    num_envs=4096,
    env_spacing=2.5,
    replicate_physics=True,
    physics=MultitaskPhysicsCfg(),
    decimation=2,
    episode_length_s=8.0,
    sim_dt=1.0 / 60.0,
)
"""Franka multi-task env cfg assembled by the registry.

The robot is shared across all clone groups (global asset).  Clone groups
partition environments by task object:

    * ``franka_lift``    — table + cube.
    * ``franka_reach``   — table.
    * ``franka_cabinet`` — Sektion cabinet + frame transformer.
"""


# ---------------------------------------------------------------------------
# Play variant
# ---------------------------------------------------------------------------


def build_play_cfg(disabled_tasks: tuple[str, ...] = ()) -> type:
    """Return a play-mode env cfg class with fewer envs and optional task filtering.

    Args:
        disabled_tasks: Clone-group names to disable.
            Choices: ``"franka_lift"``, ``"franka_reach"``, ``"franka_cabinet"``.

    Returns:
        A :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` subclass.
    """
    BaseCls = _REGISTRY.build_env_cfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=True,
        physics=MultitaskPhysicsCfg(),
        decimation=2,
        episode_length_s=8.0,
        sim_dt=1.0 / 60.0,
    )

    _disabled = set(disabled_tasks)
    _parent_post_init = BaseCls.__post_init__

    def _play_post_init(self):
        _parent_post_init(self)
        self.observations.policy.enable_corruption = False
        mdp.apply_task_filter(self, _disabled)

    return configclass(type("_RegistryFrankaPlayEnvCfg", (BaseCls,), {"__post_init__": _play_post_init}))


RegistryFrankaMultiTaskEnvCfg_PLAY = build_play_cfg()
"""Play variant with 64 envs and no observation corruption."""
