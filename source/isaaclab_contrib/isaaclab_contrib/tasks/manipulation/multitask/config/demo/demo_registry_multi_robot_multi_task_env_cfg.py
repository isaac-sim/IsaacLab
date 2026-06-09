# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registry-based multi-robot multi-task env: OpenArm-lift, Franka-cabinet, UR10-reach.

This file reproduces the same environment as
:mod:`demo_multi_robot_multi_task_env_cfg` using the modular
:class:`~...registry.MultiTaskRegistry` API.

Before (flat, ~720 lines per env)::

    # one giant configclass for scene, another for actions, obs, rewards ...

After (registry, ~25 lines)::

    env_cfg = (
        MultiTaskRegistry()
        .register(OPENARM_IK, LIFT_TASK_OPENARM)
        .register(FRANKA_JOINT, CABINET_TASK)
        .register(UR10_IK, REACH_TASK)
        .build_env_cfg(num_envs=4096)
    )

The registry handles:

* Clone-combination definitions and :class:`~isaaclab.cloner.InclusionSet` membership.
* :class:`ScatteredActionTermCfg` assembly from heterogeneous action specs.
* Scatter-observation merging (``ee_pose``, ``commands``, ``ee_pos_error``).
* Task-local observation namespacing (``cabinet_joint_pos``, ``object_pos``, …).
* Reward, termination, event, and curriculum composition.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.registry import MultiTaskRegistry
from isaaclab_contrib.tasks.manipulation.multitask.robots import (
    FRANKA_JOINT,
    OPENARM_IK,
    UR10_JOINT,
)
from isaaclab_contrib.tasks.manipulation.multitask.tasks import (
    CABINET_TASK,
    LIFT_TASK_OPENARM,
    REACH_TASK,
)

from .physics_cfg import MultitaskPhysicsCfg

# ---------------------------------------------------------------------------
# Build via registry (3 registrations → complete env cfg)
# ---------------------------------------------------------------------------

_REGISTRY = (
    MultiTaskRegistry()
    .register(OPENARM_IK, LIFT_TASK_OPENARM)
    .register(FRANKA_JOINT, CABINET_TASK)
    .register(UR10_JOINT, REACH_TASK)
)

RegistryMultiRobotMultiTaskEnvCfg = _REGISTRY.build_env_cfg(
    num_envs=4096,
    env_spacing=2.5,
    replicate_physics=True,
    physics=MultitaskPhysicsCfg(),
    decimation=2,
    episode_length_s=8.0,
    sim_dt=1.0 / 60.0,
)
"""Complete multi-robot multi-task env cfg assembled by the registry.

Groups:
    * ``openarm_lift``   — OpenArm robot, cube-lifting task.
    * ``franka_cabinet`` — Franka robot (rel-joint-pos), drawer-opening task.
    * ``ur10_reach``     — UR10 robot, 6-D pose-tracking task.
"""


# ---------------------------------------------------------------------------
# Play variant (smaller env count, no corruption)
# ---------------------------------------------------------------------------


def build_play_cfg(disabled_tasks: tuple[str, ...] = ()) -> type:
    """Return a play-mode env cfg *class* with fewer envs and optional task filtering.

    The returned object is a :class:`~isaaclab.envs.ManagerBasedRLEnvCfg`
    subclass (not an instance), so it can be passed directly to
    ``env_cfg_entry_point`` in gym registration.

    Args:
        disabled_tasks: Clone-group names to disable (weight → 0).
            Choices: ``"openarm_lift"``, ``"franka_cabinet"``, ``"ur10_reach"``.
            Leave empty to keep all groups active.

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

    return configclass(type("_RegistryPlayEnvCfg", (BaseCls,), {"__post_init__": _play_post_init}))


RegistryMultiRobotMultiTaskEnvCfg_PLAY = build_play_cfg()
"""Play variant with 64 envs and no observation corruption."""
