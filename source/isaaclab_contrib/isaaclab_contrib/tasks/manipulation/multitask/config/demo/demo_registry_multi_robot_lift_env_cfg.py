# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registry-based multi-robot lift env: OpenArm and Franka, both lifting a cube.

Matches the flat :mod:`demo_multi_robot_lift_env_cfg` layout:

    * ``openarm_lift`` — OpenArm IK (6-D) + binary gripper (1-D).
    * ``franka_lift``  — Franka IK (6-D) + binary gripper (1-D).

Action space:  ``arm`` (6-D, shared IK column) + ``gripper`` (1-D).
Observations:  ``task_onehot`` | ``ee_pose`` (7) | ``ee_pose_error`` (7) |
               ``commands`` (7) | ``ee_pos_error`` (3) |
               ``{group}_object_pos`` (3) | ``{group}_object_target_pos_error`` (3) |
               ``{group}_ee_object_pos_error`` (3) | ``actions``.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.registry import MultiTaskRegistry
from isaaclab_contrib.tasks.manipulation.multitask.robots import FRANKA_IK, OPENARM_IK
from isaaclab_contrib.tasks.manipulation.multitask.tasks import LIFT_TASK, LIFT_TASK_OPENARM

from .physics_cfg import MultitaskPhysicsCfg

# ---------------------------------------------------------------------------
# Build via registry
# ---------------------------------------------------------------------------

_REGISTRY = (
    MultiTaskRegistry()
    .register(OPENARM_IK, LIFT_TASK_OPENARM, group_name="openarm_lift")
    .register(FRANKA_IK, LIFT_TASK, group_name="franka_lift")
)

RegistryMultiRobotLiftEnvCfg = _REGISTRY.build_env_cfg(
    num_envs=4096,
    env_spacing=2.5,
    replicate_physics=True,
    physics=MultitaskPhysicsCfg(),
    decimation=2,
    episode_length_s=8.0,
    sim_dt=1.0 / 60.0,
)
"""Multi-robot lift env cfg assembled by the registry."""


# ---------------------------------------------------------------------------
# Play variant
# ---------------------------------------------------------------------------


def build_play_cfg(disabled_tasks: tuple[str, ...] = ()) -> type:
    """Return a play-mode env cfg class.

    Args:
        disabled_tasks: Clone-group names to disable.
            Choices: ``"openarm_lift"``, ``"franka_lift"``.
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

    return configclass(type("_RegistryMultiRobotLiftPlayEnvCfg", (BaseCls,), {"__post_init__": _play_post_init}))


RegistryMultiRobotLiftEnvCfg_PLAY = build_play_cfg()
"""Play variant with 64 envs and no observation corruption."""
