# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the manager-based Pendulum MARL task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_pendulum_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pole_cfg: SceneEntityCfg,
    pendulum_cfg: SceneEntityCfg,
) -> None:
    """Reset Pendulum joint positions with the final direct-task randomization.

    This event runs after :func:`isaaclab.envs.mdp.reset_scene_to_default`.
    It samples only the upper-pole position followed by the lower-pendulum
    position, preserving the direct task's random-number consumption order and
    the default velocities written by the preceding event.

    Args:
        env: The manager-based environment.
        env_ids: Environments to reset.
        pole_cfg: Scene entity selecting the upper-pole joint.
        pendulum_cfg: Scene entity selecting the lower-pendulum joint.
    """
    asset = env.scene[pole_cfg.name]
    joint_pos = asset.data.default_joint_pos.torch[env_ids].clone()
    joint_pos[:, pole_cfg.joint_ids] += sample_uniform(
        -0.25 * math.pi,
        0.25 * math.pi,
        joint_pos[:, pole_cfg.joint_ids].shape,
        joint_pos.device,
    )
    joint_pos[:, pendulum_cfg.joint_ids] += sample_uniform(
        -0.25 * math.pi,
        0.25 * math.pi,
        joint_pos[:, pendulum_cfg.joint_ids].shape,
        joint_pos.device,
    )
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
