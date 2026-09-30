# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for cube stacking.

The task ships imitation-learning configs and therefore no rewards. These terms
exist so the same scene can be post-trained with RL, and they are built out of
the task's own predicates -- ``object_grasped``, ``object_stacked`` and
``cubes_stacked`` -- so that what earns reward and what counts as success cannot
drift apart.

Stage rewards are deliberately not potential-shaped: they pay once per step for a
condition that holds, so a policy cannot farm them by repeatedly entering and
leaving a state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .observations import object_grasped, object_stacked
from .terminations import cubes_stacked

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def ee_object_distance(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    std: float = 0.1,
) -> torch.Tensor:
    """Dense reach shaping: a kernel on end-effector-to-object distance in [0, 1].

    Bounded rather than a raw negative distance, so it cannot dominate the stage
    rewards when the arm starts far from the cube.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.vector_norm(obj.data.root_pos_w.torch - ee_frame.data.target_pos_w.torch[:, 0, :], dim=1)
    return 1.0 - torch.tanh(distance / std)


def object_is_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Paid while the named cube is held."""
    return object_grasped(env, robot_cfg, ee_frame_cfg, object_cfg).float()


def object_is_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upper_object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    lower_object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Paid while the named cube rests stacked and released."""
    return object_stacked(env, robot_cfg, upper_object_cfg, lower_object_cfg).float()


def stacking_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The task's own success predicate, so reward and termination agree exactly."""
    return cubes_stacked(env).float()
