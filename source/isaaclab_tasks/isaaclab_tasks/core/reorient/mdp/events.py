# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for state-based in-hand reorientation tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from isaaclab_tasks.core.reorient.utils import sample_joint_positions_within_limits

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg


class reset_reorient_hand(ManagerTermBase):
    """Reset the hand's joints and the position targets tracking them.

    Task-local rather than :func:`~isaaclab.envs.mdp.reset_joints_by_offset`: the
    hand's PD targets must be re-seeded alongside the joint state, and the framework
    terms write joint state only.

    A term rather than a plain function so the joint limits are read once. They are a static
    model property, and on backends that keep them host-side (OvPhysX) every read stages
    through a pinned buffer and copies to the device -- a cost paid on each reset otherwise.

    The limits are cached at term construction, so limits written later in the run (for example by
    :func:`~isaaclab.envs.mdp.events.randomize_joint_parameters`) are NOT reflected here.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term and cache the robot's joint limits.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.robot: Articulation = env.scene[robot_cfg.name]
        self.joint_limits = self.robot.data.joint_limits.torch

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        joint_position_noise: float,
        joint_velocity_noise: float,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Reset the selected environments.

        Args:
            env: Environment containing the robot.
            env_ids: Environment indices to reset.
            joint_position_noise: Scale applied to sampled joint-position deltas.
            joint_velocity_noise: Joint-velocity noise half-width [rad/s].
            robot_cfg: Robot scene entity.
        """
        robot = self.robot
        default_position = robot.data.default_joint_pos.torch[env_ids]
        limits = self.joint_limits[env_ids]
        joint_position = sample_joint_positions_within_limits(default_position, limits, joint_position_noise)
        velocity_sample = math_utils.sample_uniform(-1.0, 1.0, (len(env_ids), robot.num_joints), device=env.device)
        joint_velocity = robot.data.default_joint_vel.torch[env_ids] + joint_velocity_noise * velocity_sample
        robot.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
        robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
