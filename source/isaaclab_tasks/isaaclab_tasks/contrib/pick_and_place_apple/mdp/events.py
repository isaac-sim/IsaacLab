# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom events for the H2 pick-and-place apple task."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["reset_robot_to_default_joint_positions", "reset_task_stage"]

logger = logging.getLogger(__name__)


def reset_robot_to_default_joint_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints/root directly to defaults (bypasses PD to avoid arm swing)."""
    if len(env_ids) == 0:
        return

    robot = env.scene[robot_cfg.name]

    default_joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel.torch[env_ids].clone()

    robot.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=env_ids)

    default_root_pose = robot.data.default_root_pose.torch[env_ids].clone()
    default_root_velocity = robot.data.default_root_vel.torch[env_ids].clone()
    robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=default_root_velocity, env_ids=env_ids)


def reset_task_stage(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    print_log: bool = False,
) -> None:
    """Reset stage trackers and capture the post-reset apple height reference."""
    if len(env_ids) == 0:
        return

    from .rewards import get_pnp_apple_state

    state = get_pnp_apple_state(env)
    previous_stage = state.task_stage[env_ids].clone()

    if print_log:
        exact_counts = [int((previous_stage == stage_id).sum().item()) for stage_id in range(5)]
        logger.info(
            "[PNP_STAGE_SUMMARY] total=%d exact_stage_0/1/2/3/4=%s",
            len(env_ids),
            exact_counts,
        )
        for stage_id in (1, 2, 3, 4):
            reached_mask = previous_stage >= stage_id
            reached_env_ids = env_ids[reached_mask].detach().cpu().tolist()
            logger.info(
                "[PNP_STAGE_SUMMARY] reached_stage_%d=%d/%d env_ids=%s",
                stage_id,
                len(reached_env_ids),
                len(env_ids),
                reached_env_ids,
            )

    state.task_stage[env_ids] = 0
    state.prev_stage_left_grasp[env_ids] = 0
    state.prev_stage_handover[env_ids] = 0
    state.prev_stage_place[env_ids] = 0
    state.prev_stage_release[env_ids] = 0
    state.left_dist_at_grasp[env_ids] = float("inf")
    state.stage_hold_counter[env_ids] = 0
    state.last_debug_print_step = -1

    apple = env.scene[apple_cfg.name]
    apple_z = apple.data.root_pos_w.torch[env_ids, 2]
    state.initial_apple_z[env_ids] = apple_z

    if print_log:
        logger.info("Reset task stage for %d environment(s)", len(env_ids))
