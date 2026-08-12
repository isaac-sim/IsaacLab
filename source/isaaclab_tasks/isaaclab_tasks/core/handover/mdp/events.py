# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the manager-based handover task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.utils import sample_joint_positions_within_limits

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def reset_handover_hands(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    joint_position_noise: float,
    joint_velocity_noise: float,
    right_hand_cfg: SceneEntityCfg = SceneEntityCfg("right_hand"),
    left_hand_cfg: SceneEntityCfg = SceneEntityCfg("left_hand"),
) -> None:
    """Reset both hands' joints and the position targets tracking them.

    Task-local rather than :func:`~isaaclab.envs.mdp.reset_joints_by_offset`: the
    hands' PD targets must be re-seeded alongside the joint state, and the framework
    terms write joint state only.

    Args:
        env: Environment containing both hands.
        env_ids: Environment indices to reset.
        joint_position_noise: Scale applied to sampled joint-position deltas.
        joint_velocity_noise: Joint-velocity noise half-width [rad/s].
        right_hand_cfg: Right-hand scene entity.
        left_hand_cfg: Left-hand scene entity.
    """
    for hand_cfg in (right_hand_cfg, left_hand_cfg):
        hand: Articulation = env.scene[hand_cfg.name]
        default_position = hand.data.default_joint_pos.torch[env_ids]
        limits = hand.data.joint_limits.torch[env_ids]
        joint_position = sample_joint_positions_within_limits(default_position, limits, joint_position_noise)
        velocity_sample = math_utils.sample_uniform(-1.0, 1.0, (len(env_ids), hand.num_joints), device=env.device)
        joint_velocity = hand.data.default_joint_vel.torch[env_ids] + joint_velocity_noise * velocity_sample

        hand.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
        hand.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        hand.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
