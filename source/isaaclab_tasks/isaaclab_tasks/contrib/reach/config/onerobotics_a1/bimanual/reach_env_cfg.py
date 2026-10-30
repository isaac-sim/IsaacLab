# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual specialization of the shared Reach configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.core.reach.reach_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    ObservationsCfg,
    ReachEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class AllPoseCommandsSuccess(ManagerTermBase):
    """Require both pose commands to succeed and log simultaneous episode success."""

    def __init__(self, cfg: DoneTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, command_names: tuple[str, str]) -> torch.Tensor:
        right, left = command_names
        success = mdp.pose_command_success(env, right) & mdp.pose_command_success(env, left)
        self._succeeded |= success
        return success

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Replace the per-arm command metric with simultaneous bimanual success."""
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._succeeded[env_ids].float().mean().item()
        self._succeeded[env_ids] = False


@configclass
class BimanualCommandsCfg(CommandsCfg):
    """Use a standard uniform pose command for each end effector."""

    ee_pose = None
    right_ee_pose = CommandsCfg().ee_pose.copy()
    left_ee_pose = CommandsCfg().ee_pose.copy()


@configclass
class BimanualActionsCfg(ActionsCfg):
    """Right-then-left joint-position actions."""

    arm_action = None
    right_arm_action: ActionTerm = MISSING
    left_arm_action: ActionTerm = MISSING


@configclass
class BimanualObservationsCfg(ObservationsCfg):
    """Extend the base policy with two commands and ordered arm joint observations."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        joint_pos = ObservationsCfg.PolicyCfg().joint_pos.replace(
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING, preserve_order=True)}
        )
        joint_vel = ObservationsCfg.PolicyCfg().joint_vel.replace(
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING, preserve_order=True)}
        )
        # Preserve joint positions, velocities, both commands, then actions in the policy vector.
        pose_command = None
        actions = None
        right_pose_command = ObservationsCfg.PolicyCfg().pose_command.replace(params={"command_name": "right_ee_pose"})
        left_pose_command = ObservationsCfg.PolicyCfg().pose_command.replace(params={"command_name": "left_ee_pose"})
        last_action = ObservationsCfg.PolicyCfg().actions.copy()

    policy: PolicyCfg = PolicyCfg()


@configclass
class BimanualRewardsCfg(RewardsCfg):
    """Reuse the base reward terms and regularizers for both arms."""

    end_effector_position_tracking = None
    end_effector_orientation_tracking = None
    right_end_effector_position_tracking = RewardsCfg().end_effector_position_tracking.replace(
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "right_ee_pose"}
    )
    left_end_effector_position_tracking = RewardsCfg().end_effector_position_tracking.replace(
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "left_ee_pose"}
    )
    right_end_effector_orientation_tracking = RewardsCfg().end_effector_orientation_tracking.replace(
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "right_ee_pose"}
    )
    left_end_effector_orientation_tracking = RewardsCfg().end_effector_orientation_tracking.replace(
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "left_ee_pose"}
    )


@configclass
class BimanualTerminationsCfg(TerminationsCfg):
    """Keep the shared timeout and require simultaneous pose success."""

    success = DoneTerm(func=AllPoseCommandsSuccess, params={"command_names": ("right_ee_pose", "left_ee_pose")})


@configclass
class BimanualReachEnvCfg(ReachEnvCfg):
    """Reuse the core Reach scene, events, curriculum, and control settings."""

    observations: BimanualObservationsCfg = BimanualObservationsCfg()
    actions: BimanualActionsCfg = BimanualActionsCfg()
    commands: BimanualCommandsCfg = BimanualCommandsCfg()
    rewards: BimanualRewardsCfg = BimanualRewardsCfg()
    terminations: BimanualTerminationsCfg = BimanualTerminationsCfg()
