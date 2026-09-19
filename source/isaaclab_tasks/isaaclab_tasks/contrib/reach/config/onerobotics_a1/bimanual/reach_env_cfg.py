# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetric bimanual specialization of the core Reach MDP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg

from .. import mdp as a1_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def all_pose_commands_success(env: ManagerBasedRLEnv, command_names: tuple[str, ...]) -> torch.Tensor:
    """Return true only where every named standard pose command is successful."""
    if not command_names:
        raise ValueError("At least one pose command is required for bimanual success.")
    success = mdp.pose_command_success(env, command_names[0])
    for command_name in command_names[1:]:
        success = torch.logical_and(success, mdp.pose_command_success(env, command_name))
    return success


class AllPoseCommandsSuccess(ManagerTermBase):
    """Terminate on simultaneous pose success and log the matching episode metric.

    Each standard pose command logs the same unified success-rate key during
    command-manager reset. The termination manager resets later, so this term
    replaces that per-arm value with the bimanual logical-AND success rate.
    """

    def __init__(self, cfg: DoneTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, command_names: tuple[str, ...]) -> torch.Tensor:
        success = all_pose_commands_success(env, command_names)
        self._succeeded |= success
        return success

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log and clear simultaneous episode success for the reset environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._succeeded[env_ids].float().mean().item()
        self._succeeded[env_ids] = False


@configclass
class BimanualCommandsCfg:
    """Independent reachable pose commands for the right and left end effectors."""

    right_ee_pose: a1_mdp.FkReachablePoseCommandCfg = MISSING
    left_ee_pose: a1_mdp.FkReachablePoseCommandCfg = MISSING


@configclass
class BimanualActionsCfg:
    """Right-then-left joint-position actions."""

    right_arm_action: ActionTerm = MISSING
    left_arm_action: ActionTerm = MISSING


@configclass
class BimanualObservationsCfg:
    """Policy observations matching the core Reach observation sources."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Concatenated 56-D policy observation group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})
        left_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class BimanualRewardsCfg:
    """Symmetric tracking rewards plus the core Reach regularizers."""

    right_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
        },
    )
    left_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
        },
    )
    right_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
        },
    )
    left_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
        },
    )
    success = RewTerm(func=mdp.is_terminated_term, weight=10.0, params={"term_keys": ["success"]})

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.005)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING, preserve_order=True)},
    )


@configclass
class BimanualTerminationsCfg:
    """Require simultaneous right and left pose success."""

    success = DoneTerm(
        func=AllPoseCommandsSuccess,
        params={"command_names": ("right_ee_pose", "left_ee_pose")},
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BimanualReachEnvCfg(ReachEnvCfg):
    """Core Reach scene, reset, curriculum, reward sources, and timings for two arms."""

    observations: BimanualObservationsCfg = BimanualObservationsCfg()
    actions: BimanualActionsCfg = BimanualActionsCfg()
    commands: BimanualCommandsCfg = BimanualCommandsCfg()
    rewards: BimanualRewardsCfg = BimanualRewardsCfg()
    terminations: BimanualTerminationsCfg = BimanualTerminationsCfg()
