# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-based reinforcement learning in the existing Franka cube stack scene."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ... import mdp
from ...mdp.rewards import stack_progress, stack_success
from .stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg


@configclass
class ObservationsCfg:
    """State inputs for the actor and critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Robot joints, cube poses, and previous action."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        objects = ObsTerm(func=mdp.object_obs)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Guide both placements and reward the existing success termination."""

    progress = RewTerm(func=stack_progress, weight=1.0)
    success = RewTerm(func=stack_success, weight=1000.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)


@configclass
class FrankaCubeStackRLEnvCfg(FrankaCubeStackEnvCfg):
    """Train the Franka to stack three cubes in the existing task order."""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.success.params["symmetric_height_tolerance"] = True
        self.sim.render_interval = self.decimation
