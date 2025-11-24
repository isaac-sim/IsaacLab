# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.box_pushing import mdp
from isaaclab_tasks.manager_based.box_pushing.assets.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_COMPARISON
from isaaclab_tasks.manager_based.box_pushing.box_pushing_env_cfg import DenseRewardCfg, TemporalSparseRewardCfg
from isaaclab_tasks.manager_based.box_pushing.config.franka.base_env_cfg import FrankaBoxPushingBaseEnvCfg

##
# Pre-defined configs
##
FRANKA_PANDA_PD_COMPARISON = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_PD_COMPARISON.spawn.usd_path = FRANKA_PANDA_COMPARISON.spawn.usd_path


@configclass
class FrankaBoxPushingJointPositionEnvCfg(FrankaBoxPushingBaseEnvCfg):
    robot_cfg_r2r = FRANKA_PANDA_CFG
    robot_cfg_comparison = FRANKA_PANDA_PD_COMPARISON

    def _configure_actions(self):
        self.actions.body_joint_effort = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )


@configclass
class FrankaBoxPushingJointPositionEnvCfg_Dense(FrankaBoxPushingJointPositionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards = DenseRewardCfg()


@configclass
class FrankaBoxPushingJointPositionEnvCfg_TemporalSparse(FrankaBoxPushingJointPositionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards = TemporalSparseRewardCfg()


@configclass
class FrankaBoxPushingJointPositionEnvCfg_PLAY(FrankaBoxPushingJointPositionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class FrankaBoxPushingJointPositionNoIKEnvCfg(FrankaBoxPushingJointPositionEnvCfg):
    use_ik_reset = False
    use_cached_ik = False


@configclass
class FrankaBoxPushingJointPositionNoIKEnvCfg_Dense(FrankaBoxPushingJointPositionNoIKEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = DenseRewardCfg()


@configclass
class FrankaBoxPushingJointPositionNoIKEnvCfg_TemporalSparse(FrankaBoxPushingJointPositionNoIKEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = TemporalSparseRewardCfg()
