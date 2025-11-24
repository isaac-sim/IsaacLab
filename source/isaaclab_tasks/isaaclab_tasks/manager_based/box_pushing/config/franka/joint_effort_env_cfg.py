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
from isaaclab_tasks.manager_based.box_pushing.assets.franka import FRANKA_PANDA_COMPARISON
from isaaclab_tasks.manager_based.box_pushing.assets.franka import (
    FRANKA_PANDA_ONLY_TORQUE as FRANKA_CONFIG,  # isort: skip
)
from isaaclab_tasks.manager_based.box_pushing.box_pushing_env_cfg import DenseRewardCfg, TemporalSparseRewardCfg
from isaaclab_tasks.manager_based.box_pushing.config.franka.base_env_cfg import FrankaBoxPushingBaseEnvCfg


##
# Pre-defined configs
##
@configclass
class FrankaBoxPushingEnvCfg(FrankaBoxPushingBaseEnvCfg):
    robot_cfg_r2r = FRANKA_CONFIG
    robot_cfg_comparison = FRANKA_PANDA_COMPARISON

    def _configure_actions(self):
        self.actions.body_joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=10.0,
            debug_vis=True,
        )


@configclass
class FrankaBoxPushingEnvCfg_Dense(FrankaBoxPushingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards = DenseRewardCfg()


@configclass
class FrankaBoxPushingEnvCfg_TemporalSparse(FrankaBoxPushingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards = TemporalSparseRewardCfg()


@configclass
class FrankaBoxPushingEnvCfg_PLAY(FrankaBoxPushingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class FrankaBoxPushingNoIKEnvCfg(FrankaBoxPushingEnvCfg):
    use_ik_reset = False
    use_cached_ik = False


@configclass
class FrankaBoxPushingNoIKEnvCfg_Dense(FrankaBoxPushingNoIKEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = DenseRewardCfg()


@configclass
class FrankaBoxPushingNoIKEnvCfg_TemporalSparse(FrankaBoxPushingNoIKEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = TemporalSparseRewardCfg()
