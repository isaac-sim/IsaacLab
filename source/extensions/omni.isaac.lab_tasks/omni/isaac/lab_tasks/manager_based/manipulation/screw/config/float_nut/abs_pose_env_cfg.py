# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.sensors.frame_transformer import FrameTransformerCfg, OffsetCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedEnv
import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.screw.screw_env_cfg import BaseNutTightenEnvCfg
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg

##
# Pre-defined configs
##


##
# Environment configuration
##

@configclass
class AbsFloatNutTightenEnvCfg(BaseNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = None

        self.act_lows = [-0.001, -0.001, -0.001, -0.2, -0.2, -0.2]
        self.act_highs = [0.001, 0.001, 0.001, 0.2, 0.2, 0.2]
        
        # override actions
        self.actions.nut_action = mdp.RigidObjectPoseActionTermCfg(
            asset_name="nut",
            command_type="pose",
            use_relative_mode=False,
            p_gain=5,
            d_gain=0.01,
            lows=self.act_lows,
            highs=self.act_highs,
            )



@configclass
class FloatScrewEnvCfg_PLAY(AbsFloatNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
