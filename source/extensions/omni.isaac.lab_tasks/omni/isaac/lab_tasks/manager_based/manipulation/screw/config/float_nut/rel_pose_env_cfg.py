# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from . import abs_pose_env_cfg
import omni.isaac.lab.sim as sim_utils
##
# Pre-defined configs



@configclass
class RelFloatNutTightenEnvCfg(abs_pose_env_cfg.AbsFloatNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.act_lows = [-0.001, -0.001, -0.001, -0.5, -0.5, -0.5]
        self.act_highs = [0.001, 0.001, 0.001, 0.5, 0.5, 0.5]
        
        # override actions
        self.actions.nut_action = mdp.RigidObjectPoseActionTermCfg(
            asset_name="nut",
            command_type="pose",
            use_relative_mode=True,
            p_gain=5,
            d_gain=0.01,
            act_lows=self.act_lows,
            act_highs=self.act_highs,
            )

        

@configclass
class RelFloatNutTightenEazyEnvCfg_PLAY(RelFloatNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        

@configclass
class RelFloatScrewMediumEnvCfg(RelFloatNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.nut.init_state.pos = (6.3000e-01, 4.0586e-06, 1.4904e-02)
        self.scene.nut.init_state.rot = (9.9833e-01,  1.2417e-04, -1.2629e-05,  5.7803e-02)
