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
class RelFloatScrewEazyEnvCfg(abs_pose_env_cfg.AbsFloatScrewEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.actions.nut_action.use_relative_mode = True
        # self.scene.nut.init_state.pos = (0.63, 0, 4.7518e-3)
        # self.scene.nut.init_state.rot = (9.4993e-01, -6.4670e-06, -2.1785e-05, -3.1247e-01)
        self.scene.nut.init_state.pos = (6.3000e-01, 2.0661e-06, 3.0895e-03)
        self.scene.nut.init_state.rot = (-2.1609e-01,  6.6671e-05, -6.6467e-05,  9.7637e-01)
        

@configclass
class RelFloatScrewEazyEnvCfg_PLAY(RelFloatScrewEazyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        

@configclass
class RelFloatScrewMediumEnvCfg(RelFloatScrewEazyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.nut.init_state.pos = (6.3000e-01, 4.0586e-06, 1.4904e-02)
        self.scene.nut.init_state.rot = (9.9833e-01,  1.2417e-04, -1.2629e-05,  5.7803e-02)
