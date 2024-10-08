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
class RelFloatScrewEnvCfg(abs_pose_env_cfg.AbsFloatScrewEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.actions.nut_action.use_relative_mode = True
        self.scene.nut.init_state.pos = (0.63, 0, 0.1)
        self.scene.nut.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, sleep_threshold=0.0, stabilization_threshold=0.0)

@configclass
class FrankaCubeLiftEnvCfg_PLAY(RelFloatScrewEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
