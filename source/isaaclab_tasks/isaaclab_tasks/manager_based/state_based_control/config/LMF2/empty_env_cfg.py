# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.state_based_control.config.LMF2.state_based_control_env_cfg import StateBasedControlEmptyEnvCfg

##
# Pre-defined configs
##

from isaaclab_assets.robots.lmf2 import LMF2_CFG

@configclass
class LMF2EmptyEnvCfg(StateBasedControlEmptyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to lmf2
        self.scene.robot = LMF2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class LMF2EmptyEnvCfg_PLAY(LMF2EmptyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
