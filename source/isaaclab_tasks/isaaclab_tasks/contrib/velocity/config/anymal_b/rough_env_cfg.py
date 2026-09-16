# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import dataclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_B_CFG  # isort: skip
from isaaclab.utils import replace_config


@dataclass
class AnymalBRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        # scene
        self.scene.robot = replace_config(ANYMAL_B_CFG, prim_path="{ENV_REGEX_NS}/Robot")
