# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYmal-D velocity-tracking environment on rough terrain."""

from isaaclab.utils import configclass

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

from ...velocity_env_cfg import LocomotionVelocityRoughEnvCfg


@configclass
class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the ANYmal-D velocity-tracking environment on rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = ANYMAL_D_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot", init_state=ANYMAL_D_CFG.init_state.replace(pos=(0.0, 0.0, 0.65))
        )
