# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


@configclass
class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ANYmal-D's base capsule passes within a few millimetres of the thigh colliders, so
        # the shared 1 cm Newton shape margin inflates them into constant overlap and trips
        # the base_contact termination. Other robots keep the default.
        newton_cfg = getattr(self.sim.physics, "newton_mjwarp", None)
        if newton_cfg is not None:
            newton_cfg.default_shape_cfg = newton_cfg.default_shape_cfg.replace(margin=0.0)
