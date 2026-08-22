# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .rough_env_cfg import AnymalBRoughEnvCfg


@configclass
class AnymalBFlatEnvCfg(AnymalBRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # physics
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 75
        newton_mjwarp.solver_cfg.nconmax = 15
        newton_mjwarp.solver_cfg.cone = "elliptic"
        newton_mjwarp.solver_cfg.impratio = 100.0
        newton_mjwarp.num_substeps = 1
        self.sim.physics.default = self.sim.physics.isaacsim_physx
        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # curriculum
        self.curriculum.terrain_levels = None
