# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from .rough_env_cfg import H1RoughEnvCfg


@dataclass
class H1FlatEnvCfg(H1RoughEnvCfg):
    def __post_init__(self):
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        # physics
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 65
        newton_mjwarp.solver_cfg.nconmax = 15
        self.sim.physics.default = newton_mjwarp
        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # rewards
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6
        # curriculum
        self.curriculum.terrain_levels = None
