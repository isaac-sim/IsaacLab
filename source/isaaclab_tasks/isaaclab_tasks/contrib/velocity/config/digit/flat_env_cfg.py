# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from .rough_env_cfg import DigitRoughEnvCfg


@dataclass
class DigitFlatEnvCfg(DigitRoughEnvCfg):
    def __post_init__(self):
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        # Change terrain to flat.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove height scanner.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # Remove terrain curriculum.
        self.curriculum.terrain_levels = None
