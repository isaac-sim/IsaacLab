# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for T3 Hexapod locomotion on flat terrain."""

from isaaclab.utils import configclass

from .rough_env_cfg import T3HexapodRoughEnvCfg


@configclass
class T3HexapodFlatEnvCfg(T3HexapodRoughEnvCfg):
    """Configuration for T3 Hexapod on flat terrain."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Override rewards for flat terrain
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5

        # Change terrain to flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No height scan needed for flat terrain
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # No terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class T3HexapodFlatEnvCfg_PLAY(T3HexapodFlatEnvCfg):
    """Configuration for playing/testing T3 Hexapod on flat terrain."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play
        self.observations.policy.enable_corruption = False

        # Remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
