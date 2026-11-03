# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-terrain variant of the MicroDuck velocity-tracking environment."""

from isaaclab.utils.configclass import configclass

from .velocity_env_cfg import MicroDuckVelocityRoughEnvCfg


@configclass
class MicroDuckVelocityFlatEnvCfg(MicroDuckVelocityRoughEnvCfg):
    """MicroDuck velocity-tracking environment on flat ground."""

    def __post_init__(self):
        super().__post_init__()

        # Reserve contact and constraint headroom for simultaneous body-ground contact.

        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 64
        newton_mjwarp.solver_cfg.nconmax = 10

        newton_mjwarp.solver_cfg.iterations = 10
        newton_mjwarp.solver_cfg.ls_iterations = 20
        self.sim.physics.default = newton_mjwarp

        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # a plane has no difficulty levels to progress through, and the term reads the terrain generator's
        # configuration, so it cannot merely sit inert here.
        self.curriculum.terrain_levels = None
