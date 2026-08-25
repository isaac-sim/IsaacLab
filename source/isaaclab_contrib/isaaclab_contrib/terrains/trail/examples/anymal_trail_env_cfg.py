# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""

Manager-based configuration for ANYmal-C on the contributed trail terrains.

./isaaclab.sh -p scripts/environments/zero_agent.py --task IsaacContrib-Velocity-Trail-AnymalC

"""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.contrib.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg

from isaaclab_contrib.terrains.trail.examples.trails import TRAIL_CFG


@configclass
class AnymalCTrailEnvCfg(AnymalCRoughEnvCfg):
    """ManagerBasedRLEnvCfg for ANYmal-C walking over trail terrains."""

    def __post_init__(self) -> None:
        """Configure ANYmal-C and replace the rough terrain generator with trail terrains."""
        super().__post_init__()
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TRAIL_CFG,
            max_init_terrain_level=9,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=None,
            debug_vis=False,
        )
        self.sim.physics_material = self.scene.terrain.physics_material
