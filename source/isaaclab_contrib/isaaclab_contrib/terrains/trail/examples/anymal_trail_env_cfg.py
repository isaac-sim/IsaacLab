# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""

Manager-based configuration for ANYmal-C on the contributed trail terrains.

export PYTHONPATH=/workspace/isaaclab/source/isaaclab_contrib:$PYTHONPATH

./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task IsaacContrib-Velocity-Trail-AnymalC \
  --num_envs 4

Install:
pip install manifold3d
"""

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.contrib.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg
from isaaclab_contrib.terrains.trail.examples.trails import TRAIL_CFG


@configclass
class AnymalCTrailEnvCfg(AnymalCRoughEnvCfg):
    """ManagerBasedRLEnvCfg for ANYmal-C walking over trail terrains."""

    def __post_init__(self) -> None:
        """Configure ANYmal-C and replace the rough terrain with trail terrains."""
        super().__post_init__()
        # --------------------------------------------------
        # overwrites go here...
        for trail_cfg in TRAIL_CFG.sub_terrains.values():
            trail_cfg.training = False  # do not use simplified useful for training
            trail_cfg.cut_objects_above = None  # do not cut objects
        # --------------------------------------------------
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TRAIL_CFG,
            collision_group=-1,
            visual_material=None,
            debug_vis=False,
        )
