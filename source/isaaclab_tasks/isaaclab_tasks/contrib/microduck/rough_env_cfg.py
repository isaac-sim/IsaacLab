# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Blind rough walking using microduck_rl's gentle terrain recipe."""

import trimesh
from isaaclab_newton.sim.schemas import MujocoCollisionCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.mdp.curriculums import terrain_levels_vel

from .flat_env_cfg import CurriculumCfg, MicroDuckSceneCfg, MicroDuckVelocityFlatEnvCfg


class MicroDuckTerrainImporter(terrain_gen.TerrainImporter):
    """Author upstream's terrain contact response before Newton parses the scene."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        """Import a terrain mesh and apply the softer MuJoCo contact response."""
        super().import_mesh(name, mesh)
        sim_utils.apply_collision_properties(
            f"{self.cfg.prim_path}/{name}/mesh",
            [MujocoCollisionCfg(solref=(0.04, 1.0), solimp=(0.85, 0.95, 0.001, 0.5, 2.0))],
        )


@configclass
class MicroDuckRoughSceneCfg(MicroDuckSceneCfg):
    """Walking scene with two terrain rays per foot, excluded from actor observations."""

    left_foot_height = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg/ankle_left",
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=(0.08, 0.0)),
        max_distance=1.0,
        global_world_only=True,
    )
    right_foot_height = left_foot_height.replace(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2/ankle_right"
    )


@configclass
class MicroDuckRoughCurriculumCfg(CurriculumCfg):
    """Progress through terrain difficulty according to distance walked."""

    terrain_levels = CurriculumTermCfg(func=terrain_levels_vel)


@configclass
class MicroDuckVelocityRoughEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """MicroDuck rough velocity walking with the flat task's policy interface and BAM servos."""

    scene: MicroDuckRoughSceneCfg = MicroDuckRoughSceneCfg(num_envs=4096, env_spacing=2.0)
    curriculum: MicroDuckRoughCurriculumCfg = MicroDuckRoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.class_type = MicroDuckTerrainImporter
        self.scene.terrain.max_init_terrain_level = 5
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            curriculum=True,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.25,
                    step_height_range=(0.0, 0.015),
                    step_width=0.15,
                    platform_width=2.0,
                    border_width=1.0,
                ),
                "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.30, grid_width=0.45, grid_height_range=(0.0, 0.010), platform_width=1.5
                ),
                "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.20, slope_range=(0.03, 0.10), platform_width=2.0, vertical_scale=0.001
                ),
            },
        )
        height_sensor_names = ("left_foot_height", "right_foot_height")
        self.observations.critic.foot_height.params["height_sensor_names"] = height_sensor_names
        self.rewards.foot_clearance.params["height_sensor_names"] = height_sensor_names
        self.rewards.foot_swing_height.params["height_sensor_names"] = height_sensor_names
        self.scene.left_foot_height.update_period = self.sim.dt * self.decimation
        self.scene.right_foot_height.update_period = self.sim.dt * self.decimation
        self.sim.physics.solver_cfg.nconmax = 200
        self.sim.physics.solver_cfg.njmax = 1024
        # The current MJWarp solver exhausts the flat recipe's 10-iteration budget.
        self.sim.physics.solver_cfg.iterations = 100

    def play_mode(self):
        """Use a smaller terrain map with randomly sampled difficulty for playback."""
        super().play_mode()
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.curriculum.terrain_levels = None
