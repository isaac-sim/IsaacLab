# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Equal-sized rigid-ground and MPM-ground groups for one G1 policy."""

import math

from isaaclab.utils import configclass

from .env_cfg import ROUGH_TERRAINS_DIFFICULT_CFG
from .env_cfg.mixed_scene_cfg import G1MixedTerrainSceneCfg
from .env_cfg.physics_cfg import MPM_ENTRY, G1PhysicsProxyCfg
from .env_cfg.scene_cfg import FOOT_CONTACT_MARGIN
from .mdp.curriculums import terrain_levels_vel
from .mixed_env import outside_contact_region, randomize_mpm_material
from .mpm_env_cfg import G1MovingPatchEnvCfg
from .util.visualizer import (
    MovingPatchGLVisualizerCfg,
    MovingPatchRTXVisualizerCfg,
)


@configclass
class G1MixedTerrainEnvCfg(G1MovingPatchEnvCfg):
    """Use contiguous, equal-sized MPM and rigid groups, fixed across resets."""

    scene: G1MixedTerrainSceneCfg = G1MixedTerrainSceneCfg(num_envs=64, env_spacing=4.0)

    mpm_contact_margin: float = FOOT_CONTACT_MARGIN
    """Fixed robot foot collider margin in MPM environments [m]."""

    rigid_contact_margin: float = 0.0
    """Fixed robot foot collider margin in rigid-only environments [m]."""

    foot_contact_margin_range: tuple[float, float] | None = None
    """Disabled for mixed terrain; use the fixed per-terrain margins above."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 64
        self.sim.physics = G1PhysicsProxyCfg()
        self.curriculum.terrain_levels.func = terrain_levels_vel
        self.terminations.terrain_out_of_bounds.func = outside_contact_region
        for name in ("mpm_material", "mpm_material_log"):
            term = getattr(self.events, name, None)
            if term is not None:
                term.func = randomize_mpm_material

        self.curriculum.command_vel.params["velocity_stages"] = [
            {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
            {"step": 10000 * 24, "lin_vel_x": (-1.0, 1.5), "ang_vel_z": (-1.0, 1.0)},
        ]

    def validate_config(self) -> None:
        if self.foot_contact_margin_range is not None:
            raise ValueError(
                "Mixed terrain uses mpm_contact_margin and rigid_contact_margin, not a sampled margin range."
            )
        for name in ("mpm_contact_margin", "rigid_contact_margin"):
            margin = getattr(self, name)
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.scene.num_envs < 2 or self.scene.num_envs % 2:
            raise ValueError("Mixed terrain requires an even num_envs >= 2 (half MPM, half rigid).")
        super().validate_config()
        # Keep the same foot-proxy layout in every world so Newton can compact the MPM model.
        # Worlds without particles produce no MPM reaction force.
        for entry in self.sim.physics.solver_cfg.entries:
            if entry.name == MPM_ENTRY:
                entry.solver_cfg.max_active_cell_count //= 2
                entry.solver_cfg.max_leaf_node_count //= 2
                entry.solver_cfg.max_lower_node_count //= 2
                entry.solver_cfg.max_upper_node_count = max(32, entry.solver_cfg.max_upper_node_count // 2)


@configclass
class G1MixedTerrainEnvCfg_PLAY(G1MixedTerrainEnvCfg):
    """Play one environment of each ground type without pushes or observation noise."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 2
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        for visualizer_cfg in self.sim.visualizer_cfgs:
            visualizer_cfg.headless = False

        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_DIFFICULT_CFG.copy()
        self.scene.terrain.moving_patch_terrain.particle_depth = 0.35

        self.events.mpm_material_log = None
        self.events.mpm_material = None

        # Keep the play command ranges unchanged when environments reset.
        self.curriculum.command_vel = None
        self.commands.base_velocity.ranges.lin_vel_x = (1, 1)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.5
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)

        self.sim.visualizer_cfgs = [
            MovingPatchGLVisualizerCfg(
                eye=(-2.0, -0.0, 0.5),
                lookat=(0.0, 0.0, 0.0),
                follow_body_path="/World/envs/env_0/Robot/Geometry/pelvis",
            ),
            MovingPatchRTXVisualizerCfg(
                eye=(-2.0, -0.0, 0.5),
                lookat=(0.0, 0.0, 0.0),
                # follow_body_path="/World/envs/env_0/Robot/Geometry/pelvis",
            ),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=1000, video_interval=2000),
        ]
