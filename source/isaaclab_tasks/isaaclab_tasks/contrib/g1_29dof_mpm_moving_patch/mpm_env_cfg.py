# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based G1 task configurations for alpha moving terrain."""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from .env_cfg import (
    DEFAULT_PROXY_MASS_SCALE,
    ROUGH_TERRAINS_DIFFICULT_CFG,
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1MovingPatchSceneCfg,
    G1ObservationsCfg,
    G1PhysicsCfg,
    G1RewardsCfg,
    G1TerminationsCfg,
)
from .util.solver_setting import configure_sparse_mpm_capacities
from .util.visualizer import MovingPatchGLVisualizerCfg, MovingPatchRTXVisualizerCfg


@configclass
class G1MovingPatchEnvCfg(ManagerBasedRLEnvCfg):
    """Configure G1 directly on sand with two-way coupling and local learning settings."""

    scene: G1MovingPatchSceneCfg = G1MovingPatchSceneCfg(num_envs=1, env_spacing=4.0)
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=4, use_newton_actuators=True, physics=G1PhysicsCfg())
    decimation: int = 4
    is_finite_horizon: bool = False

    # mdp terms
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    seed: int = 42

    # Terrain contact state
    foot_body_expr: str = ".*_ankle_roll_link"
    """Body-name expression selecting the feet tracked against the bed."""

    foot_contact_force_threshold: float = 5.0
    """Granular reaction magnitude above which a foot counts as in contact [N]."""

    foot_contact_margin_range: tuple[float, float] | None = None
    """Optional foot collider margins [m], evenly spaced and randomly assigned per environment.
    Applied before solver construction to both rigid and MPM contact. None preserves
    asset margins. Mixed terrain uses its fixed per-terrain margin settings instead.
    A single environment receives the lower bound. Margins remain fixed across resets.
    """

    reset_particle_jitter: float = 0.0
    """Half-width of the uniform position jitter applied when the bed is reset [m]."""

    particle_max_velocity: float = 20.0
    """Speed limit enforced on the particles by the Newton model [m/s]."""

    # -- solver capacities, scaled with the world count before the simulation is created
    proxy_mass_scale: float = DEFAULT_PROXY_MASS_SCALE
    mpm_active_cell_count_per_world: int | None = None
    """Reserved cells per world; None follows particle count, with leaf capacity as a minimum."""
    mpm_leaf_node_count_per_world: int = 1 << 13
    mpm_lower_node_count_per_world: int = 1 << 10
    mpm_upper_node_count_per_world: int = 16

    def validate_config(self) -> None:
        """Apply final terrain overrides and capacities before simulation creation.

        IsaacLab calls this through ``cfg.validate()`` after CLI/Hydra overrides.
        Repeated validation recomputes derived settings without replacing the solvers.
        """

        use_terrain_curriculum = getattr(self.curriculum, "terrain_levels", None) is not None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = use_terrain_curriculum
        if use_terrain_curriculum:
            self.scene.terrain.use_terrain_origins = True
        self.scene.configure_terrain()
        self.sim.physics.configure_terrain(self.scene.terrain.moving_patch_terrain, self.proxy_mass_scale)  # type: ignore
        configure_sparse_mpm_capacities(self)

    def __post_init__(self):
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation

        # self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.0, -15.0, 4.0), lookat=(0.0, 0.0, 0.0))
        self.sim.visualizer_cfgs = [
            MovingPatchGLVisualizerCfg(eye=(-20.0, 0.0, 6.0), lookat=(0.0, 0.0, 0.0), show_particles=True),
            MovingPatchRTXVisualizerCfg(
                eye=(-50.0, -15.0, 5.0),
                lookat=(-10.0, 0.0, 0.0),
                show_particles=True,
            ),
        ]
        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
        ]

        # self.curriculum.command_vel = None  # type: ignore


@configclass
class G1MovingPatchEnvCfg_PLAY(G1MovingPatchEnvCfg):
    """Deterministic reset and forward command for inspecting the moving terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False

        generator = ROUGH_TERRAINS_DIFFICULT_CFG
        # generator = SLOPE_TERRAINS_CFG
        # generator = WAVE_TERRAINS_CFG
        # generator = FLAT_TERRAINS_CFG

        self.scene.terrain.terrain_generator = generator
        self.scene.terrain.moving_patch_terrain.particle_depth = 0.35

        for name in ("add_base_mass", "push_robot", "physics_material", "scale_actuator_gains"):
            setattr(self.events, name, None)
        self.events.reset_base.params = {
            "pose_range": {"x": (-5.0, 5.0), "y": (-5.0, 5.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.mpm_material_log = None
        # self.events.mpm_material = None

        # override command ranges for the PLAY configuration
        self.curriculum.command_vel = None  # type: ignore
        for name in ("track_lin_vel", "track_ang_vel", "track_heading", "track_lin_vel_weight"):
            setattr(self.curriculum, name, None)

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        # self.commands.base_velocity.debug_vis = False

        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.0, -15.0, 2.0), lookat=(0.0, 0.0, 0.0))
        self.sim.visualizer_cfgs = [
            MovingPatchGLVisualizerCfg(
                eye=(-2.0, -0.0, 0.5),
                lookat=(0.0, 0.0, 0.0),
                follow_body_path="/World/envs/env_0/Robot/Geometry/pelvis",
            ),
            MovingPatchRTXVisualizerCfg(
                eye=(-2.0, -0.0, 0.5),
                lookat=(0.0, 0.0, 0.0),
                follow_body_path="/World/envs/env_0/Robot/Geometry/pelvis",
            ),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=1000, video_interval=2000),
        ]
