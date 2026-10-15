# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 29-DoF locomotion over a coupled MJWarp/MPM granular bed."""

from __future__ import annotations

import math

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1MPMSceneCfg,
    G1ObservationsCfg,
    G1RewardsCfg,
    G1TerminationsCfg,
)
from .env_cfg.physics_cfg import DEFAULT_PROXY_MASS_SCALE, configure_sparse_mpm_capacities, g1_mpm_physics_cfg
from .env_cfg.scene_cfg import APPROACH_LENGTH, MPM_VISUAL_COLOR, G1RoughMPMSceneCfg, G1TiltingMPMSceneCfg
from .mdp.rewards import base_height_above_testbed
from .mdp.terminations import root_outside_testbed
from .terrain import TerrainMPMGridCfg
from .tilting_testbed import TestbedMotionCfg

# VISUALIZER = "newton_gl"
VISUALIZER = "newton_rtx"
# VISUALIZER = "kit"


@configclass
class G1MPMEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity-tracking locomotion of the G1 on granular media.

    The task mirrors the flat soft-contact task; the analytical contact model is replaced by an
    implicit MPM bed coupled to MJWarp through lagged foot proxies.
    """

    # scene
    scene: G1MPMSceneCfg = G1MPMSceneCfg(num_envs=32, env_spacing=8.0, replicate_physics=True, clone_in_fabric=True)

    # basic settings
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()

    # mdp settings
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    seed: int = 42

    # -- granular terrain state
    foot_body_expr: str = ".*_ankle_roll_link"
    """Body-name expression selecting the feet tracked against the bed."""

    foot_contact_force_threshold: float = 5.0
    """Granular reaction magnitude above which a foot counts as in contact [N]."""

    reset_particle_jitter: float = 0.0
    """Half-width of the uniform position jitter applied when the bed is reset [m]."""

    particle_max_velocity: float = 20.0
    """Speed limit enforced on the particles by the Newton model [m/s]."""

    # -- solver capacities, scaled with the world count before the simulation is created
    proxy_mass_scale: float = DEFAULT_PROXY_MASS_SCALE
    mpm_active_cell_count_per_world: int | None = None
    """Active-cell capacity per world; None derives it from the sand particle grid."""
    mpm_leaf_node_count_per_world: int = 1 << 13
    mpm_lower_node_count_per_world: int = 1 << 10
    mpm_upper_node_count_per_world: int = 16

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 4  # 50 Hz control
        # The approach has to be crossed before the granular part of the episode even begins.
        self.episode_length_s = 10.0 + APPROACH_LENGTH
        self.is_finite_horizon = False

        # simulation settings
        self.sim = SimulationCfg(dt=1 / 200, render_interval=self.decimation)  # 200 Hz physics
        self.sim.physics = g1_mpm_physics_cfg(
            self.proxy_mass_scale, rough_terrain=isinstance(self.scene.sand.spawn, TerrainMPMGridCfg)
        )
        self.sim.use_newton_actuators = True

        # The bed is a few metres across, so the high-speed stages of the cloned command
        # curriculum would drive the robot off it within an episode.
        self.curriculum.command_vel = None  # type: ignore
        # The robot spawns on the rigid platform and the bed lies ahead of it, so the command is
        # biased forward; a backward command would spend the episode walking away from the sand.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(0.0, -6.0, 1.5),
                headless=True,
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
            ),
            NewtonRTXVisualizerCfg(
                eye=(0.0, -6.0, 1.5),
                headless=True,
            ),
            # KitVisualizerCfg(eye=(0.0, -6.0, 1.5), headless=True),
        ]

        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]

        configure_sparse_mpm_capacities(self)


@configclass
class G1MPMEnvCfg_PLAY(G1MPMEnvCfg):
    """Small playback scene with randomization disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # disable curriculum and observation corruption
        self.curriculum.track_lin_vel = None  # type: ignore
        self.curriculum.track_ang_vel = None  # type: ignore
        self.curriculum.track_heading = None  # type: ignore
        self.curriculum.track_lin_vel_weight = None  # type: ignore
        self.observations.policy.enable_corruption = False

        # remove random events
        self.events.mpm_material = None  # type: ignore
        self.events.mpm_material_log = None  # type: ignore
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(2.0, -3.0, 0.5),
                lookat=(2.0, 0.0, 0.0),
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
            ),
            NewtonRTXVisualizerCfg(
                eye=(-0.5, -4.0, 1.5),
                lookat=(-0.5, 0.0, 0.0),
            ),
            # KitVisualizerCfg(eye=(0.0, -6.0, 1.5), headless=True),
        ]

        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1RoughMPMEnvCfg(G1MPMEnvCfg):
    """G1 walking from the approach platform onto a fixed generated sand surface.

    This geometry variant retains the base task's rewards. Height and touchdown
    rewards still reference flat ground; adapt them before rough-terrain training.
    """

    scene: G1RoughMPMSceneCfg = G1RoughMPMSceneCfg(num_envs=32, env_spacing=8.0)


@configclass
class G1RoughMPMEnvCfg_PLAY(G1MPMEnvCfg_PLAY):
    """Playback with a fixed generated sand surface and no episode randomization."""

    scene: G1RoughMPMSceneCfg = G1RoughMPMSceneCfg(num_envs=1, env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(-4.4, 1.5, 3.5),
                lookat=(-1.133547, -1.052032, 0.704035),  # Pitch -34 deg, yaw -38 deg.
                focal_length=12.000874,  # Vertical FOV 65 deg.
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
            ),
            NewtonRTXVisualizerCfg(
                eye=(-4.4, 1.5, 3.5),
                lookat=(-1.133547, -1.052032, 0.704035),  # Pitch -34 deg, yaw -38 deg.
                focal_length=12.000874,  # Vertical FOV 65 deg.
            ),
            # KitVisualizerCfg(eye=(0.0, -6.0, 1.5), headless=True),
        ]

        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1TiltingMPMEnvCfg(G1MPMEnvCfg):
    """Balance on a slowly tilting sand tray driven through physical joints."""

    scene: G1TiltingMPMSceneCfg = G1TiltingMPMSceneCfg(num_envs=4, env_spacing=8.0)
    testbed_motion: TestbedMotionCfg = TestbedMotionCfg(
        roll_range=(math.radians(-15.0), math.radians(15.0)),
        pitch_range=(math.radians(-15.0), math.radians(15.0)),
        debug_vis=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.testbed_motion.validate()
        self.episode_length_s = 30.0
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.76)
        # The same dynamic tray collides with G1 in MJWarp and with particles via its MPM proxy.
        self.sim.physics.solver_cfg.entries[0].bodies.append(r"/World/envs/env_.*/Testbed")
        self.sim.physics.solver_cfg.entries[0].include_static_shapes = False
        self.sim.physics.solver_cfg.entries[1].bodies = []
        self.sim.physics.solver_cfg.proxies[0].bodies.append(r"/World/envs/env_.*/Testbed/tray")
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {}
        self.terminations.outside_workspace.func = root_outside_testbed
        self.terminations.outside_workspace.params = {"margin": 0.3}
        self.rewards.base_height.func = base_height_above_testbed


@configclass
class G1TiltingMPMEnvCfg_PLAY(G1TiltingMPMEnvCfg):
    """Deterministic balancing evaluation on the tilting sand tray."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False

        self.events.mpm_material = None
        self.events.mpm_material_log = None
        self.events.add_base_mass = None
        self.events.physics_material = None
        self.events.scale_actuator_gains = None
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        self.commands.base_velocity.debug_vis = False

        self.curriculum.track_lin_vel = None
        self.curriculum.track_ang_vel = None
        self.curriculum.track_heading = None
        self.curriculum.track_lin_vel_weight = None
        for visualizer in self.sim.visualizer_cfgs:
            visualizer.headless = False

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(2.0, 2.0, 2.0),
                lookat=(0.0, 0.0, 0.0),
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
            ),
            NewtonRTXVisualizerCfg(
                eye=(2.0, 2.0, 2.0),
                lookat=(0.0, 0.0, 0.0),
            ),
            # KitVisualizerCfg(eye=(0.0, -6.0, 1.5), headless=True),
        ]

        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]
