# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Full RL environment configuration for Eigenbot hexapod locomotion.

Ported from legged_gym (Isaac Gym) EigenbotRoughCfg / LeggedRobotCfg.
"""

from __future__ import annotations

import math

from eigenbot.assets import EIGENBOT_CFG

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------
NUM_JOINTS = 18
NUM_FEET = 6
NUM_COMMANDS = 4

# Proprioceptive observation: ang_vel(3) + imu(2) + delta_yaw(1) + gravity(3)
#   + cmd_vel(1) + dof_pos(18) + dof_vel(18) + flat(1) + not_flat(1)
#   + actions(18) + contacts(6) = 72
N_PROPRIO = 72
N_SCAN = 132  # 12 x 11 height measurement grid
N_PRIV = 9  # base_lin_vel(3) + zeros(6)
N_PRIV_LATENT = 41  # mass_params(4) + friction(1) + motor_p(18) + motor_d(18)
HISTORY_LEN = 10

NUM_OBSERVATIONS = N_PROPRIO + N_SCAN + N_PRIV + N_PRIV_LATENT + HISTORY_LEN * N_PROPRIO


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------
@configclass
class RewardScalesCfg:
    """Reward scales, applied per-step (multiplied by dt internally)."""

    termination: float = -1.0
    tracking_goal_vel: float = 4.0
    delta_yaw: float = 1.2
    lin_vel_z: float = -1.0
    ang_vel_xy: float = -0.05
    orientation: float = -1.0
    torques: float = -0.0002  # eigenbot override (base = -0.00001)
    dof_vel: float = 0.0
    dof_acc: float = -2.5e-7
    feet_air_time: float = 0.87
    collision: float = -1.0
    stumble: float = -1.5
    action_rate: float = -0.01
    stand_still: float = -0.5
    rule_1: float = 0.35
    rule_3: float = 0.1
    dof_pos_limits: float = -10.0  # eigenbot override


@configclass
class RewardsCfg:
    scales: RewardScalesCfg = RewardScalesCfg()
    only_positive_rewards: bool = False
    tracking_sigma: float = 0.25
    soft_dof_pos_limit: float = 0.9  # eigenbot override
    soft_dof_vel_limit: float = 2.5  # eigenbot override
    soft_torque_limit: float = 1.0
    base_height_target: float = 0.25  # eigenbot override
    max_contact_force: float = 100.0
    torque_limit_hard: float = 8.0
    contact_tresh: float = 0.5
    exp_coeff_rule3: float = -10.0
    stumble_tresh: float = 2.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (0.0, 0.5)
    lin_vel_y: tuple = (0.0, 0.0)
    ang_vel_yaw: tuple = (-1.0, 1.0)
    heading: tuple = (-math.pi / 3, math.pi / 3)


@configclass
class CommandsCfg:
    num_commands: int = NUM_COMMANDS
    resampling_time: float = 10.0
    heading_command: bool = True
    lin_vel_clip: float = 0.1
    rand_heading: bool = True
    curriculum: bool = True
    max_curriculum: float = 1.0
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class DomainRandCfg:
    randomize_friction: bool = True
    friction_range: tuple = (0.5, 1.25)
    randomize_base_mass: bool = False
    added_mass_range: tuple = (-1.0, 1.0)
    randomize_base_com: bool = True
    added_com_range: tuple = (-0.2, 0.2)
    push_robots: bool = True
    push_interval_s: float = 15.0
    max_push_vel_xy: float = 1.0
    randomize_motor: bool = True
    motor_strength_range: tuple = (0.8, 1.2)
    action_delay: bool = False
    action_buf_len: int = 8


@configclass
class ObsScalesCfg:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    height_measurements: float = 5.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0


@configclass
class NoiseScalesCfg:
    dof_pos: float = 0.01
    orientation: float = 0.03
    dof_vel: float = 1.5
    lin_vel: float = 0.1
    ang_vel: float = 0.2
    gravity: float = 0.05
    height_measurements: float = 0.1


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_level: float = 1.0
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


EIGENBOT_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)


# ---------------------------------------------------------------------------
# Asset body-name lists for contact sensing
# ---------------------------------------------------------------------------
FEET_BODIES = [
    "foot_input_M25_S25", "foot_input_M26_S26", "foot_input_M27_S27",
    "foot_input_M28_S28", "foot_input_M29_S29", "foot_input_M30_S30",
]

PENALIZE_CONTACT_BODIES = [
    f"bendy_input_M{i}_S{i}" for i in range(1, 19)
]

TERMINATE_CONTACT_BODIES = ["base_link"]


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------
@configclass
class EigenbotEnvCfg(DirectRLEnvCfg):
    # env
    decimation: int = 4
    episode_length_s: float = 25.0
    action_space: int = NUM_JOINTS
    observation_space: int = NUM_OBSERVATIONS
    state_space: int = 0

    # simulation – match legacy: dt=0.005, gravity=-9.81
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
        ),
    )

    # robot articulation
    robot_cfg: ArticulationCfg = EIGENBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # contact sensor – track all bodies for contact force access
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=2,
        track_air_time=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # action scale: target angle = action_scale * action + default_joint_pos
    action_scale: float = 0.25

    # terrain (defaults to flat plane; set terrain_type="generator" and
    # terrain_generator=EIGENBOT_ROUGH_TERRAIN_CFG for rough terrain)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # height scanner (12x11 = 132 rays, 0.15m spacing, offset 0.375m forward)
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    # sub-configs
    rewards: RewardsCfg = RewardsCfg()
    commands: CommandsCfg = CommandsCfg()
    domain_rand: DomainRandCfg = DomainRandCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    noise: NoiseCfg = NoiseCfg()
