# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

from omni.isaac.orbit.robots.config.anymal import ANYMAL_C_CFG
from omni.isaac.orbit.robots.legged_robot import LeggedRobotCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, SimCfg, ViewerCfg

##
# Scene settings
##


@configclass
class TerrainCfg:
    """Configuration for terrain to load."""

    # whether to enable or disable rough terrain
    use_default_ground_plane = True
    # usd file to import
    # usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/flat_plane.usd"


@configclass
class MarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd"
    # scale of the asset at import
    scale = [1.0, 0.1, 0.1]  # x,y,z


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Configuration for the goals in the environment."""

    @configclass
    class Ranges:
        """Ranges for the commands."""

        lin_vel_x: Tuple[float, float] = (-1.0, 1.0)  # min max [m/s]
        lin_vel_y: Tuple[float, float] = (-1.0, 1.0)  # min max [m/s]
        ang_vel_yaw: Tuple[float, float] = (-1.5, 1.5)  # min max [rad/s]
        heading: Tuple[float, float] = (-3.14, 3.14)  # [rad]

    curriculum = False
    max_curriculum = 1.0
    num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw
    resampling_time = 4.0  # time before commands are changed [s]
    heading_command = False  # if true: compute ang vel command from heading error
    ranges: Ranges = Ranges()


@configclass
class RandomizationCfg:
    """Randomization of scene at reset."""

    initial_base_position = {"enabled": False, "xy_range": (-1.0, 1.0)}
    """Initial XY position of the base at the time of reset."""

    initial_base_velocity = {"enabled": True, "vel_range": (-0.5, 0.5)}
    """Initial velocity of the base at the time of reset."""

    push_robot = {"enabled": True, "interval_s": 15.0, "vel_xy_range": (-1.0, 1.0)}
    """Pushes the robots at each time interval (in sec) with velocity offset (in m/s)."""

    additive_body_mass = {"enabled": True, "body_name": "base", "range": (-5.0, 5.0)}
    """Adds mass to body on the robot in the specified range."""

    feet_material_properties = {
        "enabled": True,
        "static_friction_range": (0.5, 1.5),
        "dynamic_friction_range": (0.5, 1.5),
        "restitution_range": (0.0, 0.1),
        "num_buckets": 64,
    }
    """Direct randomization of material properties."""


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
        # observation terms (order preserved)
        base_lin_vel = {"noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        base_ang_vel = {"noise": {"name": "uniform", "min": -0.2, "max": 0.2}}
        projected_gravity = {"noise": {"name": "uniform", "min": -0.05, "max": 0.05}}
        velocity_commands = {}
        dof_pos = {"noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        dof_vel = {"noise": {"name": "uniform", "min": -1.5, "max": 1.5}}
        actions = {}

    # global observation settings
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # global settings
    only_positive_rewards: bool = True

    # -- base
    lin_vel_xy_exp = {"weight": 1.0, "std": 0.25}
    ang_vel_z_exp = {"weight": 0.5, "std": 0.25}
    lin_vel_z_l2 = {"weight": -2.0}
    ang_vel_xy_l2 = {"weight": -0.05}
    flat_orientation_l2 = {"weight": -2.0}
    # base_height_l2 = {"weight": -0.5, "target_height": 0.57}

    # -- dof undesirable
    # dof_pos_limits = {"weight": 1e-2}
    # dof_vel_limits = {"weight": 1e-2}
    dof_torques_l2 = {"weight": -0.000025}
    dof_acc_l2 = {"weight": -2.5e-7}
    # dof_vel_l2 = {"weight": 0.0}

    # -- command undesirable
    action_rate_l2 = {"weight": -0.01}
    # applied_torque_limits = {"weight": 1e-2}

    # -- cosmetic
    # Note: This reward is useless till we have a proper contact sensor.
    # feet_air_time = {"weight": 2.0, "time_threshold": 0.5}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    episode_timeout = {"enabled": True}
    """Reset when episode length ended."""
    base_height_fall = {"enabled": True, "min_height": 0.4}
    """Reset when base falls below certain height."""


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 4
    # scaling of input actions
    action_scale = 0.5
    # clipping of input actions
    action_clipping = 100.0


##
# Environment configuration
##


@configclass
class VelocityEnvCfg(IsaacEnvCfg):
    """Configuration for the locomotion velocity environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=2.5, episode_length_s=20.0)
    viewer: ViewerCfg = ViewerCfg()
    # Physics settings
    # disable replicate physics to use physics domain randomization
    # TODO: This is a temporary fix. Should be resolved in the future.
    sim: SimCfg = SimCfg(dt=0.005, substeps=4, replicate_physics=False)

    # Scene Settings
    terrain: TerrainCfg = TerrainCfg()
    robot: LeggedRobotCfg = ANYMAL_C_CFG
    marker: MarkerCfg = MarkerCfg()

    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()
