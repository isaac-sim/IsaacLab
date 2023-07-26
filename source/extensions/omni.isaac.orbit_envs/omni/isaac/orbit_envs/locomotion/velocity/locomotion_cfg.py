# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.orbit.command_generators import UniformVelocityCommandGeneratorCfg
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.robots.config.anymal import ANYMAL_C_CFG
from omni.isaac.orbit.robots.legged_robot import LeggedRobotCfg
from omni.isaac.orbit.sensors.contact_sensor import ContactSensorCfg
from omni.isaac.orbit.sensors.ray_caster import GridPatternCfg, RayCasterCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.orbit_envs.locomotion.curriculum as Cur
import omni.isaac.orbit_envs.locomotion.observations as Obs
import omni.isaac.orbit_envs.locomotion.randomizations as Rand
import omni.isaac.orbit_envs.locomotion.rewards as Rew
import omni.isaac.orbit_envs.locomotion.terminations as Done
from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, SimCfg, ViewerCfg
from omni.isaac.orbit_envs.locomotion.actions import JointPositionActionCfg

##
# MDP settings
##


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # startup
    physics_material = RandTerm(
        func=Rand.physics_material,
        mode="startup",
        asset_name="robot",
        body_names=".*",
        params={
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = RandTerm(
        func=Rand.add_body_mass,
        mode="startup",
        asset_name="robot",
        body_names="base",
        params={"mass_range": (-5.0, 5.0)},
    )

    # reset
    base_external_force_torqe = RandTerm(
        func=Rand.apply_external_force_torqe,
        mode="reset",
        asset_name="robot",
        body_names="base",
        params={
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = RandTerm(
        func=Rand.reset_robot_root,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = RandTerm(
        func=Rand.reset_robot_joints_scale_defaults,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = RandTerm(
        func=Rand.push_robot,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class ActionsCfg:
    joint_pos: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot", joint_name_expr=".*", scale=0.5, offset_with_default=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
        concatenate_terms: bool = True
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=Obs.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=Obs.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=Obs.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=Obs.velocity_commands)
        dof_pos = ObsTerm(func=Obs.dof_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        dof_vel = ObsTerm(func=Obs.dof_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=Obs.actions)
        height_scan = ObsTerm(func=Obs.height_scan, sensor_name="height_scanner", noise=Unoise(n_min=-0.1, n_max=0.1))

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(func=Rew.track_lin_vel_xy_exp, weight=1.0, params={"scale": 0.25})
    track_ang_vel_z_exp = RewTerm(func=Rew.track_ang_vel_z_exp, weight=0.5, params={"scale": 0.25})
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=Rew.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=Rew.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=Rew.dof_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=Rew.dof_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=Rew.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=Rew.feet_air_time, sensor_name="contact_forces", weight=0.5, params={"time_threshold": 0.5}
    )
    undesired_contacts = RewTerm(
        func=Rew.undesired_contacts,
        sensor_name="contact_forces",
        asset_name="robot",
        body_names=".*THIGH",
        weight=-1.0,
        params={"threshold": 1.0},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=Done.time_out, time_out=True)
    base_contact = DoneTerm(
        func=Done.illegal_contact,
        sensor_name="contact_forces",
        asset_name="robot",
        body_names="base",
        params={"force_threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=Cur.terrain_levels_vel)


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # decimation: Number of control action updates @ sim dt per policy dt
    # TODO decide where to move this config
    decimation = 4


@configclass
class SensorsCfg:
    """Processing of MDP actions."""

    height_scanner = RayCasterCfg(
        prim_path_expr="Robot/base",
        pos_offset=(0.0, 0.0, 20.0),
        attach_yaw_only=True,
        pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path_expr="Robot/.*", history_length=3)


##
# Environment configuration
##


@configclass
class LocomotionEnvCfg(IsaacEnvCfg):
    """Configuration for the locomotion velocity environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=2.5, episode_length_s=20.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=True)
    # Physics settings
    # disable replicate physics to use physics domain randomization
    # TODO: This is a temporary fix. Should be resolved in the future.
    sim: SimCfg = SimCfg(dt=0.005, substeps=1, replicate_physics=False, disable_contact_processing=True)

    # Scene Settings
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
    )
    robot: LeggedRobotCfg = ANYMAL_C_CFG

    # MDP settings
    commands: UniformVelocityCommandGeneratorCfg = UniformVelocityCommandGeneratorCfg(
        robot_attr="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=False,
        ranges=UniformVelocityCommandGeneratorCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()

    # Sensor settings
    sensors: SensorsCfg = SensorsCfg()
    # We update the height scanner at the same rate as the control frequency
    sensors.height_scanner.update_period = control.decimation * sim.dt
    sensors.contact_forces.update_period = sim.dt


LocomotionEnvRoughCfg = LocomotionEnvCfg


@configclass
class LocomotionEnvRoughCfg_PLAY(LocomotionEnvRoughCfg):
    """Configuration for the locomotion velocity environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=50, env_spacing=2.5, episode_length_s=20.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=True)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        # combine_mode="multiply",
        # static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        improve_patch_friction=True,
    )
    terrain.terrain_generator.num_rows = 5
    terrain.terrain_generator.num_cols = 5
    observations: ObservationsCfg = ObservationsCfg(policy=ObservationsCfg.PolicyCfg(enable_corruption=False))
