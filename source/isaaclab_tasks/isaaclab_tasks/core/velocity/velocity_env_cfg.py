# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING, dataclass

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import config_field
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.utils import PresetCfg

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from typing import Any

##
# Physics presets
##


@dataclass
class RoughPhysicsCfg(PresetCfg):
    """Shared backend presets for locomotion velocity environments."""

    isaacsim_physx: Any = config_field(PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
    ovphysx: Any = config_field(OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
    physx: Any = config_field(PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx))
    newton_mjwarp: Any = config_field(
        NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=1000,
                nconmax=300,
                cone="pyramidal",
                impratio=1.0,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
            num_substeps=2,
            debug_mode=False,
            default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        )
    )
    newton_kamino: Any = config_field(NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64)))
    default: Any = config_field(newton_mjwarp)


##
# Scene definition
##


@dataclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain: Any = config_field(
        TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
    )
    # robots
    robot: ArticulationCfg = config_field(MISSING)
    # sensors -- the concrete implementation is selected automatically from the active physics
    # backend (Newton / PhysX / OvPhysX); backend-specific fields such as ``global_world_only`` are
    # documented on the config and ignored by the backends that do not use them.
    height_scanner: Any = config_field(
        RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            global_world_only=True,
        )
    )
    contact_forces: Any = config_field(
        ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True)
    )
    # lights
    sky_light: Any = config_field(
        AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
    )


##
# MDP settings
##


@dataclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity: Any = config_field(
        mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        )
    )


@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos: Any = config_field(
        mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    )


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel: Any = config_field(ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)))
        base_ang_vel: Any = config_field(ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)))
        projected_gravity: Any = config_field(
            ObsTerm(
                func=mdp.projected_gravity,
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )
        )
        velocity_commands: Any = config_field(
            ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        )
        joint_pos: Any = config_field(ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)))
        joint_vel: Any = config_field(ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)))
        actions: Any = config_field(ObsTerm(func=mdp.last_action))
        height_scan: Any = config_field(
            ObsTerm(
                func=mdp.height_scan,
                params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                noise=Unoise(n_min=-0.1, n_max=0.1),
                clip=(-1.0, 1.0),
            )
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = config_field(PolicyCfg())


@dataclass
class EventsCfg:
    """Configuration for events."""

    # startup
    physics_material: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 0.8),
                "dynamic_friction_range": (0.6, 0.6),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
    )

    add_base_mass: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                # Multiplicative ±25% log-uniform. Scale-invariant across robot sizes
                # (no per-robot kg overrides needed) with geometric mean 1.0 and
                # symmetric inverse perturbation (acceleration symmetric around nominal).
                "mass_distribution_params": (1 / 1.25, 1.25),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
    )

    base_com: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
            },
        )
    )

    # reset
    base_external_force_torque: Any = config_field(
        EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "force_range": (0.0, 0.0),
                "torque_range": (-0.0, 0.0),
            },
        )
    )

    reset_base: Any = config_field(
        EventTerm(
            func=mdp.reset_root_state_uniform,
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
    )

    reset_robot_joints: Any = config_field(
        EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
            },
        )
    )

    # interval
    push_robot: Any = config_field(
        EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
    )


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp: Any = config_field(
        RewTerm(
            func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
        )
    )
    track_ang_vel_z_exp: Any = config_field(
        RewTerm(
            func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
        )
    )
    # -- penalties
    lin_vel_z_l2: Any = config_field(RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0))
    ang_vel_xy_l2: Any = config_field(RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05))
    dof_torques_l2: Any = config_field(RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5))
    dof_acc_l2: Any = config_field(RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7))
    action_rate_l2: Any = config_field(RewTerm(func=mdp.action_rate_l2, weight=-0.01))
    feet_air_time: Any = config_field(
        RewTerm(
            func=mdp.feet_air_time,
            weight=0.125,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )
    )
    undesired_contacts: Any = config_field(
        RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
        )
    )
    # -- optional penalties
    flat_orientation_l2: Any = config_field(RewTerm(func=mdp.flat_orientation_l2, weight=0.0))
    dof_pos_limits: Any = config_field(RewTerm(func=mdp.joint_pos_limits, weight=0.0))


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))
    base_contact: Any = config_field(
        DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
        )
    )


@dataclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels: Any = config_field(CurrTerm(func=mdp.terrain_levels_vel))


##
# Environment configuration
##


@dataclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Simulation settings — shared physics preset (PhysX + MJWarp) for all rough-terrain envs
    sim: SimulationCfg = config_field(SimulationCfg(physics=RoughPhysicsCfg()))
    # Scene settings
    scene: MySceneCfg = config_field(MySceneCfg(num_envs=4096, env_spacing=2.5))
    # Basic settings
    observations: ObservationsCfg = config_field(ObservationsCfg())
    actions: ActionsCfg = config_field(ActionsCfg())
    commands: CommandsCfg = config_field(CommandsCfg())
    # MDP settings
    rewards: RewardsCfg = config_field(RewardsCfg())
    terminations: TerminationsCfg = config_field(TerminationsCfg())
    events: EventsCfg = config_field(EventsCfg())
    curriculum: CurriculumCfg = config_field(CurriculumCfg())

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def play_mode(self):
        """Play-mode overrides shared by the velocity-tracking environments."""
        super().play_mode()
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # remove random pushing events
        self.events.base_external_force_torque = None
        self.events.push_robot = None
