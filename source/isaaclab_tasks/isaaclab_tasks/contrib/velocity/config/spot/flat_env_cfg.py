# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import config_field, replace_config
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.utils import PresetCfg


@dataclass
class PhysicsCfg(PresetCfg):
    isaacsim_physx: Any = config_field(PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
    physx: Any = config_field(PhysxAutoCfg(isaacsim_physx=isaacsim_physx))
    default: Any = config_field(isaacsim_physx)
    newton_mjwarp: Any = config_field(
        NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=130,
                nconmax=40,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
            num_substeps=2,
            debug_mode=False,
            default_shape_cfg=NewtonShapeCfg(margin=0.01),
        )
    )
    newton_kamino: Any = config_field(NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64)))


##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)


@dataclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    joint_pos: Any = config_field(
        mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)
    )


@dataclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity: Any = config_field(
        mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
            ),
        )
    )


@dataclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel: Any = config_field(
            ObsTerm(
                func=mdp.base_lin_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.1, n_max=0.1),
            )
        )
        base_ang_vel: Any = config_field(
            ObsTerm(
                func=mdp.base_ang_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.1, n_max=0.1),
            )
        )
        projected_gravity: Any = config_field(
            ObsTerm(
                func=mdp.projected_gravity,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )
        )
        velocity_commands: Any = config_field(
            ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        )
        joint_pos: Any = config_field(
            ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )
        )
        joint_vel: Any = config_field(
            ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.5, n_max=0.5),
            )
        )
        actions: Any = config_field(ObsTerm(func=mdp.last_action))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = config_field(PolicyCfg())


@dataclass
class SpotNewtonEventCfg:
    """Newton event configuration for Spot (reset + interval only)."""

    # reset
    base_external_force_torque: Any = config_field(
        EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="body"),
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
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-1.5, 1.5),
                    "y": (-1.0, 1.0),
                    "z": (-0.5, 0.5),
                    "roll": (-0.7, 0.7),
                    "pitch": (-0.7, 0.7),
                    "yaw": (-1.0, 1.0),
                },
            },
        )
    )

    reset_robot_joints: Any = config_field(
        EventTerm(
            func=spot_mdp.reset_joints_around_default,
            mode="reset",
            params={
                "position_range": (-0.2, 0.2),
                "velocity_range": (-2.5, 2.5),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    )

    # interval
    push_robot: Any = config_field(
        EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            },
        )
    )


@dataclass
class SpotStartupEventCfg:
    """PhysX-only startup randomization for Spot."""

    # startup
    physics_material: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (0.3, 0.8),
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
                "asset_cfg": SceneEntityCfg("robot", body_names="body"),
                "mass_distribution_params": (-2.5, 2.5),
                "operation": "add",
            },
        )
    )


@dataclass
class SpotPhysxEventCfg(SpotNewtonEventCfg, SpotStartupEventCfg):
    pass


@dataclass
class SpotEventCfg(PresetCfg):
    physx: Any = config_field(SpotPhysxEventCfg())
    isaacsim_physx: Any = config_field(physx)
    default: Any = config_field(isaacsim_physx)
    newton_mjwarp: Any = config_field(SpotNewtonEventCfg())
    newton_kamino: Any = config_field(newton_mjwarp)


@dataclass
class SpotRewardsCfg:
    # -- task
    air_time: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.air_time_reward,
            weight=5.0,
            params={
                "mode_time": 0.3,
                "velocity_threshold": 0.5,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
    )
    base_angular_velocity: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.base_angular_velocity_reward,
            weight=5.0,
            params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
        )
    )
    base_linear_velocity: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.base_linear_velocity_reward,
            weight=5.0,
            params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
        )
    )
    foot_clearance: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.foot_clearance_reward,
            weight=0.5,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
    )
    gait: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.GaitReward,
            weight=10.0,
            params={
                "std": 0.1,
                "max_err": 0.2,
                "velocity_threshold": 0.5,
                "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
            },
        )
    )

    # -- penalties
    action_smoothness: Any = config_field(RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0))
    air_time_variance: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.air_time_variance_penalty,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
    )
    base_motion: Any = config_field(
        RewardTermCfg(func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    )
    base_orientation: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
    )
    foot_slip: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.foot_slip_penalty,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )
    )
    joint_acc: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.joint_acceleration_penalty,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
        )
    )
    joint_pos: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.joint_position_penalty,
            weight=-0.7,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.5,
            },
        )
    )
    joint_torques: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.joint_torques_penalty,
            weight=-5.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
    )
    joint_vel: Any = config_field(
        RewardTermCfg(
            func=spot_mdp.joint_velocity_penalty,
            weight=-1.0e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
        )
    )


@dataclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))
    body_contact: Any = config_field(
        DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
        )
    )
    terrain_out_of_bounds: Any = config_field(
        DoneTerm(
            func=mdp.terrain_out_of_bounds,
            params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
            time_out=True,
        )
    )


@dataclass
class SpotFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the Spot robot in a flat environment."""

    sim: SimulationCfg = config_field(SimulationCfg(physics=PhysicsCfg()))

    # Basic settings
    observations: SpotObservationsCfg = config_field(SpotObservationsCfg())
    actions: SpotActionsCfg = config_field(SpotActionsCfg())
    commands: SpotCommandsCfg = config_field(SpotCommandsCfg())

    # MDP setting
    rewards: SpotRewardsCfg = config_field(SpotRewardsCfg())
    terminations: SpotTerminationsCfg = config_field(SpotTerminationsCfg())
    events: SpotEventCfg = config_field(SpotEventCfg())

    def __post_init__(self):
        # post init of parent
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physics = PhysicsCfg()
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Spot-d
        self.scene.robot = replace_config(SPOT_CFG, prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=PhysxRigidBodyMaterialCfg(
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
            debug_vis=True,
        )

        # no height scan
        self.scene.height_scanner = None
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(10.5, 10.5, 0.3))
