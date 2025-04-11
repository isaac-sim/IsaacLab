# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import berkeley_humanoid.tasks.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from berkeley_humanoid.terrains.terrain_generator_cfg import ROUGH_TERRAINS_CFG

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
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
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.LidarPatternCfg(channels=2,
    #                                          vertical_fov_range=(-15.0, 15.0),
    #                                          horizontal_fov_range=(0.0, 360.0),
    #                                          horizontal_res=12.0,
    #                                          ),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3,
                                                track_air_time=True, track_pose=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10, 10.0), 
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=1,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #     lin_vel_x=(-1.0,1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0) )
        # lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(0.0,0.0) )
    # )

    base_position = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*upper_leg",
        resampling_time_range=(10, 10.0),
        #ranges=mdp.UniformPoseCommandCfg.Ranges(pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(-0.3, 0.15 ), yaw=(0.0, 0.0))
        ranges=mdp.UniformPoseCommandCfg.Ranges(pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(0.15, 0.15), yaw=(0.0, 0.0))
    )
    # base_position = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=".*upper_leg",
    #     resampling_time_range=(10, 10.0),
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(-0.28,-0.28 ), yaw=(0.0, 0.0))
    # )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*upper_leg"], scale=1.0, use_default_offset=True)

    # wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*wheel"], scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    class DLP(ObsGroup):
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_position = ObsTerm(func=mdp.joint_leg_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Unoise(n_min=-0.05, n_max=0.05))
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # up_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05),
        #                   params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*upper_leg"])})
        # wheel_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.08, n_max=0.08),
        #                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*wheel"])})
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))



        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },      
    )

    scale_all_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.5, 1.5),
                "operation": "scale"},
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (-0.01, 0.5),
                "operation": "add"},
    )

    # scale_all_joint_armature = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*upper_leg"]), "armature_distribution_params": (1.0, 1.05),
    #             "operation": "scale"},
    # )

    add_all_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "pos_distribution_params": (-0.1, 0.15),
                "operation": "add"},
    )

    scale_all_joint_friction_model = EventTerm(
        func=mdp.randomize_joint_friction_model,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "friction_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )

    # # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # track_velocity = RewTerm(func=mdp.track_velocity, weight=1.0, params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=[".*upper_leg"])})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*lower_leg"]), "threshold": 1.0},)

    joint_pos_pitch_exp = RewTerm(func=mdp.joint_pos_pitch_exp, weight=1.0, params={"command_name": "base_position", 
                                                                                    "std": math.sqrt(0.1),
                                                                                    "asset_cfg":
                                                                                    SceneEntityCfg("robot", joint_names=[".*upper_leg"])})

        # -- task #선형 속도 보상 함수 math.sqrt(0.5)
    # move = {
    # track_lin_vel_x_exp = RewTerm(
    #     func=mdp.track_lin_vel_x_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.0625)}
    # )
    # # 회전 보상 함수
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.1, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # jump_reward = RewTerm(
    #     func=mdp.base_height, weight=0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*wheel"),
    #             "sensor_cfg": SceneEntityCfg("height_scanner"),
    #             "target_height": 0.05}
    # )
    # penalize_opposite_direction = RewTerm(
    #     func=mdp.penalize_opposite_direction, weight=-0.1,
    #     params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")}
    # )
    # # y축 방향 회전 보상 함수
    # track_yaw_for_y_vel = RewTerm(
    # func=mdp.track_yaw_for_y_vel, weight=0.7, params={"command_name": "base_velocity", "std": math.sqrt(0.3)}
    # )
    # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z, weight=0.05,
    #                         params={"asset_cfg": SceneEntityCfg("robot"),
    #                             "sensor_cfg": SceneEntityCfg("height_scanner")})
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=-0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*wheel"),
    #         "threshold_min": 0.1,
    #         "threshold_max": 0.5,
    #         # "sensor_cfg2": SceneEntityCfg("height_scanner"),
    #     },
    # )
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot")})


    # joint_deviation_upper_leg = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight= 0.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*upper_leg"])},
    # )
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
        # -- optional penalties
        # "flat_orientation_l2" : RewTerm(func=mdp.flat_orientation_l2, weight=-5.0),
        # "wheel_flat_orientation_l2" : RewTerm(func=mdp.wheel_y_flat_orientation_l2, weight=-5.0),
        # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    # }

    # jump = {
    #     "track_lin_vel_xy_exp" : RewTerm( 
    #         func=mdp.track_lin_vel_x_exp, weight=1.25, params={"command_name": "jump_velocity", "std": math.sqrt(0.0625)}
    #     ),
    #     # 회전 보상 함수
    #     "track_ang_vel_z_exp" : RewTerm(
    #         func=mdp.track_ang_vel_z_exp, weight=0.4, params={"command_name": "jump_velocity", "std": math.sqrt(0.25)}
    #     ),
    # }



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 기존 타임아웃 조건
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*lower_leg" ]),
            "threshold": 0.5,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # # push force follows curriculum
    # push_force_levels = CurrTerm(func=mdp.modify_push_force,
    #                              params={"term_name": "push_robot", "max_velocity": [3.0, 3.0], "interval": 200 * 24,
    #                                      "starting_step": 300 * 24})
    # change_weight = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name":"wheel_flat_orientation_l2",
    #         "weight":-0.5,
    #         "num_steps": 200 * 24,
    #     }
    # )
    # command_velocity_x = CurrTerm(
    #     func=mdp.modify_command_velocity_x,
    #     params={
    #         "term_name": "track_velocity",
    #         "max_velocity": [-2.0, 2.0],
    #         "interval": 200 * 24,
    #         "starting_step": 200 * 24,
    #         "cmd" : "base_velocity"
    #     }
    # )
    # command_leg_pitch = CurrTerm(
    #     func=mdp.modify_command_leg_pitch,
    #     params={
    #         "term_name": "joint_pos_pitch_exp",
    #         "max_pitch": [-0.3, 0.15],
    #         "interval": 70 * 24,
    #         "cmd" : "base_position"
    #     }
    # )

    # command_velocity_z = CurrTerm(
    #     func=mdp.modify_command_velocity_z,
    #     params={
    #         "term_name": "track_ang_vel_z_exp",
    #         "max_velocity": [-2.0, 2.0],
    #         "interval": 200 * 24,
    #         "starting_step": 1500 * 24,
    #         "cmd" : "base_velocity"
    #     }
    # )



@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: RandomizationCfg = RandomizationCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
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
