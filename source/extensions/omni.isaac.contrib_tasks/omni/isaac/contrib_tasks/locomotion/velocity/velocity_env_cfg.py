from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.contrib_tasks.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, 
            dynamic_friction=1.0, 
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            intensity=1000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.13, 0.13, 0.13),
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros",
        ),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )


##
# MDP configuration
##
    

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )        


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True,)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # observation terms (order preserved)
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        actions = ObsTerm(func=mdp.last_action)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["l_ankle_pitch", "r_ankle_pitch"])},
        )
        height_scan = ObsTerm(
            func=mdp.height_scan, 
            params={"sensor_cfg": SceneEntityCfg("height_scanner")}, 
            noise=Unoise(n_min=-0.05, n_max=0.05), 
            clip=(-0.5, 0.5)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # startup
    physics_material = RandTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
    )

    add_base_mass = RandTerm(
        func=mdp.add_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_range": (-5.0, 5.0)},
    )

    # reset
    base_external_force_torque = RandTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    reset_scene = RandTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_base = RandTerm(
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

    reset_robot_joints = RandTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )
    
    # interval
    push_robot = RandTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5),}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- "True" rewards
    # Reward for staying alive
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # Rewards for tracking the velocity command
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # Penalize z-axis base linear velocity using L2-kernel
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # Penalize xy-axis base angular velocity using L2-kernel
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.0)
    # Penalize joint torques applied on the articulation using L2-kernel
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    # Penalize joint accelerations on the articulation using L2-kernel
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    # Penalize large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # Penalize the rate of change of the actions
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # Penalize undesired contacts with the ground
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, 
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["head_.*", "waist_.*", ".*_thigh_.*"]), 
            "threshold": 1.0
        },
    )
    # Penalize joint positions if they cross the soft limits
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.0)
    # Penalize joint torques if they cross the soft limits
    torques_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.0)


    # -- "Auxiliary" rewards
    # Reward for tracking upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length exceeds the maximum duration
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if impact on selected body
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )               
    # (3) Terminate if the robot falls
    # TODO: use rellative height from the contact surface instead of absolute height
    torso_height = DoneTerm(func=mdp.base_height, params={"minimum_height": 0.3})
    # (4) Terminate if the robot tilted too much
    torso_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.25})


@configclass
class CurriculumCfg:
    """Curriculum configuration for the MDP."""

    # (1) Curriculum for the velocity command
    velocity_command = CurrTerm(
        func=mdp.curriculum_velocity_command,
        params={
            "command_name": "base_velocity",
            "resampling_time_range": (10.0, 10.0),
            "rel_standing_envs": 0.02,
            "rel_heading_envs": 1.0,
            "heading_command": True,
            "ranges": {
                "lin_vel_x": (-1.0, 1.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-1.0, 1.0),
                "heading": (-math.pi, math.pi),
            },
        },
    )
    # (2) Curriculum for the terrain
    terrain = CurrTerm(
        func=mdp.curriculum_terrain,
        params={
            "terrain_type": "generator",
            "terrain_generator": ROUGH_TERRAINS_CFG,
            "max_init_terrain_level": 3,
            "collision_group": -1,
            "physics_material": sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
            ),
            "visual_material": sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                project_uvw=True,
            ),
        },
    )
    # (3) Curriculum for the robot
    robot = CurrTerm(
        func=mdp.curriculum_robot,
        params={
            "articulation_enabled": True,
            "enabled_self_collisions": True,
            "solver_position_iteration_count": 8,
            "solver_velocity_iteration_count": 1,
            "sleep_threshold": 0.005,
            "stabilization_threshold": 0.005,
        },
    )
    # (4) Curriculum for the robot's initial state
    robot_init_state = CurrTerm(
        func=mdp.curriculum_robot_init_state,
        params={
            "pos": (0.0, 0.0, 0.935),
            "rot": (1.0, 0.0, 0.0,),
            "joint_pos": {
                "l_hip_roll": 0.0,
                "l_hip_yaw": 0.0,
                "l_hip_pitch": -0.5236,
                "l_knee_pitch": 1.0472,
                "l_ankle_pitch": -0.5236,
                "l_ankle_roll": 0.0,
                "r_hip_roll": -0.0,
                "r_hip_yaw": 0.0,
                "r_hip_pitch": -0.5236,
                "r_knee_pitch": 1.0472,
                "r_ankle_pitch": -0.5236,
                "r_ankle_roll": 0.0,
                "waist_yaw": 0.0,
                "waist_pitch": 0.1,
                "waist_roll": 0.0,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "head_roll": 0.0,
                "l_shoulder_pitch": 0.0,
                "l_shoulder_roll": 0.3,
                "l_shoulder_yaw": 0.3,
                "l_elbow_pitch": -0.1,
                "l_wrist_yaw": 0.0,
                "l_wrist_roll": 0.0,
                "l_wrist_pitch": 0.0,
                "r_shoulder_pitch": 0.0,
                "r_shoulder_roll": -0.3,
                "r_shoulder_yaw": 0.3,
                "r_elbow_pitch": -0.1,
                "r_wrist_yaw": 0.0,
                "r_wrist_roll": 0.0,
                "r_wrist_pitch": 0.0,
            },
            "joint_vel": {".*": 0.0},
        },
    )


##
# Environment configuration
##
    

@configclass
class LocomotionVelocityRoughEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion task with velocity-tracking commands and rough terrain."""

    # Scene
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 0.001
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics udpate period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        if self.scene.camera is not None:
            self.scene.camera.update_period = self.sim.dt * self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt 

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and facilitates agents to adapt to different terrains
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
