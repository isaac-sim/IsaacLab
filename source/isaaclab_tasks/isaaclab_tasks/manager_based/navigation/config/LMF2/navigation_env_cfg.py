# Copyright (c) 2022-20 , The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationWithThrustersCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
# from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors.ray_caster.ray_caster_camera_cfg import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .vae.vae_image_encoder import VAEImageEncoder

##
# Pre-defined configs
##
from isaaclab.terrains.config.floating_obstacles import FLOATING_OBSTACLES_CFG 
from isaaclab.controllers.lee_velocity_control_cfg import LeeVelControllerCfg

##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a flying robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLOATING_OBSTACLES_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=None,
        debug_vis=False,
    )
    
    # robots
    robot: ArticulationWithThrustersCfg = MISSING
    # sensors
    depth_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        update_period=0.1,
        pattern_cfg=PinholeCameraPatternCfg(
            width=480, height=270
        ),
        data_types=["distance_to_image_plane"],
        max_distance=10.0,
        depth_clipping_behavior="max",
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=10,
        debug_vis=False,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="base_link",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(10.5, 11.5),
            pos_y=(1.0, 7.0),
            pos_z=(1.0, 5.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_commands = mdp.NavigationActionCfg(asset_name="robot", 
                                     joint_names=[".*"], 
                                     scale=1.0, 
                                     use_default_offset=False,
                                     controller_cfg=LeeVelControllerCfg())

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_link_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_roll_pitch = ObsTerm(func=mdp.base_roll_pitch, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        last_action = ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.0, n_max=0.0))
        depth_latent = ObsTerm(
            func=mdp.image_latents,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane", "vae": VAEImageEncoder},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.5, 1.5),
                "y": (1.0, 7.0),
                "z": (1.0, 5.0),
                "yaw": (-math.pi / 2.0, math.pi / 2.0),
            },  # yaw translated from xyzw (0, 0, 0.5236, 1) from aerial gym
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    goal_dist_exp1 = RewTerm(func=mdp.distance_to_goal_exp, weight=2.0,
                             params={
                                 "asset_cfg": SceneEntityCfg("robot"), 
                                 "std": 7.0,
                                 "command_name": "target_pose",
                                     })
    goal_dist_exp2 = RewTerm(func=mdp.distance_to_goal_exp, weight=4.0,
                             params={
                                 "asset_cfg": SceneEntityCfg("robot"), 
                                 "std": 0.5,
                                 "command_name": "target_pose",
                                 })
    velocity_reward = RewTerm(func=mdp.velocity_to_goal_reward, weight=0.5,
                              params={
                                  "asset_cfg": SceneEntityCfg("robot"), 
                                  "command_name": "target_pose",
                                  })
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-100.0,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    collision = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0},
        time_out=False,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel,
                              params={
                                  "asset_cfg": SceneEntityCfg("robot"), 
                                  "command_name": "target_pose"},
                                  )

##
# Environment configuration
##

@configclass
class NavigationVelocityFloatingObstacleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
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
