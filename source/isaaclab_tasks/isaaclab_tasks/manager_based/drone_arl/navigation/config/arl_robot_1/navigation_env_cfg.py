# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera_cfg import MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_contrib.controllers import LeeVelControllerCfg

import isaaclab_tasks.manager_based.drone_arl.mdp as mdp

##
# Pre-defined configs
##
from .scenes.obstacle_scenes.obstacle_scene import (
    OBSTACLE_SCENE_CFG,
    generate_obstacle_collection,
)


##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene configuration for drone navigation with obstacles."""

    # obstacles
    object_collection = generate_obstacle_collection(OBSTACLE_SCENE_CFG)

    # robots
    robot: MultirotorCfg = MISSING

    # sensors
    depth_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        mesh_prim_paths=[
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr=f"{{ENV_REGEX_NS}}/obstacle_{wall_name}", is_shared=False, track_mesh_transforms=True
            )
            for wall_name, _ in OBSTACLE_SCENE_CFG.wall_cfgs.items()
        ]
        + [
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr=f"{{ENV_REGEX_NS}}/obstacle_{i}", is_shared=False, track_mesh_transforms=True
            )
            for i in range(OBSTACLE_SCENE_CFG.max_num_obstacles)
        ],
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
        ),
        update_period=0.1,
        pattern_cfg=PinholeCameraPatternCfg(
            width=480, height=270, focal_length=0.193, horizontal_aperture=0.36, vertical_aperture=0.21
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

    target_pose = mdp.DroneUniformPoseCommandCfg(
        asset_name="robot",
        body_name="base_link",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.DroneUniformPoseCommandCfg.Ranges(
            pos_x=(4.0, 5.0),
            pos_y=(-3.0, 3.0),
            pos_z=(1.0, 5.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_commands = mdp.NavigationActionCfg(
        asset_name="robot",
        scale=1.0,
        offset=0.0,
        preserve_order=False,
        use_default_offset=False,
        controller_cfg=LeeVelControllerCfg(
            K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
            K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
            max_inclination_angle_rad=1.0471975511965976,
            max_yaw_rate=1.0471975511965976,
        ),
        max_magnitude=2.0,
        max_yawrate=3.14 / 3.0,
        max_inclination_angle=3.14 / 4.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_link_position = ObsTerm(
            func=mdp.generated_drone_commands,
            params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_roll_pitch = ObsTerm(func=mdp.base_roll_pitch, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        last_action = ObsTerm(
            func=mdp.last_action_navigation,
            params={"action_name": "velocity_commands"},
        )
        depth_latent = ObsTerm(
            func=mdp.ImageLatentObservation,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
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
                "x": (-5.0, -4.5),
                "y": (-3.0, 3.0),
                "z": (1.0, 5.0),
                "yaw": (-math.pi / 6.0, math.pi / 6.0),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_obstacles = EventTerm(
        func=mdp.events.reset_obstacles_with_individual_ranges,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object_collection"),
            "obstacle_configs": OBSTACLE_SCENE_CFG.obstacle_cfgs,
            "wall_configs": OBSTACLE_SCENE_CFG.wall_cfgs,
            "env_size": OBSTACLE_SCENE_CFG.env_size,
            "use_curriculum": True,
            "min_num_obstacles": OBSTACLE_SCENE_CFG.min_num_obstacles,
            "max_num_obstacles": OBSTACLE_SCENE_CFG.max_num_obstacles,
            "ground_offset": OBSTACLE_SCENE_CFG.ground_offset,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    goal_dist_exp1 = RewTerm(
        func=mdp.distance_to_goal_exp_curriculum,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 7.0,
            "command_name": "target_pose",
        },
    )
    goal_dist_exp2 = RewTerm(
        func=mdp.distance_to_goal_exp_curriculum,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.5,
            "command_name": "target_pose",
        },
    )
    velocity_reward = RewTerm(
        func=mdp.velocity_to_goal_reward_curriculum,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "target_pose",
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude_l2 = RewTerm(func=mdp.action_l2, weight=-0.05)

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

    obstacle_levels = CurrTerm(
        func=mdp.obstacle_density_curriculum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_difficulty": 10,
            "min_difficulty": 0,
        },
    )


##
# Environment configuration
##


@configclass
class NavigationVelocityFloatingObstacleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=20.5)
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
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physx.gpu_max_rigid_patch_count = 2**21
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
