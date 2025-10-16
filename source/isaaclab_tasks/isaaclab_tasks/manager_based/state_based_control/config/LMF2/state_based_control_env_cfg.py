# Copyright (c) 2022-20 , The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, MultirotorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.state_based_control.mdp as mdp


##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a flying robot."""

    # robots
    robot: MultirotorCfg = MISSING

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
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    thrust_command = mdp.ThrustActionCfg(
        asset_name="robot",
        scale=3.0, 
        offset=3.0,
        preserve_order=False,
        use_default_offset=False,
        clip= {
            "back_left_prop": (0.0, 6.0),  
            "back_right_prop": (0.0, 6.0), 
            "front_left_prop": (0.0, 6.0), 
            "front_right_prop": (0.0, 6.0),  
        }
    )

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
            noise=Unoise(n_min=-0.001, n_max=0.001)
        )
        base_roll_pitch = ObsTerm(func=mdp.base_roll_pitch, noise=Unoise(n_min=-0.001, n_max=0.001))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.001, n_max=0.001))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.001, n_max=0.001))
        last_action = ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.0, n_max=0.0))

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
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 1.0),
                "yaw": (-math.pi / 6.0, math.pi / 6.0),
                "roll": (-math.pi / 6.0, math.pi / 6.0),
                "pitch": (-math.pi / 6.0, math.pi / 6.0),
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    goal_dist_exp1 = RewTerm(func=mdp.distance_to_goal_exp, weight=10.0,
                             params={
                                 "asset_cfg": SceneEntityCfg("robot"), 
                                 "std": 7.0,
                                 "command_name": "target_pose",
                                     })
    goal_dist_exp2 = RewTerm(func=mdp.distance_to_goal_exp, weight=25.0,
                             params={
                                 "asset_cfg": SceneEntityCfg("robot"), 
                                 "std": 1.5,
                                 "command_name": "target_pose",
                                 })
    upright_posture = RewTerm(func=mdp.upright_posture_reward, weight=1.0,
                              params={
                                  "asset_cfg": SceneEntityCfg("robot"),
                                  "std": 5.0 #0.5 
                                  })
    yaw_aligned = RewTerm(func=mdp.yaw_reward, weight=2.0,
                          params={
                              "asset_cfg": SceneEntityCfg("robot"),
                              "std": 1.
                          })
    velocity_reward = RewTerm(func=mdp.velocity_to_goal_reward, weight=4.5,
                              params={
                                  "asset_cfg": SceneEntityCfg("robot"), 
                                  "command_name": "target_pose",
                                  })
    ang_vel_smooth = RewTerm(func=mdp.ang_vel_reward, weight=10.0, 
                             params={
                                 "asset_cfg": SceneEntityCfg("robot"),
                                 "std": 10.0  #1.0
                                 })
    
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude_l2 = RewTerm(func=mdp.action_l2, weight=-0.05)

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )

    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -3.0})

##
# Environment configuration
##

@configclass
class StateBasedControlEmptyEnvCfg(ManagerBasedRLEnvCfg):
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
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
