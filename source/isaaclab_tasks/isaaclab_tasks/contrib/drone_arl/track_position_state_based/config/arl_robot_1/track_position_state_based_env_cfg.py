# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING, dataclass
from typing import Any

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import config_field
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.assets import MultirotorCfg

import isaaclab_tasks.contrib.drone_arl.mdp as mdp
from isaaclab_tasks.contrib.drone_arl.mdp.commands import DroneUniformPoseCommandCfg
from isaaclab_tasks.contrib.drone_arl.mdp.rewards import (
    ang_vel_xyz_exp,
    distance_to_goal_exp,
    lin_vel_xyz_exp,
    yaw_aligned,
)


##
# Scene definition
##
@dataclass
class ArlTrackPositionStateBasedSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a flying robot."""

    # robots
    robot: MultirotorCfg = config_field(MISSING)

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

    target_pose: Any = config_field(
        DroneUniformPoseCommandCfg(
            asset_name="robot",
            body_name="base_link",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            ranges=DroneUniformPoseCommandCfg.Ranges(
                pos_x=(-0.0, 0.0),
                pos_y=(-0.0, 0.0),
                pos_z=(-0.0, 0.0),
                roll=(-0.0, 0.0),
                pitch=(-0.0, 0.0),
                yaw=(-0.0, 0.0),
            ),
        )
    )


@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    thrust_command: Any = config_field(
        mdp.ThrustActionCfg(
            asset_name="robot",
            scale=3.0,
            offset=3.0,
            preserve_order=False,
            use_default_offset=False,
            clip={
                "back_left_prop": (0.0, 6.0),
                "back_right_prop": (0.0, 6.0),
                "front_left_prop": (0.0, 6.0),
                "front_right_prop": (0.0, 6.0),
            },
        )
    )


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_link_position: Any = config_field(ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1)))
        base_orientation: Any = config_field(ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1)))
        base_lin_vel: Any = config_field(ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)))
        base_ang_vel: Any = config_field(ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1)))
        last_action: Any = config_field(ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.0, n_max=0.0)))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = config_field(PolicyCfg())


@dataclass
class EventCfg:
    """Configuration for events."""

    # reset

    reset_base: Any = config_field(
        EventTerm(
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
    )


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    distance_to_goal_exp: Any = config_field(
        RewTerm(
            func=distance_to_goal_exp,
            weight=25.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "std": 1.5,
                "command_name": "target_pose",
            },
        )
    )
    flat_orientation_l2: Any = config_field(
        RewTerm(
            func=mdp.flat_orientation_l2,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    )
    yaw_aligned: Any = config_field(
        RewTerm(
            func=yaw_aligned,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.0},
        )
    )
    lin_vel_xyz_exp: Any = config_field(
        RewTerm(
            func=lin_vel_xyz_exp,
            weight=2.5,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 2.0},
        )
    )
    ang_vel_xyz_exp: Any = config_field(
        RewTerm(
            func=ang_vel_xyz_exp,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 10.0},
        )
    )
    action_rate_l2: Any = config_field(RewTerm(func=mdp.action_rate_l2, weight=-0.05))
    action_magnitude_l2: Any = config_field(RewTerm(func=mdp.action_l2, weight=-0.05))

    termination_penalty: Any = config_field(
        RewTerm(
            func=mdp.is_terminated,
            weight=-5.0,
        )
    )


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))
    crash: Any = config_field(DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -3.0}))


##
# Environment configuration
##


@dataclass
class TrackPositionNoObstaclesEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the state-based drone pose-control environment."""

    # Scene settings
    scene: ArlTrackPositionStateBasedSceneCfg = config_field(
        ArlTrackPositionStateBasedSceneCfg(num_envs=4096, env_spacing=2.5)
    )
    # Basic settings
    observations: ObservationsCfg = config_field(ObservationsCfg())
    actions: ActionsCfg = config_field(ActionsCfg())
    commands: CommandsCfg = config_field(CommandsCfg())
    # MDP settings
    rewards: RewardsCfg = config_field(RewardsCfg())
    terminations: TerminationsCfg = config_field(TerminationsCfg())
    events: EventCfg = config_field(EventCfg())

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
        # ThrusterCfg is implemented in Isaac Lab and has no Newton-native execution path.
        self.sim.use_newton_actuators = False
