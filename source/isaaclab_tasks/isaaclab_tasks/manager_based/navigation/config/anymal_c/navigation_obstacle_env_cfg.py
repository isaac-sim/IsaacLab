# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg

LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform_angular,
        mode="reset",
        params={
            "radius_range": (1.5, 2.5),
            "angle_range": (-3.14, 3.14),
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    class SensorCfg(ObsGroup):
        """Observations for sensor group."""

        camera_data = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera")})

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    sensor: SensorCfg = SensorCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=10,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=10,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-4.0,
        params={"command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.ObstaclePose2dCommandCfg(
        asset_name="robot",
        object_name="object",
        simple_heading=False,
        resampling_time_range=(12.0, 12.0),
        debug_vis=True,
        ranges=mdp.ObstaclePose2dCommandCfg.Ranges(heading=(-math.pi, math.pi)),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


@configclass
class NavigationObstacleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.scene.env_spacing = 20
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
            update_period=0.1,
            height=64,
            width=64,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object",
            spawn=sim_utils.CylinderCfg(
                radius=0.2,
                height=0.6,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=200.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0)),
        )


class NavigationObstacleEnvCfg_PLAY(NavigationObstacleEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 20
        # disable randomization for play
        self.observations.policy.enable_corruption = False
