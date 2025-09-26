# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Code adapted from https://github.com/leggedrobotics/nav-suite

# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCameraCfg, TiledCameraCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg as LOW_LEVEL_CFGS
import isaaclab_tasks.manager_based.navigation.mdp as mdp

from .terrains import MeshPillarTerrainCfg

##
# Scene definition
##


NAV_TERRAIN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=1.0,
    border_height=-3.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pillar": MeshPillarTerrainCfg(
            proportion=0.8,
            box_objects=MeshPillarTerrainCfg.BoxCfg(
                width=(0.2, 0.4),
                length=(0.2, 0.4),
                height=(0.4, 1.5),
                num_objects=(5, 15),
                max_yx_angle=(0.0, 20.0),
            ),
            cylinder_cfg=MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.1, 0.3),
                height=(0.4, 1.5),
                num_objects=(5, 15),
                max_yx_angle=(0.0, 20.0),
            ),
        ),
        "pillar_road": MeshPillarTerrainCfg(
            proportion=0.2,
            box_objects=MeshPillarTerrainCfg.BoxCfg(
                width=(0.2, 0.4),
                length=(0.2, 0.4),
                height=(0.2, 0.4),
                num_objects=(10, 20),
            ),
            cylinder_cfg=MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.1, 0.3),
                height=(0.4, 1.5),
                num_objects=(5, 15),
                max_yx_angle=(0.0, 20.0),
            ),
            rough_terrain=HfRandomUniformTerrainCfg(
                proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            ),
        ),
    },
)


@configclass
class RayCasterNavSceneCfg(LOW_LEVEL_CFGS.MySceneCfg):
    """Configuration for a scene for training a perceptive navigation policy on an AnymalD Robot."""

    # SENSORS: Navigation Policy
    front_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            height=36,
            width=64,
        ),
        update_period=0,
        debug_vis=False,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.4761, 0.0035, 0.1055),
            rot=(0.9914449, 0.0, 0.1305262, 0.0),
            convention="world",  # 15 degrees downward tilted
        ),
        max_distance=20,
        data_types=["distance_to_image_plane"],
    )

    def __post_init__(self):
        """Post initialization."""
        # swap to navigation terrain
        self.terrain.terrain_generator = NAV_TERRAIN


@configclass
class TiledNavSceneCfg(LOW_LEVEL_CFGS.MySceneCfg):
    """Configuration for a scene for training a perceptive navigation policy on an AnymalD Robot."""

    # SENSORS: Navigation Policy
    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/camera",
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.01, 20.0),
        ),
        height=36,
        width=64,
        update_period=0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.4761, 0.0035, 0.1055),
            rot=(0.9914449, 0.0, 0.1305262, 0.0),
            convention="world",  # 15 degrees downward tilted
        ),
        data_types=["distance_to_image_plane"],
    )

    def __post_init__(self):
        """Post initialization."""
        # swap to navigation terrain
        self.terrain.terrain_generator = NAV_TERRAIN


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_command = mdp.NavigationSE2ActionCfg(
        asset_name="robot",
        low_level_action=LOW_LEVEL_CFGS.ActionsCfg().joint_pos,
        low_level_policy_file=ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt",
        clip_mode="minmax",
        clip=[
            LOW_LEVEL_CFGS.CommandsCfg().base_velocity.ranges.lin_vel_x,
            LOW_LEVEL_CFGS.CommandsCfg().base_velocity.ranges.lin_vel_y,
            LOW_LEVEL_CFGS.CommandsCfg().base_velocity.ranges.ang_vel_z,
        ],
    )


@configclass
class NavObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class NavProprioceptiveCfg(ObsGroup):
        """Proprioceptive observations for navigation policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NavExteroceptiveCfg(ObsGroup):
        """Exteroceptive observations for navigation policy group."""

        forwards_depth_image = ObsTerm(
            func=mdp.camera_image,
            params={"sensor_cfg": SceneEntityCfg("front_camera")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation Groups
    low_level_policy: LOW_LEVEL_CFGS.ObservationsCfg.PolicyCfg = LOW_LEVEL_CFGS.ObservationsCfg.PolicyCfg()
    proprioceptive: NavProprioceptiveCfg = NavProprioceptiveCfg()
    exteroceptive: NavExteroceptiveCfg = NavExteroceptiveCfg()

    def __post_init__(self):
        # adjust because the velocity commands are now given by the navigation policy
        self.low_level_policy.velocity_commands = ObsTerm(
            func=mdp.vel_commands, params={"action_term": "velocity_command"}
        )
        self.low_level_policy.actions = ObsTerm(
            func=mdp.last_low_level_action, params={"action_term": "velocity_command"}
        )


@configclass
class EventCfg:
    """Configuration for randomization."""

    reset_base = EventTerm(
        func=mdp.reset_robot_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "yaw_range": (-3.14, 3.14),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-0.5, 0.5),
            },
            "goal_command_generator_name": "goal_command",
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    .. note::
        All rewards get multiplied with weight*dt - consider this when setting weights.
        Rewards are normalized over max episode length in wandb logging.

    .. note::
        In wandb:
        - Episode Rewards are in seconds
        - Train Mean Reward is based on episode length (Rewards * Episode Length)
    """

    # -- rewards
    goal_reached_rew = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": "goal_reached"},
        weight=200.0,
    )

    # -- penalties
    lateral_movement = RewTerm(
        func=mdp.lateral_movement,
        weight=-0.01,
    )
    backward_movement = RewTerm(
        func=mdp.backwards_movement,
        weight=-0.01,
    )
    episode_termination = RewTerm(
        func=mdp.is_terminated_term,  # type: ignore
        params={"term_keys": ["base_contact"]},
        weight=-200.0,
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_reached = DoneTerm(
        func=mdp.at_goal,  # type: ignore
        params={
            "distance_threshold": 0.5,
            "command_generator_term_name": "goal_command",
        },
        time_out=False,
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
            "threshold": 0.0,
        },
        time_out=False,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    goal_command = mdp.GoalCommandCfg(
        asset_name="robot",
        grid_resolution=0.1,
        robot_length=1.0,
        raycaster_sensor="height_scanner",
        resampling_time_range=(1.0e9, 1.0e9),  # No resampling
        debug_vis=True,
        reset_pos_term_name="reset_base",
    )


##
# Environment configuration
##


@configclass
class NavEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # Basic settings
    observations: NavObservationsCfg = NavObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""

        # Simulation settings
        self.sim.dt = 0.005  # In seconds
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # General settings
        self.episode_length_s = 20

        # This sets how many times the high-level actions (navigation policy)
        # are applied to the sim before being recalculated.
        self.decimation = int(1 / self.sim.dt / 10)  # 10Hz planner frequency

        # Similar to above, the low-level actions (locomotion controller) are calculated every:
        # self.sim.dt * self.low_level_decimation, so 0.005 * 4 = 0.02 seconds, or 50Hz.
        self.low_level_decimation = 4


@configclass
class RayCasterNavEnvCfg(NavEnvCfg):
    """Configuration for the navigation environment with ray caster camera."""

    scene: RayCasterNavSceneCfg = RayCasterNavSceneCfg(num_envs=100, env_spacing=8)

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # update sensor update periods
        # We tick contact sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # We tick the cameras based on the navigation policy update period.
        if self.scene.front_camera is not None:
            self.scene.front_camera.update_period = self.decimation * self.sim.dt


@configclass
class TiledNavEnvCfg(NavEnvCfg):
    """Configuration for the navigation environment with tiled camera."""

    scene: TiledNavSceneCfg = TiledNavSceneCfg(num_envs=100, env_spacing=8)

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # update sensor update periods
        # We tick contact sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # We tick the cameras based on the navigation policy update period.
        if self.scene.front_camera is not None:
            self.scene.front_camera.update_period = self.decimation * self.sim.dt
