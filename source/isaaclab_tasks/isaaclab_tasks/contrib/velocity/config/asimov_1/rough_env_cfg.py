# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 velocity configurations."""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_assets.robots.asimov_1 import (
    ASIMOV_1_ACTION_SCALE,
    ASIMOV_1_DELAYED_CFG,
    ASIMOV_1_JOINT_NAMES,
)

from . import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.02, 0.05),
            noise_step=0.02,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            border_width=0.25,
        ),
    },
)

SLOT_0_1 = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
)
SLOT_2_3 = (
    "left_hip_yaw_joint",
    "left_knee_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
)
SLOT_4_5 = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "right_wrist_yaw_joint",
    "left_wrist_yaw_joint",
)

FEET_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
TORSO_BODY = "waist_yaw_link"

POSE_JOINT_PATTERNS = (
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_yaw_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
)


def _slot_cfg(names: tuple[str, ...]) -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(names), preserve_order=True)


def _feet_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=list(FEET_BODIES), preserve_order=True)


@configclass
class Asimov1SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ASIMOV_1_DELAYED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    feet_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        history_length=3,
        track_air_time=True,
    )
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(wrist_yaw|elbow|knee|ankle_roll)_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/pelvis_link",
            "{ENV_REGEX_NS}/Robot/waist_yaw_link",
            "{ENV_REGEX_NS}/Robot/.*_hip_pitch_link",
            "{ENV_REGEX_NS}/Robot/.*_hip_roll_link",
            "{ENV_REGEX_NS}/Robot/.*_hip_yaw_link",
            "{ENV_REGEX_NS}/Robot/.*_shoulder_roll_link",
        ],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class CommandsCfg:
    twist = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.3,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.8),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ASIMOV_1_JOINT_NAMES),
        preserve_order=True,
        scale=ASIMOV_1_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.delayed_obs,
            params={"quantity": "base_ang_vel", "min_lag": 0, "max_lag": 1},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.delayed_obs,
            params={"quantity": "projected_gravity", "min_lag": 0, "max_lag": 2},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        joint_pos_slot01 = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _slot_cfg(SLOT_0_1)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_pos_slot23 = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _slot_cfg(SLOT_2_3)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_pos_slot45 = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _slot_cfg(SLOT_4_5)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel_slot01 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _slot_cfg(SLOT_0_1)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.1,
        )
        joint_vel_slot23 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _slot_cfg(SLOT_2_3)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.1,
        )
        joint_vel_slot45 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _slot_cfg(SLOT_4_5)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.1,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        joint_pos_slot01 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_0_1)}, scale=1.0)
        joint_pos_slot23 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_2_3)}, scale=1.0)
        joint_pos_slot45 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_4_5)}, scale=1.0)
        joint_vel_slot01 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _slot_cfg(SLOT_0_1)}, scale=1.0)
        joint_vel_slot23 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _slot_cfg(SLOT_2_3)}, scale=1.0)
        joint_vel_slot45 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _slot_cfg(SLOT_4_5)}, scale=1.0)
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        foot_height = ObsTerm(func=mdp.foot_height, params={"asset_cfg": _feet_cfg()})
        foot_air_time = ObsTerm(func=mdp.foot_air_time, params={"sensor_name": "feet_contact"})
        foot_contact = ObsTerm(func=mdp.foot_contact, params={"sensor_name": "feet_contact"})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces, params={"sensor_name": "feet_contact"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FEET_BODIES)),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    qpos0_rand = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "ranges": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY]),
            "com_range": {"x": (0.0, 0.05), "y": (0.0, 0.0), "z": (0.03, 0.07)},
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
                "pitch": (-0.15, 0.15),
                "roll": (-0.1, 0.1),
            },
            "velocity_range": {
                "pitch": (-0.5, 0.5),
                "roll": (-0.3, 0.3),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    pd_gains_rand = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.3, 0.3),
                "roll": (-0.4, 0.4),
                "pitch": (-0.4, 0.4),
                "yaw": (-0.5, 0.5),
            }
        },
    )


@configclass
class RewardsCfg:
    track_linear_velocity = RewTerm(
        func=mdp.track_linear_velocity,
        weight=5.0,
        params={"command_name": "twist", "std": 0.5},
    )
    track_angular_velocity = RewTerm(
        func=mdp.track_angular_velocity,
        weight=3.0,
        params={"command_name": "twist", "std": 0.7071},
    )
    upright = RewTerm(
        func=mdp.flat_orientation,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY]),
        },
    )
    pose = RewTerm(
        func=mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POSE_JOINT_PATTERNS)),
            "command_name": "twist",
            "walking_threshold": 0.1,
            "running_threshold": 1.5,
            "pose_weight_velocity_threshold": 0.3,
            "pose_weight_slow": 2.0,
            "pose_weight_fast": 1.0,
            "disturbance_std_scale": 2.5,
            "std_standing": {".*": 0.05},
            "std_walking": {
                r".*hip_pitch.*": 0.5,
                r".*hip_roll.*": 0.15,
                r".*hip_yaw.*": 0.15,
                r".*knee.*": 0.5,
                r".*ankle_pitch.*": 0.15,
                r".*ankle_roll.*": 0.1,
                r".*waist_yaw.*": 0.15,
                r".*shoulder_pitch.*": 0.15,
                r".*shoulder_roll.*": 0.1,
                r".*shoulder_yaw.*": 0.1,
                r".*elbow.*": 0.1,
                r".*wrist.*": 0.1,
            },
            "std_running": {
                r".*hip_pitch.*": 0.5,
                r".*hip_roll.*": 0.25,
                r".*hip_yaw.*": 0.25,
                r".*knee.*": 0.5,
                r".*ankle_pitch.*": 0.25,
                r".*ankle_roll.*": 0.1,
                r".*waist_yaw.*": 0.25,
                r".*shoulder_pitch.*": 0.25,
                r".*shoulder_roll.*": 0.1,
                r".*shoulder_yaw.*": 0.1,
                r".*elbow.*": 0.1,
                r".*wrist.*": 0.1,
            },
        },
    )
    air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_name": "feet_contact",
            "threshold_min": 0.05,
            "threshold_max": 0.5,
            "command_name": "twist",
            "command_threshold": 0.5,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=-2.0,
        params={
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": _feet_cfg(),
        },
    )
    foot_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.25,
        params={
            "sensor_name": "feet_contact",
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": _feet_cfg(),
        },
    )
    foot_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.1,
        params={
            "sensor_name": "feet_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": _feet_cfg(),
        },
    )
    feet_orientation = RewTerm(
        func=mdp.feet_orientation_penalty,
        weight=-1.0,
        params={"asset_cfg": _feet_cfg()},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    body_ang_vel = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY])},
    )
    angular_momentum = RewTerm(func=mdp.angular_momentum_penalty, weight=-0.03)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    soft_landing = RewTerm(
        func=mdp.soft_landing,
        weight=-1e-5,
        params={
            "sensor_name": "feet_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POSE_JOINT_PATTERNS)),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.25,
        params={"sensor_name": "feet_contact", "ratio_threshold": 4.0},
    )
    feet_contact_force_limit = RewTerm(
        func=mdp.feet_contact_force_limit,
        weight=-5e-4,
        params={"sensor_name": "feet_contact", "max_force": 350.0},
    )
    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})


@configclass
class Asimov1RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Asimov-1 rough-terrain velocity tracking."""

    scene: Asimov1SceneCfg = Asimov1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Set simulation timing and the shared terrain material."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

    def play_mode(self):
        """Apply deterministic overrides for policy playback."""
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.qpos0_rand = None
        self.events.pd_gains_rand = None
        self.events.base_com = None
        self.events.foot_friction = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {
            k: (0.0, 0.0) for k in self.events.reset_base.params["pose_range"]
        }
        self.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in self.events.reset_base.params["velocity_range"]
        }
        self.commands.twist.ranges.lin_vel_x = (0.6, 0.8)
        self.commands.twist.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.twist.ranges.ang_vel_z = (-0.6, 0.6)
        self.scene.terrain.terrain_generator = COBBLESTONE_ROAD_CFG.replace(num_rows=5, num_cols=5, border_width=10.0)
        self.scene.terrain.max_init_terrain_level = 4
