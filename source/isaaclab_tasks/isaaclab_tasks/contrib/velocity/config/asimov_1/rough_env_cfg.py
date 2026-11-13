# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 velocity configurations.

The task builds on the shared rough-terrain velocity configuration and supports the Isaac Sim PhysX backend.
The policy is blind: the height scanner and the terrain curriculum of the shared configuration are disabled.
"""

import math

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity.velocity_env_cfg import EventsCfg, LocomotionVelocityRoughEnvCfg, MySceneCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.asimov_1 import (
    ASIMOV_1_ACTION_SCALE,
    ASIMOV_1_CFG,
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
FOOT_SITE_OFFSET = (0.05, 0.0, -0.025)
"""Position [m] of the foot site in the ankle-roll link frame."""

# Each contact filter must match exactly one body per environment, so both sides are listed. Only links
# with collision shapes in the asset are included.
SELF_COLLISION_FILTER_PATHS = [
    "{ENV_REGEX_NS}/Robot/pelvis_link",
    "{ENV_REGEX_NS}/Robot/waist_yaw_link",
    *(
        f"{{ENV_REGEX_NS}}/Robot/{side}_{link}_link"
        for side in ("left", "right")
        for link in ("hip_pitch", "hip_yaw", "shoulder_roll")
    ),
]

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
    """The feet on the robot, in the same order as :func:`_feet_sensor_cfg`."""
    return SceneEntityCfg("robot", body_names=list(FEET_BODIES), preserve_order=True)


def _feet_sensor_cfg() -> SceneEntityCfg:
    """The feet on the contact sensor, in the same order as :func:`_feet_cfg`."""
    return SceneEntityCfg("contact_forces", body_names=list(FEET_BODIES), preserve_order=True)


@configclass
class Asimov1PhysicsCfg(PresetCfg):
    """Physics configuration for the Asimov-1 velocity environment."""

    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    default = isaacsim_physx


@configclass
class Asimov1SceneCfg(MySceneCfg):
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
    robot: ArticulationCfg = ASIMOV_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # the policy is blind
    height_scanner = None
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(elbow|knee|ankle_roll)_link",
        filter_prim_paths_expr=SELF_COLLISION_FILTER_PATHS,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointZeroOffsetPositionActionCfg(
        asset_name="robot",
        joint_names=list(ASIMOV_1_JOINT_NAMES),
        preserve_order=True,
        scale=ASIMOV_1_ACTION_SCALE,
        use_default_offset=True,
        zero_offset_range=(-0.02, 0.02),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            delay_min_lag=0,
            delay_max_lag=1,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            delay_min_lag=0,
            delay_max_lag=2,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_slot01 = ObsTerm(
            func=mdp.joint_pos_rel_measured,
            params={"action_name": "joint_pos", "asset_cfg": _slot_cfg(SLOT_0_1)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos_slot23 = ObsTerm(
            func=mdp.joint_pos_rel_measured,
            params={"action_name": "joint_pos", "asset_cfg": _slot_cfg(SLOT_2_3)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos_slot45 = ObsTerm(
            func=mdp.joint_pos_rel_measured,
            params={"action_name": "joint_pos", "asset_cfg": _slot_cfg(SLOT_4_5)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
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
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_slot01 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_0_1)})
        joint_pos_slot23 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_2_3)})
        joint_pos_slot45 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_4_5)})
        joint_vel_slot01 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _slot_cfg(SLOT_0_1)})
        joint_vel_slot23 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _slot_cfg(SLOT_2_3)})
        joint_vel_slot45 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _slot_cfg(SLOT_4_5)})
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        foot_height = ObsTerm(func=mdp.foot_height, params={"asset_cfg": _feet_cfg(), "offset": FOOT_SITE_OFFSET})
        foot_air_time = ObsTerm(func=mdp.foot_air_time, params={"sensor_cfg": _feet_sensor_cfg()})
        foot_contact = ObsTerm(func=mdp.foot_contact, params={"sensor_cfg": _feet_sensor_cfg()})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces, params={"sensor_cfg": _feet_sensor_cfg()})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Asimov1EventsCfg(EventsCfg):
    physics_material = EventTerm(
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
    add_base_mass = None
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY]),
            "com_range": {"x": (0.0, 0.05), "y": (0.0, 0.0), "z": (0.03, 0.07)},
        },
    )
    base_external_force_torque = None
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
    actuator_gains = EventTerm(
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
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    # -- base motion
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.03)
    # -- posture
    pose = RewTerm(
        func=mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POSE_JOINT_PATTERNS)),
            "command_name": "base_velocity",
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
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POSE_JOINT_PATTERNS)),
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={"command_name": "base_velocity", "sensor_cfg": _feet_sensor_cfg(), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={"sensor_cfg": _feet_sensor_cfg(), "asset_cfg": _feet_cfg()},
    )
    foot_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=-2.0,
        params={
            "target_height": 0.1,
            "offset": FOOT_SITE_OFFSET,
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "sensor_cfg": _feet_sensor_cfg(),
            "asset_cfg": _feet_cfg(),
        },
    )
    foot_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.25,
        params={
            "target_height": 0.1,
            "offset": FOOT_SITE_OFFSET,
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "sensor_cfg": _feet_sensor_cfg(),
            "asset_cfg": _feet_cfg(),
        },
    )
    feet_orientation = RewTerm(func=mdp.feet_orientation_l1, weight=-1.0, params={"asset_cfg": _feet_cfg()})
    soft_landing = RewTerm(
        func=mdp.soft_landing,
        weight=-1e-5,
        params={"sensor_cfg": _feet_sensor_cfg(), "command_name": "base_velocity", "command_threshold": 0.05},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.25,
        params={"sensor_cfg": _feet_sensor_cfg(), "ratio_threshold": 4.0},
    )
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-5e-4,
        params={"sensor_cfg": _feet_sensor_cfg(), "threshold": 350.0},
    )
    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("self_collision")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})


@configclass
class Asimov1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Asimov-1 rough-terrain velocity tracking."""

    sim: SimulationCfg = SimulationCfg(physics=Asimov1PhysicsCfg())
    scene: Asimov1SceneCfg = Asimov1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: Asimov1EventsCfg = Asimov1EventsCfg()

    def __post_init__(self):
        # the terrain distribution is fixed
        self.curriculum.terrain_levels = None
        super().__post_init__()

        # commands
        self.commands.base_velocity.resampling_time_range = (3.0, 8.0)
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.rel_heading_envs = 0.3
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

    def play_mode(self):
        """Apply deterministic overrides for policy playback."""
        super().play_mode()
        self.scene.num_envs = 32
        self.actions.joint_pos.zero_offset_range = (0.0, 0.0)
        self.events.physics_material = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {
            k: (0.0, 0.0) for k in self.events.reset_base.params["pose_range"]
        }
        self.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in self.events.reset_base.params["velocity_range"]
        }
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
