# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat walking recipe ported from pollen-robotics/microduck_rl and the original Lab task."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

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
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import preset

from isaaclab_assets import MICRODUCK_BACKLASH_CFG, MICRODUCK_CFG

from . import mdp

# Pin the deployed policy order; USD articulation order differs.
MICRODUCK_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
MICRODUCK_LEG_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
MICRODUCK_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
MICRODUCK_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
MICRODUCK_TRUNK_BODY_NAME = "trunk_base"
MICRODUCK_HEAD_BODY_NAMES = ["neck", "neck_pitch(_[0-9]+)?", "yaw_roll_motion", "jaw_soft"]
_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")
_PLAY_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_JOINT_NAMES], preserve_order=True
)
_HEAD_PLAY_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_HEAD_JOINT_NAMES], preserve_order=True
)
_IMU_MISALIGNMENT_DEG = 6.0
_IMU_DELAY_UPDATE_PERIOD = 64
MICRODUCK_SOLE_TO_ANKLE_OFFSET = 0.0225
"""Standing ankle-frame height above the sole [m]."""
MICRODUCK_FOOT_TARGET_HEIGHT = 0.02 + MICRODUCK_SOLE_TO_ANKLE_OFFSET
_FOOT_SWING_HEIGHT_WEIGHT = -0.25 * (MICRODUCK_FOOT_TARGET_HEIGHT / 0.02) ** 2
MICRODUCK_STEPS_PER_ITERATION = 24


def _backlash_preset(term, func=None, **params):
    """Use ``term`` as is, or with ``func`` and ``params`` for the backlash robot under ``presets=backlash``."""
    backlash = term.replace(params={**term.params, **params})
    if func is not None:
        backlash.func = func
    return preset(default=term, backlash=backlash)


@configclass
class MicroDuckSceneCfg(InteractiveSceneCfg):
    """Scene with the MicroDuck robot on a ground plane."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False,
    )
    robot = preset(
        default=MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        backlash=MICRODUCK_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )
    # The walking USD enables sole colliders only, so self-contact is foot against foot.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_left",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_right"],
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9))
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.15,
        debug_vis=True,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
    head_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0), ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
    )
    # Retained in the deployed observation vector, although body-pose reward is disabled.
    body_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005), (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True, scale=1.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract."""

        base_ang_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.base_ang_vel_imu_misaligned,
                "min_lag": 0,
                "max_lag": 1,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        projected_gravity = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.projected_gravity_imu_misaligned,
                "min_lag": 0,
                "max_lag": 1,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # With backlash, encoders measure servo plus play angle.
        joint_pos = _backlash_preset(
            ObsTerm(
                func=mdp.joint_pos_rel_biased,
                params={"asset_cfg": _SERVO_JOINT_CFG, "biased": True},
                noise=Unoise(n_min=-0.001, n_max=0.001),
            ),
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        joint_vel = _backlash_preset(
            ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": _SERVO_JOINT_CFG},
                noise=Unoise(n_min=-0.25, n_max=0.25),
                delay_min_lag=1,
                delay_max_lag=1,
            ),
            func=mdp.joint_vel_rel_backlash,
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function (76 values)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = _backlash_preset(
            ObsTerm(func=mdp.joint_pos_rel_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False}),
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        joint_vel = _backlash_preset(
            ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG}),
            func=mdp.joint_vel_rel_backlash,
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_height = ObsTerm(func=mdp.foot_height_safe, params={"asset_cfg": _FOOT_BODY_CFG})
        foot_air_time = ObsTerm(func=mdp.foot_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": _FOOT_MATERIAL_CFG,
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="startup", params={"bias_range": (-0.015, 0.015)})
    imu_misalignment = EventTerm(
        func=mdp.randomize_imu_misalignment, mode="startup", params={"max_angle_deg": _IMU_MISALIGNMENT_DEG}
    )
    mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )
    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _HEAD_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction, mode="reset", params={"scale_range": (0.9, 1.1)}
    )
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel = RewTerm(
        func=mdp.track_linear_velocity, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.1)}
    )
    track_ang_vel = RewTerm(
        func=mdp.track_angular_velocity, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    upright = RewTerm(func=mdp.upright, weight=2.0, params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG})
    pose = RewTerm(
        func=mdp.pose_mode_switch,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std_standing": {".*hip_yaw": 0.1, ".*hip_roll": 0.05, ".*hip_pitch": 0.15, ".*knee": 0.15, ".*ankle": 0.1},
            "std_walking": {".*hip_yaw": 0.3, ".*hip_roll": 0.05, ".*hip_pitch": 0.4, ".*knee": 0.4, ".*ankle": 0.25},
            "walking_threshold": 0.01,
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    # Play hinges rest against their stops, so only servos incur the soft-limit penalty with backlash.
    dof_pos_limits = _backlash_preset(RewTerm(func=mdp.joint_pos_limits, weight=-1.0), asset_cfg=_SERVO_JOINT_CFG)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    air_time = RewTerm(
        func=mdp.feet_air_time_windowed,
        weight=3.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold_min": 0.125,
            "threshold_max": 0.3,
            "command_threshold": 0.01,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=-2.0,
        params={
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    foot_swing_height = RewTerm(
        func=mdp.foot_swing_height,
        weight=_FOOT_SWING_HEIGHT_WEIGHT,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "asset_cfg": _FOOT_BODY_CFG,
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "command_threshold": 0.01,
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.1,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    self_collisions = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG}
    )
    head_pose_tracking = _backlash_preset(
        RewTerm(
            func=mdp.head_pose_tracking,
            weight=2.0,
            params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
        ),
        backlash_cfg=_HEAD_PLAY_JOINT_CFG,
    )
    head_pose_bias = _backlash_preset(
        RewTerm(
            func=mdp.head_pose_bias_penalty,
            weight=0.0,
            params={"command_name": "head_pose", "tau_s": 1.0, "asset_cfg": _HEAD_JOINT_CFG},
        ),
        backlash_cfg=_HEAD_PLAY_JOINT_CFG,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    nan_state = DoneTerm(func=mdp.robot_state_is_nan, time_out=False, params={"sensor_names": ("contact_forces",)})


def _schedule(address: str, *stages: tuple[int, object]) -> CurrTerm:
    """Step ``address`` through ``(upstream_iteration, value)`` stages after its configured value."""
    return CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": address,
            "modify_fn": mdp.staged_value,
            "modify_params": {
                "stages": [(iteration * MICRODUCK_STEPS_PER_ITERATION, value) for iteration, value in stages]
            },
        },
    )


def _com_range(half_width: float) -> dict[str, tuple[float, float]]:
    """Symmetric per-axis center-of-mass offset range [m]."""
    return {axis: (-half_width, half_width) for axis in ("x", "y", "z")}


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate_weight = _schedule(
        "rewards.action_rate_l2.weight", (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0)
    )
    head_pose_bias_weight = _schedule("rewards.head_pose_bias.weight", (600, -1.0), (1000, -2.0), (1500, -3.0))
    standing_envs = _schedule(
        "commands.base_velocity.rel_standing_envs", (500, 0.05), (750, 0.1), (1000, 0.15), (1500, 0.2), (2000, 0.25)
    )
    head_pose_range = _schedule(
        "commands.head_pose.ranges",
        (500, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
        (1000, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
        (1500, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.2, 0.2))),
        (2000, ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-0.31, 0.31))),
    )
    com_range = _schedule(
        "events.randomize_com.params.com_range",
        (500, _com_range(0.005)),
        (1000, _com_range(0.01)),
        (1500, _com_range(0.015)),
    )
    head_com_range = _schedule(
        "events.randomize_head_com.params.com_range", (500, _com_range(0.005)), (1000, _com_range(0.01))
    )


@configclass
class MicroDuckVelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck flat velocity walking with Newton MJWarp and BAM servos."""

    sim: SimulationCfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=96,
                nconmax=16,
                # Gearbox limit contacts exhaust 10 iterations at training scale.
                iterations=preset(default=10, backlash=100),
                # The original 20-step budget exhausts line search with the current MJWarp solver.
                ls_iterations=50,
                cone="pyramidal",
                impratio=1.0,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2500000),
            default_shape_cfg=NewtonShapeCfg(margin=0.0),
            num_substeps=1,
        )
    )
    scene: MicroDuckSceneCfg = MicroDuckSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.self_collision.update_period = self.sim.dt
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))

    def play_mode(self):
        """Use the shared inference defaults and disable interval pushes."""
        super().play_mode()
        self.events.push_robot = None
