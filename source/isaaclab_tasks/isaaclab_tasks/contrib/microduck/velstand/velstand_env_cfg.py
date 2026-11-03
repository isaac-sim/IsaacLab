# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Current merged upstream walking-plus-recovery recipe, flat and production rough terrain.

Pinned formulation: pollen-robotics/microduck_rl@bae39f5900b318f7e87fc485c30a31213cfa14eb.
Select ``presets=backlash`` for the production output-encoder/serial-play robot. The selected
training lineage is 6000 added iterations from the pinned walking checkpoint with two teachers;
the historical published v8 policy additionally had a multi-stage training lineage.
"""

import math
from typing import ClassVar

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.microduck.pollen_mdp as shared
from isaaclab_tasks.contrib.microduck.assets import (
    MICRODUCK_FULLCONTACT_BACKLASH_CFG,
    MICRODUCK_FULLCONTACT_CFG,
    MICRODUCK_FULLCONTACT_COLLIDER_SHAPE_EXPR,
    MICRODUCK_FULLCONTACT_HEAD_SHAPE_EXPR,
    MICRODUCK_FULLCONTACT_SERVO_SHAPE_EXPR,
    MICRODUCK_FULLCONTACT_TRUNK_SHAPE_EXPR,
)
from isaaclab_tasks.contrib.microduck.backlash import apply_backlash_preset
from isaaclab_tasks.contrib.microduck.terrain import MicroDuckTerrainImporter, configure_foot_terrain_sensing
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_JOINT_NAMES,
    MicroDuckSceneCfg,
    MicroDuckVelocityRoughEnvCfg,
    ObservationsCfg,
)
from isaaclab_tasks.utils import preset

from . import mdp

MICRODUCK_FALLEN_TILT_DEG = 40.0
MICRODUCK_RECOVERED_TILT_DEG = 25.0
MICRODUCK_RECOVERED_HEIGHT = 0.09
MICRODUCK_TERMINATION_Z = 0.08
MICRODUCK_STAND_HEIGHT = 0.115
MICRODUCK_FELL_OVER_DISABLE_ITER = 100
MICRODUCK_RECOVERY_ECON_ITER = 600
_SERVO = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_TRUNK = SceneEntityCfg("robot", body_names=["trunk_base"])
_ALL_SHAPES = MICRODUCK_FULLCONTACT_COLLIDER_SHAPE_EXPR
_SERVO_SHAPES = MICRODUCK_FULLCONTACT_SERVO_SHAPE_EXPR
_HEAD_SHAPES = MICRODUCK_FULLCONTACT_HEAD_SHAPE_EXPR
_TRUNK_SHAPES = MICRODUCK_FULLCONTACT_TRUNK_SHAPE_EXPR
_GROUND_SHAPES = "/World/ground/.*"

PRONE_RAMP_STAGES = [
    {"step": it * 24, "params": {"prone_prob": prone, "face_down_prob": down, "side_prob": 0.5, "crouch_prob": crouch}}
    for it, prone, down, crouch in (
        (0, 0.0, 1.0, 0.0),
        (150, 0.0, 1.0, 0.2),
        (700, 0.15, 0.8, 0.2),
        (1000, 0.3, 0.65, 0.2),
        (1400, 0.45, 0.5, 0.2),
    )
]


def _ground_contact(shapes):
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[shapes],
        filter_shape_prim_expr=[_GROUND_SHAPES],
    )


@configclass
class MicroDuckVelStandSceneCfg(MicroDuckSceneCfg):
    """Full geometry and ground-filtered shape sensors, including named XL330 housings."""

    robot = MICRODUCK_FULLCONTACT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[_ALL_SHAPES],
        filter_shape_prim_expr=[_ALL_SHAPES],
    )
    servo_ground_contact = _ground_contact(_SERVO_SHAPES)
    head_ground_contact = _ground_contact(_HEAD_SHAPES)
    trunk_ground_contact = _ground_contact(_TRUNK_SHAPES)


def _body_ranges(frac):
    angle = math.radians(12) * frac
    return (
        (-0.002, 0.002),
        (-0.002, 0.002),
        (-0.025 * frac, 0.010 * frac),
        (-angle, angle),
        (-angle, angle),
        (-0.02, 0.02),
    )


def _configure(cfg):
    cfg.events.finite_reward_guard = EventTerm(func=mdp.install_finite_reward_guard, mode="startup")
    # Collapse inherited schedules BEFORE adding the run-relative recovery/body schedules.
    for term in vars(cfg.curriculum).values():
        if term is None or not hasattr(term, "params"):
            continue
        for key, stages in list(term.params.items()):
            if isinstance(stages, list) and stages and all(isinstance(v, dict) and "step" in v for v in stages):
                final = {**stages[-1], "step": 0}
                term.params[key] = [final]
                if key == "weight_stages":
                    getattr(cfg.rewards, term.params["reward_name"]).weight = final["weight"]

    cfg.commands.body_pose = mdp.StandingGatedPoseCommandCfg(
        resampling_time_range=cfg.commands.body_pose.resampling_time_range,
        ranges=_body_ranges(0.5),
        zero_command_prob=0.3,
        axis_zero_prob=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    )
    cfg.commands.base_velocity.rel_turn_in_place_envs = 0.2
    cfg.commands.base_velocity.turn_in_place_min_frac = 0.2
    cfg.curriculum.body_pose_range = CurrTerm(
        func=shared.command_range_stages,
        params={
            "command_name": "body_pose",
            "range_stages": [{"step": 0, "ranges": _body_ranges(0.5)}, {"step": 300 * 24, "ranges": _body_ranges(1.0)}],
        },
    )
    cfg.rewards.body_pose_tracking = RewTerm(
        func=mdp.body_pose_tracking_standing,
        weight=2.0,
        params={"command_name": "body_pose", "nominal_height": 0.116, "z_std": 0.01, "angle_std": math.radians(5)},
    )
    cfg.rewards.pose.func = mdp.variable_posture_body_relaxed
    cfg.rewards.pose.params.update(
        {
            "std_body": {
                r".*hip_yaw.*": 0.1,
                r".*hip_roll.*": 0.25,
                r".*hip_pitch.*": 0.4,
                r".*knee.*": 0.4,
                r".*ankle.*": 0.3,
            },
            "body_command_name": "body_pose",
        }
    )
    cfg.rewards.upright.func = mdp.upright_body_cmd_relative
    cfg.rewards.upright.params["command_name"] = "body_pose"
    cfg.rewards.track_yaw_fine = RewTerm(
        func=mdp.track_yaw_rate_fine, weight=2.0, params={"std": 0.35, "command_name": "base_velocity"}
    )
    cfg.rewards.head_pose_bias.params.update(
        {"gate_height_low": 0.09, "gate_height_high": 0.11, "gate_tilt_full_deg": 20.0, "gate_tilt_zero_deg": 40.0}
    )
    cfg.rewards.air_time.params["gate_tilt_above_deg"] = 40.0
    cfg.rewards.self_collisions.params["saturate"] = True
    cfg.rewards.upright_progress = RewTerm(func=shared.upright_progress, weight=5.0, params={"asset_cfg": _TRUNK})
    cfg.rewards.height_progress = RewTerm(
        func=shared.height_progress, weight=30.0, params={"asset_cfg": _TRUNK, "ceiling": 0.115}
    )
    cfg.rewards.com_upward_velocity = RewTerm(
        func=shared.com_upward_velocity,
        weight=0.0,
        params={"asset_cfg": _TRUNK, "max_height": 0.125, "gate_tilt_above_deg": 40.0},
    )
    cfg.rewards.fallen_tax = RewTerm(
        func=shared.fallen_state_penalty,
        weight=0.0,
        params={
            "asset_cfg": _TRUNK,
            "gate_tilt_above_deg": 40.0,
            "release_tilt_below_deg": 25.0,
            "release_z_above": 0.09,
        },
    )
    cfg.rewards.recovery_success = RewTerm(
        func=shared.recovery_success,
        weight=0.0,
        params={"asset_cfg": _TRUNK, "fallen_tilt_deg": 40.0, "min_fallen_s": 0.5, "up_tilt_deg": 25.0, "up_z": 0.09},
    )
    cfg.rewards.action_rate_l2.func = mdp.action_rate_l2_fallen_scaled
    cfg.rewards.action_rate_l2.params = {"asset_cfg": _SERVO, "fallen_scale": 0.1, "gate_tilt_above_deg": 40.0}
    cfg.rewards.joint_torque_rate_l2 = RewTerm(
        func=mdp.joint_torque_rate_l2_fallen_scaled,
        weight=-0.002,
        params={"asset_cfg": _SERVO, "fallen_scale": 0.1, "gate_tilt_above_deg": 40.0},
    )

    protection = (
        ("servo_impact", -0.02, "servo_ground_contact", 2.0),
        ("head_impact", -0.01, "head_ground_contact", 15.0),
        ("trunk_impact", -0.01, "trunk_ground_contact", 20.0),
    )
    weights = {}
    for name, weight, sensor, threshold in protection:
        setattr(
            cfg.rewards,
            name,
            RewTerm(
                func=shared.body_impact_cost,
                weight=weight * 0.25,
                params={"sensor_cfg": SceneEntityCfg(sensor), "threshold": threshold},
            ),
        )
        weights[name] = weight
    cfg.rewards.servo_acc_spike = RewTerm(
        func=mdp.servo_acc_spike_penalty, weight=-0.00025, params={"asset_cfg": _SERVO, "acc_thresh": 300.0}
    )
    cfg.rewards.servo_stall = RewTerm(
        func=mdp.servo_stall_penalty,
        weight=-0.0125,
        params={"asset_cfg": _SERVO, "torque_thresh": 0.4, "vel_thresh": 0.5},
    )
    cfg.rewards.gentle_rise = RewTerm(func=shared.trunk_vertical_accel_penalty, weight=0.00125)
    weights.update(servo_acc_spike=-0.001, servo_stall=-0.05, gentle_rise=0.005)
    for name, weight in weights.items():
        setattr(
            cfg.curriculum,
            f"{name}_weight",
            CurrTerm(
                func=shared.reward_weight_stages,
                params={
                    "reward_name": name,
                    "weight_stages": [{"step": 0, "weight": weight * 0.25}, {"step": 1000 * 24, "weight": weight}],
                },
            ),
        )
    for name, weight in (("fallen_tax", -0.5), ("recovery_success", 10.0), ("com_upward_velocity", 2.0)):
        setattr(
            cfg.curriculum,
            f"{name}_weight",
            CurrTerm(
                func=shared.reward_weight_stages,
                params={
                    "reward_name": name,
                    "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 600 * 24, "weight": weight}],
                },
            ),
        )
    cfg.events.set_ground_state = EventTerm(
        func=mdp.reset_velstand_state,
        mode="reset",
        params={
            "asset_cfg": _SERVO,
            **PRONE_RAMP_STAGES[0]["params"],
            "joint_random_prob": 0.5,
            "joint_range_frac": 0.9,
            "so3_prob": 0.25,
            "sit_prob": 0.05,
        },
    )
    cfg.curriculum.ground_state_mix = CurrTerm(
        func=shared.event_param_stages, params={"event_name": "set_ground_state", "param_stages": PRONE_RAMP_STAGES}
    )
    cfg.events.topple_push = EventTerm(
        func=shared.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )
    cfg.curriculum.topple_push_range = CurrTerm(
        func=shared.event_param_stages,
        params={
            "event_name": "topple_push",
            "inclusive": False,
            "param_stages": [
                {"step": it * 24, "params": {"velocity_range": {"x": (-v, v), "y": (-v, v)}}}
                for it, v in ((0, 0.3), (100, 0.6), (400, 0.9), (800, 1.2))
            ],
        },
    )
    # Deliberately retain upstream's historical bearing_roll right-leg selection in this task only.
    cfg.events.randomize_head_com.params["asset_cfg"] = SceneEntityCfg(
        "robot",
        body_names=["neck", "neck_pitch(_[0-9]+)?", "yaw_roll_motion", "(bottom_head_shell|jaw_soft)", "bearing_roll"],
    )
    cfg.terminations.fallen_too_long = DoneTerm(
        func=shared.fallen_too_long, params={"gate_z_below": 0.08, "gate_tilt_above_deg": 40.0, "max_duration_s": 8.0}
    )
    cfg.curriculum.fell_over_disable = CurrTerm(
        func=shared.termination_param_stages,
        params={
            "term_name": "fell_over",
            "param_stages": [
                {"step": 0, "params": {"limit_angle": math.radians(70)}},
                {"step": 100 * 24, "params": {"limit_angle": math.pi}},
            ],
        },
    )
    # Full-contact budgets are conservative until the controlled-fall profile is measured.
    cfg.sim.physics.newton_mjwarp.solver_cfg.nconmax = 200
    cfg.sim.physics.newton_mjwarp.solver_cfg.njmax = 1024
    cfg.sim.physics.default = cfg.sim.physics.newton_mjwarp
    for sensor in (cfg.scene.servo_ground_contact, cfg.scene.head_ground_contact, cfg.scene.trunk_ground_contact):
        sensor.update_period = cfg.sim.dt


@configclass
class MicroDuckVelStandRoughEnvCfg(MicroDuckVelocityRoughEnvCfg):
    """Production terrain recipe; select presets=backlash for the production robot."""

    _rough_terrain: ClassVar[bool] = True
    selected_presets = preset(default=(), backlash=("backlash",))

    scene: MicroDuckVelStandSceneCfg = MicroDuckVelStandSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure(self)
        if self._rough_terrain:
            self.scene.terrain.class_type = MicroDuckTerrainImporter
            configure_foot_terrain_sensing(self)
        apply_backlash_preset(self, MICRODUCK_FULLCONTACT_BACKLASH_CFG)

    def play_mode(self):
        """Expose full body commands and the source's accelerated protective-fall playback."""
        super().play_mode()
        self.terminations.fell_over = None
        self.curriculum.fell_over_disable = None
        self.commands.body_pose.ranges = _body_ranges(1.0)
        self.curriculum.body_pose_range = None
        self.events.push_robot.interval_range_s = (0.5, 1.0)
        self.events.topple_push.interval_range_s = (1.0, 2.0)
        self.events.topple_push.params["velocity_range"] = {"x": (-1.2, 1.2), "y": (-1.2, 1.2)}
        self.curriculum.topple_push_range = None
        if self._rough_terrain:
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.curriculum.terrain_levels = None


@configclass
class MicroDuckVelStandFlatEnvCfg(MicroDuckVelStandRoughEnvCfg):
    """Flat comparison of the same learning and recovery formulation."""

    _rough_terrain: ClassVar[bool] = False

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.sim.physics.newton_mjwarp.solver_cfg.iterations = 10
        self.sim.physics.newton_mjwarp.solver_cfg.ls_iterations = 20
