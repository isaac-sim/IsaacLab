# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Pollen Robotics' MicroDuck with Dynamixel XL330 BAM servos.

The following configurations are available:

* :data:`MICRODUCK_CFG`: the 14-joint walking robot with its standing pose.
* :data:`MICRODUCK_ALLCOLLISIONS_CFG`: the robot with additional body and head collisions.
* :data:`MICRODUCK_ROLLERS_CFG`: the robot with four passive wheels in place of its soles.
* :data:`MICRODUCK_BACKLASH_CFG`, :data:`MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG`, and
  :data:`MICRODUCK_ROLLERS_BACKLASH_CFG`: the corresponding variants with gearbox play.

Reference: https://github.com/pollen-robotics/microduck_rl
"""

from dataclasses import fields

from isaaclab_newton.sim.schemas import NewtonArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuatorCfg, BamBacklashActuatorCfg, BamMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import clone

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

_XL330_MOTOR_CFG = BamMotorCfg(
    model="m6",
    kt=0.36601349688984386,
    resistance=2.8113923539223227,
    error_gain=0.0028773775022263564,
    max_pwm=1.0,
    max_current=1.75,
    friction_base=0.004771183165566,
    friction_viscous=0.005359668274599504,
    friction_stribeck=0.004676345799486616,
    dtheta_stribeck=2.890372094130307,
    alpha=8.683259907618984,
    load_friction_motor=0.2667860954283698,
    load_friction_external=8.515871897059342e-06,
    load_friction_motor_stribeck=1.0722918395099123e-05,
    load_friction_external_stribeck=0.08077928978935671,
    load_friction_motor_quad=0.009972471242139415,
    load_friction_external_quad=0.004902565732332559,
)
"""XL330 m6 fit from Rhoban BAM revision ``62bd8ce12154340be97e06f7f41a0ca8f116d967``."""

MICRODUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/PollenRobotics/MicroDuck/microduck_walk.usd",
        activate_contact_sensors=True,
        articulation_props=NewtonArticulationCfg(self_collision_enabled=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.125),
        joint_pos={
            ".*hip_yaw": 0.0,
            "left_hip_roll": -0.0873,
            "right_hip_roll": 0.0873,
            "left_hip_pitch": -0.4579,
            "right_hip_pitch": 0.4579,
            "left_knee": -0.0049,
            "right_knee": 0.0049,
            "left_ankle": 0.4530,
            "right_ankle": -0.4530,
            "neck_pitch": 0.3491,
            "head_pitch": 0.3491,
            "head_yaw": 0.0,
            "head_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "servos": BamActuatorCfg(
            joint_names_expr=["^(?!passive_).*"],
            motor=_XL330_MOTOR_CFG,
            kp_fw=200.0,
            vin=7.4,
            vin_range=(6.5, 8.2),
            vin_drop_gain_range=(0.0, 0.2),
            vin_min=6.0,
            min_delay=3,
            max_delay=6,
        ),
    },
)
"""MicroDuck with BAM voltage control, battery sag, and a 3--6-step command delay.

Requires Newton's MJWarp solver and ``SimulationCfg.use_newton_actuators=True``.
The USD carries BAM actuator prims, passive damping, joint armature, and effort limits.
As with other explicit actuators, Lab replaces the USD actuators with this configuration's
matching parameters when spawning. Per-episode friction randomization
belongs in the task's reset events, using :func:`~isaaclab.actuators.newton.write_group_parameter`.
"""

MICRODUCK_ALLCOLLISIONS_CFG = clone(MICRODUCK_CFG)
MICRODUCK_ALLCOLLISIONS_CFG.spawn.usd_path = (
    f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/PollenRobotics/MicroDuck/microduck_allcollisions.usd"
)
"""MicroDuck with additional trunk, hip, and head collision geometry for whole-body contact."""

MICRODUCK_ROLLERS_CFG = clone(MICRODUCK_CFG)
MICRODUCK_ROLLERS_CFG.spawn.usd_path = (
    f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/PollenRobotics/MicroDuck/microduck_rollers.usd"
)
MICRODUCK_ROLLERS_CFG.init_state.pos = (0.0, 0.0, 0.145)
MICRODUCK_ROLLERS_CFG.init_state.joint_pos["passive_.*"] = 0.0
"""MicroDuck with 14 BAM servos and four passive wheel joints.

The initial root height [m] provides clearance for the wheels, which extend below the walking soles.
"""


def _with_backlash(cfg: ArticulationCfg) -> ArticulationCfg:
    """Use the corresponding USD with passive play hinges and output-side encoder feedback."""
    cfg = clone(cfg)
    cfg.spawn.usd_path = cfg.spawn.usd_path.removesuffix(".usd") + "_backlash.usd"
    servos = cfg.actuators["servos"]
    cfg.actuators["servos"] = BamBacklashActuatorCfg(
        **{field.name: getattr(servos, field.name) for field in fields(servos)}
    )
    cfg.init_state.joint_pos["passive_.*"] = 0.0
    return cfg


MICRODUCK_BACKLASH_CFG = _with_backlash(MICRODUCK_CFG)
"""Walking MicroDuck with 14 servos and 14 passive gearbox-play hinges spanning ±1°."""

MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG = _with_backlash(MICRODUCK_ALLCOLLISIONS_CFG)
"""MicroDuck with additional body collisions and passive gearbox-play hinges."""

MICRODUCK_ROLLERS_BACKLASH_CFG = _with_backlash(MICRODUCK_ROLLERS_CFG)
"""MicroDuck with 14 servos, 14 passive gearbox-play hinges, and four passive wheels."""
