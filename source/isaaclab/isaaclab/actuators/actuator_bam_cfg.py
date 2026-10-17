# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass

from .actuator_base_cfg import ActuatorBaseCfg

BAM_POSITION_BACKLASH_API = "IsaacLabPositionBacklashAPI"
"""USD API schema token of the backlash input processor that feeds back the play angle but not its velocity."""

BAM_STIFF_SOLREF_FRICTION: tuple[float, float] = (-5.0e4, -2.0e2)
"""Stiff ``(-stiffness, -damping)`` friction solref from the reference BAM implementation [N.m/rad, N.m.s/rad]."""

BAM_STIFF_SOLIMP_FRICTION: tuple[float, float, float, float, float] = (0.99, 0.9999, 0.001, 0.5, 2.0)
"""Friction-constraint impedance profile paired with :data:`BAM_STIFF_SOLREF_FRICTION` [-]."""

BAM_MODEL_FLAGS: dict[str, dict[str, int]] = {
    "m1": {"stribeck": 0, "load_dependent": 0, "quadratic": 0},
    "m2": {"stribeck": 1, "load_dependent": 0, "quadratic": 0},
    "m5": {"stribeck": 1, "load_dependent": 1, "quadratic": 0},
    "m6": {"stribeck": 1, "load_dependent": 1, "quadratic": 1},
}
"""Gearbox friction terms each BAM model enables."""


@configclass
class BamMotorCfg:
    """Identified BAM motor and gearbox fit, shared by all joints of an actuator group."""

    model: Literal["m1", "m2", "m5", "m6"] = MISSING
    """Friction variant: Coulomb, Stribeck, directional load-dependent, or quadratic."""

    kt: float = MISSING
    """Motor torque/back-EMF constant [N.m/A or V.s/rad]."""

    resistance: float = MISSING
    """Winding resistance [Ohm]."""

    error_gain: float = MISSING
    """Position-error-to-duty-cycle factor per unit of firmware gain [1/rad]."""

    max_pwm: float = 1.0
    """Maximum duty-cycle magnitude [-]."""

    max_current: float = MISSING
    """Firmware current limit [A]. Zero disables current limiting."""

    friction_base: float = MISSING
    """Coulomb friction [N.m]."""

    friction_viscous: float = MISSING
    """Viscous friction coefficient [N.m.s/rad], applied as passive joint damping."""

    friction_stribeck: float = 0.0
    """Additional near-rest friction [N.m]."""

    dtheta_stribeck: float = 1.0
    """Stribeck decay velocity [rad/s]."""

    alpha: float = 1.0
    """Stribeck decay exponent [-]."""

    load_friction_motor: float = 0.0
    """Motor-side load-dependent friction coefficient [-]."""

    load_friction_external: float = 0.0
    """External-side load-dependent friction coefficient [-]."""

    load_friction_motor_stribeck: float = 0.0
    """Motor-side near-rest load-dependent friction coefficient [-]."""

    load_friction_external_stribeck: float = 0.0
    """External-side near-rest load-dependent friction coefficient [-]."""

    load_friction_motor_quad: float = 0.0
    """Motor-side quadratic load-coupling coefficient [1/(N.m)]."""

    load_friction_external_quad: float = 0.0
    """External-side quadratic load-coupling coefficient [1/(N.m)]."""

    def validate_config(self) -> None:
        """Reject friction variants the BAM drive does not implement.

        Raises:
            ValueError: If the model is not m1, m2, m5, or m6.
        """
        if self.model not in BAM_MODEL_FLAGS:
            raise ValueError(f"Unsupported BAM model {self.model!r}; expected m1, m2, m5, or m6.")


@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """Configuration for the BAM voltage-domain servo actuator.

    Requires ``use_newton_actuators=True`` with Newton's MJWarp solver. Like other explicit
    actuators, it replaces existing USD actuators on the selected joints. Newton runs the firmware
    and motor (:class:`~newton.actuators.DriveBAM`), the supply (:class:`~newton.actuators.Battery`),
    the command delay and the effort clamp; Isaac Lab adds the random command lag and the gearbox
    friction.

    If :attr:`~isaaclab.actuators.ActuatorBaseCfg.actuator_effort_limit` is None, the torque
    limit is the stall torque ``max(vin_range) * motor.kt / motor.resistance``, using
    :attr:`vin` when :attr:`vin_range` is unset.
    """

    class_type: type | None = None
    """None: Newton builds the drive from the authored USD schema."""

    stiffness: dict[str, float] | float | None = None
    """Unused. The firmware gain is :attr:`kp_fw`."""

    damping: dict[str, float] | float | None = None
    """Unused. Damping is the motor's back-EMF."""

    motor: BamMotorCfg = MISSING
    """Identified motor and gearbox fit."""

    kp_fw: float = MISSING
    """Firmware proportional gain [-]."""

    vin: float = MISSING
    """Nominal supply voltage [V]. Overridden by :attr:`vin_range` when set."""

    vin_range: tuple[float, float] | None = None
    """Range of the per-environment supply voltage [V], sampled once and held across resets."""

    vin_drop_gain_range: tuple[float, float] | None = None
    """Range of the per-environment supply sag gain [V/(N.m)], sampled once and held across resets.

    Each environment's group joints share one battery. Its voltage is ``vin - gain * sum_j |tau_j|``
    from the previous step's torques, floored at :attr:`vin_min` and 0 V. None disables sag.
    """

    vin_min: float = -math.inf
    """Lower bound on the sagged supply voltage [V]. The default keeps only the 0 V floor."""

    min_delay: int = 0
    """Minimum command delay [physics steps]."""

    max_delay: int = 0
    """Maximum command delay [physics steps]. Zero disables the delay."""

    delay_hold_prob: float = 0.0
    """Probability of keeping the current lag instead of resampling it [-]."""

    delay_update_period: int = 0
    """Physics steps between lag resamples, with a random per-environment phase. Zero resamples every step."""

    backlash_joint_template: str | None = None
    """Name of each servo's passive gearbox-play hinge, formatted with the servo joint name.

    For example ``"passive_{}_backlash"``. The hinge must be a revolute sibling of the servo joint. When set, the
    firmware reads servo plus play angle, while back-EMF, friction, and applied torque stay on the servo joint.
    None disables backlash.
    """

    backlash_velocity_feedback: bool = False
    """Whether the firmware and back-EMF also read the play-hinge velocity.

    True uses Newton's ``NewtonActuatorBacklashAPI``, which reads servo plus play velocity. False, the default,
    keeps the servo velocity as the BAM reference does. Only used with :attr:`backlash_joint_template`.
    """

    stiff_frictionloss: bool = True
    """Use a stiff friction-constraint solver reference, since MJWarp has no noslip solver.

    Authored as the servo joints' ``mjc:solreffriction`` and ``mjc:solimpfriction``, which Newton's MuJoCo solver
    imports.
    """

    def validate_config(self) -> None:
        """Reject command-lag settings the lag sampler cannot draw from.

        Raises:
            ValueError: If the lag range is negative or reversed, :attr:`delay_hold_prob` lies outside
                [0, 1], or :attr:`delay_update_period` is negative.
        """
        if self.min_delay < 0:
            raise ValueError(f"min_delay must not be negative, got {self.min_delay}")
        if self.max_delay < self.min_delay:
            raise ValueError(f"max_delay ({self.max_delay}) must not be below min_delay ({self.min_delay})")
        if not 0.0 <= self.delay_hold_prob <= 1.0:
            raise ValueError(f"delay_hold_prob must lie in [0, 1], got {self.delay_hold_prob}")
        if self.delay_update_period < 0:
            raise ValueError(f"delay_update_period must not be negative, got {self.delay_update_period}")
