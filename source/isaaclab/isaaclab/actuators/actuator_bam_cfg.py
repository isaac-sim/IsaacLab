# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass

from .actuator_base_cfg import ActuatorBaseCfg

BAM_DRIVE_API = "NewtonBamDriveAPI"
"""USD API schema token of the BAM drive."""


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
        if self.model not in ("m1", "m2", "m5", "m6"):
            raise ValueError(f"Unsupported BAM model {self.model!r}; expected m1, m2, m5, or m6.")


@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """Configuration for the BAM voltage-domain servo actuator.

    Requires ``use_newton_actuators=True`` with Newton's MJWarp solver. Like other explicit
    actuators, it replaces existing USD actuators on the selected joints.

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

    The sagged supply is ``vin - gain * sum_j |tau_j|``. None disables sag.
    """

    vin_min: float | None = None
    """Lower bound on the sagged supply voltage [V]. None for no bound."""

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

    For example ``"passive_{}_backlash"``. When set, the firmware reads servo plus play angle, while back-EMF,
    friction, and applied torque stay on the servo joint. None disables backlash.
    """

    stiff_frictionloss: bool = True
    """Use a stiff friction-constraint solver reference, since MJWarp has no noslip solver."""
