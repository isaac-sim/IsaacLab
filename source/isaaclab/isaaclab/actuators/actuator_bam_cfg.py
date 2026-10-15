# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

from isaaclab.utils.configclass import configclass

from .actuator_base_cfg import ActuatorBaseCfg

BAM_DRIVE_API = "NewtonBamDriveAPI"
"""USD API schema token of the BAM drive."""

BAM_MODEL_FLAGS: dict[str, dict[str, int]] = {
    "m1": {"stribeck": 0, "load_dependent": 0, "quadratic": 0},
    "m2": {"stribeck": 1, "load_dependent": 0, "quadratic": 0},
    "m5": {"stribeck": 1, "load_dependent": 1, "quadratic": 0},
    "m6": {"stribeck": 1, "load_dependent": 1, "quadratic": 1},
}
"""Drive flags of the friction terms each BAM model enables."""


@configclass
class BamMotorCfg:
    """Identified BAM motor and gearbox fit, shared by all joints of an actuator group.

    A field left as None keeps the value of the joint's USD-authored BAM actuator. A new actuator, on a
    joint without one, uses the default given in the field's description; fields without a default are
    then required by the drive.
    """

    model: Literal["m1", "m2", "m5", "m6"] | None = None
    """Friction variant: Coulomb, Stribeck, directional load-dependent, or quadratic."""

    kt: float | None = None
    """Motor torque/back-EMF constant [N.m/A or V.s/rad]."""

    resistance: float | None = None
    """Winding resistance [Ohm]."""

    error_gain: float | None = None
    """Position-error-to-duty-cycle factor per unit of firmware gain [1/rad]."""

    max_pwm: float | None = None
    """Maximum duty-cycle magnitude [-]. Defaults to 1.0."""

    max_current: float | None = None
    """Firmware current limit [A]. Zero disables current limiting."""

    friction_base: float | None = None
    """Coulomb friction [N.m]."""

    friction_viscous: float | None = None
    """Viscous friction coefficient [N.m.s/rad], applied as passive joint damping.

    None keeps the joint's own viscous friction coefficient.
    """

    friction_stribeck: float | None = None
    """Additional near-rest friction [N.m]. Defaults to 0.0."""

    dtheta_stribeck: float | None = None
    """Stribeck decay velocity [rad/s]. Defaults to 1.0."""

    alpha: float | None = None
    """Stribeck decay exponent [-]. Defaults to 1.0."""

    load_friction_motor: float | None = None
    """Motor-side load-dependent friction coefficient [-]. Defaults to 0.0."""

    load_friction_external: float | None = None
    """External-side load-dependent friction coefficient [-]. Defaults to 0.0."""

    load_friction_motor_stribeck: float | None = None
    """Motor-side near-rest load-dependent friction coefficient [-]. Defaults to 0.0."""

    load_friction_external_stribeck: float | None = None
    """External-side near-rest load-dependent friction coefficient [-]. Defaults to 0.0."""

    load_friction_motor_quad: float | None = None
    """Motor-side quadratic load-coupling coefficient [1/(N.m)]. Defaults to 0.0."""

    load_friction_external_quad: float | None = None
    """External-side quadratic load-coupling coefficient [1/(N.m)]. Defaults to 0.0."""

    def validate_config(self) -> None:
        """Reject friction variants the BAM drive does not implement.

        Raises:
            ValueError: If the model is set and is not m1, m2, m5, or m6.
        """
        if self.model is not None and self.model not in BAM_MODEL_FLAGS:
            raise ValueError(f"Unsupported BAM model {self.model!r}; expected m1, m2, m5, or m6.")


@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """Configuration for the BAM voltage-domain servo actuator.

    Requires ``use_newton_actuators=True`` with Newton's MJWarp solver. Like other explicit
    actuators, it is authored onto the selected joints' USD BAM actuators: a drive value left as
    None, here or in :attr:`motor`, keeps the asset's value. Joints without a BAM actuator get a
    new one, which needs the motor fit, :attr:`kp_fw` and :attr:`vin`.

    If :attr:`~isaaclab.actuators.ActuatorBaseCfg.actuator_effort_limit` is None, the torque
    limit is the stall torque ``max(vin_range) * motor.kt / motor.resistance``, using
    :attr:`vin` when :attr:`vin_range` is unset. When the configuration does not set the
    torque constant, resistance and voltage, the USD actuator's limit applies.
    """

    class_type: type | None = None
    """None: Newton builds the drive from the authored USD schema."""

    stiffness: dict[str, float] | float | None = None
    """Unused. The firmware gain is :attr:`kp_fw`."""

    damping: dict[str, float] | float | None = None
    """Unused. Damping is the motor's back-EMF."""

    motor: BamMotorCfg | None = None
    """Identified motor and gearbox fit. None keeps the USD actuator's fit."""

    kp_fw: float | None = None
    """Firmware proportional gain [-]."""

    vin: float | None = None
    """Nominal supply voltage [V]. Overridden by :attr:`vin_range` when set."""

    vin_range: tuple[float, float] | None = None
    """Range of the per-environment supply voltage [V], sampled once and held across resets."""

    vin_drop_gain_range: tuple[float, float] | None = None
    """Range of the per-environment supply sag gain [V/(N.m)], sampled once and held across resets.

    The sagged supply is ``vin - gain * sum_j |tau_j|``, floored at :attr:`vin_min` and 0 V. None
    disables sag.
    """

    vin_min: float | None = None
    """Lower bound on the sagged supply voltage [V]. Defaults to -inf, which keeps only the 0 V floor."""

    min_delay: int | None = None
    """Minimum command delay [physics steps]. Defaults to 0."""

    max_delay: int | None = None
    """Maximum command delay [physics steps]. Zero disables the delay. Defaults to 0."""

    delay_hold_prob: float | None = None
    """Probability of keeping the current lag instead of resampling it [-]. Defaults to 0.0."""

    delay_update_period: int | None = None
    """Physics steps between lag resamples, with a random per-environment phase. Zero resamples every step.

    Defaults to 0.
    """

    backlash_joint_template: str | None = None
    """Name of each servo's passive gearbox-play hinge, formatted with the servo joint name.

    For example ``"passive_{}_backlash"``. When set, the firmware reads servo plus play angle, while back-EMF,
    friction, and applied torque stay on the servo joint. None disables backlash.
    """

    stiff_frictionloss: bool = True
    """Use a stiff friction-constraint solver reference, since MJWarp has no noslip solver."""
