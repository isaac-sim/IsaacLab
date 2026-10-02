# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass

from .actuator_base_cfg import ActuatorBaseCfg


@configclass
class BamMotorCfg:
    """Identified BAM motor and gearbox fit for one servo model.

    The fit is shared by all joints in an actuator group. Use separate groups for different
    fits. Firmware gain and supply voltage are deployment settings on :class:`BamActuatorCfg`.
    """

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
    """Required firmware current limit [A]. Set zero explicitly to disable current limiting."""

    friction_base: float = MISSING
    """Coulomb friction [N.m]."""

    friction_viscous: float = MISSING
    """Viscous friction coefficient [N.m.s/rad], used to initialize passive joint damping once."""

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
        """Reject friction variants not implemented by the BAM drive.

        Raises:
            ValueError: If the model is not m1, m2, m5, or m6.
        """
        if self.model not in ("m1", "m2", "m5", "m6"):
            raise ValueError(f"Unsupported BAM model {self.model!r}; expected m1, m2, m5, or m6.")


@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """Configuration for the BAM voltage-domain servo actuator.

    Motor and friction coefficients come from :attr:`motor`. Like other explicit actuators,
    this configuration replaces existing USD actuators on the selected joints. Newton reads
    the resulting ``NewtonBamDriveAPI`` prims without loading parameter files.

    Note:
        :attr:`~isaaclab.actuators.ActuatorBaseCfg.stiffness` and
        :attr:`~isaaclab.actuators.ActuatorBaseCfg.damping` are unused by this model. Its
        position loop runs in the firmware domain, parameterized by :attr:`kp_fw`, and its
        damping is the physical back-EMF of the motor.

    This model requires ``use_newton_actuators=True`` with Newton's MJWarp solver.
    The drive publishes its friction budget each step, initializes viscous damping once, and reads
    the external load from its generalized forces. Other backends and solvers raise an error.

    If :attr:`~isaaclab.actuators.ActuatorBaseCfg.actuator_effort_limit` is None, the motor
    torque limit [N.m] is ``max(vin_range) * motor.kt / motor.resistance``, using :attr:`vin`
    when :attr:`vin_range` is unset. An explicit actuator limit overrides this default.
    The separate joint effort limit remains unchanged.
    """

    class_type: type | None = None
    """No Isaac Lab-executed model; Newton constructs the drive from its USD schema."""

    stiffness: dict[str, float] | float | None = None
    """Unused by this model. Defaults to None so that a configuration validates unset.

    Configuration validation rejects an object that still holds the inherited ``MISSING``
    sentinel, so the field is defaulted here rather than left required.
    Leave it unset; the firmware gain is configured with :attr:`kp_fw`.
    """

    damping: dict[str, float] | float | None = None
    """Unused by this model. Defaults to None so that a configuration validates unset.

    See :attr:`stiffness`.
    """

    motor: BamMotorCfg = MISSING
    """Identified motor and gearbox fit shared by the joints in this group."""

    kp_fw: float = MISSING
    """Firmware proportional gain [-]."""

    vin: float = MISSING
    """Nominal supply voltage [V]. Overridden by :attr:`vin_range` when set."""

    vin_range: tuple[float, float] | None = None
    """Range to sample the per-environment supply voltage from [V].

    Sampled once at construction and held constant across resets, because a robot's battery
    does not change between episodes. Takes precedence over :attr:`vin`.
    """

    vin_drop_gain_range: tuple[float, float] | None = None
    """Range to sample the per-environment supply sag gain from [V/(N.m)].

    The gain models the voltage drop across the battery and wiring resistance under load,
    ``vin_eff = vin - gain * sum_j |tau_j|``. Sampled once at construction and held constant
    across resets. If None, the gain is zero and the supply does not sag.
    """

    vin_min: float | None = None
    """Lower bound on the supply voltage after the load-induced sag [V], or None for no bound."""

    min_delay: int = 0
    """Minimum command delay [physics steps]. Defaults to 0."""

    max_delay: int = 0
    """Maximum command delay [physics steps]. Defaults to 0, which disables the delay."""

    delay_hold_prob: float = 0.0
    """Probability of keeping the current lag instead of resampling it [-]. Defaults to 0."""

    delay_update_period: int = 0
    """Number of physics steps between lag resamples. Defaults to 0, which resamples every step.

    When positive, a phase offset in ``[0, delay_update_period)`` staggers the resamples rather
    than synchronizing them. The Newton drive draws the phase and lag per environment,
    shared by every joint in the group.
    """

    stiff_frictionloss: bool = True
    """Stiffen the joint friction constraint on a solver that applies the friction itself [-].

    MuJoCo Warp has no noslip solver: its friction-loss constraint stays soft and a statically
    held joint creeps. Setting this replaces the constraint's solver reference with the stiff,
    timestep-independent form the reference implementation uses.
    """
