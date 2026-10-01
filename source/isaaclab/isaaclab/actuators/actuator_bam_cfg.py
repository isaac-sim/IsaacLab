# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .actuator_base_cfg import ActuatorBaseCfg


@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """Configuration for the BAM voltage-domain servo actuator.

    Identified motor and friction coefficients are read from the asset's
    ``NewtonBamDriveAPI`` actuator prims. Configuration values explicitly override
    the USD values; no parameter file is loaded during simulation.

    Note:
        :attr:`~isaaclab.actuators.ActuatorBaseCfg.stiffness` and
        :attr:`~isaaclab.actuators.ActuatorBaseCfg.damping` are unused by this model. Its
        position loop runs in the firmware domain, parameterized by :attr:`kp_fw`, and its
        damping is the physical back-EMF of the motor.

    This model requires ``use_newton_actuators=True`` with Newton's MJWarp solver.
    The drive publishes its friction budget and viscous damping into the solver and reads
    the external load from its generalized forces. Other backends and solvers raise an error.
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

    parameter_overrides: dict[str, float | int] | None = None
    """Explicit overrides of USD BAM coefficients, keyed by snake-case parameter name.

    Unspecified values are retained per joint from the asset. For example,
    ``{"friction_base": 0.005}`` overrides Coulomb friction [N.m] for this group.
    See the BAM parameter table in the actuator guide for supported names and units.
    Unknown names and missing required coefficients raise an error at authoring time.
    Prefer :attr:`kp_fw` and :attr:`vin` for deployment settings; these take precedence
    over entries in this mapping. Solver inertia remains owned by the joint USD or
    :attr:`~isaaclab.actuators.ActuatorBaseCfg.armature`. The drive has no separate
    rotor-inertia parameter.
    """

    kp_fw: float | None = None
    """Firmware proportional gain [-]. None preserves the USD value."""

    vin: float | None = None
    """Nominal supply voltage [V]. None preserves the USD value.

    Overridden by :attr:`vin_range` when that is set.
    """

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

    friction_scale_range: tuple[float, float] | None = None
    """Range to sample the per-environment friction-budget scale from [-].

    The scale multiplies the whole velocity-independent friction budget (Coulomb, Stribeck
    and load-dependent terms). Sampled once at construction. Per-episode friction randomization
    writes the drive's ``friction_scale`` through
    :func:`~isaaclab.actuators.newton.write_group_parameter`. If None, the scale is 1.
    """

    min_delay: int = 0
    """Minimum command delay [physics steps]. Defaults to 0."""

    max_delay: int = 0
    """Maximum command delay [physics steps]. Defaults to 0, which disables the delay."""

    delay_hold_prob: float = 0.0
    """Probability of keeping the current lag instead of resampling it [-]. Defaults to 0."""

    delay_update_period: int = 0
    """Number of physics steps between lag resamples. Defaults to 0, which resamples every step.

    When positive, a phase offset in ``[0, delay_update_period)`` staggers the resamples rather
    than synchronizing them. The Newton drive draws the phase and lag per driven joint.
    """

    stiff_frictionloss: bool = True
    """Stiffen the joint friction constraint on a solver that applies the friction itself [-].

    MuJoCo Warp has no noslip solver: its friction-loss constraint stays soft and a statically
    held joint creeps. Setting this replaces the constraint's solver reference with the stiff,
    timestep-independent form the reference implementation uses.
    """
