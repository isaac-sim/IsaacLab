# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-native BAM servo controller.

The Warp controller runs inside Newton's actuator pipeline. MuJoCo Warp supplies the external
load and resolves the load-dependent friction budget alongside its other constraints through
:mod:`isaaclab_newton.physics.mjwarp_actuator_bridge`.

The controller owns the stochastic command delay, battery sag, firmware PWM controller, DC-motor
equation and gearbox friction. State is
double-buffered and CUDA-graph-safe. Identified coefficients are carried by the USD actuator prim.

The effort clamp is part of the controller: mixing a registered clamping schema with the
unregistered ``NewtonBamControlAPI`` token can hide BAM from Newton's actuator schema discovery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import warp as wp
from newton.actuators import ComponentKind, Controller, register_actuator_component

BAM_CONTROL_API: str = "NewtonBamControlAPI"
"""USD API schema token that maps an actuator prim onto :class:`ControllerBam`."""

_DELAY_RNG_STRIDE: int = 7919
"""Prime stride that decorrelates the per-DOF delay random streams."""

_is_registered: bool = False
"""Whether :func:`register_bam_actuator_component` has already run in this process."""


@wp.kernel
def _bam_motor_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    target_pos: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    kp_fw: wp.array[float],
    kp_scale: wp.array[float],
    kd_scale: wp.array[float],
    vin: wp.array[float],
    sag_gain: wp.array[float],
    kt: wp.array[float],
    resistance: wp.array[float],
    error_gain: wp.array[float],
    max_pwm: wp.array[float],
    max_current: wp.array[float],
    prev_motor_torque: wp.array[float],
    delay_ring: wp.array2d[float],
    delay_lag: wp.array[wp.int32],
    delay_fill: wp.array[wp.int32],
    delay_step_count: wp.array[wp.int32],
    delay_phase: wp.array[wp.int32],
    vin_min: float,
    env_dof_stride: int,
    min_delay: int,
    max_delay: int,
    delay_hold_prob: float,
    delay_update_period: int,
    delay_seed: int,
    motor_torque: wp.array[float],
    effective_vin: wp.array[float],
    next_ring: wp.array2d[float],
    next_lag: wp.array[wp.int32],
    next_fill: wp.array[wp.int32],
    next_step_count: wp.array[wp.int32],
):
    """Delay the command, sag the supply and run the firmware and DC-motor stages."""
    i = wp.tid()

    target = target_pos[target_pos_indices[i]]
    step_count = delay_step_count[i]
    # Reset clears the cached lag; a held or staggered update must still honor the minimum.
    lag = wp.max(delay_lag[i], min_delay)
    fill = delay_fill[i]

    if max_delay > 0:
        # Resample the lag before reading, matching the reference update policy:
        # a draw is attempted only on the environment's phase of the update period and
        # may be skipped again with probability ``delay_hold_prob``.
        should_update = True
        if delay_update_period > 0:
            should_update = ((step_count + delay_phase[i]) % delay_update_period) == 0
        rng = wp.rand_init(delay_seed, i * _DELAY_RNG_STRIDE + step_count)
        if should_update:
            if delay_hold_prob > 0.0:
                should_update = wp.randf(rng) >= delay_hold_prob
        if should_update:
            lag = wp.randi(rng, min_delay, max_delay + 1)

        # ``delay_ring[i, 0]`` is the previous step's command, so a lag of ``k`` reads
        # column ``k - 1``. The read is clamped to the number of commands seen so far,
        # which is how the reference ring behaves before it has filled up.
        if lag > 0 and fill > 0:
            column = wp.min(lag - 1, fill - 1)
            target = delay_ring[i, column]

        next_ring[i, 0] = target_pos[target_pos_indices[i]]
        for column in range(1, max_delay):
            next_ring[i, column] = delay_ring[i, column - 1]
        next_fill[i] = wp.min(fill + 1, max_delay)
    else:
        next_fill[i] = 0
    next_lag[i] = lag
    next_step_count[i] = step_count + 1

    # All joints sharing one supply sag together, so the drop is driven by the summed
    # magnitude of the torques the environment's joints drew on the previous step.
    load = float(0.0)
    base = i - (i % env_dof_stride)
    for offset in range(env_dof_stride):
        load += wp.abs(prev_motor_torque[base + offset])
    vin_eff = vin[i] - sag_gain[i] * load
    vin_eff = wp.max(vin_eff, vin_min)
    effective_vin[i] = vin_eff

    # kd_scale scales the electrical damping only: the velocity enters the firmware law
    # and the torque equation solely through the back-EMF term.
    scaled_vel = velocities[vel_indices[i]] * kd_scale[i]

    measured = positions[pos_indices[i]]

    duty = (target - measured) * (kp_fw[i] * kp_scale[i]) * error_gain[i]
    if max_current[i] > 0.0:
        duty_center = kt[i] * scaled_vel / vin_eff
        duty_span = resistance[i] * max_current[i] / vin_eff
        duty = wp.clamp(duty, duty_center - duty_span, duty_center + duty_span)
    duty = wp.clamp(duty, -max_pwm[i], max_pwm[i])

    volts = vin_eff * duty
    motor_torque[i] = kt[i] * volts / resistance[i] - (kt[i] * kt[i]) * scaled_vel / resistance[i]


@wp.kernel
def _bam_friction_kernel(
    velocities: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    motor_torque: wp.array[float],
    external_torque_in: wp.array[float],
    armature: wp.array[float],
    friction_scale: wp.array[float],
    friction_base: wp.array[float],
    friction_viscous: wp.array[float],
    friction_stribeck: wp.array[float],
    dtheta_stribeck: wp.array[float],
    alpha: wp.array[float],
    load_friction_motor: wp.array[float],
    load_friction_external: wp.array[float],
    load_friction_motor_stribeck: wp.array[float],
    load_friction_external_stribeck: wp.array[float],
    load_friction_motor_quad: wp.array[float],
    load_friction_external_quad: wp.array[float],
    max_effort: wp.array[float],
    prev_motor_torque: wp.array[float],
    prev_applied_torque: wp.array[float],
    prev_joint_vel: wp.array[float],
    needs_velocity_seed: wp.array[wp.int32],
    dt: float,
    stribeck: int,
    load_dependent: int,
    quadratic: int,
    solver_applies_friction: int,
    forces: wp.array[float],
    friction_budget: wp.array[float],
    viscous_damping: wp.array[float],
    next_prev_motor: wp.array[float],
    next_prev_applied: wp.array[float],
    next_prev_vel: wp.array[float],
    next_needs_seed: wp.array[wp.int32],
):
    """Size the gearbox friction budget and emit the actuator torque.

    Uses the solver's external load when bound, otherwise a rotor-momentum estimate.
    Solver-hosted execution publishes the budget for the solver to resolve; standalone
    controller execution applies a torque-level stiction clip.
    """
    i = wp.tid()

    joint_vel = velocities[vel_indices[i]]
    # A freshly reset joint has no previous velocity to differentiate against; reporting a
    # zero acceleration on that step avoids the spike a zeroed cache would produce.
    previous_vel = prev_joint_vel[i]
    if needs_velocity_seed[i] != 0:
        previous_vel = joint_vel

    motor_tau = motor_torque[i]
    if external_torque_in:
        ext_tau = external_torque_in[i]
    else:
        ext_tau = armature[i] * (joint_vel - previous_vel) / dt - prev_applied_torque[i]

    stribeck_coeff = float(0.0)
    if stribeck != 0:
        stribeck_coeff = wp.exp(-wp.pow(wp.abs(joint_vel) / dtheta_stribeck[i], alpha[i]))

    prev_tau = prev_motor_torque[i]
    budget = friction_base[i]
    if stribeck != 0:
        budget += stribeck_coeff * friction_stribeck[i]
    if load_dependent != 0:
        budget += wp.abs(ext_tau * load_friction_external[i] - prev_tau * load_friction_motor[i])
        if stribeck != 0:
            budget += stribeck_coeff * wp.abs(
                ext_tau * load_friction_external_stribeck[i] - prev_tau * load_friction_motor_stribeck[i]
            )
            if quadratic != 0:
                # Driving (motor wins) loads the gearbox through the external torque;
                # back-driving (load wins) loads it through the motor torque.
                abs_ext = wp.abs(ext_tau)
                abs_motor = wp.abs(prev_tau)
                quad_term = load_friction_motor_quad[i] * abs_motor * abs_motor
                if abs_motor > abs_ext:
                    quad_term = load_friction_external_quad[i] * abs_ext * abs_ext
                budget += stribeck_coeff * quad_term
    budget *= friction_scale[i]

    friction_budget[i] = budget
    viscous_damping[i] = friction_viscous[i]

    applied = motor_tau
    if solver_applies_friction == 0:
        net_tau = motor_tau + ext_tau
        # Torque that would bring the joint to a stop in one timestep.
        tau_stop = (armature[i] / dt) * joint_vel + net_tau
        clip = budget + friction_viscous[i] * wp.abs(joint_vel)
        applied = motor_tau - wp.sign(tau_stop) * wp.min(wp.abs(tau_stop), clip)
    # The safety clamp on top of the electrical limit the duty-cycle model already enforces.
    # Owned here rather than composed as a clamping component, so that the cached previous
    # effort is the one that was really applied -- and so that the actuator prim carries no
    # USD-registered schema alongside the unregistered BAM token (see the module docstring).
    applied = wp.clamp(applied, -max_effort[i], max_effort[i])
    forces[i] = applied

    next_prev_motor[i] = motor_tau
    next_prev_applied[i] = applied
    next_prev_vel[i] = joint_vel
    next_needs_seed[i] = 0


@wp.kernel
def _bam_state_reset_kernel(
    mask: wp.array[wp.bool],
    prev_motor_torque: wp.array[float],
    prev_applied_torque: wp.array[float],
    prev_joint_vel: wp.array[float],
    needs_velocity_seed: wp.array[wp.int32],
    delay_ring: wp.array2d[float],
    delay_lag: wp.array[wp.int32],
    delay_fill: wp.array[wp.int32],
    delay_step_count: wp.array[wp.int32],
    delay_phase: wp.array[wp.int32],
    delay_update_period: int,
    phase_seed: int,
):
    """Clear the previous-step caches and the delay state of the masked DOFs."""
    i = wp.tid()
    if mask:
        if not mask[i]:
            return
    prev_motor_torque[i] = 0.0
    prev_applied_torque[i] = 0.0
    prev_joint_vel[i] = 0.0
    needs_velocity_seed[i] = 1
    delay_lag[i] = 0
    delay_fill[i] = 0
    delay_step_count[i] = 0
    for column in range(delay_ring.shape[1]):
        delay_ring[i, column] = 0.0
    if delay_update_period > 0:
        delay_phase[i] = wp.randi(wp.rand_init(phase_seed, i), 0, delay_update_period)
    else:
        delay_phase[i] = 0


class ControllerBam(Controller):
    """Newton-native BAM voltage-domain servo controller.

    One step runs: delay the position command, sag the supply with the previous step's
    load, run the firmware proportional controller to a PWM duty cycle, convert that to a
    motor torque through the DC-motor equation, size the gearbox friction budget from the
    previous motor torque and the external load, and finally either publish the budget to
    the solver (:attr:`solver_applies_friction`) or clip the torque with it in place.

    The model consumes only the position target; the modelled firmware has no torque input,
    so feed-forward efforts and velocity targets are ignored.

    Per-environment randomization is exposed through the parameter arrays :attr:`vin`,
    :attr:`sag_gain`, :attr:`friction_scale`, :attr:`kp_scale` and :attr:`kd_scale`, whose
    values are updated through :func:`~isaaclab.actuators.newton.write_group_parameter`.
    """

    SHARED_PARAMS = {
        "stribeck",
        "load_dependent",
        "quadratic",
        "vin_min",
        "min_delay",
        "max_delay",
        "delay_hold_prob",
        "delay_update_period",
        "delay_seed",
    }

    external_torque: wp.array[float] | None
    """External torque on the gearbox [N.m], shape ``(N,)``, or None to use the estimator.

    A backend that can read the true generalized forces binds its own array here before the
    first step; otherwise the controller falls back to a rotor-momentum
    estimate ``armature * (dq - dq_prev) / dt - tau_applied_prev``.
    """

    solver_applies_friction: bool
    """Whether the physics solver, rather than this controller, applies the friction budget.

    Must be set before the first step, and therefore before CUDA-graph capture: the flag is
    passed to the kernel as a launch scalar and is baked into a captured graph.
    """

    env_dof_stride: int
    """Number of consecutive DOFs that share one supply, i.e. this actuator's DOFs per environment.

    Set by :class:`~isaaclab.actuators.newton.NewtonActuatorAdapter`, the first object that
    knows the environment count. Left at ``1``, the battery sag is driven by each joint's own
    load instead of the group's.
    """

    friction_budget: wp.array[float] | None
    """Velocity-independent friction budget of the last step [N.m], shape ``(N,)``."""

    viscous_damping: wp.array[float] | None
    """Viscous friction coefficient published alongside the budget [N.m.s/rad], shape ``(N,)``."""

    effective_vin: wp.array[float] | None
    """Supply voltage after the load-induced sag of the last step [V], shape ``(N,)``."""

    motor_torque: wp.array[float] | None
    """Motor-side torque of the last step, before any friction [N.m], shape ``(N,)``."""

    _PER_DOF_PARAMS = (
        "kp_fw",
        "kp_scale",
        "kd_scale",
        "vin",
        "sag_gain",
        "friction_scale",
        "kt",
        "resistance",
        "armature",
        "error_gain",
        "max_pwm",
        "max_current",
        "friction_base",
        "friction_viscous",
        "friction_stribeck",
        "dtheta_stribeck",
        "alpha",
        "load_friction_motor",
        "load_friction_external",
        "load_friction_motor_stribeck",
        "load_friction_external_stribeck",
        "load_friction_motor_quad",
        "load_friction_external_quad",
        "max_effort",
    )
    """Per-DOF parameter arrays, in the order :meth:`resolve_arguments` fills them."""

    @dataclass
    class State(Controller.State):
        """Double-buffered per-DOF state of the BAM controller."""

        prev_motor_torque: wp.array[float] | None = None
        """Motor-side torque of the previous step [N.m], shape ``(N,)``."""

        prev_applied_torque: wp.array[float] | None = None
        """Actuator torque emitted on the previous step [N.m], shape ``(N,)``."""

        prev_joint_vel: wp.array[float] | None = None
        """Joint velocity of the previous step [rad/s or m/s], shape ``(N,)``."""

        needs_velocity_seed: wp.array[wp.int32] | None = None
        """``1`` while the previous-velocity cache still has to be seeded, shape ``(N,)``."""

        delay_ring: wp.array2d[float] | None = None
        """Ring of past position commands [rad or m], shape ``(N, max(max_delay, 1))``."""

        delay_lag: wp.array[wp.int32] | None = None
        """Command lag currently applied [physics steps], shape ``(N,)``."""

        delay_fill: wp.array[wp.int32] | None = None
        """Number of valid entries in :attr:`delay_ring`, shape ``(N,)``."""

        delay_step_count: wp.array[wp.int32] | None = None
        """Steps taken since the last reset, shape ``(N,)``."""

        delay_phase: wp.array[wp.int32] | None = None
        """Per-DOF offset of the lag-resampling period, shape ``(N,)``."""

        delay_update_period: int = 0
        """Update period the phase is redrawn against on reset [physics steps]."""

        delay_seed: int = 0
        """Base seed of the phase draw."""

        reset_count: int = 0
        """Number of resets applied, which decorrelates successive phase draws."""

        def assign(self, other: ControllerBam.State) -> None:
            """Copy controller history and reset metadata while preserving array storage.

            Args:
                other: State to copy from, with matching array shapes and devices.
            """
            for field in fields(self):
                value = getattr(other, field.name)
                if isinstance(value, wp.array):
                    getattr(self, field.name).assign(value)
                else:
                    setattr(self, field.name, value)

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is not None:
                if mask.dtype is not wp.bool or mask.ndim != 1:
                    raise ValueError("BAM reset mask must be a one-dimensional Boolean array")
                if len(mask) != len(self.prev_motor_torque):
                    raise ValueError(
                        f"BAM reset mask length ({len(mask)}) must match state length ({len(self.prev_motor_torque)})"
                    )
                if mask.device != self.prev_motor_torque.device:
                    raise ValueError(
                        f"BAM reset mask device ({mask.device}) must match state device"
                        f" ({self.prev_motor_torque.device})"
                    )
            self.reset_count += 1
            wp.launch(
                _bam_state_reset_kernel,
                dim=len(self.prev_motor_torque),
                inputs=[
                    mask,
                    self.prev_motor_torque,
                    self.prev_applied_torque,
                    self.prev_joint_vel,
                    self.needs_velocity_seed,
                    self.delay_ring,
                    self.delay_lag,
                    self.delay_fill,
                    self.delay_step_count,
                    self.delay_phase,
                    self.delay_update_period,
                    self.delay_seed + self.reset_count,
                ],
                device=self.prev_motor_torque.device,
            )

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve scalar coefficients from a self-contained USD actuator prim.

        Args:
            args: Authored attribute values, keyed by snake-case name.

        Returns:
            Shared settings and per-DOF coefficients for Newton to assemble into arrays.

        Raises:
            ValueError: If coefficients are missing, names are unknown, or delays are invalid.
        """
        unknown = set(args) - cls.SHARED_PARAMS - set(cls._PER_DOF_PARAMS)
        if unknown:
            raise ValueError(f"Unknown BAM parameter(s): {', '.join(sorted(unknown))}")
        required = {
            "kt",
            "resistance",
            "armature",
            "error_gain",
            "max_pwm",
            "kp_fw",
            "vin",
            "friction_base",
            "friction_viscous",
        }
        if args.get("stribeck", 0):
            required.update(("friction_stribeck", "dtheta_stribeck", "alpha"))
        if args.get("load_dependent", 0):
            required.update(("load_friction_motor", "load_friction_external"))
            if args.get("stribeck", 0):
                required.update(("load_friction_motor_stribeck", "load_friction_external_stribeck"))
        if args.get("quadratic", 0):
            if not args.get("stribeck", 0) or not args.get("load_dependent", 0):
                raise ValueError("BAM quadratic friction requires stribeck and load_dependent")
            required.update(("load_friction_motor_quad", "load_friction_external_quad"))
        missing = required - args.keys()
        if missing:
            raise ValueError(f"{BAM_CONTROL_API} is missing coefficient(s): {', '.join(sorted(missing))}")

        min_delay = int(args.get("min_delay", 0))
        max_delay = int(args.get("max_delay", 0))
        delay_hold_prob = float(args.get("delay_hold_prob", 0.0))
        if min_delay < 0:
            raise ValueError(f"min_delay must not be negative, got {min_delay}")
        if max_delay < min_delay:
            raise ValueError(f"max_delay ({max_delay}) must not be below min_delay ({min_delay})")
        if not 0.0 <= delay_hold_prob <= 1.0:
            raise ValueError(f"delay_hold_prob must lie in [0, 1], got {delay_hold_prob}")

        resolved: dict[str, Any] = {
            "stribeck": int(args.get("stribeck", 0)),
            "load_dependent": int(args.get("load_dependent", 0)),
            "quadratic": int(args.get("quadratic", 0)),
            "vin_min": float(args.get("vin_min", -math.inf)),
            "min_delay": min_delay,
            "max_delay": max_delay,
            "delay_hold_prob": delay_hold_prob,
            "delay_update_period": int(args.get("delay_update_period", 0)),
            "delay_seed": int(args.get("delay_seed", 0)),
        }

        # Zero disables optional friction terms and the firmware current limiter.
        defaults = dict.fromkeys(cls._PER_DOF_PARAMS, 0.0)
        defaults.update(
            kp_scale=1.0, kd_scale=1.0, friction_scale=1.0, dtheta_stribeck=1.0, alpha=1.0, max_effort=math.inf
        )
        for name in cls._PER_DOF_PARAMS:
            resolved[name] = float(args.get(name, defaults[name]))
        return resolved

    def __init__(
        self,
        *,
        stribeck: int = 0,
        load_dependent: int = 0,
        quadratic: int = 0,
        vin_min: float = -math.inf,
        min_delay: int = 0,
        max_delay: int = 0,
        delay_hold_prob: float = 0.0,
        delay_update_period: int = 0,
        delay_seed: int = 0,
        **per_dof: wp.array,
    ):
        """Initialize the controller from pre-built per-DOF parameter arrays.

        Args:
            stribeck: Whether the Stribeck friction terms are active.
            load_dependent: Whether the gearbox friction grows with the transmitted torque.
            quadratic: Whether the quadratic load-coupling term is active.
            vin_min: Lower bound on the supply voltage after the load-induced sag [V].
            min_delay: Minimum command delay [physics steps].
            max_delay: Maximum command delay [physics steps]. ``0`` disables the delay.
            delay_hold_prob: Probability of keeping the current lag instead of resampling it.
            delay_update_period: Physics steps between lag resamples. ``0`` resamples every step.
            delay_seed: Base seed of the lag and phase draws.
            per_dof: One ``(N,)`` float array per entry of :attr:`_PER_DOF_PARAMS`.

        Raises:
            ValueError: If a per-DOF array is missing or its shape does not match the others.
        """
        self.stribeck = int(stribeck)
        self.load_dependent = int(load_dependent)
        self.quadratic = int(quadratic)
        self.vin_min = float(vin_min)
        self.min_delay = int(min_delay)
        self.max_delay = int(max_delay)
        self.delay_hold_prob = float(delay_hold_prob)
        self.delay_update_period = int(delay_update_period)
        self.delay_seed = int(delay_seed)

        missing = [name for name in self._PER_DOF_PARAMS if name not in per_dof]
        if missing:
            raise ValueError(f"ControllerBam is missing per-DOF parameter array(s): {', '.join(missing)}")
        unexpected = set(per_dof) - set(self._PER_DOF_PARAMS)
        if unexpected:
            raise ValueError(f"ControllerBam got unexpected parameter(s): {', '.join(sorted(unexpected))}")
        reference_shape = per_dof[self._PER_DOF_PARAMS[0]].shape
        for name in self._PER_DOF_PARAMS:
            array = per_dof[name]
            if array.shape != reference_shape:
                raise ValueError(f"'{name}' shape {array.shape} must match 'kp_fw' shape {reference_shape}")
            setattr(self, name, array)

        self.external_torque = None
        self.solver_applies_friction = False
        self.env_dof_stride = 1
        self.friction_budget = None
        self.viscous_damping = None
        self.effective_vin = None
        self.motor_torque = None
        self._next_state_arrays: dict[str, wp.array] = {}

    """
    Newton component interface.
    """

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self.friction_budget = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self.viscous_damping = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self.effective_vin = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self.motor_torque = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._next_state_arrays = {
            "prev_motor_torque": wp.zeros(num_actuators, dtype=wp.float32, device=device),
            "prev_applied_torque": wp.zeros(num_actuators, dtype=wp.float32, device=device),
            "prev_joint_vel": wp.zeros(num_actuators, dtype=wp.float32, device=device),
            "needs_velocity_seed": wp.zeros(num_actuators, dtype=wp.int32, device=device),
            "delay_ring": wp.zeros((num_actuators, max(self.max_delay, 1)), dtype=wp.float32, device=device),
            "delay_lag": wp.zeros(num_actuators, dtype=wp.int32, device=device),
            "delay_fill": wp.zeros(num_actuators, dtype=wp.int32, device=device),
            "delay_step_count": wp.zeros(num_actuators, dtype=wp.int32, device=device),
        }

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return True

    def set_env_dof_stride(self, stride: int) -> None:
        """Declare how many consecutive DOFs share one supply.

        Args:
            stride: DOFs per environment handled by this controller. The battery sag sums
                the previous motor torques over each such block.
        """
        if stride < 1:
            raise ValueError(f"env_dof_stride must be at least 1, got {stride}")
        self.env_dof_stride = int(stride)

    def state(self, num_actuators: int, device: wp.Device) -> ControllerBam.State:
        state = ControllerBam.State(
            prev_motor_torque=wp.zeros(num_actuators, dtype=wp.float32, device=device),
            prev_applied_torque=wp.zeros(num_actuators, dtype=wp.float32, device=device),
            prev_joint_vel=wp.zeros(num_actuators, dtype=wp.float32, device=device),
            needs_velocity_seed=wp.ones(num_actuators, dtype=wp.int32, device=device),
            delay_ring=wp.zeros((num_actuators, max(self.max_delay, 1)), dtype=wp.float32, device=device),
            delay_lag=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            delay_fill=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            delay_step_count=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            delay_phase=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            delay_update_period=self.delay_update_period,
            delay_seed=self.delay_seed,
        )
        # Draw the initial phase deterministically: the two ping-pong buffers must agree,
        # and a reproducible stream keeps rollouts comparable across runs.
        if self.delay_update_period > 0:
            wp.launch(
                _bam_state_reset_kernel,
                dim=num_actuators,
                inputs=[
                    None,
                    state.prev_motor_torque,
                    state.prev_applied_torque,
                    state.prev_joint_vel,
                    state.needs_velocity_seed,
                    state.delay_ring,
                    state.delay_lag,
                    state.delay_fill,
                    state.delay_step_count,
                    state.delay_phase,
                    self.delay_update_period,
                    self.delay_seed,
                ],
                device=device,
            )
        return state

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: ControllerBam.State,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        del target_vel, feedforward, target_vel_indices  # the modelled firmware has no torque input
        num_actuators = len(forces)
        scratch = self._next_state_arrays
        wp.launch(
            kernel=_bam_motor_kernel,
            dim=num_actuators,
            inputs=[
                positions,
                velocities,
                target_pos,
                pos_indices,
                vel_indices,
                target_pos_indices,
                self.kp_fw,
                self.kp_scale,
                self.kd_scale,
                self.vin,
                self.sag_gain,
                self.kt,
                self.resistance,
                self.error_gain,
                self.max_pwm,
                self.max_current,
                state.prev_motor_torque,
                state.delay_ring,
                state.delay_lag,
                state.delay_fill,
                state.delay_step_count,
                state.delay_phase,
                self.vin_min,
                self.env_dof_stride,
                self.min_delay,
                self.max_delay,
                self.delay_hold_prob,
                self.delay_update_period,
                self.delay_seed,
            ],
            outputs=[
                self.motor_torque,
                self.effective_vin,
                scratch["delay_ring"],
                scratch["delay_lag"],
                scratch["delay_fill"],
                scratch["delay_step_count"],
            ],
            device=device,
        )
        wp.launch(
            kernel=_bam_friction_kernel,
            dim=num_actuators,
            inputs=[
                velocities,
                vel_indices,
                self.motor_torque,
                self.external_torque,
                self.armature,
                self.friction_scale,
                self.friction_base,
                self.friction_viscous,
                self.friction_stribeck,
                self.dtheta_stribeck,
                self.alpha,
                self.load_friction_motor,
                self.load_friction_external,
                self.load_friction_motor_stribeck,
                self.load_friction_external_stribeck,
                self.load_friction_motor_quad,
                self.load_friction_external_quad,
                self.max_effort,
                state.prev_motor_torque,
                state.prev_applied_torque,
                state.prev_joint_vel,
                state.needs_velocity_seed,
                dt,
                self.stribeck,
                self.load_dependent,
                self.quadratic,
                int(self.solver_applies_friction),
            ],
            outputs=[
                forces,
                self.friction_budget,
                self.viscous_damping,
                scratch["prev_motor_torque"],
                scratch["prev_applied_torque"],
                scratch["prev_joint_vel"],
                scratch["needs_velocity_seed"],
            ],
            device=device,
        )

    def update_state(self, current_state: ControllerBam.State, next_state: ControllerBam.State) -> None:
        for name, scratch in self._next_state_arrays.items():
            wp.copy(getattr(next_state, name), scratch)
        # The phase only changes on reset, so it is carried across rather than recomputed.
        wp.copy(next_state.delay_phase, current_state.delay_phase)


def apply_bam_startup_sampling(controller: ControllerBam, cfg: Any) -> None:
    """Draw the start-up per-environment quantities of one BAM controller.

    A USD prim is shared by every clone, so the ranges
    :class:`~isaaclab.actuators.BamActuatorCfg` exposes (``vin_range``,
    ``vin_drop_gain_range``, ``friction_scale_range``) cannot be authored per environment.
    They are drawn here instead, once the actuator exists: one value per environment,
    shared by that environment's joints and held constant across resets.

    Args:
        controller: The BAM controller to write, already bound to its environment stride.
        cfg: The group's :class:`~isaaclab.actuators.BamActuatorCfg`.
    """
    import torch  # noqa: PLC0415

    ranges = (
        ("vin", cfg.vin_range),
        ("sag_gain", cfg.vin_drop_gain_range),
        ("friction_scale", cfg.friction_scale_range),
    )
    for attr, value_range in ranges:
        if value_range is None:
            continue
        per_env = wp.to_torch(getattr(controller, attr)).view(-1, controller.env_dof_stride)
        samples = torch.empty(per_env.shape[0], 1, device=per_env.device, dtype=per_env.dtype)
        samples.uniform_(*value_range)
        per_env.copy_(samples.expand_as(per_env))


def register_bam_actuator_component() -> None:
    """Register :class:`ControllerBam` under the ``NewtonBamControlAPI`` USD schema token.

    Idempotent: Newton warns when a token is re-registered, so repeated calls are ignored.
    Both actuator construction paths -- Newton's ``ModelBuilder.add_usd`` and the PhysX-family
    :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.from_usd` -- resolve the token
    through the same registry, so registering once covers every backend.
    """
    global _is_registered  # noqa: PLW0603
    if _is_registered:
        return
    register_actuator_component(BAM_CONTROL_API, ControllerBam, ComponentKind.CONTROLLER)
    _is_registered = True


register_bam_actuator_component()
