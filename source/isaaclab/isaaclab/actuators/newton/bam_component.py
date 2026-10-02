# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-native BAM servo drive.

The Warp drive runs inside Newton's actuator pipeline. MuJoCo Warp supplies the external
load and resolves the load-dependent friction budget alongside its other constraints through
:mod:`isaaclab_newton.physics.mjwarp_actuator_bridge`.

The drive owns the stochastic command delay, battery sag, firmware PWM control, DC-motor
equation and gearbox friction. State is
double-buffered and CUDA-graph-safe. Identified coefficients are carried by the USD actuator prim.

The effort clamp is part of the drive: mixing a registered clamping schema with the
unregistered ``NewtonBamDriveAPI`` token can hide BAM from Newton's actuator schema discovery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import warp as wp
from newton.actuators import ComponentKind, DriveBase, register_actuator_component

from .bam_kernels import _bam_friction_kernel, _bam_motor_kernel, _bam_state_reset_kernel

BAM_DRIVE_API: str = "NewtonBamDriveAPI"
"""USD API schema token that maps an actuator prim onto :class:`DriveBam`."""

_is_registered: bool = False
"""Whether :func:`register_bam_actuator_component` has already run in this process."""


class DriveBam(DriveBase):
    """Newton-native BAM voltage-domain servo drive implemented as a :class:`~newton.actuators.DriveBase`.

    One step runs: delay the position command, sag the supply with the previous step's
    unclamped motor torque, run the firmware proportional control loop to a PWM duty cycle, convert that to a
    motor torque through the DC-motor equation, size the gearbox friction budget from the
    previous clamped motor torque and the external load, and publish the budget to MJWarp.
    MJWarp applies friction alongside its other constraints; other solvers are unsupported.

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
    """Previous MJWarp solve's external gearbox load [N.m], shape ``(N,)``.

    The MJWarp bridge must bind this array before the first step and CUDA graph capture.
    """

    env_dof_stride: int
    """Consecutive DOFs sharing one supply and command delay: this actuator's DOFs per environment.

    Set by :class:`~isaaclab.actuators.newton.NewtonActuatorAdapter`, the first object that
    knows the environment count. Left at ``1``, each joint has its own battery and delay stream.
    """

    friction_budget: wp.array[float] | None
    """Velocity-independent friction budget of the last step [N.m], shape ``(N,)``."""

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
        "error_gain",
        "max_pwm",
        "max_current",
        "friction_base",
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
    class State(DriveBase.State):
        """Double-buffered per-DOF state of the BAM drive."""

        prev_motor_torque: wp.array[float] | None = None
        """Unclamped motor torque of the previous step, used for supply sag [N.m], shape ``(N,)``."""

        prev_applied_torque: wp.array[float] | None = None
        """Clamped effort emitted on the previous step, used for friction [N.m], shape ``(N,)``."""

        delay_ring: wp.array2d[float] | None = None
        """Ring of past position commands [rad or m], shape ``(N, max(max_delay, 1))``."""

        delay_lag: wp.array[wp.int32] | None = None
        """Command lag currently applied [physics steps], shape ``(N,)``."""

        delay_fill: wp.array[wp.int32] | None = None
        """Number of valid entries in :attr:`delay_ring`, shape ``(N,)``."""

        delay_step_count: wp.array[wp.int32] | None = None
        """Steps taken since the last reset, shape ``(N,)``."""

        delay_phase: wp.array[wp.int32] | None = None
        """Offset of the lag-resampling period, shared within each environment, shape ``(N,)``."""

        delay_rng_seed: wp.array[wp.int32] | None = None
        """Per-DOF seed of the current episode, read during graph replay, shape ``(N,)``."""

        delay_update_period: int = 0
        """Update period the phase is redrawn against on reset [physics steps]."""

        env_dof_stride: int = 1
        """Consecutive DOFs sharing one command-delay stream."""

        delay_seed: int = 0
        """Base seed of the lag and phase draws."""

        reset_count: int = 0
        """Number of resets applied, which decorrelates successive episodes' lag and phase draws."""

        def assign(self, other: DriveBam.State) -> None:
            """Copy drive history and reset metadata while preserving array storage.

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
                    self.delay_ring,
                    self.delay_lag,
                    self.delay_fill,
                    self.delay_step_count,
                    self.delay_phase,
                    self.delay_rng_seed,
                    self.delay_update_period,
                    self.env_dof_stride,
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
            "error_gain",
            "max_pwm",
            "kp_fw",
            "vin",
            "friction_base",
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
            raise ValueError(f"{BAM_DRIVE_API} is missing coefficient(s): {', '.join(sorted(missing))}")

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
            kp_scale=1.0,
            kd_scale=1.0,
            friction_scale=1.0,
            dtheta_stribeck=1.0,
            alpha=1.0,
            max_effort=float(args["vin"]) * float(args["kt"]) / float(args["resistance"]),
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
        """Initialize the drive from pre-built per-DOF parameter arrays.

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
            raise ValueError(f"DriveBam is missing per-DOF parameter array(s): {', '.join(missing)}")
        unexpected = set(per_dof) - set(self._PER_DOF_PARAMS)
        if unexpected:
            raise ValueError(f"DriveBam got unexpected parameter(s): {', '.join(sorted(unexpected))}")
        reference_shape = per_dof[self._PER_DOF_PARAMS[0]].shape
        for name in self._PER_DOF_PARAMS:
            array = per_dof[name]
            if array.shape != reference_shape:
                raise ValueError(f"'{name}' shape {array.shape} must match 'kp_fw' shape {reference_shape}")
            setattr(self, name, array)

        self.external_torque = None
        self.env_dof_stride = 1
        self.friction_budget = None
        self.effective_vin = None
        self.motor_torque = None
        self._next_state_arrays: dict[str, wp.array] = {}

    """
    Newton component interface.
    """

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self.friction_budget = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self.effective_vin = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self.motor_torque = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._next_state_arrays = {
            "prev_motor_torque": wp.zeros(num_actuators, dtype=wp.float32, device=device),
            "prev_applied_torque": wp.zeros(num_actuators, dtype=wp.float32, device=device),
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
        """Declare how many consecutive DOFs share one supply and command delay before creating state.

        Args:
            stride: DOFs per environment handled by this drive. The battery sag sums
                the previous motor torques over each such block; its joints share one delay draw.
        """
        if stride < 1:
            raise ValueError(f"env_dof_stride must be at least 1, got {stride}")
        self.env_dof_stride = int(stride)

    def state(self, num_actuators: int, device: wp.Device) -> DriveBam.State:
        state = DriveBam.State(
            prev_motor_torque=wp.zeros(num_actuators, dtype=wp.float32, device=device),
            prev_applied_torque=wp.zeros(num_actuators, dtype=wp.float32, device=device),
            delay_ring=wp.zeros((num_actuators, max(self.max_delay, 1)), dtype=wp.float32, device=device),
            delay_lag=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            delay_fill=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            delay_step_count=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            delay_phase=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            delay_rng_seed=wp.full(num_actuators, self.delay_seed, dtype=wp.int32, device=device),
            delay_update_period=self.delay_update_period,
            env_dof_stride=self.env_dof_stride,
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
                    state.delay_ring,
                    state.delay_lag,
                    state.delay_fill,
                    state.delay_step_count,
                    state.delay_phase,
                    state.delay_rng_seed,
                    self.delay_update_period,
                    self.env_dof_stride,
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
        state: DriveBam.State,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        if self.external_torque is None:
            raise RuntimeError("BAM requires an MJWarp solver binding before stepping (external_torque is unbound).")
        del target_vel, feedforward, target_vel_indices, dt  # the modelled firmware has no torque input
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
                state.delay_rng_seed,
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
                self.friction_scale,
                self.friction_base,
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
                state.prev_applied_torque,
                self.stribeck,
                self.load_dependent,
                self.quadratic,
            ],
            outputs=[
                forces,
                self.friction_budget,
                scratch["prev_motor_torque"],
                scratch["prev_applied_torque"],
            ],
            device=device,
        )

    def update_state(self, current_state: DriveBam.State, next_state: DriveBam.State) -> None:
        for name, scratch in self._next_state_arrays.items():
            wp.copy(getattr(next_state, name), scratch)
        # The phase and seed only change on reset.
        wp.copy(next_state.delay_phase, current_state.delay_phase)
        wp.copy(next_state.delay_rng_seed, current_state.delay_rng_seed)


def apply_bam_startup_sampling(drive: DriveBam, cfg: Any) -> None:
    """Draw the start-up per-environment quantities of one BAM drive.

    A USD prim is shared by every clone, so the ranges
    :class:`~isaaclab.actuators.BamActuatorCfg` exposes (``vin_range``,
    ``vin_drop_gain_range``) cannot be authored per environment.
    They are drawn here instead, once the actuator exists: one value per environment,
    shared by that environment's joints and held constant across resets.

    Args:
        drive: The BAM drive to write, already bound to its environment stride.
        cfg: The group's :class:`~isaaclab.actuators.BamActuatorCfg`.
    """
    import torch  # noqa: PLC0415

    ranges = (
        ("vin", cfg.vin_range),
        ("sag_gain", cfg.vin_drop_gain_range),
    )
    for attr, value_range in ranges:
        if value_range is None:
            continue
        per_env = wp.to_torch(getattr(drive, attr)).view(-1, drive.env_dof_stride)
        samples = torch.empty(per_env.shape[0], 1, device=per_env.device, dtype=per_env.dtype)
        samples.uniform_(*value_range)
        per_env.copy_(samples.expand_as(per_env))


def register_bam_actuator_component() -> None:
    """Register :class:`DriveBam` under the ``NewtonBamDriveAPI`` USD schema token.

    Idempotent: Newton warns when a token is re-registered, so repeated calls are ignored.
    Both actuator construction paths -- Newton's ``ModelBuilder.add_usd`` and the PhysX-family
    :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.from_usd` -- resolve the token
    through the same registry, so registering once covers every backend.
    """
    global _is_registered  # noqa: PLW0603
    if _is_registered:
        return
    register_actuator_component(BAM_DRIVE_API, DriveBam, ComponentKind.DRIVE)
    _is_registered = True


register_bam_actuator_component()
