# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-native BAM servo drive.

Importing this module registers :class:`DriveBam` under the ``NewtonBamDriveAPI`` USD token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp
from newton.actuators import ComponentKind, DriveBase, register_actuator_component

from isaaclab.actuators.actuator_bam_cfg import BAM_DRIVE_API, BamActuatorCfg

from .bam_kernels import bam_delay_kernel, bam_friction_kernel, bam_motor_kernel, bam_state_reset_kernel


class DriveBam(DriveBase):
    """BAM voltage-domain servo drive.

    Each step delays the position command, sags the supply, runs the firmware proportional
    loop to a PWM duty cycle, converts it to motor torque, and publishes the gearbox friction
    budget that MJWarp applies as a constraint. Only the position target is consumed.

    The drive clamps its own effort: authoring a registered clamping schema next to the
    unregistered ``NewtonBamDriveAPI`` token hides the drive from Newton's schema discovery.
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

    PER_DOF_PARAMS = (
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
    """Per-DOF parameter arrays, shape ``(N,)`` each."""

    external_torque: wp.array[float] | None
    """External gearbox load of the previous MJWarp solve [N.m], shape ``(N,)``. Bound by the MJWarp bridge."""

    env_dof_stride: int
    """DOFs per environment, which share one supply and one command-delay stream."""

    startup_settings: tuple | None
    """BAM settings the start-up sampling ran with, or None if it has not run."""

    friction_budget: wp.array[float] | None
    """Velocity-independent friction budget of the last step [N.m], shape ``(N,)``."""

    effective_vin: wp.array[float] | None
    """Supply voltage after sag of the last step [V], shape ``(N,)``."""

    motor_torque: wp.array[float] | None
    """Motor torque of the last step, before friction [N.m], shape ``(N,)``."""

    @dataclass
    class State(DriveBase.State):
        """Double-buffered per-DOF state of the BAM drive."""

        prev_motor_torque: wp.array[float] | None = None
        """Unclamped motor torque of the previous step, used for supply sag [N.m], shape ``(N,)``."""

        prev_applied_torque: wp.array[float] | None = None
        """Clamped effort of the previous step, used for friction [N.m], shape ``(N,)``."""

        delay_ring: wp.array2d[float] | None = None
        """Past position commands [rad or m], shape ``(N, max(max_delay, 1))``."""

        delay_lag: wp.array[wp.int32] | None = None
        """Current command lag [physics steps], shape ``(N,)``."""

        delay_fill: wp.array[wp.int32] | None = None
        """Number of valid entries in :attr:`delay_ring`, shape ``(N,)``."""

        delay_step_count: wp.array[wp.int32] | None = None
        """Steps since the last reset, shape ``(N,)``."""

        delay_phase: wp.array[wp.int32] | None = None
        """Offset of the lag-resampling period per environment, shape ``(N,)``."""

        delay_rng_seed: wp.array[wp.int32] | None = None
        """Episode seed per environment, shape ``(N,)``."""

        delay_update_period: int = 0
        """Lag-resampling period [physics steps]."""

        env_dof_stride: int = 1
        """DOFs sharing one command-delay stream."""

        delay_seed: int = 0
        """Base seed of the lag and phase draws."""

        reset_count: int = 0
        """Number of resets applied, which decorrelates successive episodes."""

        def assign(self, other: DriveBam.State) -> None:
            """Copy another state into this one, keeping this state's array storage."""
            self.prev_motor_torque.assign(other.prev_motor_torque)
            self.prev_applied_torque.assign(other.prev_applied_torque)
            self.delay_ring.assign(other.delay_ring)
            self.delay_lag.assign(other.delay_lag)
            self.delay_fill.assign(other.delay_fill)
            self.delay_step_count.assign(other.delay_step_count)
            self.delay_phase.assign(other.delay_phase)
            self.delay_rng_seed.assign(other.delay_rng_seed)
            self.delay_update_period = other.delay_update_period
            self.env_dof_stride = other.env_dof_stride
            self.delay_seed = other.delay_seed
            self.reset_count = other.reset_count

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            self.reset_count += 1
            wp.launch(
                bam_state_reset_kernel,
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
        """Resolve shared settings and per-DOF coefficients from a USD actuator prim.

        Args:
            args: Authored attribute values, keyed by snake-case name.

        Returns:
            Shared settings and per-DOF coefficients.

        Raises:
            ValueError: If names are unknown, coefficients are missing, or delays are invalid.
        """
        unknown = set(args) - cls.SHARED_PARAMS - set(cls.PER_DOF_PARAMS)
        if unknown:
            raise ValueError(f"Unknown BAM parameter(s): {', '.join(sorted(unknown))}")
        required = {"kt", "resistance", "error_gain", "max_pwm", "kp_fw", "vin", "friction_base"}
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
        defaults = dict.fromkeys(cls.PER_DOF_PARAMS, 0.0)
        defaults.update(
            kp_scale=1.0,
            kd_scale=1.0,
            friction_scale=1.0,
            dtheta_stribeck=1.0,
            alpha=1.0,
            max_effort=float(args["vin"]) * float(args["kt"]) / float(args["resistance"]),
        )
        for name in cls.PER_DOF_PARAMS:
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
        """Initialize the drive from Newton's resolved arguments.

        Args:
            stribeck: Whether the Stribeck friction terms are active.
            load_dependent: Whether gearbox friction grows with the transmitted torque.
            quadratic: Whether the quadratic load-coupling term is active.
            vin_min: Lower bound on the sagged supply voltage [V].
            min_delay: Minimum command delay [physics steps].
            max_delay: Maximum command delay [physics steps]. ``0`` disables the delay.
            delay_hold_prob: Probability of keeping the current lag instead of resampling it.
            delay_update_period: Physics steps between lag resamples. ``0`` resamples every step.
            delay_seed: Base seed of the lag and phase draws.
            per_dof: One ``(N,)`` array per entry of :attr:`PER_DOF_PARAMS`.
        """
        self.stribeck = stribeck
        self.load_dependent = load_dependent
        self.quadratic = quadratic
        self.vin_min = vin_min
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delay_hold_prob = delay_hold_prob
        self.delay_update_period = delay_update_period
        self.delay_seed = delay_seed
        # Newton's selection API reads and writes per-DOF parameters by attribute name.
        for name in self.PER_DOF_PARAMS:
            setattr(self, name, per_dof[name])

        self.external_torque = None
        self.env_dof_stride = 1
        self.startup_settings = None
        self.friction_budget = None
        self.effective_vin = None
        self.motor_torque = None
        self._delayed_target: wp.array[float] | None = None
        self._next_state_arrays: dict[str, wp.array] = {}

    """
    Newton component interface.
    """

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self.friction_budget = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self.effective_vin = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self.motor_torque = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._delayed_target = wp.empty(num_actuators, dtype=wp.float32, device=device)
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
        # Seed environments and phases identically in both ping-pong buffers.
        if self.max_delay > 0:
            wp.launch(
                bam_state_reset_kernel,
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
        # A drive stepped without the bridge would silently read no external load.
        if self.external_torque is None:
            raise RuntimeError("BAM requires an MJWarp solver binding before stepping (external_torque is unbound).")
        num_actuators = len(forces)
        scratch = self._next_state_arrays
        wp.launch(
            kernel=bam_delay_kernel,
            dim=num_actuators,
            inputs=[
                target_pos,
                target_pos_indices,
                state.delay_ring,
                state.delay_lag,
                state.delay_fill,
                state.delay_step_count,
                state.delay_phase,
                self.min_delay,
                self.max_delay,
                self.delay_hold_prob,
                self.delay_update_period,
                state.delay_rng_seed,
            ],
            outputs=[
                self._delayed_target,
                scratch["delay_ring"],
                scratch["delay_lag"],
                scratch["delay_fill"],
                scratch["delay_step_count"],
            ],
            device=device,
        )
        wp.launch(
            kernel=bam_motor_kernel,
            dim=num_actuators,
            inputs=[
                positions,
                velocities,
                self._delayed_target,
                pos_indices,
                vel_indices,
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
                self.vin_min,
                self.env_dof_stride,
            ],
            outputs=[self.motor_torque, self.effective_vin],
            device=device,
        )
        wp.launch(
            kernel=bam_friction_kernel,
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


def apply_bam_startup_sampling(drive: DriveBam, cfg: BamActuatorCfg) -> None:
    """Draw the per-environment supply voltage and sag gain of one BAM drive.

    A USD prim is shared by every clone, so these ranges are sampled once here instead and held
    constant across resets.

    Args:
        drive: The BAM drive, with its environment stride set.
        cfg: The group's actuator configuration.
    """
    for attr, value_range in (("vin", cfg.vin_range), ("sag_gain", cfg.vin_drop_gain_range)):
        if value_range is None:
            continue
        per_env = wp.to_torch(getattr(drive, attr)).view(-1, drive.env_dof_stride)
        samples = torch.empty(per_env.shape[0], 1, device=per_env.device, dtype=per_env.dtype)
        samples.uniform_(*value_range)
        per_env.copy_(samples.expand_as(per_env))


register_actuator_component(BAM_DRIVE_API, DriveBam, ComponentKind.DRIVE)
