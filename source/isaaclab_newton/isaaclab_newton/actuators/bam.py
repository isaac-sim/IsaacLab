# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The BAM servo parts that Newton's native BAM path does not provide.

Newton runs the firmware (:class:`~newton.actuators.DriveVoltageControlled`), the current and PWM limits
(:class:`~newton.actuators.ClampingCurrentPWM`), the DC motor (:class:`~newton.actuators.MotorDC`), the supply
(:class:`~newton.actuators.BatteryConstantResistance`), the command ring
(:class:`~newton.actuators.InputProcessorDelay`) and the play-angle read
(:class:`~newton.actuators.InputProcessorEncoder`). Isaac Lab adds a per-environment random command lag and the
gearbox friction that MJWarp applies as a DOF friction constraint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from newton.actuators import InputProcessorDelay

from isaaclab.actuators.actuator_bam_cfg import BAM_MODEL_FLAGS, BamMotorCfg

from .bam_kernels import (
    BamFrictionFit,
    bam_command_lag_kernel,
    bam_command_lag_reset_kernel,
    bam_friction_budget_kernel,
    bam_friction_reset_kernel,
    gather_dof_values_kernel,
)

if TYPE_CHECKING:
    from newton.actuators import BatteryConstantResistance

    from isaaclab.assets import Articulation


class BamCommandLag:
    """Random command lag of BAM servos, written into a Newton delay processor before each actuator step.

    :class:`~newton.actuators.InputProcessorDelay` replays commands with a fixed lag per DOF. Each step,
    :meth:`step` draws one lag per environment, shared by the environment's DOFs, and writes it into the
    processor's :attr:`~newton.actuators.InputProcessorDelay.delay_steps`. A draw is attempted every
    ``update_period`` steps on a random per-environment phase, and is skipped with probability ``hold_prob``.

    Each environment's stream is keyed by the DOF of its first slot, so lags over other DOFs draw independently. It
    restarts on :meth:`reset` from the seed ``reset_count``, the number of resets since construction.
    """

    def __init__(
        self,
        delay: InputProcessorDelay,
        slots: Sequence[int],
        dof_indices: Sequence[int],
        dofs_per_env: int,
        min_delay: int,
        max_delay: int,
        hold_prob: float = 0.0,
        update_period: int = 0,
    ):
        """Initialize the lag streams and seed the first episode.

        Args:
            delay: Delay processor whose lags are drawn.
            slots: Processor slots driven by this lag, shape ``(N,)``.
            dof_indices: DOF index of each slot, which locates its environment, shape ``(N,)``. Ordered
                environment-major, with the same DOFs in every environment.
            dofs_per_env: DOFs per environment in the layout of :paramref:`dof_indices`.
            min_delay: Minimum lag [physics steps].
            max_delay: Maximum lag [physics steps].
            hold_prob: Probability of keeping the current lag at a draw [-].
            update_period: Physics steps between draws. ``0`` draws every step.

        Raises:
            ValueError: If the lag range is invalid or exceeds the processor's buffer.
        """
        if not 0 <= min_delay <= max_delay <= delay.buf_depth:
            raise ValueError(
                f"Lag range [{min_delay}, {max_delay}] must be ordered, non-negative and within the delay buffer"
                f" depth {delay.buf_depth}."
            )
        if len(slots) != len(dof_indices):
            raise ValueError(f"Got {len(slots)} slots and {len(dof_indices)} DOF indices.")
        device = delay.delay_steps.device
        self.delay = delay
        """Delay processor whose :attr:`~newton.actuators.InputProcessorDelay.delay_steps` this lag writes."""
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.hold_prob = hold_prob
        self.update_period = update_period
        self.reset_count = 0
        """Number of resets applied, which decorrelates successive episodes."""
        self._slots = wp.array(np.asarray(slots, dtype=np.int32), dtype=wp.int32, device=device)
        self._dof_indices = wp.array(np.asarray(dof_indices, dtype=np.uint32), dtype=wp.uint32, device=device)
        self._dofs_per_env = dofs_per_env
        self._first_dof = int(dof_indices[0]) % dofs_per_env
        self.lag = wp.zeros(len(slots), dtype=wp.int32, device=device)
        """Lag of the last step [physics steps], shape ``(N,)``."""
        self.step_count = wp.zeros(len(slots), dtype=wp.int32, device=device)
        """Steps since the last reset, shape ``(N,)``."""
        self.phase = wp.zeros(len(slots), dtype=wp.int32, device=device)
        """Offset of the draw period, shape ``(N,)``."""
        self.rng_seed = wp.zeros(len(slots), dtype=wp.int32, device=device)
        """Episode seed of each slot's environment, shape ``(N,)``."""
        self._reseed(None)

    def step(self) -> None:
        """Draw this step's lags into the delay processor. Captured with the actuator step in CUDA graphs."""
        wp.launch(
            bam_command_lag_kernel,
            dim=len(self.lag),
            inputs=[
                self.lag,
                self.step_count,
                self.phase,
                self.rng_seed,
                self._slots,
                self.min_delay,
                self.max_delay,
                self.hold_prob,
                self.update_period,
            ],
            outputs=[self.delay.delay_steps],
            device=self.lag.device,
        )

    def reset(self, env_mask: wp.array | None = None) -> None:
        """Restart the lag streams of the masked environments with a new episode seed.

        Args:
            env_mask: Per-environment boolean reset mask, or None to reset every environment.
        """
        self.reset_count += 1
        self._reseed(env_mask)

    def _reseed(self, env_mask: wp.array | None) -> None:
        wp.launch(
            bam_command_lag_reset_kernel,
            dim=len(self.lag),
            inputs=[
                env_mask,
                self._dof_indices,
                self._dofs_per_env,
                self._first_dof,
                self.update_period,
                self.reset_count,
            ],
            outputs=[self.lag, self.step_count, self.phase, self.rng_seed],
            device=self.lag.device,
        )


class BamFriction:
    """Gearbox friction of BAM servos, which MuJoCo applies as a DOF friction constraint.

    Before each actuator step, :meth:`compute_budget` turns the servo velocity, the previous solve's external
    load and the previous step's applied effort into a velocity-independent friction budget, and writes it into the
    model's joint friction for the next solve. After the step, :meth:`record_applied_effort` stores the clamped
    effort for the next budget. Both run in CUDA graphs.
    """

    def __init__(self, motor: BamMotorCfg, dof_indices: wp.array, dofs_per_env: int):
        """Initialize the friction state of a set of servo DOFs.

        Args:
            motor: Identified motor and gearbox fit.
            dof_indices: DOF index of each servo slot into ``joint_qd``-shaped arrays, shape ``(N,)``.
            dofs_per_env: DOFs per environment in the layout of :paramref:`dof_indices`.
        """
        device = dof_indices.device
        num_dofs = len(dof_indices)
        self.dof_indices = dof_indices
        """DOF index of each slot into ``joint_qd``-shaped arrays, shape ``(N,)``."""
        self._dofs_per_env = dofs_per_env
        self.fit = BamFrictionFit()
        """Friction fit of the servos' motor."""
        self.fit.friction_base = motor.friction_base
        self.fit.friction_stribeck = motor.friction_stribeck
        self.fit.dtheta_stribeck = motor.dtheta_stribeck
        self.fit.alpha = motor.alpha
        self.fit.load_friction_motor = motor.load_friction_motor
        self.fit.load_friction_external = motor.load_friction_external
        self.fit.load_friction_motor_stribeck = motor.load_friction_motor_stribeck
        self.fit.load_friction_external_stribeck = motor.load_friction_external_stribeck
        self.fit.load_friction_motor_quad = motor.load_friction_motor_quad
        self.fit.load_friction_external_quad = motor.load_friction_external_quad
        flags = BAM_MODEL_FLAGS[motor.model]
        self.fit.stribeck = flags["stribeck"]
        self.fit.load_dependent = flags["load_dependent"]
        self.fit.quadratic = flags["quadratic"]
        self.friction_scale = wp.ones(num_dofs, dtype=wp.float32, device=device)
        """Multiplier of the friction budget [-], shape ``(N,)``."""
        self.external_torque = wp.zeros(num_dofs, dtype=wp.float32, device=device)
        """External gearbox load of the previous solve, as of the last budget [N.m], shape ``(N,)``."""
        self.prev_applied_torque = wp.zeros(num_dofs, dtype=wp.float32, device=device)
        """Clamped effort of the previous step [N.m], shape ``(N,)``. Zeroed on reset."""
        self.episode_start = wp.zeros(num_dofs, dtype=wp.bool, device=device)
        """Whether the next budget is the first of an episode, which reads no external load, shape ``(N,)``.

        Set on reset and cleared by the next budget.
        """
        self.friction_budget = wp.zeros(num_dofs, dtype=wp.float32, device=device)
        """Velocity-independent friction budget of the last step [N.m], shape ``(N,)``."""

    def compute_budget(
        self,
        joint_qd: wp.array,
        qfrc_bias: wp.array,
        qfrc_constraint: wp.array,
        qfrc_frictionloss: wp.array,
        joint_friction: wp.array,
    ) -> None:
        """Compute this step's friction budget and write it into the joint friction.

        The force arrays are the previous solve's, as Newton's MuJoCo solver exports them.

        Args:
            joint_qd: Joint velocities of the simulation state [rad/s].
            qfrc_bias: Gravity, Coriolis and centrifugal forces [N.m].
            qfrc_constraint: Constraint forces, including each DOF's friction loss [N.m].
            qfrc_frictionloss: Each DOF's own friction-loss force [N.m].
            joint_friction: Joint friction the solver applies [N.m]; the servo DOFs receive the budget.

        All arrays are indexed by :attr:`dof_indices`.
        """
        wp.launch(
            bam_friction_budget_kernel,
            dim=len(self.dof_indices),
            inputs=[
                joint_qd,
                self.dof_indices,
                qfrc_bias,
                qfrc_constraint,
                qfrc_frictionloss,
                self.prev_applied_torque,
                self.episode_start,
                self.friction_scale,
                self.fit,
            ],
            outputs=[self.external_torque, self.friction_budget, joint_friction],
            device=self.dof_indices.device,
        )

    def record_applied_effort(self, joint_f: wp.array) -> None:
        """Store the effort the actuators applied this step.

        Args:
            joint_f: Control efforts after the actuator step [N.m], indexed by :attr:`dof_indices`.
        """
        wp.launch(
            gather_dof_values_kernel,
            dim=len(self.dof_indices),
            inputs=[joint_f, self.dof_indices],
            outputs=[self.prev_applied_torque],
            device=self.dof_indices.device,
        )

    def reset(self, env_mask: wp.array | None = None) -> None:
        """Start a new episode in the masked environments.

        Clears their applied-effort history, and their next budget reads no external load: the solver keeps a
        reset environment's loads until it solves again. State writes without a reset, such as pushes, keep the
        load.

        Args:
            env_mask: Per-environment boolean reset mask, or None to reset every environment.
        """
        wp.launch(
            bam_friction_reset_kernel,
            dim=len(self.dof_indices),
            inputs=[env_mask, self.dof_indices, self._dofs_per_env],
            outputs=[self.prev_applied_torque, self.episode_start],
            device=self.dof_indices.device,
        )


@dataclass
class BamGroup:
    """Isaac Lab-side runtime of one Newton-executed BAM actuator group.

    Slot arrays are ordered environment-major, then by the group's joints.
    """

    num_envs: int
    """Number of environments."""

    friction: BamFriction
    """Gearbox friction state of the group's servos. Read and write it through :attr:`friction_scale` and
    :attr:`friction_budget`."""

    battery: BatteryConstantResistance
    """Newton batteries that include the group's, one per environment."""

    battery_ids: slice
    """Indices of the group's batteries in :attr:`battery`, environment-major and evenly spaced."""

    @property
    def friction_scale(self) -> wp.array:
        """Live multiplier of the friction budget [-], shape ``(num_envs, num_group_joints)``. Writable."""
        return self.friction.friction_scale.reshape((self.num_envs, -1))

    @property
    def friction_budget(self) -> wp.array:
        """Friction budget of the last step [N.m], shape ``(num_envs, num_group_joints)``."""
        return self.friction.friction_budget.reshape((self.num_envs, -1))

    @property
    def nominal_voltage(self) -> wp.array:
        """Live unloaded supply voltage [V], shape ``(num_envs,)``. Writable.

        A write takes effect at the battery update that follows the next actuator step, or at the next reset.
        """
        return self.battery.nominal_voltage[self.battery_ids]

    @property
    def battery_resistance(self) -> wp.array:
        """Live internal resistance of the battery [Ohm], shape ``(num_envs,)``. Writable."""
        return self.battery.resistance[self.battery_ids]


def get_bam_group(articulation: Articulation, name: str) -> BamGroup:
    """Return the Isaac Lab-side runtime of a Newton-executed BAM actuator group.

    Args:
        articulation: Newton articulation that owns the group.
        name: Actuator group name.

    Returns:
        The group's friction and supply state.

    Raises:
        KeyError: If the articulation has no Newton-executed BAM group *name*.
    """
    groups = articulation._actuator_control.bam_groups
    if name not in groups:
        raise KeyError(f"No Newton-executed BAM actuator group '{name}'.")
    return groups[name]
