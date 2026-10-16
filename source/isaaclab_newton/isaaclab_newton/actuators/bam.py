# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The BAM servo part that Newton's native BAM path does not provide.

Newton runs the firmware and motor (:class:`~newton.actuators.DriveBAM`), the supply
(:class:`~newton.actuators.Battery`), the random command lag (:class:`~newton.actuators.InputProcessorRandomDelay`),
the play-angle read (:class:`~newton.actuators.InputProcessorBacklash`) and the effort clamp. Isaac Lab adds the
gearbox friction that MJWarp applies as a DOF friction constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp

from isaaclab.actuators.actuator_bam_cfg import BAM_MODEL_FLAGS, BamMotorCfg

from .bam_kernels import bam_friction_budget_kernel, gather_dof_values_kernel, zero_masked_envs_kernel

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.assets import Articulation


class BamFriction:
    """Gearbox friction of BAM servos, which MJWarp applies as a DOF friction constraint.

    Before each actuator step, :meth:`compute_budget` turns the servo velocity, the previous solve's external
    load and the previous step's applied effort into a velocity-independent friction budget. After the step,
    :meth:`record_applied_effort` stores the clamped effort for the next budget. Both run in CUDA graphs.
    """

    COEFFICIENTS: ClassVar[tuple[str, ...]] = (
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
    )
    """Fit coefficients copied per DOF from :class:`~isaaclab.actuators.BamMotorCfg`."""

    def __init__(self, motor: BamMotorCfg, dof_indices: wp.array, dofs_per_env: int):
        """Initialize the friction state of a set of servo DOFs.

        Args:
            motor: Identified motor and gearbox fit.
            dof_indices: DOF index of each servo slot into ``joint_qd``-shaped arrays, shape ``(N,)``.
            dofs_per_env: DOFs per environment in the layout of :paramref:`dof_indices`.
        """
        device = dof_indices.device
        num_dofs = len(dof_indices)
        flags = BAM_MODEL_FLAGS[motor.model]
        self.stribeck = flags["stribeck"]
        self.load_dependent = flags["load_dependent"]
        self.quadratic = flags["quadratic"]
        self.dof_indices = dof_indices
        """DOF index of each slot into ``joint_qd``-shaped arrays, shape ``(N,)``."""
        self._dofs_per_env = dofs_per_env
        self.coefficients = {
            name: wp.array(np.full(num_dofs, getattr(motor, name), dtype=np.float32), device=device)
            for name in self.COEFFICIENTS
        }
        """Fit coefficients per slot, shape ``(N,)`` each."""
        self.friction_scale = wp.ones(num_dofs, dtype=wp.float32, device=device)
        """Multiplier of the friction budget [-], shape ``(N,)``."""
        self.external_torque = wp.zeros(num_dofs, dtype=wp.float32, device=device)
        """External gearbox load of the previous solve [N.m], shape ``(N,)``."""
        self.prev_applied_torque = wp.zeros(num_dofs, dtype=wp.float32, device=device)
        """Clamped effort of the previous step [N.m], shape ``(N,)``. Zeroed on reset."""
        self.friction_budget = wp.zeros(num_dofs, dtype=wp.float32, device=device)
        """Velocity-independent friction budget of the last step [N.m], shape ``(N,)``."""

    def compute_budget(self, joint_qd: wp.array) -> None:
        """Compute this step's friction budget.

        Args:
            joint_qd: Joint velocities of the simulation state [rad/s], indexed by :attr:`dof_indices`.
        """
        wp.launch(
            bam_friction_budget_kernel,
            dim=len(self.dof_indices),
            inputs=[
                joint_qd,
                self.dof_indices,
                self.external_torque,
                self.prev_applied_torque,
                self.friction_scale,
                *(self.coefficients[name] for name in self.COEFFICIENTS),
                self.stribeck,
                self.load_dependent,
                self.quadratic,
            ],
            outputs=[self.friction_budget],
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
        """Clear the applied-effort history of the masked environments.

        Args:
            env_mask: Per-environment boolean reset mask, or None to reset every environment.
        """
        wp.launch(
            zero_masked_envs_kernel,
            dim=len(self.dof_indices),
            inputs=[env_mask, self.dof_indices, self._dofs_per_env],
            outputs=[self.prev_applied_torque],
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
    """Gearbox friction of the group's servos."""

    adapter: NewtonActuatorAdapter
    """Adapter whose batteries power the group."""

    battery_offset: int
    """Index of environment 0's battery in :attr:`NewtonActuatorAdapter.battery`."""

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
        """Live unloaded supply voltage [V], shape ``(num_envs,)``. Writable."""
        return self.adapter.battery.nominal_voltage[self.battery_offset : self.battery_offset + self.num_envs]

    @property
    def sag_gain(self) -> wp.array:
        """Live supply sag per unit of total motor torque [V/(N.m)], shape ``(num_envs,)``. Writable."""
        return self.adapter.battery.sag_gain[self.battery_offset : self.battery_offset + self.num_envs]

    @property
    def voltage(self) -> wp.array:
        """Supply voltage of the last step [V], shape ``(num_envs,)``."""
        return self.adapter.battery.voltage[self.battery_offset : self.battery_offset + self.num_envs]


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
    groups = getattr(getattr(articulation, "_actuator_control", None), "bam_groups", {})
    if name not in groups:
        raise KeyError(f"No Newton-executed BAM actuator group '{name}'.")
    return groups[name]
