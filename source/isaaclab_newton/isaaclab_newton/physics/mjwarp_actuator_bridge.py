# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step MuJoCo Warp DOF channel for Newton actuator drives.

Drives that publish a per-step dry-friction budget or read the generalized load on a DOF go
through :class:`MjWarpActuatorBridge`, which reads and writes
:attr:`~newton.solvers.SolverMuJoCo.mjw_model` and ``mjw_data`` directly. The pinned Newton can
publish per-step joint friction through ``ModelFlags.JOINT_DOF_FORCE_PROPERTIES``; moving the
friction publication onto it is a follow-up. Newton has no API for the generalized load on a
DOF, so that readback stays on ``mjw_data``.

The published friction is written into the MuJoCo Warp model after any joint-property resync,
so :attr:`~isaaclab.assets.Articulation.data` joint friction shows the authored seed value, not
the live budget.
"""

from __future__ import annotations

from typing import ClassVar

import mujoco
import numpy as np
import warp as wp
from newton import Model
from newton.solvers import SolverBase, SolverMuJoCo

from .mjwarp_actuator_bridge_kernels import (
    gather_external_torque_kernel,
    publish_dof_friction_kernel,
    stiffen_friction_constraint_kernel,
    stiffen_model_friction_solver_kernel,
    vec5,
)


class MjWarpActuatorBridge:
    """Per-step MuJoCo Warp DOF channel for one Newton actuator."""

    STIFF_SOLREF_FRICTION: ClassVar[tuple[float, float]] = (-5.0e4, -2.0e2)
    """Stiff ``(-stiffness, -damping)`` friction solref from the reference BAM implementation [N.m/rad, N.m.s/rad]."""

    STIFF_SOLIMP_FRICTION: ClassVar[tuple[float, float, float, float, float]] = (0.99, 0.9999, 0.001, 0.5, 2.0)
    """Friction-constraint impedance profile paired with :attr:`STIFF_SOLREF_FRICTION` [-]."""

    @staticmethod
    def is_available(solver: SolverBase | None) -> bool:
        """Return whether *solver* steps MuJoCo Warp.

        CPU MuJoCo is unsupported: it still builds the MuJoCo Warp model, but steps the host MuJoCo
        model, which the bridge neither writes nor reads.
        """
        return isinstance(solver, SolverMuJoCo) and not solver.use_mujoco_cpu

    def __init__(self, solver: SolverMuJoCo, model: Model, dof_indices: wp.array, device: str):
        """Bind the bridge to a MuJoCo Warp solver.

        Args:
            solver: The active MuJoCo Warp solver.
            model: The Newton model the solver was built from.
            dof_indices: Global Newton DOF index of each actuator slot, shape ``(N,)``.
            device: Warp device the actuator arrays live on.

        Raises:
            RuntimeError: If the solver's DOF fields are not stored per world, so per-environment
                writes would alias one buffer.
        """
        self._device = device
        mjw_model = solver.mjw_model
        mjw_data = solver.mjw_data
        dof_map = solver.mjc_dof_to_newton_dof
        num_worlds = dof_map.shape[0]
        if mjw_model.dof_frictionloss.shape[0] != num_worlds:
            raise RuntimeError(
                "MuJoCo Warp's 'dof_frictionloss' is not expanded per world"
                f" (got {mjw_model.dof_frictionloss.shape[0]} rows for {num_worlds} worlds), so"
                " per-environment friction writes would alias a single buffer."
            )
        self._mjw_model = mjw_model
        self._mjw_data = mjw_data
        self._dof_map = dof_map
        self._launch_dim = (num_worlds, dof_map.shape[1])
        self._dof_indices = dof_indices
        self._model = model

        # Newton DOF -> actuator slot, -1 for DOFs this actuator does not drive.
        slots = np.full(model.joint_dof_count, -1, dtype=np.int32)
        indices = dof_indices.numpy().astype(np.int64)
        slots[indices] = np.arange(len(indices), dtype=np.int32)
        self._newton_dof_to_slot = wp.array(slots, dtype=wp.int32, device=device)
        self._friction_constraint_type = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)

    def publish_dof_friction(self, friction_budget: wp.array) -> None:
        """Write the actuator's dry-friction budget into ``dof_frictionloss`` for the next solve.

        Args:
            friction_budget: Velocity-independent friction budget per slot [N.m], shape ``(N,)``.
        """
        wp.launch(
            publish_dof_friction_kernel,
            dim=self._launch_dim,
            inputs=[self._dof_map, self._newton_dof_to_slot, friction_budget],
            outputs=[self._mjw_model.dof_frictionloss],
            device=self._device,
        )

    def gather_external_torque(self, external_torque: wp.array) -> None:
        """Fill *external_torque* with the previous solve's external load per slot.

        Applied body wrenches are not included.

        Args:
            external_torque: Destination, shape ``(N,)``. Written with
                ``-qfrc_bias + qfrc_constraint - qfrc_own_friction`` [N.m].
        """
        efc = self._mjw_data.efc
        wp.launch(
            gather_external_torque_kernel,
            dim=self._launch_dim,
            inputs=[
                self._dof_map,
                self._newton_dof_to_slot,
                self._mjw_data.qfrc_bias,
                self._mjw_data.qfrc_constraint,
                efc.type,
                efc.id,
                efc.force,
                self._mjw_data.nefc,
                self._friction_constraint_type,
            ],
            outputs=[external_torque],
            device=self._device,
        )

    def stiffen_friction_constraint(self) -> None:
        """Stiffen the managed DOFs' friction constraint in MuJoCo Warp and in the Newton model.

        Writing the Newton model too keeps the stiff values across joint-property resyncs.
        """
        solref = wp.vec2(*self.STIFF_SOLREF_FRICTION)
        solimp = vec5(*self.STIFF_SOLIMP_FRICTION)
        wp.launch(
            stiffen_friction_constraint_kernel,
            dim=self._launch_dim,
            inputs=[self._dof_map, self._newton_dof_to_slot, solref, solimp],
            outputs=[self._mjw_model.dof_solref, self._mjw_model.dof_solimp],
            device=self._device,
        )
        mujoco_attrs = self._model.mujoco
        wp.launch(
            stiffen_model_friction_solver_kernel,
            dim=len(self._dof_indices),
            inputs=[self._dof_indices, solref, solimp],
            outputs=[mujoco_attrs.solreffriction, mujoco_attrs.solimpfriction],
            device=self._device,
        )
