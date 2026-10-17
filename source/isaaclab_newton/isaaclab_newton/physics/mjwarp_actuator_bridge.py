# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step MuJoCo Warp DOF channel for actuated DOFs.

Actuator models that publish a per-step dry-friction budget or read the generalized load on a DOF,
such as BAM's gearbox friction, go through :class:`MjWarpActuatorBridge`, which reads and writes
:attr:`~newton.solvers.SolverMuJoCo.mjw_model` and ``mjw_data`` directly. The pinned Newton can
publish per-step joint friction through ``ModelFlags.JOINT_DOF_FORCE_PROPERTIES``; moving the
friction publication onto it is a follow-up. Newton has no API for the generalized load on a
DOF, so that readback stays on ``mjw_data``.

The published friction is written into the MuJoCo Warp model after any joint-property resync,
so :attr:`~isaaclab.assets.Articulation.data` joint friction shows the authored seed value, not
the live budget.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp
from newton import Model
from newton.solvers import SolverBase, SolverMuJoCo

from .mjwarp_actuator_bridge_kernels import gather_external_torque_kernel, publish_dof_friction_kernel


class MjWarpActuatorBridge:
    """Per-step MuJoCo Warp DOF channel for a set of actuated DOFs."""

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
            dof_indices: Global Newton DOF index of each slot, shape ``(N,)``.
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

        # Newton DOF -> slot, -1 for DOFs outside the set.
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
