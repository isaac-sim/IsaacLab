# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step MuJoCo Warp DOF channel for Newton actuator components.

Some actuator models need more of the solver than
:class:`~newton.actuators.DriveBase` is handed: the BAM servo model publishes a
load-dependent dry-friction budget every control step and reads a generalized load estimate on
the gearbox. Newton exposes neither -- there is no supported per-step override of
``dof_frictionloss``, and ``State``'s extended-attribute whitelist carries
only ``mujoco:qfrc_actuator``. Both quantities *are* reachable through
:attr:`~newton.solvers.SolverMuJoCo.mjw_model` / ``mjw_data``, which are public attributes
without a stability contract.

This module is the single place that touches them. Everything else in Isaac Lab goes through
:class:`MjWarpActuatorBridge`, so a MuJoCo Warp layout change is a one-file fix and the
coupling is auditable.

Ordering contract
-----------------

Writes here go *straight into the MuJoCo Warp model* and are therefore invisible to
``model.joint_friction`` and to Isaac Lab's joint-property readers. Within one physics step
the order is:

1. :meth:`NewtonManager.step <isaaclab_newton.physics.NewtonManager.step>` drains the model
   change set eagerly, so a pending ``JOINT_DOF_PROPERTIES`` notification re-synchronizes
   ``dof_frictionloss`` from ``model.joint_friction`` **before** the graph replay;
2. :meth:`gather_external_torque` runs on the pre-actuator hook, reading the previous solve's
   forces;
3. the actuators compute;
4. :meth:`publish_dof_friction` runs on the post-actuator hook and overwrites the DOFs it
   owns, **after** any notify-driven re-sync and before the substeps consume them.

The published budget is therefore always the actuator's, but a reader of
:attr:`~isaaclab.assets.Articulation.data` joint friction sees the seeded authored value, not
the live budget. Report the live budget through the actuator's own telemetry instead.
"""

from __future__ import annotations

import numpy as np
import warp as wp

vec5 = wp.types.vector(length=5, dtype=wp.float32)
"""Solver-impedance vector type, matching MuJoCo Warp's ``dof_solimp`` element type."""


@wp.kernel(enable_backward=False)
def _publish_dof_friction_kernel(
    mjc_dof_to_newton_dof: wp.array2d[wp.int32],
    newton_dof_to_slot: wp.array[wp.int32],
    friction_budget: wp.array[float],
    dof_frictionloss: wp.array2d[float],
):
    """Scatter an actuator's per-DOF dry-friction budget into the MuJoCo model."""
    world, mjc_dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if newton_dof < 0:
        return
    slot = newton_dof_to_slot[newton_dof]
    if slot < 0:
        return
    dof_frictionloss[world, mjc_dof] = friction_budget[slot]


@wp.kernel(enable_backward=False)
def _gather_external_torque_kernel(
    mjc_dof_to_newton_dof: wp.array2d[wp.int32],
    newton_dof_to_slot: wp.array[wp.int32],
    qfrc_bias: wp.array2d[float],
    qfrc_constraint: wp.array2d[float],
    efc_type: wp.array2d[wp.int32],
    efc_id: wp.array2d[wp.int32],
    efc_force: wp.array2d[float],
    nefc: wp.array[wp.int32],
    friction_constraint_type: int,
    external_torque: wp.array[float],
):
    """Gather the external load on each managed DOF from the previous solve.

    The load the gearbox works against is the gravity/Coriolis bias plus the constraint
    forces, **minus** the DOF-friction constraint the actuator itself injected on the previous
    solve: leaving it in would make the load-dependent friction terms feed back on themselves.
    Applied body wrenches and other externally applied force channels are not included.
    """
    world, mjc_dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if newton_dof < 0:
        return
    slot = newton_dof_to_slot[newton_dof]
    if slot < 0:
        return
    own_friction = float(0.0)
    for row in range(nefc[world]):
        if efc_type[world, row] == friction_constraint_type and efc_id[world, row] == mjc_dof:
            own_friction += efc_force[world, row]
    external_torque[slot] = -qfrc_bias[world, mjc_dof] + qfrc_constraint[world, mjc_dof] - own_friction


@wp.kernel(enable_backward=False)
def _stiffen_friction_constraint_kernel(
    mjc_dof_to_newton_dof: wp.array2d[wp.int32],
    newton_dof_to_slot: wp.array[wp.int32],
    solref: wp.vec2,
    solimp: vec5,
    dof_solref: wp.array2d[wp.vec2],
    dof_solimp: wp.array2d[vec5],
):
    """Write a stiff friction-constraint solver reference onto the managed DOFs."""
    world, mjc_dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if newton_dof < 0:
        return
    if newton_dof_to_slot[newton_dof] < 0:
        return
    dof_solref[world, mjc_dof] = solref
    dof_solimp[world, mjc_dof] = solimp


@wp.kernel(enable_backward=False)
def _stiffen_model_friction_solver_kernel(
    dof_indices: wp.array[wp.uint32],
    solref: wp.vec2,
    solimp: vec5,
    model_solref: wp.array[wp.vec2],
    model_solimp: wp.array[vec5],
):
    """Mirror the stiffening onto the Newton model so a property re-sync reproduces it."""
    i = wp.tid()
    dof = wp.int32(dof_indices[i])
    model_solref[dof] = solref
    model_solimp[dof] = solimp


class MjWarpActuatorBridge:
    """Per-step MuJoCo Warp DOF channel for one Newton actuator.

    Owns the Newton-DOF to controller-slot mapping and the kernels that move data between an
    actuator component and MuJoCo Warp's device arrays. Construction validates that the solver
    really exposes per-world DOF storage: a non-expanded model field aliases one buffer across
    worlds, and per-environment writes into it would be silently wrong.
    """

    STIFF_SOLREF_FRICTION: tuple[float, float] = (-5.0e4, -2.0e2)
    """Timestep-independent ``(-stiffness, -damping)`` friction solref [N.m/rad, N.m.s/rad].

    MuJoCo Warp has no noslip solver, so the friction-loss constraint stays soft and a
    statically held joint creeps. These are the values the reference BAM implementation uses
    as the GPU-side substitute (``bam/mjlab.py``).
    """

    STIFF_SOLIMP_FRICTION: tuple[float, float, float, float, float] = (0.99, 0.9999, 0.001, 0.5, 2.0)
    """Friction-constraint impedance profile paired with :attr:`STIFF_SOLREF_FRICTION` [-]."""

    @staticmethod
    def is_available(solver: object) -> bool:
        """Return whether *solver* exposes the MuJoCo Warp device model this bridge needs.

        BAM requires MuJoCo's Warp model to publish its per-step friction budget. The
        native CPU MuJoCo backend does not expose that model and is unsupported.

        Args:
            solver: The active Newton solver, or ``None``.
        """
        return getattr(solver, "mjw_model", None) is not None

    def __init__(self, solver: object, dof_indices: wp.array, num_newton_dofs: int, device: str):
        """Bind the bridge to a MuJoCo Warp solver.

        Args:
            solver: The active :class:`~newton.solvers.SolverMuJoCo`.
            dof_indices: Global Newton DOF index of each controller slot, shape ``(N,)``.
            num_newton_dofs: Total DOF count of the Newton model.
            device: Warp device the actuator arrays live on.

        Raises:
            RuntimeError: If the solver does not expose the MuJoCo Warp model and data, or if
                its DOF fields are not stored per world.
        """
        self._device = device
        mjw_model = getattr(solver, "mjw_model", None)
        mjw_data = getattr(solver, "mjw_data", None)
        dof_map = getattr(solver, "mjc_dof_to_newton_dof", None)
        if mjw_model is None or mjw_data is None or dof_map is None:
            raise RuntimeError(
                "Per-step actuator friction requires the MuJoCo Warp solver's device model;"
                " it is unavailable on this solver (CPU MuJoCo is not supported)."
            )
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

        # Reverse map: Newton DOF -> controller slot, -1 for DOFs this actuator does not drive.
        slots = np.full(num_newton_dofs, -1, dtype=np.int32)
        indices = dof_indices.numpy().astype(np.int64)
        slots[indices] = np.arange(len(indices), dtype=np.int32)
        self._newton_dof_to_slot = wp.array(slots, dtype=wp.int32, device=device)

        import mujoco  # noqa: PLC0415

        self._friction_constraint_type = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)

    def publish_dof_friction(self, friction_budget: wp.array) -> None:
        """Write the actuator's dry-friction budget into the solver.

        MuJoCo Warp's friction-loss constraint reads ``dof_frictionloss``, so an in-place
        write takes effect on the next solver launch, including a replayed CUDA graph.

        Args:
            friction_budget: Velocity-independent friction budget per slot [N.m], shape ``(N,)``.
        """
        wp.launch(
            _publish_dof_friction_kernel,
            dim=self._launch_dim,
            inputs=[self._dof_map, self._newton_dof_to_slot, friction_budget],
            outputs=[self._mjw_model.dof_frictionloss],
            device=self._device,
        )

    def gather_external_torque(self, external_torque: wp.array) -> None:
        """Fill an actuator input array with the previous solve's external load per slot.

        This estimate includes bias and constraint forces, but excludes applied body wrenches
        and other externally applied force channels.

        Args:
            external_torque: Destination, shape ``(N,)``. Written with
                ``-qfrc_bias + qfrc_constraint - qfrc_own_friction`` [N.m].
        """
        efc = self._mjw_data.efc
        wp.launch(
            _gather_external_torque_kernel,
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
        """Stiffen the managed DOFs' friction constraint, once, at bind time.

        Written both into the MuJoCo Warp model (so it takes effect immediately) and into the
        Newton model's ``mujoco.solreffriction`` / ``solimpfriction`` attributes (so a later
        joint-property notification re-synchronizes to the same values instead of reverting to
        MuJoCo's soft defaults).
        """
        solref = wp.vec2(*self.STIFF_SOLREF_FRICTION)
        solimp = vec5(*self.STIFF_SOLIMP_FRICTION)
        wp.launch(
            _stiffen_friction_constraint_kernel,
            dim=self._launch_dim,
            inputs=[self._dof_map, self._newton_dof_to_slot, solref, solimp],
            outputs=[self._mjw_model.dof_solref, self._mjw_model.dof_solimp],
            device=self._device,
        )
        model_solref, model_solimp = self._model_friction_solver_attributes()
        if model_solref is None or model_solimp is None:
            return
        wp.launch(
            _stiffen_model_friction_solver_kernel,
            dim=len(self._dof_indices),
            inputs=[self._dof_indices, solref, solimp],
            outputs=[model_solref, model_solimp],
            device=self._device,
        )

    def _model_friction_solver_attributes(self) -> tuple[wp.array | None, wp.array | None]:
        """Return the Newton model's friction ``solref`` / ``solimp`` custom attributes."""
        from isaaclab_newton.physics.newton_manager import NewtonManager  # noqa: PLC0415

        model = NewtonManager.backend.model  # noqa: SLF001
        mujoco_attrs = getattr(model, "mujoco", None) if model is not None else None
        if mujoco_attrs is None:
            return None, None
        return getattr(mujoco_attrs, "solreffriction", None), getattr(mujoco_attrs, "solimpfriction", None)
