# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging

import warp as wp
from newton import Model, eval_fk
from newton._src.solvers.kamino._src.core.joints import JointDoFType
from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.kinematics.joints import (
    extract_actuators_state_from_joints,
)
from newton._src.solvers.kamino._src.kinematics.resets import _reset_joints_of_select_worlds
from newton.solvers import SolverKamino

from isaaclab.physics import PhysicsManager

from .kamino_manager_cfg import KaminoSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)

# Kamino joint DoF type for a 6-DoF free joint (a floating base's root joint).
_FREE_DOF_TYPE = wp.constant(int(JointDoFType.FREE))


@wp.kernel(enable_backward=False)
def _gather_base_state_from_joints(
    base_joint_index: wp.array(dtype=wp.int32),
    joint_dof_type: wp.array(dtype=wp.int32),
    joint_coords_offset: wp.array(dtype=wp.int32),
    joint_dofs_offset: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    # outputs
    base_q: wp.array(dtype=wp.transformf),
    base_u: wp.array(dtype=vec6f),
):
    """Gather each world's base pose/twist from the base joint coordinates in the joint state.

    For each world ``wid``, when its base joint (``base_joint_index[wid]``) is a free joint,
    its 7 coordinates in ``joint_q`` (position + ``xyzw`` quaternion) and 6 DoFs in
    ``joint_qd`` (linear + angular twist) are exactly the base body's pose/twist relative to
    the world, expressed in the base joint frame -- i.e. precisely the ``base_q`` / ``base_u``
    the Kamino FK reset expects. They are written into the dense per-world output arrays.

    Reading from ``joint_q`` / ``joint_qd`` (NOT ``body_q``) is required: for a floating base,
    a root pose/velocity write (``write_root_*_to_sim``) lands in the free joint's
    ``joint_q[0:7]`` / ``joint_qd[0:6]`` head, while ``body_q`` is only refreshed *by* this FK
    reconcile and is stale at this point.

    Fixed-based systems carry no base pose in the joint state, so the output defaults to identity
    pose / zero twist; the FK solver then keeps the base at its reference pose. Worlds without a
    base joint (``base_joint_index < 0``) are likewise left at the default.
    """
    wid = wp.tid()
    # Default: identity pose / zero twist (keeps fixed / anchored bases at their reference).
    base_q[wid] = wp.transform_identity()
    base_u[wid] = vec6f(0.0)

    base_jid = base_joint_index[wid]
    if base_jid < 0:
        return
    if joint_dof_type[base_jid] != _FREE_DOF_TYPE:
        return

    c = joint_coords_offset[base_jid]
    d = joint_dofs_offset[base_jid]
    base_q[wid] = wp.transformf(
        wp.vec3(joint_q[c + 0], joint_q[c + 1], joint_q[c + 2]),
        wp.quat(joint_q[c + 3], joint_q[c + 4], joint_q[c + 5], joint_q[c + 6]),
    )
    base_u[wid] = vec6f(
        joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2], joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5]
    )


class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Kamino solver.

    Uses Newton's :class:`CollisionPipeline` unless
    :attr:`KaminoSolverCfg.use_collision_detector` is ``True``, in which case
    Kamino's internal collision detector handles contact generation.
    """

    _base_q: wp.array | None = None
    """Persistent per-world base body poses fed to the FK reset. Shape ``(num_worlds,)``, dtype ``wp.transformf``."""

    _base_u: wp.array | None = None
    """Persistent per-world base body twists fed to the FK reset. Shape ``(num_worlds,)``, dtype ``vec6f``."""

    @classmethod
    def _get_kamino_solver_cfg(cls) -> KaminoSolverCfg:
        cfg = PhysicsManager._cfg
        if cfg is None:
            raise RuntimeError("Physics manager is not initialized.")
        solver_cfg = getattr(cfg, "solver_cfg", None)
        if not isinstance(solver_cfg, KaminoSolverCfg):
            raise TypeError(f"Expected KaminoSolverCfg, got {type(solver_cfg).__name__}.")
        return solver_cfg

    @classmethod
    def _eval_fk(cls, world_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Reconcile body state from joint coordinates for the Kamino solver.

        Uses Kamino's loop-closure FK (:meth:`_forward_kamino`) when
        :attr:`KaminoSolverCfg.use_fk_solver` is enabled.

        For the Kamino (maximal-coordinate) solver, ``_forward_kamino`` runs ``solver.reset()``,
        which overwrites the authoritative ``state_0.body_q`` / ``body_qd``. Restricting the
        solve to ``world_mask`` keeps in-flight (non-reset) worlds untouched; a ``None``
        ``world_mask`` reconciles all worlds (the full-reconcile path used by
        :meth:`forward` when :attr:`_forward_full_reconcile` is set, and at initial setup).

        Args:
            world_mask: Per-world mask of worlds to reconcile (``None`` means all).
            fk_mask: Per-articulation mask, used only on the non-``use_fk_solver`` ``eval_fk``
                fallback.
        """
        if cls._get_kamino_solver_cfg().use_fk_solver:
            cls._forward_kamino(world_mask=world_mask)
        else:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, fk_mask)

    @classmethod
    def _forward_kamino(cls, world_mask: wp.array | None = None) -> None:
        """Kamino-specific forward kinematics via ``solver.reset()``.

        Used when :attr:`KaminoSolverCfg.use_fk_solver` is ``True``. Extracts actuated
        coordinates from Newton ``joint_q`` / ``joint_qd`` and passes ``actuator_q`` /
        ``actuator_u`` so Kamino FK resolves passive joints.

        The per-environment base (root) pose and twist are gathered from the base joint's
        coordinates in ``state_0.joint_q`` / ``joint_qd`` and passed as ``base_q`` / ``base_u``.
        This anchors the FK solve to the per-environment root state that resets write into the
        free joint's ``joint_q[0:7]`` / ``joint_qd[0:6]`` head (floating bases). Fixed
        bases carry no base pose in the joint state and keep their original pose.

        Args:
            world_mask: Per-world mask indicating which worlds to reset.
                Shape ``(num_worlds,)``, dtype ``wp.int32``. If None, resets all worlds.
        """

        solver_kamino = cls._solver._solver_kamino
        model_kamino = cls._solver._model_kamino
        effective_world_mask = world_mask if world_mask is not None else solver_kamino._all_worlds_mask

        # Extract the actuated joint state from the full joint state.
        extract_actuators_state_from_joints(
            model=model_kamino,
            world_mask=effective_world_mask,
            joint_q=cls._state_0.joint_q,
            joint_u=cls._state_0.joint_qd,
            actuator_q=solver_kamino._actuators_q,
            actuator_u=solver_kamino._actuators_u,
        )

        # Gather the per-world base pose/twist from the base joint coordinates in the joint
        # state (where a floating base's root pose/velocity write lands).
        wp.launch(
            _gather_base_state_from_joints,
            dim=model_kamino.size.num_worlds,
            inputs=[
                model_kamino.info.base_joint_index,
                model_kamino.joints.dof_type,
                model_kamino.joints.coords_offset,
                model_kamino.joints.dofs_offset,
                cls._state_0.joint_q,
                cls._state_0.joint_qd,
            ],
            outputs=[
                cls._base_q,
                cls._base_u,
            ],
            device=model_kamino.device,
        )

        # Run FK
        cls._solver.reset(
            state_out=cls._state_0,
            actuator_q=solver_kamino._actuators_q,
            actuator_u=solver_kamino._actuators_u,
            base_q=cls._base_q,
            base_u=cls._base_u,
            world_mask=world_mask,
        )

        # The Kamino FK reset only writes the *actuated* joint coordinates back into the joint
        # state; passive joint coordinates/velocities are left stale. Recompute the full joint
        # state from the reconciled body poses so reads of ``joint_pos`` / ``joint_vel`` reflect
        # the FK solution.
        # TODO: remove once Kamino FK writes the full joint state.
        joints = model_kamino.joints
        scratch = solver_kamino._data.joints
        wp.launch(
            _reset_joints_of_select_worlds,
            dim=model_kamino.size.sum_of_num_joints,
            inputs=[
                True,  # reset_constraints: only touches the scratch reaction buffer
                effective_world_mask,
                joints.wid,
                joints.dof_type,
                joints.num_dynamic_cts,
                joints.num_kinematic_cts,
                joints.coords_offset,
                joints.dofs_offset,
                joints.dynamic_cts_offset_joint_cts,
                joints.kinematic_cts_offset_joint_cts,
                joints.bid_B,
                joints.bid_F,
                joints.B_r_Bj,
                joints.F_r_Fj,
                joints.X_j,
                joints.q_j_0,
                cls._state_0.body_q,
                cls._state_0.body_qd,
                scratch.lambda_j,
            ],
            outputs=[
                scratch.p_j,
                scratch.r_j,
                scratch.dr_j,
                cls._state_0.joint_q,
                cls._state_0.joint_qd,
                scratch.lambda_j,
            ],
            device=model_kamino.device,
        )

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: KaminoSolverCfg) -> None:
        """Construct :class:`SolverKamino` and populate the base-class slots.

        Sets :attr:`NewtonManager._needs_collision_pipeline` to ``True`` only
        when ``use_collision_detector=False`` (Kamino's internal detector
        handles contacts otherwise).

        Configures the shared FK-reconcile flags for the Kamino (maximal-coordinate)
        solver.

        Also allocates the persistent per-world base pose/twist buffers (:attr:`_base_q` /
        :attr:`_base_u`) used by :meth:`_forward_kamino` to feed the FK reset with the
        per-environment root state.
        """
        NewtonManager._solver = SolverKamino(model, solver_cfg.to_solver_config())
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_collision_detector

        # Because Kamino treats body states as the simulation state. Environments that
        # have been reset (via joint states) therefore must be reconciled before each step.
        NewtonManager._reconcile_fk_before_step = True
        # For Kamino we should not run FK for worlds that were not reset.
        # This would modify the body states of the worlds that were not reset.
        NewtonManager._forward_full_reconcile = False

        model_kamino = NewtonManager._solver._model_kamino
        num_worlds = model_kamino.size.num_worlds
        cls._base_q = wp.zeros(num_worlds, dtype=wp.transformf, device=model_kamino.device)
        cls._base_u = wp.zeros(num_worlds, dtype=vec6f, device=model_kamino.device)
