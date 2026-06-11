# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging

import warp as wp
from newton import JointType, Model, eval_fk
from newton.solvers import SolverKamino

from isaaclab.physics import PhysicsManager
from isaaclab.utils.timer import Timer

from .kamino_manager_cfg import KaminoSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Kamino solver.

    Uses Newton's :class:`CollisionPipeline` unless
    :attr:`KaminoSolverCfg.use_collision_detector` is ``True``, in which case
    Kamino's internal collision detector handles contact generation.
    """

    @classmethod
    def _forward_kamino(cls, world_mask: wp.array | None = None) -> None:
        """Kamino-specific forward kinematics via ``solver.reset()``.

        Kamino's ``joint_q`` / ``joint_u`` include coordinates for **all** joints
        (including free joints), so we pass Newton's full state arrays directly.

        For floating-base models the first joint per world is a free joint and
        ``base_q`` / ``base_u`` are reconstructed from the first 7 coords / 6 dofs
        of ``joint_q`` / ``joint_qd``. Fixed-base models (e.g. Cartpole) omit
        ``base_q`` / ``base_u`` and reset through ``joint_q`` / ``joint_u`` only.
        After the solver reset, a full ``eval_fk`` (mask ``None``) overwrites
        ``body_q`` for a consistent frame convention.

        Consumes the pending reset state: zeroes the world / FK reset masks and clears
        :attr:`_reset_pending` so that callers (:meth:`forward`, :meth:`step`) do not
        redo the work.

        Args:
            world_mask: Per-world mask indicating which worlds to reset.
                Shape ``(num_worlds,)``, dtype ``wp.bool``. If None, resets all worlds.
        """
        _model = cls._model
        _nw = max(int(getattr(_model, "world_count", 0) or 0), 1)

        _coord_count = int(getattr(_model, "joint_coord_count", _model.joint_count))
        _dof_count = int(getattr(_model, "joint_dof_count", _model.joint_count))
        _coords_per_world = _coord_count // _nw
        _dofs_per_world = _dof_count // _nw
        _jq = wp.to_torch(cls._state_0.joint_q).reshape(_nw, _coords_per_world)
        _ju = wp.to_torch(cls._state_0.joint_qd).reshape(_nw, _dofs_per_world)
        _has_free_base = _coords_per_world >= 7 and int(_model.joint_type.numpy()[0]) == int(JointType.FREE)
        if _has_free_base:
            _base_q = wp.from_torch(_jq[:, :7].contiguous(), dtype=wp.transformf)
            _base_u = wp.from_torch(_ju[:, :6].contiguous(), dtype=wp.spatial_vectorf)
        else:
            _base_q = None
            _base_u = None

        cls._solver.reset(
            state=cls._state_0,
            world_mask=world_mask,
            joint_q=cls._state_0.joint_q,
            joint_u=cls._state_0.joint_qd,
            base_q=_base_q,
            base_u=_base_u,
        )

        # Overwrite body_q via Newton's full eval_fk for a consistent frame convention.
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)
        cls._update_sensors(None)
        NewtonManager._world_reset_mask.zero_()
        NewtonManager._fk_reset_mask.zero_()
        NewtonManager._reset_pending = False

    @classmethod
    def forward(cls) -> None:
        """Update kinematics without stepping physics.

        Runs the Kamino solver reset over the accumulated world mask so ``body_q`` is
        consistent with the written ``joint_q`` before observations / rendering read it
        (the explicit-reset path ``env.reset()`` -> ``sim.forward()``, and the lazy
        read path via ``ArticulationData._ensure_fk_fresh``).

        Gated on :attr:`_reset_pending`: with no pending reset the solver reset and FK
        would be a no-op, so the launch is skipped entirely. :meth:`_forward_kamino`
        consumes the masks and clears the flag so a following :meth:`step` does not
        redo the work.
        """
        if not cls._reset_pending:
            return
        cls._forward_kamino(world_mask=cls._world_reset_mask)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        if cls._reset_pending:
            cls._forward_kamino(world_mask=cls._world_reset_mask)

        # Notify solver of model changes
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._solver.notify_model_changed(change)
                NewtonManager._model_changes = set()

        # Lazy CUDA graph capture: deferred from initialize_solver() when RTX was active.
        # By the time step() is first called, RTX has fully initialized (all cudaImportExternalMemory
        # calls are done) and is idle between render frames — giving us a clean capture window.
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        if cls._graph_capture_pending and cfg is not None and cfg.use_cuda_graph and "cuda" in device:  # type: ignore[union-attr]
            NewtonManager._graph_capture_pending = False
            NewtonManager._graph = cls._capture_relaxed_graph(device)
            if cls._graph is not None:
                # Kamino: StateKamino.from_newton() lazily allocates body_f_total,
                # joint_q_prev, and joint_lambdas via wp.clone/wp.zeros during the
                # first step() inside graph capture. Replay once to pin those
                # memory-pool addresses before any eager solver.reset() call.
                wp.capture_launch(cls._graph)
                logger.info("Newton CUDA graph captured (deferred relaxed mode, RTX-compatible)")
            else:
                logger.warning("Newton deferred CUDA graph capture failed; using eager execution")

        # Step simulation (graphed or not; _graph is None when capture is disabled or failed)
        if cfg is not None and cfg.use_cuda_graph and cls._graph is not None and "cuda" in device:  # type: ignore[union-attr]
            wp.capture_launch(cls._graph)
        else:
            with wp.ScopedDevice(device):
                cls._simulate_physics_only()
        if cls._usdrt_stage is not None:
            cls._mark_transforms_dirty()

        # Launch solver-specific debug logging after stepping.
        cls._log_solver_debug()

        PhysicsManager._sim_time += cls._solver_dt * cls._num_substeps

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: KaminoSolverCfg) -> None:
        """Construct :class:`SolverKamino` and populate the base-class slots.

        Sets :attr:`NewtonManager._needs_collision_pipeline` to ``True`` only
        when ``use_collision_detector=False`` (Kamino's internal detector
        handles contacts otherwise).

        Applies :attr:`KaminoSolverCfg.max_contacts_per_world`, when set, by overriding
        ``model.rigid_contact_max`` before solver construction. This bounds GPU memory
        for contact-rich multi-env training that would otherwise over-allocate from
        ``geoms.world_minimum_contacts``.
        """
        if solver_cfg.max_contacts_per_world is not None:
            model.rigid_contact_max = int(solver_cfg.max_contacts_per_world) * model.world_count
            logger.info(
                "[KAMINO] Capping rigid_contact_max to %d (%d/world * %d worlds)",
                model.rigid_contact_max,
                solver_cfg.max_contacts_per_world,
                model.world_count,
            )
        NewtonManager._solver = SolverKamino(model, solver_cfg.to_solver_config())
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_collision_detector
        NewtonManager._reset_passive_joints = True

    @classmethod
    def _capture_or_defer_cuda_graph(cls) -> None:
        """Capture the physics CUDA graph, or defer if RTX is initializing."""
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        use_cuda_graph = cfg is not None and cfg.use_cuda_graph and "cuda" in device  # type: ignore[union-attr]

        with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
            if not use_cuda_graph:
                NewtonManager._graph = None
                return
            if cls._usdrt_stage is None:
                # No RTX active — use standard Warp capture (cudaStreamCaptureModeGlobal).
                with wp.ScopedCapture() as capture:
                    cls._simulate_physics_only()
                NewtonManager._graph = capture.graph
                logger.info("Newton CUDA graph captured (standard Warp mode)")

                # TODO: streamline this with base NewtonManager
                # Kamino: StateKamino.from_newton() lazily allocates body_f_total,
                # joint_q_prev, and joint_lambdas via wp.clone/wp.zeros during the
                # first step() inside graph capture. Replay once to pin those
                # memory-pool addresses before any eager solver.reset() call.
                wp.capture_launch(cls._graph)
            else:
                # RTX is active during initialization — cudaImportExternalMemory and other
                # non-capturable RTX ops run on background CUDA streams right now.
                # Defer capture to the first step() call, after RTX is fully initialized
                # and idle between render frames (clean capture window).
                NewtonManager._graph = None
                NewtonManager._graph_capture_pending = True
                logger.info("Newton CUDA graph capture deferred until first step() (RTX active)")
