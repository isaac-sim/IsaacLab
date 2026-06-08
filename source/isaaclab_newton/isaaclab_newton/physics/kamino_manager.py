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
        """Reset Kamino solver state to match the current joint configuration.

        Two reset paths are selected at solver-construction time via
        :attr:`KaminoSolverCfg.use_fk_solver`:

        * ``use_fk_solver=True`` (default): ``solver.reset(joint_q=..., joint_u=...,
          base_q=..., base_u=...)`` runs Kamino's Gauss-Newton FK so ``body_q`` is
          consistent with the written ``joint_q`` (needed for non-trivial joint reset
          targets on closed-loop assets).
        * ``use_fk_solver=False``: ``solver.reset(base_q=..., base_u=...)`` routes to
          ``_reset_to_base_state`` -- bodies are placed by transforming the model's
          reference body poses by ``base_q`` with **no FK solver**. This is the fast
          path for tasks that pin every joint coord to ``0`` (the assembled
          closed-loop-valid configuration) and sidesteps an FK-solver buffer overflow
          at high ``num_envs``. It is only correct when the written ``joint_q`` equals
          the model default (all zeros).

        Args:
            world_mask: Per-world mask indicating which worlds to reset.
                Shape ``(num_worlds,)``, dtype ``wp.int32``. If None, resets all worlds.
        """
        _model = cls._model
        _nw = max(int(getattr(_model, "world_count", 0) or 0), 1)
        _first_joint_is_free = _model.joint_count > 0 and int(_model.joint_type.numpy()[0]) == int(JointType.FREE)

        _base_q = None
        _base_u = None
        if _first_joint_is_free:
            _coord_count = int(getattr(_model, "joint_coord_count", _model.joint_count))
            _dof_count = int(getattr(_model, "joint_dof_count", _model.joint_count))
            if _nw > 0 and _coord_count % _nw == 0 and _dof_count % _nw == 0:
                _coords_per_world = _coord_count // _nw
                _dofs_per_world = _dof_count // _nw
                if _coords_per_world >= 7 and _dofs_per_world >= 6:
                    _jq = wp.to_torch(cls._state_0.joint_q).reshape(_nw, _coords_per_world)
                    _ju = wp.to_torch(cls._state_0.joint_qd).reshape(_nw, _dofs_per_world)
                    _base_q = wp.from_torch(_jq[:, :7].contiguous(), dtype=wp.transformf)
                    _base_u = wp.from_torch(_ju[:, :6].contiguous(), dtype=wp.spatial_vectorf)
        else:
            _body_count = int(getattr(_model, "body_count", 0) or 0)
            if _body_count > 0 and _nw > 0 and _body_count % _nw == 0:
                _bodies_per_world = _body_count // _nw
                if _model.joint_count > 0:
                    _joints_per_world = _model.joint_count // _nw if _nw > 0 else _model.joint_count
                    _joint_child_np = _model.joint_child.numpy()[: max(_joints_per_world, 0)]
                    _per_world_children = {int(c) for c in _joint_child_np if int(c) >= 0}
                else:
                    _per_world_children = set()
                if 0 not in _per_world_children:
                    _bq = wp.to_torch(cls._state_0.body_q).reshape(_nw, _bodies_per_world, 7)
                    _bu = wp.to_torch(cls._state_0.body_qd).reshape(_nw, _bodies_per_world, 6)
                    _base_q = wp.from_torch(_bq[:, 0, :].contiguous(), dtype=wp.transformf)
                    _base_u = wp.from_torch(_bu[:, 0, :].contiguous(), dtype=wp.spatial_vectorf)

        _use_fk = bool(getattr(getattr(cls._solver, "_config", None), "use_fk_solver", True))
        if _use_fk:
            cls._solver.reset(
                state=cls._state_0,
                world_mask=world_mask,
                joint_q=cls._state_0.joint_q,
                joint_u=cls._state_0.joint_qd,
                base_q=_base_q,
                base_u=_base_u,
            )
        elif _base_q is not None:
            # No-FK reset honoring the per-world target base pose: ``base_q`` routes Kamino to
            # ``_reset_to_base_state``, which rigidly transforms the assembled reference
            # configuration (valid at ``joint_q == 0``) so each world keeps its origin offset
            # and the closed-loop constraints stay satisfied. ``base_u`` (zero on reset) zeroes
            # all body velocities.
            cls._solver.reset(
                state=cls._state_0,
                world_mask=world_mask,
                base_q=_base_q,
                base_u=_base_u,
            )
        else:
            # Fallback when the base body could not be identified: restore the assembled
            # model-default state (consistent, but ignores the per-world target pose).
            cls._solver.reset(
                state=cls._state_0,
                world_mask=world_mask,
            )

        # Snap ``joint_q_prev`` to the reset joint coords for the reset worlds: the no-FK
        # reset leaves it stale, so the Moreau integrator would otherwise emit a spurious
        # ``(q_new - q_prev)/dt`` velocity on the first post-reset step. The mask keeps
        # other worlds' history intact.
        _jqp = getattr(cls._state_0, "joint_q_prev", None)
        if _jqp is not None and world_mask is not None:
            _jq_t = wp.to_torch(cls._state_0.joint_q)
            _jqp_t = wp.to_torch(_jqp)
            _mask = wp.to_torch(world_mask).bool()
            _nw_m = _mask.shape[0]
            if _nw_m > 0 and _jq_t.shape[0] % _nw_m == 0:
                _c = _jq_t.shape[0] // _nw_m
                _jqp_t.view(_nw_m, _c)[_mask] = _jq_t.view(_nw_m, _c)[_mask]

        # Overwrite body_q via Newton's eval_fk for a consistent frame convention. For
        # closed-loop assets the fk mask is zero-length, so this is a no-op and body poses
        # come from the base-state reset above.
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, cls._fk_reset_mask)

        # Sync state_1 to match state_0 so both states are consistent for the dual-state
        # stepping scheme.
        if cls._state_1 is not None and cls._state_1 is not cls._state_0:
            for attr in ("joint_q", "joint_qd", "body_q", "body_qd"):
                _src = getattr(cls._state_0, attr, None)
                _dst = getattr(cls._state_1, attr, None)
                if _src is not None and _dst is not None:
                    wp.copy(_dst, _src)
            for attr in ("joint_q_prev", "joint_lambdas"):
                _src = getattr(cls._state_0, attr, None)
                _dst = getattr(cls._state_1, attr, None)
                if _src is not None and (
                    _dst is None or _dst.shape != _src.shape or _dst.dtype != _src.dtype or _dst.device != _src.device
                ):
                    setattr(cls._state_1, attr, wp.clone(_src))
                elif _src is not None and _dst is not None:
                    wp.copy(_dst, _src)

    @classmethod
    def forward(cls) -> None:
        """Update kinematics without stepping physics.

        For Kamino, consume any pending reset flagged by :meth:`invalidate_fk` so the
        explicit-reset path (``env.reset()`` -> ``sim.forward()``) makes ``body_q``
        consistent with the reset ``joint_q`` before observations are read. Falls back
        to a full ``eval_fk`` when no reset is pending.
        """
        if cls._kamino_needs_reset:
            cls._forward_kamino(world_mask=cls._world_reset_mask)
            # Clear the flag on the BASE class: assigning through ``cls`` would shadow the
            # base attribute and make every later ``invalidate_fk`` go unobserved.
            NewtonManager._kamino_needs_reset = False
            if cls._world_reset_mask is not None:
                cls._world_reset_mask.zero_()
            if cls._fk_reset_mask is not None:
                cls._fk_reset_mask.zero_()
            return
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Run solver.reset() only when invalidate_fk() flagged a pending reset: it rewrites
        # joint_q_prev and re-snaps body_q, so it must not run on no-reset steps.
        if cls._kamino_needs_reset:
            cls._forward_kamino(world_mask=cls._world_reset_mask)
            # Clear on the BASE class (see ``forward`` for the shadowing rationale).
            NewtonManager._kamino_needs_reset = False

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

        # Ensure body_q is up-to-date before collision detection.
        # After env resets, joint_q is written but body_q (used by
        # broadphase/narrowphase) is stale until FK runs.
        # Only runs FK for dirtied articulations via the accumulated mask.
        if cls._needs_collision_pipeline:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, cls._fk_reset_mask)

        # Zero both masks after consumption
        NewtonManager._world_reset_mask.zero_()
        NewtonManager._fk_reset_mask.zero_()

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
