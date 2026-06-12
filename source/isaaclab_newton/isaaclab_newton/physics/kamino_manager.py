# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging

import warp as wp
from newton import Model, eval_fk
from newton.solvers import SolverKamino

from isaaclab.physics import PhysicsManager

from .kamino_manager_cfg import KaminoSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


<<<<<<< HEAD
def _model_has_loop_closing_joints(model: Model) -> bool:
    """Return whether ``model`` contains converted loop-closing articulation joints.

    Newton stores regular tree joints in ``[articulation_start[i], articulation_end[i])`` and
    loop-closing joints in ``[articulation_end[i], articulation_start[i + 1])``. Loop closures
    are present when the next articulation sentinel exceeds the tree joint end for any
    articulation.

    Args:
        model: Finalized Newton model to inspect.

    Returns:
        ``True`` if at least one articulation has loop-closing joints.
    """
    articulation_start = model.articulation_start
    articulation_end = model.articulation_end
    if articulation_start is None or articulation_end is None:
        return False
    articulation_start_np = articulation_start.numpy()
    articulation_end_np = articulation_end.numpy()
    if articulation_end_np.shape[0] == 0:
        return False
    return bool((articulation_start_np[1:] > articulation_end_np).any())


=======
>>>>>>> 9cab5c87665 (Rebase MJWarp reset fix on latest Newton)
class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Kamino solver.

    Uses Newton's :class:`CollisionPipeline` unless
    :attr:`KaminoSolverCfg.use_collision_detector` is ``True``, in which case
    Kamino's internal collision detector handles contact generation.
    """

<<<<<<< HEAD
    # Annotate the concrete solver type.
    _solver: SolverKamino

=======
>>>>>>> 9cab5c87665 (Rebase MJWarp reset fix on latest Newton)
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
    def _eval_fk_impl(cls, world_reset_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body states from joint coordinates.

<<<<<<< HEAD
        For the Kamino (maximal-coordinate) solver, body poses/velocities are the authoritative
        simulation state. When :attr:`KaminoSolverCfg.use_fk_solver` is enabled, this calls
        :meth:`SolverKamino.reset`, which runs Kamino's loop-closure forward kinematics: it reads
        body poses/velocities from the joint coordinates (including the base body's pose/twist)
        and writes back a consistent full joint and body state.

        When ``use_fk_solver`` is disabled, falls back to Newton's articulated ``eval_fk`` over
        ``fk_mask``; the caller is then responsible for writing constraint-consistent joint values.

=======
>>>>>>> 9cab5c87665 (Rebase MJWarp reset fix on latest Newton)
        Args:
            world_reset_mask: Per-world mask passed to :meth:`SolverKamino.reset` (``None`` means all).
            fk_mask: Per-articulation mask of articulations to update (``None`` means all).
        """
<<<<<<< HEAD
        if cls._get_kamino_solver_cfg().use_fk_solver:
            cls._solver.reset(
                cls._state_0,
                world_mask=world_reset_mask,
                config=SolverKamino.ResetConfig.from_joints(),
            )
=======
        cls._solver.reset(
            cls._state_0,
            joint_q=cls._state_0.joint_q,
            joint_u=cls._state_0.joint_qd,
            world_mask=world_mask,
        )

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Kamino: run solver.reset() with the accumulated world mask to reinitialise
        # internal state (warm-start containers, constraint multipliers) for reset worlds.
        # Note: runs every step. solver.reset() with an all-False world_mask is a no-op
        # (kernels check mask per-world and skip). The cost of a no-op launch is negligible
        # compared to the complexity of maintaining a separate boolean guard.
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
>>>>>>> 9cab5c87665 (Rebase MJWarp reset fix on latest Newton)
        else:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, fk_mask)

            # Reset solver internals without performing Kamino's FK.
            cls._solver.reset(
                cls._state_0,
                world_mask=world_reset_mask,
                config=SolverKamino.ResetConfig.preserve(),
            )

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: KaminoSolverCfg) -> None:
        """Construct :class:`SolverKamino` and populate the base-class slots.

        Sets :attr:`NewtonManager._needs_collision_pipeline` to ``True`` only
        when ``use_collision_detector=False`` (Kamino's internal detector
        handles contacts otherwise).
<<<<<<< HEAD

        Sets :attr:`NewtonManager._needs_fk_before_step` because Kamino treats body state as
        authoritative: reset worlds (written via joint coordinates) must be reconciled before each
        step. The shared :attr:`NewtonManager._world_reset_mask` and
        :attr:`NewtonManager._fk_reset_mask` restrict that pre-step reconcile and
        :meth:`NewtonManager.forward` to reset worlds, so non-reset worlds keep their live,
        authoritative body state through Kamino's :meth:`_eval_fk_impl` overwrite.

        Raises:
            RuntimeError: If the model has more than one articulation per environment. The Kamino
                interface in IsaacLab currently only supports one articulation per environment.
=======
>>>>>>> 9cab5c87665 (Rebase MJWarp reset fix on latest Newton)
        """

        # Set the max contacts per world if specified.
        if solver_cfg.max_contacts_per_world is not None:
            model.rigid_contact_max = int(solver_cfg.max_contacts_per_world) * model.world_count
            logger.info(
                "[KAMINO] Capping rigid_contact_max to %d (%d/world * %d worlds)",
                model.rigid_contact_max,
                solver_cfg.max_contacts_per_world,
                model.world_count,
            )

        # Set the use_fk_solver flag based on the model's articulation structure if not specified by user.
        if solver_cfg.use_fk_solver is None:
            solver_cfg.use_fk_solver = _model_has_loop_closing_joints(model)

        if solver_cfg.use_fk_solver and model.articulation_count != model.world_count:
            raise RuntimeError(
                "The Kamino FK solver requires exactly one articulation per environment, but the model"
                f" has {model.articulation_count} articulations across {model.world_count} environments."
                " Multiple articulations per environment are not yet supported in Kamino's FK solver."
            )

        NewtonManager._solver = SolverKamino(model, solver_cfg.to_solver_config())
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_collision_detector
<<<<<<< HEAD
        NewtonManager._needs_fk_before_step = True
=======

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
>>>>>>> 9cab5c87665 (Rebase MJWarp reset fix on latest Newton)
