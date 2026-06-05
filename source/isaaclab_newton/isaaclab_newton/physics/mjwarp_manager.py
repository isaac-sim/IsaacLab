# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import inspect
import logging

import numpy as np
import warp as wp
from newton import Contacts, Model
from newton.solvers import SolverMuJoCo

from isaaclab.physics import PhysicsManager

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` specialization for the MuJoCo Warp solver.

    Owns construction of :class:`SolverMuJoCo`, contact-buffer allocation in
    both internal-MuJoCo and Newton-pipeline contact modes, and the debug
    convergence logging emitted from :meth:`_log_solver_debug` when
    :attr:`NewtonCfg.debug_mode` is enabled.
    """

    # Persistent bool mask co-located with MJWarp data, populated from
    # :attr:`NewtonManager._world_reset_mask` when the two arrays live on
    # different devices.  :func:`mujoco_warp.reset_data` launches on Warp's
    # current device when no explicit device is passed, so
    # :meth:`_reset_solver_internals` scopes the call to the MJWarp data
    # device and mirrors the mask there when needed.  Lazily allocated in
    # :meth:`_reset_solver_internals` and released via
    # :meth:`_solver_specific_clear`.
    _mjwarp_reset_mask: wp.array | None = None

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct :class:`SolverMuJoCo` and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so
        non-constructor metadata (``solver_type``, ``class_type``) and the
        ignored deprecated ``ls_parallel`` field are not forwarded. Sets
        :attr:`NewtonManager._needs_collision_pipeline` to
        ``True`` only when ``use_mujoco_contacts=False``.
        """
        ignored = {"class_type", "solver_type", "ls_parallel"}
        valid = set(inspect.signature(SolverMuJoCo.__init__).parameters) - {"self", "model"} - ignored
        kwargs = {k: v for k, v in solver_cfg.to_dict().items() if k in valid}
        NewtonManager._solver = SolverMuJoCo(model, **kwargs)
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_mujoco_contacts

        cfg = PhysicsManager._cfg
        # Cross-config validation that needs both halves.
        if solver_cfg.use_mujoco_contacts and cfg.collision_cfg is not None:
            raise ValueError(
                "NewtonCfg: collision_cfg cannot be set when "
                "solver_cfg.use_mujoco_contacts=True. Either set "
                "use_mujoco_contacts=False or remove collision_cfg."
            )

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Allocate contact buffers.

        Delegates to the base implementation when Newton's
        :class:`CollisionPipeline` is active.  When ``use_mujoco_contacts=True``
        the solver runs MuJoCo's internal collision detection, so this method
        instead pre-allocates a :class:`Contacts` buffer sized to the solver's
        maximum contact count; ``solver.update_contacts`` later populates it
        from MuJoCo data for contact-sensor reporting.
        """
        if cls._needs_collision_pipeline:
            super()._initialize_contacts()
            return
        if cls._solver is not None:
            NewtonManager._contacts = Contacts(
                rigid_contact_max=cls._solver.get_max_contact_count(),
                soft_contact_max=0,
                device=PhysicsManager._device,
                requested_attributes=cls._model.get_requested_contact_attributes(),
            )

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array) -> None:
        """Clear MuJoCo Warp solver-internal state for flagged worlds.

        Calls :func:`mujoco_warp.reset_data` with the accumulated per-world
        reset bitmask to zero ``qacc_warmstart``, ``qfrc_applied``,
        ``xfrc_applied``, ``qacc``, contact arrays, energy, and solver
        counters for worlds that an env reset flagged since the previous
        step.  Without this, a NaN produced in one solve persists across
        :meth:`isaaclab.envs.ManagerBasedEnv.reset` because the next solver
        substep warm-starts from the NaN — the world is then permanently
        dead.  See https://github.com/newton-physics/newton/issues/1266 and
        https://github.com/google-deepmind/mujoco_warp/discussions/1112.

        ``reset_data`` also overwrites ``mjw_data.qpos``/``qvel`` with the
        model defaults, but :class:`SolverMuJoCo` re-syncs them from
        ``state.joint_q``/``joint_qd`` on the very next step (its
        ``update_data_interval`` defaults to ``1``), so IsaacLab's reset
        pose still wins.  Until ``SolverMuJoCo.reset()`` lands upstream
        (newton#2657), reaching into ``solver.mjw_model``/``solver.mjw_data``
        directly is the documented workaround.

        No-ops when ``world_mask`` is ``None``: :func:`mujoco_warp.reset_data`
        with ``reset=None`` short-circuits its per-world guard
        (``wp.static(reset is not None)``) and resets **every** world to
        model defaults, which would silently nuke all sims after any
        unusual call sequence (e.g. invocation between
        :meth:`~isaaclab_newton.physics.NewtonManager.clear` and the next
        :meth:`~isaaclab_newton.physics.NewtonManager.start_simulation`).

        When :attr:`NewtonManager._world_reset_mask` and the MJWarp data live
        on different devices, the mask is copied into
        :attr:`_mjwarp_reset_mask` co-located with the data.  The
        ``reset_data`` call runs under ``wp.ScopedDevice`` so Warp
        launches on the data device instead of the current default device.

        Args:
            world_mask: Per-world bool mask of shape ``(world_count,)``;
                ``True`` for worlds that need their MJWarp internals cleared.
                ``None`` is treated as a no-op.
        """
        if world_mask is None:
            return
        # Deferred so the docs build, which does not install ``mujoco_warp``,
        # can still autodoc :class:`NewtonMJWarpManager` from this module.
        import mujoco_warp

        target_device = cls._solver.mjw_data.xfrc_applied.device
        if world_mask.device != target_device:
            mirror = cls._mjwarp_reset_mask
            if mirror is None or mirror.shape != world_mask.shape or mirror.device != target_device:
                cls._mjwarp_reset_mask = wp.zeros(world_mask.shape[0], dtype=wp.bool, device=target_device)
                mirror = cls._mjwarp_reset_mask
            wp.copy(mirror, world_mask)
            world_mask = mirror
        with wp.ScopedDevice(target_device):
            mujoco_warp.reset_data(cls._solver.mjw_model, cls._solver.mjw_data, reset=world_mask)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Release the on-mjw-device reset-mask mirror."""
        cls._mjwarp_reset_mask = None

    @classmethod
    def _log_solver_debug(cls) -> None:
        """Optionally log MuJoCo solver convergence at the end of step."""
        cfg = PhysicsManager._cfg
        if cfg is not None and cfg.debug_mode:  # type: ignore[union-attr]
            data = cls._get_solver_convergence_steps()
            logger.info(f"Solver convergence data: {data}")
            if data["max"] == cls._solver.mjw_model.opt.iterations:
                logger.warning(f"Solver didn't converge! max_iter={data['max']}")

    @classmethod
    def _get_solver_convergence_steps(cls) -> dict[str, float | int]:
        """Return MuJoCo Warp solver convergence statistics.

        Reads ``mjw_data.solver_niter`` (only available on
        :class:`SolverMuJoCo`) and summarizes per-environment iteration counts.
        """
        niter = cls._solver.mjw_data.solver_niter.numpy()
        return {
            "max": np.max(niter),
            "mean": np.mean(niter),
            "min": np.min(niter),
            "std": np.std(niter),
        }
