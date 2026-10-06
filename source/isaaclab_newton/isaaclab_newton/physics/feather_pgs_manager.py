# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FeatherPGS Newton manager."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import warp as wp
from newton import Contacts, Model, State
from newton.solvers import SolverFeatherPGS, SolverMuJoCo

from isaaclab.physics import PhysicsManager

from .feather_pgs_manager_cfg import FeatherPGSSolverCfg
from .newton_manager import NewtonManager


class NewtonFeatherPGSManager(NewtonManager):
    """:class:`NewtonManager` specialization for the FeatherPGS solver.

    FeatherPGS steps with separate input and output states and uses Newton's collision pipeline.
    """

    # MuJoCo attributes carry the joint properties authored for MuJoCo-based assets.
    _builder_attribute_solvers = (SolverFeatherPGS, SolverMuJoCo)

    @classmethod
    def create_fixed_tendon_control(cls, articulation):
        """Keep imported tendon metadata without tendon commands.

        FeatherPGS does not simulate tendons. Articulation tendon-target setters still reject commands.
        """
        return

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: FeatherPGSSolverCfg) -> SolverFeatherPGS:
        """Construct the configured FeatherPGS solver."""
        return SolverFeatherPGS(model, **cls._filter_solver_kwargs(SolverFeatherPGS, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: FeatherPGSSolverCfg) -> None:
        """Construct :class:`SolverFeatherPGS` and populate the base-class slots."""
        collision_cfg = cls._collision_cfg
        contact_capacity = None if collision_cfg is None else collision_cfg.resolve_rigid_contact_max(model.world_count)
        if contact_capacity is not None:
            # FeatherPGS sizes its contact scratch from the model when it is constructed.
            model.rigid_contact_max = contact_capacity
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = True

    @classmethod
    def _collision_pipeline_args(cls) -> dict[str, Any]:
        """Add the predictive-contact extension of the solver configuration to the collision pipeline."""
        pipeline_args = super()._collision_pipeline_args()
        pipeline_args["speculative_contact_gap_max"] = PhysicsManager._cfg.solver_cfg.speculative_contact_gap_max
        return pipeline_args

    @classmethod
    def _collide(cls, state: State, contacts: Contacts) -> None:
        """Generate contacts, predicting them over the full physics step when predictive contacts are enabled."""
        cls._collision_pipeline.collide(state, contacts, dt=cls._solver_dt * cls._num_substeps)

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Host-prepared contact torsion reads contacts back every step, so it runs without a CUDA graph."""
        solver_cfg = PhysicsManager._cfg.solver_cfg
        return solver_cfg.contact_torsion_radius == 0.0 or solver_cfg.contact_torsion_device

    @classmethod
    def _capture_graph(cls, capture_target: Callable[[], None]) -> wp.Graph:
        """Prepare device contact torsion for the captured states, then record the step."""
        cls._solver.prepare_contact_torsion_capture(cls.backend.state_0, cls.backend.state_1)
        return super()._capture_graph(capture_target)

    @classmethod
    def _check_solver_status(cls) -> None:
        """Raise contact-torsion errors and, when requested, dropped constraint rows or contacts."""
        cls._solver.validate_contact_torsion()
        if PhysicsManager._cfg.solver_cfg.raise_on_constraint_overflow:
            cls._solver.check_constraint_capacity()

    @classmethod
    def _simulate_full(cls) -> None:
        """Seed the solver's double-buffer events inside a CUDA graph capture, then step."""
        cls._seed_double_buffer_events_if_capturing()
        super()._simulate_full()

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Seed the solver's double-buffer events inside a CUDA graph capture, then step."""
        cls._seed_double_buffer_events_if_capturing()
        super()._simulate_physics_only()

    @classmethod
    def _seed_double_buffer_events_if_capturing(cls) -> None:
        device = PhysicsManager._device
        if device is not None and "cuda" in device and wp.get_stream(device).is_capturing:
            cls._solver.seed_double_buffer_events()
