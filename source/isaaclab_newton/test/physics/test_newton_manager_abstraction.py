# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the per-solver :class:`NewtonManager` abstraction.

Covers:

* :attr:`NewtonSolverCfg.class_type` resolves to the matching manager subclass.
* :meth:`NewtonCfg.__post_init__` propagates ``solver_cfg.class_type`` onto
  :attr:`NewtonCfg.class_type` so that ``SimulationContext`` picks the right
  manager.
* Each leaf manager subclasses :class:`NewtonManager` and implements
  :meth:`_build_solver` (with the abstract base raising ``NotImplementedError``).
* The cross-config validation in :meth:`NewtonMJWarpManager._build_solver`
  rejects the ``MJWarp + use_mujoco_contacts=True + collision_cfg`` combination.
* Manager name dispatch (used by :class:`InteractiveScene` and the various
  factory dispatchers) still starts with ``"newton"``.
* End-to-end: spinning up a simulation with each solver builds the correct
  solver, sets the right ``_use_single_state`` / ``_needs_collision_pipeline``
  flags, and lands canonical state on :class:`NewtonManager` so that external
  ``NewtonManager._foo`` reads keep working.
"""

from __future__ import annotations

import pytest
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    KaminoSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonFeatherstoneManager,
    NewtonKaminoManager,
    NewtonManager,
    NewtonMJWarpManager,
    NewtonSolverCfg,
    NewtonXPBDManager,
    XPBDSolverCfg,
)
from newton.solvers import SolverFeatherstone, SolverKamino, SolverMuJoCo, SolverXPBD

from isaaclab.sim import SimulationCfg, build_simulation_context

# ---------------------------------------------------------------------------
# Lightweight (no sim) parametrisation
# ---------------------------------------------------------------------------

# (solver_cfg_factory, expected_manager, expected_solver_cls,
#  expected_use_single_state, expected_needs_collision_pipeline)
SOLVER_MATRIX = [
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=True),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        False,
        id="mjwarp_internal_contacts",
    ),
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=False),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        True,
        id="mjwarp_newton_pipeline",
    ),
    pytest.param(
        lambda: XPBDSolverCfg(),
        NewtonXPBDManager,
        SolverXPBD,
        False,
        True,
        id="xpbd",
    ),
    pytest.param(
        lambda: FeatherstoneSolverCfg(),
        NewtonFeatherstoneManager,
        SolverFeatherstone,
        False,
        True,
        id="featherstone",
    ),
    pytest.param(
        lambda: KaminoSolverCfg(use_collision_detector=True),
        NewtonKaminoManager,
        SolverKamino,
        False,
        False,
        id="kamino_internal_contacts",
    ),
    pytest.param(
        lambda: KaminoSolverCfg(use_collision_detector=False),
        NewtonKaminoManager,
        SolverKamino,
        False,
        True,
        id="kamino_newton_pipeline",
    ),
]


# ---------------------------------------------------------------------------
# class_type wiring (no SimulationContext required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_solver_cfg_class_type_resolves_to_subclass(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """Each ``*SolverCfg.class_type`` resolves to its matching manager subclass."""
    solver_cfg = solver_cfg_factory()
    # ``class_type`` is a lazy ``"module:Class"`` reference; calling its
    # ``_resolve()`` returns the actual class. ``__name__`` works without
    # forcing import (LazyType caches metadata) and is sufficient identity.
    assert solver_cfg.class_type.__name__ == expected_manager.__name__


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_newton_cfg_post_init_propagates_class_type(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """``NewtonCfg.__post_init__`` lifts ``solver_cfg.class_type`` onto ``NewtonCfg.class_type``."""
    cfg = NewtonCfg(solver_cfg=solver_cfg_factory())
    assert cfg.class_type.__name__ == expected_manager.__name__


# ---------------------------------------------------------------------------
# Manager class hierarchy and factory contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manager", [NewtonMJWarpManager, NewtonXPBDManager, NewtonFeatherstoneManager, NewtonKaminoManager]
)
def test_subclass_of_newton_manager(manager):
    """All concrete managers inherit from :class:`NewtonManager`."""
    assert issubclass(manager, NewtonManager)
    # Subclasses must override the abstract factory.
    assert manager._build_solver is not NewtonManager._build_solver


def test_abstract_build_solver_raises():
    """Calling :meth:`_build_solver` on the abstract base raises."""
    with pytest.raises(NotImplementedError):
        NewtonManager._build_solver(model=None, solver_cfg=NewtonSolverCfg())


@pytest.mark.parametrize(
    "manager", [NewtonMJWarpManager, NewtonXPBDManager, NewtonFeatherstoneManager, NewtonKaminoManager]
)
def test_manager_name_starts_with_newton(manager):
    """The ``"newton"`` prefix is required by :class:`InteractiveScene` and the
    various backend factories that dispatch on ``physics_manager.__name__.lower()``.
    """
    assert manager.__name__.lower().startswith("newton")


# ---------------------------------------------------------------------------
# End-to-end: build each solver via SimulationContext
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, expected_solver_cls,"
    " expected_use_single_state, expected_needs_collision_pipeline",
    SOLVER_MATRIX,
)
def test_initialize_solver_populates_canonical_state(
    solver_cfg_factory,
    expected_manager,
    expected_solver_cls,
    expected_use_single_state,
    expected_needs_collision_pipeline,
):
    """End-to-end: ``SimulationContext`` resolves the right manager subclass and
    ``initialize_solver`` lands the right solver + flags on :class:`NewtonManager`.

    External code reads :class:`NewtonManager` attributes directly (``_solver``,
    ``_use_single_state``, ``_needs_collision_pipeline``).  Even though dispatch
    runs through a leaf subclass (e.g. :class:`NewtonMJWarpManager`), shared
    state is assigned through the explicit base class so that those reads keep
    working regardless of which leaf is active.  This test is the regression
    guard for that contract.

    The builder is pre-populated with a minimal one-body / one-joint scene
    (instead of relying on a USD stage) for two reasons:

    1. :class:`SolverMuJoCo` requires at least one joint to convert the model
       to MJCF; a ground-plane-only scene fails MJCF conversion.
    2. Pre-populating ``NewtonManager._builder`` causes
       :meth:`NewtonManager.start_simulation` to skip
       :meth:`instantiate_builder_from_stage`, so the test does not depend on
       USD asset packages.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg_factory(), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        # Resolved manager class matches the expected leaf.
        resolved_manager = sim.physics_manager
        # ``physics_manager`` is a LazyType proxy — compare by ``__name__`` to
        # avoid forcing identity-by-id checks against the unresolved proxy.
        assert resolved_manager.__name__ == expected_manager.__name__
        assert resolved_manager.__name__.lower().startswith("newton")

        # Pre-populate the builder with a minimal scene so MJCF conversion has
        # something to work with.
        builder = NewtonManager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)

        # Force resolution and bring up the solver.
        sim.reset()

        # Canonical state lives on the base class.
        assert NewtonManager._solver is not None
        assert isinstance(NewtonManager._solver, expected_solver_cls)
        assert NewtonManager._use_single_state is expected_use_single_state
        assert NewtonManager._needs_collision_pipeline is expected_needs_collision_pipeline

        # ``_contacts`` is allocated whichever way contacts are handled
        # (MuJoCo internal buffer or Newton pipeline output).
        # Kamino with internal contacts does not currently set NewtonManager._contacts.
        if expected_solver_cls is not SolverKamino:
            assert NewtonManager._contacts is not None

        # One step should not raise — proves the dispatch wiring lines up
        # end-to-end.  (We do not assert physics; that's covered by the
        # asset/sensor test suites.)
        sim.step(render=False)


def test_mjwarp_internal_contacts_with_collision_cfg_raises():
    """Combining ``use_mujoco_contacts=True`` with a ``collision_cfg`` is rejected.

    The check lives in :meth:`NewtonMJWarpManager._build_solver` because it
    needs both the solver cfg subtype and the parent :class:`NewtonCfg`, so it
    fires during :meth:`NewtonManager.initialize_solver` (i.e. on
    ``sim.reset()``) rather than at cfg construction time.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True),
            collision_cfg=NewtonCollisionPipelineCfg(),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = NewtonManager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)

        with pytest.raises(ValueError, match="collision_cfg cannot be set"):
            sim.reset()
