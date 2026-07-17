Newton Manager Abstraction
==========================

Newton exposes multiple solver families, and Isaac Lab keeps that flexibility by
making each solver an implementation detail of a small
:class:`~isaaclab_newton.physics.NewtonManager` subclass. The simulation context
still sees a normal physics manager; the solver configuration decides which
manager class is used.

For most new Newton solvers, the integration surface is intentionally small:

* define a solver config that inherits from
  :class:`~isaaclab_newton.physics.NewtonSolverCfg`;
* point the config's ``class_type`` at a manager subclass;
* implement ``_build_solver()`` in that manager;
* set the three base-manager slots: ``_solver``, ``_use_single_state``, and
  ``_needs_collision_pipeline``.

The existing MuJoCo Warp, XPBD, Featherstone, and Kamino managers are examples
of this pattern.


Adding a Solver Manager
-----------------------

The solver config carries both user-tunable solver parameters and the manager
dispatch target:

.. code-block:: python

    from isaaclab_newton.physics import NewtonManager, NewtonSolverCfg
    from isaaclab.utils.configclass import configclass


    @configclass
    class MySolverCfg(NewtonSolverCfg):
        class_type: type[NewtonManager] | str = "{DIR}.my_solver_manager:NewtonMySolverManager"
        solver_type: str = "my_solver"
        iterations: int = 16


``NewtonCfg`` copies ``solver_cfg.class_type`` into its own ``class_type`` in
``__post_init__``. User code keeps the normal shape:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import NewtonCfg

    sim_cfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MySolverCfg(iterations=32),
            num_substeps=2,
        )
    )


The manager then owns solver construction:

.. code-block:: python

    from newton import Model
    from newton.solvers import SolverMySolver

    from isaaclab_newton.physics import NewtonManager


    class NewtonMySolverManager(NewtonManager):
        @classmethod
        def _build_solver(cls, model: Model, solver_cfg: MySolverCfg) -> None:
            NewtonManager._solver = SolverMySolver(model, iterations=solver_cfg.iterations)
            NewtonManager._use_single_state = False
            NewtonManager._needs_collision_pipeline = True


``_use_single_state`` tells the base manager whether the solver advances in
place or swaps input/output states. ``_needs_collision_pipeline`` tells the base
manager whether to allocate and pass Newton collision-pipeline contacts to the
solver. A solver with its own internal contact detector can set it to ``False``.

Optional Overrides
------------------

Most managers only implement ``_build_solver()``. Override more only when the
solver actually needs it:

* ``_initialize_contacts()``: allocate custom contact buffers or support an
  internal contact detector.
* ``_step_solver(state_0, state_1, control, contacts, substep_dt)``: change
  one substep of solver execution while keeping the base simulation loop.
* ``_simulate_physics_only()``: add per-step work around the base substep loop,
  such as rebuilding a BVH.
* ``_reset_solver_internals(world_mask)``: clear solver-owned state before
  ``forward()`` consumes reset masks.
* ``step()``: perform solver-specific model-change notification or other
  pre-step work before delegating to the base manager.
* ``start_simulation()`` or ``instantiate_builder_from_stage()``: customize model
  building or post-finalize setup.
* ``_register_builder_attributes(builder)``: register solver-specific Newton
  custom attributes (particle, shape, body) on the builder before particles or
  finalize run. The active manager class invokes this hook from
  ``create_builder()``, ``start_simulation()``, and
  ``instantiate_builder_from_stage()``.
  :class:`~isaaclab_newton.physics.NewtonMPMManager` is the in-tree example —
  it registers ``mpm:young_modulus`` and the rest of the implicit MPM
  particle attributes.
* ``_prepare_builder_for_finalize(builder)``: normalize imported or replicated
  builder data right before ``ModelBuilder.finalize()``.
  :class:`~isaaclab_newton.physics.NewtonMPMManager` uses this to clear mass and
  inertia on kinematic bodies so implicit MPM treats them as massless colliders.
* ``_supports_cuda_graph_capture()``: return ``False`` to opt the solver out of
  CUDA graph capture and fall back to eager execution. Defaults to ``True``;
  :class:`~isaaclab_newton.physics.NewtonMPMManager` returns ``True`` only for a
  fixed grid, since sparse/dense MPM grids reallocate as particles move.
* ``_solver_specific_clear()``: release any class-level state owned by the
  solver manager.

Keep the manager name prefixed with ``Newton`` and the solver config grouped
with the other Newton solver configs so autocomplete and backend discovery stay
predictable.


Proxy-Coupled Solvers
---------------------

The Newton proxy coupler partitions one model between named solvers and
exchanges selected rigid bodies through virtual proxies:

* :class:`~isaaclab_contrib.coupling.CouplerEntryCfg` assigns model ownership
  and a solver configuration to each named entry.
* :class:`~isaaclab_contrib.coupling.CouplerProxyMappingCfg` maps selected
  source bodies into a destination entry.
* :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` combines the entries and
  proxy mappings.

:class:`~isaaclab_contrib.coupling.NewtonCouplerManager` resolves ownership,
validates the model partition, and constructs the Newton ``SolverCoupledProxy``.
The component solvers remain unchanged.

.. warning::

   The pinned Newton proxy solver clears lagged feedback and proxy contact
   caches for every replicated world when any world is reset. Use synchronized
   whole-batch resets until mask-aware upstream reset support is pinned.

The Franka soft-body and cloth lifting tasks use MJWarp for the robot and VBD
for the deformable object through this proxy coupling:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run python scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

         ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

Use ``--task Isaac-Lift-Cloth-Franka`` for the cloth variant.

For an opt-in manual MJWarp and VBD example, import
:mod:`isaaclab_contrib.custom_coupling`. The import registers
``IsaacContrib-Lift-Soft-Franka-Custom-Coupling`` and demonstrates custom
substep ordering outside the generic coupler.


When to Add a Coupled Manager
-----------------------------

Add a coupled manager when one solver cannot own the whole model step by itself:

* rigid bodies should use one solver while particles or cloth use another;
* contact detection is shared, but each solver consumes the contacts
  differently;
* you need a custom force, impulse, or state exchange between solvers;
* the substep order is part of the algorithm.

Use a normal single-solver manager when all physics can be advanced by one
Newton solver. Use a coupled manager only for the small amount of glue that is
truly solver-specific.
