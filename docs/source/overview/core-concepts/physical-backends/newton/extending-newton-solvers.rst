.. _newton-extending-solvers:

Extending Newton Solvers
========================

This page is for contributors adding a Newton solver to Isaac Lab or building a
custom coupled solver. It describes the
:class:`~isaaclab_newton.physics.NewtonManager` extension contract: what the
manager owns, when its hooks run, and what a subclass must provide.

If you only need to select and configure a shipped solver, use the user-facing
pages instead: :doc:`/source/concepts/backends_and_presets` for backend and
preset selection, :doc:`index` for the per-solver guides, and
:ref:`newton-coupled-solvers` for choosing a coupling approach.


When a Solver Manager Is Needed
-------------------------------

Each Newton solver is exposed as a small
:class:`~isaaclab_newton.physics.NewtonManager` subclass. The simulation context
only sees a physics manager; the solver configuration decides which subclass is
active. Write a new subclass when:

* a Newton solver has no Isaac Lab manager yet;
* the solver needs its own contact allocation, builder attributes, or reset
  handling;
* several solvers must advance one shared model and the substep order is part
  of the algorithm.

Do not write one when an existing solver can advance the whole model, or when
the scene can be partitioned into named solver entries. Partitioning is already
covered by :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` and
:class:`~isaaclab_contrib.coupling.CouplerAdmmCfg`, which
:class:`~isaaclab_contrib.coupling.NewtonCouplerManager` resolves into entry
views over a shared model. Prefer that path for mixed rigid and deformable
scenes. Write a coupled manager only when contact detection is shared but each
solver consumes the contacts differently, or when the exchange between solvers
is a custom force, impulse, or state transfer.


Responsibilities and Boundaries
-------------------------------

:class:`~isaaclab_newton.physics.NewtonManager` is a class-level singleton: all
state lives on the base class and there are no instance methods, so exactly one
manager subclass is active per simulation.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Owner
     - Responsibility
   * - Simulation context
     - Resolves the manager subclass from
       :attr:`~isaaclab_newton.physics.NewtonCfg.class_type` and drives the
       public lifecycle calls.
   * - ``NewtonManager`` base
     - Builder, finalized ``Model``, states, control, collision pipeline and
       contacts, substep iteration, CUDA graph capture, Fabric and USD
       synchronization, sensors, and reset masks.
   * - Manager subclass
     - Solver construction and any solver-owned internal state, contact
       buffers, or builder attributes.
   * - Coupler entry
     - A disjoint part of the shared model, when the active manager is a
       coupler.
   * - Task configuration
     - A :class:`~isaaclab_newton.physics.NewtonSolverCfg` subclass whose
       ``class_type`` points at the manager, plus entry ownership selectors for
       coupled setups.

:class:`~isaaclab_newton.physics.NewtonCfg` copies ``solver_cfg.class_type``
onto its own :attr:`~isaaclab_newton.physics.NewtonCfg.class_type` during
post-initialization, so task configuration keeps the normal
``NewtonCfg(solver_cfg=...)`` shape and never names the manager directly.


Lifecycle
---------

The public entry points run in this order. Subclass hooks are private and are
invoked by the base class.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Public call
     - What happens
     - Subclass hooks
   * - :meth:`~isaaclab_newton.physics.NewtonManager.initialize`
     - Binds the simulation context, gravity, and the scene data backend.
     - none
   * - :meth:`~isaaclab_newton.physics.NewtonManager.create_builder`,
       :meth:`~isaaclab_newton.physics.NewtonManager.set_builder`, or
       :meth:`~isaaclab_newton.physics.NewtonManager.instantiate_builder_from_stage`
     - Creates or imports the ``ModelBuilder``.
     - ``_register_builder_attributes()``, except through ``set_builder()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.start_simulation`
     - Finalizes the model, then allocates states, reset masks, and Fabric
       prims.
     - ``_register_builder_attributes()``,
       ``_prepare_builder_for_finalize()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.initialize_solver`
     - Builds the solver, checks that it was assigned, and allocates contacts.
     - ``_build_solver()``, ``_initialize_contacts()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.reset`
     - A hard reset re-runs ``start_simulation()`` and ``initialize_solver()``
       against a re-finalized model; a soft reset reuses the existing model,
       solver, contacts, and graph.
     - the two rows above, on a hard reset only
   * - :meth:`~isaaclab_newton.physics.NewtonManager.step`
     - Runs one actuator pass plus ``num_substeps`` solver substeps, then
       updates sensors.
     - ``_reset_solver_internals()``, ``_simulate_physics_only()``,
       ``_step_solver()``, ``_check_solver_status()``,
       ``_log_solver_debug()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.pre_render`
     - Refreshes kinematics, then writes body, cable, and particle state to
       Fabric for rendering.
     - ``_reset_solver_internals()``, through ``forward()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.close` and
       :meth:`~isaaclab_newton.physics.NewtonManager.clear`
     - Releases the solver, model, and all class-level state.
     - ``_solver_specific_clear()``

``_build_solver()`` runs after the model is finalized, so it may size solver
resources from the real model. ``_register_builder_attributes()`` runs before
particles are added and before ``finalize()``, so it is the only place to
register Newton custom attributes.

``step()`` takes one of two paths, selected by
:meth:`~isaaclab_newton.physics.NewtonManager.handles_decimation`. When every
actuator is on the graph-safe Newton fast path, actuators and substeps run
together inside one graph and ``step()`` folds the whole decimation loop in, so
``_step_solver()`` runs ``decimation x num_substeps`` times per call. Otherwise
actuators run eagerly, only the substeps are graphed through
``_simulate_physics_only()``, and the environment drives decimation by calling
``step()`` repeatedly. Both paths reach ``_step_solver()`` through the same
substep loop.

CUDA graph capture is not tied to one call. ``initialize_solver()`` captures
unless the graph-safe path is active; in that case
:meth:`~isaaclab_newton.physics.NewtonManager.set_decimation` captures once the
decimation is known, and the RTX path defers capture to the first ``step()``.
Every route consults ``_supports_cuda_graph_capture()``; only the non-RTX route
also consults ``_requires_initial_reset_before_graph_capture()``.

.. warning::

   With ``_use_single_state = False`` the base manager ping-pongs
   ``NewtonManager._state_0`` and ``NewtonManager._state_1`` after each substep,
   except on the final substep of an odd count, where it copies instead. Never
   cache a ``State`` reference in ``_build_solver()``; read the current state
   through the class attribute on each use.


Extension Contract
------------------

A subclass must implement ``_build_solver()`` and assign four slots on
``NewtonManager`` itself, not on ``cls``, so external readers see the canonical
state regardless of which subclass is active.
:meth:`~isaaclab_newton.physics.NewtonManager.initialize_solver` raises
``RuntimeError`` if ``_solver`` is still unset. The other three slots are not
validated and silently keep their defaults, so a subclass that forgets them runs
with no collision pipeline and double-buffered states.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Slot
     - Meaning
   * - ``_solver``
     - The constructed Newton ``SolverBase`` instance.
   * - ``_use_single_state``
     - ``True`` if the solver steps in place on one ``State``; ``False`` if it
       needs separate input and output states.
   * - ``_needs_collision_pipeline``
     - ``True`` if the base manager should own Newton's collision pipeline;
       ``False`` if the solver detects contacts internally.
   * - ``_supports_rigid_body_force_input``
     - ``True`` if the solver consumes external rigid-body forces from
       ``State.body_f``.

.. code-block:: python

   from newton import Model
   from newton.solvers import SolverMySolver

   from isaaclab.utils.configclass import configclass
   from isaaclab_newton.physics import NewtonManager, NewtonSolverCfg


   @configclass
   class MySolverCfg(NewtonSolverCfg):
       class_type: type[NewtonManager] | str = "{DIR}.my_solver_manager:NewtonMySolverManager"
       solver_type: str = "my_solver"
       iterations: int = 16


   class NewtonMySolverManager(NewtonManager):
       @classmethod
       def _build_solver(cls, model: Model, solver_cfg: MySolverCfg) -> None:
           NewtonManager._solver = SolverMySolver(model, iterations=solver_cfg.iterations)
           NewtonManager._use_single_state = False
           NewtonManager._needs_collision_pipeline = True
           NewtonManager._supports_rigid_body_force_input = True

Override anything else only when the solver needs it:

* ``_create_solver()``: construct a solver without mutating manager state, so a
  coupler can nest this solver.
* ``_initialize_contacts()``: allocate custom contact buffers.
* ``_step_solver(state_0, state_1, control, contacts, substep_dt)``: change one
  substep while keeping the base simulation loop.
* ``_simulate_physics_only()``: add per-step work around the substep loop.
* ``_reset_solver_internals()``: clear solver-owned state for reset worlds.
* ``_register_builder_attributes()``: register Newton custom particle, shape, or
  body attributes on the builder.
  :class:`~isaaclab_newton.physics.NewtonMPMManager` registers the per-particle
  ``mpm:*`` material attributes here.
* ``_prepare_builder_for_finalize()``: normalize imported or replicated builder
  data immediately before ``finalize()``.
  :class:`~isaaclab_newton.physics.NewtonMPMManager` clears mass and inertia on
  kinematic bodies here so implicit MPM treats them as massless colliders.
* ``_supports_cuda_graph_capture()``: return ``False`` to fall back to eager
  execution.
* ``_requires_initial_reset_before_graph_capture()``: delay headless capture
  until the first post-reset step.
* ``_solver_specific_clear()``: release class-level state the subclass owns.
* ``_check_solver_status()`` and ``_log_solver_debug()``: report solver-specific
  failures and diagnostics.

Raise from ``_build_solver()`` on an unsupported configuration rather than
silently degrading;
:class:`~isaaclab_contrib.coupling.NewtonCouplerManager` follows this pattern
and rejects solver configurations it cannot nest.

Keep the manager name prefixed with ``Newton`` and group the solver
configuration with the other Newton solver configurations so autocomplete and
backend discovery stay predictable.


Coupling Paths
--------------

Four architectures are available, and they differ in who owns the substep:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Path
     - Structure
   * - Standalone solver
     - One manager subclass, one solver, one model. The base class owns the
       substep loop.
   * - Proxy coupling
     - :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` partitions the model
       into named entries.
       :class:`~isaaclab_contrib.coupling.NewtonCouplerManager` builds a Newton
       coupled solver that exposes source bodies to the destination solver as
       proxies and returns lagged feedback. No new manager is required.
   * - ADMM coupling
     - :class:`~isaaclab_contrib.coupling.CouplerAdmmCfg` uses the same manager
       and entry model, but Newton creates symmetric interface constraints and
       iterates the sub-solvers. No new manager is required.
   * - Custom shared-model manager
     - A subclass constructs several sub-solvers itself and overrides
       ``_step_solver()`` to fix the substep order.
       :class:`~isaaclab_contrib.custom_coupling.newton_manager_cfg.CoupledMJWarpVBDSolverCfg`
       is the in-tree example: it stores an
       :class:`~isaaclab_newton.physics.MJWarpSolverCfg` and a
       :class:`~isaaclab_newton.physics.VBDSolverCfg`, and its manager clears
       force accumulators, detects contacts once, injects soft-to-rigid
       reactions into ``body_f`` when ``coupling_mode="two_way"``, then advances
       MJWarp on its own internal contacts and VBD on the detected ones.

Use a custom shared-model manager only when the substep order itself is the
algorithm. It bypasses entry ownership resolution, so it cannot reuse the
coupler's selectors or validation. The coupler paths keep the base manager in
charge of state allocation, substep iteration, and synchronization; only the
coupling policy is configured.

For the trade-offs between proxy and ADMM, and for the configuration fields of
each, see :ref:`newton-coupled-solvers`.


Related Documentation
---------------------

* :ref:`newton-using-vbd`: VBD setup and contact tuning.
* :ref:`mjwarp-solver-tuning`: MJWarp tuning.
* :doc:`../../multi_backend_architecture`: adding a whole physics backend.
* :doc:`/source/api/lab_newton/isaaclab_newton.physics`: manager and solver
  configuration API reference.
