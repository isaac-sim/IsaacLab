.. _newton-extending-solvers:

Extending Newton Solvers
========================

To add a Newton solver or a custom coupled solver, implement a
:class:`~isaaclab_newton.physics.NewtonManager` subclass. The sections below explain
its hooks, resource ownership, and extension requirements.

If you only need to select and configure a shipped solver, use the user-facing
pages instead: :doc:`/source/concepts/backends_and_presets` for backend and
preset selection, the :ref:`solver tuning guides <solver-tuning>` for shipped solvers, and
:ref:`newton-coupled-solvers` for choosing a coupling approach.


When a Solver Manager Is Needed
-------------------------------

Each Newton solver is exposed as a
:class:`~isaaclab_newton.physics.NewtonManager` subclass. Write a new one when:

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

:class:`~isaaclab_newton.physics.NewtonManager` subclasses are instantiated by the
simulation context, which owns exactly one manager instance as
:attr:`~isaaclab.sim.SimulationContext.backend_manager`. The manager owns its
builder and the :class:`~isaaclab_newton.physics.NewtonBackend` finalized from it;
everything bound to the finalized model lives on the backend, and a hard reset
closes it and finalizes a new one.

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
     - Builder, site requests and replication outputs
       (:class:`~isaaclab_newton.physics.NewtonCloneRecord`), substep iteration,
       CUDA graph capture, step callbacks, and Fabric and USD synchronization.
   * - :class:`~isaaclab_newton.physics.NewtonBackend`
     - Finalized ``Model``, states, control, solver, collision pipeline and
       contacts, sensors, Newton actuators, and reset masks.
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
onto its own :attr:`~isaaclab_newton.physics.NewtonCfg.class_type`, so task
configuration never names the manager directly.

Managers can also run without a ``SimulationContext``. Give each its own populated builder; managers do not share
state:

.. code-block:: python

   from isaaclab_newton.physics import NewtonCfg, NewtonMJWarpManager

   cfg = NewtonCfg()
   left = NewtonMJWarpManager(builder_a, cfg, dt=0.01, device="cuda:0")
   right = NewtonMJWarpManager(builder_b, cfg, dt=0.01, device="cuda:0")
   left.reset()
   right.reset()
   left.step()  # right's model, state, and time remain unchanged


Lifecycle
---------

The public entry points run in this order. Subclass hooks are private; the base
class invokes them.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Public call
     - What happens
     - Subclass hooks
   * - :meth:`~isaaclab_newton.physics.NewtonManager.initialize`
     - Stores the simulation context and adopts its physics configuration.
     - none
   * - ``get_newton_builder(sim)``
     - Returns the manager's :attr:`~isaaclab_newton.physics.NewtonManager.builder`, created through its
       ``create_builder()`` factory and kept across hard resets. The cloner imports declared prototypes and
       composes worlds into it.
     - ``_register_builder_attributes()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.start_simulation`
     - Finalizes the builder into a new :class:`~isaaclab_newton.physics.NewtonBackend`, which allocates states,
       control, and reset masks.
     - ``_prepare_builder_for_finalize()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.initialize_solver`
     - Builds the solver, checks that it was assigned, and allocates contacts.
     - ``_build_solver()``, ``_initialize_contacts()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.reset`
     - A hard reset re-runs ``start_simulation()`` and ``initialize_solver()``
       against a re-finalized model; a soft reset reuses the existing model,
       solver, contacts, and graph.
     - ``_prepare_builder_for_finalize()``, ``_build_solver()``,
       ``_initialize_contacts()`` (hard reset only)
   * - :meth:`~isaaclab_newton.physics.NewtonManager.step`
     - Runs one actuator pass plus ``num_substeps`` solver substeps, then
       updates sensors.
     - ``_reset_solver_internals()``, ``_simulate_physics_only()``,
       ``_step_solver()``, ``_check_solver_status()``,
       ``_log_solver_debug()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.pre_render`
     - Refreshes kinematics, then writes body, cable, and particle state to
       Fabric for rendering.
     - ``_reset_solver_internals()`` (via ``forward()``)
   * - :meth:`~isaaclab_newton.physics.NewtonManager.close` and
       :meth:`~isaaclab_newton.physics.NewtonManager.clear`
     - Releases the solver, model, and all manager state.
     - ``_solver_specific_clear()``

``_build_solver()`` runs after the model is finalized, so it may size solver
resources from the real model. ``_register_builder_attributes()`` runs before
particles are added and before ``finalize()``, so it is the only place to
register Newton custom attributes.

``step()`` takes one of two paths, selected by
:meth:`~isaaclab_newton.physics.NewtonManager.handles_decimation`. When every
actuator is on the graph-safe Newton fast path, actuators and substeps run
together inside one graph and ``step()`` runs the whole decimation loop, so
``_step_solver()`` runs ``decimation x num_substeps`` times per call. Otherwise
actuators run eagerly, only the substeps are graphed through
``_simulate_physics_only()``, and the environment drives decimation by calling
``step()`` repeatedly. Both paths reach ``_step_solver()`` through the same
substep loop.

Graph capture happens at one of several points. ``initialize_solver()`` captures
unless the graph-safe path is active; in that case
:meth:`~isaaclab_newton.physics.NewtonManager.set_decimation` captures once the
decimation is known, and the RTX path defers capture to the first ``step()``.
All routes check ``_supports_cuda_graph_capture()``. The non-RTX route
additionally checks ``_requires_initial_reset_before_graph_capture()``.

.. warning::

   With ``_use_single_state = False`` the base manager ping-pongs
   ``self.backend.state_0`` and ``self.backend.state_1`` after each substep,
   except on the final substep of an odd count, where it copies instead. Never
   cache a ``State`` reference in ``_build_solver()``; read the current state
   through the manager on each use.


Extension Contract
------------------

A subclass must implement ``_build_solver()`` and assign four slots on the
manager instance (``self``).
:meth:`~isaaclab_newton.physics.NewtonManager.initialize_solver` raises
``RuntimeError`` if ``_solver`` is still unset. The other three are not
validated and keep their defaults, so a subclass that forgets them runs with
double-buffered states, no collision pipeline, and no visualizer force input.

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

   from isaaclab.utils import configclass
   from isaaclab_newton.physics import NewtonManager, NewtonSolverCfg


   @configclass
   class MySolverCfg(NewtonSolverCfg):
       class_type: type[NewtonManager] | str = "{DIR}.my_solver_manager:NewtonMySolverManager"
       solver_type: str = "my_solver"
       iterations: int = 16


   class NewtonMySolverManager(NewtonManager):
       def _build_solver(self, model: Model, solver_cfg: MySolverCfg) -> None:
           self._solver = SolverMySolver(model, iterations=solver_cfg.iterations)
           self._use_single_state = False
           self._needs_collision_pipeline = True
           self._supports_rigid_body_force_input = True

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
* ``_prepare_builder_for_finalize()``: normalize imported or replicated builder
  data immediately before ``finalize()``.
* ``_supports_cuda_graph_capture()``: return ``False`` to fall back to eager
  execution.
* ``_requires_initial_reset_before_graph_capture()``: delay headless capture
  until the first post-reset step.
* ``_solver_specific_clear()``: release manager state the subclass owns.
* ``_check_solver_status()`` and ``_log_solver_debug()``: run after stepping.

:class:`~isaaclab_newton.physics.NewtonMPMManager` overrides both builder hooks.

Raise from ``_build_solver()`` on an unsupported configuration rather than
silently degrading. Name the manager ``Newton<Solver>Manager``.


Coupling Paths
--------------

The four architectures differ in what drives the substep loop:

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
       is the in-tree example. Its manager clears force accumulators, detects
       contacts once, injects soft-to-rigid reactions into ``body_f`` when
       ``coupling_mode="two_way"``, then advances MJWarp on its own internal
       contacts and VBD on the detected ones.

A custom shared-model manager bypasses entry ownership resolution, so it cannot
reuse the coupler's selectors or validation. For the proxy and ADMM trade-offs,
see :ref:`newton-coupled-solvers`.


.. _related-documentation:

Where to go next
----------------

* :doc:`add_physics_backend`: adding a whole physics backend.
* :doc:`/source/api/lab_newton/isaaclab_newton.physics`: manager and solver
  configuration API reference.
