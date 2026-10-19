.. _newton-extending-solvers:

Extending Newton Solvers
========================

To add a Newton solver or a custom coupled solver, implement a
:class:`~isaaclab_newton.physics.NewtonSolver` adapter. The sections below explain
its hooks, resource ownership, and extension requirements.

If you only need to select and configure a shipped solver, use the user-facing
pages instead: :doc:`/source/concepts/backends_and_presets` for backend and
preset selection, the :ref:`solver tuning guides <solver-tuning>` for shipped solvers, and
:ref:`newton-coupled-solvers` for choosing a coupling approach.


When a Solver Adapter Is Needed
-------------------------------

Each Newton solver is exposed as a :class:`~isaaclab_newton.physics.NewtonSolver` subclass, such as
:class:`~isaaclab_newton.physics.MJWarpSolverAdapter`. An adapter instance holds its solver configuration and no
runtime state. Write a new one when:

* a Newton solver has no Isaac Lab adapter yet;
* the solver needs its own contact allocation, builder attributes, or reset
  handling;
* several solvers must advance one shared model and the substep order is part
  of the algorithm.

Do not write one when an existing solver can advance the whole model, or when
the scene can be partitioned into named solver entries. Partitioning is already
covered by :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` and
:class:`~isaaclab_contrib.coupling.CouplerAdmmCfg`, which
:class:`~isaaclab_contrib.coupling.CouplerSolverAdapter` resolves into entry
views over a shared model. Prefer that path for mixed rigid and deformable
scenes. Write a coupled solver adapter only when contact detection is shared but each
solver consumes the contacts differently, or when the exchange between solvers
is a custom force, impulse, or state transfer.


Responsibilities and Boundaries
-------------------------------

Data and behavior are split. :class:`~isaaclab_newton.physics.NewtonBackend` holds everything bound to one finalized
model, and the functions in :mod:`isaaclab_newton.physics.newton_backend` operate on it explicitly, in the style of
``mj_step(m, d)``. A solver adapter is an instance built from its solver configuration
(:func:`~isaaclab_newton.physics.newton_solver.create_solver_adapter`); its hooks take the backend or builder they act
on as an argument and keep no runtime state, so one adapter can serve several backends.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Owner
     - Responsibility
   * - Simulation context
     - Resolves the manager from :attr:`~isaaclab_newton.physics.NewtonCfg.class_type` (or adopts the one passed in)
       and drives the public lifecycle calls.
   * - :class:`~isaaclab_newton.physics.NewtonManager`
     - The :attr:`~isaaclab_newton.physics.NewtonManager.builder` and the active
       :attr:`~isaaclab_newton.physics.NewtonManager.backend`, which it finalizes and closes itself, plus what must
       survive a hard reset: site
       requests, the replication outputs (:class:`~isaaclab_newton.physics.NewtonCloneRecord`), and the decimation
       setting. Each instance owns its construction data, callbacks and views; closing it cannot clear another manager.
   * - :class:`~isaaclab_newton.physics.NewtonBackend`
     - Model, states, control, solver, collision pipeline and contacts, sensors, Newton actuators, step callbacks,
       reset masks, and the compiled :class:`~isaaclab_newton.physics.StepGraph`. A hard reset or close discards it.
   * - Solver adapter (``<Solver>SolverAdapter``)
     - Solver construction, capabilities, stepping, per-world reset of solver-owned buffers, forward kinematics, and
       builder attributes, as methods over an explicit backend or builder.
   * - Coupler entry
     - A disjoint part of the shared model, when the selected solver adapter is a coupler.
   * - Task configuration
     - A :class:`~isaaclab_newton.physics.NewtonSolverCfg` subclass whose ``class_type`` points at the solver adapter, plus
       entry ownership selectors for coupled setups.

``NewtonCfg.class_type`` selects the integration manager, while ``solver_cfg.class_type`` selects the solver
adapter. Neither field is derived from the other. Asset factories use the manager's ``backend_name`` key, so a
custom class name does not change backend selection.


Independent and Custom Managers
-------------------------------

Managers can run without a ``SimulationContext``. Give each manager its own populated builder:

.. code-block:: python

   from isaaclab_newton.physics import NewtonCfg, NewtonManager

   cfg = NewtonCfg()
   left = NewtonManager(builder_a, cfg, dt=0.01, device="cuda:0")
   right = NewtonManager(builder_b, cfg, dt=0.01, device="cuda:0")
   try:
       left.reset()
       right.reset()
       left.step()   # right's model, state, callbacks and time remain unchanged
       right.step()
   finally:
       left.close()
       right.close()

To customize Isaac Lab integration, subclass ``NewtonManager`` and select it with
``NewtonCfg(class_type=MyManager, solver_cfg=...)``, or pass an instance with
``SimulationContext(sim_cfg, backend_manager=my_manager)``; the context then owns its lifetime. Override
``finalize_backend`` to customize model construction, and call the base implementation from overridden lifecycle
methods. A custom solver only needs a ``NewtonSolver`` adapter, not a new manager.


Lifecycle
---------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Public call
     - What happens
     - Solver hooks
   * - ``get_newton_builder(sim)``
     - Returns the physics manager's builder, created through the selected solver adapter's ``create_builder()``. The cloner imports
       declared prototypes and composes worlds into it.
     - ``register_builder_attributes()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.reset` (hard)
     - Dispatches ``MODEL_INIT``, finalizes the model into a new backend, dispatches ``PHYSICS_READY`` so consumers
       bind views, sensors, actuators, and step callbacks, then constructs the solver and contacts and runs FK. Does
       not advance physics. A soft reset keeps the backend.
     - ``prepare_solver_builder()``, ``validate_cfg()``, ``create_solver()``, ``initialize_output_state()``,
       ``uses_collision_pipeline()``, ``create_collision_pipeline()``, ``create_contacts()``, ``prepare_contacts()``,
       ``eval_fk()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.step`
     - Runs :func:`~isaaclab_newton.physics.newton_backend.forward` for authored state, builds the step graph on the
       first step after a structural change, runs that step eagerly, and captures the graph for later steps.
     - ``reset_solver()``, ``eval_fk()``, ``prepare_step()``, ``step_solver()``, ``check_status()``, ``log_debug()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.close`
     - Closes the backend and clears the session state.
     - none

``register_builder_attributes()`` runs before particles are added and before ``finalize()``, so it is the only place
to register Newton custom attributes. The solver is constructed after ``PHYSICS_READY``, so it sees model properties
that consumers author while binding.


The Step
--------

Each :meth:`~isaaclab_newton.physics.NewtonManager.step` launches one :class:`~isaaclab_newton.physics.StepGraph`. It
covers the whole decimation loop when the manager handles decimation
(:meth:`~isaaclab_newton.physics.NewtonManager.handles_decimation`) and one physics step otherwise. Pre-step
callbacks run once first. Every physics step then runs ``collide -> control callbacks -> Newton actuators ->
substeps -> finish_step() -> post-step callbacks -> sensors``, where every substep runs ``staged forces -> state-force callbacks ->
step_solver()``, and post-actuator callbacks run after the actuators of the last physics step. Each list of
:class:`~isaaclab_newton.physics.StepCallbacks` has one cadence.

The graph binds every buffer when it is built. Double-buffered solvers alternate input and output states at build
time, and each physics step ends in ``state_0``, so ``manager.backend.state_0`` is always the same object.
Consecutive graphable operations are captured together; operations that cannot be captured, such as actuators with Python history indices or eager callbacks, run in place between captured segments. Consumers add operations with
:meth:`~isaaclab_newton.physics.NewtonManager.register_control_callback` and its siblings while ``PHYSICS_READY`` dispatches.

The backend functions read no manager state, so backends with different solvers can be built from their own models
and stepped side by side:

.. code-block:: python

   from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBackend, NewtonCfg, VBDSolverCfg
   from isaaclab_newton.physics import newton_backend as nb

   rigid = NewtonBackend(rigid_model, NewtonCfg(solver_cfg=MJWarpSolverCfg()), dt=0.01)
   cloth = NewtonBackend(cloth_model, NewtonCfg(solver_cfg=VBDSolverCfg()), dt=0.01)
   for backend in (rigid, cloth):
       nb.init_solver(backend)
       backend.steps_per_call = 2

   for _ in range(100):
       nb.step(rigid)
       nb.step(cloth)

:func:`~isaaclab_newton.physics.newton_backend.step` runs the first launch after the step graph is built eagerly,
then captures the graph for later launches when the backend's device and solver allow it.
:func:`~isaaclab_newton.physics.newton_backend.prepare` builds the step graph and allocates scratch without advancing
physics. :func:`~isaaclab_newton.physics.newton_backend.capture_graph` records Torch and Warp on one stream and keeps
both libraries' allocations alive; use it rather than a bare capture when Torch work is recorded.

Property writers call ``mark_model_changed(backend, flags)``; ``notify_model_changes`` commits the queued
categories once at the next state read or step.


Extension Contract
------------------

Subclass :class:`~isaaclab_newton.physics.NewtonSolver`, implement
:meth:`~isaaclab_newton.physics.NewtonSolver.create_solver`, and set the capability class attributes when they differ
from the defaults:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Attribute
     - Meaning
   * - ``single_state``
     - ``True`` if the solver steps in place on one ``State``.
   * - ``supports_deterministic``
     - ``True`` if the solver honors a Warp determinism guarantee.
   * - ``supports_contact_sensors``
     - ``False`` if Newton contact sensors cannot read the solver's contacts.
   * - ``prepares_step``
     - ``True`` if ``prepare_step()`` does work once per physics step.
   * - ``finishes_step``
     - ``True`` if ``finish_step()`` publishes derived state once per physics step.
   * - ``ignored_model_changes``
     - Model changes the solver does not apply after construction, mapped to a one-time warning.
   * - ``captures_model_changes``
     - ``True`` if ``notify_model_changed()`` can be recorded into a CUDA graph. Model changes then run in a
       conditional graph that skips the refresh when nothing changed; otherwise the solver is notified eagerly on
       the host.
   * - ``solver_class``
     - Newton solver the default ``create_solver()`` constructs from the matching configuration fields.
   * - ``builder_attribute_solvers``
     - Newton solvers whose custom builder attributes are registered before import.

.. code-block:: python

   import warp as wp
   from newton import Model
   from newton.solvers import SolverMySolver

   from isaaclab.utils import configclass
   from isaaclab_newton.physics import NewtonSolver, NewtonSolverCfg


   @configclass
   class MySolverCfg(NewtonSolverCfg):
       class_type: type[NewtonSolver] | str = "{DIR}.my_solver_manager:MySolverAdapter"
       solver_type: str = "my_solver"
       iterations: int = 16


   class MySolverAdapter(NewtonSolver):
       supports_deterministic = True

       def create_solver(self, model: Model, deterministic_mode=wp.DeterministicMode.NOT_GUARANTEED):
           return SolverMySolver(model, iterations=self.cfg.iterations)

Override anything else only when the solver needs it. The configuration is :attr:`~isaaclab_newton.physics.NewtonSolver.cfg`;
every other input is an argument, and hooks must not keep runtime state on the adapter:

* ``step_solver()``: change one substep, for example to run a projection after the solve.
* ``reset_solver()``: clear solver-owned history for masked worlds without touching authored joint state.
* ``eval_fk()``: use a solver-specific forward kinematics.
* ``prepare_step()``: refresh acceleration structures once per physics step; must stay graphable.
* ``finish_step()``: publish derived state, such as joint coordinates, once per physics step after the substeps.
* ``initialize_output_state()``: initialize the output state of double-buffered solvers before capture.
* ``uses_collision_pipeline()``, ``create_collision_pipeline()``, ``create_contacts()``, and ``prepare_contacts()``:
  detect contacts internally, customize Newton's collision pipeline, allocate the contacts the solver reports, or bind
  solver buffers to them.
* ``supports_body_forces()`` and ``supports_graph_capture()``: report configuration-dependent capabilities.
* ``register_builder_attributes()``, ``registers_builder_attributes_from()``, and ``prepare_solver_builder()``:
  register Newton custom attributes and normalize the builder before ``finalize()``.
* ``validate_cfg()``: reject settings that conflict across the physics and solver configurations.
* ``check_status()`` and ``log_debug()``: run after stepping.
* ``create_fixed_tendon_control()``: build a fixed-tendon command adapter (MJWarp only).

:class:`~isaaclab_newton.physics.MPMSolverAdapter` overrides both builder hooks. Subclasses of it that need a
custom implicit MPM solver override ``instantiate_solver(model, config)``, which receives the resolved
``SolverImplicitMPM.Config``.

Raise from a hook on an unsupported configuration rather than silently degrading. Name the adapter
``<Solver>SolverAdapter``. State a solver needs between steps belongs on the Newton solver object itself; a custom
coupled solver is a :class:`newton.solvers.SolverBase` subclass holding its sub-solvers.


Coupling Paths
--------------

The four architectures differ in what drives the substep loop:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Path
     - Structure
   * - Standalone solver
     - One manager, one solver, one model. The step graph owns the substep loop.
   * - Proxy coupling
     - :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` partitions the model
       into named entries.
       :class:`~isaaclab_contrib.coupling.CouplerSolverAdapter` builds a Newton
       coupled solver that exposes source bodies to the destination solver as
       proxies and returns lagged feedback. No new manager is required.
   * - ADMM coupling
     - :class:`~isaaclab_contrib.coupling.CouplerAdmmCfg` uses the same manager
       and entry model, but Newton creates symmetric interface constraints and
       iterates the sub-solvers. No new manager is required.
   * - Custom shared-model manager
     - ``create_solver()`` returns a :class:`newton.solvers.SolverBase` subclass
       that holds several sub-solvers and fixes the substep order in its
       ``step()``.
       :class:`~isaaclab_contrib.custom_coupling.newton_manager_cfg.CoupledMJWarpVBDSolverCfg`
       is the in-tree example. Its adapter clears force accumulators, detects
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
