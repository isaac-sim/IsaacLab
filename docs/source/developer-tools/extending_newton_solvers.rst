.. _newton-extending-solvers:

Extending Newton Solvers
========================

To add a Newton solver or a custom coupled solver, write a
:class:`~isaaclab_newton.physics.SolverSpec`. The sections below explain its
functions, resource ownership, and extension requirements.

If you only need to select and configure a shipped solver, use the user-facing
pages instead: :doc:`/source/concepts/backends_and_presets` for backend and
preset selection, the :ref:`solver tuning guides <solver-tuning>` for shipped solvers, and
:ref:`newton-coupled-solvers` for choosing a coupling approach.


When a Solver Specification Is Needed
-------------------------------------

Isaac Lab describes each Newton solver type with a :class:`~isaaclab_newton.physics.SolverSpec`, such as
:data:`~isaaclab_newton.physics.MJWARP_SPEC`: how to build the solver from its configuration, what it supports, and
the lifecycle functions that differ from Newton's defaults. A spec holds functions and constants, never state; the
solver configuration selects it through :attr:`~isaaclab_newton.physics.NewtonSolverCfg.class_type`. Write a new one when:

* a Newton solver has no Isaac Lab spec yet;
* the solver needs its own contact allocation, builder attributes, or reset
  handling;
* several solvers must advance one shared model and the substep order is part
  of the algorithm.

Do not write one when an existing solver can advance the whole model, or when
the scene can be partitioned into named solver entries. Partitioning is already
covered by :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` and
:class:`~isaaclab_contrib.coupling.CouplerAdmmCfg`, which
:data:`~isaaclab_contrib.coupling.COUPLER_SPEC` resolves into entry views over a
shared model. Prefer that path for mixed rigid and deformable scenes. Write a
custom coupled solver only when contact detection is shared but each solver
consumes the contacts differently, or when the exchange between solvers is a
custom force, impulse, or state transfer.


Responsibilities and Boundaries
-------------------------------

Data and behavior are split. :class:`~isaaclab_newton.physics.NewtonBackend` holds everything bound to one finalized
model, and the functions in :mod:`isaaclab_newton.physics.newton_backend` operate on it explicitly, in the style of
``mj_step(m, d)``. Newton's solver object (:attr:`NewtonBackend.solver <isaaclab_newton.physics.NewtonBackend>`) is
the only solver instance. The spec's functions take it and its configuration explicitly
(:func:`~isaaclab_newton.physics.solver_spec` resolves the spec of a configuration), so a coupled solver calls the
specs of its entries with each entry's solver and configuration.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Owner
     - Responsibility
   * - Simulation context
     - Resolves the manager from :attr:`~isaaclab_newton.physics.NewtonCfg.class_type` (or adopts the one passed in),
       drives the public lifecycle calls, and owns the shared builder and backend in its registry.
   * - :class:`~isaaclab_newton.physics.NewtonManager`
     - What must survive a hard reset: site requests, the replication outputs
       (:class:`~isaaclab_newton.physics.NewtonCloneRecord`), the decimation setting, and the state-force callbacks.
       It finalizes the :attr:`~isaaclab_newton.physics.NewtonManager.backend` through the registry and closes it on a
       hard reset.
   * - :class:`~isaaclab_newton.physics.NewtonBackend`
     - Model, states, control, solver, collision pipeline and contacts, sensors, Newton actuators, step callbacks,
       reset masks, and the compiled :class:`~isaaclab_newton.physics.StepGraph`. A hard reset or close discards it.
   * - Solver spec (``<SOLVER>_SPEC``)
     - Solver construction, capabilities, stepping, per-world reset of solver-owned buffers, forward kinematics, and
       builder attributes, as functions over an explicit solver, configuration, state, or builder.
   * - Coupler entry
     - A disjoint part of the shared model, when the selected spec is a coupler.
   * - Task configuration
     - A :class:`~isaaclab_newton.physics.NewtonSolverCfg` subclass whose ``class_type`` names the solver spec, plus
       entry ownership selectors for coupled setups.

``NewtonCfg.class_type`` selects the integration manager, while ``solver_cfg.class_type`` selects the solver spec. Neither
field is derived from the other. Asset factories use the manager's ``backend_name`` key, so a
custom class name does not change backend selection.


Custom Managers
---------------

To customize Isaac Lab integration, subclass ``NewtonManager`` and select it with
``NewtonCfg(class_type=MyManager, solver_cfg=...)``, or pass an instance with
``SimulationContext(sim_cfg, backend_manager=my_manager)``; the context then owns its lifetime. Call the base
implementation from overridden lifecycle methods. A custom solver only needs a ``SolverSpec``, not a new manager.


Lifecycle
---------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Public call
     - What happens
     - Solver hooks
   * - ``sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))``
     - Returns the shared builder, created by :func:`~isaaclab_newton.physics.create_builder`. The cloner imports
       declared prototypes and composes worlds into it.
     - ``attribute_solvers``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.reset` (hard)
     - Dispatches ``MODEL_INIT``, finalizes the model into a new backend, dispatches ``PHYSICS_READY`` so consumers
       bind views, sensors, actuators, and step callbacks, then constructs the solver and contacts and runs FK. Does
       not advance physics. A soft reset keeps the backend.
     - ``prepare_builder``, ``resolve_cfg``, ``validate``, ``build``, ``init_state``, ``external_contacts``,
       ``create_collision_pipeline``, ``create_contacts``, ``prepare_contacts``, ``eval_fk``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.step`
     - Runs :func:`~isaaclab_newton.physics.newton_backend.forward` for authored state, builds the step graph on the
       first step after a structural change, runs that step eagerly, and captures the graph for later steps.
     - ``reset``, ``eval_fk``, ``prepare_step``, ``step``, ``finish_step``, ``check_status``, ``log_debug``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.close`
     - Closes the backend and clears the session state.
     - none

The custom attributes of ``attribute_solvers`` are registered before particles are added and before ``finalize()``,
so it is the only place to declare Newton custom attributes. The solver is constructed after ``PHYSICS_READY``, so it sees model properties
that consumers author while binding.


The Step
--------

Each :meth:`~isaaclab_newton.physics.NewtonManager.step` launches one :class:`~isaaclab_newton.physics.StepGraph`. It
covers the whole decimation loop when the manager handles decimation
(:meth:`~isaaclab_newton.physics.NewtonManager.handles_decimation`) and one physics step otherwise. Pre-step
callbacks run once first. Every physics step then runs ``collide -> control callbacks -> Newton actuators ->
substeps -> finish_step -> post-step callbacks -> sensors``, where every substep runs ``staged forces -> state-force
callbacks -> step``, and post-actuator callbacks run after the actuators of the last physics step. Each list of
:class:`~isaaclab_newton.physics.StepCallbacks` has one cadence.

The graph binds every buffer when it is built. Double-buffered solvers alternate input and output states at build
time, and each physics step ends in ``state_0``, so ``manager.backend.state_0`` is always the same object.
Consecutive graphable operations are captured together; operations that cannot be captured, such as actuators with Python history indices or eager callbacks, run in place between captured segments. Consumers add operations with
:meth:`~isaaclab_newton.physics.NewtonManager.register_control_callback` and its siblings while ``PHYSICS_READY`` dispatches.

The backend functions read no manager state, so backends with different solvers can be built from their own builders
and stepped side by side:

.. code-block:: python

   from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBackend, NewtonBackendCfg, NewtonCfg, VBDSolverCfg
   from isaaclab_newton.physics import newton_backend as nb

   def backend_cfg(solver_cfg):
       return NewtonBackendCfg(physics_cfg=NewtonCfg(solver_cfg=solver_cfg), device="cuda:0")

   rigid = NewtonBackend(backend_cfg(MJWarpSolverCfg()), rigid_builder, dt=0.01)
   cloth = NewtonBackend(backend_cfg(VBDSolverCfg()), cloth_builder, dt=0.01)
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

Define a :class:`~isaaclab_newton.physics.SolverSpec` with a ``build`` function and the fields that differ from the
defaults, and point the solver configuration's ``class_type`` at it; ``instantiate(solver_cfg, model, mode)`` then
builds the solver. Every field has a default, so a simple solver needs only ``build``;
:func:`~isaaclab_newton.physics.build_from_cfg` constructs a Newton solver from the matching configuration fields:

.. code-block:: python

   from newton.solvers import SolverMySolver

   from isaaclab.utils import configclass
   from isaaclab_newton.physics import NewtonSolverCfg, SolverSpec, build_from_cfg


   @configclass
   class MySolverCfg(NewtonSolverCfg):
       class_type: SolverSpec | str = "{DIR}.my_solver:MY_SOLVER_SPEC"
       solver_type: str = "my_solver"
       iterations: int = 16


   MY_SOLVER_SPEC = SolverSpec(build=build_from_cfg(SolverMySolver), deterministic=True)

Functions called before the model exists take the configuration; functions called afterwards take Newton's solver
and the configuration, plus the state, masks, or contacts they act on. None of them read the backend, and none keep
state: state a solver needs between steps belongs on the Newton solver object.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``attribute_solvers``
     - Newton solvers whose custom builder attributes and USD schemas the configuration needs before import.
   * - ``prepare_builder``
     - Normalize the complete builder before ``finalize()``, e.g. VBD coloring.
   * - ``resolve_cfg``
     - Resolve model-dependent configuration defaults; the backend keeps the result as ``backend.solver_cfg``.
   * - ``validate``
     - Reject settings that conflict across the physics and solver configurations.
   * - ``build``
     - Construct Newton's solver for a model.
   * - ``init_state``
     - Initialize the output state of double-buffered solvers before capture.
   * - ``single_state``, ``deterministic``, ``contact_sensors``, ``captures_model_changes``
     - Constant capabilities: stepping in place on one ``State``, honoring a Warp determinism guarantee, readable
       contacts for Newton contact sensors, and recording ``notify_model_changed()`` into a conditional CUDA graph
       that skips the refresh when nothing changed.
   * - ``body_forces``, ``supports_capture``
     - Configuration-dependent capabilities: consuming applied body forces, and CUDA graph capture.
   * - ``ignored_model_changes``
     - Model changes the solver does not apply after construction, mapped to a one-time warning.
   * - ``coupling_error``
     - Why the solver cannot run as an entry of a coupled solver, or ``None``.
   * - ``external_contacts``, ``create_collision_pipeline``, ``create_contacts``, ``prepare_contacts``
     - Detect contacts internally, customize Newton's collision pipeline, allocate the contacts the solver reports,
       or bind solver buffers to them.
   * - ``prepare_step``, ``finish_step``
     - Refresh acceleration structures before, or publish derived state such as joint coordinates after, each physics
       step. ``None`` (the default) skips them; both must be CUDA-graph-safe.
   * - ``step``
     - Change one substep, for example to run a projection after the solve.
   * - ``reset``, ``eval_fk``
     - Clear solver-owned history for masked worlds without touching authored joint state; solver-specific forward
       kinematics.
   * - ``check_status``, ``log_debug``
     - Run after stepping.
   * - ``fixed_tendon_control``
     - Build a fixed-tendon command adapter (MJWarp only).

Variants derive with :func:`dataclasses.replace`. A task that needs a different Newton solver class, such as a custom
implicit MPM solver, replaces only ``build``:
``replace(MPM_SPEC, build=mpm_builder(lambda model, config, cfg: MyMPM(model, config)))`` keeps the shared MPM
configuration resolution. Subclass a Newton solver only to change the physics; integration behavior belongs in spec
functions. Raise from a function on an unsupported configuration rather than silently degrading.


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
       :data:`~isaaclab_contrib.coupling.COUPLER_SPEC` builds a Newton
       coupled solver that exposes source bodies to the destination solver as
       proxies and returns lagged feedback. Each entry's solver is built and
       queried through the entry's own spec. No new manager is required.
   * - ADMM coupling
     - :class:`~isaaclab_contrib.coupling.CouplerAdmmCfg` uses the same manager
       and entry model, but Newton creates symmetric interface constraints and
       iterates the sub-solvers. No new manager is required.
   * - Custom shared-model solver
     - ``build`` returns a :class:`newton.solvers.SolverBase` subclass
       that holds several sub-solvers, built through their own specs, and fixes
       the substep order in its ``step()``.
       :class:`~isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_solver_cfg.CoupledMJWarpVBDSolverCfg`
       is the in-tree example. Its solver clears force accumulators, detects
       contacts once, injects soft-to-rigid reactions into ``body_f`` when
       ``coupling_mode="two_way"``, then advances MJWarp on its own internal
       contacts and VBD on the detected ones.

A custom shared-model solver bypasses entry ownership resolution, so it cannot
reuse the coupler's selectors or validation. For the proxy and ADMM trade-offs,
see :ref:`newton-coupled-solvers`.


.. _related-documentation:

Where to go next
----------------

* :doc:`add_physics_backend`: adding a whole physics backend.
* :doc:`/source/api/lab_newton/isaaclab_newton.physics`: manager and solver
  configuration API reference.
