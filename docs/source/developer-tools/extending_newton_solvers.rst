.. _newton-extending-solvers:

Extending Newton Solvers
========================

To add a Newton solver or a custom coupled solver, write an Isaac Lab solver, a
:class:`~isaaclab_newton.physics.NewtonSolver` subclass, and its configuration, a
:class:`~isaaclab_newton.physics.NewtonSolverCfg` subclass whose ``class_type`` selects it. The sections below explain
what the solver overrides, resource ownership, and extension requirements.

If you only need to select and configure a shipped solver, use the user-facing
pages instead: :doc:`/source/concepts/backends_and_presets` for backend and
preset selection, the :ref:`solver tuning guides <solver-tuning>` for shipped solvers, and
:ref:`newton-coupled-solvers` for choosing a coupling approach.


When a New Solver Is Needed
---------------------------

Every Newton solver runs through one generic path written against Newton's
:class:`~newton.solvers.SolverBase` API: build the solver from the configuration fields matching its constructor,
step it, reset it for masked worlds, evaluate forward kinematics, and collide with Newton's
:class:`~newton.CollisionPipeline`. An Isaac Lab solver names its Newton solver class
(:attr:`~isaaclab_newton.physics.NewtonSolver.solver_class`) and overrides capabilities and methods only where its
solver differs. Write a new one when:

* a Newton solver has no Isaac Lab configuration yet;
* the solver needs its own construction, contact allocation, or reset handling;
* several solvers must advance one shared model and the substep order is part
  of the algorithm.

Do not write one when an existing solver can advance the whole model, or when
the scene can be partitioned into named solver entries. Partitioning is already
covered by :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` and
:class:`~isaaclab_contrib.coupling.CouplerAdmmCfg`, which resolve into entry views over a
shared model. Prefer that path for mixed rigid and deformable scenes. Write a
custom coupled solver only when contact detection is shared but each solver
consumes the contacts differently, or when the exchange between solvers is a
custom force, impulse, or state transfer.


Responsibilities and Boundaries
-------------------------------

:class:`~isaaclab_newton.physics.NewtonBackend` holds everything bound to one finalized model, and the
:class:`~isaaclab_newton.physics.NewtonManager` drives it. The backend's solver
(:attr:`NewtonBackend.solver <isaaclab_newton.physics.NewtonBackend>`) is Newton's solver, built with
``solver_cfg.class_type.build(solver_cfg, model, mode)``. The Isaac Lab solver is never instantiated and holds no state:
each method is a class method that takes the configuration and Newton's solver. A coupled solver builds each entry's
Newton solver through the entry's own Isaac Lab solver.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Owner
     - Responsibility
   * - Simulation context
     - Builds the manager from :attr:`~isaaclab_newton.physics.NewtonCfg.class_type` with ``instantiate(cfg)``,
       drives the public lifecycle calls, and owns the shared builder and backend in its registry.
   * - :class:`~isaaclab_newton.physics.NewtonManager`
     - What must survive a hard reset: site requests, the replication outputs
       (:class:`~isaaclab_newton.physics.NewtonCloneRecord`), and the decimation setting. It finalizes the
       :attr:`~isaaclab_newton.physics.NewtonManager.backend` through the registry, closes it on a hard reset, and runs
       the step, holding the step callbacks and the captured step.
   * - :class:`~isaaclab_newton.physics.NewtonBackend`
     - Model, states, control, solver, collision pipeline and contacts, sensors, Newton actuators, and reset masks.
       A hard reset or close discards it.
   * - Solver configuration
     - Solver parameters, and ``class_type``, which selects the Isaac Lab solver.
   * - Isaac Lab solver
     - What differs from the generic path: construction, capabilities, stepping, per-world reset of solver-owned
       buffers, forward kinematics, contacts, and builder preparation.
   * - Coupler entry
     - A disjoint part of the shared model, when the solver configuration is a coupler.

``NewtonCfg.class_type`` selects the integration manager independently of the solver. Asset factories use the
manager's ``backend_name`` key, so a custom class name does not change backend selection.


Custom Managers
---------------

To customize Isaac Lab integration, subclass ``NewtonManager`` and select it with
``NewtonCfg(class_type=MyManager, solver_cfg=...)``. Call the base implementation from overridden lifecycle methods.
A custom solver only needs an Isaac Lab solver and its configuration, not a new manager.


Lifecycle
---------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Public call
     - What happens
     - Isaac Lab solver
   * - :meth:`~isaaclab_newton.physics.NewtonManager.initialize`
     - Stores the simulation context and acquires the shared :attr:`~isaaclab_newton.physics.NewtonManager.builder`,
       created by :func:`~isaaclab_newton.physics.create_builder`. The cloner later imports declared prototypes and
       composes worlds into it.
     - ``solver_classes``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.reset` (hard)
     - Dispatches ``MODEL_INIT``, finalizes the model into a new backend, dispatches ``PHYSICS_READY`` so consumers
       bind views, sensors, actuators, and step callbacks, then constructs the solver and contacts and runs FK. Does
       not advance physics. A soft reset keeps the backend.
     - ``prepare_builder``, ``build``, ``init_state``, ``uses_collision_pipeline``,
       ``create_collision_pipeline``, ``create_contacts``, ``prepare_contacts``, ``eval_fk``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.step`
     - Runs :meth:`~isaaclab_newton.physics.NewtonManager.forward` for authored state, runs the first step after a
       structural change eagerly, and captures it as a CUDA graph for later steps.
     - ``reset``, ``eval_fk``, ``pre_step``, ``step``, ``post_step``, ``check_step``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.close`
     - Closes the backend and clears the session state.
     - none

The custom attributes of the Newton solvers in ``solver_classes`` are registered before particles are added and
before ``finalize()``. The solver is constructed after ``PHYSICS_READY``, so it sees model properties that consumers
author while binding.


The Step
--------

Each :meth:`~isaaclab_newton.physics.NewtonManager.step` covers the whole decimation loop when the manager handles
decimation (:meth:`~isaaclab_newton.physics.NewtonManager.handles_decimation`: an articulation runs its Newton
actuators in the step and all of them are CUDA-graph-safe), and one physics step otherwise. In the first case every
physics step runs ``collide -> pre-actuator callbacks -> Newton actuators -> post-actuator callbacks -> substeps``,
and post-step callbacks and sensors run once at the end. Otherwise the actuators and their callbacks run eagerly once,
followed by ``collide -> substeps -> post-step callbacks -> sensors``. Every substep runs ``staged body forces ->
state-force callbacks -> step``, and ``pre_step`` and ``post_step`` run around each physics step's substeps.

Every physics step ends in ``state_0``, so ``manager.backend.state_0`` is always the same object. The captured step
records the callbacks, so they must be graph-safe. Consumers register them with
:meth:`~isaaclab_newton.physics.NewtonManager.register_post_step_callback` and its siblings while ``PHYSICS_READY``
dispatches.

Property writers call :meth:`~isaaclab_newton.physics.NewtonManager.add_model_change`; the manager commits the queued
categories once at the next step.


Extension Contract
------------------

Subclass :class:`~isaaclab_newton.physics.NewtonSolver`, name the Newton solver class, and point a
configuration at it. A solver whose constructor takes the configuration fields as keyword arguments needs nothing
else:

.. code-block:: python

   from isaaclab.utils import configclass
   from isaaclab_newton.physics import NewtonSolverCfg, NewtonSolver

   from my_package.solvers import SolverMySolver


   class MySolver(NewtonSolver):
       solver_class = SolverMySolver
       deterministic = True


   @configclass
   class MySolverCfg(NewtonSolverCfg):
       class_type: type | str = "my_package.my_solver:MySolver"

       iterations: int = 16

``MySolver.build(cfg, model, mode, env_template)`` returns Newton's solver. Every method is a class method:
methods called before the model exists take the configuration, and the rest also take Newton's solver, e.g.
``MySolver.reset(cfg, solver, state, world_mask)``. State a solver needs between steps belongs on Newton's
solver. Override only what differs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Meaning
   * - ``solver_class``
     - Newton solver class that ``build`` constructs.
   * - ``single_state``, ``deterministic``, ``contact_sensors``
     - Class-level capabilities: stepping in place on one ``State``, honoring a Warp determinism guarantee, and
       readable contacts for Newton contact sensors.
   * - ``ignored_model_changes``
     - :class:`newton.ModelFlags` the solver does not apply after construction, mapped to a one-time warning.
   * - ``solver_classes``
     - Newton solvers whose custom builder attributes and USD schemas the builder needs before import; defaults to
       ``solver_class``.
   * - ``prepare_builder``
     - Normalize the complete builder before ``finalize()``, e.g. VBD coloring.
   * - ``applied_deterministic_mode``, ``coupling_error``
     - The determinism mode the solver runs with for a request, and why the solver cannot run as a coupled-solver
       entry.
   * - ``shares_materials``
     - Whether contacts read one material table shared by all environments, e.g. Kamino's own collision detector.
   * - ``build``
     - Construct Newton's solver for a model, given the determinism mode and the clone plan's environment template.
   * - ``init_state``
     - Initialize the output state of double-buffered solvers before capture.
   * - ``applies_body_forces``, ``supports_capture``
     - Configuration-dependent capabilities: consuming applied body forces, and CUDA graph capture.
   * - ``uses_collision_pipeline``, ``create_collision_pipeline``, ``create_contacts``, ``prepare_contacts``
     - Detect contacts internally, customize Newton's collision pipeline, allocate the contacts the solver reports, or
       bind solver buffers to them.
   * - ``pre_step``, ``post_step``
     - Refresh acceleration structures before, or publish derived state such as joint coordinates after, each physics
       step. Both must be CUDA-graph-safe.
   * - ``step``
     - Change one substep, for example to run a projection after the solve.
   * - ``reset``, ``eval_fk``
     - Clear solver-owned history for masked worlds without touching authored joint state; solver-specific forward
       kinematics.
   * - ``check_step``
     - Raise asynchronous failures and log diagnostics after a step launch.
   * - ``fixed_tendon_control``
     - Build a fixed-tendon command adapter (MJWarp only).

A task that needs a different Newton solver class subclasses the Isaac Lab solver it extends and selects it through
its configuration's ``class_type``: a custom VBD solver only sets ``solver_class``, and a custom implicit MPM solver
overrides ``build`` around ``isaaclab_newton.physics.mpm_solver.mpm_solver_config``, which keeps the shared MPM
configuration resolution. Subclass a Newton solver only to change the physics; integration behavior belongs in the
Isaac Lab solver. Raise from a method on an unsupported configuration rather than silently degrading.


Coupling Paths
--------------

The four architectures differ in what drives the substep loop:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Path
     - Structure
   * - Standalone solver
     - One manager, one solver, one model. The captured step owns the substep loop.
   * - Proxy coupling
     - :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` partitions the model
       into named entries and builds a Newton coupled solver that exposes source
       bodies to the destination solver as proxies and returns lagged feedback.
       Each entry's Newton solver is built through the entry's own
       configuration. No new manager is required.
   * - ADMM coupling
     - :class:`~isaaclab_contrib.coupling.CouplerAdmmCfg` uses the same manager
       and entry model, but Newton creates symmetric interface constraints and
       iterates the sub-solvers. No new manager is required.
   * - Custom shared-model solver
     - ``build`` returns a :class:`newton.solvers.SolverBase` subclass
       that holds several sub-solvers, built from their own configurations,
       and fixes the substep order in its ``step()``.
       :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolver`
       is the in-tree example. Its Newton solver clears force accumulators, detects
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
