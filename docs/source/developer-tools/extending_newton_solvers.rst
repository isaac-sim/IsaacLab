.. _newton-extending-solvers:

Extending Newton Solvers
========================

This page is for contributors adding a Newton solver to Isaac Lab or building a
custom coupled solver. It describes the
:class:`~isaaclab_newton.physics.NewtonManager` extension contract: what the
manager owns, when its hooks run, and what a subclass must provide.

If you only need to select and configure a shipped solver, use the user-facing
pages instead: :doc:`/source/concepts/backends_and_presets` for backend and
preset selection, the :ref:`solver tuning guides <solver-tuning>` for shipped solvers, and
:ref:`newton-coupled-solvers` for choosing a coupling approach.


When a Solver Manager Is Needed
-------------------------------

Each Newton solver is exposed as a :class:`~isaaclab_newton.physics.NewtonManager` subclass, such as
:class:`~isaaclab_newton.physics.NewtonMJWarpManager`. Write a new one when:

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

Data and behavior are split. :class:`~isaaclab_newton.physics.NewtonBackend` holds everything bound to one finalized
model, and the functions in :mod:`isaaclab_newton.physics.newton_backend` operate on it explicitly, in the style of
``mj_step(m, d)``. A solver manager contributes only stateless classmethod hooks that take the backend as an argument.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Owner
     - Responsibility
   * - Simulation context
     - Resolves the manager from :attr:`~isaaclab_newton.physics.NewtonCfg.class_type`, owns the shared builder and
       the backend through its registry, and drives the public lifecycle calls.
   * - :class:`~isaaclab_newton.physics.NewtonManager`
     - The active :attr:`~isaaclab_newton.physics.NewtonManager.backend`, plus what must survive a hard reset: site
       requests, the replication outputs (:class:`~isaaclab_newton.physics.NewtonCloneRecord`), and the decimation
       setting.
   * - :class:`~isaaclab_newton.physics.NewtonBackend`
     - Model, states, control, solver, collision pipeline and contacts, sensors, Newton actuators, step callbacks,
       reset masks, and the compiled :class:`~isaaclab_newton.physics.StepGraph`. A hard reset or close discards it.
   * - Solver manager (``Newton<Solver>Manager``)
     - Solver construction, capabilities, stepping, per-world reset of solver-owned buffers, forward kinematics, and
       builder attributes, as classmethods over an explicit backend.
   * - Coupler entry
     - A disjoint part of the shared model, when the active manager is a coupler.
   * - Task configuration
     - A :class:`~isaaclab_newton.physics.NewtonSolverCfg` subclass whose ``class_type`` points at the manager, plus
       entry ownership selectors for coupled setups.

:class:`~isaaclab_newton.physics.NewtonCfg` copies ``solver_cfg.class_type`` onto its own
:attr:`~isaaclab_newton.physics.NewtonCfg.class_type`, so task configuration never names the manager directly.


Lifecycle
---------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Public call
     - What happens
     - Solver hooks
   * - ``sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))``
     - Acquires the shared builder through the active manager's ``create_builder()`` factory. The cloner imports
       declared prototypes and composes worlds into it.
     - ``register_builder_attributes()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.reset` (hard)
     - Dispatches ``MODEL_INIT``, finalizes the model into a new backend, dispatches ``PHYSICS_READY`` so consumers
       bind views, sensors, actuators, and step callbacks, then constructs the solver and contacts and runs FK. Does
       not advance physics. A soft reset keeps the backend.
     - ``prepare_solver_builder()``, ``validate_cfg()``, ``create_solver()``, ``initialize_output_state()``,
       ``uses_collision_pipeline()``, ``create_contacts()``, ``prepare_contacts()``, ``eval_fk()``
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
(:meth:`~isaaclab_newton.physics.NewtonManager.handles_decimation`) and one physics step otherwise. Every physics step
runs ``collide -> CONTROL -> Newton actuators -> POST_ACTUATOR -> substeps``, where every substep runs
``staged forces -> STATE_FORCE -> step_solver()``; ``POST_STEP`` callbacks and sensors run once at the end.

The graph binds every buffer when it is built. Double-buffered solvers alternate input and output states at build
time, and each physics step ends in ``state_0``, so ``NewtonManager.get_state_0()`` is always the same object.
Consecutive graphable operations are captured together; operations that cannot be captured, such as TorchScript
actuators or eager callbacks, run in place between captured segments. Consumers add operations with
:meth:`~isaaclab_newton.physics.NewtonManager.register_step_callback` while ``PHYSICS_READY`` dispatches.

The backend functions read no manager state, so backends with different solvers can be built from their own models
and stepped side by side, or recorded into one caller-owned CUDA graph:

.. code-block:: python

   from isaaclab_newton.physics import NewtonBackend
   from isaaclab_newton.physics import newton_backend as nb

   rigid = NewtonBackend(rigid_model, NewtonCfg(solver_cfg=MJWarpSolverCfg()), dt=0.01)
   cloth = NewtonBackend(cloth_model, NewtonCfg(solver_cfg=VBDSolverCfg()), dt=0.01)
   for backend in (rigid, cloth):
       nb.init_solver(backend)
       nb.step(backend, steps=2)  # eager warm-up; also builds the step graph

   with wp.ScopedCapture() as capture:
       nb.record_step(rigid, steps=2)
       nb.record_step(cloth, steps=2)
   wp.capture_launch(capture.graph)


Extension Contract
------------------

Subclass :class:`~isaaclab_newton.physics.NewtonManager`, implement
:meth:`~isaaclab_newton.physics.NewtonManager.create_solver`, and set the capability class attributes when they differ
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
   * - ``ignored_model_changes``
     - Model changes the solver does not apply after construction, mapped to a one-time warning.
   * - ``builder_attribute_solvers``
     - Newton solvers whose custom builder attributes are registered before import.

.. code-block:: python

   import warp as wp
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
       supports_deterministic = True

       @classmethod
       def create_solver(
           cls, model: Model, solver_cfg: MySolverCfg, deterministic_mode=wp.DeterministicMode.NOT_GUARANTEED
       ):
           return SolverMySolver(model, iterations=solver_cfg.iterations)

Override anything else only when the solver needs it. Every hook takes the backend explicitly and must not keep
class state:

* ``step_solver()``: change one substep, for example to run a projection after the solve.
* ``reset_solver()``: clear solver-owned history for masked worlds without touching authored joint state.
* ``eval_fk()``: use a solver-specific forward kinematics.
* ``prepare_step()``: refresh acceleration structures once per physics step; must stay graphable.
* ``initialize_output_state()``: initialize the output state of double-buffered solvers before capture.
* ``uses_collision_pipeline()``, ``create_contacts()``, and ``prepare_contacts()``: detect contacts internally,
  allocate the contacts the solver reports, or bind solver buffers to them.
* ``supports_body_forces()`` and ``supports_graph_capture()``: report configuration-dependent capabilities.
* ``register_builder_attributes()``, ``registers_builder_attributes_from()``, and ``prepare_solver_builder()``:
  register Newton custom attributes and normalize the builder before ``finalize()``.
* ``validate_cfg()``: reject settings that conflict across the physics and solver configurations.
* ``check_status()`` and ``log_debug()``: run after stepping.
* ``create_fixed_tendon_control()``: build a fixed-tendon command adapter (MJWarp only).

:class:`~isaaclab_newton.physics.NewtonMPMManager` overrides both builder hooks.

Raise from a hook on an unsupported configuration rather than silently degrading. Name the manager
``Newton<Solver>Manager``. State a solver needs between steps belongs on the Newton solver object itself; a custom
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
       :class:`~isaaclab_contrib.coupling.NewtonCouplerManager` builds a Newton
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
       is the in-tree example. Its manager clears force accumulators, detects
       contacts once, injects soft-to-rigid reactions into ``body_f`` when
       ``coupling_mode="two_way"``, then advances MJWarp on its own internal
       contacts and VBD on the detected ones.

A custom shared-model manager bypasses entry ownership resolution, so it cannot
reuse the coupler's selectors or validation. For the proxy and ADMM trade-offs,
see :ref:`newton-coupled-solvers`.


Related Documentation
---------------------

* :doc:`add_physics_backend`: adding a whole physics backend.
* :doc:`/source/api/lab_newton/isaaclab_newton.physics`: manager and solver
  configuration API reference.
