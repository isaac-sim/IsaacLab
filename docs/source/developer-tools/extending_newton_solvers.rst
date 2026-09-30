.. _newton-extending-solvers:

Extending Newton Solvers
========================

This page is for contributors adding a Newton solver to Isaac Lab or building a
custom coupled solver. It describes the
:class:`~isaaclab_newton.physics.NewtonManager` extension contract: what the
manager owns, when its hooks run, and what a subclass must provide.

If you only need to select and configure a shipped solver, use the user-facing
pages instead: :doc:`/source/concepts/backends_and_presets` for backend and
preset selection, :doc:`/source/concepts/solver-tuning/index` for the per-solver guides, and
:ref:`newton-coupled-solvers` for choosing a coupling approach.


When a Solver Manager Is Needed
-------------------------------

Each Newton solver is exposed as a :class:`~isaaclab_newton.physics.NewtonSolverBinding` selected by a
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

:class:`~isaaclab_newton.physics.NewtonManager` is a facade over three pieces of data with separate lifetimes. A
solver contributes only a :class:`~isaaclab_newton.physics.NewtonSolverBinding`; the concrete manager exists so that
:attr:`~isaaclab_newton.physics.NewtonCfg.class_type` can select the binding.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Owner
     - Responsibility
   * - Simulation context
     - Resolves the manager from :attr:`~isaaclab_newton.physics.NewtonCfg.class_type`, owns the native builder
       and model through its backend registry, and drives the public lifecycle calls.
   * - Build requests
     - Sites, extended state and contact attributes, per-world builder hooks, and the cloner's outputs. Collected
       before the model is finalized and kept across hard resets.
   * - Runtime
     - Everything bound to one finalized model: the solver binding, collision pipeline and contacts, sensors,
       Newton actuators, consumer stages, reset masks, and the compiled step program. A hard reset or close
       discards it.
   * - Solver binding
     - Solver construction, capabilities, stepping, per-world reset of solver-owned buffers, forward kinematics,
       and builder attributes.
   * - Coupler entry
     - A disjoint part of the shared model, when the active binding is a coupler.
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
     - Binding hooks
   * - ``sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))``
     - Acquires the shared builder through the active manager's ``create_builder()`` factory. The cloner imports
       declared prototypes and composes worlds into it.
     - ``register_builder_attributes()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.start_simulation`
     - Applies build requests, finalizes the model, creates the runtime and schema, then dispatches
       ``PHYSICS_READY`` so consumers bind views, sensors, actuators, and stages.
     - ``prepare_builder()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.initialize_solver`
     - Constructs the binding and contacts and runs FK. Does not advance physics.
     - ``validate_cfg()``, ``__init__``/``construct()``, ``initialize_output_state()``, ``create_contacts()``,
       ``prepare_contacts()``, ``eval_fk()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.step`
     - Reconciles authored state, compiles the step program on the first step after a structural change (and
       captures it on CUDA), then runs it.
     - ``reset()``, ``eval_fk()``, ``prepare_step()``, ``step()``, ``check_status()``, ``log_debug()``
   * - :meth:`~isaaclab_newton.physics.NewtonManager.reset`
     - A hard reset discards the runtime and model, then re-runs ``start_simulation()`` and
       ``initialize_solver()``; a soft reset keeps them.
     - as above
   * - :meth:`~isaaclab_newton.physics.NewtonManager.close`
     - Releases the runtime and build requests.
     - none

``register_builder_attributes()`` runs before particles are added and before ``finalize()``, so it is the only place
to register Newton custom attributes. The binding is constructed after the model is finalized, so it may size solver
resources from the real model.


The Step Program
----------------

Each :meth:`~isaaclab_newton.physics.NewtonManager.step` runs one compiled program. It covers the whole decimation
loop when Newton actuators are active (:meth:`~isaaclab_newton.physics.NewtonManager.handles_decimation`) and one
physics step otherwise. Every physics step runs ``collide -> Newton actuators -> CONTROL stages -> substeps``, where
every substep runs ``SUBSTEP stages -> binding.step() -> clear forces``; ``POST_STEP`` stages and sensors run once at
the end.

The runtime is plain data. The functions in :mod:`isaaclab_newton.physics.runtime` take it explicitly
(``runtime.step(rt, steps, capture)``, ``runtime.add_stage(rt, stage)``), and the manager only supplies the active
simulation's runtime, so runtimes bound to different Newton backends can be driven side by side.

The program binds every buffer when it is compiled. Double-buffered solvers alternate input and output states at
compile time, and each physics step ends in ``state_0``, so ``NewtonManager.get_state_0()`` is always the same object.
Consecutive graph-safe operations are captured together; operations that are not graph-safe, such as TorchScript
actuators or eager consumer stages, run in place between captured segments. Consumers add operations with
:meth:`~isaaclab_newton.physics.NewtonManager.add_stage` while ``PHYSICS_READY`` dispatches.


Extension Contract
------------------

Implement :meth:`~isaaclab_newton.physics.NewtonSolverBinding.create` and set the capability attributes in
``__init__`` when they differ from the defaults (double-buffered states, Newton's collision pipeline, body-force
input, contact sensors):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Attribute
     - Meaning
   * - ``single_state``
     - ``True`` if the solver steps in place on one ``State``.
   * - ``needs_collision_pipeline``
     - ``False`` if the solver detects contacts internally; implement ``create_contacts()`` to report them.
   * - ``supports_body_forces``
     - ``True`` if the solver consumes external rigid-body forces from ``State.body_f``.
   * - ``supports_contact_sensors``
     - ``False`` if Newton contact sensors cannot read the solver's contacts.
   * - ``ignored_model_changes``
     - Model changes the solver does not apply after construction, mapped to a one-time warning.

.. code-block:: python

   import warp as wp
   from newton import Model
   from newton.solvers import SolverMySolver

   from isaaclab.utils import configclass
   from isaaclab_newton.physics import NewtonManager, NewtonSolverBinding, NewtonSolverCfg


   @configclass
   class MySolverCfg(NewtonSolverCfg):
       class_type: type[NewtonManager] | str = "{DIR}.my_solver_manager:NewtonMySolverManager"
       solver_type: str = "my_solver"
       iterations: int = 16


   class MySolverBinding(NewtonSolverBinding):
       @classmethod
       def create(cls, model: Model, solver_cfg: MySolverCfg, deterministic_mode=wp.DeterministicMode.NOT_GUARANTEED):
           return SolverMySolver(model, iterations=solver_cfg.iterations)


   class NewtonMySolverManager(NewtonManager):
       solver_binding = MySolverBinding

Override anything else only when the solver needs it:

* ``step()``: change one substep, for example to run a projection after the solve.
* ``reset()``: clear solver-owned history for masked worlds without touching authored joint state.
* ``eval_fk()``: use a solver-specific forward kinematics.
* ``prepare_step()``: refresh acceleration structures once per physics step; must stay graph-safe.
* ``initialize_output_state()``: initialize the output state of double-buffered solvers before capture.
* ``create_contacts()`` and ``prepare_contacts()``: allocate internal contacts or bind solver buffers to them.
* ``register_builder_attributes()``, ``registers_builder_attributes_from()``, and ``prepare_builder()``: register
  Newton custom attributes and normalize the builder before ``finalize()``.
* ``validate_cfg()``: reject settings that conflict across the physics and solver configurations.
* ``supports_graph_capture``: return ``False`` to fall back to eager execution.
* ``check_status()`` and ``log_debug()``: run after stepping.
* ``create_fixed_tendon_control()``: build a fixed-tendon command adapter (MJWarp only).

:class:`~isaaclab_newton.physics.MPMSolverBinding` overrides both builder hooks.

Raise from the binding on an unsupported configuration rather than silently degrading. Name the binding
``<Solver>SolverBinding`` and the manager ``Newton<Solver>Manager``.


Coupling Paths
--------------

The four architectures differ in what drives the substep loop:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Path
     - Structure
   * - Standalone solver
     - One binding, one solver, one model. The step program owns the substep loop.
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
     - A binding constructs several sub-solvers in ``construct()`` and overrides
       ``step()`` to fix the substep order.
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
