.. _mjwarp-solver-tuning:

MJWarp Solver
=============

The MuJoCo-Warp solver from Google DeepMind is the primary, validated solver
for the Newton backend in Isaac Lab. It is enabled by setting
:attr:`~isaaclab_newton.physics.NewtonCfg.solver_cfg` to a
:class:`~isaaclab_newton.physics.MJWarpSolverCfg`, usually exposed as the
``newton_mjwarp`` physics preset on a task configuration. Newton ships with
support for other solvers as well — see :doc:`kamino-solver` and
:ref:`hydra-backend-solver-presets` for how presets are selected. For details
on how solver-specific managers are implemented, or how to add a new solver
manager, see :doc:`newton-manager-abstraction`.

For a diagnose-first workflow, parameter-ordering guidance, and links to the current
solver-specific references, see Newton's
`Simulation Tuning guide <https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html>`__.
The companion `Solver Tuning Reference
<https://newton-physics.github.io/newton/latest/concepts/simulation_tuning_solvers.html>`__
lists the options supported by each Newton solver, while the `MuJoCo-Warp Contact Tuning guide
<https://newton-physics.github.io/newton/latest/concepts/simulation_tuning_mujoco.html>`__
explains the MuJoCo constraint model and contact-parameter mapping.


The mechanical scene still comes from USD, but solver configuration is backend-specific.
:class:`~isaaclab_newton.physics.NewtonCfg` replaces
:class:`~isaaclab_physx.physics.PhysxCfg` inside an MJWarp physics preset; the simulation ``dt``
remains on :class:`~isaaclab.sim.SimulationCfg`.

Do not translate PhysX knobs one-for-one
----------------------------------------

PhysX and MJWarp expose some related concepts, but their parameters have different scopes,
algorithms, and units. A matching number rarely means matching behavior.

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - PhysX configuration surface
     - Closest MJWarp surface
     - Migration rule
   * - Scene min/max solver iterations and per-actor position or velocity iterations
     - ``iterations``, ``ls_iterations``, and ``tolerance``
     - There is no numeric conversion. MJWarp's values are solver-wide nonlinear and line-search
       convergence caps, not per-actor TGS or PGS sweeps.
   * - ``gpu_max_rigid_contact_count``, ``gpu_max_rigid_patch_count``, and found/lost-pair buffers
     - ``nconmax`` and ``njmax``; or ``collision_cfg.rigid_contact_max`` and
       ``collision_cfg.max_triangle_pairs`` when using Newton contacts
     - PhysX buffers are scene-wide GPU allocations. ``nconmax`` and ``njmax`` are per environment.
       Size them from MJWarp contact and constraint demand, not from the PhysX allocation.
   * - ``bounce_threshold_velocity``
     - Contact material, timestep, damping, and the active MuJoCo ``solref``/``solimp`` mapping
     - MJWarp has no direct bounce-threshold replacement. Reproduce the intended contact response
       instead of copying the threshold.
   * - ``friction_correlation_distance``
     - Collision geometry and contact count, ``cone``, ``impratio``, and material friction
     - There is no distance-to-``impratio`` conversion. Establish the expected contacts first, then
       tune friction behavior.
   * - ``solver_type`` TGS or PGS
     - ``solver="newton"`` or ``"cg"``
     - These are different algorithms. Maintained Isaac Lab tasks use ``"newton"``; do not infer a
       pairing from the names.
   * - ``enable_ccd``
     - Contact representation and ``ccd_iterations``
     - ``ccd_iterations`` is a convergence cap for convex collision detection, not an enable flag
       equivalent to PhysX CCD.
   * - ``enable_stabilization``
     - Model validation, smaller solver ``dt``, substeps, damping, and convergence work
     - There is no direct MJWarp stabilization pass. Diagnose the underlying failure.

Start from an explicit baseline
-------------------------------

The following baseline makes the frequently relied-on values visible. ``iterations``,
``ls_iterations``, and ``tolerance`` match the current
:class:`~isaaclab_newton.physics.MJWarpSolverCfg` defaults; ``integrator="implicitfast"`` is an
intentional override used by nearly every maintained Isaac Lab MJWarp environment because it is a
better starting point for stiff drives and contact than the Isaac Lab configuration default of
``"euler"``.

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    solver_cfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        njmax=50,
        nconmax=20,
        cone="pyramidal",
        impratio=1.0,
        iterations=100,
        ls_iterations=50,
        tolerance=1.0e-6,
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,  # Set True during tuning, False for production.
    )
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=newton_cfg)

Enable ``debug_mode`` during tuning so Isaac Lab reports the minimum, mean, maximum, and standard
deviation of the per-environment solver iteration count and warns when an environment reaches the
``iterations`` cap. Disable it for production training after the solver budget is validated.

Repository starting profiles
----------------------------

The checked-in task configurations show four recurring profiles. These values are **initial
budgets**, not universal safe values. Validate them with the task's worst-case reset, contact, and
command distribution, then keep only the capacity and convergence work that the task needs.

.. list-table::
   :header-rows: 1
   :widths: 24 13 13 18 14 18

   * - Task profile
     - ``njmax``
     - ``nconmax``
     - Friction starting point
     - Substeps
     - Other starting values
   * - Simple articulation or reach
     - ``50``
     - ``20``
     - pyramidal, ``impratio=1``
     - ``1``
     - ``update_data_interval=1``
   * - Legged locomotion
     - ``100``
     - ``40``
     - pyramidal, ``impratio=1``
     - ``1``
     - Reduce to measured demand
   * - Dexterous reorientation or handover
     - ``200``
     - ``70``
     - elliptic, ``impratio=10``
     - ``2``
     - ``update_data_interval=2``
   * - Dense manipulation or many objects
     - ``300``
     - ``200``
     - task-dependent
     - ``2``
     - Validate the contact pipeline

The maintained locomotion presets span roughly ``njmax=52`` to ``130`` and ``nconmax=10`` to
``40``. Cartpole uses only ``5`` and ``3``. Dexterous tasks use ``80`` to ``200`` constraint rows,
``70`` contacts, two substeps, and commonly an elliptic cone with ``impratio=10``. Dense
manipulation configurations use ``300`` and ``200``. These ranges explain the profiles; they are
not guarantees that a new task will fit.

Some Anymal locomotion presets use an elliptic cone with ``impratio=100``. Treat that as a
robot-specific tuned result, not a default for locomotion or grasping. Likewise, dense
manipulation configurations that set ``ls_iterations=15`` are reducing the class default of ``50``
for measured performance; ``15`` is not a higher-accuracy setting.

Tune parameters by function
---------------------------

Capacity: ``njmax`` and ``nconmax``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nconmax`` bounds contact points per environment. ``njmax`` bounds constraint rows per
environment; one contact can generate multiple rows, and joint limits, drives, equality
constraints, and mimic constraints also consume rows. Do not derive ``njmax`` directly from
``nconmax``.

Increase capacity first when the solver reports overflow, contacts disappear only in dense
states, equality constraints are missing, or behavior changes with the number of objects. Add
headroom above the measured worst case, including randomized geometry and resets. Once the full
distribution is stable, reduce capacity in a bounded performance sweep. Increasing either value
cannot repair wrong collision geometry, insufficient friction, poor convergence, or invalid
resets.

Integration: ``integrator`` and ``num_substeps``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start rigid articulated tasks with ``integrator="implicitfast"``. Use ``num_substeps=1`` for simple
tasks and most locomotion; test ``2`` for dexterous contact, high drive stiffness, light objects,
or impact-heavy motion. Each solver substep uses ``SimulationCfg.dt / num_substeps``. Substeps do
not change the policy period ``SimulationCfg.dt * env.decimation``.

Prefer adding a substep or reducing ``dt`` before making contact substantially harder. Compare
runtime and the same physical metrics at every candidate. Do not change policy decimation merely
to hide a solver instability.

Convergence: ``iterations``, ``ls_iterations``, and ``tolerance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep ``solver="newton"`` and the default ``iterations=100``, ``ls_iterations=50``, and
``tolerance=1e-6`` for the first validated baseline. If ``debug_mode`` shows environments reaching
the iteration cap or a reproducible residual or task metric improves with more work, sweep one cap
at a time. Stop when the metric plateaus. Lowering ``tolerance`` requests tighter convergence;
raising it permits earlier exit.

Do not raise iterations to compensate for bad inertia, reset penetration, missing collision
geometry, unsupported constraints, excessive drive stiffness, or undersized capacity. Conversely,
do not copy the ``ls_iterations=15`` performance choice from dense manipulation until the task
converges reliably with the larger default.

Friction: ``cone`` and ``impratio``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``cone="pyramidal"`` with ``impratio=1`` as the inexpensive baseline. Test
``cone="elliptic"`` for contact-rich manipulation or when physically valid friction and normal
force still produce slip. Elliptic cones are a closer friction representation but cost more.
``impratio`` controls frictional relative to normal constraint impedance; try ``10`` for a
validated grasp before considering larger values. Recheck convergence and runtime whenever the
cone changes because this changes the constraint formulation.

Do not use ``impratio`` to compensate for missing contacts, incorrect friction coefficients,
insufficient gripper effort, sparse collision geometry, or a controller that opens the grasp.

Contact path: ``use_mujoco_contacts`` and ``collision_cfg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep ``use_mujoco_contacts=True`` for the simplest baseline. This uses MuJoCo's internal collision
detection and forbids setting :attr:`~isaaclab_newton.physics.NewtonCfg.collision_cfg`.

Set ``use_mujoco_contacts=False`` when the task specifically needs Newton's collision pipeline,
for example non-convex meshes, SDF or hydroelastic contacts, or the maintained rough-terrain and
dense-manipulation patterns. Then configure
:class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg` on
:class:`~isaaclab_newton.physics.NewtonCfg`. The maintained rough-terrain presets use
``default_shape_cfg.margin=0.01`` and raise ``max_triangle_pairs`` from ``1_000_000`` to
``2_500_000``; the Lift pattern raises ``rigid_contact_max`` for its unusually dense scene.
Increase those capacities only in response to the matching overflow or missing-contact evidence.

``collision_decimation=0`` collides once per physics tick. A positive value re-collides every N
solver substeps and only matters when the Newton collision pipeline is active and
``num_substeps > 1``. Refresh contacts more often for fast-changing manipulation contacts before
reducing it for performance.

Collision convergence: ``ccd_iterations``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep the default ``35``. Increase it when GJK/EPA explicitly warns that
``opt.ccd_iterations`` is insufficient or a reproducible complex-geometry collision fails to
converge. Handover uses ``50`` for multi-finger geometry; a coupled cloth example uses ``100``.
This knob does not enable PhysX-style continuous collision detection and should not be raised for
unrelated contact softness or solver residuals.

State synchronization: ``update_data_interval``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep ``1`` while bringing up a task. Several two-substep humanoid and dexterous environments use
``2`` to reduce joint-state resynchronization frequency. Increase the interval only after resets,
Newton-side state writes, contact reporting, and sensors are verified. ``0`` disables updates
after initialization and is unsuitable for a normal migration unless the data flow has been
audited explicitly.

Diagnostic and model-conversion options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following settings are useful, but they are not general stability knobs:

* ``save_to_mjcf`` exports the converted model for inspecting joints, actuators, contacts, and
  resolved MuJoCo options.
* ``use_mujoco_cpu=True`` is a diagnostic comparison path, not the GPU training configuration.
* ``disable_contacts=True`` isolates whether a failure originates in contact; do not train the
  final contact task with it.
* ``default_actuator_gear`` and ``actuator_gears`` change the actuator transmission model. Source
  them from the mechanism or an identified model rather than tuning them as solver convergence
  parameters.
* ``use_cuda_graph=False`` can aid debugging but severely reduces performance.

Recommended tuning sequence
---------------------------

#. Reproduce one failure with a fixed initial state and command.
#. Validate units, mass, inertia, collision geometry, joint topology, drives, reset overlap, and
   supported MJWarp features.
#. Establish ``dt``, ``integrator="implicitfast"``, and a task-appropriate substep count.
#. Size ``nconmax`` and ``njmax``. If using Newton contacts, size the collision-pipeline buffers.
#. Enable ``debug_mode`` and sweep convergence settings only when the solver reaches its cap or a
   recorded residual or task metric justifies more work.
#. Tune the friction cone, ``impratio``, and contact material against penetration, slip, energy,
   success, and runtime metrics.
#. Optimize line searches, state-update frequency, collision refresh, capacities, and substeps
   only after behavior is acceptable.

See :doc:`migrating-assets-from-physx-to-newton` for the complete model and asset audit. For the
authoritative MuJoCo contact formulation, consult the `MuJoCo computation documentation
<https://mujoco.readthedocs.io/en/stable/computation/index.html#contact>`__ and `option reference
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-impratio>`__.
