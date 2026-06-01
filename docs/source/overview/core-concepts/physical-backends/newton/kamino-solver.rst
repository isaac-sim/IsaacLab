.. _newton-kamino-solver:

Kamino Solver
=============

Kamino is a Newton solver, not a separate Isaac Lab physics backend. In Isaac Lab,
Kamino is enabled by selecting a :class:`~isaaclab_newton.physics.NewtonCfg` whose
``solver_cfg`` is :class:`~isaaclab_newton.physics.KaminoSolverCfg`.
This is usually exposed as a ``newton_kamino`` physics preset on the task configuration.

Kamino support is currently beta. A task that works with PhysX or with Newton's
MuJoCo-Warp solver may still need task-specific asset, collision, reset, and solver
tuning before it works well with Kamino.


Start from a Supported Newton Task
----------------------------------

Before adding Kamino, first make sure the task runs with the Newton backend:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 128 --viz newton physics=newton_mjwarp

Then run the same task with the Kamino preset if it is available:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 128 --viz newton physics=newton_kamino

At the time of writing, the ``newton_kamino`` preset is defined for
``Isaac-Cartpole-Direct``, ``Isaac-Ant-Direct-v0``, ``Isaac-Cartpole``,
and ``Isaac-Ant-v0``. Passing ``physics=newton_kamino`` to another task does not
automatically enable Kamino; the task must define and validate its own ``newton_kamino``
preset.


Add a Kamino Physics Preset
---------------------------

Tasks that support multiple physics options usually store ``SimulationCfg.physics``
as a :class:`~isaaclab_tasks.utils.hydra.PresetCfg`. First import the Newton
solver config types used by the presets:

.. code-block:: python

    from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg

Then add a ``newton_kamino`` entry beside the existing ``default``, ``physx``, and
``newton_mjwarp`` entries:

.. literalinclude:: ../../../../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env_cfg.py
    :language: python
    :start-at: class CartpolePhysicsCfg
    :end-at: ovphysx: OvPhysxCfg = OvPhysxCfg()
    :emphasize-lines: 16-38

The important pieces are:

* Add a ``newton_kamino`` preset whose value is :class:`~isaaclab_newton.physics.NewtonCfg`.
* Set ``solver_cfg=KaminoSolverCfg(...)`` inside that Newton config.
* Keep the preset at the same config path used by the task's
  :class:`~isaaclab.sim.SimulationCfg`, for example ``env.sim.physics``.

You can select the preset globally:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole physics=newton_kamino

or select the physics field directly:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole env.sim.physics=newton_kamino

Use the direct path override when only one task field should use the Kamino preset.
Use ``physics=newton_kamino`` when you want every matching preset field in the task config
to resolve to ``newton_kamino``.
Isaac Lab training scripts accept these Hydra overrides after the regular command
line flags; no separator is needed for the examples above.


Check Task and Asset Compatibility
----------------------------------

Kamino uses the Newton model built from the task assets. When adding Kamino to a
new task, validate the following before tuning solver parameters:

* The task must already be compatible with the Newton backend. If ``physics=newton_mjwarp``
  fails during model construction, fix the asset or task configuration first.
* The assets should use Newton-supported rigid bodies, articulations, and collision
  geometry. PhysX-only features, unsupported schemas, or missing collision shapes
  can prevent Newton model creation or produce unusable contacts.
* Reset logic should write consistent root and joint state through Isaac Lab asset
  APIs. Kamino uses a forward-kinematics reset path after state writes so maximal
  coordinate body poses match the reduced joint state.
* Sensor, renderer, and visualizer presets remain separate from the solver preset.
  Kamino can share the Newton-compatible sensors and renderers used by the task,
  but each sensor and renderer combination still needs its own validation.
* Contact-heavy tasks usually need their own collision mode, substep count, and
  P-ADMM iteration/tolerance settings. Start from the validated Cartpole or Ant
  preset that most closely resembles the task.

For a small articulated system with simple contacts, the Cartpole preset uses
Kamino's internal collision detector. For Ant, the preset uses Newton's collision
pipeline and two substeps. These choices are task-specific; treat them as starting
points rather than universal defaults.


Kamino Solver Parameters
------------------------

The following fields are specific to :class:`~isaaclab_newton.physics.KaminoSolverCfg`.
They are grouped by the part of the solver they affect.

Core Integration
^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``integrator``
      - Default: ``"euler"``. Time integration scheme. ``"moreau"`` is used by the validated Kamino task presets.
    * - ``use_fk_solver``
      - Default: ``True``. Enables Kamino's forward-kinematics solver for resets. Keep this enabled for Isaac Lab tasks unless you have a task-specific reset path.
    * - ``rotation_correction``
      - Default: ``"twopi"``. Rotation correction mode for maximal-coordinate bodies. Valid values are ``"twopi"``, ``"continuous"``, and ``"none"``.
    * - ``angular_velocity_damping``
      - Default: ``0.0``. Damps angular velocity. Higher values can suppress spin but also remove physical energy from the system.


Collision Handling
^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``use_collision_detector``
      - Default: ``False``. Selects Kamino's internal collision detector when ``True``. When ``False``, Isaac Lab uses Newton's collision pipeline for contact generation.
    * - ``collision_detector_pipeline``
      - Default: ``None``. Internal Kamino collision detector pipeline. Common values are ``"primitive"`` and ``"unified"``. Only used when ``use_collision_detector=True``.
    * - ``collision_detector_max_contacts_per_pair``
      - Default: ``None``. Maximum contacts generated per candidate geometry pair by the internal Kamino collision detector.
    * - ``constraints_delta``
      - Default: ``1.0e-6``. Contact penetration margin [m] used by Kamino constraint stabilization.


Constraint Stabilization
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``constraints_alpha``
      - Default: ``0.01``. Baumgarte stabilization for bilateral joint constraints. Increasing it can reduce joint constraint drift but may make the solve stiffer.
    * - ``constraints_beta``
      - Default: ``0.01``. Baumgarte stabilization for unilateral joint-limit constraints.
    * - ``constraints_gamma``
      - Default: ``0.01``. Baumgarte stabilization for unilateral contact constraints.


P-ADMM Solver Controls
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``padmm_max_iterations``
      - Default: ``200``. Maximum number of P-ADMM iterations per solver step. Higher values can improve convergence and increase runtime.
    * - ``padmm_primal_tolerance``
      - Default: ``1e-6``. Primal residual convergence tolerance.
    * - ``padmm_dual_tolerance``
      - Default: ``1e-6``. Dual residual convergence tolerance.
    * - ``padmm_compl_tolerance``
      - Default: ``1e-6``. Complementarity residual convergence tolerance for contacts and unilateral constraints.
    * - ``padmm_rho_0``
      - Default: ``1.0``. Initial P-ADMM penalty parameter. This influences how strongly constraint residuals are penalized early in the solve.
    * - ``padmm_eta``
      - Default: ``1e-5``. Proximal regularization parameter. It must be greater than zero.
    * - ``padmm_use_acceleration``
      - Default: ``True``. Enables acceleration in the P-ADMM iterations. This usually improves convergence but should be validated per task.
    * - ``padmm_warmstart_mode``
      - Default: ``"containers"``. Warm-start source for P-ADMM. Valid values are ``"none"``, ``"internal"``, and ``"containers"``.
    * - ``padmm_contact_warmstart_method``
      - Default: ``"key_and_position"``. Contact warm-start matching method. The validated presets use ``"geom_pair_net_force"``.
    * - ``padmm_use_graph_conditionals``
      - Default: ``True``. Uses CUDA graph conditional nodes for the iterative solver when ``True``. Setting it to ``False`` unrolls to fixed loops over the maximum iteration count.


Sparsity, Dynamics, and Debugging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``sparse_jacobian``
      - Default: ``False``. Uses sparse Jacobian computation. This is enabled in the validated Kamino task presets.
    * - ``sparse_dynamics``
      - Default: ``False``. Uses sparse dynamics computation.
    * - ``dynamics_preconditioning``
      - Default: ``True``. Enables preconditioning for constrained dynamics. Preconditioning can improve P-ADMM convergence.
    * - ``collect_solver_info``
      - Default: ``False``. Collects solver convergence and performance information. Enable only for debugging because it significantly increases runtime.
    * - ``compute_solution_metrics``
      - Default: ``False``. Computes solution metrics at each step. Enable only for debugging because it significantly increases runtime.


Tuning Workflow
---------------

Use the following sequence when bringing up a new Kamino task:

1. Run the task with ``physics=newton_mjwarp`` and fix Newton model construction or task
   compatibility issues first.
2. Add a ``newton_kamino`` preset with conservative values copied from the closest
   validated task.
3. Run a small smoke test with a low environment count and a visualizer.
4. Increase ``num_envs`` and profile only after the task is stable.
5. Tune ``num_substeps``, ``padmm_max_iterations``, and the P-ADMM tolerances
   together. Raising iteration count without checking tolerances can hide a
   poorly scaled constraint setup.
6. Enable ``collect_solver_info`` or ``compute_solution_metrics`` only while
   debugging convergence. Disable them for training and benchmarks.
