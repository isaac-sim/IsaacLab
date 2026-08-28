.. _newton-kamino-solver:

Kamino Solver
=============

Kamino is a Newton solver, not a separate Isaac Lab physics backend. In Isaac Lab,
Kamino is enabled by selecting a :class:`~isaaclab_newton.physics.NewtonCfg` whose
``solver_cfg`` is :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` or
:class:`~isaaclab_newton.physics.KaminoDVISolverCfg`.
This is usually exposed as a ``newton_kamino`` physics preset on the task configuration.

Kamino support is currently beta. A task that works with PhysX or with Newton's
MuJoCo-Warp solver may still need task-specific asset, collision, reset, and solver
tuning before it works well with Kamino.


Start from a Supported Newton Task
----------------------------------

Before adding Kamino, first make sure the task runs with the Newton backend:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run python scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 128 --viz newton physics=newton_mjwarp

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 128 --viz newton physics=newton_mjwarp

Then run the same task with the Kamino preset if it is available:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run python scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 128 --viz newton physics=newton_kamino

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 128 --viz newton physics=newton_kamino

At the time of writing, the ``newton_kamino`` preset is defined for
``Isaac-Cartpole-Direct``, ``Isaac-Ant-Direct``, ``Isaac-Cartpole``,
``Isaac-Pendulum-MARL-Direct``, ``Isaac-Ant``, and several locomotion tasks. Passing
``physics=newton_kamino`` to another task does not automatically enable Kamino;
the task must define and validate its own ``newton_kamino`` preset.


Add a Kamino Physics Preset
---------------------------

Tasks that support multiple physics options usually store ``SimulationCfg.physics``
as a :class:`~isaaclab_tasks.utils.hydra.PresetCfg`. First import the Newton
solver config types used by the presets:

.. code-block:: python

    from isaaclab_newton.physics import (
        KaminoCollisionDetectorCfg,
        KaminoDVISolverCfg,
        KaminoPADMMSolverCfg,
        MJWarpSolverCfg,
        NewtonCfg,
    )

Then add a ``newton_kamino`` entry beside the existing ``default``, ``physx``, and
``newton_mjwarp`` entries:

.. literalinclude:: ../../../../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env_cfg.py
    :language: python
    :start-at: class CartpolePhysicsCfg
    :end-before: class CartpoleEnvCfg
    :emphasize-lines: 18-25

The important pieces are:

* Add a ``newton_kamino`` preset whose value is :class:`~isaaclab_newton.physics.NewtonCfg`.
* Construct a :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` or
  :class:`~isaaclab_newton.physics.KaminoDVISolverCfg`.
* Keep the preset at the same config path used by the task's
  :class:`~isaaclab.sim.SimulationCfg`, for example ``env.sim.physics``.

Choosing PADMM vs DVI
---------------------

Kamino exposes two concrete forward-dynamics solver configurations:

* :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg`: robust proximal ADMM;
  recommended for contact-heavy tasks.
* :class:`~isaaclab_newton.physics.KaminoDVISolverCfg`: faster projected-dual
  iterations; best for mechanisms with relatively few active contacts. It defaults
  ``dynamics.preconditioning`` to ``False``.

Construct the concrete solver configuration directly:

.. code-block:: python

    from isaaclab_newton.physics import KaminoDVISolverCfg, KaminoPADMMSolverCfg

    newton_kamino = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(use_collision_detector=True))
    newton_kamino_dvi = NewtonCfg(solver_cfg=KaminoDVISolverCfg(integrator="moreau"))

You can select the preset globally:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task=Isaac-Cartpole physics=newton_kamino

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Cartpole physics=newton_kamino

or select the physics field directly:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task=Isaac-Cartpole env.sim.physics=newton_kamino

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Cartpole env.sim.physics=newton_kamino

Use the direct path override when only one task field should use the Kamino preset.
Use ``physics=newton_kamino`` when you want every matching preset field in the task config
to resolve to ``newton_kamino``.
Isaac Lab training commands accept these Hydra overrides after the regular command
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

The following fields are shared by
:class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` and
:class:`~isaaclab_newton.physics.KaminoDVISolverCfg`. They are grouped by the part
of the solver they affect.

Core Integration
^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``integrator``
      - Default: ``"moreau"``. Time integration scheme. Use ``"euler"`` for explicit Euler integration.
    * - ``use_fk_solver``
      - Default: ``None`` (auto). Enables Kamino's forward-kinematics solver for resets when required.
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
    * - ``collision_detector.pipeline``
      - Default: ``None``. Internal Kamino collision detector pipeline. Common values are ``"primitive"`` and ``"unified"``. ``None`` uses Newton's default (``"unified"``). Only used when ``use_collision_detector=True``.
    * - ``collision_detector.broadphase``
      - Default: ``None``. Broad-phase algorithm. ``None`` uses Newton's default.
    * - ``collision_detector.bvtype``
      - Default: ``None``. Bounding-volume type. ``None`` uses Newton's default.
    * - ``collision_detector.max_contacts``
      - Default: ``None``. Model-wide contact buffer capacity cap.
    * - ``collision_detector.max_contacts_per_world``
      - Default: ``None``. Per-world contact buffer capacity override.
    * - ``collision_detector.max_contacts_per_pair``
      - Default: ``None``. Maximum contacts generated per candidate geometry pair by the internal Kamino collision detector.
    * - ``collision_detector.max_triangle_pairs``
      - Default: ``None``. Maximum triangle-primitive shape pairs in narrow phase.
    * - ``collision_detector.default_gap``
      - Default: ``None``. Default detection gap [m] applied as a floor to per-geometry gaps.
    * - ``max_contacts_per_world``
      - Default: ``None``. Caps per-world contact pre-allocation passed to Kamino. When ``None``, Kamino falls back to the collision pipeline default, which can over-allocate for contact-rich assets.
    * - ``constraints.delta``
      - Default: ``1.0e-6``. Contact penetration margin [m] used by Kamino constraint stabilization.


Constraint Stabilization
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``constraints.alpha``
      - Default: ``0.1``. Baumgarte stabilization for bilateral joint constraints. Increasing it can reduce joint constraint drift but may make the solve stiffer.
    * - ``constraints.beta``
      - Default: ``0.01``. Baumgarte stabilization for unilateral joint-limit constraints.
    * - ``constraints.gamma``
      - Default: ``0.01``. Baumgarte stabilization for unilateral contact constraints.


P-ADMM Solver Controls
^^^^^^^^^^^^^^^^^^^^^^

Configured through :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` at
``solver_cfg.dynamics_solver_cfg``.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``solver_cfg.dynamics_solver_cfg.max_iterations``
      - Default: ``100``. Maximum number of P-ADMM iterations per solver step. Higher values can improve convergence and increase runtime.
    * - ``solver_cfg.dynamics_solver_cfg.primal_tolerance``
      - Default: ``1e-4``. Primal residual convergence tolerance.
    * - ``solver_cfg.dynamics_solver_cfg.dual_tolerance``
      - Default: ``1e-4``. Dual residual convergence tolerance.
    * - ``solver_cfg.dynamics_solver_cfg.compl_tolerance``
      - Default: ``1e-4``. Complementarity residual convergence tolerance for contacts and unilateral constraints.
    * - ``solver_cfg.dynamics_solver_cfg.restart_tolerance``
      - Default: ``0.999``. Combined primal-dual residual tolerance for acceleration restarts.
    * - ``solver_cfg.dynamics_solver_cfg.rho_0``
      - Default: ``0.05``. Initial P-ADMM penalty parameter. This influences how strongly constraint residuals are penalized early in the solve.
    * - ``solver_cfg.dynamics_solver_cfg.rho_min``
      - Default: ``1e-5``. Lower bound on the penalty parameter.
    * - ``solver_cfg.dynamics_solver_cfg.a_0``
      - Default: ``1.0``. Initial acceleration parameter.
    * - ``solver_cfg.dynamics_solver_cfg.alpha``
      - Default: ``10.0``. Primal-dual residual threshold for penalty updates.
    * - ``solver_cfg.dynamics_solver_cfg.tau``
      - Default: ``1.5``. Penalty increase/decrease factor.
    * - ``solver_cfg.dynamics_solver_cfg.eta``
      - Default: ``1e-5``. Proximal regularization parameter. It must be greater than zero.
    * - ``solver_cfg.dynamics_solver_cfg.penalty_update_freq``
      - Default: ``1``. Frequency of penalty updates. Zero disables updates.
    * - ``solver_cfg.dynamics_solver_cfg.penalty_update_method``
      - Default: ``"fixed"``. Penalty update method. Valid values are ``"fixed"`` and ``"balanced"``.
    * - ``solver_cfg.dynamics_solver_cfg.linear_solver_tolerance``
      - Default: ``0.0``. Absolute tolerance for the iterative linear solver. Zero leaves it unchanged.
    * - ``solver_cfg.dynamics_solver_cfg.linear_solver_tolerance_ratio``
      - Default: ``0.0``. Ratio adapting the linear solver tolerance from the ADMM primal residual.
    * - ``solver_cfg.dynamics_solver_cfg.use_acceleration``
      - Default: ``True``. Enables acceleration in the P-ADMM iterations. This usually improves convergence but should be validated per task.
    * - ``solver_cfg.dynamics_solver_cfg.warmstart_mode``
      - Default: ``"containers"``. Warm-start source for P-ADMM. Valid values are ``"none"``, ``"internal"``, and ``"containers"``.
    * - ``solver_cfg.dynamics_solver_cfg.contact_warmstart_method``
      - Default: ``"geom_pair_net_force"``. Contact warm-start matching method.
    * - ``solver_cfg.dynamics_solver_cfg.use_graph_conditionals``
      - Default: ``False``. Uses CUDA graph conditional nodes for the iterative solver when ``True``. Setting it to ``False`` unrolls to fixed loops over the maximum iteration count.


DVI Solver Controls
^^^^^^^^^^^^^^^^^^^

Configured through :class:`~isaaclab_newton.physics.KaminoDVISolverCfg` at
``solver_cfg.dynamics_solver_cfg``.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``solver_cfg.dynamics_solver_cfg.max_alternating_iterations``
      - Default: ``20``. Maximum outer DVI iterations.
    * - ``solver_cfg.dynamics_solver_cfg.tolerance``
      - Default: ``1e-5``. Convergence tolerance on the projected update size.
    * - ``solver_cfg.dynamics_solver_cfg.regularization``
      - Default: ``1e-6``. Diagonal regularization added to each projected update denominator.
    * - ``solver_cfg.dynamics_solver_cfg.omega``
      - Default: ``1.0``. Relaxation factor applied to projected Gauss-Seidel updates.
    * - ``solver_cfg.dynamics_solver_cfg.inequality_sweeps_per_iteration``
      - Default: ``1``. Projected Gauss-Seidel sweeps per DVI iteration.
    * - ``solver_cfg.dynamics_solver_cfg.bilateral_solve_interval``
      - Default: ``1``. DVI iterations between repeated direct bilateral solves.
    * - ``solver_cfg.dynamics_solver_cfg.bilateral_solver_type``
      - Default: ``"LLTB"``. Direct linear solver for bilateral constraints. Use ``"LLTBRCM"`` for large sparse systems.
    * - ``solver_cfg.dynamics_solver_cfg.warmstart_mode``
      - Default: ``"containers"``. Warm-start source for DVI. Valid values are ``"none"``, ``"internal"``, and ``"containers"``.
    * - ``solver_cfg.dynamics_solver_cfg.contact_warmstart_method``
      - Default: ``"key_and_position_with_net_force_backup"``. Contact warm-start method for container warm-starts.


Forward Kinematics Reset
^^^^^^^^^^^^^^^^^^^^^^^^

Configured through :class:`~isaaclab_newton.physics.KaminoFKCfg` at ``solver_cfg.fk``.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``fk.use_regularization``
      - Default: ``True``. Regularizes the FK reset solve with a Tikhonov term on body poses.
    * - ``fk.regularization_weight``
      - Default: ``1e-5``. Weight of the FK reset regularizer when ``fk.use_regularization=True``.
    * - ``fk.tolerance``
      - Default: ``1e-5``. Convergence tolerance of the FK reset solve.


Material Mixing
^^^^^^^^^^^^^^^

Configured through :class:`~isaaclab_newton.physics.KaminoMaterialsCfg` at ``solver_cfg.materials``.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``materials.friction_mix_mode``
      - Default: ``"average"``. How friction coefficients are mixed for a contact pair. Valid values are ``"average"``, ``"multiply"``, ``"max"``, and ``"min"``.
    * - ``materials.restitution_mix_mode``
      - Default: ``"min"``. How restitution coefficients are mixed for a contact pair.


Sparsity, Dynamics, and Debugging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``sparse_jacobian``
      - Default: ``None``. Uses sparse Jacobian computation. ``None`` lets Newton pick per backend.
    * - ``sparse_dynamics``
      - Default: ``False``. Uses sparse dynamics computation.
    * - ``dynamics.preconditioning``
      - Default: ``True``. Enables preconditioning for constrained dynamics. Must be ``False`` for DVI.
    * - ``dynamics.linear_solver_type``
      - Default: ``"LLTB"``. Linear solver for the dynamics problem. The DVI config defaults it to ``"LLTBRCM"``.
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
5. Tune ``num_substeps``, ``padmm.max_iterations``, and the P-ADMM tolerances
   together. Raising iteration count without checking tolerances can hide a
   poorly scaled constraint setup.
6. Enable ``collect_solver_info`` or ``compute_solution_metrics`` only while
   debugging convergence. Disable them for training and benchmarks.
