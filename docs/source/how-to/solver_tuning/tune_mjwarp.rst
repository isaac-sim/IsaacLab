.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _mjwarp-solver-tuning:

Tune MJWarp
===========

This how-to tunes the MuJoCo-Warp (MJWarp) solver after an asset and task can
be constructed by the Newton backend. The generated API documentation for
:class:`~isaaclab_newton.physics.NewtonCfg` and
:class:`~isaaclab_newton.physics.MJWarpSolverCfg` is authoritative for every
configuration field and its current default.

Prerequisites
-------------

First follow :doc:`/source/how-to/prepare_asset_for_newton` and reproduce one failure with a
fixed initial state, seed, and action sequence. Before changing solver
settings, check the mechanical model, collision geometry, reset overlap,
actuator limits, and unsupported features. A solver setting cannot correct an
invalid asset or controller.

Use maintained task configurations as evidence for a similar workload, not as
defaults to copy. For example, compare the small
`Cartpole configuration <https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env_cfg.py>`__
with the contact-rich
`Allegro Hand configuration <https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_tasks/isaaclab_tasks/core/reorient/config/allegro_hand/allegro_hand_common.py>`__.
Their values only apply after validating the new task's reset and contact
distribution.

Start from an explicit baseline
-------------------------------

Make the selected solver, integration method, timestep, substeps, and
diagnostics visible in the task configuration. This is an illustrative baseline;
measure and set contact capacity for the task rather than treating omitted or
checked-in values as universal.

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    solver_cfg = MJWarpSolverCfg(solver="newton", integrator="implicitfast")
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=True,
    )
    sim_cfg = SimulationCfg(dt=1 / 120, physics=newton_cfg)

Record the fixed-state behavior, warnings, contact count, penetration, slip,
task metric, and runtime before changing one variable at a time. Turn off
``debug_mode`` only after the solver budget has been validated for the
full reset, command, and randomization distribution.

Size contact and constraint capacity
------------------------------------

``nconmax`` bounds contact points per environment. ``njmax`` bounds constraint
rows per environment; contacts can consume multiple rows, and joint limits,
drives, equality constraints, and mimic constraints also consume rows. Do not
derive ``njmax`` directly from ``nconmax``.

Increase the relevant capacity when diagnostics show an overflow, contacts
disappear in dense states, or behavior changes with object count. Measure the
worst case across resets and randomized scenes, add task-specific headroom, and
then verify that the same fixed-state result remains stable. Capacity cannot fix
incorrect collision geometry, contact material, or convergence.

Choose timestep and substeps
----------------------------

Each solver substep uses ``SimulationCfg.dt / NewtonCfg.num_substeps``. Start
with a policy period that the controller can support, then compare a smaller
``dt`` or more substeps when high drive stiffness, light objects, impacts, or
dense contact cause instability. Substeps do not change the policy period
``SimulationCfg.dt * env.decimation``.

Use the same physical metrics and fixed-state reproduction for every candidate.
Do not change policy decimation merely to conceal an unstable physics step.

Validate convergence
--------------------

Keep ``NewtonCfg.debug_mode`` enabled while diagnosing. It reports
per-environment solver iteration statistics and warns when an environment
reaches the ``iterations`` cap. Only after the model, reset, contact path, and
capacities are valid should a cap hit or a reproducible metric justify sweeping
``iterations``, ``ls_iterations``, or ``tolerance``.

Sweep one convergence limit at a time and stop when the physical and task
metrics plateau. More iterations cannot compensate for bad inertia, penetration
at reset, missing collision geometry, unsupported constraints, or excessive
drive stiffness.

Tune friction and contact behavior
----------------------------------

Validate colliders, contact locations, normal force, material friction, and
``condim`` before changing global friction settings. ``cone`` selects the
MuJoCo friction-cone representation; test an elliptic cone when a physically
valid contact model still slips and the additional cost is warranted.
``impratio`` changes frictional impedance relative to normal impedance. Treat
both settings as a contact-formulation change, and recheck penetration, slip,
energy, convergence, task success, and runtime after each change.

Do not use ``impratio`` to mask missing contacts, incorrect material friction,
insufficient gripper effort, or a controller that opens the grasp. For the
underlying contact formulation, see the `MuJoCo contact documentation
<https://mujoco.readthedocs.io/en/stable/computation/index.html#contact>`__ and
`option reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-impratio>`__.

Choose the contact pipeline
---------------------------

Use ``use_mujoco_contacts=True`` for the simplest baseline. It selects
MuJoCo's internal collision detection and cannot be combined with
:attr:`~isaaclab_newton.physics.NewtonCfg.collision_cfg`.

Set ``use_mujoco_contacts=False`` only when the task needs Newton's collision
pipeline, such as for non-convex meshes, SDF or hydroelastic contacts. Configure
:class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg` on
:class:`~isaaclab_newton.physics.NewtonCfg`, then size its collision buffers
from observed overflow or missing-contact evidence. ``collision_decimation``
matters only for this pipeline and when more than one solver substep is used;
refresh contacts more often for fast-changing contacts before reducing work for
performance.

Optimize only after validation
------------------------------

After the task is stable over its full distribution, measure the cost of each
change and retain only work that improves the recorded outcome. Reduce excess
capacity, line-search work, collision refresh, substeps, and state
synchronization frequency one at a time.

``ccd_iterations`` is a GJK/EPA collision-convergence cap, not a PhysX-style
continuous-collision-detection switch. Increase it only for a warning or a
reproducible complex-geometry collision failure. Keep
``update_data_interval=1`` until resets, Newton-side state writes, contact
reporting, and sensors are verified; a larger interval reduces synchronization
work but can expose stale data. ``save_to_mjcf``, ``use_mujoco_cpu``, and
``disable_contacts`` are diagnostic tools, not production tuning targets.

The diagnose-first order is: validate the model and fixed reproduction; choose
timestep and substeps; size capacities; validate convergence; tune contact
behavior and the contact pipeline; then optimize measured costs.
