.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _solver-tuning:

Solver Tuning
=============

Use these guides after the selected backend can construct the task and its
assets. Reproduce one physical or task-level problem with a fixed initial
state, seed, and action sequence before changing solver settings. Exact fields
and defaults remain in the generated API reference.


SemiImplicit
------------

Select Newton's maximal-coordinate semi-implicit solver with
:class:`~isaaclab_newton.physics.SemiImplicitSolverCfg`:

.. code-block:: python

   from isaaclab.sim import SimulationCfg
   from isaaclab_newton.physics import NewtonCfg, SemiImplicitSolverCfg

   sim_cfg = SimulationCfg(
       dt=1.0 / 120.0,
       physics=NewtonCfg(
           solver_cfg=SemiImplicitSolverCfg(),
           num_substeps=4,
       ),
   )

:class:`~isaaclab_newton.physics.NewtonSemiImplicitManager` uses the shared
Newton collision, external-force, reset, double-buffer, and CUDA-graph
pipelines. The solver configuration exposes Newton's constructor settings:
``angular_damping=0.05`` [1/s], ``friction_smoothing=1.0`` [m/s],
``joint_attach_ke=1.0e4`` [N/m for translation, N m/rad for rotation],
``joint_attach_kd=1.0e2`` [N s/m for translation, N m s/rad for rotation],
and ``enable_tri_contact=True``.

This solver advances maximal-coordinate body state. It supports prismatic,
revolute, ball, fixed, free, distance, and D6 joints. It does not support rod
joints, equality or mimic constraints, joint armature, joint friction, effort
or velocity limits, or joint target mode; ball-joint limits and targets are not
enforced. Its semi-implicit integration is not unconditionally stable, so
reduce the timestep or increase :attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps`
when stiffness or damping makes a scene unstable. Switching from another solver
usually requires task-specific retuning and validation.

Set :attr:`~isaaclab_newton.physics.NewtonCfg.deterministic` or
``deterministic_mode`` when the task requires Newton's deterministic execution.
Isaac Lab forwards the resulting Warp deterministic mode to
``SolverSemiImplicit`` and to its collision pipeline.

.. toctree::
   :maxdepth: 1

   tune_physx
   tune_mjwarp
   tune_kamino
   tune_vbd
