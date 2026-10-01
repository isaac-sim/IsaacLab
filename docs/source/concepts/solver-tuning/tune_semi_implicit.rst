.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _semi-implicit-solver-tuning:

Tune SemiImplicit
=================

This guide covers Newton's maximal-coordinate semi-implicit solver. The
generated API documentation for :class:`~isaaclab_newton.physics.NewtonCfg` and
:class:`~isaaclab_newton.physics.SemiImplicitSolverCfg` is authoritative for
every configuration field and its current default.

Select the solver
-----------------

Select the solver through the Newton physics configuration:

.. code-block:: python

   from isaaclab.sim import SimulationCfg
   from isaaclab_newton.physics import NewtonCfg, SemiImplicitSolverCfg

   sim_cfg = SimulationCfg(
       dt=1.0 / 120.0,
       physics=NewtonCfg(solver_cfg=SemiImplicitSolverCfg(), num_substeps=4),
   )

:class:`~isaaclab_newton.physics.NewtonSemiImplicitManager` uses the shared
Newton collision, external-force, reset, double-buffer, and CUDA-graph
pipelines. The solver integrates body state; the manager updates joint
coordinates after every substep so root and joint data stay consistent.

Check supported features
------------------------

The solver supports prismatic, revolute, ball, fixed, free, distance, and D6
joints. It does not support rod joints, equality or mimic constraints, joint
armature, joint friction, effort or velocity limits, or joint target mode.
Ball-joint limits and targets are not enforced. Check that the task does not
rely on these features before tuning; switching from another solver usually
requires task-specific retuning and validation.

Tune stability
--------------

Semi-implicit integration is not unconditionally stable. When stiffness or
damping makes a scene unstable, reduce the time step or increase
:attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps` before changing other
settings. Joints are held together by attachment springs:
:attr:`~isaaclab_newton.physics.SemiImplicitSolverCfg.joint_attach_ke` and
:attr:`~isaaclab_newton.physics.SemiImplicitSolverCfg.joint_attach_kd` trade
joint drift against the time step the scene requires.

Set :attr:`~isaaclab_newton.physics.NewtonCfg.deterministic` or
:attr:`~isaaclab_newton.physics.NewtonCfg.deterministic_mode` when the task
requires deterministic execution; Isaac Lab forwards the mode to the solver and
its collision pipeline.
