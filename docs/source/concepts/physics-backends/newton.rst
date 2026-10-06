.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-backend:

Newton
======

Newton is a Warp-native backend that can run without Isaac Sim. Its Isaac Lab
integration is beta and supports multiple solver families. Configure the
backend with :class:`~isaaclab_newton.physics.NewtonCfg` and choose a solver
configuration through its ``solver_cfg`` field.

Task and component coverage is task-specific. Check task ``--help`` and
:doc:`the environment catalog </source/setup/environments>` for current
presets; the presence of a solver configuration does not imply that a task
supports it. Start with :doc:`/source/how-to/prepare_asset_for_newton` before
tuning a new asset or task.

Solver options
--------------

The links below cover Isaac Lab configuration and workflows. For solver
formulations, feature support, and contact models, use Newton's
`solver guide <https://newton-physics.github.io/newton/latest/solvers/index.html>`__
and its linked solver references. The generated Isaac Lab APIs remain the
source of truth for configuration fields and defaults.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Isaac Lab configuration
     - Workflow
   * - :class:`~isaaclab_newton.physics.MJWarpSolverCfg`
     - :ref:`mjwarp-solver-tuning` for the primary validated solver path
   * - :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` and
       :class:`~isaaclab_newton.physics.KaminoDVISolverCfg`
     - :ref:`kamino-solver-tuning` for the beta Kamino path
   * - :class:`~isaaclab_newton.physics.VBDSolverCfg`
     - :ref:`newton-using-vbd` for cloth, soft bodies, and coupled scenes
   * - :class:`~isaaclab_newton.physics.MPMSolverCfg`
     - :ref:`newton-using-mpm` for scene construction and
       :ref:`newton-tuning-mpm` for parameter studies

Additional solver configurations are
:class:`~isaaclab_newton.physics.XPBDSolverCfg` and
:class:`~isaaclab_newton.physics.FeatherstoneSolverCfg`. Use their generated
API references and the Newton solver guide when evaluating a task-specific
preset.

Related workflows
-----------------

Use :doc:`/source/how-to/transfer_policies_between_physx_and_newton` to
validate a policy across backends. Experimental specialist guides cover
:ref:`deformables`, :ref:`warp-environments`, and :ref:`warp-env-migration`.
Backend developers can also read
:doc:`/source/developer-tools/extending_newton_solvers`.
