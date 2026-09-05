.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-kamino-solver:
.. _kamino-solver-tuning:

Tune Kamino
===========

Kamino is a beta Newton solver selected through a task-exposed
``newton_kamino`` preset. This guide isolates and measures a reproducible
problem before changing its solver settings. The generated API documentation
for :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` and
:class:`~isaaclab_newton.physics.KaminoDVISolverCfg` remains authoritative for
current fields and defaults.

Prerequisites
-------------

Read :ref:`backends-and-presets` for preset semantics and
:ref:`hydra-backend-solver-presets` for task integration. Prepare the asset for
Newton with :doc:`/source/how-to/prepare_asset_for_newton`. Confirm that the
task exposes ``newton_kamino`` through its ``--help`` output or the
:doc:`environment catalog </source/setup/environments>`; do not infer
Kamino support from another Newton preset.

Start from an explicit baseline
-------------------------------

Run one small visual smoke test before training or benchmarking:

.. code-block:: bash

    uv run python scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 4 --viz newton physics=newton_kamino

Then fix the initial state, seed, action sequence, and reset distribution. For
that reproduction, record penetration, slip, energy, a task metric, a
convergence diagnostic, and runtime. Change one decision at a time so every
candidate can be compared with the same baseline.

Choose PADMM or DVI
-------------------

Choose PADMM first for contact-heavy tasks. DVI is a candidate for mechanisms
with relatively few active contacts. If a DVI task needs an explicit
:class:`~isaaclab_newton.physics.KaminoDynamicsCfg` override, disable
preconditioning; otherwise leave the dynamics override unset so Newton selects
the compatible defaults.

Validate reset and state consistency
------------------------------------

Validate root and joint state writes through the Isaac Lab asset APIs before
tuning the solver. Kamino's maximal-coordinate body poses must agree with the
reduced joint state after every reset. Investigate body-pose and joint-state
disagreement, including custom reset code and reset overlaps, before changing
convergence or stabilization settings.

Choose and size collision handling
----------------------------------

Compare Kamino's internal detector with Newton's collision pipeline using the
same full reset and contact distribution. ``use_collision_detector`` on the
Kamino solver selects the internal path, which
:class:`~isaaclab_newton.physics.KaminoCollisionDetectorCfg` configures; when
it is disabled, :class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg`
configures Newton's path. Size the active path only from observed capacity
failures, missing contacts, or warnings, and use those generated APIs for the
exact controls. Collision capacity cannot correct invalid geometry or reset
penetration.

Tune timestep and substeps
--------------------------

Hold the policy period and fixed reproduction constant. Only after the model
and reset are valid, compare a smaller simulation timestep or more
:attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps` for the same policy
period. Recheck the recorded physical, convergence, task, and runtime metrics;
do not change policy timing merely to hide an unstable simulation step.

Tune convergence and stabilization
----------------------------------

Tune :class:`~isaaclab_newton.physics.KaminoPADMMCfg` iteration and residual
controls only for a reproducible PADMM convergence failure. For DVI, tune
:class:`~isaaclab_newton.physics.KaminoDVICfg` alternating iterations and
tolerance only for the same kind of evidence. Change
:class:`~isaaclab_newton.physics.KaminoConstraintsCfg` stabilization controls
only after collision and reset validation. More solver work cannot repair
inconsistent state, invalid collision geometry, or reset overlap.

Optimize only after validation
------------------------------

Disable diagnostics before performance measurements. Retain only changes that
improve the recorded metrics, and separately measure sparse Jacobian, CUDA
graph, collision, substep, and iteration costs when each applies to the
selected solver and workload. For the parallel workflow, see :doc:`tune_mjwarp`;
it does not replace Kamino-specific validation or tuning.

The diagnose-first order is: validate the baseline, reset state, and collision
path; compare timestep and substeps; investigate convergence and stabilization;
then optimize measured costs.
