.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-kamino-solver:

Enable and Tune Kamino
======================

Kamino is a Newton solver, not a separate Isaac Lab physics backend. Enable it
by selecting a :class:`~isaaclab_newton.physics.NewtonCfg` whose ``solver_cfg``
is :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` or
:class:`~isaaclab_newton.physics.KaminoDVISolverCfg`. Tasks normally expose
this configuration as a ``newton_kamino`` physics preset.

Kamino support is beta. The generated API documentation for
:class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg`,
:class:`~isaaclab_newton.physics.KaminoDVISolverCfg`, and their nested
:class:`~isaaclab_newton.physics.KaminoPADMMCfg`,
:class:`~isaaclab_newton.physics.KaminoDVICfg`,
:class:`~isaaclab_newton.physics.KaminoDynamicsCfg`,
:class:`~isaaclab_newton.physics.KaminoConstraintsCfg`,
:class:`~isaaclab_newton.physics.KaminoFKCfg`,
:class:`~isaaclab_newton.physics.KaminoCollisionDetectorCfg`, and
:class:`~isaaclab_newton.physics.KaminoMaterialsCfg` is the field reference
for current configuration options and defaults.

Prerequisites
-------------

First understand :ref:`backends-and-presets` and prepare the asset for Newton
with :doc:`/source/how-to/prepare_asset_for_newton`. Start with a task that already runs with
``physics=newton_mjwarp``. A task supports Kamino only when it exposes a
``newton_kamino`` preset; use the :doc:`environment catalog
</source/overview/environments>` or the task's ``--help`` output to discover
available presets. Do not assume that a task supporting another Newton solver
also supports Kamino.

Run one small visual smoke test before training or benchmarking:

.. code-block:: bash

    uv run python scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 128 --viz newton physics=newton_kamino

Choose PADMM or DVI
-------------------

Choose :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` for robust
proximal ADMM on contact-heavy tasks. Choose
:class:`~isaaclab_newton.physics.KaminoDVISolverCfg` for faster projected-dual
iterations on mechanisms with relatively few active contacts. DVI requires
preconditioning to be disabled; leave ``dynamics`` unset unless the task needs
an explicit :class:`~isaaclab_newton.physics.KaminoDynamicsCfg` override, and
then set ``preconditioning=False``.

Add a Kamino physics preset
---------------------------

Import :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` and
:class:`~isaaclab_newton.physics.NewtonCfg`, then add this field to the task's
existing :class:`~isaaclab_tasks.utils.hydra.PresetCfg` subclass. Keep the
existing alternatives, especially its ``default`` field, and keep the preset at
the same configuration path used by :class:`~isaaclab.sim.SimulationCfg`, such
as ``env.sim.physics``:

.. code-block:: python

    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )

Then select ``physics=newton_kamino`` to apply matching physics presets. Use a
targeted override such as ``env.sim.physics=newton_kamino`` when only that
configuration field should change. See :ref:`backends-and-presets` for preset
resolution and override precedence.

Validate task and asset compatibility
-------------------------------------

Before tuning solver settings, verify that the task builds a Newton model with
MJWarp, and then check the Kamino smoke test across representative reset states
and actions. Confirm that assets use Newton-supported rigid bodies,
articulations, and collision geometry; PhysX-only features, unsupported
schemas, and missing collision shapes can prevent model construction or yield
unusable contacts.

Validate reset logic through the Isaac Lab asset APIs. Kamino uses a
forward-kinematics reset path so maximal-coordinate body poses agree with the
reduced joint state after a state write. Custom reset code can break this
assumption by writing inconsistent root or joint state. Sensors, renderers, and
visualizers are separate presets and each selected combination needs its own
validation.

Choose collision handling
-------------------------

Set ``use_collision_detector=True`` to use Kamino's internal collision detector.
This is a useful starting point for small articulated systems with simple
contacts. Leave it ``False`` to use Newton's collision pipeline, then configure
:class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg` on
:class:`~isaaclab_newton.physics.NewtonCfg`. Prefer the pipeline when task
assets or contact behavior need its capabilities, and validate the chosen path
with the task's full reset and contact distribution.

The generated API references for
:class:`~isaaclab_newton.physics.KaminoCollisionDetectorCfg` and
:class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg` describe the
respective collision controls and capacity settings.

Tune and validate convergence
-----------------------------

Use the same diagnose-first order as :doc:`tune_mjwarp`: validate the asset,
reset, and fixed reproduction; choose timestep and substeps; validate contact
capacity and collision handling; then tune convergence. Start from the closest
maintained Kamino preset, record penetration, slip, energy, task metrics, and
runtime, and change one variable at a time.

For PADMM, sweep iteration limits and residual tolerances together only when a
reproducible convergence problem remains after model and contact validation.
For DVI, sweep its alternating iterations and tolerance against the same fixed
reproduction. Use solver diagnostics only while investigating convergence, then
disable them before training or benchmarking. Increasing iterations cannot fix
invalid collision geometry, reset overlap, unsupported constraints, or excessive
drive stiffness.
