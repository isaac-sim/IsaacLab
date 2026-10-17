.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _physics-backends:

Physics Backends
================

Isaac Lab supports PhysX, Newton, and OvPhysX through shared asset, sensor, and
scene APIs. Choose a backend for the runtime and solvers you need, then select a
preset supported by your task. See :ref:`backends-and-presets` for launch commands.


Choose a backend
----------------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: IsaacSim PhysX
      :link: /source/concepts/physics-backends/physx
      :link-type: doc

      PhysX simulation through Isaac Sim and Omniverse Kit, with TGS and PGS
      solvers.

   .. grid-item-card:: Newton
      :link: /source/concepts/physics-backends/newton
      :link-type: doc

      Warp-native simulation without Isaac Sim. Choose from MJWarp, Kamino,
      VBD, and MPM solvers.

   .. grid-item-card:: OvPhysX
      :link: /source/concepts/physics-backends/ovphysx
      :link-type: doc

      PhysX simulation without Omniverse Kit, configured through USD. Install
      with the ``ovphysx`` extra.

Follow :ref:`isaaclab-installation-root` for the supported installation paths
and current optional extras. Do not combine OvPhysX with a Kit runtime in the
same process.

For the general PhysX--Newton policy validation workflow, see
:doc:`/source/how-to/transfer_policies_between_physx_and_newton`. When a
checkpoint's joint or body ordering differs from the target backend, use the
:ref:`articulation ordering guide <joint-and-body-ordering>`.


.. _physics-backends-physx:

IsaacSim PhysX
--------------

IsaacSim PhysX runs through Isaac Sim and Omniverse Kit. It uses the Temporal
Gauss-Seidel (TGS) solver by default, with Projective Gauss-Seidel (PGS) also
available. Configure scene settings through :class:`~isaaclab_physx.physics.PhysxCfg`
and asset properties through :doc:`schema_cfgs`. Large or contact-rich scenes may
need additional GPU buffer capacity.

See :doc:`the IsaacSim PhysX guide <physics-backends/physx>` for runtime details
and :ref:`physx-solver-tuning` for solver and buffer settings.


.. _physics-backends-newton:

Newton
------

Newton runs on Warp without requiring Isaac Sim. Its solvers cover different
simulation needs: MJWarp and Kamino for rigid robots, VBD for cloth and soft bodies,
and MPM for particle materials. Configure the backend with
:class:`~isaaclab_newton.physics.NewtonCfg` and select a solver through
``solver_cfg``. Native Warp arrays are available for custom simulation code.

See :doc:`the Newton guide <physics-backends/newton>` for solver choices, tuning
guides, and native data access.


.. _physics-backends-ovphysx:

OvPhysX
--------

OvPhysX runs PhysX without Isaac Sim or Omniverse Kit. It supports rigid objects,
articulations, contact and motion sensors, and CUDA-based deformables, and can
pair with OVRTX for rendering. Scene-level solver settings come primarily from
USD's ``PhysicsScene``; :class:`~isaaclab_ov.physics.OvPhysxCfg` holds additional
settings such as GPU buffer capacities. It cannot share a process with Kit.

See :doc:`the OvPhysX guide <physics-backends/ovphysx>` for installation, supported
features, and their configuration limits.


Find supported tasks and APIs
-----------------------------

Backend support is task-specific. Use :doc:`/source/isaaclab/environments` to
browse registered environments and presets, or ask a task for its current
choices:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct --help

The help output is the source of truth for that task's ``physics=`` choices.
For native solver data and views, see
:doc:`/source/concepts/native-physics-api/index`. For field-level configuration,
follow the generated API links for :class:`~isaaclab_physx.physics.PhysxCfg`,
:class:`~isaaclab_newton.physics.NewtonCfg`, its solver configurations, and
:class:`~isaaclab_ov.physics.OvPhysxCfg`.


.. toctree::
   :hidden:

   /source/concepts/physics-backends/physx
   /source/concepts/physics-backends/newton
   /source/concepts/physics-backends/ovphysx
