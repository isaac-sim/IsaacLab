.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _physics-backends:

Physics Backends
================

Isaac Lab supports three physics backends behind its common asset, sensor, and
scene APIs. Choose the backend that matches the runtime and solver family you
need, then choose a preset that the target task exposes. See
:ref:`backends-and-presets` for preset selection and
:ref:`backend-architecture` for the implementation architecture.


Backend and solver terminology
------------------------------

A **backend** integrates a physics runtime with Isaac Lab and owns simulation
lifecycle and data exchange. A **solver** is the numerical method that advances
the physical state within that backend. Newton is therefore not synonymous
with MJWarp: Newton hosts MJWarp, Kamino, VBD, MPM, and other solver paths.

A **physics preset** is a task-supported configuration that selects a backend
and, when applicable, one of its solvers. Solver settings and behavior are not
numerically portable between solver families. Expect to retune a task when
changing solvers; see :ref:`solver-differences`.


Choose a backend
----------------

.. list-table::
   :header-rows: 1
   :widths: 13 23 13 21 30

   * - Backend
     - Runtime requirement
     - Maturity
     - Solver family
     - Intended use
   * - PhysX
     - Isaac Sim and Omniverse Kit
     - Established reference
     - PhysX TGS by default; PGS available
     - Broadly validated Isaac Lab workflows and behavior-parity reference
   * - Newton
     - Warp-native; Isaac Sim is not required for kit-less execution
     - Beta
     - Multiple solvers, with MJWarp as the primary validated path
     - Kit-less simulation and workflows that need Newton solver choices
   * - OvPhysX
     - Optional ``ovphysx`` runtime extra; no Omniverse Kit
     - Experimental
     - PhysX, configured primarily through USD
     - Experimental kit-less PhysX workflows

Follow :ref:`isaaclab-installation-root` for the supported installation paths
and current optional extras. Do not combine OvPhysX with a Kit runtime in the
same process.

For the general PhysX--Newton policy validation workflow, see
:doc:`/source/how-to/transfer_policies_between_physx_and_newton`. When a
checkpoint's joint or body ordering differs from the target backend, use the
:doc:`articulation ordering guide
</source/overview/core-concepts/physical-backends/sim-to-sim-policy-transfer>`.


.. _physics-backends-physx:

PhysX
-----

PhysX is Isaac Lab's established reference backend. It runs through Isaac Sim,
uses the Temporal Gauss-Seidel (TGS) solver by default, and also offers the
Projective Gauss-Seidel (PGS) solver. Its GPU scene capacities are static, so
contact-rich or large vectorized scenes may require explicit buffer sizing.
Keep TGS as the starting point for articulated robots; evaluate PGS when a
stiff legacy asset behaves poorly with TGS. The generated API remains the
source of truth for solver selection values and defaults.

Configure scene-level fields with
:class:`~isaaclab_physx.physics.PhysxCfg`; the generated API reference is the
authoritative field list. Per-actor physical properties remain on the USD
schema configuration described in
:doc:`/source/overview/core-concepts/schema_cfgs`.
For a diagnose-first workflow covering solver selection, iterations, contacts,
stability, and GPU capacities, see :ref:`physx-solver-tuning`.


.. _physics-backends-newton:

Newton
------

Newton is a Warp-native backend that can run without Isaac Sim. Its Isaac Lab
integration is beta and supports multiple solver families rather than one
fixed solver. Configure the backend with
:class:`~isaaclab_newton.physics.NewtonCfg` and select a solver configuration
such as :class:`~isaaclab_newton.physics.MJWarpSolverCfg`.

Task and component coverage is narrower and task-specific. Check task
``--help`` and :doc:`the environment catalog </source/overview/environments>`
for current presets, the generated configuration APIs for available symbols,
and the specialist guides below for solver-specific workflows and limitations.

Start with :doc:`/source/how-to/prepare_asset_for_newton`, then use
:doc:`/source/how-to/solver_tuning/tune_mjwarp` for the primary validated solver path or
:doc:`/source/how-to/solver_tuning/tune_kamino` for the beta Kamino path. Experimental
specialist guides cover :ref:`newton-using-vbd`, :ref:`newton-using-mpm`,
:ref:`newton-using-cables`, :ref:`warp-environments`, and
:ref:`warp-env-migration`. Backend developers can also read
:doc:`/source/overview/core-concepts/physical-backends/newton/newton-manager-abstraction`.


.. _physics-backends-ovphysx:

OvPhysX
-------

OvPhysX is an experimental kit-less PhysX path. It drives PhysX without
Omniverse Kit, while scene-level solver settings primarily come from the USD
``PhysicsScene`` prim. Only settings that are not represented in USD live on
:class:`~isaaclab_ov.physics.OvPhysxCfg`; consult its generated API reference
for those fields.

Feature coverage is still evolving. See the
:doc:`OvPhysX specialist page
</source/overview/core-concepts/physical-backends/ovphysx/index>` for current
limitations and runtime constraints.


Find supported tasks and APIs
-----------------------------

Backend support is task-specific. Use :doc:`/source/overview/environments` to
browse registered environments and presets, or ask a task for its current
choices:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct --help

The help output is the source of truth for that task's ``physics=`` choices.
For native solver data and views, see
:doc:`/source/how-to/native_physics_api/index`. For field-level configuration,
follow the generated API links for :class:`~isaaclab_physx.physics.PhysxCfg`,
:class:`~isaaclab_newton.physics.NewtonCfg`, its solver configurations, and
:class:`~isaaclab_ov.physics.OvPhysxCfg`.


.. toctree::
   :hidden:

   /source/overview/core-concepts/physical-backends/ovphysx/index
