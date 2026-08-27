.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-coupled-solvers:

Coupled Solvers
===============

Newton can partition one model between multiple solvers and exchange state and
forces between them during each simulation step. This lets a task combine
solver families that target different physics, such as an MJWarp rigid robot
interacting with VBD cloth or an MPM material.

Isaac Lab exposes this experimental framework through
:mod:`isaaclab_contrib.coupling`. The adapter turns configuration selectors
into named Newton solver entries, constructs the selected coupling algorithm,
and integrates it with the normal :class:`~isaaclab_newton.physics.NewtonCfg`
lifecycle. For the shared-model architecture, iteration algorithms, supported
constraint rows, and solver-specific implementation details, see Newton's
`Coupled Solvers concept page
<https://newton-physics.github.io/newton/stable/concepts/coupling.html>`_.


The coupling model
------------------

A coupled simulation starts from one Newton model. Instead of giving the whole
model to one solver, Isaac Lab partitions it into named entries. Each entry
selects a solver and owns a disjoint part of the model:

.. code-block:: text

   Shared Newton model
   ├── "rigid" entry → MJWarp owns robot bodies, joints, and shapes
   ├── "soft" entry  → VBD owns deformable particles and static shapes
   └── interface     → proxy or ADMM coupling connects the entries

Each solver receives a view of the shared model and advances only the elements
owned by its entry. An element can belong to at most one entry; unassigned
elements remain outside the nested solvers. Keep an articulation in one entry
unless a supported ADMM constraint intentionally connects bodies owned by
different entries.

Isaac Lab resolves ownership selectors, constructs the Newton entry views, and
runs the coupled solver through the normal Newton backend lifecycle. Newton
owns the coupling algorithms and the exchange of poses, forces, and constraint
information between entries.


Choose proxy or ADMM coupling
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Approach
     - How it works
     - When to use it
   * - Proxy
     - A source-owned body or particle appears as a virtual endpoint in a
       destination solver. The destination returns feedback on a later pass or
       iteration.
     - Use when the interaction is naturally directional, such as a rigid
       collider inside a deformable or particle solve. Proxy coupling can
       reuse the destination solver's contact path and is the established path
       for Isaac Lab's coupled MJWarp--VBD and rigid--MPM tasks.
   * - ADMM
     - The coupler creates interface constraints between entries, iterates the
       sub-solvers, and applies equal and opposite interface forces.
     - Use when the interface should be symmetric, especially for supported
       cross-entry joints, body--particle attachments, or frictional contacts.
       ADMM has more tuning parameters and supports a narrower set of
       constraint rows.

Proxy coupling is usually the simpler starting point for collider-style
rigid--deformable interaction. Use ``mode="lagged"`` first; the
``"staggered"`` mode uses a newer source state but is more sensitive to the
timestep and ordering. Increase coupling iterations only after each entry is
stable on its own.

ADMM is a better fit when assigning a source and destination would make the
physical interface artificially one-way. Its fixed iteration count and
penalty, proximal, and stabilization parameters are part of the coupled
constraint solve, so tune them together with the timestep and the participating
solvers. Newton's concept page is the source of truth for the currently
supported joints, contacts, and limitations.


Configure a coupled solver
--------------------------

In Isaac Lab, :class:`~isaaclab_contrib.coupling.CouplerEntryCfg` defines each
entry's solver and ownership. Use
:class:`~isaaclab_contrib.coupling.CouplerProxyCfg` or
:class:`~isaaclab_contrib.coupling.CouplerAdmmCfg` as the
:class:`~isaaclab_newton.physics.NewtonCfg` solver configuration.

The following configuration mirrors the maintained Franka rigid--deformable tasks. It
assigns the complete robot to MJWarp, particles and static collision geometry
to VBD, and exposes only the hand and fingers as VBD proxy colliders:

.. code-block:: python

   from isaaclab_contrib.coupling import (
       CouplerEntryCfg,
       CouplerProxyCfg,
       CouplerProxyMappingCfg,
   )
   from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, VBDSolverCfg

   entries = [
       CouplerEntryCfg(
           name="rigid",
           solver_cfg=MJWarpSolverCfg(),
           bodies=[r"/World/envs/env_[^/]+/Robot"],
       ),
       CouplerEntryCfg(
           name="soft",
           solver_cfg=VBDSolverCfg(),
           all_particles=True,
           include_static_shapes=True,
       ),
   ]

   physics = NewtonCfg(
       solver_cfg=CouplerProxyCfg(
           entries=entries,
           proxies=[
               CouplerProxyMappingCfg(
                   source="rigid",
                   destination="soft",
                   bodies=[
                       r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_hand",
                       r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_(left|right)finger",
                   ],
                   mode="lagged",
               )
           ],
           iterations=1,
       ),
       num_substeps=2,
   )

For ADMM, keep the ownership entries and replace the proxy mapping with the
symmetric interfaces that should be coupled:

.. code-block:: python

   from isaaclab_contrib.coupling import CouplerAdmmCfg

   physics = NewtonCfg(
       solver_cfg=CouplerAdmmCfg(
           entries=entries,
           contact_pairs=[("rigid", "soft")],
           iterations=5,
           rho=1.0,
       ),
       num_substeps=2,
   )

Set ``contact_pairs=None`` to generate every distinct entry pair, or use an
empty list to disable ADMM contact coupling while retaining supported
cross-entry joints and attachments. The generated
:doc:`coupling configuration API
</source/api/lab_contrib/isaaclab_contrib.coupling>` lists the current fields
and defaults.


Start from a maintained task
----------------------------

The :ref:`newton-vbd-proxy-coupling` guide contains the complete configuration
and runnable commands for the Franka soft-body tasks. Start from that example
when building a proxy-coupled rigid--deformable environment, then narrow entry
ownership and proxy selectors to the bodies that participate in the
interaction.

Coupling remains experimental. The Isaac Lab adapter currently does not
support nested couplers or Newton contact sensors, and proxy coupling supports
at most two entries. Some solver modes require manager-specific lifecycle work
and cannot be nested in a coupler. Validate each entry independently before
tuning the coupled result, and consult the Newton concept page for the current
algorithm-level support and limitations.
