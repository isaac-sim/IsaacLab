.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-coupled-solvers:

Coupled Solvers
===============

.. warning::

   Coupled solvers are experimental and exposed through
   :mod:`isaaclab_contrib.coupling`. Their API, behavior, feature support,
   performance, and implementation may change.

Newton can partition one model between multiple solvers and exchange state and
forces between them during each simulation step. This lets a task combine
solver families that target different physics, such as an MJWarp rigid robot
interacting with VBD cloth or an MPM material.

Isaac Lab exposes this framework through
:mod:`isaaclab_contrib.coupling`. The adapter turns configuration selectors
into named Newton solver entries, constructs the selected coupling algorithm,
and integrates it with :class:`~isaaclab_newton.physics.NewtonCfg`. For the
shared-model architecture, iteration algorithms, supported constraint rows,
and solver-specific implementation details, see Newton's
`Coupled Solvers concept page
<https://newton-physics.github.io/newton/stable/concepts/coupling.html>`_.


The coupling model
------------------

A coupled simulation starts from one Newton model. Instead of giving the whole
model to one solver, Isaac Lab partitions it into named entries. Each entry
selects a solver and owns a disjoint part of the model.

The animation below shows how source-owned gripper bodies interact with
material owned by a destination solver through local proxies:

.. raw:: html

   <figure>
     <video controls playsinline preload="none" width="1280" height="866"
            style="width:100%;height:auto;"
            poster="../../_static/newton/proxy-coupling.jpg"
            aria-label="Proxy coupling between source and destination solvers"
            aria-describedby="coupling-animation-caption">
       <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/proxy-coupling-explainer.mp4" type="video/mp4">
       <a href="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/proxy-coupling-explainer.mp4">Watch the proxy coupling animation.</a>
     </video>
     <figcaption id="coupling-animation-caption">
       <p>Filled teal fingers belong to the source solver; outlined blue fingers are
         destination-local proxies of those same bodies. The blue object represents
         material owned by the destination solver.</p>
       <p>The upper arrow synchronizes source state with the proxies, including pose
         and velocity. Contact produces the force and torque feedback shown by the
         lower arrow, which returns to a later source solve or coupling iteration.</p>
       <p><a href="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/proxy-coupling-explainer.mp4">Open the animation</a> to pause or scrub through the exchange.</p>
     </figcaption>
   </figure>

Each solver receives a view of the shared model. Only the elements owned by
its entry are reconciled into the final shared state. An element can belong to
at most one entry; unassigned elements remain outside the nested solvers.
Keep each articulation in a single entry.

Isaac Lab resolves ownership selectors, constructs the Newton entry views, and
runs the coupled solver through the normal Newton backend lifecycle. Newton
owns the coupling algorithms and the exchange of poses, forces, and constraint
information between entries.


Choose Proxy or ADMM Coupling
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

Proxy coupling can have lower coupling overhead because it reuses the
destination solver's contact path and may work with one pass, but its
directional exchange is timestep- and ordering-sensitive. ADMM represents a
symmetric interface, but every coupling iteration advances the participating
solvers again. Additional passes or iterations can improve coupled response and
interface convergence at a higher runtime cost. Neither approach is uniformly
more accurate; compare them on task-relevant physical metrics.


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
cross-entry joints and attachments.


Tune Coupling
-------------

Stabilize each entry independently before changing coupling controls.

* ``CouplerEntryCfg.substeps`` changes the time resolution for one entry; more
  substeps add solver work.
* For proxy coupling, ``mode`` controls exchange ordering, ``iterations``
  controls relaxation passes, ``mass_scale`` changes proxy effective inertia in
  the destination, and ``collide_interval`` controls contact refresh frequency.
* For ADMM, ``iterations`` controls interface passes, ``rho`` sets the penalty
  weight, ``gamma`` adds proximal inertia and velocity weighting, and
  ``baumgarte`` adds positional-error correction.

More substeps or iterations can improve stability or convergence, but cost
runtime and cannot repair an unstable entry. The generated
:doc:`coupling configuration API
</source/api/lab_contrib/isaaclab_contrib.coupling>` lists every field and
default; Newton's concept page explains the underlying algorithms.


ADMM Contact Capacity
^^^^^^^^^^^^^^^^^^^^^

ADMM's internal contact buffers are independent of ``NewtonCfg.collision_cfg``.
Increase ``CouplerAdmmCfg.contact_max_triangle_pairs`` for triangle-pair overflows,
or ``contact_reduction_hashtable_size_factor`` for contact-reduction hash table
warnings. Both default to ``None`` (Newton's defaults); explicit values require
support in Newton's ``SolverCoupledADMM.Config``. Capacities cover all environments
in one process, independently on each rank in multi-GPU jobs.

With ``rigid_contact_matching="latest"`` or ``"sticky"``, triangle-pair capacity
must be less than ``2**20``; larger values require ``"disabled"``. The hash table
can grow independently while retaining matching:

.. code-block:: python

    coupling_cfg = CouplerAdmmCfg(
        entries=entries,
        rigid_contact_matching="latest",
        contact_max_triangle_pairs=1_000_000,
        contact_reduction_hashtable_size_factor=2.0,
    )


Rigid--MPM Comparison Recordings
--------------------------------

For proxy coupling, ``mass_scale`` changes a source body's effective mass and
inertia in the destination view; it does not change the authored rigid-body
mass. Start at ``1`` and increase it only when the rigid solver strongly
constrains the body during MPM contact. Check both supported and freely moving
cases, since a large scale can suppress legitimate motion.

The G1 recordings show one policy walking across sand, snow, and clay. The
one-way recording uses lower-leg proxies, while the two-way recording uses
full-body proxies. They are presentation examples, **not** a controlled
single-variable comparison; keep the policy, seed, commands, and proxy geometry
identical when measuring a coupling effect.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: One-way, lower-leg proxies

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_g1_one_way.mp4" type="video/mp4">
         </video>

   .. grid-item-card:: Two-way, full-body proxies

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_g1_two_way.mp4" type="video/mp4">
         </video>


Start from a maintained task
----------------------------

The :ref:`newton-vbd-proxy-coupling` guide contains the complete configuration
and runnable commands for the Franka soft-body tasks. Start from that example
when building a proxy-coupled rigid--deformable environment, then narrow entry
ownership and proxy selectors to the bodies that participate in the
interaction.

Current Isaac Lab limitations include no support for nested couplers or Newton
contact sensors, and proxy coupling supports at most two entries. Some solver
modes require manager-specific lifecycle work and cannot be nested in a
coupler. Validate each entry independently before tuning the coupled result,
and consult the Newton concept page for current algorithm-level support and
limitations.
