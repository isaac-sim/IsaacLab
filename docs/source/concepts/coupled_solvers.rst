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

The proxy example below shows how the hand and fingers of an MJWarp-owned
robot interact with VBD-owned cloth through two views of that same model:

.. raw:: html

   <figure class="coupling-diagram" aria-labelledby="coupling-diagram-caption">
     <div class="coupling-diagram-heading">
       <strong>One shared Newton model</strong>
       <span>MJWarp–VBD proxy example</span>
     </div>
     <div class="coupling-diagram-views">
       <div class="coupling-diagram-view coupling-diagram-source">
         <p class="coupling-diagram-role">Source view</p>
         <p class="coupling-diagram-title"><code>rigid</code> · MJWarp</p>
         <p>Owns robot bodies, joints, and shapes</p>
         <svg viewBox="0 0 300 215" role="img" aria-labelledby="coupling-source-title coupling-source-desc">
           <title id="coupling-source-title">Robot with selected hand and fingers</title>
           <desc id="coupling-source-desc">A solid robot arm belongs to MJWarp. The hand and fingers at its tip are highlighted as the bodies selected for the proxy mapping.</desc>
           <g class="coupling-diagram-arm">
             <path d="M40 184 V145 L91 111 L144 54 H205 V74"/>
             <circle cx="40" cy="145" r="9"/>
             <circle cx="91" cy="111" r="9"/>
             <circle cx="144" cy="54" r="9"/>
             <path d="M22 190 H58"/>
           </g>
           <g class="coupling-diagram-hand">
             <rect x="175" y="74" width="60" height="24" rx="4"/>
             <path d="M175 98 H187 V135 H198 V147 H175 Z"/>
             <path d="M223 98 H235 V147 H212 V135 H223 Z"/>
           </g>
         </svg>
         <p class="coupling-diagram-selection">Selected: hand + fingers</p>
         <p class="coupling-diagram-detail">Solid shapes: owned robot</p>
       </div>
       <div class="coupling-diagram-exchange">
         <div class="coupling-diagram-transfer">
           <p>Pose + velocity</p>
           <span class="coupling-diagram-arrow coupling-diagram-state-arrow" aria-hidden="true"></span>
           <p class="coupling-diagram-detail">Synchronize the proxy before the destination solve</p>
         </div>
         <div class="coupling-diagram-feedback">
           <p>Forces + torques</p>
           <span class="coupling-diagram-arrow coupling-diagram-feedback-arrow" aria-hidden="true"></span>
           <p class="coupling-diagram-detail">Return feedback for the next source pass or iteration</p>
         </div>
       </div>
       <div class="coupling-diagram-view coupling-diagram-destination">
         <p class="coupling-diagram-role">Destination view</p>
         <p class="coupling-diagram-title"><code>soft</code> · VBD</p>
         <p>Owns cloth particles and static shapes</p>
         <svg viewBox="0 0 300 215" role="img" aria-labelledby="coupling-destination-title coupling-destination-desc">
           <title id="coupling-destination-title">Cloth contacting the proxy hand and fingers</title>
           <desc id="coupling-destination-desc">Dashed outlines represent the same selected hand and fingers in VBD's view. Solid cloth particles contact the fingers. The rest of the robot is not exposed as a proxy.</desc>
           <g class="coupling-diagram-cloth">
             <path d="M85 160 L110 147 L135 140 L160 147 L185 160 M85 178 L110 165 L135 158 L160 165 L185 178 M85 196 L110 183 L135 176 L160 183 L185 196 M85 160 V196 M110 147 V183 M135 140 V176 M160 147 V183 M185 160 V196"/>
             <circle cx="85" cy="160" r="4"/><circle cx="110" cy="147" r="4"/><circle cx="135" cy="140" r="4"/><circle cx="160" cy="147" r="4"/><circle cx="185" cy="160" r="4"/>
             <circle cx="85" cy="178" r="4"/><circle cx="110" cy="165" r="4"/><circle cx="135" cy="158" r="4"/><circle cx="160" cy="165" r="4"/><circle cx="185" cy="178" r="4"/>
             <circle cx="85" cy="196" r="4"/><circle cx="110" cy="183" r="4"/><circle cx="135" cy="176" r="4"/><circle cx="160" cy="183" r="4"/><circle cx="185" cy="196" r="4"/>
           </g>
           <g class="coupling-diagram-hand coupling-diagram-proxy" transform="translate(-70 0)">
             <rect x="175" y="74" width="60" height="24" rx="4"/>
             <path d="M175 98 H187 V135 H198 V147 H175 Z"/>
             <path d="M223 98 H235 V147 H212 V135 H223 Z"/>
           </g>
         </svg>
         <p class="coupling-diagram-selection">Dashed shapes: the same selected bodies as proxies</p>
         <p class="coupling-diagram-detail">Solve local contacts with the cloth</p>
       </div>
     </div>
     <figcaption id="coupling-diagram-caption">
       MJWarp owns the robot; VBD sees the selected hand and fingers as virtual colliders alongside its cloth particles.
       VBD solves contact using destination-local virtual inertia and returns force and torque feedback for a later source pass.
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
