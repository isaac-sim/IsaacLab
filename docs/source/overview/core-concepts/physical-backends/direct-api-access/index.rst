.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Direct Physics Engine API Access
================================

.. warning::

   Native physics APIs are backend-specific escape hatches. They can bypass
   Isaac Lab's buffering, validation, ordering, and lifecycle management. Use
   the unified asset and sensor APIs unless an engine-native capability or a
   lower-overhead data path is required.

When to use native access
-------------------------

Use native access when a task depends on an engine-specific capability or when
the unified APIs do not do what you need. Keep this access close to the
component that owns the simulation state, and prefer the portable Isaac Lab
APIs for task logic that does not need it.

Why there is no unified low-level view
--------------------------------------

The backends expose fundamentally different access models. PhysX and OvPhysX
use explicit pull/push operations, so data is refreshed or published only when
the corresponding method is called. Newton instead exposes live arrays owned
by ``Model``, ``State``, ``Control``, and ``Contacts``. Changes to those arrays
immediately affect the owning object, although derived state and solver caches
can still require explicit synchronization.

PhysX organizes access into typed views for physics-object families. OvPhysX
organizes access into bindings selected by tensor type;
:class:`~isaaclab_ov.sim.views.OvPhysxView` is an Isaac Lab convenience
manager over those bindings. Newton selections describe subsets and batched
layouts without imposing an asset type. A single facade would erase these
ownership and synchronization differences and reduce the engines to a
least-common-denominator API. Isaac Lab preserves native access so advanced
users retain engine-specific performance and capabilities.

How the access models differ
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 20 17 18 18 20 20

   * - Backend
     - Native entry point
     - Selection model
     - Data ownership
     - Read/write model
     - Synchronization
     - Invalidation
   * - PhysX
     - :meth:`~isaaclab_physx.physics.PhysxManager.get_physics_sim_view`
     - Typed views selected by prim globs
     - Engine/view-owned buffers
     - Getter/setter pull/push
     - Explicit setters and occasional kinematic refresh
     - Reacquire after view invalidation, hard reset, or teardown
   * - Newton
     - :meth:`~isaaclab_newton.physics.NewtonManager.get_model`,
       :meth:`~isaaclab_newton.physics.NewtonManager.get_state_0`,
       :meth:`~isaaclab_newton.physics.NewtonManager.get_control`, and
       :meth:`~isaaclab_newton.physics.NewtonManager.get_contacts`
     - Generic labels and ``ArticulationView``
     - Live engine-owned Warp arrays
     - Direct pointer or selection reads/writes
     - Forward kinematics and model-change notification when applicable
     - Reacquire current state across state-buffer swaps and all objects after
       model rebuild
   * - OvPhysX
     - :meth:`~isaaclab_ov.physics.OvPhysxManager.get_physx_instance`
       or :class:`~isaaclab_ov.sim.views.OvPhysxView`
     - A tensor type plus pattern/prim list
     - Caller-owned transfer buffers backed by engine bindings
     - Explicit ``read()``/``write()`` or guarded convenience methods
     - Caller respects access mode, device, dtype, and shape
     - Reacquire after stage/runtime teardown

Choosing an access level
------------------------

#. Prefer the unified Isaac Lab data and write APIs for portable task code.
#. Reuse an Isaac Lab-owned native handle when its selection already matches.
#. Construct raw engine access only for selections or capabilities not exposed
   by the owning Isaac Lab object.

.. toctree::
   :maxdepth: 1
   :hidden:

   physx
   newton
   ovphysx
