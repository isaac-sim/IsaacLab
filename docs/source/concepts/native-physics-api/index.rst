.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _native-physics-api:

Native Physics API Access
=========================

.. warning::

   Native physics APIs are backend-specific escape hatches. They can bypass
   Isaac Lab's buffering, validation, ordering, and lifecycle management. Use
   the unified asset and sensor APIs unless an engine-native capability or a
   lower-overhead data path is required.

Choose an access level
----------------------

Use native access only when the unified APIs do not provide a capability or
data path that the workload needs. Keep it close to the component that owns the
simulation state. For the portable API boundary and backend lifecycle, see
:ref:`backend-architecture`.

#. Use the unified Isaac Lab data and write APIs for portable task code.
#. Reuse an Isaac Lab-owned native handle when its selection already matches.
#. Create raw engine access only for a selection or capability that the owning
   Isaac Lab object does not expose.

Ownership and synchronization
-----------------------------

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

.. toctree::
   :maxdepth: 1
   :hidden:

   physx
   newton
   ovphysx
