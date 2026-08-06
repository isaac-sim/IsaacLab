# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

PhysX Tensor API
================

The PhysX Tensor API is an engine-native interface for data paths that need
typed PhysX views or capabilities beyond the unified Isaac Lab APIs. It is
backend-specific: use Isaac Lab asset and sensor APIs unless native access is
needed for the workload.

Mental model
------------

The API starts from a ``SimulationView`` and creates
typed views over selected physics objects. A view owns an engine-backed
selection; getters pull data from that selection and setters publish data back
to it. A raw view selection uses the Tensor API's glob syntax, which is distinct
from Isaac Lab's regular-expression syntax.

Lifecycle prerequisite
----------------------

:class:`~isaaclab_physx.physics.PhysxManager` creates its
``SimulationView`` with the Warp frontend. Low-level
code must run only after physics initialization and simulation reset, when the
PhysX Tensor API view has been created:

.. code-block:: python

   from isaaclab_physx.physics import PhysxManager

   simulation_view = PhysxManager.get_physics_sim_view()
   if simulation_view is None:
       raise RuntimeError("PhysX Tensor API is not ready; initialize and reset the simulation first.")

Reuse an Isaac Lab view
-----------------------

When an Isaac Lab asset already has the desired selection, reuse its
:attr:`~isaaclab_physx.assets.Articulation.root_view` rather than creating a
second view:

.. code-block:: python

   robot = scene["robot"]
   articulation_view = robot.root_view
   joint_positions = articulation_view.get_dof_positions()

:attr:`~isaaclab_physx.assets.Articulation.root_view` is backend-specific and
typed. An articulation, rigid object, rigid-object collection, or deformable
can expose a different native PhysX view type, so choose methods that match the
returned view rather than assuming one common interface.

Create a raw typed view
-----------------------

Create a view directly only when no Isaac Lab-owned selection matches the
needed objects or capability. The path below uses Tensor API wildcards, not an
Isaac Lab regular expression:

.. code-block:: python

   rigid_body_view = simulation_view.create_rigid_body_view(
       "/World/envs/env_*/Object"
   )

Discover available view factories
---------------------------------

The available factories are provided by the installed PhysX version. Discover
them at runtime instead of maintaining a hand-written inventory:

.. code-block:: python

   view_factories = sorted(
       name
       for name in dir(simulation_view)
       if name.startswith("create_") and name.endswith("_view")
   )
   print(view_factories)

Read and write data
-------------------

Getters and setters make the transfer boundary explicit. Clone a returned
Warp buffer before editing it locally, then use the setter to publish the
result:

.. code-block:: python

   import warp as wp

   poses = wp.clone(rigid_body_view.get_transforms())
   indices = wp.array(
       range(poses.shape[0]),
       dtype=wp.int32,
       device=poses.device,
   )
   rigid_body_view.set_transforms(poses, indices)

Cloning makes ownership explicit. Callers can modify the clone with Warp before
the setter, but edits to a local buffer do not publish themselves. The
``indices`` array selects the complete view and matches the setter's required
``int32`` dtype and device; the setter performs the write. For active Tensor API
contracts that require link transforms to be refreshed after joint-state writes, call
``update_articulations_kinematic()``.
Not every setter requires that refresh; follow the method-level behavior in the
upstream reference.

Sensor view families
--------------------

Isaac Lab sensors internally create the following native view families. These
are implementation details for understanding data sources, not additional
sensor APIs; prefer the sensor's public data interface when it provides the
needed information. Consult the upstream reference for method-level behavior.

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Isaac Lab sensor
     - Native PhysX view family
     - Scope
   * - Contact sensor
     - ``RigidBodyView`` and ``RigidContactView``
     - Tracked-body state and contact reporting.
   * - Frame transformer
     - ``RigidBodyView``
     - Tracked-body transforms.
   * - IMU
     - ``RigidBodyView``
     - Rigid-body motion state.
   * - PVA sensor
     - ``RigidBodyView``
     - Rigid-parent motion state.
   * - Joint-wrench sensor
     - ``ArticulationView``
     - Articulation joint-wrench state.
   * - Ray caster
     - ``RigidBodyView``
     - Tracked-body transforms; ray intersection is not a PhysX Tensor API
       view.

Ownership, synchronization, and invalidation
--------------------------------------------

Preserve a view's expected selection ordering, device, dtype, and tensor shape
when supplying values to a setter. Check and reacquire views after a hard
reset, object removal, stage reload, or manager teardown. Prefer public Isaac
Lab sensor data when direct native contact or motion data is not required.

Authoritative reference
-----------------------

The upstream API documents view-specific factory, getter, setter, and
synchronization behavior: `Omni Physics Python APIs <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/pythonapi.html>`_.
