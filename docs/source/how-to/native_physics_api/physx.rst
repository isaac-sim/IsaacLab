.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

PhysX Tensor API
================

The PhysX Tensor API provides typed views for data paths and capabilities that
require engine-native access.

Mental model
------------

The API starts from a ``SimulationView`` and creates typed views over selected
physics objects. Access is entirely method-based: a getter pulls data from the
view, and an independent setter publishes data back. Returned buffers are not
live pointers, so editing a getter result does not update the simulation. Raw
views select objects with PhysX Tensor API wildcard patterns.

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

Reuse Isaac Lab-owned access
----------------------------

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

Create raw access
-----------------

Create a view directly only when no Isaac Lab-owned selection matches the
needed objects or capability. In this Tensor API pattern, ``*`` selects the
matching object below every cloned environment:

.. code-block:: python

   rigid_body_view = simulation_view.create_rigid_body_view(
       "/World/envs/env_*/Object"
   )

Read/write semantics
--------------------

Each getter and setter is a separate operation. Clone a returned Warp buffer
before editing it locally, then call the matching setter to publish the result:

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

Access the contact view
-----------------------

The PhysX contact sensor exposes its native ``RigidContactView`` through
:attr:`~isaaclab_physx.sensors.ContactSensor.contact_view`. Use it when the
public sensor data does not expose the required contact details:

.. code-block:: python

   contact_sensor = scene["contact_sensor"]
   friction_forces, _, buffer_counts, buffer_start_indices = (
       contact_sensor.contact_view.get_friction_data(dt=sim.cfg.dt)
   )

Other Isaac Lab sensors may use ordinary rigid-body or articulation views
internally, but those views are not sensor-specific low-level interfaces.

Ownership, synchronization, and invalidation
--------------------------------------------

Preserve a view's expected selection ordering, device, dtype, and tensor shape
when supplying values to a setter. Check and reacquire views after a hard
reset, object removal, stage reload, or manager teardown. Prefer public Isaac
Lab sensor data when direct native contact or motion data is not required.

Authoritative references
------------------------

* `Omni Physics Python API reference <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/pythonapi.html>`_
