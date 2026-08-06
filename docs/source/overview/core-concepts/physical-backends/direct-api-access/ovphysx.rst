# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

OvPhysX Tensor Bindings
=======================

Mental model
------------

OvPhysX exposes generic ``TensorBinding`` objects. A binding combines a prim
selection with one installed ``TensorType``; it is not an asset-specific native
storage object. Callers explicitly pull values into their own buffer with
``read()`` and push a buffer back with ``write()``. Isaac Lab optionally wraps
these bindings in :class:`~isaaclab_ovphysx.sim.views.OvPhysxView` for a
string-keyed, guarded convenience surface.

Lifecycle prerequisite
----------------------

Create bindings only after the OvPhysX simulation has initialized and reset,
when its native runtime is live. A stage reload, reset path that rebuilds the
stage or runtime, and manager teardown invalidate existing bindings and views;
reacquire the runtime handle and then recreate bindings and views afterwards.

Access the native runtime
-------------------------

Get the active OvPhysX handle from the manager:

.. code-block:: python

   from isaaclab_ovphysx.physics import OvPhysxManager

   physx = OvPhysxManager.get_physx_instance()
   if physx is None:
       raise RuntimeError("OvPhysX has not been constructed; initialize and reset the simulation first.")

A non-``None`` handle proves only that the manager constructed an OvPhysX instance; it is not a
readiness predicate for the current simulation context. The manager can retain that cached handle
after teardown. Initialize and reset the current context before access, and always reacquire the
handle before creating new bindings or views.

Create a raw tensor binding
---------------------------

Create a raw binding when the desired selection is not already owned by an
Isaac Lab asset. The following representative pose binding is float32 and
device-resident; it is not a generic allocator for every tensor type:

.. code-block:: python

   import warp as wp
   from isaaclab.physics import PhysicsManager
   from ovphysx.types import TensorType

   binding = physx.create_tensor_binding(
       pattern="/World/envs/env_*/Object",
       tensor_type=TensorType.RIGID_BODY_POSE,
   )
   poses = wp.empty(
       tuple(binding.shape),
       dtype=wp.float32,
       device=PhysicsManager.get_device(),
   )
   binding.read(poses)
   binding.write(poses)

Before allocating a buffer, inspect ``binding.shape``, ``binding.dtype``, and
the binding's count and path metadata. Match the binding shape and DLPack scalar
metadata exactly, and respect the binding's native residency and access mode.
``RIGID_BODY_POSE`` has the float32 layout used above, but another tensor type
can have a different scalar dtype, shape, native device, or write permission.

Reuse or create an ``OvPhysxView``
----------------------------------

Reuse the convenience view already owned by an Isaac Lab asset when its
selection matches:

.. code-block:: python

   object_view = scene["object"].root_view
   poses = object_view.get_attribute("rigid_body_pose")

For a new selection, construct :class:`~isaaclab_ovphysx.sim.views.OvPhysxView`
with the native runtime and a Tensor API pattern:

.. code-block:: python

   from isaaclab.physics import PhysicsManager
   from isaaclab_ovphysx.sim.views import OvPhysxView

   view = OvPhysxView(
       physx,
       pattern="/World/envs/env_*/Object",
       device=PhysicsManager.get_device(),
   )

Discover tensor types and availability
--------------------------------------

Discover installed tensor types at runtime instead of maintaining an enum
inventory:

.. code-block:: python

   from ovphysx.types import TensorType

   tensor_types = [
       tensor_type
       for tensor_type in TensorType
       if tensor_type.name != "INVALID"
   ]
   print(tensor_types)

The view separates its installed vocabulary from selection-specific
availability:

.. code-block:: python

   print(view.attribute_names)

   if view.try_binding_for("rigid_body_pose") is not None:
       poses = view.get_attribute("rigid_body_pose")
       view.read_into("rigid_body_pose", poses)
       view.set_attribute("rigid_body_pose", poses)

   print(view.available_attributes)

``attribute_names`` is the installed ``TensorType`` vocabulary, not selection
availability. ``try_binding_for`` attempts to create a binding for this
selection and returns ``None`` when a valid type is unavailable. By contrast,
``available_attributes`` lists bindings already instantiated for the view.
``binding_for`` returns a raw binding and bypasses the view's device, dtype,
shape, and read-only guards.

Read and write data
-------------------

``binding.read(buffer)`` explicitly pulls the selected values into the
caller-owned buffer; editing that buffer has no engine effect until
``binding.write(buffer)`` explicitly pushes it. The guarded
:meth:`~isaaclab_ovphysx.sim.views.OvPhysxView.get_attribute`,
:meth:`~isaaclab_ovphysx.sim.views.OvPhysxView.read_into`, and
:meth:`~isaaclab_ovphysx.sim.views.OvPhysxView.set_attribute` paths provide the
same pull/push boundary while validating the binding's shape, scalar dtype,
native device, and writable access.

Ownership, synchronization, and invalidation
--------------------------------------------

State tensors normally live on the simulation device, while some property
tensors are CPU-only. Allocate or supply a buffer on the binding's native
device and match its DLPack ``code``, ``bits``, and ``lanes`` scalar metadata
as well as its shape. :class:`~isaaclab_ovphysx.sim.views.OvPhysxView` can
reinterpret supported flat scalar layouts as structured Warp values, but it
does not perform implicit CPU/GPU staging.

A valid ``TensorType`` need not apply to the selected prims. Use
``try_binding_for`` for optional bindings, and do not treat a vocabulary entry
as a promise of availability. Reacquire raw bindings and convenience views
after reset paths that rebuild the stage or runtime, or after teardown.

Authoritative references
------------------------

The generated Isaac Lab reference for
:class:`~isaaclab_ovphysx.sim.views.OvPhysxView` documents the convenience
manager. The repository metadata pins OvPhysX to a version but does not provide
a versioned official OvPhysX documentation URL, so inspect the installed
``TensorType`` and binding metadata at runtime.
