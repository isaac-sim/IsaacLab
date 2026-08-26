.. _how-to-torch-array:
.. _working-with-simulation-data:

Access Simulation Data with Torch and Warp
==========================================

.. currentmodule:: isaaclab.utils.warp

Asset and sensor data properties return :class:`ProxyArray`, which provides zero-copy access to
the same simulation data through explicit Torch and Warp representations. Select the representation
required by the code that consumes the data, and re-access properties when simulation buffers may
have been rebuilt.

.. note::

   The :class:`ProxyArray` design is inspired by the ``ProxyArray`` class from
   `mujocolab/mjlab <https://github.com/mujocolab/mjlab>`_ (BSD-3-Clause).


Choose an Array Representation
------------------------------

Use ``.torch`` for PyTorch operations. It returns a cached, zero-copy :class:`torch.Tensor` view.
Use ``.warp`` when an API requires the underlying :class:`warp.array` or Warp-specific attributes
such as ``ptr`` or ``strides``.

.. code-block:: python

   robot = env.scene["robot"]
   joint_pos = robot.data.joint_pos

   # PyTorch code uses the cached tensor view.
   joint_pos_mean = torch.mean(joint_pos.torch, dim=1)

   # APIs that require an actual warp.array use the underlying array.
   joint_pos_ptr = joint_pos.warp.ptr

A :class:`ProxyArray` can be passed directly to a Warp kernel through the CUDA array interface, so
kernel launches do not need to unwrap it first:

.. code-block:: python

   wp.launch(my_kernel, inputs=[robot.data.joint_pos], ...)


Understand Zero-Copy Access
---------------------------

The ``.torch`` and ``.warp`` accessors share the same underlying buffer. Changes made through one
representation are visible through the other, and repeated ``.torch`` access on the same
:class:`ProxyArray` returns the cached tensor view.

This shared view does not replace public asset write methods. Use the appropriate ``write_*`` method
when changing simulation state. If a value must remain unchanged while the simulation continues,
copy it explicitly:

.. code-block:: python

   joint_pos_snapshot = robot.data.joint_pos.torch.clone()


Keep Views Valid Across Resets
------------------------------

Treat the arrays returned by ``.torch`` and ``.warp`` as borrowed views of simulation-owned data.
Some operations, especially full resets with the Newton backend, may replace the underlying buffers.
Previously retained views may then refer to stale or freed memory.

Re-access the data property after a reset. Clone the value before the reset when a snapshot must
remain valid:

.. code-block:: python

   joint_pos_before_reset = robot.data.joint_pos.torch.clone()
   env.reset()
   joint_pos_after_reset = robot.data.joint_pos.torch


Handle Temporary Compatibility Paths
------------------------------------

Implicit Torch operations on :class:`ProxyArray` and ``wp.to_torch(proxy_array)`` remain temporarily
supported for migrated code, but emit a one-time :class:`DeprecationWarning`. Use explicit ``.torch``
access in new code because these compatibility paths will be removed in a future release.

For the complete Isaac Lab 2.x migration procedure, see :ref:`torcharray-migration`.


Related Documentation
---------------------

- :ref:`backend-architecture` explains why :class:`ProxyArray` is part of the portable backend API.
- :class:`ProxyArray` provides the generated class and member reference.
- :ref:`torcharray-migration` covers changes required when migrating Isaac Lab 2.x code.
