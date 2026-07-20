.. _how-to-torch-array:

Working with ProxyArray
=======================

.. currentmodule:: isaaclab.utils.warp

Isaac Lab data classes return :class:`ProxyArray` — a lightweight, warp-first wrapper that
provides zero-copy access to simulation data as either a :class:`warp.array` or a
:class:`torch.Tensor`.

.. note::

   ``ProxyArray`` is inspired by the ``ProxyArray`` class from
   `mujocolab/mjlab <https://github.com/mujocolab/mjlab>`_ (BSD-3-Clause).
   The design adapts the same dual-accessor pattern to Isaac Lab's warp-based data pipeline.


Quick Start
~~~~~~~~~~~

Every property on asset and sensor data classes (e.g., ``robot.data.joint_pos``,
``sensor.data.net_forces_w``) returns a ``ProxyArray``:

.. code-block:: python

   robot = env.scene["robot"]

   # Explicit torch access (preferred)
   joint_positions = robot.data.joint_pos.torch        # torch.Tensor, cached zero-copy view
   gravity_proj = robot.data.projected_gravity_b.torch  # torch.Tensor

   # Explicit warp access (for kernel interop)
   wp_array = robot.data.joint_pos.warp                # warp.array, the original buffer

   # Pass directly to warp kernels — no unwrapping needed
   wp.launch(my_kernel, inputs=[robot.data.joint_pos], ...)  # works via __cuda_array_interface__


The ``.torch`` and ``.warp`` Accessors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``.torch`` returns a cached, zero-copy :class:`torch.Tensor` view of the underlying warp array
  (via :func:`warp.to_torch`). The tensor is created on first access and reused on subsequent
  calls. Since it is a zero-copy view, writes to the tensor are visible through the warp array
  and vice versa.

- ``.warp`` returns the original :class:`warp.array`. Use this when you need to pass data to
  warp kernels explicitly or access warp-specific attributes (``ptr``, ``strides``, etc.).


Deprecation Bridge
~~~~~~~~~~~~~~~~~~

To ease migration, ``ProxyArray`` includes a deprecation bridge that allows existing code to
treat it as a ``torch.Tensor`` temporarily:

.. code-block:: python

   # These still work but emit a one-time DeprecationWarning:
   result = torch.sum(robot.data.joint_pos)           # via __torch_function__
   value = robot.data.joint_pos[0, 3]                 # via __getitem__
   result = robot.data.joint_pos + 1.0                # via __add__
   legacy = wp.to_torch(robot.data.joint_pos)         # temporary shim

   # Preferred (no warning):
   result = torch.sum(robot.data.joint_pos.torch)
   value = robot.data.joint_pos.torch[0, 3]
   result = robot.data.joint_pos.torch + 1.0
   tensor = robot.data.joint_pos.torch

The ``wp.to_torch()`` compatibility path is a temporary shim for code that was migrated
before ``ProxyArray`` exposed explicit accessors. It returns the same zero-copy tensor as
``.torch`` and emits a one-time ``DeprecationWarning``. The shim and the other deprecation
bridges will be removed in a future release. Migrate to explicit ``.torch`` access now.


Migrating from Isaac Lab 2.x
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Isaac Lab 2.x, data properties returned ``torch.Tensor`` directly. In 3.0, they return
``ProxyArray``. Append ``.torch`` to get the tensor:

.. code-block:: python

   # BEFORE (Isaac Lab 2.x) — properties returned torch.Tensor directly
   joint_pos = robot.data.joint_pos
   first_contact = sensor.compute_first_contact(dt)

   # AFTER (Isaac Lab 3.0) — properties return ProxyArray
   joint_pos = robot.data.joint_pos.torch
   first_contact = sensor.compute_first_contact(dt).torch

.. note::

   Passing a ``ProxyArray`` to ``wp.to_torch()`` is temporarily supported by a compatibility
   shim and returns the same cached zero-copy tensor as ``.torch``. This path emits a
   one-time ``DeprecationWarning`` and will be removed in a future release. Use ``.torch``
   in new code.


Backend Differences
~~~~~~~~~~~~~~~~~~~

While the ``ProxyArray`` interface is identical across backends, the underlying data refresh
model differs:

**PhysX (pull-to-refresh):**
  Properties pull fresh data from the PhysX tensor API on first access per simulation step,
  then cache the result. The underlying GPU buffers are stable and pre-allocated — the
  ``ProxyArray`` wrapper is created once and reused safely across steps.

**Newton (auto-refresh with wrapper replacement):**
  The simulation automatically refreshes GPU buffers each step. On full simulation resets,
  buffers may be re-created. The Newton backend creates new ``ProxyArray`` wrappers for the
  new warp arrays, invalidating any previously cached torch tensors.

In both cases, ``.torch`` always returns a view of the current simulation state for the
current step.

.. warning::

   Do not cache ``.torch`` results across simulation resets. After a reset (especially on
   Newton), previously obtained tensors may point to stale or freed GPU memory. Always
   re-access the property after a reset.


API Reference
~~~~~~~~~~~~~

.. autoclass:: ProxyArray
   :members:
   :undoc-members:
   :no-index:
