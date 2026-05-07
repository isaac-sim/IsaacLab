Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab.sensors.camera.CameraData` now stores its array fields
  as :class:`~isaaclab.utils.warp.proxy_array.ProxyArray` wrappers around
  :class:`warp.array` primary buffers. The change covers
  :attr:`~isaaclab.sensors.camera.CameraData.pos_w`,
  :attr:`~isaaclab.sensors.camera.CameraData.quat_w_world`,
  :attr:`~isaaclab.sensors.camera.CameraData.intrinsic_matrices`,
  every entry in :attr:`~isaaclab.sensors.camera.CameraData.output`, and the
  derived :attr:`~isaaclab.sensors.camera.CameraData.quat_w_ros` /
  :attr:`~isaaclab.sensors.camera.CameraData.quat_w_opengl` properties.
  Migration: read torch tensors via the ``.torch`` accessor (zero-copy) and pass
  ``.warp`` to warp kernels (e.g. ``camera.data.output["rgb"].torch``,
  ``camera.data.pos_w.warp``). Existing code that uses the field directly as a
  tensor still works through :class:`ProxyArray`'s deprecation bridge but emits
  a one-time :class:`DeprecationWarning`.
* **Breaking:** :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCasterCameraData`
  ``image_mesh_ids`` is likewise a :class:`ProxyArray` over a ``wp.int16`` warp
  buffer; access via ``.torch`` for tensor reads.
* **Breaking:** :attr:`~isaaclab.sensors.camera.Camera.frame` and
  :attr:`~isaaclab.sensors.ray_caster.RayCasterCamera.frame` now return a
  :class:`ProxyArray` over a ``wp.int64`` array instead of a ``torch.Tensor``.
  Migration: use ``camera.frame.torch`` for the existing tensor view.
* **Breaking:**
  :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs` and
  :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_camera` now type
  their array parameters as :class:`warp.array` rather than
  :class:`torch.Tensor`. Renderer subclasses receive the underlying
  :attr:`ProxyArray.warp` buffers directly from the camera; consumers that
  implemented :class:`BaseRenderer` must update their signatures.
* :meth:`~isaaclab.sensors.camera.Camera.set_intrinsic_matrices`,
  :meth:`~isaaclab.sensors.camera.Camera.set_world_poses`, and
  :meth:`~isaaclab.sensors.camera.Camera.set_world_poses_from_view` (plus their
  :class:`~isaaclab.sensors.ray_caster.RayCasterCamera` analogues) now accept
  ``wp.array | ProxyArray | torch.Tensor | numpy.ndarray`` inputs, with
  :class:`warp.array` documented as the canonical type. Existing
  ``torch.Tensor`` callers continue to work without changes.
