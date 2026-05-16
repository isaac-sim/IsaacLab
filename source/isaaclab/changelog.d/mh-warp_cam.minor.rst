Changed
^^^^^^^

* Changed :class:`~isaaclab.sensors.camera.CameraData` to expose all sensor buffers as
  :class:`~isaaclab.utils.warp.ProxyArray` instead of :class:`torch.Tensor`. The fields
  :attr:`~isaaclab.sensors.camera.CameraData.pos_w` (``wp.vec3f``),
  :attr:`~isaaclab.sensors.camera.CameraData.quat_w_world` (``wp.quatf``),
  :attr:`~isaaclab.sensors.camera.CameraData.intrinsic_matrices` (``wp.mat33f``), and all
  entries in :attr:`~isaaclab.sensors.camera.CameraData.output` are now backed by warp arrays.
  Use ``.torch`` for a zero-copy :class:`torch.Tensor` view or ``.warp`` to pass the array
  directly to a warp kernel. Existing code using these fields as tensors (indexing, arithmetic,
  :func:`torch.testing.assert_close`, etc.) continues to work via the
  :class:`~isaaclab.utils.warp.ProxyArray` deprecation bridge with a one-time
  :class:`DeprecationWarning`.
* Updated :meth:`~isaaclab.renderers.BaseRenderer.set_outputs` and
  :meth:`~isaaclab.renderers.BaseRenderer.update_camera` in :class:`~isaaclab.renderers.BaseRenderer`
  to accept :class:`~isaaclab.utils.warp.ProxyArray` arguments instead of :class:`torch.Tensor`.
