Changed
^^^^^^^

* Updated :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` to accept
  :class:`~isaaclab.utils.warp.ProxyArray` in :meth:`set_outputs` and :meth:`update_camera`,
  matching the updated :class:`~isaaclab.renderers.BaseRenderer` interface. Output buffers are
  reinterpreted directly from the ProxyArray's underlying warp array, removing the previous
  :func:`warp.from_torch` conversion path.
