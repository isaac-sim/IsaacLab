Changed
^^^^^^^

* Updated :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` to accept
  :class:`~isaaclab.utils.warp.ProxyArray` in :meth:`set_outputs` and :meth:`update_camera`,
  matching the updated :class:`~isaaclab.renderers.BaseRenderer` interface. Output buffers are
  accessed via ``.warp`` directly, avoiding intermediate :func:`warp.from_torch` conversions.
