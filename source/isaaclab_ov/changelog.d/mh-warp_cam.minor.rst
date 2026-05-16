Changed
^^^^^^^

* Updated :class:`~isaaclab_ov.renderers.OVRTXRenderer` to accept
  :class:`~isaaclab.utils.warp.ProxyArray` in :meth:`set_outputs` and :meth:`update_camera`,
  matching the updated :class:`~isaaclab.renderers.BaseRenderer` interface. Output buffers are
  accessed via their underlying warp array directly.
