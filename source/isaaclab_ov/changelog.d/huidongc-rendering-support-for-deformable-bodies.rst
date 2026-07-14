Added
^^^^^

* Added deformable body rendering support in
  :class:`~isaaclab_ov.renderers.OVRTXRenderer` for Newton surface and volume
  deformables. :meth:`~isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer.update_geometries`
  syncs ``particle_q`` mesh points into OVRTX bindings each frame through
  asynchronous zero-copy handoffs.
