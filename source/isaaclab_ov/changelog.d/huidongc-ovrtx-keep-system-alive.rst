Fixed
^^^^^

* Set ``keep_system_alive=True`` on the internal OVRTX ``RendererConfig`` in
  :class:`~isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer` so the renderer
  system is not torn down prematurely during pytest sessions.
