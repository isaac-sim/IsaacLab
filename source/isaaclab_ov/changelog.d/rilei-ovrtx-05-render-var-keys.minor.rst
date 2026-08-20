Added
^^^^^

* Added support for ovrtx 0.5, which keys ``FrameOutput.render_vars`` by the absolute RenderVar prim
  path instead of the AOV source name. :class:`~isaaclab_ov.renderers.OVRTXRenderer` now selects the
  key layout from the installed ovrtx version, so ovrtx 0.4 remains supported unchanged.
