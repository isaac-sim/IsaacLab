* Added support for :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` to the OVRTX renderer.
  With an integer ``M``, a camera sensor has ``ceil(num_envs / M)`` render products. Each one renders
  a contiguous range of environments into its own tiled atlas, and writes that range of the output buffers.
  All render products of a sensor use the same RenderVars.
* Added a split search to the OVRTX renderer for camera sensors with
  :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` set to ``"auto"``. If the first frame
  fails, the renderer halves the render products of one sensor and renders the frame again. With several
  sensors, it picks the one with the most atlas pixels times samples per pixel. A warning then gives
  the value that skips the search. The value ``"all"`` or an integer turns the search off for a sensor.
  The search needs OVRTX 0.7 or later, because only these versions fail a frame above the buffer limit.
