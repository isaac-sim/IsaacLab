* Added support for :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` to the OVRTX renderer.
  A camera sensor now has ``ceil(num_envs / max_envs_per_render_product)`` render products. Each one renders
  a contiguous range of environments into its own tiled atlas, and writes that range of the output buffers.
  All render products of a sensor use the same RenderVars.
* Added a split search to the OVRTX renderer for camera sensors with
  :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` set to ``None``. If the first frame
  fails, the renderer halves the largest render product and renders the frame again. A warning then gives
  the value that skips the search. The search needs OVRTX 0.7 or later, because only these versions fail
  a frame above the buffer limit.
