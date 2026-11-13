* Added support for :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` to the OVRTX renderer.
  One camera sensor now owns ``ceil(num_envs / max_envs_per_render_product)`` render products that share
  one set of RenderVars and write dim-0 slices of the sensor's output buffers, so one tiled atlas no longer
  has to hold every environment.
* Added a render product search to the OVRTX renderer for camera sensors without
  :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product`. While the first frame fails, the
  renderer halves the largest render product and renders again. A warning then names the limit that
  skips the search. The search needs OVRTX 0.7 or later, which fails a frame above the buffer limit.
