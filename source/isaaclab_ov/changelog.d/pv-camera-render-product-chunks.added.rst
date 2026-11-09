* Added support for :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` to the OVRTX renderer.
  One camera sensor now owns ``ceil(num_envs / max_envs_per_render_product)`` render products that share
  one set of RenderVars and write dim-0 slices of the sensor's output buffers, so one tiled atlas no longer
  has to hold every environment.
