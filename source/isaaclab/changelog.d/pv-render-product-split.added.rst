* Added :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` to split a camera sensor over several
  render products. Each render product has a smaller tiled atlas, so a split can keep each renderer buffer of the
  atlas below the 4 GiB per-buffer limit. The default ``"auto"`` starts with one render product. If the first frame
  fails, the OVRTX renderer searches for a split by itself. ``"all"`` always keeps one render product. An integer
  sets the maximum number of environments in one render product, and only the OVRTX renderer accepts it.
* Added ``RendererCfg.supports_render_product_split`` for renderers that can split a camera sensor.
