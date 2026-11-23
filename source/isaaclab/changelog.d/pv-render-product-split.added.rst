* Added :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` to split a camera sensor over several
  render products. Each render product has a smaller tiled atlas, so a split can keep each renderer buffer of the
  atlas below the 4 GiB per-buffer limit. The default ``"all"`` keeps one render product. An integer sets the
  maximum number of environments in one render product. Only the OVRTX renderer accepts an integer.
* Added ``RendererCfg.supports_render_product_split`` for renderers that can split a camera sensor.
