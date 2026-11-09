* Added :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` to split one camera sensor into
  several render products with smaller tiled atlases, so large environment counts stay below the renderer's
  4 GiB per-buffer limit. ``None`` keeps one render product. Only the OVRTX renderer supports the field.
* Added :attr:`~isaaclab.renderers.RendererCfg.supports_render_product_split` so a camera configuration
  rejects the split at configuration time for renderers that render through one render product.
