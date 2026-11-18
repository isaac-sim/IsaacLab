* Added :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` to split a camera sensor over render
  products, which can keep each buffer below the 4 GiB limit. The default ``"auto"`` lets the OVRTX renderer
  split after a failed first frame, ``"all"`` keeps one render product, and an integer sets the maximum.
* Added :attr:`~isaaclab.renderers.RendererCfg.supports_render_product_split` for renderers that can split.
