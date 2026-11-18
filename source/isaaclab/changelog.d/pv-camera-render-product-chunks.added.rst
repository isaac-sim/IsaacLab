* Added :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product`. An integer below the
  environment count splits one camera sensor into several render products, each with a smaller tiled
  atlas. Smaller atlases need smaller renderer buffers, so a small enough value keeps each buffer below
  the 4 GiB limit. The default ``"auto"`` starts with one render product, and the OVRTX renderer can
  split it after a failed first frame. The value ``"all"`` always keeps one render product. Only the
  OVRTX renderer accepts an integer.
* Added :attr:`~isaaclab.renderers.RendererCfg.supports_render_product_split`. A renderer that cannot split
  a camera sensor keeps it ``False``. The camera then rejects a split when it gets
  its renderer, before the scene is cloned.
