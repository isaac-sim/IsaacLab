* Added :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product`. It splits one camera sensor into
  several render products, each with a smaller tiled atlas. With many environments, this keeps each
  renderer buffer below the 4 GiB limit. The default ``None`` starts with one render product. Only the
  OVRTX renderer supports the field.
* Added :attr:`~isaaclab.renderers.RendererCfg.supports_render_product_split`. A renderer that cannot split
  a camera sensor keeps it ``False``. The camera then rejects a split when it gets
  its renderer, before the scene is cloned.
