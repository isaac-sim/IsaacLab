* Added ``env_mask`` to the ``reset`` of every manager and :class:`~isaaclab.managers.EventManager.apply` (a
  :class:`~isaaclab.utils.warp.ProxyArray`), and to :class:`~isaaclab.scene.InteractiveScene`, the assets and the
  actuator models (a ``wp.array`` of ``wp.bool``).
* Added :func:`~isaaclab.utils.env_mask_from_ids`, :func:`~isaaclab.utils.env_ids_from_mask`,
  :func:`~isaaclab.utils.takes_env_mask` and :func:`~isaaclab.utils.env_selection_kwargs`, which dispatch an
  environment selection to a mask-native or an index-based callable.
* Added :meth:`~isaaclab.terrains.TerrainImporter.update_env_origins_mask`.
