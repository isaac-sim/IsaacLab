* :func:`~isaaclab.utils.warp.memory.resolve_ids`, :func:`~isaaclab.utils.warp.memory.resolve_1d_indices` and
  :func:`~isaaclab.utils.warp.memory.resolve_1d_mask` accept a :class:`~isaaclab.utils.warp.ProxyArray` selector
  and use its Warp array, so asset write methods take the selections of
  :class:`~isaaclab.managers.SceneEntityCfg` as they are. ``resolve_ids`` rejected a ``ProxyArray``.
