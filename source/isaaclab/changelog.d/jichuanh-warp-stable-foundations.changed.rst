* **Breaking:** ``len()`` of a :class:`~isaaclab.utils.warp.ProxyArray` raises a ``TypeError``; use
  ``.shape[0]``. Torch read a sized ``ProxyArray`` shorter than 32 elements used as an index as a tuple of
  indices, so ``tensor[proxy]`` selected the wrong elements. It now selects like the wrapped array.
* :func:`~isaaclab.utils.warp.memory.resolve_ids`, :func:`~isaaclab.utils.warp.memory.resolve_1d_indices` and
  :func:`~isaaclab.utils.warp.memory.resolve_1d_mask` accept a :class:`~isaaclab.utils.warp.ProxyArray` selector
  and use its Warp array, so asset write methods take the selections of
  :class:`~isaaclab.managers.SceneEntityCfg` as they are. ``resolve_ids`` rejected a ``ProxyArray``.
