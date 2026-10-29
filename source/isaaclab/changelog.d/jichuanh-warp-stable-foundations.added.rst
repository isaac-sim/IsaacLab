* Added :meth:`~isaaclab.managers.SceneEntityCfg.resolve_warp`, which fills the new ``joint_indices``,
  ``joint_mask``, ``body_indices`` and ``body_mask`` fields of :class:`~isaaclab.managers.SceneEntityCfg`
  with :class:`~isaaclab.utils.warp.ProxyArray` index and mask arrays of the resolved selection. Warp terms read
  them through ``.warp``. Managers that run Warp terms call it after
  :meth:`~isaaclab.managers.SceneEntityCfg.resolve`. Copies of the configuration share them.
* Added :meth:`~isaaclab.utils.warp.ProxyArray.__copy__`, which wraps the same Warp array, and
  :meth:`~isaaclab.utils.warp.ProxyArray.__deepcopy__`, which wraps a copy of it. A deep copy returned a
  ``torch.Tensor`` and a shallow copy recursed without end.
* Added :mod:`isaaclab.utils.warp.capture`, which records CUDA graphs of Warp stages:
  :class:`~isaaclab.utils.warp.capture.CapturedStage` and the :func:`~isaaclab.utils.warp.capture.captured` and
  :func:`~isaaclab.utils.warp.capture.eager` helpers. They moved from ``isaaclab_experimental``.
