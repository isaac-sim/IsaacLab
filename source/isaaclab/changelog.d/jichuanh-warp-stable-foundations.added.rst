* Added the ``joint_ids_wp`` and ``body_ids_wp`` fields to :class:`~isaaclab.managers.SceneEntityCfg`: ``int32``
  Warp arrays of the resolved joint and body indices, in the order of ``joint_ids`` and ``body_ids``, for Warp terms.
  :meth:`~isaaclab.managers.SceneEntityCfg.resolve` fills them; ``joint_ids`` and ``body_ids`` are unchanged.
* Added :meth:`~isaaclab.utils.warp.ProxyArray.__copy__`, which wraps the same Warp array, and
  :meth:`~isaaclab.utils.warp.ProxyArray.__deepcopy__`, which wraps a copy of it. A deep copy returned a
  ``torch.Tensor`` and a shallow copy recursed without end.
* Added the ``|``, ``&``, ``^`` and ``~`` operators and the in-place operators to
  :class:`~isaaclab.utils.warp.ProxyArray`. They act on its torch view with the one-time deprecation warning, and the
  in-place ones keep the wrapper bound to its Warp memory.
* Added the ``ISAACLAB_PROXY_ARRAY_STRICT`` environment variable: set to ``1``, implicit torch use of a
  :class:`~isaaclab.utils.warp.ProxyArray` raises instead of warning, which lists the call sites to migrate to
  ``.torch`` before the deprecation bridge is removed after the next release.
