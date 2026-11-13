* Added the ``joint_ids_wp`` and ``body_ids_wp`` fields to :class:`~isaaclab.managers.SceneEntityCfg`: ``int32``
  Warp arrays of the resolved joint and body indices, in the order of ``joint_ids`` and ``body_ids``, for Warp terms.
  The new :meth:`~isaaclab.managers.SceneEntityCfg.resolve_warp_ids`, which the Warp managers call after
  :meth:`~isaaclab.managers.SceneEntityCfg.resolve`, fills them; :meth:`~isaaclab.managers.SceneEntityCfg.resolve`,
  ``joint_ids`` and ``body_ids`` are unchanged. Copies of the configuration, :func:`copy.deepcopy` included, share
  them.
