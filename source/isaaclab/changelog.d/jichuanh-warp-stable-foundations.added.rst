* Added :meth:`~isaaclab.managers.SceneEntityCfg.resolve_warp`, which fills the new ``joint_ids_wp``,
  ``joint_mask_wp``, ``body_ids_wp`` and ``body_mask_wp`` fields of :class:`~isaaclab.managers.SceneEntityCfg`
  with Warp index and mask arrays of the resolved selection. Managers that run Warp terms call it after
  :meth:`~isaaclab.managers.SceneEntityCfg.resolve`.
