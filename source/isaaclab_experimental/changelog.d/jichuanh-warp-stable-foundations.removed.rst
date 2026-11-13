* **Breaking:** Removed the Warp subclass ``isaaclab_experimental.managers.SceneEntityCfg``; the name now resolves to
  :class:`isaaclab.managers.SceneEntityCfg`, whose ``joint_ids_wp`` and ``body_ids_wp`` the Warp managers fill. Its
  ``joint_mask`` and ``from_stable`` are removed: Warp terms loop over ``joint_ids_wp``.
