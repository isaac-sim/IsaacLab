* Added :func:`~isaaclab.utils.env_mask_from_ids`, :func:`~isaaclab.utils.env_ids_from_mask`,
  :func:`~isaaclab.utils.env_selection_kwargs` and :func:`~isaaclab.utils.takes_env_mask` to select environments
  for terms that take either a mask or indices.
* Added :meth:`~isaaclab.envs.ManagerBasedEnv._reset_mask`, which reset the environments selected by a mask without
  synchronizing the device, and :attr:`~isaaclab.managers.ManagerBase.takes_env_ids`.
* Added the ``joint_mask``, ``fixed_tendon_mask`` and ``body_mask`` fields of
  :class:`~isaaclab.managers.SceneEntityCfg`, filled by :meth:`~isaaclab.managers.SceneEntityCfg.finalize`, so event
  terms selected joints, fixed tendons and bodies for mask writers without allocating on every call.
