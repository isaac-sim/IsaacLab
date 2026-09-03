Fixed
^^^^^

* Fixed the :class:`~isaaclab.envs.ManagerBasedEnv`, :class:`~isaaclab.envs.DirectRLEnv` and
  :class:`~isaaclab.envs.DirectMARLEnv` destructors emitting a spurious ``AttributeError: '...'
  object has no attribute '_is_closed'`` traceback when environment construction failed before
  initialization completed. The traceback was printed ahead of the real construction error.
