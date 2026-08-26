Fixed
^^^^^

* Fixed the :class:`~isaaclab.envs.ManagerBasedEnv`, :class:`~isaaclab.envs.DirectRLEnv` and
  :class:`~isaaclab.envs.DirectMARLEnv` destructors raising ``AttributeError: '...' object has no
  attribute '_is_closed'`` when environment construction failed before initialization completed.
  The original construction error is now reported instead of being masked by the destructor.
