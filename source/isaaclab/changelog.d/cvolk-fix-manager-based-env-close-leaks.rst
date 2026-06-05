Fixed
^^^^^

* Fixed a memory leak in :meth:`~isaaclab.envs.ManagerBasedEnv.close`,
  :meth:`~isaaclab.envs.DirectRLEnv.close` and :meth:`~isaaclab.envs.DirectMARLEnv.close`
  where the cached observation buffers and the :class:`gym.spaces` observation/action
  spaces were never released, causing host and GPU memory to accumulate on each
  environment construct/teardown cycle.
