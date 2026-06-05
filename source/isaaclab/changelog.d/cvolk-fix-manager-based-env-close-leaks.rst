Fixed
^^^^^

* Fixed a memory leak in :meth:`~isaaclab.envs.ManagerBasedEnv.close` where the cached
  observation buffer and the :class:`gym.spaces.Box` observation/action spaces were never
  released, causing host and GPU memory to accumulate on each environment
  construct/teardown cycle.
