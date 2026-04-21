Changelog
---------

0.0.3 (2026-04-20)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added automatic dump of the fully-resolved env config to
  ``<log_dir>/params/resolved_env.yaml`` from
  :class:`~isaaclab_experimental.envs.ManagerBasedEnvWarp` and
  :class:`~isaaclab_experimental.envs.DirectRLEnvWarp` at the end of
  ``__init__`` via the shared :func:`~isaaclab.utils.io.dump_resolved_cfg`
  utility, matching the stable env base classes.


0.0.2 (2026-03-16)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :class:`~isaaclab_experimental.envs.DirectRLEnvWarp` not being recognized by
  RL library wrappers (e.g. :class:`~isaaclab_rl.rl_games.RlGamesVecEnvWrapper`) that
  check for :class:`~isaaclab.envs.DirectRLEnv` via ``isinstance``. Changed base class
  from :class:`gym.Env` to :class:`~isaaclab.envs.DirectRLEnv`; all methods are
  overridden so behavior is unchanged.


0.0.1 (2026-01-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the ``isaaclab_experimental`` package.
