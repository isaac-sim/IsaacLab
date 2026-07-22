Added
^^^^^

* Added :meth:`~isaaclab.envs.ManagerBasedRLEnvCfg.play_mode`,
  :meth:`~isaaclab.envs.DirectRLEnvCfg.play_mode` and
  :meth:`~isaaclab.envs.DirectMARLEnvCfg.play_mode` to define play-mode overrides on the
  environment configuration. Play scripts call this method after the configuration is loaded.
  The base implementations cap the number of environments at 50 and disable observation
  corruption/noise; task configurations can override the method to customize playback behavior.
