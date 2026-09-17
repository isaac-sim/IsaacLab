Fixed
^^^^^

* Fixed :attr:`~isaaclab.envs.mimic_env_cfg.DataGenConfig.max_num_failures` being ignored. The field
  documented a cap on failed generation attempts but was never read, so a run with
  :attr:`~isaaclab.envs.mimic_env_cfg.DataGenConfig.generation_guarantee` enabled retried without
  bound on a task with a low success rate. Its default is now ``None`` (no limit) and setting it to
  an integer stops generation once that many attempts have failed.
