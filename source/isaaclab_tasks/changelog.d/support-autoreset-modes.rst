Fixed
^^^^^

* Changed :class:`~isaaclab_tasks.core.dexsuite.dexsuite_env_cfg.DexsuiteReorientEnvCfg` to derive
  from :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` instead of
  :class:`~isaaclab.envs.ManagerBasedEnvCfg`, so the RL-specific configuration fields are inherited
  rather than set ad hoc in ``__post_init__``.
