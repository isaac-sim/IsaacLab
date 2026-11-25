* Disabled static/kinematic-vs-kinematic SDF collision pairs (``include_static_kinematic_pairs=False``) and
  reduced MJWarp solver capacities (``njmax=1024``, ``nconmax=256``) in
  :class:`~isaaclab_tasks.contrib.nist.nist_env_cfg.NISTEnvCfg`, improving 4,096-env training throughput by ~31%
  with zero solver overflows.
