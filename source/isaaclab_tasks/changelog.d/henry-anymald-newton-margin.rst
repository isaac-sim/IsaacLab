Fixed
^^^^^

* Fixed ANYmal-D rough-terrain locomotion terminating on spurious base contacts under the
  Newton MJWarp backend by setting the Newton shape margin to zero in
  :class:`~isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg.AnymalDRoughEnvCfg`.
  Other robots keep the shared default.
