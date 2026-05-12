Added
^^^^^

* Added ``ovphysx`` preset to
  :class:`~isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg.LocomotionVelocityRoughEnvCfg.RoughPhysicsCfg`
  for use under the OVPhysX backend. ``RoughPhysicsCfg`` now exposes an
  ``ovphysx`` member so ``Isaac-Velocity-Rough-Anymal-D-v0`` selects the
  right physics + contact-sensor configuration when run with
  ``presets=ovphysx``.
