Removed
^^^^^^^

* Removed per-robot ``play_mode()`` overrides from Cassie, G1, and H1 rough velocity
  environment configs. The base :meth:`~isaaclab_tasks.core.velocity.LocomotionVelocityRoughEnvCfg.play_mode`
  is now used directly without robot-specific command range overrides.
