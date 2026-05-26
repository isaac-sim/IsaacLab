Fixed
^^^^^

* Fixed :class:`~isaaclab.sensors.ContactSensor` (and other lazy-eval sensors) returning
  stale pre-reset data when :meth:`~isaaclab.scene.InteractiveScene.reset` was called
  inside an environment step without a subsequent physics step (e.g. inside
  :meth:`~isaaclab.envs.ManagerBasedRLEnv._reset_idx`). The shared
  ``reset_envs_kernel`` now clears the per-env outdated flag instead of setting it,
  so an immediate read after reset returns the zeros that the sensor's ``reset``
  override wrote rather than re-fetching a physics buffer that has not been
  stepped since. The flag is re-armed by the next
  :meth:`~isaaclab.scene.InteractiveScene.update`.
