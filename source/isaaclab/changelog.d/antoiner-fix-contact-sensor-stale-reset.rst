Fixed
^^^^^

* Fixed lazy-eval sensors (contact, IMU, PVA, joint-wrench across PhysX and Newton)
  returning stale pre-reset data when :meth:`~isaaclab.scene.InteractiveScene.reset` was
  called inside an environment step without a subsequent physics step (e.g. inside
  :meth:`~isaaclab.envs.ManagerBasedRLEnv._reset_idx`). Each step-dependent sensor's
  update kernel now skips envs whose ``timestamp`` is still ``0`` (the signal that no
  physics step has occurred since the last reset), so the next ``data`` access returns
  the values that the sensor's ``reset()`` populated rather than re-reading a physics
  buffer that holds pre-reset values.
