Added
^^^^^

* Added :meth:`~isaaclab.sensors.SensorBase._mark_envs_up_to_date` to suppress the lazy
  ``_update_outdated_buffers`` refetch for a set of envs. Step-dependent sensors call this
  from their ``reset()`` override after populating ``_data`` with post-reset values
  (typically zeros), so the next ``data`` access returns those values rather than
  re-fetching a physics buffer that has not been stepped since the reset.

Fixed
^^^^^

* Fixed lazy-eval sensors (contact, IMU, PVA, joint-wrench across PhysX and Newton)
  returning stale pre-reset data when :meth:`~isaaclab.scene.InteractiveScene.reset` was
  called inside an environment step without a subsequent physics step (e.g. inside
  :meth:`~isaaclab.envs.ManagerBasedRLEnv._reset_idx`).
