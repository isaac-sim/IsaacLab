* **Breaking:** Removed ``ManagerCallSwitch``, ``ManagerCallMode`` and the ``MANAGER_CALL_CONFIG``
  environment variable. Warp environments construct their Warp managers directly and keep a manager's
  stages eager when one of its terms is not capturable.
* **Breaking:** Replaced ``WarpGraphCache.capture_or_replay`` with
  :meth:`~isaaclab_experimental.utils.WarpGraphCache.call`. Stages are recorded from the first environment
  step without an eager warm-up, so a stage runs exactly once per call.
* Changed the Warp manager-based environments to reset environments by boolean mask. The mask is converted
  to environment indices once per reset, and only when a command, curriculum or recorder term is active.
* Changed :class:`~isaaclab_experimental.envs.DirectRLEnvWarp` to skip its reset stage on steps where no
  environment terminated.
* Changed the Warp ``push_by_setting_velocity``, ``apply_external_force_torque``, ``reset_root_state_uniform``
  and ``randomize_rigid_body_com`` event terms to classes that allocate their buffers when the term is
  created. They read their ranges on every call, so a changed range applies to the next event.
* Changed the Warp event manager to record its reset and interval stages when a non-capturable term runs
  only at startup, such as ``randomize_rigid_body_com`` in the velocity tasks.
* Changed the Warp ``randomize_rigid_body_material`` and ``randomize_rigid_body_mass`` adapters to reject
  event modes other than ``startup``.
