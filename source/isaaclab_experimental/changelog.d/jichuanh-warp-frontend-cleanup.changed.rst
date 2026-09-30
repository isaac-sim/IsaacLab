* **Breaking:** Removed ``ManagerCallSwitch``, ``ManagerCallMode`` and the ``MANAGER_CALL_CONFIG``
  environment variable. Warp environments construct their Warp managers directly and keep a manager's
  stages eager when one of its terms is not capturable.
* **Breaking:** Replaced ``WarpGraphCache.capture_or_replay`` with
  :meth:`~isaaclab_experimental.utils.WarpGraphCache.call`. Stages are recorded from the first environment
  step without an eager warm-up, so a stage runs exactly once per call.
