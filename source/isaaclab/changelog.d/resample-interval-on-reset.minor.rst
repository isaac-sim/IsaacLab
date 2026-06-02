Added
^^^^^

* Added :attr:`~isaaclab.managers.EventTermCfg.resample_interval_on_reset` to allow ``"interval"``
  event terms to keep their per-environment timer across resets while still firing asynchronously
  per environment. Defaults to ``True`` to preserve the existing behavior.
