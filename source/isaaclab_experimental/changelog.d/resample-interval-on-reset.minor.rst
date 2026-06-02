Added
^^^^^

* Added support for :attr:`~isaaclab.managers.EventTermCfg.resample_interval_on_reset` in the
  experimental Warp-first event manager, allowing ``"interval"`` event terms to keep their
  per-environment timer across resets while still firing asynchronously per environment.
