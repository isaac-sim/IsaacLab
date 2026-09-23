Added
^^^^^

* Added ``isaaclab.managers.DelayCfg`` as a standard manager term configuration, used through
  ``ObservationTermCfg(func=DelayCfg(term=..., params=...))`` with per-environment latency, refresh
  cadence, frame holds, and partial resets using the same buffer primitive as legacy delayed actuator targets.

Changed
^^^^^^^

* Replaced history shifting in ``DelayBuffer`` with ring storage and device-side indexing, including CUDA graph
  replay. Removed full-buffer clears and backfills on reset.

Fixed
^^^^^

* Fixed ``DelayBuffer.set_time_lag`` tensor subset updates to accept both supported integer dtypes.
