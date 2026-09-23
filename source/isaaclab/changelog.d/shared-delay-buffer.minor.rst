Added
^^^^^

* Added ``isaaclab.utils.DelayCfg`` for observations, policy actions, and actuator commands or efforts,
  including nested composition, independent sampling, cadence, holds, and partial resets. Observation terms
  use ``ObservationTermCfg(func=DelayCfg(term=..., params=...))``; actions and actuators wrap their term config
  directly. Input actuator delay retained current joint feedback; output delay held the computed effort.
* Added callable actuator and delay-buffer execution while preserving ``compute()`` callers and overrides.
  Joint action callables returned commands for submission after wrapper evaluation.

Changed
^^^^^^^

* Replaced history shifting in ``DelayBuffer`` with ring storage and device-side indexing, including CUDA graph
  replay. Removed full-buffer clears and backfills on reset.

Fixed
^^^^^

* Fixed ``DelayBuffer.set_time_lag`` tensor subset updates to accept both supported integer dtypes.
