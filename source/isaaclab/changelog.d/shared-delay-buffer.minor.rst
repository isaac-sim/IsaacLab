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
* Replaced ``DelayedPDActuatorCfg`` and ``DelayedPDActuator`` classes with deprecated composition constructors,
  scheduled for removal in 3.2. **Breaking:** subclassing and type checks against these names are no longer
  supported. Wrap an ``IdealPDActuatorCfg`` (or custom controller config) in
  ``DelayCfg(on="input", min_lag=..., max_lag=..., resample="reset", term=...)`` instead.
  Controller fields now belong to ``cfg.term``. Removed delay from ``RemotizedPDActuator``; move its config's
  ``min_delay``/``max_delay`` to the same outer composition. Configured native actuators used the shared delay
  instead of installing a second Newton delay; both lag bounds and reset sampling now applied on every backend.

Fixed
^^^^^

* Fixed ``DelayBuffer.set_time_lag`` tensor subset updates to accept both supported integer dtypes.
