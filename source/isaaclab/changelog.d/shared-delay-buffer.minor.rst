Added
^^^^^

* Added ``delay_min_lag`` and ``delay_max_lag`` to ``ObservationTermCfg``. Lags were sampled per environment
  at initialization and reset, then held for the episode, matching the delayed PD actuator's lag policy.
* Applied delay after observation modifiers, noise, clipping, and scaling, before history. Recorded samples
  advanced only with ``update_history=True``; extra reads left delay history unchanged.
* Shared ``DelayBuffer`` between observations and existing delayed actuators, retaining the actuator API and
  physics-step behavior. Added optional non-recording reads and replaced history shifting with device-indexed
  ring storage, including CUDA graph replay and isolated partial resets.

Fixed
^^^^^

* Fixed ``DelayBuffer.set_time_lag`` subset updates to accept both supported integer dtypes.
