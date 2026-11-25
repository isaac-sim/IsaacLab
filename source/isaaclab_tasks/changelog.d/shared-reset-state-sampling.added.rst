* Reused the core Lift task's existing ``conditional_reset`` event as the reset-bank reference.
  It kept initialization-time validation, diversity selection, and the 10-episode success history.
  It used shared native storage, sampling, and public mask writers for replay and controller targets.
  Captured replay was validated on Newton/MJWarp.
