* Reused the core Lift task's existing ``conditional_reset`` event as the reset-bank reference.
  It kept initialization-time validation and diversity selection, and used shared native storage,
  sampling, and public mask writers for captured replay and controller targets.
