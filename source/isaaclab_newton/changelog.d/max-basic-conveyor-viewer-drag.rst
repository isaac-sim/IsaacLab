Added
^^^^^

* Added per-substep state-force callbacks to the Newton physics manager for
  interactive simulation forces.

Fixed
^^^^^

* Avoided unsafe CUDA graph re-capture when interactive forces are enabled for
  fixed-grid MPM or the Newton-native actuator path by including startup
  callbacks in a deferred first capture. Late callback changes fall back to
  eager execution only when re-capture is unsupported.
