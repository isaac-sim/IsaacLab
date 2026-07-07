Added
^^^^^

* Added per-substep state-force callbacks to the Newton physics manager for
  interactive simulation forces.

Fixed
^^^^^

* Avoided unsafe CUDA graph re-capture when interactive forces are enabled for
  fixed-grid MPM or the Newton-native actuator path by using eager execution.
