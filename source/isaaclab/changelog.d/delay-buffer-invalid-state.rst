Fixed
^^^^^

* Fixed ``DelayBuffer.set_time_lag`` leaving invalid per-batch delays in the buffer by validating requested
  values before assignment, without cloning the live lag tensor.
