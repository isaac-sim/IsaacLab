Fixed
^^^^^

* Fixed ``DelayBuffer.set_time_lag`` leaving invalid lag values active after raising validation errors, so rejected
  updates no longer alter subsequent delayed outputs.
