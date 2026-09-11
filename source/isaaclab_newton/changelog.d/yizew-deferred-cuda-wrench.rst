Fixed
^^^^^

* Fixed deferred Newton CUDA graph capture with Kit/RTX so the internal warmup step no longer leaks
  state into the first real graph replay.
