Fixed
^^^^^

* Fixed intermittent segmentation faults after a CUDA graph capture by restoring the full
  collection that runs when the capture window ends. Scoping it to generation 0 left cycles
  that were promoted before the window but became unreachable during it -- a previous
  ``wp.Graph``/``State`` released on a hard reset, for instance -- to the periodic collector,
  which freed their Warp arrays long after the capture stream was destroyed.
