Fixed
^^^^^

* Fixed ``NewtonManager.update_visualization_state`` reusing a cached shadow-body index mapping
  from a different sim device (same body count, e.g. a ``--device cpu`` run following a
  ``--device cuda`` run within the same process). ``get_transforms``'s conversion kernel launches
  on the sim backend's device, so a stale mapping on the wrong device could crash with a CUDA
  illegal memory access. The cache is now also invalidated on a device change.
