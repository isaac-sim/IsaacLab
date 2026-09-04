Fixed
^^^^^

* Reduced camera-task runtime overhead by fusing image normalization with layout conversion,
  avoiding a device-to-host camera-mask check for sensors updated every step, and removing an
  unnecessary concatenation copy for single-term observation groups.
