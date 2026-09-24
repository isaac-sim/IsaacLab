Fixed
^^^^^

* Fixed joint-wrench sensors applying an extra frame transformation to PhysX readings, which already used
  the child-side joint frame and anchor. Removed the redundant USD frame buffers. Force and torque values
  changed for joints with non-identity child frames; the sensor's documented frame convention was preserved.
* Replaced circular frame-conversion checks with a shared Newton/PhysX integration test using a known mass,
  gravity, and lever arm to calculate the expected nonzero wrench independently.
