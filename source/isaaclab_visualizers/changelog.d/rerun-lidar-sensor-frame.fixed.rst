* Fixed Rerun lidar points sitting beside the scanned geometry. Sensor-frame returns are
  expressed with ``+X`` forward, ``+Y`` left, and ``+Z`` up, and are now rewritten onto the prim's
  camera axes before the world pose is applied.
