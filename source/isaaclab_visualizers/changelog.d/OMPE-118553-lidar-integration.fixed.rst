* Fixed sensor-frame lidar hits sitting beside the scanned geometry. ``SENSOR`` returns use
  ``+X`` forward, ``+Y`` left, and ``+Z`` up, and are rewritten onto the prim's camera axes before
  the world pose is applied.
