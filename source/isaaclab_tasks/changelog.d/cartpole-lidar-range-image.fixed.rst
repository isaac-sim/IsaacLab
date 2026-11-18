* Fixed ``presets=lidar`` so the policy can learn from the scan. The actor now reads a stacked
  range image through the camera CNN instead of a flat list of every point.
  **Breaking:** the policy observation is ``(2, 49, 161)`` instead of the joint state concatenated
  with every lidar coordinate. The critic still reads the joint state.
