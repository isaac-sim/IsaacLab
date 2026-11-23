* Added :class:`~isaaclab.sensors.Lidar` for tiled lidar returns from Lidar generic wpm, with a
  solid-state or rotary scan. The default scan is rotary, matching an ``OmniLidar`` prim. Each environment keeps one Cartesian point
  slot per fired ray, in meters, in the configured ``WORLD`` or ``SENSOR`` frame. The rendered pose
  is the configured offset composed with the parent. A partial reset clears only the selected
  environments. Missed rays keep their slots.
