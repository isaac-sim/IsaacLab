* Added :class:`~isaaclab.sensors.Lidar` for tiled OVRTX solid-state lidar returns. Each environment
  keeps one Cartesian point slot per ray, in meters, in the configured ``WORLD`` or ``SENSOR`` frame.
  The rendered pose is the configured offset composed with the parent. A partial reset clears only
  the selected environments. Missed rays keep their slots.
