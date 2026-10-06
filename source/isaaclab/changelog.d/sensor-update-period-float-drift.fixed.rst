* Fixed :class:`~isaaclab.sensors.SensorBase` refreshing sensors later than their ``update_period`` once the
  sensor clock had run for some seconds. Whether a sensor is due was decided by subtracting two float32
  timestamps that grow with the episode, and their rounding error exceeded the ``1e-6`` tolerance, so sensors
  with ``update_period`` set to a multiple of the physics time step skipped refreshes (for example, the height
  scanner of the locomotion tasks from 16 s into an episode, and sensors with long update periods at small
  time steps within the first second). The time since the last refresh is now accumulated in float64.
