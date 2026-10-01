* Fixed the air and contact times of :class:`~isaaclab_newton.sensors.ContactSensor` drifting with the age of the
  sensor clock. Each refresh added the difference of two growing float32 timestamps, which was off by up to 8e-5 s
  after 40 s of simulated time. It now adds the float64 time since the last refresh kept by the sensor base class.
