Changed
^^^^^^^

* Changed the IMU sensor to read linear and angular accelerations from the PhysX solver
  (:meth:`RigidBodyView.get_accelerations`) instead of finite-differencing the body velocity
  between updates. The reported acceleration now includes the rigid-body transport terms for the
  sensor offset from the center of mass and no longer produces spurious spikes when velocities are
  written directly (for example on environment resets or teleports). The reading is available from
  the first sensor update and is independent of the sensor update period.
