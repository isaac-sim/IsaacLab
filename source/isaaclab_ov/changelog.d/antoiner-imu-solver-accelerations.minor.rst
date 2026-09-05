Changed
^^^^^^^

* Changed the IMU and PVA sensors to read rigid-body accelerations from the solver through the
  ``RIGID_BODY_ACCELERATION`` tensor binding, including the transport terms for the sensor offset
  from the center of mass, instead of finite-differencing the body velocity between updates. The
  reported acceleration is available from the first update, is independent of the sensor update
  period, and no longer spikes when velocities are written directly (for example on environment
  resets or teleports).
