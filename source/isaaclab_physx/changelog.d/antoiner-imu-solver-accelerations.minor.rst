Changed
^^^^^^^

* Changed the IMU and PVA sensors to read linear and angular accelerations from the PhysX solver
  (:meth:`RigidBodyView.get_accelerations`) instead of finite-differencing the body velocity
  between updates. The reported acceleration now includes the rigid-body transport terms for the
  sensor offset from the center of mass and no longer produces spurious spikes when velocities are
  written directly (for example on environment resets or teleports). The reading is available from
  the first sensor update and is independent of the sensor update period. The PVA sensor keeps
  reporting a kinematic acceleration (no gravity bias), so a body in free fall now reads ``-g``
  instead of a finite-difference estimate.

* Changed the PVA sensor's world-frame gravity direction from a public per-instance
  ``GRAVITY_VEC_W`` proxy array to the internal ``_gravity_vec_w`` scene-wide vector, matching the
  OvPhysX sensor. Gravity is scene-wide on this backend, so the per-instance buffer held one
  broadcast value. Code reading ``pva_sensor.GRAVITY_VEC_W`` should read
  ``pva_sensor.data.projected_gravity_b`` instead; the identically-named attribute on the asset
  data classes is unaffected.
