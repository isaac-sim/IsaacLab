Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.sensors.contact_sensor.ContactSensor` registering the same leaf
  body once per matching ancestor when the sensor ``prim_path`` used a mid-path wildcard (for
  example ``Robot/.*/left_ankle_roll_link``), which inflated the sensor and filter counts and
  tripped the physics-cloned init guard. Body discovery now relies on the shared prim
  resolver's unique results.
