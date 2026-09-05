Fixed
^^^^^

* Fixed :meth:`~isaaclab.sensors.contact_sensor.BaseContactSensor.compute_first_contact` and
  :meth:`~isaaclab.sensors.contact_sensor.BaseContactSensor.compute_first_air` silently missing
  touchdowns and lift-offs once the simulation had run for a few seconds (issue #7283). Their
  ``abs_tol`` argument now defaults to ``None``, which resolves to half the sensor update interval
  instead of a fixed ``1e-8``. The old value was around 100x smaller than the float32 rounding
  error of the sensor clock, so most transitions were dropped. Callers that relied on the previous
  behavior can pass ``abs_tol=1e-8`` explicitly.
