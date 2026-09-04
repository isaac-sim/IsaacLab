Fixed
^^^^^

* Fixed ``ContactSensor.compute_first_contact()`` and ``ContactSensor.compute_first_air()``
  silently missing most touchdown / lift-off transitions once the simulation clock grows
  (issue #7283). The air/contact timers are differences of float32 timestamps, so their
  quantization error grows with the clock magnitude and quickly exceeds the default
  ``abs_tol``. Transitions are now latched by the sensor update kernels together with the
  age of the ended phase, and the comparison carries an ``interval / 2`` margin instead of
  resting exactly on the boundary.
