Changed
^^^^^^^

* Changed the IMU sensor to read rigid-body accelerations from the solver through the
  ``RIGID_BODY_ACCELERATION`` tensor binding when the installed ``ovphysx`` wheel provides it,
  including the transport terms for the sensor offset from the center of mass. On wheels without
  the binding the sensor keeps the previous finite-difference behavior.
