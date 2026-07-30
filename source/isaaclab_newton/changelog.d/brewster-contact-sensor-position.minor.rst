Added
^^^^^

* Added support for :attr:`~isaaclab.sensors.ContactSensorCfg.track_contact_points` to the Newton
  contact sensor. When filter objects are configured, :attr:`~isaaclab_newton.sensors.contact_sensor.ContactSensorData.contact_pos_w`
  reports the average contact position per filter object, weighted by contact-force magnitude.
