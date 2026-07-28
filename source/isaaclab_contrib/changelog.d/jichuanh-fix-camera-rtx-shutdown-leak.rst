Fixed
^^^^^

* Fixed ``VisuoTactileSensor`` never releasing the camera it owns. It called that camera's
  ``__del__`` directly, which runs the method body without destroying the object; it now
  forwards ``close()``.
