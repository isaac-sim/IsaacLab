Added
^^^^^

* Added :class:`~isaaclab.actuators.ActuatorCollection` as the runtime
  actuator API for articulation actuator commands, telemetry, and
  actuator-resolved gains.

Deprecated
^^^^^^^^^^

* Deprecated articulation-level actuator command setters and actuator command
  properties on articulation data in favor of
  :attr:`~isaaclab.assets.Articulation.actuators`.
