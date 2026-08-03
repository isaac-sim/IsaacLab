Added
^^^^^

* Added :class:`~isaaclab.actuators.ActuatorCollection` as the runtime
  actuator API, with separate command and processed joint-command views,
  telemetry, and actuator-resolved gains.
* Added execution aggregation for disjoint stateless actuator groups while
  preserving named group configuration and access.

Deprecated
^^^^^^^^^^

* Deprecated articulation-level actuator command setters and actuator command
  properties on articulation data in favor of the command view on
  :attr:`~isaaclab.assets.Articulation.actuators`.
