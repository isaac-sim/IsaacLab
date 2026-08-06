Added
^^^^^

* Added :class:`~isaaclab.actuators.ActuatorCollection` as the runtime
  actuator API, with separate command, processed joint-command, and telemetry
  views.
* Added execution aggregation for disjoint stateless actuator groups while
  preserving named group configuration and access.

Deprecated
^^^^^^^^^^

* Deprecated articulation-level actuator command setters and command and
  torque-telemetry properties on articulation data. Use the command view and
  ``computed_torque`` or ``applied_torque`` views on
  :attr:`~isaaclab.assets.Articulation.actuators` instead.

Changed
^^^^^^^

* Changed :class:`~isaaclab.actuators.ActuatorCollection` so named groups retain
  their configuration and access identity while compatible groups can share
  execution.
* Changed :attr:`~isaaclab.actuators.ImplicitActuatorCfg.velocity_limit` to
  populate the actuator soft velocity-limit view. Use ``velocity_limit_sim``
  to configure the solver velocity clamp.
* **Breaking:** Changed actuator collection membership to be fixed at
  construction. Configure groups through
  :attr:`~isaaclab.assets.ArticulationCfg.actuators` before constructing the
  articulation; runtime assignment to or deletion from
  :attr:`~isaaclab.assets.Articulation.actuators` raises :class:`TypeError`.
