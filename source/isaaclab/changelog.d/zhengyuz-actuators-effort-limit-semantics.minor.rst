Changed
^^^^^^^

* **Breaking:** Changed the effort-limit semantics of implicit actuators.
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit` now describes the actuator's
  rated force or torque reflected at the joint, while
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit_sim` remains the solver-level
  clamp. Setting both to different values is now valid instead of raising ``ValueError``.
  Configurations that set only one field, set equal values, or set neither behave as before.
