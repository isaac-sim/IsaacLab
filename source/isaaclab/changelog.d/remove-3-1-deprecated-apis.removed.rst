* **Breaking:** Removed the actuator limit aliases deprecated in Isaac Lab 3.0 for removal in 3.1,
  including configuration fields, constructor arguments, and runtime properties. Use
  ``actuator_effort_limit`` and ``actuator_velocity_limit`` for model limits, and
  ``joint_effort_limit`` and ``joint_velocity_limit`` for solver limits. For implicit actuator
  constructors, replace ``effort_limit`` with ``joint_effort_limit``. For implicit configurations
  that previously supplied only ``effort_limit``, set both ``actuator_effort_limit`` and
  ``joint_effort_limit`` to preserve both limits; replace a lone ``velocity_limit_sim`` with both
  ``joint_velocity_limit`` and ``actuator_velocity_limit`` to preserve the solver and soft limits.
