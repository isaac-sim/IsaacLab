Changed
^^^^^^^

* **Breaking:** Changed the velocity-limit semantics of implicit actuators.
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit` now describes the motor's rated
  speed: it populates the articulation data buffers (e.g.
  :attr:`~isaaclab.assets.ArticulationData.soft_joint_vel_limits`, read by velocity-limit
  terminations and rewards) and is not pushed to the physics solver, while
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit_sim` remains the solver-level
  clamp. Setting both on an implicit actuator is now valid (previously a ``ValueError``), and
  ``velocity_limit`` alone is no longer silently ignored with a deprecation warning. Existing
  configurations that set only one of the two attributes behave exactly as before.
