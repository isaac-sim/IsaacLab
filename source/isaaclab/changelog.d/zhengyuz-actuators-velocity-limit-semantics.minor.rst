Changed
^^^^^^^

* **Breaking:** Changed the velocity-limit semantics of implicit actuators.
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit` now describes the joint's peak
  velocity: it populates the articulation data buffers (e.g.
  :attr:`~isaaclab.assets.ArticulationData.soft_joint_vel_limits`, read by velocity-limit
  terminations and rewards) and is not pushed to the physics solver, while
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit_sim` remains the solver-level
  clamp. Setting both on an implicit actuator is now valid (previously a ``ValueError``).
  Configurations that set only ``velocity_limit_sim`` or neither attribute behave exactly as
  before. Configurations that set only ``velocity_limit`` change behavior: the value was
  previously ignored for implicit actuators (with a deprecation warning) and now feeds the
  velocity-limit data buffers, which affects velocity-limit terminations and rewards; a
  warning is emitted at actuator construction in this case. To keep the old behavior, remove
  ``velocity_limit`` from the actuator configuration; to impose a solver-level clamp, use
  ``velocity_limit_sim``.
