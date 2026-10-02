* Added :class:`~isaaclab_newton.physics.MjWarpActuatorBridge` to support BAM servo actuators
  with solver-resolved gearbox friction and external-load feedback.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_pre_actuator_callback` and
  :meth:`~isaaclab_newton.physics.NewtonManager.register_solver_init_callback` for actuator updates
  before each physics step and initialization after the solver became available.
