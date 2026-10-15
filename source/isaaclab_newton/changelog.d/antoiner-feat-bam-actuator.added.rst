* Added :mod:`isaaclab_newton.actuators`, the BAM servo parts Newton's native BAM path lacks: a
  per-environment random command lag, the gearbox friction budget, and a position-only backlash
  processor, plus :class:`~isaaclab_newton.physics.MjWarpActuatorBridge`, which couples the friction to
  MJWarp friction and load.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_pre_actuator_callback` and
  :meth:`~isaaclab_newton.physics.NewtonManager.register_solver_init_callback`.
