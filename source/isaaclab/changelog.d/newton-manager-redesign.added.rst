* Added a ``dof_count`` argument, :attr:`~isaaclab.actuators.newton.NewtonActuatorAdapter.computed_effort`,
  :attr:`~isaaclab.actuators.newton.NewtonActuatorAdapter.state_buffers`,
  :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.zero_outputs`, and
  :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.reset_dofs` so a compiled physics step can bind actuator
  history explicitly and serve worlds with different DOF layouts.
