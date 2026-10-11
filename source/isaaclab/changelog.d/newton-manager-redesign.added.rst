* Added :attr:`~isaaclab.actuators.newton.NewtonActuatorAdapter.computed_effort`,
  :attr:`~isaaclab.actuators.newton.NewtonActuatorAdapter.state_buffers`,
  :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.zero_outputs`, and
  :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.reset_dofs` so a compiled physics step can bind actuator
  history explicitly and serve worlds with different DOF layouts.
* Added :attr:`~isaaclab.scene.InteractiveScene.assets`, the scene assets in the order they write to the simulation,
  and a ``physics_manager`` argument to :class:`~isaaclab.envs.ManagerBasedEnv` and
  :class:`~isaaclab.envs.ManagerBasedRLEnv` for a manager the task prepares before its scene is built.
