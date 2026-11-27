* Added :attr:`~isaaclab.actuators.ActuatorBase.capturable`, which says whether a CUDA graph can replay an actuator's
  compute and masked reset with Torch work inside it: the implicit, ideal PD, and DC motor actuators. The delayed
  actuators, whose masked resets draw from Torch's generator, and the network actuators are not capturable.
* Added :meth:`~isaaclab.physics.PhysicsManager.supports_step_capture` and
  :meth:`~isaaclab.physics.PhysicsManager.capture`, with which environments that replay their steps as CUDA graphs
  record them as the physics backend records its own step, and :meth:`~isaaclab.sim.SimulationContext.after_replay`,
  which does the host work of the physics steps that a replayed graph ran, such as counting them.
