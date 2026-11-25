* Added :attr:`~isaaclab.actuators.ActuatorBase.capturable`, which says whether a CUDA graph can replay an actuator's
  compute and masked reset with Torch work inside it. Delayed actuators qualify: their masked resets draw delays
  from a device stream, which a replayed graph advances. The network actuators, which subclass a capturable motor
  model, are not capturable.
* Added :meth:`~isaaclab.physics.PhysicsManager.supports_step_capture` and
  :meth:`~isaaclab.physics.PhysicsManager.capture`, with which environments that replay their steps as CUDA graphs
  record them as the physics backend records its own step, and :meth:`~isaaclab.sim.SimulationContext.after_replay`,
  which does the host work of the physics steps that a replayed graph ran, such as counting them.
