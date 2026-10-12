* Added :class:`~isaaclab.utils.StepGraph` and :class:`~isaaclab.utils.StepOp`: a step as straight-line operations
  whose graphable runs replay from CUDA graphs and whose other operations run eagerly in place.
* Added :meth:`~isaaclab.physics.PhysicsManager.step_ops`, :meth:`~isaaclab.physics.PhysicsManager.capture_graph` and
  :meth:`~isaaclab.physics.PhysicsManager.finish_step`, and :meth:`~isaaclab.sim.SimulationContext.finish_physics_step`,
  so a caller can schedule and capture the physics step together with its own work. The defaults expose nothing.
