Changed
^^^^^^^

* Routed OVPhysX articulation actuator setup, compute, reset, and command
  submission through :class:`~isaaclab.actuators.ActuatorCollection`.

Added
^^^^^

* Added OVPhysX execution of supported native explicit actuators through the
  shared host adapter when
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` is enabled.
