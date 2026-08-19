Changed
^^^^^^^

* Changed the Unitree Go2 velocity tasks to execute their DC motor actuators
  through the backend-native path by default
  (:attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` is now ``True``).
  Set ``env.sim.use_newton_actuators=false`` to restore Isaac Lab-side actuator
  execution.
