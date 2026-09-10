Changed
^^^^^^^

* Changed :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` to ``True`` by default so
  supported explicit actuator models use the native Newton actuator path. Set it to ``False``
  to restore the deprecated Isaac Lab execution path.
