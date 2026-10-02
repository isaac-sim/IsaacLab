* Added :class:`~isaaclab.actuators.BamActuatorCfg` and :class:`~isaaclab.actuators.BamMotorCfg`
  for identified servo models with voltage control, independently seeded per-environment command delay,
  and load-dependent friction.
  BAM required native Newton actuator execution with the MJWarp solver and an explicit firmware current limit.
* Added per-environment voltage and supply-sag sampling, plus runtime parameter access through
  :func:`~isaaclab.actuators.newton.read_group_parameter` and
  :func:`~isaaclab.actuators.newton.write_group_parameter`, including friction randomization via reset events.
* Updated native actuators to Newton 1.6's drive API. Group parameter helpers used ``"drive"``
  instead of ``"controller"``; direct Newton access used ``actuator.drive``.
