Changed
^^^^^^^

* Changed :attr:`~isaaclab.actuators.DCMotorCfg.saturation_effort` to accept a joint-name-pattern
  dictionary in addition to a scalar. Joints in one actuator group that sit behind different gear
  reductions can now be given their own stall torque, which previously required splitting them into
  separate actuator groups. Existing scalar configurations are unaffected.
