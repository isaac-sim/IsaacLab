* **Breaking:** Changed Newton-native actuator authoring to write an explicit actuator configuration onto
  a joint's USD-authored ``NewtonActuator`` prim when its drive schema matches, instead of replacing the
  prim. Values the configuration leaves unset (``None``, or joints a regex mapping does not match) now keep
  the asset's value instead of code defaults such as zero gains; the configuration type still removes
  clamping and delay schemas it does not model. Set a value explicitly to restore the previous result.
* **Breaking:** Changed the :attr:`~isaaclab.actuators.DelayedPDActuatorCfg.max_delay` default from 0 to
  None, which keeps a USD actuator's delay on the Newton path. Set ``max_delay=0`` to remove that delay.
  The Isaac Lab model treats None as 0.
* Changed :attr:`~isaaclab.actuators.DCMotorCfg.saturation_effort` to accept None, which keeps the
  saturation effort of a USD DC motor actuator on the Newton path.
* **Breaking:** Changed the :class:`~isaaclab.actuators.BamActuatorCfg` and
  :class:`~isaaclab.actuators.BamMotorCfg` fit and deployment fields to default to None, which keeps the
  USD BAM actuator's value, so a task can override one setting of a fitted servo. BAM authoring also stopped
  resetting the ``kp_scale``, ``kd_scale``, ``friction_scale``, ``sag_gain``, and ``delay_seed`` values of an
  existing BAM actuator. Set these fields explicitly, or author or randomize the runtime values, to keep the
  previous result. A joint without a BAM actuator still needs the motor fit, ``kp_fw``, and ``vin``.
* Changed Newton-native authoring of delayed PD actuators to stop writing ``newton:maxDelay``, which is not
  a ``NewtonActuatorDelayAPI`` attribute.
