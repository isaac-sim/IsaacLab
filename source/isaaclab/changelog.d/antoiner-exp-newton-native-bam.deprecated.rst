* Deprecated :attr:`~isaaclab.actuators.BamActuatorCfg.vin_drop_gain_range` [V/(N.m)] for removal in Isaac Lab
  3.2, since Newton's battery drops its voltage with the supply current, not the motor torque. Use
  :attr:`~isaaclab.actuators.BamActuatorCfg.battery_resistance_range` [Ohm] instead. A set sag gain range is
  converted to ``vin_drop_gain_range * motor.kt`` with a ``DeprecationWarning``; that drops the voltage per
  ampere of supply current as the sag gain did per ampere of motor current, but the duty cycle weights the
  supply current, so the supply sags less.
