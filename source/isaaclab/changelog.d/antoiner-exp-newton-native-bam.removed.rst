* **Breaking:** Removed ``isaaclab.actuators.actuator_bam_cfg.BAM_DRIVE_API``. BAM servos author Newton's
  ``NewtonVoltageControlAPI``, ``NewtonCurrentPWMClampingAPI`` and ``NewtonDCMotorAPI`` instead; look up BAM
  actuator prims by those schemas.
* **Breaking:** Removed ``BamActuatorCfg.vin_drop_gain_range`` [V/(N.m)], since Newton's battery drops its
  voltage with the supply current, not the motor torque. Use
  :attr:`~isaaclab.actuators.BamActuatorCfg.battery_resistance_range` [Ohm]; ``sag_gain * motor.kt`` gives the
  same drop per unit of motor current.
