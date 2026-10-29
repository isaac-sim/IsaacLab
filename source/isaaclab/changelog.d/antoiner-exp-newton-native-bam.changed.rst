* Changed :class:`~isaaclab.actuators.BamActuatorCfg` authoring to Newton's native BAM composition, with the
  tokens of newton-usd-schemas#97: ``NewtonVoltageControlAPI`` (``newton:kp`` is ``kp_fw`` and
  ``newton:errorGain`` the fit's ``error_gain``), ``NewtonCurrentPWMClampingAPI``, ``NewtonDCMotorAPI``, a
  ``NewtonActuatorDelayAPI`` buffer sized by ``max_delay``, ``NewtonActuatorEncoderAPI`` bound through
  ``newton:outputJoint`` to the play hinge, and one ``NewtonConstantResistanceBattery`` prim per group that
  the servos draw from through ``newton:battery``. The stiff friction constraint is authored as the servo
  joints' ``mjc:solreffriction`` and ``mjc:solimpfriction``. Gearbox friction and the command lag are not
  authored; Isaac Lab runs them from the configuration.
* **Breaking:** Changed the BAM supply sag to the law of Newton's ``BatteryConstantResistance``: each
  environment's voltage drops with the signed current its group's servos draw during their PWM on-time,
  ``max(vin - R * max(sum_j duty_j * I_j, 0), vin_min, 0)``, instead of ``vin - sag_gain * sum_j |tau_j|``. A
  back-driven servo now returns current to the others. Set the battery resistance instead of the sag gain;
  ``sag_gain * motor.kt`` gives the same drop per unit of motor current. Policies trained with sag see a
  different supply and need retraining.
* **Breaking:** Changed BAM servos to apply no effort clamp after the motor, and
  :class:`~isaaclab.actuators.BamActuatorCfg` to reject ``actuator_effort_limit``. Newton has no stage after
  its motors, and its clamps bound a motor's voltage command. The stall-torque cap
  ``max(vin_range) * kt / resistance`` no longer applies; the firmware current limit holds the torque at
  ``kt * max_current`` unless the PWM limit overrides it at high joint speed. Limit the torque through
  ``motor.max_current`` and ``motor.max_pwm`` instead.
* Changed :class:`~isaaclab.actuators.newton.NewtonActuatorAdapter` to report the applied motor torque as the
  computed effort of actuators with a motor, whose drive computes a voltage, not an effort.
* Changed :class:`~isaaclab.actuators.newton.NewtonActuatorAdapter` to build Newton actuators with
  :class:`~newton.actuators.InputProcessorDelay` instead of the deprecated ``Delay``, and with their authored
  motor, on its ``vin``, on PhysX and OVPhysX.
