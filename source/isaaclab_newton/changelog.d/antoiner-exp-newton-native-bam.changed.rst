* Changed Newton-executed BAM servos to run on Newton's :class:`~newton.actuators.DriveVoltageControlled`,
  :class:`~newton.actuators.ClampingCurrentPWM`, :class:`~newton.actuators.MotorDC`,
  :class:`~newton.actuators.BatteryConstantResistance`, :class:`~newton.actuators.InputProcessorDelay` and
  :class:`~newton.actuators.InputProcessorEncoder`. The Newton manager passes the model's batteries to the
  actuator adapter, which steps them after the actuators.
* Changed the first BAM friction budget of an episode to read no external load. The solver keeps a reset
  environment's loads until it solves again, so the budget read the terminated episode's last load. State writes
  without a reset, such as pushes, keep the load.
* Changed the BAM supply to one battery per actuator group and environment. Separate BAM groups of one
  robot now sag independently, also when Newton merges their actuators.
* Changed the BAM command lag to one random stream per actuator group and environment. Groups with the same
  delay settings no longer draw the same lag in an environment, and lag sequences differ from earlier releases.
* Changed the Newton dependency pin from ``42f7def1`` to ``b9452c2f``, the merge of newton#4504 and newton#4797 on
  a fork, with ``mujoco`` and ``mujoco-warp`` 3.16.
