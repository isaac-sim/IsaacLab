Fixed
^^^^^

* Fixed :class:`~isaaclab.actuators.ActuatorNetLSTM` clipping its output with a zero joint velocity. The DC-motor
  torque-speed limits now use the current joint velocity, as :class:`~isaaclab.actuators.ActuatorNetMLP` does.
