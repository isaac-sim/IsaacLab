Fixed
^^^^^

* Fixed :class:`~isaaclab.actuators.ActuatorNetLSTM` clipping its output with a zero joint velocity. The DC-motor
  torque-speed limits now use the current joint velocity, as :class:`~isaaclab.actuators.ActuatorNetMLP` does.
* Removed redundant velocity copies from DC-motor and neural-network actuator clipping while preserving the
  measured joint velocity.
