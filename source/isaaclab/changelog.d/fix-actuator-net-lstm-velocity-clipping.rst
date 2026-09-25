Fixed
^^^^^

* Fixed :class:`~isaaclab.actuators.ActuatorNetLSTM` clipping its output with a zero joint velocity. The DC-motor
  torque-speed limits now use the current joint velocity, as :class:`~isaaclab.actuators.ActuatorNetMLP` does.
* Passed measured joint velocity directly into actuator clipping, removing the cached velocity and its redundant
  copies. Custom overrides of the private ``_clip_effort`` method now take ``(effort, joint_vel)``.
