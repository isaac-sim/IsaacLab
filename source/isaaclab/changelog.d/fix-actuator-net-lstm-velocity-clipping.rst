Fixed
^^^^^

* Fixed :class:`~isaaclab.actuators.ActuatorNetLSTM` clipping its output with a zero joint velocity. The DC-motor
  torque-speed limits now use the current joint velocity, as :class:`~isaaclab.actuators.ActuatorNetMLP` does.
* Passed measured joint velocity directly into DC-motor clipping, removing the cached velocity and its redundant
  copies. The base ``_clip_effort`` accepts ``(effort, *args, **kwargs)``; custom overrides used by explicit PD and
  neural-network actuators must accept ``(effort, joint_vel)``. Implicit actuator clipping still takes only effort.
