Fixed
^^^^^

* Corrected the SO101 Keyboard task's initial pose for the SysID asset: rotated
  the robot base toward the keyboard and restored the task's zero joint-position
  reset-IK seed. Shared SO101 defaults, actuator parameters, and reset IK budgets
  remained unchanged. Removed the need for a manual task-specific base rotation.
