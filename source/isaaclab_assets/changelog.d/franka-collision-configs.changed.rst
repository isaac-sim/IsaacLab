* Enabled full primitive arm and gripper collisions in ``FRANKA_PANDA_CFG`` and its high-PD derivative.
  Added ``FRANKA_MINIMAL_CFG`` using the same USD and actuator settings with only hand and fingertip
  colliders for applications that do not need arm contacts. Requalify policies when changing collision scope.
