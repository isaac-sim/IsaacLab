* **Breaking:** Enabled full primitive arm and gripper collisions in ``FRANKA_PANDA_CFG`` and its high-PD
  derivative. Use ``FRANKA_MINIMAL_CFG`` or select ``Colliders=gripper_only`` to retain the previous reduced
  collision scope. Requalify policies against the chosen robot configuration.
