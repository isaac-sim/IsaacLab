* Added ``FRANKA_PANDA_FLAT_CFG`` and ``FRANKA_PANDA_FLAT_HIGH_PD_CFG`` for the shared flat Franka
  asset. These configurations selected gripper-only colliders and the ``panda_arm`` actuator group;
  select the ``Physics`` variant for the target backend. Arm effort limits matched the Panda's
  87 Nm shoulder and 12 Nm forearm limits; checkpoints trained with different limits require requalification.
