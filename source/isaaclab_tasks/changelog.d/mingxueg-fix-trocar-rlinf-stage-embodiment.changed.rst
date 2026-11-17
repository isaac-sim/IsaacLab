* **Breaking:** Replaced ``G1_29DOF_BODY_JOINT_INDICES`` and ``G1_DEX3_JOINT_INDICES`` in the assemble trocar
  config with ``G1_29DOF_BODY_JOINT_NAMES`` and ``G1_DEX3_JOINT_NAMES``. The observations now select joints by
  name, since the joint order differs between physics backends.
