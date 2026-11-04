* **Breaking:** Changed :obj:`~isaaclab_assets.robots.franka.FRANKA_ROBOTIQ_GRIPPER_CFG` to spawn a corrected Franka
  asset and select its ``Robotiq_2F_85`` gripper variant, because the arm links of the Isaac Sim asset carry
  placeholder inertias (centers of mass at the link origins, inertias near zero). The asset is temporarily hosted on
  Hugging Face at a pinned revision; the path changes to the Isaac Lab Nucleus once it is uploaded there. The arm
  links take the inertial properties of the Isaac Lab Franka asset, the Franka hand body ``panda_hand`` is replaced by
  the flange body ``panda_link8`` that carries the gripper, the finger pads have convex-decomposition colliders with a
  friction material, and the arm effort limits are the Franka torque limits (87 and 12 N·m instead of 5200 and
  720 N·m). Rename ``panda_hand`` to ``panda_link8`` in body selectors and requalify policies trained with the
  previous configuration.
