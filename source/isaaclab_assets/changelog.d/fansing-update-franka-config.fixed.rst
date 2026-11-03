* Fixed :obj:`~isaaclab_assets.robots.franka.FRANKA_ROBOTIQ_GRIPPER_CFG` simulating placeholder arm inertias: it
  spawns :obj:`~isaaclab_assets.robots.franka.FRANKA_ROBOTIQ_USD_PATH`, the Isaac Sim Franka asset with the Robotiq
  2F-85 gripper whose arm links take the Isaac Lab Franka inertial properties, whose Franka hand is replaced by the
  gripper flange and whose finger pads have a friction material.
* Fixed :obj:`~isaaclab_assets.robots.franka.FRANKA_ROBOTIQ_GRIPPER_CFG` letting the arm fall on Newton, which ignores
  the disabled gravity of rigid bodies: MuJoCo Warp compensates gravity on its bodies and joints.
* Fixed the arm effort limits of :obj:`~isaaclab_assets.robots.franka.FRANKA_ROBOTIQ_GRIPPER_CFG` exceeding the Franka
  torque limits (5200 and 720 N·m instead of 87 and 12 N·m).
