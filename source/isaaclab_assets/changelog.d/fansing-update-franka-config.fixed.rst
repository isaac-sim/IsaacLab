* Fixed :obj:`~isaaclab_assets.robots.franka.FRANKA_ROBOTIQ_GRIPPER_CFG` sagging under gravity on Newton, which reads
  the disabled gravity of rigid bodies only on the physics scene: MuJoCo Warp now compensates gravity on the bodies,
  matching the weightless arm on PhysX.
