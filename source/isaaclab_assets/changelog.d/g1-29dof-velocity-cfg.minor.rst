Added
^^^^^

* Added :data:`~isaaclab_assets.robots.unitree.G1_29DOF_VELOCITY_CFG`, the 29-DoF counterpart of
  :data:`~isaaclab_assets.robots.unitree.G1_MINIMAL_CFG` for the velocity locomotion tasks, on the
  ``Isaac/Robots/Unitree/G1/g1.usd`` asset Unitree ships today. Joint names differ from
  :data:`~isaaclab_assets.robots.unitree.G1_CFG`: ``torso_joint`` is ``waist_yaw_joint``, and
  ``elbow_pitch_joint`` / ``elbow_roll_joint`` are ``elbow_joint`` / ``wrist_roll_joint``. Unlike
  ``g1_minimal.usd`` this asset carries collision geometry on the pelvis, knees, waist and wrists,
  so a task using it needs a base-height constraint; without one a crouched gait survives the whole
  episode and ``success_rate`` ends at 0.010 instead of 1.000.
