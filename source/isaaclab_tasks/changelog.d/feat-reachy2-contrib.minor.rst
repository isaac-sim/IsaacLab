Added
^^^^^

* Added reach task environments for the Pollen Robotics Reachy 2 bimanual humanoid:
  :class:`~isaaclab_tasks.contrib.reach.config.reachy2.joint_pos_env_cfg.Reachy2RightReachEnvCfg`,
  :class:`~isaaclab_tasks.contrib.reach.config.reachy2.joint_pos_env_cfg.Reachy2LeftReachEnvCfg`, and
  :class:`~isaaclab_tasks.contrib.reach.config.reachy2.bimanual_joint_pos_env_cfg.Reachy2BimanualReachEnvCfg`
  (both arms tracking independent end-effector targets). Gym IDs:
  ``IsaacContrib-Reach-Reachy2-Right``, ``IsaacContrib-Reach-Reachy2-Left``,
  ``IsaacContrib-Reach-Reachy2-Bimanual`` (each with a ``-Play`` variant).
* Added a cube lift task environment for the Reachy 2 right arm with binary gripper
  control: :class:`~isaaclab_tasks.contrib.lift.config.reachy2.joint_pos_env_cfg.Reachy2CubeLiftEnvCfg`.
  Gym ID: ``IsaacContrib-Lift-Cube-Reachy2`` (with a ``-Play`` variant).
