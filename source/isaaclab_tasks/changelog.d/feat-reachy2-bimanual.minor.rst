Added
^^^^^

* Added a bimanual reach task environment for the Pollen Robotics Reachy 2 humanoid:
  :class:`~isaaclab_tasks.contrib.reach.config.reachy2.bimanual_joint_pos_env_cfg.Reachy2BimanualReachEnvCfg`.
  Both arms simultaneously track independent end-effector pose targets (14-DOF action
  space, 82-dim observations). Gym IDs: ``IsaacContrib-Reach-Reachy2-Bimanual`` and
  ``IsaacContrib-Reach-Reachy2-Bimanual-Play``.
