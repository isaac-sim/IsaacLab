Fixed
^^^^^

* Fixed gripper-object penetration in the SO-101 stack tasks: the moving jaw tunneled into
  grasped objects because the articulation position drive was resolved after the contacts.
  Added :class:`~isaaclab_tasks.contrib.stack.config.so101.stack_joint_pos_env_cfg.SO101StackPhysicsCfg`
  enabling ``solve_articulation_contact_last`` for these tasks.
