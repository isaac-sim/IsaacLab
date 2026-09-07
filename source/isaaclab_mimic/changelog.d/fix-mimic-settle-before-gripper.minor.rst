Added
^^^^^

* Added a settle-before-gripper hold to Mimic data generation. When
  :attr:`~isaaclab.envs.mimic_env_cfg.SubTaskConfig.num_settle_steps_before_gripper` is set, the
  target of every gripper transition in a subtask segment is held, with the pre-transition gripper
  action and no action noise, for that many steps before the transition is commanded, so the generated
  arm settles where the source arm was when its operator acted the gripper. On the Franka cube stack
  task this raised the generation success rate from 35% to 54% (five scene draws, 300 attempts each).
  Off by default.
