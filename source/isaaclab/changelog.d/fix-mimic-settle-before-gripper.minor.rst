Added
^^^^^

* Added :attr:`~isaaclab.envs.mimic_env_cfg.SubTaskConfig.num_settle_steps_before_gripper`, which holds
  the target of each gripper transition of a Mimic subtask for a number of noise-free steps before the
  transition is commanded. Source demonstrations are replayed by frame index, which drops the pause a
  human operator makes before acting the gripper; the generated arm is still moving when the gripper
  closes or opens. Defaults to 0, which keeps the current behavior.
