* Changed :class:`~isaaclab.managers.RewardManager` to count a reward term value that is not finite, for instance
  after an environment's simulation diverged, as zero for that step, so it reaches neither the learner nor the
  episodic sums. The same applies to ``extras["final_obs"]``. The new
  :attr:`~isaaclab.envs.ManagerBasedRLEnvCfg.drop_non_finite_values` flag (default ``True``) disables both.
