* Changed :meth:`~isaaclab.managers.RewardManager.compute` to count a reward term value that is not finite, for
  instance after an environment's simulation diverged, as zero for that step, so it reaches neither the learner nor
  the episodic sums.
