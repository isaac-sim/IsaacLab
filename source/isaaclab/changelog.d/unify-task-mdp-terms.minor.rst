Added
^^^^^

* Added :class:`~isaaclab.envs.mdp.rewards.survival_success_rate`, :func:`~isaaclab.envs.mdp.rewards.terminated_penalty`
  and :func:`~isaaclab.envs.mdp.rewards.joint_pos_target_l2` reward terms, previously duplicated across the cartpole,
  locomotion and DR-legs task packages.
* Added :class:`~isaaclab.envs.mdp.curriculums.DifficultyScheduler` and
  :func:`~isaaclab.envs.mdp.curriculums.initial_final_interpolate_fn` for adaptive domain randomization curricula,
  previously local to the lift task package. The scheduler reads the success flag from the reward term named by the new
  ``success_term_name`` parameter (default ``"success"``).
