Fixed
^^^^^

* Fixed :func:`~isaaclab.envs.multi_agent_to_single_agent` to expose the terminal observations that
  :class:`~isaaclab.envs.DirectMARLEnv` captures per agent (``extras[agent]["final_obs"]``, see
  :attr:`~isaaclab.envs.DirectMARLEnvCfg.compute_final_obs`) as the concatenated single-agent
  ``extras["final_obs"]`` entry, so single-agent RL wrappers bootstrap time-outs of converted multi-agent tasks
  correctly. The converted environment now also exposes :attr:`extras` like :class:`~isaaclab.envs.DirectRLEnv`.
