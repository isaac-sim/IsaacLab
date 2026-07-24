Changed
^^^^^^^

* Changed :meth:`~isaaclab_rl.rsl_rl.RslRlVecEnvWrapper.get_observations` to read
  the environment-owned observation buffer instead of calling private environment
  methods. The returned observations now match the latest reset/step returns,
  including observation-noise corruption that the private path skipped, and
  multi-agent environments converted with
  :func:`~isaaclab.envs.utils.multi_agent_to_single_agent` train with RSL-RL
  without environment-side accommodations.
