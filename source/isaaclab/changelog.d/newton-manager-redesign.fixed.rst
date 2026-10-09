* Fixed state-dependent action terms, such as operational-space control and relative joint position, being applied
  once per environment step when the physics backend ran the decimation loop. :class:`~isaaclab.envs.ManagerBasedEnv`
  now lets the backend run the loop only when no action term must run before every physics step.
