* Fixed state-dependent action terms, such as operational-space control and relative joint position, being applied
  once per environment step when the physics backend folded the decimation loop. :class:`~isaaclab.envs.ManagerBasedEnv`
  now lets the backend fold the loop only when no action term must run before every physics step.
