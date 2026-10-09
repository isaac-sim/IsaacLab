* Fixed state-dependent action terms, such as operational-space control and relative joint position, being applied
  once per environment step when the physics backend ran the decimation loop. :class:`~isaaclab.envs.ManagerBasedEnv`
  now lets the backend run the loop only when no action term must run before every physics step.
* Fixed physics callbacks registered through a physics manager subclass and through its parent receiving the same
  id, which let one registration overwrite another. :meth:`~isaaclab.physics.PhysicsManager.register_callback` now
  uses one counter for every subclass.
