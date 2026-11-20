* Added :attr:`~isaaclab.envs.mdp.actions.JointActionCfg.chunk_length`, so that joint position, velocity and effort
  actions can apply a chunk of commands in order over the physics steps of an environment step. A policy that plans
  action chunks then acts once per chunk. Added
  :attr:`~isaaclab.envs.mdp.actions.JointPositionActionCfg.interpolate` to move to one target in equal increments
  instead. The default, one command per environment step, is unchanged.
