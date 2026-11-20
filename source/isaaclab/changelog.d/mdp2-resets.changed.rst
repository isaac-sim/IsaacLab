* Changed a ``state_ids`` request of -1 in :meth:`~isaaclab.envs.reset.ResetStateSampler.reset` to draw from the
  scores for that environment, so that one captured reset can replay some environments and sample the others. It
  was an invalid request that left the environment's start unchanged.
