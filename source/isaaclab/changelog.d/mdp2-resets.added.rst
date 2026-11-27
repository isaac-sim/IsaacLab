* Added an optional ``rng`` argument to :meth:`~isaaclab.envs.reset.ResetStateSampler.reset` so that callers'
  per-environment int32 seeds, such as MDP2's random streams, drive the draws.
* Added :meth:`~isaaclab.envs.reset.ResetStateBuffer.set_state`, the inverse of ``get_state``, to replace stored states
  while running, and :meth:`~isaaclab.envs.reset.ResetStateSampler.forget` to clear the outcomes of replaced entries.
* Added ``sample_unrequested`` to :meth:`~isaaclab.envs.reset.ResetStateSampler.reset`, with which a ``state_ids``
  request of -1 draws from the scores for that environment, so that one captured reset can replay some environments
  and sample the others.
