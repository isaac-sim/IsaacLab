* Added an optional ``rng`` argument to :meth:`~isaaclab.envs.reset.ResetStateSampler.reset` so that callers'
  per-environment int32 seeds, such as MDP2's random streams, drive the draws.
* Added :meth:`~isaaclab.envs.reset.ResetStateBuffer.set_state`, the inverse of ``get_state``, to replace stored states
  while running, and :meth:`~isaaclab.envs.reset.ResetStateSampler.forget` to clear the outcomes of replaced entries.
* Added per-environment sampling to explicit requests: a ``state_ids`` request of -1 in
  :meth:`~isaaclab.envs.reset.ResetStateSampler.reset` draws from the scores instead of being invalid.
