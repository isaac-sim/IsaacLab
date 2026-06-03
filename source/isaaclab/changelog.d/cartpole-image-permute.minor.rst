Added
^^^^^

* Added a ``permute`` argument to :func:`~isaaclab.envs.mdp.image` that returns
  image observations in channel-first ``[num_envs, channel, height, width]``
  layout. Defaults to ``False``, preserving the existing channel-last output.
