Fixed
^^^^^

* Fixed :meth:`~isaaclab.envs.DirectRLEnv.reset` to store the observation buffer
  like :meth:`~isaaclab.envs.DirectRLEnv.step` already does, and exposed the
  latest observations on the multi-agent-to-single-agent adapter through the
  same public buffer.
