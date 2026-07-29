Fixed
^^^^^

* Fixed :meth:`~isaaclab.envs.DirectRLEnv.reset` to store the observation buffer like
  :meth:`~isaaclab.envs.DirectRLEnv.step`, and exposed it on the
  multi-agent-to-single-agent adapter.
