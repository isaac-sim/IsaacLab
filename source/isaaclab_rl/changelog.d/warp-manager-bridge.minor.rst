Added
^^^^^

* Added ``--frontend {torch,warp}`` to the shared reinforcement-learning
  training and play CLIs (all supported RL libraries) for selecting the
  environment runtime; default ``torch`` is unchanged. The ``rlinf``
  integration constructs environments inside the external framework and is
  not frontend-routable.
