Fixed
^^^^^

* Fixed the RLinf extension rejecting a direct ``full_weights.pt`` path in ``rl_model_path``. It now
  accepts either the file or the ``global_step_<N>`` directory holding it, as its error message
  already stated.
