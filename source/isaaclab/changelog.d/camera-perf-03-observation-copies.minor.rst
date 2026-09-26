Added
^^^^^

* Added :attr:`~isaaclab.managers.ObservationTermCfg.clone_output` to skip the observation manager's
  copy for terms that return a new tensor on every call, such as normalized images.

Changed
^^^^^^^

* Removed redundant image copies: single-term observation groups without history skipped the
  concatenation copy, and ``image_features`` created its normalization statistics once.
