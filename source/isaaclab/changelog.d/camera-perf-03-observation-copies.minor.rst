Added
^^^^^

* Added :attr:`~isaaclab.managers.ObservationTermCfg.clone_output` to skip the observation manager's
  copy for terms that return a new tensor on every call, such as normalized images.

Changed
^^^^^^^

* Changed the default of ``clone`` in :func:`~isaaclab.envs.mdp.observations.image` to False,
  because the observation manager already copies every term's output. Direct callers that mutate
  an unnormalized result should pass ``clone=True``.
* Removed redundant image copies: single-term observation groups without history skip the
  concatenation copy, :class:`~isaaclab.envs.mdp.observations.stacked_image` no longer clones on top
  of the manager, and ``image_features`` creates its normalization statistics once.
