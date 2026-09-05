Added
^^^^^

* Added :class:`~isaaclab.utils.Checkpoint` for weights a component loads at runtime -- a vision
  feature extractor trained with the policy, or a frozen encoder fetched from a URL. Declaring it on
  the component's own configuration is enough for the checkpoint tooling to publish and fetch a run
  artifact, and :meth:`~isaaclab.utils.Checkpoint.resolve` gives the component its local file
  without knowing any naming convention; task configurations declare nothing. A declaration must name
  exactly one of ``run_glob`` and ``url``, and a fetch records the copy it downloaded in
  :attr:`~isaaclab.utils.Checkpoint.local_path`, which :meth:`~isaaclab.utils.Checkpoint.resolve`
  returns before searching a directory.
