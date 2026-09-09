Added
^^^^^

* Added :class:`~isaaclab.utils.Checkpoint` for weights a training run writes beside the policy, such
  as a vision feature extractor. Declaring it on the component's own configuration is enough for the
  checkpoint tooling to publish and fetch it, and :meth:`~isaaclab.utils.Checkpoint.resolve` gives the
  component its local file without knowing any naming convention; task configurations declare nothing.
  A fetch records the copy it downloaded in :attr:`~isaaclab.utils.Checkpoint.local_path`, which
  :meth:`~isaaclab.utils.Checkpoint.resolve` returns before searching a directory.
