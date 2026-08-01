Fixed
^^^^^

* Fixed published pre-trained checkpoints failing to download after the task renames:
  :meth:`~isaaclab_rl.utils.pretrained_checkpoint.get_published_pretrained_checkpoint` now retries
  with the legacy ``<task>-v0`` name when the current task name is not found on the asset server.
  Also made the ``h1_locomotion`` demo exit with a clear message instead of an opaque
  ``AttributeError`` when no checkpoint could be retrieved.
