Added
^^^^^

* Added :func:`~isaaclab.utils.configclass.find_cfgs` to collect every instance of a config type nested in a
  resolved config, and :func:`~isaaclab.utils.io.latest_file` to return the most recently modified match of a
  glob pattern in a directory.
* Added :attr:`~isaaclab.utils.Checkpoint.local_path`, which the pretrained-checkpoint fetch sets to the
  downloaded copy of a declared checkpoint. :meth:`~isaaclab.utils.Checkpoint.resolve` returns it first.

Changed
^^^^^^^

* Changed :meth:`~isaaclab.utils.Checkpoint.resolve` to no longer search the run directory for a copy named
  ``<stem>_<name><ext>``; a fetched copy is announced through ``local_path`` instead of by filename.
