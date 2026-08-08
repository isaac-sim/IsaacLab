Fixed
^^^^^

* Fixed :func:`~isaaclab_tasks.utils.parse_cfg.get_checkpoint_path` raising a bare ``FileNotFoundError``
  when the log directory does not exist. It now raises the documented ``ValueError`` naming the directory
  and pattern, matching the behavior when the directory exists but holds no runs.
