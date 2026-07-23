Added
^^^^^

* Added a ``--train_env_cfg`` flag to the play entrypoints that plays the training environment
  configuration as-is, skipping the play-mode overrides defined by the environment configuration's
  ``play_mode`` method.
* Added :func:`~isaaclab_rl.entrypoints.common.resolve_play_task_name` that redirects a retired
  ``-Play`` task id to its training task id with a deprecation warning.
