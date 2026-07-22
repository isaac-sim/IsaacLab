Changed
^^^^^^^

* **Breaking:** Removed the ``*_PLAY`` environment configuration classes and the ``-Play`` gym task
  registrations. Play-mode overrides are now defined by overriding ``play_mode`` on the training
  environment configuration and are applied automatically by the play scripts. Use the training task
  id with ``play.py`` (a ``-Play`` id is redirected with a deprecation warning), and pass
  ``--train_env_cfg`` to play the training configuration as-is.

Added
^^^^^

* Added a ``play_mode`` argument to :func:`~isaaclab_tasks.utils.hydra.hydra_task_config`,
  :func:`~isaaclab_tasks.utils.hydra.resolve_task_config` and
  :func:`~isaaclab_tasks.utils.hydra.register_task` that applies the environment configuration's
  ``play_mode`` overrides after preset resolution.
