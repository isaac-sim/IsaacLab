Changed
^^^^^^^

* **Breaking:** Removed the ``*_PLAY`` environment configuration classes and the ``-Play`` gym task
  registrations. Play-mode overrides are now defined by overriding ``play_mode`` on the training
  environment configuration and are applied automatically by the play scripts. Use the training task
  id with ``play.py``, and pass ``--train_env_cfg`` to play the training configuration as-is.
