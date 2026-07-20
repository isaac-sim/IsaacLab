Added
^^^^^

* Added reusable unified training and playback entrypoints under :mod:`isaaclab_rl`, including the
  :class:`~isaaclab_rl.TrainingRequest` and :class:`~isaaclab_rl.PlaybackRequest` programmatic APIs
  and the :func:`~isaaclab_rl.train`, :func:`~isaaclab_rl.play`, :func:`~isaaclab_rl.run_train_cli`,
  and :func:`~isaaclab_rl.run_play_cli` functions.

Removed
^^^^^^^

* Removed the deprecated per-library scripts under ``scripts/reinforcement_learning/<library>/``
  (``train.py``, ``play.py``, and ``cli_args.py``). Use the unified
  ``scripts/reinforcement_learning/train.py`` and ``play.py`` executables with
  ``--rl_library <library>``, or the programmatic :func:`~isaaclab_rl.train` and
  :func:`~isaaclab_rl.play` APIs instead.
* Removed the ``--use_last_checkpoint`` flag from the RL-Games ``play`` entrypoint.
  Use ``--checkpoint latest`` to select the newest checkpoint instead.

Fixed
^^^^^

* Fixed the ``--deterministic`` flag not configuring PyTorch deterministic operations in the RL-Games,
  RSL-RL, Stable-Baselines3, and skrl backends of the unified ``train`` and ``play`` entrypoints.
