Added
^^^^^

* Added a ``preferred_checkpoint`` regex argument to :func:`~isaaclab_tasks.utils.parse_cfg.get_checkpoint_path`
  that is matched before ``checkpoint`` and wins when it matches, otherwise resolution falls back to ``checkpoint``.

Fixed
^^^^^

* Fixed ``rl_games`` and ``sb3`` play failing to load a checkpoint on short runs where the preferred
  best/final checkpoint has not been written yet. They now prefer the best (``rl_games``) or final
  (``sb3``) checkpoint and fall back to the latest available checkpoint when it is missing. Numbered
  checkpoint filenames are now sorted naturally so epoch 10 is selected after epoch 9.
